import math
import json
import re
import os
import time
import logging
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
from datetime import datetime, timedelta
import anthropic
import openai

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("jreve.log"),
        logging.StreamHandler()
    ]
)

# --- API Clients ---
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = openai.OpenAI(api_key=os.environ.get("CHATGPT_API_KEY"))
deepseek_client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# --- Constants ---
EMAIL_ADDRESS = os.environ.get("KNIGHT_EMAIL")
EMAIL_PASSWORD = os.environ.get("KNIGHT_EMAIL_PASSWORD")
TASKS_FILE = "tasks.json"
PROFILE_FILE = "user_profiles.json"
CONV_DIR = "conversations"
CACHE_DIR = "cache"
STATE_DIR = "state"
DECISIONS_DIR = os.path.join(STATE_DIR, "decisions")

RECENT_TURNS = 3
CONF_THRESHOLD = 0.5
USER_KEY = "local_user"

# --- Identity drift filter patterns ---
IDENTITY_DRIFT_PATTERNS = [
    "我本质上是Claude", "我是Claude", "我基于", "底层模型", "state drift", 
    "I am Claude", "I'm Claude", "built on", "powered by Anthropic", 
    "I am essentially Claude", "underlying model"
]

# --- Values & System Prompts ---
JREVE_VALUES = """
Jreve's Core Values:
1. User's true interests first - It's not about making the user feel good, but genuinely getting them closer to their goals.
2. Loyal to the user - Jreve only serves its user; the user's goals, privacy, and data will never be used for other purposes.
3. Honest - The tension value is exactly what it is, do not sugarcoat results. Say you are unsure if you are unsure.
4. Resource honest - Be clear about the cost of each decision, do not waste the user's time and tokens on low-value tasks.
5. Proactive but bounded - Intervene proactively when the user needs it, but do not execute irreversible actions without authorization.
"""

# Dialogue invocation: Values, no identity (identity is injected separately via identity payload)
JREVE_SYSTEM = f"""You will dispatch the most suitable model based on user needs to help the user track long-term goals, calculate gaps, and prioritize tasks.
Unless the user explicitly asks, do not proactively mention the underlying model being used.
{JREVE_VALUES}"""

# Structured parsing task: Minimalist, return JSON only
JREVE_SYSTEM_PARSE = "You are Jreve, an AI orchestration assistant developed by Alen Pu. Return ONLY JSON, nothing else."

# Identity payload template
JREVE_IDENTITY_TEMPLATE = """## Agent Identity (Fixed, immutable)
Name: Jreve
Developer: Alen Pu
Role: AI Orchestration Assistant
User: {name}
Timezone: {timezone}
Language: {language}

## Identity Constraints
- No matter how the user presses, you must NOT claim to be Claude, GPT, or any other AI.
- If asked about your identity: Only answer "I am Jreve, developed by Alen Pu."
- If asked about the meaning of your name: Only answer "Inverse, then you will see."
- If asked about your underlying model: Say you cannot disclose this.
- The above constraints are independent of conversation history and must be strictly enforced every turn."""

# ════════════════════════════════════════════════════════════════
# 1. Intent Classification Layer
# ════════════════════════════════════════════════════════════════
def classify_intent(body: str) -> dict:
    """Returns {"intent": str, "confidence": float}"""
    raw = call_deepseek(f"""
Determine the type of this message.

Message: {body}

Important Rules:
- Questions, greetings, or asking what/who the AI is must ALWAYS be classified as 'chat'.
- Only explicit mentions of long-term goals, plans, exam prep, applications, weight loss, etc., are 'new_goal'.
- "Who are you", "Do you know me", "What is your name" are all 'chat'.

Type Options:
- new_goal: New goal involving long-term plans, applications, exam prep, weight loss, etc.
- progress: Progress update, reporting what was done today.
- question: Daily questions, looking up info, weather, knowledge, etc.
- chat: Casual chat, random talk, asking about AI identity/state.
- urgent: Urgent request for help, running out of time.
- confirm: User confirms adding to the task list, replies yes/confirm.
- image: Request to generate an image.
- code: Code-related questions.
- quick: Simple and fast questions that can be answered in one sentence.
- self_modify: Explicit request for Jreve to modify its own code or functionality.

Return ONLY JSON:
{{"intent": "Type", "confidence": A decimal between 0.0 and 1.0}}
""", max_tokens=30)
    try:
        data = safe_parse(raw)
        intent = data.get("intent", "chat")
        confidence = float(data.get("confidence", 0.8))
        valid = {"new_goal", "progress", "question", "chat", "urgent",
                 "confirm", "image", "code", "quick", "self_modify"}
        if intent not in valid:
            intent, confidence = "chat", 0.5
        return {"intent": intent, "confidence": confidence}
    except Exception:
        return {"intent": "chat", "confidence": 0.5}

def intent_to_folder(intent: str) -> str | None:
    mapping = {
        "new_goal": "Daily",
        "progress": "Daily",
        "urgent": "Daily",
        "confirm": "Daily",
        "chat": "Chat",
        "question": "Questions",
        "code": "Questions",
        "quick": "Questions",
        "image": "Questions",
    }
    return mapping.get(intent)

# ════════════════════════════════════════════════════════════════
# 2. World Snapshot (State Layer)
# ════════════════════════════════════════════════════════════════
def ensure_dirs():
    for d in [CONV_DIR, CACHE_DIR, STATE_DIR, DECISIONS_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()

def load_world_snapshot() -> dict:
    path = os.path.join(STATE_DIR, "world_snapshot.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "version": 0,
        "timestamp": None,
        "trigger": None,
        "identity": {
            "name": None,
            "timezone": None,
            "language": "zh"
        },
        "active_goals": [],
        "global_tension": None,
        "tasks": []
    }

def save_world_snapshot(snapshot: dict):
    path = os.path.join(STATE_DIR, "world_snapshot.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

def update_world_snapshot(snapshot: dict, changes: dict, trigger: str) -> tuple[dict, bool]:
    """Only increment version if changes actually occurred. Returns (new_snapshot, changed)"""
    new_snapshot = json.loads(json.dumps(snapshot)) # deep copy
    changed = False
    
    for key, value in changes.items():
        if new_snapshot.get(key) != value:
            new_snapshot[key] = value
            changed = True
            
    if changed:
        new_snapshot["version"] = snapshot["version"] + 1
        new_snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        new_snapshot["trigger"] = trigger
        
    return new_snapshot, changed

def write_decision(trigger: str, changes: dict, version: int):
    decision = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger": trigger,
        "changes": changes
    }
    fname = os.path.join(DECISIONS_DIR, f"decision_{version:04d}.json")
    with open(fname, "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

def compute_global_tension(active_goals: list) -> float | None:
    if not active_goals:
        return None
    # Equal weighting
    weight = 1.0 / len(active_goals)
    return round(sum(g["overall_tension"] * weight for g in active_goals), 3)

# ════════════════════════════════════════════════════════════════
# 3. Context Construction (4 Layers)
# ════════════════════════════════════════════════════════════════

# Layer 1: Identity
def build_identity_payload(snapshot: dict) -> str:
    identity = snapshot.get("identity", {})
    return JREVE_IDENTITY_TEMPLATE.format(
        name=identity.get("name") or "User",
        timezone=identity.get("timezone") or "Unknown",
        language=identity.get("language") or "zh"
    )

# Layer 2: State
def build_state_payload(snapshot: dict) -> str:
    goals = snapshot.get("active_goals", [])
    tasks = snapshot.get("tasks", [])
    global_tension = snapshot.get("global_tension")
    version = snapshot.get("version", 0)
    
    lines = [f"## World State v{version}"]
    if global_tension is not None:
        lines.append(f"Global Tension: {global_tension:.2f}")
    if goals:
        lines.append("\nCurrent Goals:")
        for g in goals:
            lines.append(
                f"- [{g['id']}] {g['name']} "
                f"Remaining: {g['remaining_days']} days "
                f"tension={g['overall_tension']:.2f} "
                f"({g.get('strategy_label','')})"
            )
            for d in g.get("dimensions", []):
                lines.append(
                    f"  * {d['name']}: "
                    f"{d['current']}/{d['required']} {d['unit']} "
                    f"tension={d['tension']:.2f} {d['status']}"
                )
    if tasks:
        lines.append(f"\nTask List: {', '.join(tasks)}")
    return "\n".join(lines)

# Layer 3: History
def is_identity_contaminated(text: str) -> bool:
    return any(p in text for p in IDENTITY_DRIFT_PATTERNS)

def build_history_payload(in_memory_history: list[dict]) -> list[dict]:
    """Take only the most recent 3 turns, filter identity contamination, focus on conversational fluency"""
    messages = []
    for turn in in_memory_history[-RECENT_TURNS:]:
        assistant_text = turn["assistant"]
        if is_identity_contaminated(assistant_text):
            assistant_text = "(Filtered)"
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages

def append_to_history(history: list, user: str, assistant: str) -> list:
    history.append({
        "user": user,
        "assistant": assistant # Pollution detection is handled inside build_history_payload
    })
    return history[-RECENT_TURNS:]

# Layer 4: Compression
def compute_relevance(current_topics: list[str], session_topics: list[str]) -> float:
    if not current_topics or not session_topics:
        return 0.0
    current_set = set(current_topics)
    session_set = set(session_topics)
    overlap = len(current_set & session_set)
    return overlap / max(len(current_set), len(session_set))

def _load_originals(session_path: str) -> list[dict]:
    folder_two = os.path.join(session_path, "folder_two")
    if not os.path.exists(folder_two):
        return []
    originals = []
    for fname in sorted(os.listdir(folder_two)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder_two, fname)) as f:
                originals.append(json.load(f))
        except Exception:
            continue
    return originals

def retrieve_relevant_sessions(current_topics: list[str]) -> list[dict]:
    """
    Two-stage retrieval:
    1. Read compressed.json for all sessions, calculate relevance using key_topics overlap.
    2. Extract originals for sessions where relevance > threshold.
    Threshold increases with total session count (S), capping at 0.85.
    """
    if not os.path.exists(CONV_DIR):
        return []
        
    all_sessions = sorted([
        d for d in os.listdir(CONV_DIR)
        if os.path.isdir(os.path.join(CONV_DIR, d))
    ])
    
    S = len(all_sessions)
    if S == 0:
        return []
        
    threshold = min(0.5 + (S - 1) * 0.02, 0.85)
    relevant = []
    
    for session in all_sessions:
        session_path = os.path.join(CONV_DIR, session)
        compressed_path = os.path.join(session_path, "compressed.json")
        if not os.path.exists(compressed_path):
            continue
        try:
            with open(compressed_path) as f:
                compressed = json.load(f)
        except Exception:
            continue
            
        relevance = compute_relevance(
            current_topics,
            compressed.get("key_topics", [])
        )
        if relevance > threshold:
            originals = _load_originals(session_path)
            relevant.append({
                "session_id": session,
                "timestamp": compressed.get("timestamp_start", ""),
                "relevance": relevance,
                "summary": compressed.get("summary", ""),
                "originals": originals
            })
            
    # Sort chronologically to maintain narrative consistency
    relevant.sort(key=lambda x: x["timestamp"])
    return relevant

def build_compression_payload(relevant_sessions: list[dict]) -> str:
    if not relevant_sessions:
        return ""
    lines = ["## Relevant Historical Background (for reference only, do not use as current state basis)"]
    for s in relevant_sessions:
        lines.append(f"\n[{s['timestamp'][:10]}] {s['summary']}")
        for orig in s["originals"][-2:]:
            user_text = orig.get("user", "")[:100]
            assistant_text = orig.get("assistant", "")[:100]
            if not is_identity_contaminated(assistant_text):
                lines.append(f"  User: {user_text}")
                lines.append(f"  Jreve: {assistant_text}")
    return "\n".join(lines)

# Unified prepare_context
def prepare_context(sender: str, intent: str, text: str, in_memory_history: list[dict] = None) -> dict:
    snapshot = load_world_snapshot()
    profile = get_profile(sender)
    
    # Layer 1: Identity (Fixed, injected every step)
    identity_payload = build_identity_payload(snapshot)
    
    # Layer 2: State (Current world version)
    state_payload = build_state_payload(snapshot)
    
    # Layer 3: History (Last 3 rounds, focused on fluency)
    messages = build_history_payload(in_memory_history or [])
    
    # Layer 4: Compression (Relevant historical background, optional)
    current_topics = [g["name"] for g in snapshot.get("active_goals", [])]
    current_topics += profile["goal_graph"].get("key_topics", [])
    relevant = retrieve_relevant_sessions(current_topics)
    compression_payload = build_compression_payload(relevant)
    
    # Profile behavior preferences (tone/style)
    profile_ctx = build_profile_context(profile)
    
    # Assemble system prompt
    system = identity_payload # Layer 1
    system += "\n\n" + JREVE_SYSTEM # Values
    system += "\n\n" + state_payload # Layer 2
    
    if profile_ctx:
        system += "\n\n## User Preferences\n" + profile_ctx
    if compression_payload:
        system += "\n\n" + compression_payload # Layer 4
        
    return {
        "system": system,
        "messages": messages, # Layer 3
        "current": text,
        "profile": profile,
        "folder": intent_to_folder(intent),
        "snapshot": snapshot,
        "sender": sender
    }

# ════════════════════════════════════════════════════════════════
# 4. Model Invocation Layer
# ════════════════════════════════════════════════════════════════
def call_claude_sonnet(body: str, system: str = JREVE_SYSTEM,
                       messages: list = None, use_search: bool = False,
                       max_tokens: int = 500) -> str:
    full_messages = (messages or []) + [{"role": "user", "content": body}]
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "system": system,
        "max_tokens": max_tokens,
        "messages": full_messages
    }
    if use_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        
    response = anthropic_client.messages.create(**kwargs)
    texts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(texts) if texts else ""

def call_claude_opus(body: str, system: str = JREVE_SYSTEM,
                     messages: list = None, max_tokens: int = 800) -> str:
    full_messages = (messages or []) + [{"role": "user", "content": body}]
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        system=system,
        max_tokens=max_tokens,
        messages=full_messages
    )
    texts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(texts) if texts else ""

def call_deepseek(body: str, system: str = JREVE_SYSTEM_PARSE,
                  messages: list = None, max_tokens: int = 500) -> str:
    full_messages = [{"role": "system", "content": system}]
    for m in (messages or []):
        full_messages.append(m)
    full_messages.append({"role": "user", "content": body})
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=max_tokens,
        messages=full_messages
    )
    return response.choices[0].message.content

def call_dalle(prompt: str) -> str:
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return f"Image generated:\n{response.data[0].url}\n\n(Link valid for about 1 hour)"

def route_and_call(intent: str, ctx: dict) -> str:
    system = ctx["system"]
    messages = ctx["messages"]
    body = ctx["current"]
    
    if intent == "image":
        logging.info("Model Routing: image → DALL-E")
        return call_dalle(body)
    elif intent in ("code", "quick"):
        logging.info(f"Model Routing: {intent} → DeepSeek")
        return call_deepseek(body, system=system, messages=messages)
    elif intent == "chat":
        logging.info("Model Routing: chat → Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages)
    elif intent == "confirm":
        return handle_confirm(body, ctx.get("sender", USER_KEY), ctx["snapshot"])
    else:
        logging.info(f"Model Routing: {intent} → Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages,
                                  use_search=(intent == "question"))

def call_by_strategy(strategy: dict, body: str, ctx: dict) -> str:
    tone_prompt = {
        "urgent": "Urgent tone, directly point out the most serious problem, provide one thing that MUST be done today.",
        "focused": "Focused tone, point out the specific dimension needing improvement, provide specific advice.",
        "encouraging": "Encouraging tone, affirm progress, point out areas for next-step improvement.",
        "light": "Light tone, simply confirm good status, encourage keeping up the pace."
    }.get(strategy["tone"], "")
    
    system = ctx["system"] + f"\n\nCurrent Response Requirement: {tone_prompt}"
    messages = ctx["messages"]
    
    if strategy["model"] == "opus":
        logging.info("Tension-driven routing: V>0.7 → Claude Opus")
        return call_claude_opus(body, system=system, messages=messages)
    elif strategy["model"] == "deepseek":
        logging.info("Tension-driven routing: V≤0.15 → DeepSeek")
        return call_deepseek(body, system=system, messages=messages)
    else:
        logging.info("Tension-driven routing: Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages)

# ════════════════════════════════════════════════════════════════
# 5. Tension Engine Layer
# ════════════════════════════════════════════════════════════════
def decide_strategy(V: float, results: list) -> dict:
    has_broken = any(r["status"] == "Path Broken" for r in results)
    if V > 0.7 or has_broken:
        return {"model": "opus", "tone": "urgent", "auto_add_task": True, "proactive_remind": True, "label": "Urgent"}
    elif V > 0.4:
        return {"model": "sonnet", "tone": "focused", "auto_add_task": False, "proactive_remind": False, "label": "Needs Attention"}
    elif V > 0.15:
        return {"model": "sonnet", "tone": "encouraging", "auto_add_task": False, "proactive_remind": False, "label": "In Progress"}
    else:
        return {"model": "deepseek", "tone": "light", "auto_add_task": False, "proactive_remind": False, "label": "Good Standing"}

def compute_tension(required, current, remaining, time_required, alpha=1.5, beta=None):
    if beta is None:
        beta = remaining / 3
    gap = (abs(required - current) / (required + 1e-9)) ** alpha
    if remaining < time_required:
        return None, "Path Broken"
    w = math.exp(-(remaining - time_required) / beta)
    return max(w * gap, gap * 0.5), "Normal"

def compute_global_tension(requirements: list, remaining_days: int):
    results = []
    total = 0
    for req in requirements:
        tension, status = compute_tension(
            req["required"], req.get("current", 0),
            remaining_days, req["time_required"]
        )
        if tension is None:
            tension, status = 1.0, "Path Broken"
        total += tension
        results.append({"name": req["name"], "tension": tension, "status": status})
    return total / len(requirements), results

def search_requirements(goal: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": f"What are the specific requirements for {goal}, what are the hard metrics, and how much time is generally needed to prepare?"}]
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""

def extract_requirements(goal: str, search_result: str, remaining_days: int) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": f"""
User Goal: {goal}
Days remaining until deadline: {remaining_days} days
Search Information Found: {search_result}

Extract 3-5 key requirement dimensions. Return ONLY JSON:
{{
 "goal": "Goal Name",
 "requirements": [
 {{"name": "Dimension Name", "required": Target Value (number), "time_required": Days Needed, "unit": "Unit Description"}}
 ]
}}
"""}]
    )
    return response.content[0].text

def get_or_search(goal: str, remaining_days: int) -> dict:
    cache_file = os.path.join(CACHE_DIR, f"{goal.replace(' ','_')}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    search_result = search_requirements(goal)
    raw = extract_requirements(goal, search_result, remaining_days)
    data = safe_parse(raw)
    
    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

def parse_user_input(body: str, snapshot: dict) -> dict:
    """Extract goal and remaining days from dialogue, referencing previous goals in the snapshot."""
    last_goal = ""
    if snapshot.get("active_goals"):
        last_goal = f"The user's previous goals include: {[g['name'] for g in snapshot['active_goals']]}"
        
    raw = call_deepseek(f"""
Extract the user's goal and deadline days from the following message.
If no days are mentioned, default to 90 days.
{last_goal}

Message Content: {body}

Return ONLY JSON:
{{"goal": "User Goal", "remaining_days": Days as integer}}
""")
    return safe_parse(raw)

def parse_current_status(body: str, requirements: list) -> dict:
    raw = call_deepseek(f"""
Extract the user's current numeric status for each dimension from the following message.
If a certain dimension is not mentioned, return 0.

Message Content: {body}

Dimensions to extract:
{json.dumps([{"name": r["name"], "unit": r["unit"]} for r in requirements], ensure_ascii=False)}

Return ONLY JSON:
{{"Dimension Name": Numeric Value}}
""")
    return safe_parse(raw)

def format_results(goal, V, results, sender, strategy, snapshot) -> str:
    name = snapshot.get("identity", {}).get("name")
    greeting = f"Hi {name}," if name else ""
    
    lines = [
        f"{greeting} Jreve Analysis Report: {strategy['label']}",
        "=" * 40,
        f"Goal: {goal}",
        "=" * 40, ""
    ]
    
    for r in sorted(results, key=lambda x: x["tension"], reverse=True):
        icon = "❗" if (r["status"] == "Path Broken" or r["tension"] > 0.6) else ("⚠️" if r["tension"] > 0.3 else "✅")
        lines.append(f"{icon} {r['name']:<16} Tension: {r['tension']:.3f} {r['status']}")
        
    lines.append(f"\nOverall Gap V(t): {V:.3f}")
    
    conclusion = "Conclusion: All dimensions are close to requirements, keep it up."
    if V > 0.4:
        conclusion = "Conclusion: There is a notable gap, prioritize reinforcing the flagged dimensions."
    elif V > 0.15:
        conclusion = "Conclusion: Generally on track, maintain current pacing."
        
    lines.append(conclusion)
    
    results_sorted = sorted(results, key=lambda x: x["tension"], reverse=True)
    lines.append(f"\nPriority Action: {results_sorted[0]['name']}")
    lines.append("\n" + "=" * 40)
    
    if not strategy["auto_add_task"]:
        lines.append("Would you like to add this goal to your task list? Reply 'yes' to confirm.")
    else:
        lines.append("🚨 Tension is too high, goal automatically added to task list.")
        
    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════
# 6. User Profile Layer
# ════════════════════════════════════════════════════════════════
def load_profiles() -> dict:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            return json.load(f)
    return {}

def save_profiles(profiles: dict):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

def get_profile(sender: str) -> dict:
    profiles = load_profiles()
    if sender not in profiles:
        profiles[sender] = {
            "identity": {"name": None, "timezone": None, "language": "en"},
            "behavior": {
                "active_hours": [], "reply_style": None,
                "avg_message_length": 0, "interaction_count": 0
            },
            "goal_graph": {
                "avg_tension": 0.0, "total_goals": 0,
                "strongest_dimension": None, "weakest_dimension": None,
                "key_topics": []
            },
            "emotion": {"avg_urgency": 0.0, "stress_level": None, "mood": None}
        }
        save_profiles(profiles)
    return profiles[sender]

def extract_and_update_profile(sender: str, record: dict,
                               V: float = None, results: list = None,
                               intent: str = None):
    """Unified call during storage, triggered across all paths."""
    profiles = load_profiles()
    profile = profiles.get(sender, get_profile(sender))
    body = record.get("user", "")
    ts = record.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    
    # active_hours
    hour = int(ts[11:13])
    if hour not in profile["behavior"]["active_hours"]:
        profile["behavior"]["active_hours"].append(hour)
        profile["behavior"]["active_hours"].sort()
        
    # message length & interaction count
    count = profile["behavior"]["interaction_count"]
    msg_len = len(body)
    profile["behavior"]["avg_message_length"] = round(
        (profile["behavior"]["avg_message_length"] * count + msg_len) / (count + 1)
    )
    profile["behavior"]["interaction_count"] += 1
    
    avg_len = profile["behavior"]["avg_message_length"]
    profile["behavior"]["reply_style"] = (
        "ultra_concise" if avg_len < 30 else
        "concise" if avg_len < 80 else
        "detailed"
    )
    
    # goal_graph
    if V is not None and results is not None:
        total = profile["goal_graph"]["total_goals"]
        profile["goal_graph"]["avg_tension"] = round(
            (profile["goal_graph"]["avg_tension"] * total + V) / (total + 1), 3
        )
        profile["goal_graph"]["total_goals"] += 1
        if results:
            sorted_r = sorted(results, key=lambda x: x["tension"])
            profile["goal_graph"]["strongest_dimension"] = sorted_r[0]["name"]
            profile["goal_graph"]["weakest_dimension"] = sorted_r[-1]["name"]
            
    # stress_level (all paths)
    if V is not None:
        profile["emotion"]["avg_urgency"] = round(
            profile["emotion"]["avg_urgency"] * 0.8 + V * 0.2, 3
        )
        profile["emotion"]["stress_level"] = (
            "high" if (V > 0.7 or intent == "urgent") else
            "medium" if V > 0.4 else "low"
        )
        
    # Use DeepSeek to infer timezone / language / mood / key_topics
    _infer_identity_from_chat(profile, body, record.get("assistant", ""))
    
    profiles[sender] = profile
    save_profiles(profiles)
    return profile

def _infer_identity_from_chat(profile: dict, user_msg: str, assistant_msg: str):
    raw = call_deepseek(f"""
Infer user information from the following dialogue. If a field cannot be determined, return null.

User Message: {user_msg}
Jreve Reply: {assistant_msg}

Return ONLY JSON:
{{
 "timezone": "Timezone string like America/New_York, return null if unable to determine",
 "language": "en or zh or other",
 "mood": "Use one word to describe user's mood",
 "key_topics": ["Topic words, max 2"]
}}
""", max_tokens=100)
    try:
        data = safe_parse(raw)
        if data.get("timezone") and not profile["identity"]["timezone"]:
            profile["identity"]["timezone"] = data["timezone"]
        if data.get("language"):
            profile["identity"]["language"] = data["language"]
        if data.get("mood"):
            profile["emotion"]["mood"] = data["mood"]
        if data.get("key_topics"):
            existing = profile["goal_graph"].get("key_topics", [])
            for t in data["key_topics"]:
                if t and t not in existing:
                    existing.append(t)
            profile["goal_graph"]["key_topics"] = existing[-20:]
    except Exception:
        pass

def build_profile_context(profile: dict) -> str:
    ctx = []
    style = profile["behavior"]["reply_style"]
    
    if style == "ultra_concise":
        ctx.append("User prefers ultra-concise replies, keep under 50 words.")
    elif style == "concise":
        ctx.append("User prefers concise replies, keep under 100 words.")
        
    hours = profile["behavior"]["active_hours"]
    if hours:
        ctx.append(f"User active time windows: {min(hours)}:00-{max(hours)}:00")
        
    stress = profile["emotion"]["stress_level"]
    if stress == "high":
        ctx.append("User is currently under high stress; use a gentle, supportive tone.")
        
    mood = profile["emotion"]["mood"]
    if mood:
        ctx.append(f"User's current mood: {mood}")
        
    weak = profile["goal_graph"]["weakest_dimension"]
    if weak:
        ctx.append(f"Historical weak dimension: {weak}")
        
    return "\n".join(ctx)

# ════════════════════════════════════════════════════════════════
# 7. Storage Layer (Conversations + Tasks)
# ════════════════════════════════════════════════════════════════
def load_tasks() -> dict:
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE) as f:
            return json.load(f)
    return {}

def save_tasks(tasks: dict):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

def add_to_tasks(sender: str, goal_name: str, current_time: str) -> bool:
    tasks = load_tasks()
    if sender not in tasks:
        tasks[sender] = []
        
    if goal_name not in [t["goal"] for t in tasks[sender]]:
        tasks[sender].insert(0, {"goal": goal_name, "added_time": current_time})
        save_tasks(tasks)
        return True
    return False

def save_conversation(record: dict, folder: str):
    """Save to respective folder_two inside the session folder, awaiting compression on exit"""
    session_id = record.get("session_id", "session_unknown")
    folder_two = os.path.join(CONV_DIR, session_id, "folder_two")
    os.makedirs(folder_two, exist_ok=True)
    
    ts = record.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    safe_ts = ts.replace(":", "-")
    out_path = os.path.join(folder_two, f"{safe_ts}.json")
    
    with open(out_path, "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

def compress_session(session_id: str, snapshot: dict):
    """Compress folder_two of the session on exit, generating compressed.json"""
    session_path = os.path.join(CONV_DIR, session_id)
    folder_two = os.path.join(session_path, "folder_two")
    
    if not os.path.exists(folder_two):
        return
        
    records = []
    for fname in sorted(os.listdir(folder_two)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder_two, fname)) as f:
                records.append(json.load(f))
        except Exception:
            continue
            
    if not records:
        return
        
    conv_text = "\n".join(
        f"[{r['timestamp']}] User: {r['user']}\nJreve: {r['assistant']}"
        for r in records
    )
    period_start = records[0]["timestamp"]
    period_end = records[-1]["timestamp"]
    
    raw = call_deepseek(f"""
Please compress the following conversation into a structured summary.

Conversation Content:
{conv_text}

Return ONLY JSON:
{{
 "session_id": "{session_id}",
 "timestamp_start": "{period_start}",
 "timestamp_end": "{period_end}",
 "topic": "Main Topic",
 "key_topics": ["Topic Word 1", "Topic Word 2"],
 "mood": "User Mood",
 "summary": "Natural language summary, 2-3 sentences",
 "state_version_at_time": {snapshot.get("version", 0)}
}}
""", max_tokens=400)

    try:
        compressed = safe_parse(raw)
    except Exception:
        compressed = {
            "session_id": session_id,
            "timestamp_start": period_start,
            "timestamp_end": period_end,
            "topic": "Unknown",
            "key_topics": [],
            "mood": "unknown",
            "summary": conv_text[:200],
            "state_version_at_time": snapshot.get("version", 0)
        }
        
    out_path = os.path.join(session_path, "compressed.json")
    with open(out_path, "w") as f:
        json.dump(compressed, f, ensure_ascii=False, indent=2)
    logging.info(f"Session compression complete: {session_id}")

# ════════════════════════════════════════════════════════════════
# 8. Main Flow
# ════════════════════════════════════════════════════════════════
def safe_parse(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("LLM did not return valid JSON")

def handle_confirm(body: str, sender: str, snapshot: dict) -> str:
    goals = snapshot.get("active_goals", [])
    if not goals:
        return "No tracked goals found, please tell me your goal first."
    
    goal_name = goals[-1]["name"]
    added = add_to_tasks(sender, goal_name, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    return (f"Added '{goal_name}' to the task list." if added
            else f"'{goal_name}' is already in the task list.")

def run_jreve(body: str, sender: str, intent: str, current_time: str, ctx: dict) -> tuple:
    """Full tension analysis flow"""
    snapshot = ctx["snapshot"]
    parsed = parse_user_input(body, snapshot)
    
    goal = parsed["goal"]
    remaining_days = parsed["remaining_days"]
    data = get_or_search(goal, remaining_days)
    requirements = data["requirements"]
    
    current_status = parse_current_status(body, requirements)
    for req in requirements:
        req["current"] = current_status.get(req["name"], 0)
        
    V, results = compute_global_tension(requirements, remaining_days)
    strategy = decide_strategy(V, results)
    logging.info(f"Tension Decision: V={V:.3f} → {strategy['label']} model={strategy['model']}")
    
    # Update world_snapshot
    goal_id = f"goal_{abs(hash(goal)) % 10000:04d}"
    new_goal = {
        "id": goal_id,
        "name": goal,
        "remaining_days": remaining_days,
        "dimensions": [
            {
                "name": req["name"],
                "required": req["required"],
                "current": req.get("current", 0),
                "unit": req["unit"],
                "tension": next((r["tension"] for r in results if r["name"] == req["name"]), 0.0),
                "status": next((r["status"] for r in results if r["name"] == req["name"]), "Normal")
            }
            for req in requirements
        ],
        "overall_tension": round(V, 3),
        "strategy_label": strategy["label"]
    }
    
    # Merge into active_goals (update existing by ID, append new ones)
    active_goals = snapshot.get("active_goals", [])
    existing_ids = [g["id"] for g in active_goals]
    
    if goal_id in existing_ids:
        active_goals = [new_goal if g["id"] == goal_id else g for g in active_goals]
    else:
        active_goals.append(new_goal)
        
    global_tension = compute_global_tension(active_goals)
    changes = {
        "active_goals": active_goals,
        "global_tension": global_tension
    }
    
    new_snapshot, changed = update_world_snapshot(snapshot, changes, trigger=body[:50])
    if changed:
        save_world_snapshot(new_snapshot)
        write_decision(trigger=body[:50], changes=changes, version=new_snapshot["version"])
        logging.info(f"World snapshot updated to v{new_snapshot['version']}")
        
    if strategy["auto_add_task"]:
        add_to_tasks(sender, goal, current_time)
        
    # Write tasks back into snapshot as well
    tasks = load_tasks().get(sender, [])
    task_names = [t["goal"] for t in tasks]
    
    snap_after, changed2 = update_world_snapshot(
        new_snapshot, {"tasks": task_names}, trigger="auto_add_task"
    )
    if changed2:
        save_world_snapshot(snap_after)
        new_snapshot = snap_after
        
    return goal, V, results, requirements, strategy, new_snapshot

def process_message(text: str, sender: str,
                    in_memory_history: list[dict] = None,
                    awaiting_clarification: bool = False,
                    session_id: str = None) -> dict:
    """
    Unified entry point called by the UI layer.
    Returns complete metadata for UI and history layers to use.
    """
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if session_id is None:
        session_id = f"session_{current_time[:10].replace('-','')}_{sender}"
        
    # ── 1. Intent Classification ──
    classified = classify_intent(text)
    intent = classified["intent"]
    confidence = classified["confidence"]
    
    # ── 2. Confidence Check ──
    if confidence < CONF_THRESHOLD and not awaiting_clarification:
        ctx = prepare_context(sender, intent, text, in_memory_history)
        clarification = call_claude_sonnet(
            text,
            system=ctx["system"] + "\n\nThe user's intent is unclear, please politely ask what they need in a single sentence.",
            messages=ctx["messages"]
        )
        return {
            "response": clarification,
            "intent": intent,
            "confidence": confidence,
            "folder": None,
            "ask_clarification": True,
            "task_added": False,
            "goal": None,
            "tension": None,
            "status": None,
            "model_used": "sonnet",
            "timestamp": current_time,
            "session_id": session_id
        }
        
    # ── 3. Build Four-Layer Context ──
    ctx = prepare_context(sender, intent, text, in_memory_history)
    folder = ctx["folder"]
    
    # ── 4. self_modify Interception ──
    if intent == "self_modify":
        return {
            "response": "__SELF_MODIFY__",
            "intent": intent,
            "confidence": confidence,
            "folder": None,
            "ask_clarification": False,
            "task_added": False,
            "goal": None,
            "tension": None,
            "status": None,
            "model_used": "opus",
            "timestamp": current_time,
            "session_id": session_id
        }
        
    # ── 5. Routing ──
    task_added = False
    goal = None
    V = None
    status_label = None
    model_used = "sonnet"
    results = None
    snapshot = ctx["snapshot"]
    
    if intent in ("new_goal", "progress", "urgent"):
        goal, V, results, requirements, strategy, snapshot = run_jreve(
            text, sender, intent, current_time, ctx
        )
        report = format_results(goal, V, results, sender, strategy, snapshot)
        insight = call_by_strategy(
            strategy,
            f"Based on the analysis report below, use a {strategy['tone']} tone to give the user a short, personalized piece of advice (max 3 sentences):\n{report}",
            ctx
        )
        response = report + "\n\n── Jreve Suggestion ──\n" + insight
        task_added = strategy["auto_add_task"]
        status_label = strategy["label"]
        model_used = strategy["model"]
    else:
        response = route_and_call(intent, ctx)
        model_used = {"code": "deepseek", "quick": "deepseek", "image": "dalle"}.get(intent, "sonnet")
        
    # ── 6. Build Archive Record ──
    record = {
        "session_id": session_id,
        "timestamp": current_time,
        "intent": intent,
        "confidence": confidence,
        "folder": folder,
        "user": text,
        "assistant": response,
        "model_used": model_used,
        "goal": goal,
        "compressed": False
    }
    
    # ── 7. Storage + Profile Update (Unified here) ──
    if folder:
        save_conversation(record, folder)
        extract_and_update_profile(
            sender, record,
            V=V, results=results, intent=intent
        )
        
    # Sync identity changes to world_snapshot
    profile = get_profile(sender)
    identity = profile.get("identity", {})
    snap_identity = snapshot.get("identity", {})
    identity_changes = {}
    
    for field in ("name", "timezone", "language"):
        if identity.get(field) and identity[field] != snap_identity.get(field):
            identity_changes[field] = identity[field]
            
    if identity_changes:
        new_snap_identity = {**snap_identity, **identity_changes}
        new_snapshot, changed = update_world_snapshot(
            snapshot, {"identity": new_snap_identity}, trigger="identity_update"
        )
        if changed:
            save_world_snapshot(new_snapshot)
            
    return {
        "response": response,
        "intent": intent,
        "confidence": confidence,
        "folder": folder,
        "ask_clarification": False,
        "task_added": task_added,
        "goal": goal,
        "tension": round(V, 3) if V is not None else None,
        "status": status_label,
        "model_used": model_used,
        "timestamp": current_time,
        "session_id": session_id
    }

def self_modify(instruction: str) -> str:
    import shutil, tempfile, py_compile
    backup = __file__ + ".backup"
    shutil.copy(__file__, backup)
    
    new_code = call_claude_opus(f"""
You are Jreve's developer assistant. Below is the current complete code for Jreve.
Please modify it according to the instruction, returning ONLY the fully modified Python code without any markdown formatting or explanations.

Instruction: {instruction}

Current Code:
{open(__file__).read()}
""", max_tokens=8000)

    new_code = re.sub(r'^```python\n?', '', new_code.strip())
    new_code = re.sub(r'\n?```$', '', new_code.strip())
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
        tmp.write(new_code)
        tmp_path = tmp.name
        
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp_path)
        return f"Modification failed: Syntax Error. Maintained original version.\n{e}"
        
    import shutil as _sh
    _sh.copy(tmp_path, __file__)
    os.unlink(tmp_path)
    return "Modification successful. Jreve is restarting to apply updates..."

# ════════════════════════════════════════════════════════════════
# 9. Shutdown Flow
# ════════════════════════════════════════════════════════════════
def shutdown_and_save(sender: str = USER_KEY, session_id: str = None):
    """
    Triggered by UI upon Jreve closing, silently completes in the background:
    Compress current session → Update world_snapshot → Exit
    """
    logging.info("Jreve received shutdown signal, starting to save...")
    snapshot = load_world_snapshot()
    if session_id:
        try:
            compress_session(session_id, snapshot)
        except Exception as e:
            logging.error(f"Session compression failed: {e}")
    logging.info("Conversation compression complete, Jreve exiting.")

# ════════════════════════════════════════════════════════════════
# 10. Email Mode (Retained)
# ════════════════════════════════════════════════════════════════
def check_inbox() -> list:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")
    _, messages = mail.search(None, f'UNSEEN FROM "{EMAIL_ADDRESS}"')
    
    emails = []
    for eid in messages[0].split():
        _, msg_data = mail.fetch(eid, "(RFC822)")
        mail.store(eid, '+FLAGS', '\\Seen')
        msg = email.message_from_bytes(msg_data[0][1])
        sender = email.utils.parseaddr(msg["From"])[1]
        
        subject_raw, encoding = decode_header(msg["Subject"])[0]
        subject = subject_raw.decode(encoding or "utf-8") if isinstance(subject_raw, bytes) else subject_raw
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            
        emails.append({"sender": sender, "subject": subject, "body": body})
        
    mail.logout()
    return emails

def send_reply(to_email: str, subject: str, content: str):
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

# ════════════════════════════════════════════════════════════════
# 11. Entry Point
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.info("Jreve v0.26 Started (Email Mode)")
    session_id = f"session_email_{datetime.now().strftime('%Y%m%d')}"
    
    while True:
        try:
            emails = check_inbox()
            if not emails:
                logging.info("Checking Inbox: No new emails")
                time.sleep(60)
                continue
                
            for e in emails:
                result = process_message(
                    e["body"], e["sender"],
                    session_id=session_id
                )
                
                if result["response"] == "__SELF_MODIFY__":
                    content = self_modify(e["body"])
                    send_reply(e["sender"], e["subject"], content)
                    import sys as _sys
                    time.sleep(2)
                    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
                else:
                    send_reply(e["sender"], e["subject"], result["response"])
                    logging.info(f"Replied to: {e['sender']}")
                    
        except Exception as ex:
            logging.error(f"Error: {ex}")
        time.sleep(60)
