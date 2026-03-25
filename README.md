# Jreve

Jreve is a personal AI assistant that helps you close the gap between where you are and where you want to be.

Most AI assistants answer questions. Jreve tracks goals.

You tell it what you're working toward — a job application, an exam, a fitness target — and it measures how far you are from getting there, then helps you figure out what to do about it.

---

## The Core Idea: Tension

Jreve uses a mathematical model to calculate the "tension" between your current state and your goal. Think of tension like the distance remaining on a road trip — except it also accounts for how much time you have left and how hard each part of the journey is.

For each goal, Jreve breaks it down into dimensions. If your goal is to get into a graduate program, the dimensions might be GPA, test scores, research experience, and recommendation letters. For each dimension, Jreve calculates a tension value between 0 and 1.

The formula looks like this:

```
gap  = (|required - current| / required) ^ α
w    = exp(-(remaining_days - time_required) / β)
V(t) = w × gap
```

- **gap** measures how far you are from the requirement, scaled by difficulty
- **w** is a time pressure weight — the closer your deadline, the higher it gets
- **V(t)** is the final tension score for that dimension

If `remaining_days < time_required`, the path is broken — there is not enough time left to close the gap through normal effort. Jreve flags this immediately.

The overall tension across all dimensions is averaged into a single global score. This score drives everything: which model gets called, what tone the response uses, and whether your goal gets added to the task list automatically.

```
V < 0.15   → On track         → Light response, DeepSeek
V < 0.40   → Making progress  → Encouraging, Claude Sonnet  
V < 0.70   → Needs focus      → Direct, Claude Sonnet
V ≥ 0.70   → Critical         → Urgent, Claude Opus
```

---

## Architecture

Jreve routes between multiple AI models depending on what you need and how urgent your situation is. You never have to think about which model is being used — Jreve decides.

```
Claude Opus     → High tension goals, code self-modification
Claude Sonnet   → Conversations, goal analysis, web search
DeepSeek        → Intent classification, parsing, low-tension responses
DALL-E 3        → Image generation
```

Every message goes through four layers before reaching a model:

```
Layer 1  Identity     Fixed agent identity, injected every turn, never drifts
Layer 2  State        Structured world snapshot — your goals, tension scores, tasks
Layer 3  History      Last 3 conversation turns, for fluency only
Layer 4  Compression  Relevant past sessions retrieved by topic similarity
```

The key design decision here: **state is the source of truth, not history**. Your goals, progress, and tension scores live in a versioned `world_snapshot.json`. History is only used to make the conversation feel natural — it does not carry identity or decisions.

Each time your state changes, a new version is written. Nothing is overwritten.

```
state/
    world_snapshot.json     ← current version
    decisions/
        decision_0001.json  ← what changed and why
        decision_0002.json
        ...
```

Past conversations are compressed and stored in a two-level folder structure. When you start a new session, Jreve retrieves only the sessions that are relevant to your current topic — older sessions with low relevance stay on disk but do not consume context.

```
conversations/
    session_20260324_001/
        compressed.json         ← used for topic relevance matching
        folder_two/
            original_001.json   ← injected if session is relevant
```

---

## Getting Started

**Requirements**

- Python 3.11+
- PyQt6
- An Anthropic API key
- A DeepSeek API key
- An OpenAI API key (for DALL-E, optional)

**Install dependencies**

```bash
pip install anthropic openai PyQt6
```

**Set environment variables**

```bash
export ANTHROPIC_API_KEY="your key"
export DEEPSEEK_API_KEY="your key"
export CHATGPT_API_KEY="your key"  # optional
```

**Run**

```bash
python3 Jreve_7.py
```

The first time you launch, Jreve will ask what you want to be called. After that, it starts learning your goals, schedule, and working style from your conversations.

---

## What Jreve Tracks

Once you share a goal, Jreve automatically:

- Searches for the standard requirements for that goal
- Breaks it into measurable dimensions
- Calculates how far you are from each one
- Tracks your progress over time
- Adjusts the urgency of its responses based on how much time you have left

You can also just chat, ask questions, generate images, or get help with code. Jreve classifies what you need and routes accordingly.

---

## Identity

Jreve is developed by Alen Pu.

If you ask what the name means, it will tell you: *Inverse, then you will see.*

---

## License

See `LICENSE` for details.
