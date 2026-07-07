# Token-Efficient AI Agent — Track 1

A general-purpose AI agent that solves natural-language tasks across eight
capability domains using **Fireworks AI** models, optimised to pass an accuracy
gate while spending the **fewest possible tokens**.

---

## 1. Problem Statement

Build a containerised AI agent that:

1. Reads tasks from `/input/tasks.json` on startup.
2. Answers each task by calling Fireworks AI models (all inference must go
   through `FIREWORKS_BASE_URL`).
3. Writes results to `/output/results.json` before exiting.

### Capability categories (evaluated across all eight)

| #   | Category                 | What it covers                                                       |
| --- | ------------------------ | -------------------------------------------------------------------- |
| 1   | Factual knowledge        | Explaining concepts, definitions, how things work                    |
| 2   | Mathematical reasoning   | Multi-step arithmetic, percentages, word problems, projections       |
| 3   | Sentiment classification | Labelling sentiment and justifying the classification                |
| 4   | Text summarisation       | Condensing passages to a specific format or length constraint        |
| 5   | Named entity recognition | Extracting and labelling entities (person, org, location, date) JSON |
| 6   | Code debugging           | Identifying bugs and providing corrected implementations             |
| 7   | Logical / deductive      | Constraint-based puzzles where all conditions must be satisfied      |
| 8   | Code generation          | Writing correct, well-structured functions from a spec               |

### I/O contract

**Input** — `/input/tasks.json`

```json
[
  {
    "task_id": "t1",
    "prompt": "Summarise the following text in one sentence: ..."
  },
  { "task_id": "t2", "prompt": "..." }
]
```

**Output** — `/output/results.json`

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

### Runtime environment variables (injected by the harness — never hardcode)

| Variable             | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `FIREWORKS_API_KEY`  | Provided by the harness — use this key, not your own.     |
| `FIREWORKS_BASE_URL` | Base URL for **all** Fireworks calls.                     |
| `ALLOWED_MODELS`     | Comma-separated permitted model IDs (revealed at launch). |

### Hard rules

- Exit code `0` on success, non-zero on failure.
- Maximum runtime: **10 minutes**.
- Only models in `ALLOWED_MODELS` are permitted.
- `/output/results.json` must be valid JSON (malformed output scores zero).
- All inference must go through `FIREWORKS_BASE_URL` (bypassing it records zero
  tokens and invalidates the submission).
- Do not hardcode or cache answers (evaluation uses unseen prompt variants).
- Image compressed size must not exceed **10 GB**.
- Submissions are rate-limited to **10 per hour** per team.

### Scoring

1. **Accuracy gate** — an LLM judge evaluates each answer against expected
   intent. Submissions below the threshold are excluded from the leaderboard.
2. **Token efficiency** — passing submissions are ranked ascending by **total
   tokens (input + output)** recorded by the judging proxy. Fewer tokens wins.

> Key insight: the token ranking is **raw token count, not weighted by model
> size**. Model choice therefore matters for _accuracy and one-shot success_,
> not for the token score directly.

---

## 2. Our Solution Approach

The strategy is a two-phase discipline: **pass the accuracy gate first, then
minimise tokens down to (not through) that gate.**

### Core principles

1. **Free local routing.** A local, zero-token classifier tags each task with
   its category. This costs nothing and picks the right prompt strategy.
2. **Minimal prompts.** Terse, category-specific system prompts. No few-shot
   examples unless a category provably fails without them — every prompt token
   is paid on every call.
3. **Constrained outputs.** Each category has a hard `max_tokens` and an output
   shape that requests the _minimum_ the judge needs (a label + short reason,
   a length-bound summary, compact JSON, code only, etc.).
4. **One call per task.** No multi-pass self-critique or speculative retries.
   Retries happen only on API errors.
5. **Accuracy-safe reasoning.** Math and logic tasks get a brief scratchpad,
   then a clearly delimited final answer — enough reasoning to stay correct,
   no more.
6. **Never score zero.** Robust error handling guarantees a valid, complete
   `results.json` even if individual calls fail.

### Request pipeline (per task)

```
tasks.json
   │
   ▼
[ router.py ]  local, zero-token category classification
   │
   ▼
[ prompts.py ] select category template + max_tokens
   │
   ▼
[ fireworks_client.py ] single Fireworks call via FIREWORKS_BASE_URL
   │
   ▼
[ solver.py ] validate / fallback on error
   │
   ▼
results.json
```

### Why this maps to a real-world problem

This is **LLM cost optimisation under a quality SLA** — the exact challenge of
running LLMs in production: get correct answers while minimising the API bill.
The accuracy gate is the quality SLA; the token ranking is the cost. Every
technique here (intent routing, prompt compression, output constraints, eval
observability, graceful degradation) is standard production practice.

---

## 3. Project Structure

```
token-efficient-agent/
├── README.md                 # this file
├── Dockerfile                # lean container definition
├── requirements.txt          # runtime dependencies
├── .dockerignore
├── .gitignore
├── .env.example              # template for local dev (never commit real keys)
│
├── src/                      # application code
│   ├── __init__.py
│   ├── main.py               # entrypoint: orchestrates read → solve → write
│   ├── config.py             # loads env vars (API key, base URL, models)
│   ├── categories.py         # canonical category names + per-category config
│   ├── router.py             # local, zero-token category classifier
│   ├── prompts.py            # per-category system prompts + max_tokens
│   ├── fireworks_client.py   # Fireworks API wrapper (OpenAI-compatible)
│   ├── solver.py             # per-task orchestration + fallback handling
│   └── io_utils.py           # read tasks.json / write results.json
│
├── eval/                     # local evaluation harness (not shipped in image)
│   ├── __init__.py
│   ├── run_eval.py           # run agent over sample tasks, sweep settings
│   ├── scorer.py             # accuracy checks + token accounting
│   └── datasets/
│       └── sample_tasks.json # representative tasks per category
│
├── tests/                    # unit tests for stable components
│   ├── __init__.py
│   ├── test_router.py
│   └── test_io.py
│
└── data/                     # local-run I/O (mirrors container /input, /output)
    ├── input/
    │   └── tasks.json        # sample input for local runs
    └── output/
        └── .gitkeep
```

### Why this structure is stable

- **`src/` separates concerns by responsibility**, not by category. Adding or
  tuning a category touches only `prompts.py` and `categories.py` — never the
  orchestration, I/O, or client code.
- **Routing, prompting, and the API client are decoupled**, so swapping the
  classifier (rules → trained model) or the model-selection logic is isolated.
- **`eval/` lives outside `src/`** and is excluded from the image, keeping the
  container lean while giving us a full offline tuning loop.
- **`data/` mirrors the container's `/input` and `/output`**, so local runs and
  graded runs behave identically.

---

## 4. Local Development

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Copy env template and fill in dev values
cp .env.example .env

# 3. Run against local sample tasks
python -m src.main            # reads data/input, writes data/output

# 4. Run the eval harness (accuracy + token report)
python -m eval.run_eval
```

> For local dev you may use a `.env` file. The **submitted container must read
> config purely from the environment** — do not bundle `.env` in the image.

## 5. Build & Submit

```bash
# Build
docker build -t token-efficient-agent .

# Test locally (mount input/output, inject env)
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  -v "$(pwd)/data/input:/input" \
  -v "$(pwd)/data/output:/output" \
  token-efficient-agent

# Push to a public registry
docker tag token-efficient-agent <registry>/token-efficient-agent:latest
docker push <registry>/token-efficient-agent:latest
```

## 6. Launch-Day Checklist

1. Populate `ALLOWED_MODELS` and run the eval harness.
2. Per category, pick the smallest model that clears the accuracy gate.
3. Sweep each template's verbosity / `max_tokens` down to the gate.
4. Run full eval, confirm the gate passes with margin, then trim tokens.
5. Build, push, submit — mind the 10/hour rate limit.
