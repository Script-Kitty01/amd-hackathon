# Design Document

## Overview

This design implements the `cascade-improvements` requirements as a **two-tier,
accuracy-gate-first router**:

```
/input/tasks.json
    │
    ▼
[ router.route ]  local, zero-token classifier → category + complexity
    │
    ▼
[ tier 0: deterministic solvers ]  provably-correct-or-abstain (0 tokens)
    │  abstain
    ▼
[ tier 1: Fireworks remote ]  category-specialised model, compressed prompt,
    │                          validated + retried, tokens counted
    ▼
[ finalize + validate ] → /output/results.json  (one entry per task, always valid)
```

There is **no bundled local LLM**. Tier 0 answers only what it can prove correct
(currently arithmetic); everything else escalates to Fireworks with a
category-appropriate model and a compressed prompt. Correctness is protected by an
abstain-when-uncertain policy and a post-response validation/retry step; tokens are
minimised by compression, an in-run cache, terse prompts, and tight `max_tokens`.

The design reuses the existing module layout (`router`, `categories`, `prompts`,
`fireworks_client`, `solver`, `cascade`, `io_utils`, `local_solvers`, `finalize`) and
modifies behaviour rather than restructuring, so changes stay isolated and testable.

## Requirements Coverage

| Requirement                                    | Addressed by                                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| R1 Clear accuracy gate                         | Abstain-when-uncertain + validated/retried remote + public-validation eval                          |
| R2 Answer validation                           | `validate.py` form/sanity checks in `Cascade`; empty/truncation retry in `FireworksClient`/`Solver` |
| R3 Conservative deterministic (over-routing=0) | `local_solvers` abstain rules + eval over-routing metric                                            |
| R4 Category-specialised model selection        | `categories.select_model` MODEL_HINTS + `reasoning_effort` control                                  |
| R5 Category answer quality                     | Per-category prompts in `prompts.py` (mixed sentiment, exact summary format, NER labels)            |
| R6 Lean image                                  | `Dockerfile` python-slim, no model, no runtime download                                             |
| R7 Time budget/concurrency                     | `main.run` ThreadPool + `RUNTIME_BUDGET_SECONDS` deadline                                           |
| R8 Token efficiency                            | `compress.py` + `Cascade` normalized cache + `prompts` caps                                         |
| R9 Harness contract                            | `io_utils` tolerant read / atomic write; `main` fallback-per-task, exit 0                           |
| R10 Eval harness                               | `eval/run_eval.py` full-cascade + deterministic-only modes + public dataset                         |

## Architecture

Two tiers behind a zero-token router:

1. **Router (local, 0 tokens):** classifies each task into one of eight categories and
   estimates complexity. Drives both prompt and model choice.
2. **Tier 0 — Deterministic solvers (0 tokens):** answer only provably-correct patterns
   (currently arithmetic); abstain otherwise (over-routing target = 0).
3. **Tier 1 — Fireworks remote (counted tokens):** category-specialised model, compressed
   prompt, response validated and retried on empty/truncation.

An in-run normalized cache short-circuits duplicate prompts (0 tokens). A final
validate + finalize step guarantees a well-formed, complete `results.json`. There is no
bundled local LLM tier. Data flows strictly downward; each tier either resolves the task
or escalates.

## Components and Interfaces

### Routing (tier selection)

- `router.route(prompt) -> RouteResult(category, confidence, complexity, ambiguous)`
  — unchanged scoring classifier (zero tokens). Category drives prompt + model choice.

### Tier 0 — Deterministic solvers (`local_solvers.py`)

- Registry maps a category to solver(s); only `MATH` is enabled by default.
- Each solver returns `Solution(answer, confidence)` or `None` (abstain).
- **Over-routing = 0 (R3):** solvers abstain on any multi-step/ambiguous input; they
  answer only strict, unambiguous patterns (e.g. `X% of Y`, single discount, explicit
  aggregates, two-operand word arithmetic).
- New: an accepted deterministic answer still passes the tier-0 sanity check in
  `validate.py` (e.g. numeric result parses) before being shipped.

### Tier 1 — Fireworks remote (`solver.py`, `fireworks_client.py`, `categories.py`)

- **Model selection (R4):** `select_model(category, ALLOWED_MODELS)` uses `MODEL_HINTS`
  — code→kimi/code, math/logic→minimax, language→gemma — always returning a model that
  is present in `ALLOWED_MODELS`, else the first allowed model. A different escalation
  model is tried only when the primary errors/empties/truncates.
- **Thinking control (R4.5):** `FIREWORKS_REASONING_EFFORT` (default unset) is passed
  via `extra_body`; on a model that rejects it (400), the client learns to drop it for
  that model (0-token failure) and retries.
- **Validation + retry (R2):** `FireworksClient.complete` returns text + usage +
  `finish_reason`. The solver treats empty or `finish_reason == "length"` (truncated)
  as failure and escalates.

### Prompting (`prompts.py`) — R5, R8

- Terse, category-specific system prompts merged into a single user turn (Gemma-safe).
- Category specifics required by the validation examples:
  - **Sentiment:** instruct to label Positive/Negative/Neutral/**Mixed** and, when the
    text has both positive and negative aspects, acknowledge **both** in the reason.
  - **Summarization:** echo and obey the exact stated constraint (sentence/bullet/word
    counts); output only the summary.
  - **NER:** label each entity as PERSON/ORGANIZATION/LOCATION/DATE (match the example
    format) or emit the compact JSON with those types.
- Per-category `max_tokens` caps sized to avoid truncation while staying lean.

### Prompt compression (`compress.py`, new) — R8

- `compress(prompt) -> str`: collapse repeated whitespace/blank lines, strip greeting/
  filler lead-ins, trim trailing politeness — **while preserving fenced code blocks and
  the semantic content verbatim**. Applied only to the remote prompt, never to the task
  text used for routing or caching keys in a way that changes meaning.
- Pure function, unit-tested; must be idempotent and must never drop code or numbers.

### Answer validation (`validate.py`, new) — R2, R5

- `is_valid(category, answer) -> bool` form checks:
  - non-empty after strip;
  - NER: parses as JSON with the expected keys (or contains labelled entities);
  - sentiment: contains a recognised label token;
  - generic: not equal to the fallback sentinel.
- Used by `Cascade` to decide accept-vs-escalate, and as the final guard before write.

### In-run cache (`cache.py` or in `Cascade`) — R8

- Thread-safe dict keyed on a **normalized** prompt (lowercased, whitespace-collapsed).
- Hit → reuse stored `answer`/`category`, record **0** additional tokens, tier=`cache`.
- In-run only; nothing persisted across runs (compliant with "do not cache answers").

### Orchestration (`cascade.py`, `main.py`) — R1, R7, R9

- `Cascade.solve(task_id, prompt)`: cache → route → tier-0 solver (validate) →
  Fireworks (compressed prompt, validated, retried) → finalize. Returns
  `CascadeOutcome(task_id, answer, category, total_tokens, tier)`.
- `main.run`: read tasks (tolerant) → thread pool (bounded) with a wall-clock deadline →
  default every task to a fallback answer → write atomically → exit 0.

### Eval harness (`eval/run_eval.py`, `eval/datasets/public_validation.json`) — R10

- Encode the public validation tasks (T01–T05 + variants) with pass criteria.
- Modes: full cascade (accuracy + tokens + tier breakdown), `--cloud-only` baseline,
  and a deterministic-only offline mode reporting an **over-routing count** (target 0).

## Data Models

```python
@dataclass(frozen=True)
class RouteResult:
    category: Category
    confidence: float
    complexity: str          # "easy" | "complex"
    ambiguous: bool

@dataclass(frozen=True)
class Solution:              # tier-0 deterministic result
    answer: str
    confidence: float

@dataclass
class LLMResult:             # Fireworks response
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str       # NEW: to detect truncation

@dataclass
class CascadeOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int
    tier: str                # deterministic | remote | cache | fallback
```

## Error Handling

- **Empty / truncated remote answer:** escalate to the escalation model; if still bad,
  write the best available text or the fallback sentinel (never crash).
- **Remote API error / rate limit:** exponential backoff retry in `FireworksClient`;
  give up after N attempts and escalate/fallback.
- **Malformed input task:** `read_tasks` skips non-dict items and defaults missing
  fields; a task with no usable prompt still gets a fallback answer.
- **Time budget exceeded:** stop scheduling new work; unfinished tasks keep their
  fallback answer; results file is still written.
- **Any per-task exception:** caught in `main.run`; the task keeps its fallback; the run
  completes and exits 0.

## Correctness Properties

### Property 1: No empty/truncated answer shipped when a retry is possible

If a remote response is empty or `finish_reason == "length"` (truncated), the system
escalates to the escalation model; a fallback sentinel is written only after all
attempts are exhausted.
**Validates: Requirements 2.2, 2.3**

### Property 2: Deterministic answers are correct or absent

A tier-0 solver returns a result only for patterns it can compute exactly; every other
input abstains (over-routing count = 0).
**Validates: Requirements 3.1, 3.2**

### Property 3: Compression preserves meaning

`compress` never removes code, numbers, or entities, and is idempotent:
`compress(compress(x)) == compress(x)`.
**Validates: Requirements 8.1**

### Property 4: Cache reuse is answer-preserving and free

A normalized-prompt cache hit returns the same answer as the first computation and adds
zero remote tokens.
**Validates: Requirements 8.2**

### Property 5: Completeness

For every input `task_id` there is exactly one output entry with the same id; the run
always writes valid JSON and exits 0.
**Validates: Requirements 9.1, 9.2, 9.3**

### Property 6: Compliance

Remote calls only ever use a model present in `ALLOWED_MODELS` and go through
`FIREWORKS_BASE_URL`; nothing is persisted across runs.
**Validates: Requirements 4.4**

## Testing Strategy

- **Unit:** router categorisation (incl. no-misroute cases), each deterministic solver
  (correct cases + abstain cases), `compress` (idempotent, preserves code/numbers),
  `validate` (accept/reject per category), model selection over the real ALLOWED_MODELS,
  cache reuse, `io_utils` read/write hardening.
- **Integration (offline, fake client):** full read→cascade→write; every task gets a
  non-empty answer; results JSON valid and complete; one entry per task id preserved.
- **Eval (with stand-in remote):** run the public validation dataset; assert overall
  pass rate ≥ 90% target and report tokens + tier breakdown; deterministic-only mode
  asserts over-routing count == 0.
- **Container check (build machine):** build image, confirm compressed size and a clean
  `docker run` over sample input produces valid `/output/results.json`.
