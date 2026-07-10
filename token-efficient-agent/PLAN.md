# Execution Plan — Token-Efficient Agent (Track 1)

Single source of truth for architecture, milestones, and trackable subtasks.

> **Strategy pivot (2026-07-09):** LabLab Admin confirmed a task's final answer
> **may be produced by a local model**; the goal is to use local models to
> minimize Fireworks calls (local inference = 0 tokens). Original design: a
> local-first confidence cascade.
>
> **HARDWARE REALITY (2026-07-10) — critical:** the grading box is **CPU-only,
> 4 GB RAM, 2 vCPU, no GPU.** A 7–9 GB local LLM (e.g. `gemma4:e4b`/`e2b`)
> **cannot load** in 4 GB, and even a ~2 GB model is too slow/RAM-heavy on
> 2 vCPU to be worth it. So the "answer everything with local gemma" result we
> measured on a dev machine **does NOT transfer to grading.**
>
> **Revised grading architecture:** bundle **no general local LLM**. Keep the
> deterministic local solvers (math / sentiment / NER via spaCy) — they use
> negligible RAM, run instantly, and answer their categories at **0 tokens**.
> Everything needing a real LLM (factual, summarization, logic, code, hard math)
> goes to the **Fireworks allowed models**, with tokens minimized (terse prompts,
> tight `max_tokens`, thinking disabled, per-category model routing). The code
> already degrades to this automatically when no local LLM is configured
> (`LocalLLM.from_env()` returns None → cascade = deterministic → Fireworks).
> The local gemma path remains available for richer dev/self-host environments.

---

## 1. Target Architecture — Local-First Confidence Cascade

```
/input/tasks.json
   │
   ▼
[ router.classify ]  local regex/scoring classifier — 0 tokens        (top layer)
   │  category + confidence + complexity
   ▼
[ local solver tool (per category) ]  deterministic / NLP             (cheap tier)
   │  try_solve -> (answer, confidence) | ABSTAIN
   │     high confidence ─────────────────────────────► answer  (0 tokens)
   │     abstain / low confidence
   ▼
[ local LLM: gemma via ROCm ]  generative, on-box                     (local tier)
   │     confident ─────────────────────────────────────► answer  (0 tokens)
   │     low confidence
   ▼
[ Fireworks API (ALLOWED_MODELS) ]  paid fallback, tiered cheap→strong (fallback)
   │
   ▼
/output/results.json
```

**Core principles**

1. **Local-first.** Every stage that answers locally costs zero tokens. Fireworks
   is the fallback of last resort, not the default path.
2. **Confidence by construction.** Each tool answers only when it is confident
   (returns an answer) or _abstains_ and passes the task down the cascade. No
   separate ML confidence model — use each tool's intrinsic certainty signals +
   simple, calibrated thresholds (per mentor guidance).
3. **Protect the accuracy gate.** A confidently-wrong local answer that skips
   Fireworks is the main risk. Thresholds are **calibrated against the judge** on
   the eval set; reasoning-heavy categories escalate conservatively.
4. **Minimize what Fireworks produces.** When we must call Fireworks, ask for the
   bare minimum and do formatting/assembly locally (free).
5. **Never score zero.** Always emit a valid, complete `results.json`.

---

## 2. Resolved Questions

1. **Documents to retrieve over?** No — tasks are standalone prompts. RAG
   (chunking / vector store / retrieval / top-K) is **not applicable** and stays
   out. The local-first ruling does not create a corpus.
2. **Local answering allowed?** YES (LabLab Admin, 2026-07-09). Local models may
   produce final answers; goal is to minimize Fireworks calls. → local-first.
3. **Gemma 4** — bonus for best use. Running gemma **locally** as the primary
   generative tier is now both the token strategy _and_ the bonus. AMD hardware
   (ROCm) is the likely target for local inference.
4. **Fireworks model IDs** — KNOWN: `minimax-m3`, `kimi-k2p7-code`,
   `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`. Fallback
   prefers `kimi-k2p7-code` (code) and `minimax-m3` (reasoning) via MODEL_HINTS;
   the launch-day sweep confirms/overrides per category. 3 of 5 are Gemma 4 —
   same family as the local `gemma4:e4b`, reinforcing the gemma-4 bonus story.

---

## 3. Component Inventory

Status: ✅ done · 🟡 partial · ⬜ missing · ❌ dropped

| Component                          | File                              | Status | Role in cascade                                               |
| ---------------------------------- | --------------------------------- | ------ | ------------------------------------------------------------- | --------- |
| Config loader (env-only)           | `src/config.py`                   | ✅     | Fireworks creds + budget.                                     |
| Task I/O                           | `src/io_utils.py`                 | ✅     | Read/write, atomic valid JSON.                                |
| Router / classifier                | `src/router.py`                   | ✅     | Top layer: category + confidence + complexity.                |
| Fireworks client                   | `src/fireworks_client.py`         | ✅     | Fallback tier transport.                                      |
| Categories + model policy          | `src/categories.py`               | ✅     | Fallback-tier model selection.                                |
| Solver (Fireworks tiered)          | `src/solver.py`                   | 🟡     | Becomes the **fallback** stage of the cascade.                |
| Entrypoint                         | `src/main.py`                     | 🟡     | Must drive the cascade, not just the solver.                  |
| Eval runner + scorer               | `eval/`                           | 🟡     | Needs judge-based accuracy + per-stage token/escalation view. |
| Sample dataset                     | `eval/datasets/sample_tasks.json` | 🟡     | Expand to 8 cats + easy/complex.                              |
| **Local solver interface**         | —                                 | ⬜     | `try_solve(prompt)->(answer,conf)                             | ABSTAIN`. |
| **Math solver (deterministic)**    | —                                 | ⬜     | Parse+evaluate; conf = clean parse.                           |
| **Sentiment (local classifier)**   | —                                 | ⬜     | conf = class probability.                                     |
| **NER (local, spaCy)**             | —                                 | ⬜     | conf = entity/model score.                                    |
| **Local LLM runtime (gemma/ROCm)** | —                                 | ⬜     | Generative local tier.                                        |
| **Cascade orchestrator**           | —                                 | ⬜     | Runs stages, applies thresholds, escalates.                   |
| **Confidence thresholds/config**   | —                                 | ⬜     | Calibrated per tool against the judge.                        |
| **In-run dedup cache**             | —                                 | ⬜     | Reuse identical-prompt results.                               |
| Container                          | `Dockerfile`                      | 🟡     | Must bundle gemma weights, < 10 GB, ROCm.                     |
| RAG / vector DB                    | —                                 | ❌     | Dropped — no corpus.                                          |

---

## 4. Milestones & Acceptance Criteria

| Milestone                             | Done when                                                                                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **M0 — Plan** ✅                      | Architecture + backlog captured (this file).                                                                                                   |
| **M1 — Local routing (top layer)** ✅ | Router returns category + confidence + complexity; validated on samples.                                                                       |
| **M2 — Fireworks fallback tier** ✅   | Tiered cheap→strong Fireworks escalation + data-driven model preference built and tested.                                                      |
| **M3 — Local answering layer**        | Deterministic solvers (math/NER/sentiment) + local gemma runtime each answer with a confidence signal or abstain.                              |
| **M4 — Confidence & cascade**         | Orchestrator runs local→Fireworks with calibrated thresholds; escalates only on low confidence; gate protected.                                |
| **M5 — Validate & test**              | Judge-based eval reports accuracy + tokens + escalation rate; dataset covers 8 cats; always-valid output; fits 10-min budget incl. model load. |
| **M6 — Optimize (local-first)**       | Dedup cache live; thresholds tuned to maximize local-answer rate at/above gate; Fireworks output minimized.                                    |
| **M7 — Ship**                         | Image < 10 GB with gemma weights, env-only, ROCm inference verified in-container; full run passes; submitted.                                  |

---

## 5. Subtask Backlog

Points: 1 trivial · 2 small · 3 moderate · 5 substantial · 8 large.

### M0 — Plan ✅

- [x] **T1** (2) Component inventory + breakdown.
- [x] **T2** (2) Milestones + acceptance criteria.

### M1 — Local routing (top layer) ✅

- [x] **T3** (3) Validate router on samples (fixed logic detection, 8/8).
- [x] **T4** (5) Scoring router: category + confidence + `ambiguous` + safe default.
- [x] **T5** (3) Easy/complex complexity heuristic on `RouteResult`.

### M2 — Fireworks fallback tier ✅

- [x] **T6** (5) Tiered escalation (now the paid fallback stage).
- [x] **T7** (2) Data-driven `MODEL_PREFERENCE` via config file.
- [x] **T8** (5) `run_eval.py --sweep` model sweep + recommendation.
- [x] **T19** (3) 🎁 Gemma preference in sweep recommendation.

### M3 — Local answering layer ✅ (T25 partial)

- [x] **T20** (5) Local solver interface + registry in `src/local_solvers.py`: `try_solve(prompt) -> Solution(answer, confidence) | None (abstain)`, keyed by category. No third-party deps.
- [x] **T21** (3) Deterministic math solver: `X% of Y`, percentage discount/increase on a price, bare arithmetic; abstains on anything unclear to protect the gate. 8 tests.
- [x] **T22** (3) Sentiment: local lexicon solver with negation handling; confidence from margin. Abstains without signal.
- [x] **T23** (3) NER: spaCy `en_core_web_sm` solver (conf 0.85, accepted locally) with the regex heuristic as fallback if the model is absent → compact JSON. Now answers locally at 0 tokens.
- [x] **T24** (8) Local LLM tier (`src/local_llm.py`): OpenAI-compatible client for a local endpoint (Ollama/gemma), enabled via env, self-consistency confidence, graceful abstain when absent. Tokens ignored (local=0).
- [~] **T25** (3) Summarization/factual handled generically by the local LLM tier. Map-reduce/refine for over-long single inputs = deferred nice-to-have.

### M4 — Confidence & cascade orchestration ✅

- [x] **T26** (5) Cascade orchestrator (`src/cascade.py`): router → local solver → local LLM → Fireworks, threshold-gated per hop; records tier + tokens. Wired into `main.py`.
- [x] **T27** (5) Threshold machinery (`src/thresholds.py`): per-tier/category cutoffs, overlaid from `config/confidence.json`. Live calibration against the judge = launch-day.
- [x] **T28** (3) Escalation policy: conservative defaults — trust deterministic math; force local-LLM math/logic to escalate to Fireworks.

### M5 — Validate & test ✅

- [x] **T9** (3) Scorer reports tokens, tier breakdown, and **local-answer rate**; eval now runs the cascade.
- [x] **T10** (3) `sample_tasks.json` expanded to all 8 categories × easy/complex (16 tasks, with `expected`).
- [x] **T11** (2) `read_tasks` hardened: tolerates missing file/keys, non-dict items, non-list, unicode. Tests added.
- [x] **T12** (3) Concurrency test: 60 tasks through the cascade in a thread pool, no lost/duplicate ids, all answered. (Local-model load-time budget = launch-day, T30.)

### M6 — Optimize (local-first) ✅ (T14/T15 launch-day)

- [x] **T13** (2) In-run dedup cache in `Cascade`: identical prompts reuse the first result (0 extra tokens), thread-safe.
- [x] **T29** (3) `src/finalize.py`: extract value from math `Answer:` line, compact NER JSON, strip — applied to every accepted answer.
- [~] **T14** (5) Fallback prompts already terse in `prompts.py`; dynamic few-shot + tuning = launch-day (needs real models).
- [~] **T15** (3) `max_tokens` caps exist per category; sweeping them to the gate = launch-day.

### M7 — Ship (T30/T18 launch-day)

- [x] **T17** (3) Dockerfile lean (python-slim), deps trimmed to `openai`, env-only, copies `config/`. ROCm+gemma bundling documented as an option.
- [ ] **T30** (3) Verify local inference in-container on target AMD hardware within budget — **launch-day (needs hardware)**.
- [~] **T18** (3) Offline integration test passes (read→cascade→write→valid results.json). Live submission = launch-day.

### Dropped

- [x] ~~**T16** Dynamic RAG (adaptive top-K)~~ — no corpus.

---

## 6. Suggested Order

M3 (local solvers, deterministic first: math → sentiment → NER → local LLM) →
M4 (orchestrator + threshold calibration) → M5 (validate) → M6 (optimize) →
M7 (ship). Deterministic solvers are the safest zero-token wins; do them first,
then the local LLM, then wire the cascade and calibrate against the judge.

---

## 7. Key Risks

- **Confidently-wrong local answers** miss the gate for zero token savings →
  mitigated by judge-calibrated thresholds + conservative reasoning cutoffs (T27/T28).
- **Image size**: gemma weights + deps must stay < 10 GB (quantized) (T17).
- **Runtime**: local model load + inference must fit 10 min at task volume (T12/T30).
- **Local-answer ambiguity boundary**: routing/post-processing are clearly local;
  keep answer-generation confined to gemma/tools + Fireworks as ruled.
