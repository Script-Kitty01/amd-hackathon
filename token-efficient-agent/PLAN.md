# Execution Plan — Token-Efficient Agent (Track 1)

Single source of truth for architecture, milestones, and trackable subtasks.

> **Strategy pivot (2026-07-09):** The organizers (LabLab Admin) confirmed a
> task's final answer **may be produced by a local model**, as long as it meets
> the accuracy threshold — and that the _goal_ is to use local models as much as
> possible to minimize Fireworks API calls. Since local inference counts as
> **zero tokens**, the winning design is a **local-first confidence cascade**:
> answer everything we can locally, and call Fireworks only when a local answer
> can't confidently clear the gate. This reshapes M3+ below.

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
4. **Fireworks model IDs** — unknown until `ALLOWED_MODELS` is published at
   launch; fallback-tier tuning is launch-day work.

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

### M3 — Local answering layer ⬅ IN PROGRESS

- [x] **T20** (5) Local solver interface + registry in `src/local_solvers.py`: `try_solve(prompt) -> Solution(answer, confidence) | None (abstain)`, keyed by category. No third-party deps.
- [x] **T21** (3) Deterministic math solver: `X% of Y`, percentage discount/increase on a price, bare arithmetic; abstains on anything unclear to protect the gate. 8 tests.
- [ ] **T22** (3) Sentiment: local classifier/lexicon; confidence = class probability.
- [ ] **T23** (3) NER: local spaCy pipeline → required JSON shape; confidence = entity/model scores.
- [ ] **T24** (8) Local LLM runtime: gemma via Ollama/llama.cpp on ROCm; wrapper matching the solver interface; confidence via logprobs and/or self-consistency.
- [ ] **T25** (3) Summarization/factual handling via local LLM (+ extractive fallback for over-long inputs: map-reduce/refine on the _task's own_ text).

### M4 — Confidence & cascade orchestration

- [ ] **T26** (5) Cascade orchestrator: router → local tool → local LLM → Fireworks, abstain/threshold-gated at each hop; token + stage accounting.
- [ ] **T27** (5) Threshold calibration: on eval set, find per-tool confidence cutoffs where answers pass the judge; persist to `config/confidence.json`.
- [ ] **T28** (3) Escalation policy: conservative cutoffs on reasoning categories (logic/math word problems/code) to protect the gate.

### M5 — Validate & test

- [ ] **T9** (3) Judge-based scorer: accuracy gate signal + tokens + **escalation rate** per category.
- [ ] **T10** (3) Expand `sample_tasks.json`: all 8 categories + easy/complex variants.
- [ ] **T11** (2) Harden pipeline so `results.json` is always valid (bad input, empty prompt, unicode).
- [ ] **T12** (3) Concurrency + 10-min budget **including local model load time**.

### M6 — Optimize (local-first)

- [ ] **T13** (2) In-run dedup cache for identical prompts.
- [ ] **T29** (3) Minimize Fireworks output: request minimum, assemble/format locally (NER JSON, sentiment wording, extract `Answer:` line).
- [ ] **T14** (5) Fallback-tier prompt tuning + dynamic few-shot **only** where a category misses the gate.
- [ ] **T15** (3) `max_tokens` tuning for the Fireworks fallback tier.

### M7 — Ship

- [ ] **T17** (3) Dockerfile: bundle gemma weights, ROCm base, env-only, image < 10 GB.
- [ ] **T30** (3) Verify local inference runs in-container on target hardware within the 10-min budget.
- [ ] **T18** (3) Full integration run → submit (10/hr limit).

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
