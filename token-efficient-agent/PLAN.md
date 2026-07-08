# Execution Plan — Token-Efficient Agent (Track 1)

Phase 0 artifact: component inventory, milestones, and trackable subtasks.
Keep this file updated as the single source of truth for status.

---

## 1. Component Inventory

Status legend: ✅ done · 🟡 partial / needs tuning · ⬜ missing · ❔ conditional

| Component                 | File                              | Status | Notes                                                               |
| ------------------------- | --------------------------------- | ------ | ------------------------------------------------------------------- |
| Config loader (env-only)  | `src/config.py`                   | ✅     | Reads API key, base URL, `ALLOWED_MODELS`, budget.                  |
| Task I/O                  | `src/io_utils.py`                 | ✅     | Atomic, valid-JSON write. Solid.                                    |
| Categories + model select | `src/categories.py`               | 🟡     | `MODEL_PREFERENCE` all index 0 — needs launch-day tuning.           |
| Router / classifier       | `src/router.py`                   | 🟡     | Regex/keyword only. No confidence, ambiguity, or complexity signal. |
| Prompt templates          | `src/prompts.py`                  | 🟡     | Terse per-category specs exist; need token/accuracy tuning.         |
| Fireworks client          | `src/fireworks_client.py`         | ✅     | Single deterministic call, temp=0, usage tracked.                   |
| Solver orchestration      | `src/solver.py`                   | 🟡     | One retry + fallback. No tiered escalation, no dedup cache.         |
| Entrypoint                | `src/main.py`                     | ✅     | Concurrent, wall-clock budget, safe fallbacks.                      |
| Eval runner               | `eval/run_eval.py`                | 🟡     | Single-model run only. No model sweep.                              |
| Scorer                    | `eval/scorer.py`                  | 🟡     | Naive substring match; token accounting good. No per-model view.    |
| Sample dataset            | `eval/datasets/sample_tasks.json` | ❔     | Verify it covers all 8 categories + easy/complex variants.          |
| Unit tests                | `tests/`                          | 🟡     | `test_io`, `test_router` exist; extend as logic grows.              |
| Container                 | `Dockerfile`                      | ❔     | Verify build + image < 10 GB, no bundled `.env`.                    |
| Tiered model escalation   | —                                 | ⬜     | Cheap-first, escalate on low-confidence/failure.                    |
| Ambiguous-query routing   | —                                 | ⬜     | Fallback category + tie-break rules.                                |
| In-run dedup cache        | —                                 | ⬜     | Reuse byte-identical prompt results within one run.                 |
| RAG / vector DB           | —                                 | ❔     | Only if tasks include a document corpus. Currently N/A.             |

---

## 2. Open Questions (resolve before building)

1. **Do tasks include documents to retrieve over?** RESOLVED: No — tasks are standalone prompts. RAG dropped (Task 16 = N/A).
2. **Exact Fireworks model IDs** — unknown until `ALLOWED_MODELS` is revealed at launch. All model tuning is launch-day work.
3. **Gemma 4 = bonus, not mandate.** RESOLVED: it is a _bonus challenge_ to make the best use of gemma 4, not a hard requirement to use only gemma. Tiered model selection stays; additionally, deliberately route categories where gemma clears the accuracy gate cheaply to gemma to capture the bonus (see T19). Confirm the exact gemma model ID available via `FIREWORKS_BASE_URL` at launch.

---

## 3. Milestones & Acceptance Criteria

| Milestone                | Covers      | Done when                                                                                                              |
| ------------------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| **M0 — Plan**            | Tasks 1–2   | This file exists, subtasks tracked, open questions listed.                                                             |
| **M1 — Routing**         | Tasks 3–5   | Router validated on samples; ambiguous inputs route to a safe default; complexity signal available.                    |
| **M2 — Model selection** | Tasks 6–8   | Tiered escalation works; `MODEL_PREFERENCE` driven by eval sweep output.                                               |
| **M3 — Validate & test** | Tasks 9–12  | Scorer reports accuracy + tokens; dataset covers 8 categories; always-valid `results.json`; runs within 10-min budget. |
| **M4 — Optimize**        | Tasks 13–16 | Dedup cache live; prompts + `max_tokens` swept to the accuracy gate.                                                   |
| **M5 — Ship**            | Tasks 17–18 | Image builds < 10 GB, env-only; full integration run passes; submitted.                                                |

---

## 4. Subtask Backlog (19 items · 69 pts, 61 without RAG)

Points scale: 1 trivial · 2 small · 3 moderate · 5 substantial · 8 large.

### M0 — Plan & breakdown ✅ COMPLETE

- [x] **T1** (2) Inventory components + split into workable units (this file).
- [x] **T2** (2) Define milestones + acceptance criteria (Jira CSV available on request).

### M1 — Routing / classification ✅ COMPLETE

- [x] **T3** (3) Validate `router.py` on sample tasks — was 7/8; fixed logic detection (ranking/deduction puzzles), now 8/8. Regression test added.
- [x] **T4** (5) Scoring-based router: highest-score category wins; confidence = winning share of total signal; `ambiguous` flag + `factual` safe default. `route()` added, `classify()` kept stable.
- [x] **T5** (3) Complexity heuristic (easy/complex) on `RouteResult` from category class, length, reasoning cues, and clause density — feeds tiered escalation (T6).

### M2 — Model selection ✅ COMPLETE (logic built; final tuning is launch-day)

- [x] **T6** (5) Tiered escalation in `solver.py`: route() picks the tier — easy/high-confidence starts on cheap preferred model, complex/ambiguous goes straight to strong; escalate on error/empty; tokens summed across attempts. Tested with a fake client.
- [x] **T7** (2) `MODEL_PREFERENCE` data-driven: `load_model_preference()` overlays `config/model_preference.json` at startup; safe all-zero default when absent.
- [x] **T8** (5) `run_eval.py --sweep`: runs every ALLOWED_MODEL × task, prints pass-rate/token table per category, writes recommended `config/model_preference.json`. (Run at launch with real models.)
- [x] **T19** (3) 🎁 Sweep recommendation prefers a gemma model among gate-passers when within 10% of best tokens, so gemma is chosen wherever competitive. (Confirm gemma model ID at launch.)

### M3 — Validate & test

- [ ] **T9** (3) Verify scorer: accuracy gate + token accounting.
- [ ] **T10** (3) Expand `sample_tasks.json`: all 8 categories + easy/complex variants.
- [ ] **T11** (2) Harden fallback so `results.json` is always valid.
- [ ] **T12** (3) Test concurrency + 10-min wall-clock budget under load.

### M4 — Optimization

- [ ] **T13** (2) In-run dedup cache for identical prompts (no persisted answers).
- [ ] **T14** (5) Tune per-category system prompts for terseness.
- [ ] **T15** (3) Sweep per-category `max_tokens` down to the accuracy gate.
- [x] ~~**T16** (8) Dynamic RAG (adaptive top-K)~~ — DROPPED: tasks are standalone, no corpus.

### M5 — Ship

- [ ] **T17** (2) Docker build < 10 GB, env-only config (no bundled `.env`).
- [ ] **T18** (3) Full integration run → submit (mind 10/hr rate limit).

---

## 5. Suggested Order (mentor's sequence)

M0 → M1 (classification) → M3 (validate & test) → M2 + M4 (model + prompt/RAG optimization) → M5 (ship).
Start with the easiest, self-contained pieces. Do not start feature work before M0 is complete.
