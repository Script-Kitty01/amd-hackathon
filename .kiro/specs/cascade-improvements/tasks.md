# Implementation Plan

- [x] 1. Detect truncation in the Fireworks client
  - Add `finish_reason` to `LLMResult` and capture it in `FireworksClient.complete`.
  - _Requirements: 2.3_

- [x] 2. Prompt compression module (`src/compress.py`)
  - Meaning-preserving, idempotent; collapse whitespace, strip filler lead-ins,
    preserve fenced code blocks and all numbers/entities.
  - Unit tests: idempotence, code/number preservation, whitespace collapse.
  - _Requirements: 8.1_

- [x] 3. Answer validation module (`src/validate.py`)
  - `is_valid(category, answer)`: non-empty, not the fallback sentinel; NER must be
    JSON-ish; sentiment must contain a recognised label.
  - Unit tests per category (accept/reject).
  - _Requirements: 2.1, 2.4, 5.1, 5.3_

- [x] 4. Escalate on empty/truncated/invalid remote answers (`src/solver.py`)
  - Compress the prompt before the remote call; treat empty, `finish_reason=="length"`,
    or `is_valid`==False as failure and try the escalation model; keep best partial.
  - Tests with a fake client: truncated→escalate, invalid→escalate, valid→accept.
  - _Requirements: 2.1, 2.2, 2.3, 8.1_

- [x] 5. Normalized in-run cache (`src/cascade.py`)
  - Key on normalized prompt (strip + collapse whitespace + lowercase); hit → reuse,
    0 tokens, tier=`cache`.
  - Test: near-duplicate prompts hit the cache.
  - _Requirements: 8.2_

- [x] 6. Category answer-quality prompts (`src/prompts.py`)
  - Sentiment: allow Mixed + justification must cover both sides.
  - Summarization: obey the exact stated format/length constraint.
  - NER: label every entity (PERSON/ORGANIZATION/LOCATION/DATE) as compact JSON.
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. Category-specialised model selection (verify `src/categories.py`)
  - Confirm MODEL_HINTS route code→kimi, math/logic→minimax, language→gemma, and only
    ever return an ALLOWED_MODELS entry; keep `reasoning_effort` handling.
  - Tests over the real allowed-model list.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 8. Two-tier wiring (`src/main.py`, Dockerfile)
  - Confirm no local LLM is configured in the shipped path (deterministic + Fireworks),
    lean python-slim image, no model, no runtime download, atomic write, exit 0.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.1, 9.2, 9.3, 9.4_

- [x] 9. Public validation eval harness (`eval/datasets/public_validation.json`)
  - Encode the published T01–T05 tasks with pass criteria; add a deterministic-only
    over-routing check (target 0).
  - _Requirements: 10.1, 10.2, 10.3, 3.2_

- [x] 10. Full test pass + offline integration
  - Run the whole suite; confirm all green and the offline integration produces a
    valid, complete results.json.
  - _Requirements: 1.2, 9.1, 9.2_
