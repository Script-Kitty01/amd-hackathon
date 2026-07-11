# Requirements Document

## Introduction

This spec covers a set of improvements to the Track 1 token-efficient routing agent
(`token-efficient-agent`), adopting proven, general techniques observed in comparable
open-source agents while keeping all implementation original.

Context and constraints (from the official Track 1 spec and judging FAQ):

- **Grading hardware:** CPU-only, 4 GB RAM, 2 vCPU, no GPU.
- **Scoring:** an accuracy gate (approx. 80%) is applied first; only submissions that
  pass are ranked, ascending, by total Fireworks tokens. **Correctness comes first;
  token efficiency matters only after the gate is cleared.**
- **A prior submission scored ~31.6% accuracy** — well below the gate — so the primary
  goal of this spec is to reliably clear the accuracy gate, then minimise tokens.
- **Compliance:** all remote inference goes through `FIREWORKS_BASE_URL` using only
  `ALLOWED_MODELS`; a bundled local model (any model, ≤10 GB image) may answer for
  zero token cost; `.env` is never bundled.

**Architecture decision:** a **two-tier** design — deterministic solvers (tier 0,
zero tokens, provably-correct-or-abstain) → Fireworks remote (tier 1). **No bundled
local LLM.** On the 4 GB / 2 vCPU box a small local model is too weak/slow to trust
against the gate and adds image/RAM/latency risk; deterministic solvers cover the
provable cases for free, and everything else goes to the strong Fireworks models.

## Glossary

- **Accuracy gate:** The minimum answer-quality threshold (approx. 80%) a submission
  must clear to be eligible for token ranking.
- **Cascade / tier:** The ordered chain a task flows through — deterministic solver
  (tier 0), Fireworks remote model (tier 1). (No bundled local LLM tier.)
- **Deterministic solver:** Rule/computation-based answerer (e.g. arithmetic) that
  returns a provably correct result or abstains.
- **Validation gate:** A form-level check applied to a local answer before it is
  accepted; failure causes escalation to the next tier.
- **Over-routing:** A cheaper tier answering a task it should have escalated (risking
  accuracy). Target count: zero.
- **Escalate:** Pass a task to the next, more capable tier.
- **Local answer / local tokens:** Output produced inside the container; counts toward
  accuracy but not toward the token score.
- **Remote / Fireworks tier:** Inference via `FIREWORKS_BASE_URL` using a model from
  `ALLOWED_MODELS`; its tokens count toward the leaderboard score.
- **ALLOWED_MODELS:** The comma-separated list of permitted Fireworks model IDs
  injected by the harness at runtime.
- **Thinking model:** A model that emits hidden reasoning tokens, which can truncate
  answers and inflate token cost if not bounded.

## Requirements

### Requirement 1: Reliably clear the accuracy gate

**User Story:** As a hackathon participant, I want the agent to reliably clear the ~80%
accuracy gate, so that my submission is eligible for the token-efficiency leaderboard.

#### Acceptance Criteria

1. WHEN the agent produces a final answer for a task, THE system SHALL prioritise
   correctness over token cost.
2. WHEN the agent is evaluated on the public validation tasks (factual, math,
   sentiment, summarization, NER), THE system SHALL satisfy each task's stated pass
   criteria for a target overall pass rate of at least 90% (margin above the ~80% gate).
3. WHERE a task's correct answer is uncertain at a cheaper tier, THE system SHALL
   escalate to a more capable tier rather than emit a low-confidence answer.

### Requirement 2: Answer validation before shipping

**User Story:** As a participant, I want every answer validated before it is written,
so that an empty, malformed, or truncated answer never causes an accuracy-gate or
schema failure.

#### Acceptance Criteria

1. WHEN a deterministic solver produces an answer, IF it does not pass a sanity/form
   check for its category, THEN the system SHALL abstain and escalate to Fireworks.
2. WHEN a Fireworks response is empty or whitespace-only, THE system SHALL retry with
   the escalation model rather than ship an empty answer.
3. WHEN a Fireworks response is truncated (finished at the token cap mid-answer), THE
   system SHALL treat it as a failure and retry the escalation model.
4. WHERE a category has a required output form (e.g. NER JSON, sentiment label), THE
   final written answer SHALL conform to that form.

### Requirement 3: Conservative deterministic routing (over-routing = 0)

**User Story:** As a participant, I want the deterministic solvers to answer only when
provably correct, so that free answers never reduce accuracy.

#### Acceptance Criteria

1. WHERE a deterministic solver cannot compute a provably correct result, THE system
   SHALL abstain and pass the task to the next tier.
2. WHEN measured on the deterministic evaluation set, THE system SHALL report an
   over-routing count (tasks answered deterministically that should have escalated)
   of zero.
3. WHEN a math task involves multiple steps or ambiguous parsing, THE deterministic
   math solver SHALL abstain rather than attempt a single-step guess.

### Requirement 4: Category-specialised remote model selection

**User Story:** As a participant, I want each escalated task sent to the most suitable
allowed model, so that remote calls succeed in one shot and spend the fewest tokens.

#### Acceptance Criteria

1. WHEN a code task is escalated, THE system SHALL prefer a code-specialised model from
   `ALLOWED_MODELS` (e.g. a Kimi/code model) when present.
2. WHEN a math or logic task is escalated, THE system SHALL prefer a reasoning-capable
   allowed model (e.g. Minimax) when present.
3. WHEN a language task (factual, sentiment, NER, summarization) is escalated, THE
   system SHALL prefer a Gemma-4 allowed model when present.
4. THE system SHALL only ever select a model that is present in `ALLOWED_MODELS`, and
   SHALL fall back to the first allowed model when no preferred match exists.
5. WHERE an allowed model emits hidden reasoning tokens ("thinking"), THE system SHALL
   suppress or bound that behaviour to avoid truncated answers and inflated tokens.

### Requirement 5: Category-specific answer quality

**User Story:** As a participant, I want answers to match each category's required form
and content, so that they pass the judge's category-specific criteria.

#### Acceptance Criteria

1. WHEN a sentiment task describes a mixed experience, THE system SHALL produce a label
   that is not "Negative-only" and a justification that acknowledges BOTH the positive
   and negative aspects.
2. WHEN a summarization task states an exact format constraint (e.g. exactly two
   sentences, exactly three bullets each within a word limit), THE system SHALL produce
   output that meets that constraint exactly.
3. WHEN an NER task requests labelled entities, THE system SHALL return every expected
   entity with a correct entity-type label.
4. WHEN a factual task is answered, THE system SHALL give a specific, on-topic answer
   and SHALL NOT return a generic or evasive response.

### Requirement 6: Lean, reliable self-contained image

**User Story:** As a participant, I want a small container that builds reliably and runs
entirely offline on the grading box, so that the submission is not rejected for build,
pull, timeout, or missing-output errors.

#### Acceptance Criteria

1. WHEN the image is built, THE build SHALL complete deterministically with no
   background server process and no model download during build.
2. THE container SHALL NOT download any model or large file at run time.
3. THE compressed image size SHALL be well under 10 GB (target < 1 GB, since no model
   weights are bundled).
4. THE runtime memory footprint SHALL fit comfortably within 4 GB RAM / 2 vCPU.
5. WHEN the container starts with no arguments, THE entrypoint SHALL run automatically,
   read config from the environment, and require no local files beyond `/input`.

### Requirement 7: Time budget and concurrency

**User Story:** As a participant, I want the run to always finish within the time limit,
so that the submission is never rejected for a timeout.

#### Acceptance Criteria

1. WHILE processing tasks, THE system SHALL respect a configurable global time budget
   below the 10-minute hard cap.
2. WHEN the time budget is exceeded mid-run, THE system SHALL stop making new remote
   calls and write whatever results it has (with fallbacks for unfinished tasks).
3. THE system SHALL issue remote calls concurrently (bounded worker pool) to use the
   budget efficiently.

### Requirement 8: Token-efficiency optimisations (after correctness)

**User Story:** As a participant, I want lean remote calls and reused work, so that my
passing submission ranks well on total tokens.

#### Acceptance Criteria

1. WHEN a prompt is sent to a remote model, THE system SHALL first compress it by
   removing filler and redundant whitespace WHILE preserving code blocks and meaning.
2. WHEN two tasks in a run have equivalent (normalised) prompts, THE system SHALL reuse
   the first result for the second and record zero additional remote tokens.
3. THE system SHALL bound remote output with per-category `max_tokens` caps sized to
   avoid truncation while minimising output length.
4. THE system SHALL use terse, category-specific system prompts that request only the
   minimum the judge needs.

### Requirement 9: Always-valid, complete output (harness contract)

**User Story:** As a participant, I want a complete, valid results file for every run,
so that the submission is never scored zero for schema or missing-task errors.

#### Acceptance Criteria

1. THE system SHALL read tasks from `/input/tasks.json` and write answers to
   `/output/results.json` as valid JSON.
2. THE system SHALL return exactly one result per input task, preserving `task_id`
   values exactly.
3. IF an individual task fails, THEN the system SHALL record a fallback answer for it
   and continue, still writing a complete results file and exiting with code 0.
4. THE system SHALL write `/output/results.json` before exiting.

### Requirement 10: Public validation eval harness

**User Story:** As a participant, I want to run the published validation tasks end-to-end
locally, so that I can confirm the accuracy gate and token spend before submitting.

#### Acceptance Criteria

1. THE eval harness SHALL run the public validation tasks through the full cascade and
   report per-task pass/fail against the stated criteria.
2. THE eval harness SHALL report total remote tokens and the per-tier route breakdown
   (deterministic / remote).
3. THE eval harness SHALL be runnable offline for the deterministic tier without any
   remote calls.
