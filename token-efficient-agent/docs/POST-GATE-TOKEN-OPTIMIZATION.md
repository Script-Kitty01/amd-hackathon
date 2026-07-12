# Post-Gate Token Optimization (do NOT apply until we clear the 80% accuracy gate)

> Scope: these are **token-efficiency** tactics for _after_ we're on the leaderboard.
> Our current blocker is the **accuracy gate**, not tokens. Reasoning models are
> generally _more_ accurate on hard math/logic, so moving those to a cheaper
> non-reasoning model saves tokens but can cost accuracy — the wrong trade while
> we're gate-blocked. Apply the items here only once accuracy is comfortably
> above the gate, and only if a change can be validated (proxy judge or a real
> submission cycle) without dropping below it.
>
> Source: distilled from an external deep-research pass (competitor/companion
> docs). Numbers there were 2026-dated, vendor/benchmark-sourced, and heavily
> caveated — treat all specific figures as **directional**, confirm before trust.

## Current architecture (baseline these changes against)

- Two-tier: deterministic math solver (0 tokens) + Fireworks for everything else.
- No bundled local LLM (dropped due to 4 GB RAM / can't-validate risk).
- Routing today: language -> `gemma-4-31b-it`; math/logic -> `minimax-m3`;
  code -> `kimi-k2p7-code`. Escalation fallback -> `gemma-4-31b-it-nvfp4`.
- Already done: strip `<think>` reasoning traces; terse prompts; per-category
  `max_tokens`; keep full multi-part answers; high concurrency; never-empty backstop.

## The core lever: the reasoning-token tax

- `minimax-m3` and `kimi-k2p7-code` are **reasoning models** — they emit counted
  chain-of-thought by default. A CoT trace can be several times the answer's
  tokens. Every reasoning-model call is a token tax taken on purpose.
- Gemma variants are **terse / non-reasoning by default** and (reportedly) use a
  larger-vocab tokenizer, so identical text costs fewer counted tokens.

## Ordered post-gate experiments (each gated on "accuracy still holds")

1. **Suppress reasoning on `minimax-m3` where the gate allows.** Its thinking is
   reportedly togglable off at request time. Our client already supports
   `FIREWORKS_REASONING_EFFORT` via `extra_body` and drops it gracefully on a 400.
   Try `reasoning_effort=none`/`low` for math/logic and re-measure accuracy.
2. **Gemma-first on escalation for code.** A/B `gemma-4-31b-it` (non-reasoning,
   denser tokenizer) against `kimi-k2p7-code` for code_debug / code_gen. If Gemma
   clears the gate, it likely wins big on tokens. Kimi also over-engineers by
   default — add a "simplest correct solution, function only" nudge if kept.
3. **Gemma-first for math/logic**, escalating to `minimax-m3` (thinking-on) only
   on validation failure. Careful: do NOT over-condense CoT on math — concise-CoT
   on weaker models has shown large accuracy drops on math specifically.
4. **Tokenizer sanity check** (only possible with real endpoint access): send one
   fixed answer string through all 5 models via the proxy, record counted tokens,
   confirm the Gemma-cheapest ordering on the actual scorer.
5. **Tighter output policy** where the judge tolerates it: shorter `max_tokens`,
   `stop` sequences (`\n\n`, closing code fence), compact structure with SHORT
   keys for NER (long JSON keys cost more than they save).

## Do NOT do (net-negative or wrong phase)

- Do **not** import the `caveman` / `ponytail` skills — they add large system-prompt
  overhead per call and assume multi-turn prompt caching we don't have; benchmarks
  show them net-negative on single-shot metered scoring. The _idea_ (one-line terse
  instruction) is already in our prompts.
- Do **not** strip hedging/qualifiers aggressively — it can flip judged meaning on
  sentiment ("slightly negative" -> "negative") and nuanced factual answers.
- Do **not** pivot to a bundled local model blind. Legitimate as a zero-token path,
  but it needs on-box calibration we can't currently do, and an unvalidated 2-3B
  model that emits confident-but-wrong answers reproduces the original gate failure.

## Claims to verify before trusting (unconfirmed)

- Does the judging proxy actually count reasoning/CoT tokens? (Assumed yes — the
  whole Gemma-first thesis depends on it.)
- Do the served `gemma-4-*-it` endpoints emit a `<think>` trace by default? If so,
  force non-thinking or they lose the terse advantage.
- Exact param name to toggle `minimax-m3` thinking via the OpenAI-compatible API.
- Whether any listed model can 404 as "not deployed" in the grading env (likely a
  non-issue since the harness controls deployment of `ALLOWED_MODELS`).
