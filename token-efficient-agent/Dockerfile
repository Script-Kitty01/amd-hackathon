# Lean Track 1 image: deterministic local solvers (free, 0 tokens) + Fireworks
# for everything that needs real understanding. No bundled LLM — a 2-3B model
# that fits the 4 GB CPU grading box is too weak to clear the accuracy gate and
# only produces confident-but-wrong answers, so all language tasks go to
# Fireworks via FIREWORKS_BASE_URL. FIREWORKS_* config is read from the
# environment at runtime — never bundled.

FROM python:3.11-slim

WORKDIR /app

# Python app dependencies (just the OpenAI-compatible client).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# NOTE: we deliberately do NOT force reasoning_effort=none. The reasoning-heavy
# categories (multi-step math, constraint logic, code) need the thinking models
# (minimax/kimi) to actually reason to be correct; the prompt max_tokens are
# sized generously so reasoning + the final answer both fit without truncation.
# Non-thinking models (gemma) simply ignore reasoning and answer directly.

ENTRYPOINT ["python", "-m", "src.main"]
