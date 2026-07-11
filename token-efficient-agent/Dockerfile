# Lean Track 1 image: deterministic local solvers (free, 0 tokens) + Fireworks
# for everything that needs real understanding. No bundled LLM — a 2-3B model
# that fits the 4 GB CPU grading box is too weak to clear the accuracy gate and
# only produces confident-but-wrong answers, so all language tasks go to
# Fireworks via FIREWORKS_BASE_URL. FIREWORKS_* config is read from the
# environment at runtime — never bundled.

# Pin the platform: the grading box is linux/amd64. Building with an explicit
# platform guarantees the image runs there (no "exec format error" crash).
FROM --platform=linux/amd64 python:3.11-slim

# Fail fast and flush logs so any startup error is visible in the harness logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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

# Runs with no arguments; reads FIREWORKS_* from the environment (injected by
# the harness). Always writes /output/results.json and exits 0.
ENTRYPOINT ["python", "-m", "src.main"]
