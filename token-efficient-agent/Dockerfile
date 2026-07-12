# Track 1 — Token-Efficient Agent
# Lean image: Gemma-first Fireworks routing + deterministic local solvers (0 tokens).
# Reads FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS from the environment.
# Reads /input/tasks.json, writes /output/results.json, exits 0.

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install only the OpenAI-compatible client (the sole runtime dependency).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY config/ ./config/

# Create output directory (the harness mounts /input and /output, but we
# create /output here as a safety net in case it doesn't exist).
RUN mkdir -p /output

# The entrypoint reads tasks from /input/tasks.json, solves them via the
# cascade (local solvers -> Fireworks fallback), writes /output/results.json,
# and always exits 0 with valid JSON output.
ENTRYPOINT ["python", "-m", "src.main"]
