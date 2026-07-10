# Lean agent image. Runs the local-first cascade: local deterministic solvers +
# (optional) a local LLM reached via LOCAL_LLM_BASE_URL, with Fireworks as the
# paid fallback. Config is injected purely via environment at runtime.
#
# Local LLM (gemma) deployment options:
#   A) Co-located Ollama service: set LOCAL_LLM_BASE_URL to its OpenAI-compatible
#      endpoint. Keeps THIS image tiny.
#   B) Self-contained: switch the base image to a ROCm runtime and bundle
#      Ollama + a quantized gemma (keep total compressed size < 10 GB).
# Option A keeps the image lean and is the default below.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Application code.
COPY src/ ./src/
# Launch-day tuning outputs (model preference / confidence thresholds), if any.
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "src.main"]
