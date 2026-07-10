# Self-contained Track 1 image: bundles a small local model (llama3.2:3b) served
# by Ollama on CPU, plus the deterministic solvers and the Fireworks fallback.
#
# Grading box: CPU-only, 4 GB RAM, 2 vCPU, image <= 10 GB.
#   - Local tier: llama3.2:3b (~2 GB) via Ollama at localhost:11434 (free tokens).
#   - Fireworks tier: harness-injected ALLOWED_MODELS handle the hard tail.
# FIREWORKS_* config is read from the environment at runtime — never bundled.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# Install the Ollama CPU runtime.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python app dependencies + the spaCy NER model.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Bundle the local model weights into the image (start server, wait, pull).
# On failure, dump the ollama log + disk usage so the build output shows why.
RUN which ollama && ollama --version && df -h /; \
    ollama serve > /tmp/ollama.log 2>&1 & \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
    if ! ollama pull llama3.2:3b; then \
        echo '=== ollama serve log ==='; cat /tmp/ollama.log; df -h /; exit 1; \
    fi; \
    ollama list

COPY src/ ./src/
COPY config/ ./config/
COPY entrypoint.sh /entrypoint.sh
# Strip any CR (Windows CRLF) so the shebang works, then make executable.
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# In-container local model endpoint (not secrets; safe to set here).
ENV LOCAL_LLM_BASE_URL=http://localhost:11434/v1 \
    LOCAL_LLM_MODEL=llama3.2:3b \
    LOCAL_LLM_API_KEY=local \
    LOCAL_LLM_SAMPLES=1

ENTRYPOINT ["/entrypoint.sh"]
