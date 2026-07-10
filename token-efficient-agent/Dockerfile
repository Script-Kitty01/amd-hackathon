# Self-contained Track 1 image: bundles a small local model (llama3.2:3b) served
# by Ollama on CPU, plus the deterministic solvers and the Fireworks fallback.
#
# Design for the grading box (CPU-only, 4 GB RAM, 2 vCPU, image <= 10 GB):
#   - Local tier: llama3.2:3b (~2 GB, 4-bit) via Ollama at localhost:11434.
#     Its inference is free (0 tokens) and answers the easy categories.
#   - Fireworks tier: the harness-injected ALLOWED_MODELS handle the hard tail.
# Config (FIREWORKS_*) is read from the environment at runtime — never bundled.

FROM ollama/ollama:latest

# Python + our app dependencies on top of the Ollama (Ubuntu) base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt \
    && python3 -m spacy download en_core_web_sm

# Bundle the local model weights into the image (start server, pull, stop).
RUN ollama serve & \
    sleep 8 && \
    ollama pull llama3.2:3b && \
    sleep 2

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

# The ollama base image sets ENTRYPOINT to the ollama binary — override it.
ENTRYPOINT ["/entrypoint.sh"]
