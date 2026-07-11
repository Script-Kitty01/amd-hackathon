# Self-contained Track 1 image: bundles a small local model (llama3.2:3b) served
# by Ollama on CPU, plus the deterministic solvers and the Fireworks fallback.

FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        curl \
        ca-certificates \
        procps \
        zstd && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "click>=8.1,<9"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Bundle the local model into the image
RUN which ollama && \
    ollama --version && \
    df -h / && \
    ollama serve >/tmp/ollama.log 2>&1 & \
    for i in $(seq 1 60); do \
        ollama list >/dev/null 2>&1 && break; \
        sleep 1; \
    done && \
    if ! ollama pull llama3.2:3b; then \
        echo "===== Ollama Log ====="; \
        cat /tmp/ollama.log; \
        echo "===== Disk Usage ====="; \
        df -h; \
        exit 1; \
    fi && \
    ollama list

COPY src/ ./src/
COPY config/ ./config/
COPY entrypoint.sh /entrypoint.sh

RUN sed -i 's/\r$//' /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENV LOCAL_LLM_BASE_URL=http://localhost:11434/v1
ENV LOCAL_LLM_MODEL=llama3.2:3b
ENV LOCAL_LLM_API_KEY=local
ENV LOCAL_LLM_SAMPLES=1

ENTRYPOINT ["/entrypoint.sh"]