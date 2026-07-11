# Lean Track 1 image: deterministic local solvers (free, 0 tokens) + Fireworks
# for everything that needs real understanding. No bundled LLM — a 2-3B model
# that fits the 4 GB CPU grading box is too weak to clear the accuracy gate and
# only produces confident-but-wrong answers, so all language tasks go to
# Fireworks via FIREWORKS_BASE_URL. FIREWORKS_* config is read from the
# environment at runtime — never bundled.

FROM python:3.11-slim

WORKDIR /app

# Python app dependencies + the spaCy NER model (used only by local solvers,
# which are disabled by default for the accuracy-first baseline).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY config/ ./config/

# Thinking models (e.g. minimax) burn the max_tokens budget on hidden reasoning
# and truncate the final answer. Requesting no extra reasoning keeps answers
# intact and cuts tokens. The client automatically drops this param for any
# model that rejects it (costs 0 tokens), so it's safe across all ALLOWED_MODELS.
ENV FIREWORKS_REASONING_EFFORT=none

ENTRYPOINT ["python", "-m", "src.main"]
