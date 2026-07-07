FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ship only the application code (eval/, tests/, data/ excluded via .dockerignore).
COPY src/ ./src/

# Config is injected purely via environment at runtime.
ENTRYPOINT ["python", "-m", "src.main"]
