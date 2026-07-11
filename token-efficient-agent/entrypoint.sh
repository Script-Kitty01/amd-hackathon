#!/usr/bin/env sh
# Start the bundled local model server, wait for it, warm the model, then run
# the agent. When the agent exits, the container exits with its status code.
set -e

ollama serve &

# Wait for the local server to accept requests.
i=0
until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "warning: local model server did not come up; continuing (will use Fireworks)" >&2
        break
    fi
    sleep 1
done

# Warm the model so the first task doesn't pay load latency (best-effort).
ollama run "${LOCAL_LLM_MODEL:-llama3.2:3b}" "ok" >/dev/null 2>&1 || true

# Run the agent (reads FIREWORKS_* + LOCAL_LLM_* from the environment).
exec python3 -m src.main
