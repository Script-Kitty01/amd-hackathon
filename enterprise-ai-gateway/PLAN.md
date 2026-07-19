# Enterprise AI Gateway — Build Plan

> **Status:** Core components built & tested ✅ | Server + remaining modules pending

---

## ✅ COMPLETED — What's Been Built & Tested

### 1. Project Scaffolding

| File                  | Description                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `gateway/__init__.py` | Package init                                                                                                 |
| `gateway/config.py`   | `GatewayConfig`, `ProviderConfig` dataclasses, `.env` loader, `load_config()`                                |
| `requirements.txt`    | 50-line dependency manifest (FastAPI, OpenAI, Anthropic, Gemini, sentence-transformers, Presidio, JWT, etc.) |

### 2. Provider Layer (`gateway/providers/`)

| File                    | Description                                                                         | Status                |
| ----------------------- | ----------------------------------------------------------------------------------- | --------------------- |
| `base.py`               | Abstract `ProviderClient` ABC, `ProviderResponse` dataclass                         | ✅ Built              |
| `openai_provider.py`    | `OpenAIProvider` via `AsyncOpenAI` (covers OpenAI/Fireworks/Groq/Together)          | ✅ Built, lazy import |
| `anthropic_provider.py` | `AnthropicProvider` via `AsyncAnthropic`                                            | ✅ Built, lazy import |
| `gemini_provider.py`    | `GeminiProvider` via `google-generativeai` (sync → thread pool)                     | ✅ Built, lazy import |
| `registry.py`           | `ProviderRegistry` with lazy instantiation, model→provider mapping, health tracking | ✅ Built & tested     |

### 3. Routing Layer (`gateway/routing/`)

| File        | Description                                                                                                      | Status            |
| ----------- | ---------------------------------------------------------------------------------------------------------------- | ----------------- |
| `engine.py` | `RoutingEngine` with `decide()` + `execute()` fallback chain, `classify_prompt()` regex classifier               | ✅ Built & tested |
| `scorer.py` | `ModelScorer` with 7-dimension weighted scoring (cost, quality, latency, availability, policy, budget, priority) | ✅ Built & tested |

### 4. Middleware Layer (`gateway/middleware/`)

| File              | Description                                                                                 | Status            |
| ----------------- | ------------------------------------------------------------------------------------------- | ----------------- |
| `auth.py`         | `AuthManager` — JWT + API key auth, `User` dataclass, default dev users, FastAPI dependency | ✅ Built & tested |
| `logging.py`      | `GatewayLogger` (JSON-lines to daily files), `RequestTracker` with full metadata            | ✅ Built & tested |
| `rate_limiter.py` | `RateLimiter` — sliding-window in-memory, burst + per-minute limits                         | ✅ Built & tested |

### 5. Security Layer (`gateway/security/`)

| File                    | Description                                                                   | Status            |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------- |
| `pii_scanner.py`        | `PIIScanner` — regex fallback + Presidio integration, entity-specific masking | ✅ Built & tested |
| `injection_detector.py` | `InjectionDetector` — critical/high/medium regex patterns, risk scoring 0..1  | ✅ Built & tested |

### 6. Cache Layer (`gateway/cache/`)

| File                | Description                                                                              | Status                      |
| ------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| `semantic_cache.py` | `SemanticCache` — sentence-transformers embeddings, cosine similarity, LRU eviction, TTL | ✅ Built, ⚠️ not yet tested |

### 7. Lazy Import Fixes (Critical for testability)

All provider SDKs (`openai`, `anthropic`, `google-generativeai`) and auth deps (`jose`, `fastapi`) are now **lazy-imported** inside method bodies. This means:

- The entire codebase can be imported and tested **without installing any provider SDKs**
- Only the specific SDK is loaded when a provider is actually used
- Files fixed: `registry.py`, `openai_provider.py`, `anthropic_provider.py`, `gemini_provider.py`, `scorer.py`, `auth.py`, `rate_limiter.py`

### 8. Dependencies Installed

- `sentence-transformers` (all-MiniLM-L6-v2 for semantic cache)
- `python-jose[cryptography]` (JWT auth)
- `passlib[bcrypt]` (password hashing)

---

## ❌ REMAINING — What Still Needs To Be Built

---

### Phase 1: Wire Up The Server (Critical Path)

#### 1.1 `gateway/server.py` — FastAPI Application

The central orchestrator. Request flow:

```
Request → Auth → Rate Limit → PII Scan → Injection Check → Cache? → Route → Execute → Log → Response
```

**Lifespan startup:** Load `GatewayConfig` from env → create all components (registry, scorer, engine, auth, rate limiter, logger, PII scanner, injection detector, cache) → store in `app.state`.

**`POST /v1/chat/completions`** (OpenAI-compatible schema):
1. `get_current_user` dependency extracts `User` from Bearer JWT or `X-API-Key` header
2. `RateLimiter.check(user_id)` — raises 429 if exceeded
3. `PIIScanner.scan_and_mask(prompt)` — masks PII, sets `tracker.pii_detected`
4. `InjectionDetector.scan(prompt)` — if `blocked=True`, return 400 immediately
5. `SemanticCache.lookup(prompt)` — if hit, return cached response instantly
6. `RoutingEngine.decide(prompt, department, priority, budget, blocked_models)` → `RouteDecision`
7. `RoutingEngine.execute(decision, system, user, max_tokens, temperature)` with fallback chain
8. `RequestTracker` records tokens, cost, latency, route scores
9. `GatewayLogger.log(tracker)` writes JSON-lines log
10. Return OpenAI-formatted response: `{"id":..., "object":"chat.completion", "choices":[...], "usage":{...}}`

**`GET /v1/models`** — returns `registry.all_models` with cost info from `registry.model_info()`

**`GET /health`** — returns `{"status": "ok", "providers": {...}}` with health status per provider

#### 1.2 `run.py` — Entry Point

```python
import uvicorn
from gateway.server import app
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

#### 1.3 `.env.example` — Environment Template

All env vars documented: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `JWT_SECRET`, `CACHE_TTL`, `RATE_LIMIT_PER_MINUTE`, etc.

---

### Phase 2: Budget Manager

#### 2.1 `gateway/budget/manager.py` — BudgetTracker

```
BudgetManager
  ├── check_budget(user_id, estimated_cost) → bool   # can we afford this call?
  ├── record_spend(user_id, model, tokens, cost)      # log the spend
  ├── get_remaining(user_id) → float                  # remaining this month
  └── get_usage_report(department) → dict             # aggregated stats
```

#### 2.2 `gateway/budget/store.py` — SQLite Ledger

SQLAlchemy + aiosqlite. Table: `spend_log` (user_id, department, model, prompt_tokens, completion_tokens, cost_usd, timestamp). Monthly budgets come from `User.monthly_budget_usd` in auth.

---

### Phase 3: Response Validation

#### 3.1 `gateway/validation/response_validator.py`

```
ResponseValidator
  └── validate(response_text, expected_schema=None) → ValidationResult
        ├── empty_check: is response blank?
        ├── truncation_check: does it look cut off?
        ├── hallucination_markers: "as an AI", refusal patterns
        ├── schema_check: Pydantic validate against expected JSON schema
        └── safety_check: profanity, harmful content regex
```

---

### Phase 4: Analytics Dashboard

#### 4.1 `gateway/dashboard/routes.py` — FastAPI APIRouter

Routes: `/dashboard` (HTML page), `/dashboard/api/stats` (JSON: total cost, calls, avg latency), `/dashboard/api/usage` (JSON: per-model breakdown, per-user activity). Reads from JSON-lines log files + budget SQLite DB.

#### 4.2 `gateway/dashboard/templates/` — Jinja2 HTML

Overview page with cost charts, model usage pie, user activity table. Uses Chart.js (CDN) for visualizations.

#### 4.3 `gateway/dashboard/static/` — CSS/JS

Minimal custom CSS for the dashboard layout.

---

### Phase 5: Deployment & DevOps

#### 5.1 `Dockerfile`

Multi-stage: `python:3.12-slim` → install deps → copy code → non-root user → `CMD ["uvicorn", "gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]`

#### 5.2 `docker-compose.yml`

Services: gateway (port 8000) + redis (for distributed cache) + optional postgres.

#### 5.3 `tests/` — Test Suite

pytest + pytest-asyncio + httpx. Unit tests for all modules, integration tests for the server endpoints.

---

### Phase 6: Polish

| #   | Task                            | Description                                                                |
| --- | ------------------------------- | -------------------------------------------------------------------------- |
| 6.1 | Test semantic cache             | Verify sentence-transformers embeddings + cosine similarity matching works |
| 6.2 | Add `__init__.py` to empty dirs | `budget/`, `dashboard/`, `validation/` need proper package inits           |
| 6.3 | README                          | Usage docs, architecture diagram, quickstart                               |

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │     FastAPI Server           │
                    │  (gateway/server.py)         │
                    └──────────┬──────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │  Middleware  │   │   Security  │   │    Cache    │
     │  • Auth      │   │  • PII      │   │  • Semantic │
     │  • Rate Limit│   │  • Injection│   │  • TTL+LRU  │
     │  • Logging   │   │             │   │             │
     └──────┬───────┘   └──────┬──────┘   └──────┬──────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Routing Engine     │
                    │  • Classify prompt   │
                    │  • Score models      │
                    │  • Fallback chain    │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │   OpenAI    │   │  Anthropic  │   │   Gemini    │
     │  Provider   │   │  Provider   │   │  Provider   │
     └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Quick Test Command

```powershell
cd enterprise-ai-gateway
python -c "
from gateway.providers.registry import ProviderRegistry
from gateway.routing.scorer import ModelScorer, RoutingContext, TaskCategory
from gateway.routing.engine import RoutingEngine, classify_prompt
from gateway.middleware.auth import AuthManager
from gateway.middleware.rate_limiter import RateLimiter, RateLimitConfig
from gateway.middleware.logging import GatewayLogger, RequestTracker
from gateway.security.pii_scanner import PIIScanner
from gateway.security.injection_detector import InjectionDetector
from gateway.cache.semantic_cache import SemanticCache
from gateway.config import ProviderConfig
print('ALL MODULES IMPORT OK — no SDKs required!')
"
```
