"""Authentication middleware — JWT tokens and API keys.

Supports two auth methods:
  1. Bearer JWT tokens (for dashboard / admin users)
  2. X-API-Key header (for service-to-service / programmatic access)

In production, replace the in-memory user store with a database.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import DATA_DIR


@dataclass
class User:
    username: str
    api_key_hash: str
    department: str
    role: str  # "admin" | "developer" | "viewer"
    priority: int = 0  # 0=normal, 1=high, 2=critical
    monthly_budget_usd: float = 1000.0
    allowed_models: list[str] | None = None  # None = all allowed
    blocked_models: list[str] | None = None


class AuthManager:
    """Handles JWT creation/verification and API key validation."""

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        expire_minutes: int = 60,
        users_path: Path | None = None,
    ) -> None:
        self._secret = secret
        self._algorithm = algorithm
        self._expire_minutes = expire_minutes
        self._users: dict[str, User] = {}
        self._api_key_index: dict[str, User] = {}

        # Load users from file or use defaults
        users_path = users_path or DATA_DIR / "users.json"
        self._load_users(users_path)

    def _load_users(self, path: Path) -> None:
        """Load users from JSON file, or create defaults."""
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                for entry in raw:
                    user = User(**entry)
                    self._users[user.username] = user
                    self._api_key_index[user.api_key_hash] = user
                return
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        # Default users for development
        defaults = [
            User(
                username="admin",
                api_key_hash=_hash_key("admin-key-change-me"),
                department="engineering",
                role="admin",
                priority=2,
                monthly_budget_usd=5000.0,
            ),
            User(
                username="developer",
                api_key_hash=_hash_key("dev-key-change-me"),
                department="engineering",
                role="developer",
                priority=0,
                monthly_budget_usd=1000.0,
            ),
            User(
                username="viewer",
                api_key_hash=_hash_key("viewer-key-change-me"),
                department="marketing",
                role="viewer",
                priority=0,
                monthly_budget_usd=200.0,
            ),
        ]
        for user in defaults:
            self._users[user.username] = user
            self._api_key_index[user.api_key_hash] = user

    def create_token(self, username: str) -> str:
        """Create a JWT for a user."""
        from jose import jwt
        user = self._users.get(username)
        if user is None:
            raise ValueError(f"Unknown user: {username}")

        payload = {
            "sub": username,
            "department": user.department,
            "role": user.role,
            "priority": user.priority,
            "exp": int(time.time()) + self._expire_minutes * 60,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def verify_token(self, token: str) -> dict:
        """Verify a JWT and return its payload."""
        from jose import JWTError, jwt
        from fastapi import HTTPException, status
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
            username = payload.get("sub")
            if username not in self._users:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
            return payload
        except JWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    def verify_api_key(self, api_key: str) -> User:
        """Verify an API key and return the user."""
        from fastapi import HTTPException, status
        key_hash = _hash_key(api_key)
        user = self._api_key_index.get(key_hash)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return user

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# --- FastAPI dependency ---

_AUTH_HEADER = "X-API-Key"


async def get_current_user(request, auth: AuthManager) -> User:
    """FastAPI dependency: extract user from Bearer token or X-API-Key header."""
    from fastapi import HTTPException, status
    # Try API key first
    api_key = request.headers.get(_AUTH_HEADER)
    if api_key:
        return auth.verify_api_key(api_key)

    # Try Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = auth.verify_token(token)
        username = payload["sub"]
        user = auth.get_user(username)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
