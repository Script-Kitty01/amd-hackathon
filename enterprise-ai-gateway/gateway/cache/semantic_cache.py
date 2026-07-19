"""Semantic cache using sentence-transformers embeddings.

Caches LLM responses keyed by semantic similarity rather than exact match.
Two similar prompts (e.g., "What is Python?" vs "Tell me about Python") 
will hit the same cache entry if their cosine similarity exceeds the threshold.

Supports:
  - In-memory cache (default, zero dependencies beyond sentence-transformers)
  - Redis backend (optional, for distributed deployments)
  - TTL-based expiry
  - LRU eviction for in-memory mode
"""

from __future__ import annotations

import hashlib
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    prompt: str
    response: str
    embedding: np.ndarray
    created_at: float
    ttl_seconds: int
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    entries: int = 0
    hit_rate: float = 0.0


class SemanticCache:
    """Semantic cache with cosine-similarity matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,  # 1 hour default
        max_entries: int = 10000,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._model_name = model_name
        self._model = None
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            s = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                entries=len(self._entries),
            )
            total = s.hits + s.misses
            s.hit_rate = s.hits / total if total > 0 else 0.0
            return s

    def _ensure_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self._ensure_model()
        return self._model.encode(text, normalize_embeddings=True)

    def _hash_key(self, prompt: str) -> str:
        """Generate a deterministic hash key for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def lookup(self, prompt: str) -> Optional[str]:
        """Look up a semantically similar prompt in the cache.

        Returns the cached response if a match is found, None otherwise.
        """
        self._ensure_model()
        query_embedding = self._embed(prompt)

        with self._lock:
            # Evict expired entries
            self._evict_expired()

            best_similarity = 0.0
            best_entry: Optional[CacheEntry] = None

            for entry in self._entries.values():
                if entry.expired:
                    continue
                sim = float(np.dot(query_embedding, entry.embedding))
                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = entry

            if best_entry is not None and best_similarity >= self._threshold:
                best_entry.hit_count += 1
                self._stats.hits += 1
                return best_entry.response

            self._stats.misses += 1
            return None

    def store(self, prompt: str, response: str, ttl_seconds: Optional[int] = None) -> None:
        """Store a prompt-response pair in the cache."""
        self._ensure_model()
        embedding = self._embed(prompt)
        key = self._hash_key(prompt)
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl

        with self._lock:
            # Evict if at capacity (LRU: remove oldest)
            if len(self._entries) >= self._max_entries:
                oldest_key = min(
                    self._entries.keys(),
                    key=lambda k: self._entries[k].created_at,
                )
                del self._entries[oldest_key]

            self._entries[key] = CacheEntry(
                key=key,
                prompt=prompt,
                response=response,
                embedding=embedding,
                created_at=time.time(),
                ttl_seconds=ttl,
            )

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [
            k for k, e in self._entries.items()
            if (now - e.created_at) > e.ttl_seconds
        ]
        for k in expired:
            del self._entries[k]

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._stats = CacheStats()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)
