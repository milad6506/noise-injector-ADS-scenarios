"""Pluggable LLM backend for LLM-powered corruption ops.

Bring-your-own API: any OpenAI-compatible endpoint (OpenAI, OpenRouter, Together,
vLLM, ollama, ...). The API key is read from an environment variable — never passed
on the command line, never written to manifests or code.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


class LLMBackend:
    def __init__(self, base_url: str, model: str, api_key_env: str = "NOISE_INJECTOR_API_KEY", timeout_s: float = 60.0):
        if not base_url:
            raise ValueError("LLM backend requires base_url")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self._key = None

    @property
    def key(self) -> str | None:
        if self._key is None:
            self._key = os.environ.get(self.api_key_env)
        return self._key

    @property
    def cache_key_base(self) -> str:
        return f"{self.base_url}|{self.model}"

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class FakeLLMBackend(LLMBackend):
    """Offline deterministic stub for tests/demos: returns canned rewrites without any network."""

    def __init__(self, canned: str = "wrong-answer-stub", **kwargs):
        super().__init__(base_url="http://localhost:0", model="fake-llm", **kwargs)
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
        self.calls.append((system, user))
        return self.canned


class LLMCache:
    """Disk cache keyed by (backend, model, prompt-hash) — makes LLM corruptions reproducible."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            if self.path.exists():
                with open(self.path) as f:
                    self._data = json.load(f)
            else:
                self._data = {}
        return self._data

    def get(self, backend_key: str, prompt_key: str):
        return self._load().get(f"{backend_key}|{prompt_key}")

    def put(self, backend_key: str, prompt_key: str, value: str):
        data = self._load()
        data[f"{backend_key}|{prompt_key}"] = value
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp.replace(self.path)

    @staticmethod
    def prompt_hash(system: str, user: str) -> str:
        return hashlib.sha256(f"{system}||{user}".encode()).hexdigest()[:24]


def make_backend(base_url: str | None, model: str | None, api_key_env: str = "NOISE_INJECTOR_API_KEY", cache_path: Path | None = None):
    if not base_url or not model:
        raise ValueError(
            "LLM-powered ops need a backend: pass --llm-base-url and --llm-model "
            f"(API key read from env var {api_key_env})"
        )
    backend = LLMBackend(base_url, model, api_key_env)
    cache = LLMCache(cache_path) if cache_path else None
    return backend, cache


def cached_chat(backend: LLMBackend, cache: LLMCache | None, system: str, user: str, **kw) -> str:
    pkey = LLMCache.prompt_hash(system, user)
    if cache is not None:
        hit = cache.get(backend.cache_key_base, pkey)
        if hit is not None:
            return hit
    for attempt in range(3):
        try:
            out = backend.chat(system, user, **kw)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0 + attempt)
    if cache is not None:
        cache.put(backend.cache_key_base, pkey, out)
    return out
