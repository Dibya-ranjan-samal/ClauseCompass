"""LLM access - Groq (hosted) or Ollama (local) - with a SQLite response cache.

The cache matters more than it looks: the generation eval runs 10+ times today
over 148 questions plus one judge call per claim. Without it that is thousands
of inference calls; with it, one per distinct prompt.

No `anthropic`/`openai`/`groq` SDK - httpx is already a dependency, and both
Ollama and Groq's OpenAI-compatible endpoint speak plain HTTP. The two providers
share the cache and `complete()`; only `_call` differs. `make_llm()` picks one:
GROQ_API_KEY set -> Groq, otherwise local Ollama. LLM_PROVIDER forces either.

The recorded eval numbers were measured on local qwen2.5, so the eval and
audit scripts still construct OllamaProvider explicitly.
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Ollama defaults num_ctx to 2048 and truncates over-long prompts SILENTLY
# (Day 1 F8). Six clause chunks plus the JSON schema runs past that.
NUM_CTX = 8192

GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# USD per 1M tokens (input, output), from console.groq.com/docs/models in
# October 2026. A model missing here is costed at 0 - check the console.
GROQ_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}
# gpt-oss reasoning tokens count against max_completion_tokens. Without
# headroom a long think spends the whole budget and returns empty content with
# finish_reason "length" - which the answer parser would score as a refusal.
REASONING_HEADROOM = 2048


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float          # 0.0 for local
    latency_ms: float
    model: str
    cached: bool = False
    truncated: bool = False


class _CachedProvider:
    model: str

    def __init__(self, cache_path="data/llm_cache.db"):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(cache_path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    def _key_parts(self, system, user, max_tokens) -> list: ...

    def _call(self, system, user, max_tokens) -> LLMResponse: ...

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        blob = json.dumps(self._key_parts(system, user, max_tokens), sort_keys=True)
        k = hashlib.sha256(blob.encode()).hexdigest()
        row = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            return LLMResponse(**{**json.loads(row[0]), "cached": True})

        resp = self._call(system, user, max_tokens)
        d = resp.__dict__.copy()
        d.pop("cached")
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                          (k, json.dumps(d)))
        self.conn.commit()
        return resp


class OllamaProvider(_CachedProvider):
    def __init__(self, model="qwen2.5:3b", cache_path="data/llm_cache.db",
                 num_ctx=NUM_CTX, temperature=0.0):
        super().__init__(cache_path)
        self.model, self.num_ctx, self.temperature = model, num_ctx, temperature

    @property
    def name(self):
        return f"ollama:{self.model}"

    def _key_parts(self, system, user, max_tokens):
        # Unchanged from before Groq existed, so the existing cache still hits.
        return [self.model, system, user, max_tokens, self.num_ctx, self.temperature]

    def ping(self) -> bool:
        try:
            return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
        except Exception:
            return False

    def _call(self, system, user, max_tokens):
        t0 = time.perf_counter()
        r = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=600.0, json={
            "model": self.model, "system": system, "prompt": user, "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens}})
        r.raise_for_status()
        b = r.json()
        used = b.get("prompt_eval_count") or 0
        return LLMResponse(
            text=b["response"].strip(), input_tokens=used,
            output_tokens=b.get("eval_count") or 0, cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000, model=self.model,
            truncated=used >= self.num_ctx * 0.98)


class GroqProvider(_CachedProvider):
    def __init__(self, model=DEFAULT_GROQ_MODEL, api_key=None,
                 cache_path="data/llm_cache.db", temperature=0.0,
                 reasoning_effort="low", max_retries=4):
        super().__init__(cache_path)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GroqProvider needs GROQ_API_KEY")
        self.model, self.temperature = model, temperature
        self.reasoning_effort, self.max_retries = reasoning_effort, max_retries
        self._ping_at, self._ping_ok = 0.0, False

    @property
    def name(self):
        return f"groq:{self.model}"

    @property
    def _reasons(self):
        return self.model.startswith("openai/gpt-oss")

    def _key_parts(self, system, user, max_tokens):
        return ["groq", self.model, system, user, max_tokens, self.temperature,
                self.reasoning_effort if self._reasons else None]

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def ping(self) -> bool:
        """Lists models, which also proves the key works. Cached for a minute:
        the UI polls /healthz every 15 s."""
        if time.monotonic() - self._ping_at > 60:
            try:
                self._ping_ok = httpx.get(f"{GROQ_URL}/models", headers=self._headers(),
                                          timeout=5).status_code == 200
            except Exception:
                self._ping_ok = False
            self._ping_at = time.monotonic()
        return self._ping_ok

    def _call(self, system, user, max_tokens):
        body = {"model": self.model, "temperature": self.temperature,
                "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if self._reasons:
            body.update(reasoning_effort=self.reasoning_effort, include_reasoning=False,
                        max_completion_tokens=max_tokens + REASONING_HEADROOM)

        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            r = httpx.post(f"{GROQ_URL}/chat/completions", headers=self._headers(),
                           json=body, timeout=120.0)
            # 429 is the free tier's per-minute limit; 5xx is transient.
            if r.status_code in (429, 500, 502, 503) and attempt < self.max_retries:
                wait = float(r.headers.get("retry-after") or 2 ** attempt)
                time.sleep(min(wait, 30))
                continue
            break
        err = {}
        if r.is_error:
            try:
                err = r.json()["error"]
            except Exception:
                raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:300]}") from None
        if err.get("code") == "tool_use_failed" and err.get("failed_generation"):
            # gpt-oss answers any prompt that DESCRIBES tools with a native
            # function call, even with none declared and even in JSON mode, and
            # Groq rejects it with 400. The rejected text is still the model's
            # answer ({"name", "arguments"}), so return it; the agent loop
            # accepts that shape. Groq reports no usage here, so cost reads 0.
            return LLMResponse(
                text=err["failed_generation"].strip(), input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=(time.perf_counter() - t0) * 1000,
                model=self.model)
        if err:
            raise RuntimeError(f"Groq HTTP {r.status_code}: {err.get('message', err)}")

        b = r.json()
        choice = b["choices"][0]
        usage = b.get("usage") or {}
        n_in, n_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        p_in, p_out = GROQ_PRICES.get(self.model, (0.0, 0.0))
        return LLMResponse(
            text=(choice["message"].get("content") or "").strip(),
            input_tokens=n_in, output_tokens=n_out,
            cost_usd=(n_in * p_in + n_out * p_out) / 1e6,
            latency_ms=(time.perf_counter() - t0) * 1000, model=self.model,
            truncated=choice.get("finish_reason") == "length")


def make_llm(model: str | None = None):
    """GROQ_API_KEY set -> Groq (GROQ_MODEL, default gpt-oss-120b); otherwise
    local Ollama. LLM_PROVIDER=groq|ollama overrides the choice."""
    provider = os.environ.get("LLM_PROVIDER") or (
        "groq" if os.environ.get("GROQ_API_KEY") else "ollama")
    if provider == "groq":
        return GroqProvider(model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    if provider == "ollama":
        return OllamaProvider(model or "qwen2.5:3b")
    raise ValueError(f"LLM_PROVIDER must be groq or ollama, not {provider!r}")
