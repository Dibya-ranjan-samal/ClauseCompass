"""Ollama LLM access with a SQLite response cache.

The cache matters more than it looks: the generation eval runs 10+ times today
over 148 questions plus one judge call per claim. Without it that is thousands
of local inference calls; with it, one per distinct prompt.

No `anthropic`/`openai` SDK - httpx is already a dependency and Ollama speaks
plain HTTP. No LLMProvider Protocol either: there is one provider, and an
interface with one implementation is speculative. `model` is a constructor
argument, which is all the 3b-vs-7b swap and the Task 8 comparison need.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

URL = "http://localhost:11434/api/generate"
# Ollama defaults num_ctx to 2048 and truncates over-long prompts SILENTLY
# (Day 1 F8). Six clause chunks plus the JSON schema runs past that.
NUM_CTX = 8192


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float          # 0.0 for local, kept so the schema matches paid providers
    latency_ms: float
    model: str
    cached: bool = False
    truncated: bool = False


class OllamaProvider:
    def __init__(self, model="qwen2.5:3b", cache_path="data/llm_cache.db",
                 num_ctx=NUM_CTX, temperature=0.0):
        self.model, self.num_ctx, self.temperature = model, num_ctx, temperature
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(cache_path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    @property
    def name(self):
        return f"ollama:{self.model}"

    def _key(self, system, user, max_tokens):
        blob = json.dumps([self.model, system, user, max_tokens, self.num_ctx,
                           self.temperature], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> LLMResponse:
        k = self._key(system, user, max_tokens)
        row = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            return LLMResponse(**{**json.loads(row[0]), "cached": True})

        t0 = time.perf_counter()
        r = httpx.post(URL, timeout=600.0, json={
            "model": self.model, "system": system, "prompt": user, "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens}})
        r.raise_for_status()
        b = r.json()
        used = b.get("prompt_eval_count") or 0
        resp = LLMResponse(
            text=b["response"].strip(), input_tokens=used,
            output_tokens=b.get("eval_count") or 0, cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000, model=self.model,
            truncated=used >= self.num_ctx * 0.98)
        d = resp.__dict__.copy()
        d.pop("cached")
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                          (k, json.dumps(d)))
        self.conn.commit()
        return resp
