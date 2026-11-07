import httpx
import pytest

from clausecompass.generation import llm


def _ok(content="{}", finish="stop", usage=(1000, 200)):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}})


@pytest.fixture
def calls(monkeypatch):
    """Queue responses for httpx.post and record each request body."""
    sent, queue = [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return queue.pop(0)

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return sent, queue


def groq(tmp_path, model="openai/gpt-oss-120b"):
    return llm.GroqProvider(model, api_key="test", cache_path=tmp_path / "c.db")


def test_reasoning_model_gets_headroom_and_hidden_reasoning(calls, tmp_path):
    """Reasoning tokens share the completion budget: without headroom a long
    think returns empty content, which the parser would score as a refusal."""
    sent, queue = calls
    queue.append(_ok())
    groq(tmp_path).complete("sys", "user", max_tokens=700)
    assert sent[0]["max_completion_tokens"] == 700 + llm.REASONING_HEADROOM
    assert sent[0]["include_reasoning"] is False
    assert sent[0]["reasoning_effort"] == "low"


def test_non_reasoning_model_gets_plain_request(calls, tmp_path):
    sent, queue = calls
    queue.append(_ok())
    groq(tmp_path, "llama-3.3-70b-versatile").complete("sys", "user", max_tokens=700)
    assert sent[0]["max_completion_tokens"] == 700
    assert "reasoning_effort" not in sent[0] and "include_reasoning" not in sent[0]


def test_cost_from_usage_and_truncation_flag(calls, tmp_path):
    _, queue = calls
    queue.append(_ok("partial", finish="length", usage=(1_000_000, 1_000_000)))
    r = groq(tmp_path).complete("sys", "user")
    assert r.cost_usd == pytest.approx(0.15 + 0.60)
    assert r.truncated and r.text == "partial"


def test_second_identical_call_is_served_from_cache(calls, tmp_path):
    sent, queue = calls
    queue.append(_ok("answer"))
    p = groq(tmp_path)
    p.complete("sys", "user")
    again = p.complete("sys", "user")
    assert len(sent) == 1 and again.cached and again.text == "answer"


def test_rate_limit_is_retried(calls, tmp_path):
    sent, queue = calls
    queue += [httpx.Response(429, headers={"retry-after": "0"}), _ok("after retry")]
    assert groq(tmp_path).complete("sys", "user").text == "after retry"
    assert len(sent) == 2


def test_auth_error_surfaces_groq_message(calls, tmp_path):
    _, queue = calls
    queue.append(httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(RuntimeError, match="401: Invalid API Key"):
        groq(tmp_path).complete("sys", "user")


def test_make_llm_picks_provider_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # providers create data/llm_cache.db relative to cwd
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.make_llm().name == "ollama:qwen2.5:3b"
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert llm.make_llm().name == f"groq:{llm.DEFAULT_GROQ_MODEL}"
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm.make_llm().name.startswith("ollama:")


def test_rejected_native_tool_call_is_returned_as_text(calls, tmp_path):
    """gpt-oss emits a native function call when a prompt describes tools; Groq
    400s it but includes the generation, which is the model's real answer."""
    _, queue = calls
    gen = '{"name": "search_clauses", "arguments": {"query": "cap"}}'
    queue.append(httpx.Response(400, json={"error": {
        "code": "tool_use_failed", "message": "Tool choice is none, but model called a tool",
        "failed_generation": gen}}))
    assert groq(tmp_path).complete("sys", "user").text == gen
