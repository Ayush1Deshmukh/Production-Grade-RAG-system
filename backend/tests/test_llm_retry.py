"""Transient provider failures must be retried, not surfaced as a 500."""

import httpx
import pytest
from openai import RateLimitError

from app.rag.chain import _invoke_llm_with_retry


class _FakeChain:
    """Chain stub that fails `failures` times before returning a value."""

    def __init__(self, failures: int, error: Exception | None = None):
        self.failures = failures
        self.calls = 0
        self.error = error or RateLimitError(
            "queue_exceeded",
            response=httpx.Response(429, request=httpx.Request("POST", "http://cerebras")),
            body=None,
        )

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "answer"


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    chain = _FakeChain(failures=2)
    result = await _invoke_llm_with_retry(chain, {"question": "q"}, [])
    assert result == "answer"
    assert chain.calls == 3


@pytest.mark.asyncio
async def test_gives_up_after_three_attempts():
    chain = _FakeChain(failures=99)
    with pytest.raises(RateLimitError):
        await _invoke_llm_with_retry(chain, {"question": "q"}, [])
    assert chain.calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_transient_errors():
    """A schema/parsing bug should fail fast rather than burn three calls."""
    chain = _FakeChain(failures=99, error=ValueError("bad schema"))
    with pytest.raises(ValueError):
        await _invoke_llm_with_retry(chain, {"question": "q"}, [])
    assert chain.calls == 1
