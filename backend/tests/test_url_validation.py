import pytest

from app.ingestion.url_validation import (
    UrlNotAllowed,
    validate_ingest_url,
    validate_ingest_urls,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "http://127.0.0.1:6333/collections",         # loopback / local Qdrant
        "http://localhost:8000/health",
        "http://10.0.0.5/internal",                  # private range
        "http://192.168.1.1/admin",
        "http://[::1]:8000/",                        # IPv6 loopback
    ],
)
def test_rejects_internal_targets(url):
    """The server must refuse to fetch anything that isn't publicly routable."""
    with pytest.raises(UrlNotAllowed):
        validate_ingest_url(url, allowlist=[])


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/data",
        "gopher://example.com/",
        "not-a-url",
    ],
)
def test_rejects_non_http_schemes(url):
    with pytest.raises(UrlNotAllowed):
        validate_ingest_url(url, allowlist=[])


def test_rejects_host_outside_allowlist():
    with pytest.raises(UrlNotAllowed, match="allowlist"):
        validate_ingest_url(
            "https://evil.example.com/page", allowlist=["python.langchain.com"]
        )


def test_allows_subdomain_of_allowlisted_host():
    # Resolution still applies, so this only asserts the allowlist branch itself.
    from app.ingestion.url_validation import _host_allowed

    assert _host_allowed("docs.python.langchain.com", ["python.langchain.com"])
    assert _host_allowed("python.langchain.com", ["python.langchain.com"])
    assert not _host_allowed("python.langchain.com.evil.net", ["python.langchain.com"])


def test_reports_every_bad_url_at_once():
    with pytest.raises(UrlNotAllowed) as exc:
        validate_ingest_urls(
            ["file:///etc/passwd", "http://127.0.0.1/"], allowlist=[]
        )
    message = str(exc.value)
    assert "file:///etc/passwd" in message
    assert "127.0.0.1" in message
