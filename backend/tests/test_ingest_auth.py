"""The /ingest endpoint must never be callable without the shared secret."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import ingest as ingest_module
from app.api import security as security_module


@pytest.fixture
def client(monkeypatch):
    """Isolated app with just the ingest router — no lifespan, no network."""

    async def _never_runs(*args, **kwargs):
        raise AssertionError("ingest_pipeline must not run for a rejected request")

    monkeypatch.setattr(ingest_module, "ingest_pipeline", _never_runs)
    monkeypatch.setattr(
        ingest_module, "load_urls", lambda urls: (_ for _ in ()).throw(
            AssertionError("load_urls must not run for a rejected request")
        )
    )

    app = FastAPI()
    app.include_router(ingest_module.router, prefix="/api/v1")
    return TestClient(app)


def _set_key(monkeypatch, key: str):
    for module in (security_module, ingest_module):
        monkeypatch.setattr(module.settings, "ingest_api_key", key, raising=False)


def test_disabled_when_no_key_configured(client, monkeypatch):
    _set_key(monkeypatch, "")
    r = client.post("/api/v1/ingest", json={"urls": ["https://example.com"]})
    assert r.status_code == 503
    assert "disabled" in r.json()["detail"].lower()


def test_rejects_missing_key(client, monkeypatch):
    _set_key(monkeypatch, "s3cret")
    r = client.post("/api/v1/ingest", json={"urls": ["https://example.com"]})
    assert r.status_code == 401


def test_rejects_wrong_key(client, monkeypatch):
    _set_key(monkeypatch, "s3cret")
    r = client.post(
        "/api/v1/ingest",
        json={"urls": ["https://example.com"]},
        headers={"X-API-Key": "guess"},
    )
    assert r.status_code == 401


def test_valid_key_still_blocks_internal_urls(client, monkeypatch):
    """Authentication is not authorisation to fetch the internal network."""
    _set_key(monkeypatch, "s3cret")
    r = client.post(
        "/api/v1/ingest",
        json={"urls": ["http://169.254.169.254/latest/meta-data/"]},
        headers={"X-API-Key": "s3cret"},
    )
    assert r.status_code == 400
    assert "non-public" in r.json()["detail"]
