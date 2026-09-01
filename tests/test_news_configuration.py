"""Optional news token is write-only, encrypted and consumed only at runtime."""

import asyncio

import httpx
import pytest

from cryptotrader.runtime_config.models import SecurityConfig
from cryptotrader.runtime_config.secrets import TokenPayload


async def test_inactive_news_token_requires_the_configured_api_key(api_harness):
    runtime = api_harness.runtime
    snapshot = await runtime.repository.put_token(
        runtime.snapshot.revision, "api-access", TokenPayload(token="fixture-access")
    )
    document = snapshot.document.model_copy(update={"security": SecurityConfig(enabled=True)})
    pending = await runtime.repository.replace(snapshot.revision, document)
    runtime.snapshot = await runtime.repository.mark_applied(pending.revision)
    revision = runtime.snapshot.revision
    for key, expected in ((None, 401), ("fixture-wrong", 401), ("fixture-access", 200)):
        response = await api_harness.client.put(
            "/api/config/credentials/news-provider",
            headers={"X-API-Key": key} if key else {},
            json={"expected_revision": revision, "token": "fixture-news-secret"},
        )
        assert response.status_code == expected
        assert "fixture-news-secret" not in response.text
    assert runtime.snapshot.document.scheduler.automation_enabled is False


async def test_news_mutation_waits_for_actual_apply_barrier_without_persisting_early(api_harness):
    runtime = api_harness.runtime
    pending = await runtime.repository.replace(
        runtime.snapshot.revision, runtime.snapshot.document.model_copy(update={})
    )
    runtime.snapshot = await runtime.repository.mark_applied(pending.revision)
    revision = runtime.snapshot.revision
    started = asyncio.Event()

    async def save():
        started.set()
        return await api_harness.client.put(
            "/api/config/credentials/news-provider",
            json={"expected_revision": revision, "token": "fixture-queued-news"},
        )

    async with runtime.application_barrier():
        queued = asyncio.create_task(save())
        await started.wait()
        await asyncio.sleep(0)
        assert not queued.done()
        assert (await runtime.repository.get_or_create()).revision == revision
        assert (await api_harness.client.get("/api/config")).status_code == 503
    assert (await queued).status_code == 200
    assert (await runtime.repository.get_or_create()).revision == revision + 1
    assert runtime.snapshot.document.scheduler.automation_enabled is False


@pytest.mark.parametrize(
    ("protected", "applying", "expected"), [(False, False, 200), (True, False, 401), (False, True, 200)]
)
async def test_inactive_news_token_http_mutation_keeps_auth_and_apply_admission(
    api_harness, protected, applying, expected
):
    runtime = api_harness.runtime
    current = runtime.snapshot
    document = current.document.model_copy(update={"security": SecurityConfig(enabled=protected)})
    pending = await runtime.repository.replace(current.revision, document)
    runtime.snapshot = await runtime.repository.mark_applied(pending.revision)
    runtime.application_in_progress = applying
    revision = runtime.snapshot.revision
    response = await api_harness.client.put(
        "/api/config/credentials/news-provider",
        json={
            "expected_revision": revision,
            "token": "fixture-news-marker",
        },
    )
    assert response.status_code == expected
    assert "fixture-news-marker" not in response.text
    saved = await runtime.repository.get_or_create()
    assert saved.document.scheduler.automation_enabled is False
    if expected == 200:
        assert saved.revision == revision + 1
        payload = (await api_harness.client.get("/api/config")).json()
        assert payload["document"]["market_data"]["news_credential_configured"] is True
        assert payload["document"]["market_data"]["news_credential_updated_at"]
        assert "fixture-news-marker" not in str(payload)
        stale = await api_harness.client.put(
            "/api/config/credentials/news-provider",
            json={
                "expected_revision": revision,
                "token": "fixture-stale-marker",
            },
        )
        assert stale.status_code == 409
    else:
        assert saved.revision == revision


@pytest.mark.parametrize("configured", [False, True])
async def test_runtime_injects_optional_vault_news_token_into_actual_news_request(api_harness, monkeypatch, configured):
    import cryptotrader.runtime as runtime_module
    from cryptotrader.cycle_events import NullCycleEventSink

    repository = api_harness.runtime.repository
    snapshot = await repository.get_or_create()
    if configured:
        snapshot = await repository.put_token(
            snapshot.revision, "news-provider", TokenPayload(token="fixture-runtime-news")
        )
    monkeypatch.setattr(runtime_module.SignalComponentRegistry, "discover", lambda *a, **k: object())
    _, _, markets = await runtime_module._discover_registry_graph(snapshot.document, NullCycleEventSink(), repository)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"Data": [{"TITLE": "News", "title": "News", "BODY": "Details"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("cryptotrader.data.news.httpx.AsyncClient", lambda **kwargs: client)
    articles = await markets.require("default").aggregator.news._collect_cryptocompare("BTC")
    assert [article.title for article in articles] == ["News"]
    if configured:
        assert requests[0].url.host == "data-api.coindesk.com"
        assert requests[0].headers["Authorization"] == "Bearer fixture-runtime-news"
    else:
        assert requests[0].url.host == "min-api.cryptocompare.com"
        assert "Authorization" not in requests[0].headers
    assert "fixture-runtime-news" not in snapshot.document.model_dump_json()
