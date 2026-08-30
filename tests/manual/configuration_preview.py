"""Loopback-only actual SPA + real configuration routers, with disposable data.

No application Runtime, model, market source, scheduler, or execution cycle is
constructed. Only local Paper connection checks are allowed. See the acceptance
guide for building and installing the example in an isolated target directory.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from api.dependencies import verify_api_key
from api.routes import config, venues
from cryptotrader.configuration.catalog import configuration_catalog
from cryptotrader.runtime_config.models import RuntimeConfigDocument
from cryptotrader.runtime_config.repository import API_ACCESS_CREDENTIAL_REF, RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault, TokenPayload
from cryptotrader.venues.ccxt_base import VenueOperationError
from cryptotrader.venues.paper import PaperVenueAdapter
from cryptotrader.venues.registry import VenueAdapterRegistry
from tests.factories.runtime_config import active_document, runtime_document


class RejectedExternalAdapter:
    def __init__(self, adapter_id):
        self.adapter_id = adapter_id

    def capabilities(self, environment):
        del environment
        return PaperVenueAdapter().capabilities("paper")

    async def connect(self, connection, credentials):
        del connection, credentials
        error = VenueOperationError("Fixture intentionally rejects external authentication")
        error.code = "authentication_failed"
        raise error


class PreviewRuntime:
    def __init__(self, repository, snapshot):
        self.repository, self.snapshot = repository, snapshot
        self.application_in_progress = False
        self.fail_next_publish = False
        self.lock = asyncio.Lock()
        catalog = configuration_catalog()
        self.signal_registry = SimpleNamespace(installed_ids=lambda: frozenset(catalog.components))
        self.market_registry = SimpleNamespace(installed_ids=lambda: frozenset(catalog.market_sources))
        self.venue_registry = VenueAdapterRegistry(
            (PaperVenueAdapter(), RejectedExternalAdapter("okx"), RejectedExternalAdapter("bybit"))
        )

    @asynccontextmanager
    async def application_barrier(self):
        async with self.lock:
            self.application_in_progress = True
            try:
                yield
            finally:
                self.application_in_progress = False

    async def prepare_candidate(self, snapshot):
        if snapshot.document.system.active != self.snapshot.document.system.active:
            raise ValueError("Fixture refuses activation")
        return self

    async def close(self):
        pass

    async def publish_candidate(self, candidate, pending):
        if self.fail_next_publish:
            self.fail_next_publish = False
            raise RuntimeError("Intentional fixture publication failure")

    async def activate_applied(self, snapshot):
        # Publication acknowledgement only: no trading Runtime exists here.
        self.snapshot = snapshot
        self.venue_registry = VenueAdapterRegistry(
            (PaperVenueAdapter(), RejectedExternalAdapter("okx"), RejectedExternalAdapter("bybit"))
        )

    async def fail_closed(self, snapshot):
        self.snapshot = snapshot


async def fixture_safety(request: Request, call_next):
    path = request.url.path
    if path == "/api/config" and request.method == "PUT":
        payload = await request.json()
        desired = payload.get("document", {})
        activation = (
            desired.get("system", {}).get("active") != request.app.state.runtime.snapshot.document.system.active
        )
        if activation or desired.get("execution", {}).get("live_order_execution_enabled"):
            return JSONResponse({"detail": "Fixture refuses activation and live execution"}, status_code=403)
    allowed_write = path == "/api/config" or path.startswith(
        (
            "/api/config/credentials/",
            "/api/venue-connections",
            "/__fixture__/",
        )
    )
    if request.method != "GET" and not allowed_write:
        return JSONResponse({"detail": "Fixture refuses execution, order and model operations"}, status_code=403)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'"
    )
    response.headers["X-Configuration-Preview"] = "disposable-fixture-no-runtime"
    return response


def preview_document(active_ui: bool):
    document = active_document() if active_ui else runtime_document()
    if active_ui:
        raw = document.model_dump(mode="json")
        raw["infrastructure"]["redis_url"] = "redis://127.0.0.1:1/0"
        raw["signals"]["components"] = [{"component_id": "kronos", "enabled": True, "weight": 1, "parameters": {}}]
        document = RuntimeConfigDocument.model_validate(raw)
    return document


def spa_response(spa_dir: Path, path: str):
    if path.startswith("api/"):
        return JSONResponse({"detail": "Fixture endpoint unavailable"}, status_code=404)
    candidate = (spa_dir / path).resolve()
    if not candidate.is_relative_to(spa_dir.resolve()):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    if candidate.is_file() and candidate.name != "index.html":
        return FileResponse(candidate)
    html = (spa_dir / "index.html").read_text()
    html = html.replace("<title>CryptoTrader AI</title>", "<title>隔离验收数据 · CryptoTrader AI</title>")
    html = html.replace(
        "<body>",
        (
            '<body><div role="note" style="padding:4px 12px;background:#fff4c2;'
            'color:#342700;font:14px system-ui">隔离验收数据 · 无交易运行时 · 仅本地配置保存与检查</div>'
        ),
    )
    return HTMLResponse(html)


async def create_preview(data_dir: Path, spa_dir: Path, *, active_ui=False):
    document = preview_document(active_ui)
    repository = RuntimeConfigRepository(
        f"sqlite+aiosqlite:///{data_dir / 'configuration-preview.db'}",
        CredentialVault("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        default_factory=lambda: document,
    )
    app = FastAPI()
    app.state.runtime = PreviewRuntime(repository, await repository.get_or_create())

    app.middleware("http")(fixture_safety)

    app.include_router(config.router, dependencies=[Depends(verify_api_key)])
    app.include_router(venues.router, dependencies=[Depends(verify_api_key)])

    @app.post("/__fixture__/fail-next-publication")
    async def fail_publication():
        app.state.runtime.fail_next_publish = True
        return {"fixture": "next save will persist then fail publication"}

    @app.post("/__fixture__/protect")
    async def protect():
        current = await repository.get_or_create()
        saved = await repository.put_token(
            current.revision, API_ACCESS_CREDENTIAL_REF, TokenPayload(token="fixture-access-only")
        )
        raw = saved.document.model_dump(mode="json")
        raw["security"]["enabled"] = True
        saved = await repository.replace(saved.revision, RuntimeConfigDocument.model_validate(raw))
        app.state.runtime.snapshot = await repository.mark_applied(saved.revision)
        return {"fixture": "protected with documented disposable key"}

    @app.get("/api/backtest/sessions", dependencies=[Depends(verify_api_key)])
    async def sessions():
        return {"sessions": ["fixture-saved-parameters"]}

    @app.get("/api/backtest/sessions/fixture-saved-parameters", dependencies=[Depends(verify_api_key)])
    async def saved_parameters():
        return {
            "name": "fixture-saved-parameters",
            "params": {"pair": "ETH/USDT", "start": "2025-01-01", "end": "2025-02-01", "initial_capital": 2500},
            "result": {},
            "saved_at": "2026-08-30T00:00:00Z",
        }

    @app.get("/api/scheduler/rules", dependencies=[Depends(verify_api_key)])
    async def rules():
        return []

    @app.get("/api/scheduler/status", dependencies=[Depends(verify_api_key)])
    async def scheduler_status():
        return {"enabled": False, "next_pair": None, "next_run_at": None, "redis_available": False}

    @app.get("/api/scheduler/triggers", dependencies=[Depends(verify_api_key)])
    async def history():
        return {"items": [], "total": 0, "page": 1, "size": 20}

    @app.get("/{path:path}")
    def spa(path: str):
        return spa_response(spa_dir, path)

    return app


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spa-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--active-ui", action="store_true", help="Seed operational UI only; never activate a Runtime")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cryptotrader-config-preview-") as directory:
        app = await create_preview(Path(directory), args.spa_dir, active_ui=args.active_ui)
        sys.stdout.write(
            f"DISPOSABLE FIXTURE http://127.0.0.1:{args.port} database={directory} active_ui={args.active_ui}\n"
        )
        sys.stdout.flush()
        await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")).serve()


if __name__ == "__main__":
    asyncio.run(main())
