"""Independent, explicitly started read-only account owner."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime

from cryptotrader.runtime_config.repository import CredentialNotConfigured


class AccountSyncError(RuntimeError):
    pass


def _validate_page(page, cursor, window, expected_start):
    if page.coverage_start is None or page.coverage_end is None:
        raise ValueError("history coverage unavailable")
    if page.coverage_start > page.coverage_end:
        raise ValueError("history coverage invalid")
    current_window = (page.coverage_start, page.coverage_end)
    if window is None and expected_start is not None and page.coverage_start != expected_start:
        raise ValueError("history coverage has a gap")
    if window is not None and window != current_window:
        raise ValueError("history pagination changed its fixed window")
    if page.next_cursor is None or page.next_cursor == cursor:
        raise ValueError("history checkpoint did not progress")
    return current_window


async def _read_history(session, kind, prior, target):
    cursor = prior.get("cursor")
    first_start = prior.get("coverage_start")
    expected_start = datetime.fromisoformat(prior["coverage_end"]) if prior.get("coverage_end") else None
    window, items = None, []
    fetch = session.fetch_fills if kind == "fills" else session.fetch_funding
    while True:
        page = await fetch(cursor)
        window = _validate_page(page, cursor, window, expected_start)
        if first_start is None:
            first_start = page.coverage_start.isoformat()
        items.extend(page.items)
        cursor = page.next_cursor
        if page.complete and page.coverage_end >= target:
            return tuple(items), {
                "cursor": cursor,
                "coverage_start": first_start,
                "coverage_end": page.coverage_end.isoformat(),
                "complete": True,
                "from_inception": prior.get("from_inception", getattr(session, "history_from_inception", False)),
            }
        if page.complete:
            if expected_start is not None and page.coverage_end <= expected_start:
                raise ValueError("history window did not progress")
            expected_start = page.coverage_end
            window = None


class AccountSyncService:
    def __init__(self, store, session_provider):
        self.store = store
        self.session_provider = session_provider
        self._locks = {}

    async def sync(self, connection_id):
        async with self._locks.setdefault(connection_id, asyncio.Lock()):
            try:
                async with self.session_provider(connection_id) as session:
                    return await self._sync_session(session, connection_id)
            except asyncio.CancelledError:
                raise
            except CredentialNotConfigured:
                reason = "账户凭据未配置"
            except Exception:
                # Never persist a connector exception containing credentials or raw requests.
                reason = "账户同步失败，请检查连接权限与历史覆盖范围"  # noqa: RUF001
            await self.store.failed(connection_id, reason)
            raise AccountSyncError(reason) from None

    async def sync_session(self, session, connection_id):
        """Serialize the complete ledger even when the caller already owns a venue session."""
        async with self._locks.setdefault(connection_id, asyncio.Lock()):
            return await self._sync_session(session, connection_id)

    async def _sync_session(self, session, connection_id):
        """Caller holds this service's account lock for the whole history batch."""
        snapshot = await session.fetch_account()
        if snapshot.connection_id != connection_id:
            raise ValueError("account identity mismatch")
        status = await self.store.status(connection_id)
        checkpoints, batches = {}, {}
        for kind in ("fills", "funding"):
            batches[kind], checkpoints[kind] = await _read_history(
                session, kind, status["coverage"].get(kind, {}), snapshot.observed_at
            )
        await self.store.ingest(snapshot, batches["fills"], batches["funding"], checkpoints=checkpoints)
        return snapshot


class AccountSyncOwner:
    def __init__(self, service, repository):
        self.service = service
        self.repository = repository
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="account-read-sync")

    def refresh(self):
        self._wake.set()

    async def stop(self):
        self._stop.set()
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            snapshot = await self.repository.get_existing()
            for connection in snapshot.document.execution.connections:
                if self._stop.is_set():
                    break
                with suppress(AccountSyncError):
                    await self.service.sync(connection.id)
            if not self._stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), snapshot.document.accounts.sync_interval_seconds)
