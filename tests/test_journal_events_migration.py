"""Operational event writes never own journal schema DDL."""


async def test_missing_event_journal_soft_fails_without_creating_database(tmp_path):
    from cryptotrader.journal.events import _write_journal_event

    path = tmp_path / "unmigrated-events.sqlite"
    await _write_journal_event(
        f"sqlite+aiosqlite:///{path}",
        trace_id="trace-missing-schema",
        event_type="phase1_rejected",
        pair="BTC/USDT",
        payload={"reason": "fixture"},
    )

    assert not path.exists()
