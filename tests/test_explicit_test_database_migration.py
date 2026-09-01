"""Test databases are migrated by their owner, never by a global monkeypatch."""

import pytest


async def test_ordinary_unmigrated_test_store_is_not_secretly_given_tables(tmp_path):
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.migrations.schema import MigrationRequired

    path = tmp_path / "ordinary-unmigrated.sqlite"
    store = AccountStore(f"sqlite+aiosqlite:///{path}")

    with pytest.raises(MigrationRequired, match="migration required"):
        await store.latest("missing")

    assert not path.exists()
