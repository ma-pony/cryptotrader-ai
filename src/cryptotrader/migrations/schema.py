"""Read-only guard for the explicitly migrated Workbench database schema."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from cryptotrader.db import get_engine


class MigrationRequired(RuntimeError):  # noqa: N818 - public typed boundary.
    def __init__(self, missing_tables: tuple[str, ...]) -> None:
        self.missing_tables = missing_tables
        names = ", ".join(missing_tables)
        super().__init__(f"workbench migration required; missing tables: {names}")


def _missing_sqlite_file(database_url: str) -> bool:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or url.database in {None, "", ":memory:"}:
        return False
    return not Path(url.database).exists()


async def require_tables(database_url: str, table_names) -> None:
    """Validate table presence without creating a database file or changing schema."""

    expected = frozenset(table_names)
    if _missing_sqlite_file(database_url):
        raise MigrationRequired(tuple(sorted(expected)))
    engine = await get_engine(database_url)
    async with engine.connect() as connection:
        current = await connection.run_sync(lambda sync: frozenset(inspect(sync).get_table_names()))
    missing = tuple(sorted(expected - current))
    if missing:
        raise MigrationRequired(missing)
