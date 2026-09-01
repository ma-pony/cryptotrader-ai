"""File sessions are retired.

New runs use BacktestService/BacktestStore. Explicit legacy import is available
only as migrations.workbench.import_backtest_session(database_url, source).
There is no default home-directory lookup, file writer, or dual-read fallback.
"""
