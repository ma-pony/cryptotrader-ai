from pathlib import Path


def test_legacy_trading_architecture_is_absent():
    root = Path(__file__).parents[1]
    assert not (root / "src/cryptotrader/graph.py").exists()
    assert not (root / "src/cryptotrader/state.py").exists()
    assert not (root / "src/cryptotrader/nodes").exists()

    forbidden = (
        "signal_engine",
        "TradeVerdict",
        "verdict_source",
        "build_trading_graph",
        "build_kronos_graph",
        "build_backtest_graph",
        "build_initial_state",
        "verdict_partial",
    )
    runtime = "\n".join(path.read_text(errors="ignore") for path in (root / "src").rglob("*.py"))
    for symbol in forbidden:
        assert symbol not in runtime
