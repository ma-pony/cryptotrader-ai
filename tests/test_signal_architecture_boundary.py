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
        "position_scale",
    )
    runtime = "\n".join(path.read_text(errors="ignore") for path in (root / "src").rglob("*.py"))
    for symbol in forbidden:
        assert symbol not in runtime


def test_obsolete_mid_cycle_user_injection_is_absent_from_current_runtime():
    root = Path(__file__).parents[1]
    paths = [*(root / "src").rglob("*.py"), *(root / "config").rglob("*.toml")]
    current_contract_tests = [path for path in (root / "tests").rglob("*.py") if path != Path(__file__)]
    runtime_and_tests = "\n".join(path.read_text(errors="ignore") for path in [*paths, *current_contract_tests])
    forbidden = "ste" + "er"

    assert forbidden not in runtime_and_tests.lower()
