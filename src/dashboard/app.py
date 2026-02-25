"""Streamlit dashboard for CryptoTrader AI."""

from __future__ import annotations

import asyncio
import streamlit as st


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


st.set_page_config(page_title="CryptoTrader AI", layout="wide")
page = st.sidebar.radio("Navigation", ["Overview", "Decisions", "Risk Status", "Backtest"])


if page == "Overview":
    st.title("Overview")
    from cryptotrader.portfolio.manager import PortfolioManager
    import os
    pm = PortfolioManager(os.environ.get("DATABASE_URL"))
    portfolio = _run(pm.get_portfolio())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"${portfolio['total_value']:,.2f}")
    col2.metric("Daily PnL", f"${_run(pm.get_daily_pnl()):,.2f}")
    col3.metric("Drawdown", f"{_run(pm.get_drawdown()):.2%}")
    positions = portfolio.get("positions", {})
    if positions:
        st.subheader("Positions")
        st.table([{"pair": k, **v} for k, v in positions.items()])

elif page == "Decisions":
    st.title("Recent Decisions")
    from cryptotrader.journal.store import JournalStore
    import os
    store = JournalStore(os.environ.get("DATABASE_URL"))
    commits = _run(store.log(limit=20))
    if not commits:
        st.info("No decisions recorded yet.")
    for c in commits:
        action = c.verdict.action if c.verdict else "N/A"
        conf = c.verdict.confidence if c.verdict else 0
        div = c.divergence if c.divergence is not None else 0.0
        with st.expander(f"{c.hash} | {c.pair} | {action} | conf={conf:.0%} | div={div:.2%}"):
            st.json({"timestamp": str(c.timestamp), "pair": c.pair,
                      "debate_rounds": c.debate_rounds,
                      "risk_gate": c.risk_gate.passed if c.risk_gate else None,
                      "pnl": c.pnl})

elif page == "Risk Status":
    st.title("Risk Status")
    from cryptotrader.risk.state import RedisStateManager
    import os
    rsm = RedisStateManager(os.environ.get("REDIS_URL"))
    if rsm.available:
        hourly, daily = _run(rsm.get_trade_counts())
        cb = _run(rsm.is_circuit_breaker_active())
        st.metric("Hourly Trades", hourly)
        st.metric("Daily Trades", daily)
        st.metric("Circuit Breaker", "ACTIVE" if cb else "Inactive")
        if cb and st.button("Reset Circuit Breaker"):
            _run(rsm.reset_circuit_breaker())
            st.rerun()
    else:
        st.warning("Redis not connected")

elif page == "Backtest":
    st.title("Backtest")
    col1, col2 = st.columns(2)
    pair = col1.text_input("Pair", "BTC/USDT")
    interval = col2.selectbox("Interval", ["1h", "4h", "1d"])
    start = col1.date_input("Start")
    end = col2.date_input("End")
    if st.button("Run Backtest"):
        from cryptotrader.backtest.engine import BacktestEngine
        with st.spinner("Running backtest..."):
            engine = BacktestEngine(pair, str(start), str(end), interval)
            result = _run(engine.run())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return", f"{result.total_return:.2%}")
        c2.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
        c3.metric("Max DD", f"{result.max_drawdown:.2%}")
        c4.metric("Win Rate", f"{result.win_rate:.2%}")
        if result.equity_curve:
            st.line_chart(result.equity_curve)
