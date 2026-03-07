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
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    config = load_config()
    pm = PortfolioManager(config.infrastructure.database_url)
    portfolio = _run(pm.get_portfolio())
    daily_pnl = _run(pm.get_daily_pnl())
    drawdown = _run(pm.get_drawdown())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"${portfolio['total_value']:,.2f}")
    col2.metric(
        "Daily PnL", f"${daily_pnl:,.2f}", delta=f"{daily_pnl:+,.2f}" if daily_pnl != 0 else None, delta_color="normal"
    )
    col3.metric(
        "Drawdown", f"{drawdown:.2%}", delta=f"{drawdown:.2%}" if drawdown != 0 else None, delta_color="inverse"
    )

    # Equity curve chart
    snapshots = _run(pm._load_snapshots("default"))
    if snapshots:
        import pandas as pd

        df = pd.DataFrame(snapshots)
        if "timestamp" in df.columns and "total_value" in df.columns:
            df = df.set_index("timestamp")
            st.subheader("Equity Curve")
            st.line_chart(df["total_value"])

    # Positions table
    positions = portfolio.get("positions", {})
    if positions:
        st.subheader("Positions")
        st.table(
            [
                {
                    "Pair": k,
                    "Amount": f"{v['amount']:.6f}",
                    "Avg Price": f"${v['avg_price']:,.2f}",
                    "Value": f"${v['amount'] * v['avg_price']:,.2f}",
                }
                for k, v in positions.items()
            ]
        )
    else:
        st.info("No open positions.")

elif page == "Decisions":
    st.title("Recent Decisions")
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import JournalStore

    config = load_config()
    store = JournalStore(config.infrastructure.database_url)
    commits = _run(store.log(limit=20))
    if not commits:
        st.info("No decisions recorded yet.")
    for c in commits:
        action = c.verdict.action if c.verdict else "N/A"
        conf = c.verdict.confidence if c.verdict else 0
        div = c.divergence if c.divergence is not None else 0.0
        risk_icon = "✅" if (c.risk_gate and c.risk_gate.passed) else "❌"
        with st.expander(f"{risk_icon} {c.pair} | {action.upper()} | conf={conf:.0%} | div={div:.2%} | {c.hash[:8]}"):
            col1, col2 = st.columns(2)
            col1.write(f"**Timestamp:** {c.timestamp}")
            col1.write(f"**Debate Rounds:** {c.debate_rounds}")
            col2.write(f"**Risk Gate:** {'PASS' if c.risk_gate and c.risk_gate.passed else 'REJECT'}")
            if c.risk_gate and not c.risk_gate.passed:
                col2.write(f"**Rejected by:** {c.risk_gate.rejected_by}")
                col2.write(f"**Reason:** {c.risk_gate.reason}")
            if c.pnl is not None:
                col2.write(f"**PnL:** ${c.pnl:,.2f}")

            # Agent analyses detail
            if c.analyses:
                st.subheader("Agent Analyses")
                agent_cols = st.columns(len(c.analyses))
                for i, (aid, analysis) in enumerate(c.analyses.items()):
                    with agent_cols[i]:
                        if hasattr(analysis, "direction"):
                            direction = analysis.direction
                            confidence = analysis.confidence
                            reasoning = analysis.reasoning
                        elif isinstance(analysis, dict):
                            direction = analysis.get("direction", "?")
                            confidence = analysis.get("confidence", 0)
                            reasoning = analysis.get("reasoning", "")
                        else:
                            continue
                        dir_color = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
                        st.write(f"**{aid}** {dir_color}")
                        st.write(f"Direction: {direction}")
                        st.write(f"Confidence: {confidence:.0%}")
                        if reasoning:
                            st.caption(reasoning[:200])

            # Verdict detail
            if c.verdict:
                st.subheader("Verdict")
                st.write(f"**Action:** {c.verdict.action} | **Confidence:** {c.verdict.confidence:.0%}")
                if hasattr(c.verdict, "reasoning") and c.verdict.reasoning:
                    st.caption(c.verdict.reasoning)

elif page == "Risk Status":
    st.title("Risk Status")
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    rsm = RedisStateManager(config.infrastructure.redis_url)
    if rsm.available:
        hourly, daily = _run(rsm.get_trade_counts())
        cb = _run(rsm.is_circuit_breaker_active())
        col1, col2, col3 = st.columns(3)
        col1.metric("Hourly Trades", hourly)
        col2.metric("Daily Trades", daily)
        col3.metric("Circuit Breaker", "🔴 ACTIVE" if cb else "🟢 Inactive")
        if cb and st.button("Reset Circuit Breaker"):
            _run(rsm.reset_circuit_breaker())
            st.rerun()
    else:
        st.warning("Redis not connected — risk state unavailable.")

    # Show risk config summary
    try:
        from cryptotrader.config import load_config

        config = load_config()
        st.subheader("Risk Parameters")
        st.table(
            [
                {"Check": "Max Position", "Value": f"{config.risk.position.max_single_pct:.0%}"},
                {"Check": "Max Exposure", "Value": f"{config.risk.position.max_total_exposure_pct:.0%}"},
                {"Check": "Daily Loss Limit", "Value": f"{config.risk.loss.max_daily_loss_pct:.0%}"},
                {"Check": "Max Drawdown", "Value": f"{config.risk.loss.max_drawdown_pct:.0%}"},
                {"Check": "Cooldown", "Value": f"{config.risk.cooldown.same_pair_minutes}m"},
            ]
        )
    except Exception:
        pass

elif page == "Backtest":
    st.title("Backtest")
    col1, col2 = st.columns(2)
    pair = col1.text_input("Pair", "BTC/USDT")
    interval = col2.selectbox("Interval", ["1h", "4h", "1d"])
    start = col1.date_input("Start")
    end = col2.date_input("End")
    capital = col1.number_input("Initial Capital", value=10000, step=1000)
    if st.button("Run Backtest"):
        from cryptotrader.backtest.engine import BacktestEngine

        with st.spinner("Running backtest..."):
            engine = BacktestEngine(pair, str(start), str(end), interval, initial_capital=capital)
            result = _run(engine.run())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return", f"{result.total_return:.2%}")
        c2.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
        c3.metric("Max DD", f"{result.max_drawdown:.2%}")
        c4.metric("Win Rate", f"{result.win_rate:.2%}")
        if result.equity_curve:
            st.subheader("Equity Curve")
            st.line_chart(result.equity_curve)
        if result.trades:
            st.subheader(f"Trades ({len(result.trades)})")
            import pandas as pd

            df = pd.DataFrame(result.trades)
            st.dataframe(df, use_container_width=True)
