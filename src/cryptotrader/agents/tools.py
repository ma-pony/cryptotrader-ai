"""Agent tools for progressive disclosure pattern."""

import asyncio
import os

from langchain_core.tools import tool

from cryptotrader.agents.skills import load_skill_content


@tool
def load_skill(skill_name: str) -> str:
    """Load detailed analysis framework for specific trading concepts.

    Available skills:
    - funding_rate_analysis: Perpetual funding rate interpretation
    - btc_dominance_analysis: BTC dominance and altcoin implications
    - liquidation_cascade_analysis: Liquidation volume and cascade risk
    - fear_greed_interpretation: Fear & Greed Index contrarian signals

    Use this when you need detailed guidance on interpreting specific indicators.
    """
    return load_skill_content(skill_name)


@tool
def load_past_experience(context: str) -> str:
    """Load relevant past trading decisions and their outcomes.

    Use this when you need historical context to inform your analysis.
    Provide a brief description of what you're analyzing (e.g., 'BTC bullish breakout').

    Returns: Summary of similar past decisions and their results.
    """
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.learning.verbal import format_experience_text, get_experience

    db_url = os.environ.get("DATABASE_URL")
    store = JournalStore(db_url)

    async def _fetch() -> str:
        cases = await get_experience(store, {"context": context})
        return await format_experience_text(cases)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            experience = pool.submit(asyncio.run, _fetch()).result()
    else:
        experience = asyncio.run(_fetch())
    return experience if experience else "No relevant past experience found."
