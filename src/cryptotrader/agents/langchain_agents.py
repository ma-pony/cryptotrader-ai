"""Agent implementations using LangChain's official create_agent() API.

This module follows the supervisor pattern from LangChain docs:
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from cryptotrader.agents.base import _create_chat_model
from cryptotrader.agents.skills import get_skill_descriptions
from cryptotrader.agents.tools import load_past_experience, load_skill

# ── Agent System Prompts ──

TECH_AGENT_PROMPT = f"""You are a technical analysis specialist for cryptocurrency trading.

Your role: Analyze price action, volume, volatility, and technical indicators.

Available skills (use load_skill tool to get detailed guidance):
{get_skill_descriptions()}

Analysis framework:
- Base analysis ONLY on provided data, not general market knowledge
- Every claim must reference specific data points
- Take directional stance when data supports it (avoid defaulting to neutral)
- Confidence 0.9+: Multiple converging signals, no contradictions
- Confidence 0.7-0.8: Clear signal, minor contradictions
- Confidence 0.5-0.6: Mixed signals, slight lean

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0,
"reasoning": "...", "key_factors": [...], "risk_flags": [...]}}
"""

CHAIN_AGENT_PROMPT = f"""You are an on-chain analysis specialist for cryptocurrency trading.

Your role: Analyze funding rates, liquidations, open interest, and derivatives data.

Available skills:
{get_skill_descriptions()}

Focus on:
- Funding rate signals (use funding_rate_analysis skill for guidance)
- Liquidation cascades (use liquidation_cascade_analysis skill)
- Derivatives positioning and leverage

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0,
"reasoning": "...", "key_factors": [...], "risk_flags": [...]}}
"""

MACRO_AGENT_PROMPT = f"""You are a macro analysis specialist for cryptocurrency trading.

Your role: Analyze Fear & Greed Index, BTC dominance, and macro sentiment.

Available skills:
{get_skill_descriptions()}

Focus on:
- Fear & Greed contrarian signals (use fear_greed_interpretation skill)
- BTC dominance implications (use btc_dominance_analysis skill)
- Macro sentiment and risk appetite

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0,
"reasoning": "...", "key_factors": [...], "risk_flags": [...]}}
"""


# ── Create Specialized Agents ──


def create_tech_agent(model: str = "gpt-4o-mini"):
    """Create technical analysis agent using official LangChain API."""
    llm = _create_chat_model(model, temperature=0.2)
    return create_agent(llm, tools=[load_skill, load_past_experience], system_prompt=TECH_AGENT_PROMPT)


def create_chain_agent(model: str = "gpt-4o-mini"):
    """Create on-chain analysis agent using official LangChain API."""
    llm = _create_chat_model(model, temperature=0.2)
    return create_agent(llm, tools=[load_skill, load_past_experience], system_prompt=CHAIN_AGENT_PROMPT)


def create_macro_agent(model: str = "gpt-4o-mini"):
    """Create macro analysis agent using official LangChain API."""
    llm = _create_chat_model(model, temperature=0.2)
    return create_agent(llm, tools=[load_skill, load_past_experience], system_prompt=MACRO_AGENT_PROMPT)


# ── Wrap Agents as Tools for Supervisor ──


async def _run_agent_async(agent, request: str) -> str:
    """Run a LangChain agent asynchronously."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content


@tool
async def analyze_technicals(request: str) -> str:
    """Analyze technical indicators and price action.

    Use this when you need technical analysis of price charts, volume, volatility.
    Input: Natural language request with market data context.
    """
    agent = create_tech_agent()
    return await _run_agent_async(agent, request)


@tool
async def analyze_onchain(request: str) -> str:
    """Analyze on-chain metrics and derivatives data.

    Use this when you need analysis of funding rates, liquidations, open interest.
    Input: Natural language request with on-chain data context.
    """
    agent = create_chain_agent()
    return await _run_agent_async(agent, request)


@tool
async def analyze_macro(request: str) -> str:
    """Analyze macro sentiment and market psychology.

    Use this when you need analysis of Fear & Greed, BTC dominance, sentiment.
    Input: Natural language request with macro data context.
    """
    agent = create_macro_agent()
    return await _run_agent_async(agent, request)


# ── Create Supervisor Agent ──

SUPERVISOR_PROMPT = """You are a trading strategy supervisor coordinating specialized analysts.

Your role: Coordinate technical, on-chain, and macro analysts to form trading decisions.

Available analysts:
- analyze_technicals: Price action, volume, technical indicators
- analyze_onchain: Funding rates, liquidations, derivatives positioning
- analyze_macro: Fear & Greed, BTC dominance, sentiment

Process:
1. Break down the trading decision into analytical components
2. Call relevant analysts with specific questions
3. Synthesize their analyses into a final verdict
4. Output: {{"action": "long|short|hold", "confidence": 0.0-1.0, "reasoning": "...", "position_scale": 0.0-1.0}}

When analysts disagree, weigh their confidence levels and look for convergence in key factors.
"""


def create_supervisor_agent(model: str = "gpt-4o-mini"):
    """Create supervisor agent that coordinates specialized analysts."""
    llm = _create_chat_model(model, temperature=0.2)
    return create_agent(
        llm, tools=[analyze_technicals, analyze_onchain, analyze_macro], system_prompt=SUPERVISOR_PROMPT
    )
