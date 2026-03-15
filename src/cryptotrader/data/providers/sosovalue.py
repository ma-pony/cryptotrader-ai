"""SoSoValue API provider — ETF flow data and crypto news."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

ETF_BASE = "https://api.sosovalue.xyz/openapi/v2/etf"
NEWS_BASE = "https://openapi.sosovalue.com/api/v1/news"


async def fetch_etf_metrics(api_key: str, etf_type: str = "us-btc-spot") -> dict:
    """Fetch current ETF data metrics (daily net inflow, AUM, holdings, etc.)."""
    if not api_key:
        logger.warning("SoSoValue API key not set, skipping ETF metrics")
        return {}
    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:  # nosec S501 — third-party data source SoSoValue uses self-signed cert, confirmed no sensitive data in transit
            r = await c.post(
                f"{ETF_BASE}/currentEtfDataMetrics",
                headers={"x-soso-api-key": api_key, "Content-Type": "application/json"},
                json={"type": etf_type},
            )
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                logger.warning("SoSoValue ETF metrics error: %s", body.get("msg"))
                return {}
            data = body.get("data", {})
            # Extract key aggregate metrics
            result = {}
            for field in (
                "totalNetAssets",
                "dailyNetInflow",
                "cumNetInflow",
                "dailyTotalValueTraded",
                "totalTokenHoldings",
            ):
                obj = data.get(field)
                if obj and obj.get("value") is not None:
                    result[field] = float(obj["value"])
            # Top ETF flows
            etf_list = data.get("list", [])
            top_flows = []
            for etf in etf_list[:5]:
                inflow_obj = etf.get("dailyNetInflow", {})
                if inflow_obj and inflow_obj.get("value") is not None:
                    top_flows.append(
                        {
                            "ticker": etf.get("ticker", ""),
                            "dailyNetInflow": float(inflow_obj["value"]),
                        }
                    )
            if top_flows:
                result["topEtfFlows"] = top_flows
            return result
    except Exception:
        logger.warning("SoSoValue ETF metrics fetch failed", exc_info=True)
        return {}


async def fetch_etf_history(api_key: str, etf_type: str = "us-btc-spot") -> list[dict]:
    """Fetch historical ETF inflow chart (up to 300 days).

    Returns list of {date, totalNetInflow, totalValueTraded, totalNetAssets, cumNetInflow}.
    """
    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:  # nosec S501 — third-party data source SoSoValue uses self-signed cert, confirmed no sensitive data in transit
            r = await c.post(
                f"{ETF_BASE}/historicalInflowChart",
                headers={"x-soso-api-key": api_key, "Content-Type": "application/json"},
                json={"type": etf_type},
            )
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                return []
            data = body.get("data", [])
            return data if isinstance(data, list) else data.get("list", [])
    except Exception:
        logger.warning("SoSoValue ETF history fetch failed", exc_info=True)
        return []


async def fetch_news(api_key: str, page_size: int = 20) -> list[dict]:
    """Fetch featured crypto news from SoSoValue."""
    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=15, verify=False) as c:  # nosec S501 — third-party data source SoSoValue uses self-signed cert, confirmed no sensitive data in transit
            r = await c.get(
                f"{NEWS_BASE}/featured",
                headers={"x-soso-api-key": api_key},
                params={"pageNum": 1, "pageSize": page_size},
            )
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                return []
            items = body.get("data", {}).get("list", [])
            return [
                {
                    "title": _extract_title(item),
                    "category": item.get("category", ""),
                    "releaseTime": item.get("releaseTime", 0),
                    "author": item.get("author", ""),
                }
                for item in items
            ]
    except Exception:
        logger.warning("SoSoValue news fetch failed", exc_info=True)
        return []


def _extract_title(item: dict) -> str:
    """Extract title from multilanguage content."""
    ml = item.get("multilanguageContent", [])
    if isinstance(ml, list):
        for entry in ml:
            if isinstance(entry, dict) and entry.get("title"):
                return entry["title"]
    elif isinstance(ml, dict):
        for lang in ("en", "zh-CN", "zh-TW"):
            content = ml.get(lang, {})
            if content.get("title"):
                return content["title"]
    return ""
