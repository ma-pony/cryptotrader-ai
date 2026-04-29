"""Tests for API context_builder — multimodal message construction."""

from __future__ import annotations

from cryptotrader.config import AppConfig, ChartAnalysisConfig


def _cfg(**overrides) -> AppConfig:
    """Build AppConfig with optional chart_analysis overrides."""
    chart = ChartAnalysisConfig(**overrides)
    cfg = AppConfig()
    cfg.chart_analysis = chart
    return cfg


def test_vision_model_includes_image():
    """When model supports vision and dataUrl is valid, image block is included."""
    from api.context_builder import build_multimodal_messages

    payloads = [{"timeframe": "1h", "description": "Bullish", "dataUrl": "data:image/png;base64,abc"}]
    msgs, degraded = build_multimodal_messages(payloads, "gpt-4o", _cfg())
    assert not degraded
    content = msgs[0].content
    types = [b["type"] for b in content if isinstance(b, dict)]
    assert "image_url" in types
    assert "text" in types


def test_non_vision_model_no_image():
    """When model is not in vision_models, image is not included."""
    from api.context_builder import build_multimodal_messages

    payloads = [{"timeframe": "1h", "description": "Bullish", "dataUrl": "data:image/png;base64,abc"}]
    msgs, degraded = build_multimodal_messages(payloads, "deepseek-chat", _cfg())
    assert not degraded
    content = msgs[0].content
    types = [b["type"] for b in content if isinstance(b, dict)]
    assert "image_url" not in types
    assert "text" in types


def test_image_too_large_degrades():
    """When dataUrl exceeds max_image_bytes, image is dropped and degraded=True."""
    from api.context_builder import build_multimodal_messages

    large_url = "data:image/png;base64," + "A" * 5_000_000
    payloads = [{"timeframe": "1h", "description": "Test", "dataUrl": large_url}]
    msgs, degraded = build_multimodal_messages(payloads, "gpt-4o", _cfg())
    assert degraded
    content = msgs[0].content
    types = [b["type"] for b in content if isinstance(b, dict)]
    assert "image_url" not in types


def test_multiple_timeframes_separator():
    """Multiple payloads produce timeframe separator markers."""
    from api.context_builder import build_multimodal_messages

    payloads = [
        {"timeframe": "15m", "description": "Short term"},
        {"timeframe": "4h", "description": "Long term"},
    ]
    msgs, degraded = build_multimodal_messages(payloads, "deepseek-chat", _cfg())
    assert not degraded
    content = msgs[0].content
    texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert any("15m" in t for t in texts)
    assert any("4h" in t for t in texts)


def test_null_dataurl_no_error():
    """When dataUrl is None, no error occurs."""
    from api.context_builder import build_multimodal_messages

    payloads = [{"timeframe": "1h", "description": "Test", "dataUrl": None}]
    msgs, degraded = build_multimodal_messages(payloads, "gpt-4o", _cfg())
    assert not degraded
    content = msgs[0].content
    types = [b["type"] for b in content if isinstance(b, dict)]
    assert "image_url" not in types
    assert "text" in types


def test_two_payloads_distinct_timeframe_sections():
    """Two payloads produce two distinct timeframe sections with clear labels."""
    from api.context_builder import build_multimodal_messages

    payloads = [
        {"timeframe": "15m", "description": "Short-term RSI=65", "dataUrl": "data:image/png;base64,abc"},
        {"timeframe": "4h", "description": "Long-term SMA crossover", "dataUrl": None},
    ]
    msgs, degraded = build_multimodal_messages(payloads, "gpt-4o", _cfg())
    assert not degraded

    content = msgs[0].content
    texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
    full_text = "\n".join(texts)

    assert "=== 时间周期: 15m ===" in full_text
    assert "=== 时间周期: 4h ===" in full_text
    assert "Short-term RSI=65" in full_text
    assert "Long-term SMA crossover" in full_text

    types = [b["type"] for b in content if isinstance(b, dict)]
    assert "image_url" in types
