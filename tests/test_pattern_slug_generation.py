"""tests/test_pattern_slug_generation.py — spec 021 T007

5 用例覆盖 _make_pattern_slug 规则：
  1. empty input → "unnamed"
  2. non-alnum chars → 替换为 -
  3. truncate at 60 chars
  4. collision -2 / -3 后缀
  5. non-ascii only input → "unnamed"
"""

import pytest

from cryptotrader.learning.memory import _make_pattern_slug


def test_slug_empty_input(tmp_path):
    """空字符串 → base="unnamed"，无冲突时直接返回。"""
    slug = _make_pattern_slug("", tmp_path)
    assert slug == "unnamed"


def test_slug_non_alnum_chars(tmp_path):
    """非 alnum 字符替换为 -，前后 - 去除。"""
    slug = _make_pattern_slug("Volume Spike + RSI Overbought!", tmp_path)
    assert slug == "volume-spike-rsi-overbought"
    # 验证无前导/尾随 -
    assert not slug.startswith("-")
    assert not slug.endswith("-")


def test_slug_truncate_60_chars(tmp_path):
    """超过 60 字符时截断到 60。"""
    long_text = "a" * 80
    slug = _make_pattern_slug(long_text, tmp_path)
    assert len(slug) <= 60


def test_slug_collision_uses_n_suffix(tmp_path):
    """已存在同名文件时加 -2 后缀；再次冲突加 -3。"""
    # 先创建 base.md
    (tmp_path / "volume-spike.md").write_text("existing")
    slug2 = _make_pattern_slug("Volume Spike", tmp_path)
    assert slug2 == "volume-spike-2"

    # 再创建 -2.md
    (tmp_path / "volume-spike-2.md").write_text("existing2")
    slug3 = _make_pattern_slug("Volume Spike", tmp_path)
    assert slug3 == "volume-spike-3"


def test_slug_non_ascii_only_input(tmp_path):
    """纯非 ASCII 输入（如中文）→ base 为空 → "unnamed"。"""
    slug = _make_pattern_slug("量化信号", tmp_path)
    assert slug == "unnamed"
