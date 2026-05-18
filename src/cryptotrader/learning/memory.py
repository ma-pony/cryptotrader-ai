"""PatternRecord 数据模型与文件加载 helper。

agent_memory/<agent>/patterns/<name>.md 格式（spec 014 既有；spec 024 冷启动补完）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 支持的 maturity 状态（spec 014 FSM）
VALID_MATURITY = frozenset({"observed", "probationary", "active", "deprecated", "archived"})
VALID_AGENTS = frozenset({"tech", "chain", "news", "macro"})


@dataclass
class PnLTrack:
    """Pattern 的 PnL 追踪记录（spec 014 既有）。"""

    pnls: list[float] = field(default_factory=list)


@dataclass
class PatternRecord:
    """单条 pattern 蒸馏数据（agent_memory/<agent>/patterns/<name>.md frontmatter）。

    spec 014 既有 entity；本模块仅提供读取 helper，不修改 schema。
    """

    name: str
    agent: str
    description: str = ""
    maturity: str = "observed"
    regime_tags: list[str] = field(default_factory=list)
    pnl_track: PnLTrack = field(default_factory=PnLTrack)
    source_cycles: list[str] = field(default_factory=list)
    body: str = ""
    version: int = 1
    manually_edited: bool = False
    created: str | None = None


def _load_pattern_record(path: Path) -> PatternRecord | None:
    """从单个 pattern .md 文件加载 PatternRecord。

    文件格式：YAML frontmatter + markdown body。
    解析失败返回 None（logger.warning），不抛异常。
    跳过 .gitkeep / .lock 文件（调用方负责过滤，此处也做一次兜底）。
    """
    if path.suffix not in {".md"} or path.name in {".gitkeep", ".lock"}:
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("无法读取 pattern 文件 %s: %s", path, exc)
        return None

    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter

        fm, body = parse_frontmatter(content, path=path)
    except Exception as exc:
        logger.warning("pattern frontmatter 解析失败 %s: %s", path, exc)
        return None

    # 必填字段校验
    name = fm.get("name")
    agent = fm.get("agent")
    if not name or not agent:
        logger.warning("pattern 缺少必填字段 name/agent: %s", path)
        return None

    maturity = str(fm.get("maturity", "observed"))

    # pnl_track 兼容两种格式：
    #   新格式（spec 024）: {pnls: [1.2, -0.5, ...]}
    #   旧格式（memory_old_format）: {cases: N, wins: N, win_rate: 0.8, avg_pnl: 45.0, ...}
    raw_pnl = fm.get("pnl_track") or {}
    if isinstance(raw_pnl, dict):
        raw_pnls = raw_pnl.get("pnls")
        if isinstance(raw_pnls, list):
            pnl_track = PnLTrack(pnls=[float(v) for v in raw_pnls if v is not None])
        else:
            # 旧格式：没有 pnls 列表，用空 list（仅保留结构）
            pnl_track = PnLTrack(pnls=[])
    else:
        pnl_track = PnLTrack(pnls=[])

    regime_tags: list[str] = []
    raw_tags = fm.get("regime_tags")
    if isinstance(raw_tags, list):
        regime_tags = [str(t) for t in raw_tags]

    source_cycles: list[str] = []
    raw_cycles = fm.get("source_cycles")
    if isinstance(raw_cycles, list):
        source_cycles = [str(c) for c in raw_cycles]

    return PatternRecord(
        name=str(name),
        agent=str(agent),
        description=str(fm.get("description", "")),
        maturity=maturity,
        regime_tags=regime_tags,
        pnl_track=pnl_track,
        source_cycles=source_cycles,
        body=body.strip(),
        version=int(fm.get("version", 1)),
        manually_edited=bool(fm.get("manually_edited", False)),
        created=str(fm.get("created")) if fm.get("created") else None,
    )
