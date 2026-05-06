"""数据模型 — Skill / PatternRecord / CaseRecord 等核心实体。

与 data-model.md 一一对齐，作为整个 014-agent-skills-protocol-migration 的数据基础。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# ── Maturity 枚举 ──

Maturity = Literal["observed", "probationary", "active", "deprecated"]


# ── PnL 追踪 ──


@dataclass
class PnLTrack:
    """单条 pattern 的 PnL 追踪数据。"""

    cases: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    last_active: str = ""  # ISO date string

    def update(self, pnl: float) -> None:
        """增量更新：新增一条 PnL 数据。"""
        self.cases += 1
        if pnl > 0:
            self.wins += 1
        self.win_rate = self.wins / self.cases if self.cases > 0 else 0.0
        # 移动平均
        prev_total = self.avg_pnl * (self.cases - 1)
        self.avg_pnl = (prev_total + pnl) / self.cases


# ── 核心实体 ──


@dataclass
class Skill:
    """高层能力包，对应 agent_skills/<name>/SKILL.md。

    initial 5 个，运行时可增长；遵循 Anthropic Skills 协议。
    """

    name: str
    description: str
    scope: str  # "shared" | "agent:<agent_id>"
    body: str
    file_path: Path = field(default_factory=lambda: Path("."))
    manually_edited: bool = False
    version: str = "1.0"
    mtime: float = 0.0  # 磁盘 mtime，用于缓存失效

    @property
    def agent_id(self) -> str | None:
        """从 scope 中提取 agent_id；shared → None。"""
        if self.scope.startswith("agent:"):
            return self.scope.split(":", 1)[1]
        return None

    @property
    def is_shared(self) -> bool:
        return self.scope == "shared"


@dataclass
class PatternRecord:
    """memory 层数据，对应 agent_memory/<agent>/patterns/<name>.md。"""

    name: str
    agent: str  # tech / chain / news / macro
    description: str
    body: str
    regime_tags: list[str] = field(default_factory=list)
    pnl_track: PnLTrack = field(default_factory=PnLTrack)
    maturity: Maturity = "observed"
    source_cycles: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=lambda: datetime.now(UTC))
    file_path: Path = field(default_factory=lambda: Path("."))
    manually_edited: bool = False
    version: int = 1


@dataclass
class CaseRecord:
    """memory 层数据，对应 agent_memory/cases/<cycle_id>.md（per-cycle 单文件）。"""

    cycle_id: str
    timestamp: datetime
    pair: str
    snapshot_summary: dict = field(default_factory=dict)
    agent_analyses: dict[str, str] = field(default_factory=dict)  # agent_id -> analysis text
    verdict_action: str = "hold"  # long / short / hold / close
    verdict_reasoning: str = ""
    applied_patterns: list[str] = field(default_factory=list)  # "<agent>::<pattern>" 格式
    risk_gate_passed: bool = True
    execution_status: dict | None = None
    final_pnl: float | None = None
    file_path: Path = field(default_factory=lambda: Path("."))


@dataclass
class AgentSkillSet:
    """单 agent 一次 cycle 加载的 skill 集合（动态 list[Skill]）。"""

    agent_id: str
    skills: list[Skill] = field(default_factory=list)

    def get_combined_body(self) -> str:
        """拼接所有 skill body，供 middleware 注入 system prompt。"""
        parts = []
        for skill in self.skills:
            parts.append(f"\n\n## Skill: {skill.name}\n\n{skill.body}")
        return "".join(parts)


@dataclass
class ReflectionRun:
    """一次 reflection 任务的结构化日志。"""

    cases_processed: int = 0
    patterns_created: int = 0
    patterns_updated: int = 0
    patterns_archived: int = 0
    failed_agents: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class CurationRun:
    """一次 SKILL.md 整理任务的日志。"""

    skill_name: str = ""
    patterns_included: int = 0
    draft_path: str = ""
    used_llm: bool = False
    skipped: bool = False
    skip_reason: str = ""
