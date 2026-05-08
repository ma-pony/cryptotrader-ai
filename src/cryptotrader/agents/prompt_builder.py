"""Agent prompt externalization — PromptBuilder + Provider 协议 + TokenBudgetEnforcer.

本模块实现 spec 017：把 4 个 analysis agent 的 ROLE 系统提示词从 Python 源码外置到
config/agents/<name>.md（YAML frontmatter + Markdown body），由 PromptBuilder 在运行时拼装。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ── Token 估算（CJK-aware，与 spec 014 算法一致）───────────────────────────────


def _estimate_tokens(text: str) -> int:
    """CJK-aware token 估算：ASCII÷4 + CJK÷1.5（误差 < 10% vs tiktoken）。

    复用 spec 014 的既有算法，本模块自带实现以避免循环依赖。
    如 cryptotrader.learning.context 后续落地同名函数，可替换此处 import。
    """
    ascii_chars = 0
    cjk_chars = 0
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs ranges
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
            cjk_chars += 1
        else:
            ascii_chars += 1
    return int(ascii_chars / 4) + int(cjk_chars / 1.5) + 1


# ── ConfigValidationError ───────────────────────────────────────────────────────


class ConfigValidationError(Exception):
    """启动期 config 校验失败时抛出，含失败文件路径与原因。"""

    def __init__(self, file_path: Path, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Config 校验失败 [{file_path}]: {reason}")


# ── AgentConfig dataclass ───────────────────────────────────────────────────────

# 5 个核心必填 section
_REQUIRED_SECTIONS = frozenset({"system_prompt", "user_tail", "available_skills", "recent_memory", "output_schema"})

# 默认 slot 分配
_DEFAULT_SYSTEM_SLOT = ["system_prompt", "available_skills", "output_schema"]
_DEFAULT_USER_SLOT = ["recent_memory", "snapshot", "portfolio", "agent_analyses", "user_tail"]


@dataclass
class AgentConfig:
    """单个 agent 配置的内存表示，由 ConfigLoader 从 config/agents/<agent_id>.md 解析。"""

    agent_id: str
    description: str
    sections: list[str]
    budget: int
    priority: dict[str, int]
    body_sections: dict[str, str]
    slot_overrides: dict[str, list[str]] = field(default_factory=dict)

    @property
    def system_slot(self) -> list[str]:
        """返回应进入 SystemMessage 的 section 名列表。"""
        return self.slot_overrides.get("system", _DEFAULT_SYSTEM_SLOT)

    @property
    def user_slot(self) -> list[str]:
        """返回应进入 HumanMessage 的 section 名列表。"""
        return self.slot_overrides.get("user_tail", _DEFAULT_USER_SLOT)


# ── ConfigLoader ────────────────────────────────────────────────────────────────


class ConfigLoader:
    """从 config/agents/<agent_id>.md 加载并校验 AgentConfig。

    满足 9 项校验规则（见 contracts/agent-config-schema.md）。
    """

    @staticmethod
    def load(path: Path) -> AgentConfig:
        """加载并校验 agent config 文件，失败抛 ConfigValidationError。"""
        # 规则 1：文件可读
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigValidationError(path, f"无法读取 config 文件: {e}") from e

        # frontmatter / body 切分
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not m:
            raise ConfigValidationError(path, "无法找到 YAML frontmatter（缺少 --- 分隔符）")
        fm_text, body_text = m.group(1), m.group(2)

        # 规则 2：YAML 可解析
        try:
            fm = yaml.safe_load(fm_text)
        except yaml.YAMLError as e:
            raise ConfigValidationError(path, f"YAML 解析失败: {e}") from e

        if not isinstance(fm, dict):
            raise ConfigValidationError(path, "YAML frontmatter 应为 dict 类型")

        # 规则 3：必填字段齐全
        for required_field in ("agent_id", "description", "sections", "budget", "priority"):
            if required_field not in fm:
                raise ConfigValidationError(path, f"缺少必填字段: {required_field!r}")

        agent_id: str = fm["agent_id"]
        description: str = fm["description"]
        sections: list[str] = fm["sections"]
        budget: int = fm["budget"]
        priority: dict[str, int] = fm["priority"]
        slot_overrides: dict[str, list[str]] = fm.get("slot_overrides", {}) or {}

        # 规则 4：agent_id 与文件名匹配
        expected_name = path.stem
        if agent_id != expected_name:
            raise ConfigValidationError(path, f"agent_id ({agent_id!r}) 与文件名 ({expected_name!r}) 不匹配")

        # 规则 5：budget > 0
        if not isinstance(budget, int) or budget <= 0:
            raise ConfigValidationError(path, f"budget 必须 > 0，当前值: {budget!r}")

        # 规则 6：sections 含 5 个核心必填项
        missing_required = _REQUIRED_SECTIONS - set(sections)
        if missing_required:
            raise ConfigValidationError(path, f"sections 缺少必需项: {sorted(missing_required)}")

        # 解析 body sections
        body_sections = ConfigLoader._parse_body_sections(body_text)

        # 规则 7：body 中 section 与 sections 声明一一对应
        for sec_name in sections:
            if sec_name not in body_sections:
                raise ConfigValidationError(path, f"section {sec_name!r} 在 body 中未找到")

        # 规则 8：priority 中每个 key 都在 sections 中（动态 section 例外）
        _dynamic_sections = {"snapshot", "portfolio", "agent_analyses"}
        for pkey in priority:
            if pkey not in sections and pkey not in _dynamic_sections:
                raise ConfigValidationError(path, f"priority 引用了未声明的 section: {pkey!r}")

        # 规则 9：slot_overrides 校验
        if slot_overrides:
            all_slot_sections: list[str] = []
            for slot_name, slot_secs in slot_overrides.items():
                for sec in slot_secs:
                    # 动态 section（snapshot/portfolio/agent_analyses）允许不在 sections 声明中
                    _dynamic = {"snapshot", "portfolio", "agent_analyses"}
                    if sec not in sections and sec not in _dynamic:
                        raise ConfigValidationError(
                            path, f"slot_overrides[{slot_name!r}] 引用了未声明的 section: {sec!r}"
                        )
                all_slot_sections.extend(slot_secs)
            # 检查 system / user_tail 是否有交集
            sys_secs = set(slot_overrides.get("system", []))
            usr_secs = set(slot_overrides.get("user_tail", []))
            overlap = sys_secs & usr_secs
            if overlap:
                raise ConfigValidationError(path, f"slot_overrides system 与 user_tail 有交集: {sorted(overlap)}")

        return AgentConfig(
            agent_id=agent_id,
            description=description,
            sections=list(sections),
            budget=budget,
            priority=dict(priority),
            body_sections=body_sections,
            slot_overrides=slot_overrides,
        )

    @staticmethod
    def _parse_body_sections(body_text: str) -> dict[str, str]:
        """把 Markdown body 按 '## section_name' 标题切分为 dict。"""
        sections: dict[str, str] = {}
        current_name: str | None = None
        current_lines: list[str] = []
        for line in body_text.splitlines():
            if line.startswith("## "):
                if current_name is not None:
                    sections[current_name] = "\n".join(current_lines).strip()
                current_name = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_name is not None:
            sections[current_name] = "\n".join(current_lines).strip()
        return sections


# ── Skill dataclass ─────────────────────────────────────────────────────────────


@dataclass
class Skill:
    """单个技能的数据载体（沿用 spec 014 schema）。"""

    skill_id: str
    description: str
    tags: list[str]
    steps: list[str]
    body: str = ""


# ── MemoryProvider Protocol ─────────────────────────────────────────────────────


class MemoryProvider(Protocol):
    """记忆数据源协议接口；本 spec 提供 DefaultMemoryProvider，spec 018 提供进化版实现。"""

    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str:
        """返回已格式化的 markdown 记忆文本；空记忆返回固定占位"暂无历史记忆"。"""
        ...


# ── SkillProvider Protocol ──────────────────────────────────────────────────────


class SkillProvider(Protocol):
    """技能数据源协议接口；本 spec 提供 DefaultSkillProvider，spec 018 提供进化版实现。"""

    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]:
        """返回 ranked skill 列表，长度 ≤ k；空返回 []。"""
        ...


# ── DefaultMemoryProvider ───────────────────────────────────────────────────────


class DefaultMemoryProvider:
    """默认 MemoryProvider 实现，复用 spec 014 agent_memory/<agent_id>/{patterns.md, cases.jsonl}。"""

    def __init__(
        self,
        memory_root: Path | None = None,
        top_k_patterns: int = 5,
        top_k_cases: int = 3,
    ) -> None:
        self._root = memory_root or Path("agent_memory")
        self._k_patterns = top_k_patterns
        self._k_cases = top_k_cases

    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str:
        """读取 patterns.md + cases.jsonl 返回格式化 markdown。"""
        agent_dir = self._root / agent_id
        if not agent_dir.exists():
            return "暂无历史记忆"

        patterns = self._read_patterns(agent_dir / "patterns.md")
        cases = self._read_cases(agent_dir / "cases.jsonl")

        if not patterns and not cases:
            return "暂无历史记忆"

        parts = []
        if patterns:
            parts.append("### Patterns\n" + "\n".join(f"- {p}" for p in patterns[: self._k_patterns]))
        if cases:
            parts.append("### Cases\n" + "\n".join(f"- {c}" for c in cases[: self._k_cases]))
        return "\n\n".join(parts)

    def _read_patterns(self, path: Path) -> list[str]:
        """每行 '- ' 开头视作一条 pattern。"""
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln[2:].strip() for ln in lines if ln.startswith("- ")]

    def _read_cases(self, path: Path) -> list[str]:
        """逐行解析 JSONL，解析失败行跳过 + warning。"""
        if not path.exists():
            return []
        cases: list[str] = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                logger.warning("cases.jsonl 第 %d 行解析失败: %s", len(cases) + 1, e)
                continue
            cases.append(self._format_case(obj))
        return cases

    def _format_case(self, obj: dict) -> str:
        cid = obj.get("case_id", "?")
        ctx = obj.get("context", "")
        outcome = obj.get("outcome", "")
        pnl = obj.get("pnl", "")
        return f"[{cid}] {ctx} → {outcome} (PnL: {pnl})"


# ── DefaultSkillProvider ────────────────────────────────────────────────────────


class DefaultSkillProvider:
    """默认 SkillProvider 实现，扫描 agent_skills/<id>/SKILL.md，按 agent_id tag 过滤。"""

    def __init__(self, skills_root: Path | None = None) -> None:
        self._root = skills_root or Path("agent_skills")
        self._cache: list[Skill] | None = None

    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]:
        """返回 agent_id 在 tags 中的 top-k skills（本 spec 简单 keyword match）。"""
        skills = self._load_all()
        relevant = [s for s in skills if agent_id in s.tags]
        return relevant[:k]

    def _load_all(self) -> list[Skill]:
        """延迟加载并缓存所有 SKILL.md，失败单条跳过 + warning。"""
        if self._cache is not None:
            return self._cache
        if not self._root.exists():
            self._cache = []
            return []
        skills: list[Skill] = []
        for skill_dir in sorted(self._root.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            try:
                skill = self._parse_skill_md(md_path)
                skills.append(skill)
            except Exception as e:
                logger.warning("解析 %s 失败: %s", md_path, e)
        self._cache = skills
        return skills

    def _parse_skill_md(self, path: Path) -> Skill:
        """解析 SKILL.md：YAML frontmatter（必填：skill_id / description / tags / steps）+ body。"""
        content = path.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not m:
            raise ValueError(f"SKILL.md 缺少 YAML frontmatter: {path}")
        fm_text, body = m.group(1), m.group(2)
        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict):
            raise ValueError(f"SKILL.md frontmatter 应为 dict: {path}")
        for fld in ("skill_id", "description", "tags", "steps"):
            if fld not in fm:
                raise ValueError(f"SKILL.md 缺少必填字段 {fld!r}: {path}")
        return Skill(
            skill_id=str(fm["skill_id"]),
            description=str(fm["description"]),
            tags=list(fm["tags"]),
            steps=list(fm["steps"]),
            body=body.strip(),
        )


# ── EnforceResult + TokenBudgetEnforcer ────────────────────────────────────────


@dataclass
class EnforceResult:
    """TokenBudgetEnforcer 输出的 dataclass（spec 017 FR-X12）。"""

    final_sections: dict[str, str]
    dropped_sections: list[str]
    degraded_sections: list[str]
    prompt_size_pre: int
    prompt_size_post: int
    budget: int


_PROTECTED_SECTIONS = frozenset({"system_prompt", "output_schema"})


class TokenBudgetEnforcer:
    """按优先级丢/降 section 至 token 预算内（spec 017 FR-X11）。

    优先级数字越大越先丢；system_prompt / output_schema 强制保留。
    """

    def enforce(
        self,
        sections: dict[str, str],
        budget: int,
        priority: dict[str, int],
        protected: frozenset[str] = _PROTECTED_SECTIONS,
    ) -> EnforceResult:
        """执行 token 预算检查与丢/降流程，返回 EnforceResult。"""
        # 深拷贝 sections，避免修改传入 dict
        working = dict(sections)
        prompt_size_pre = sum(_estimate_tokens(v) for v in working.values())

        dropped: list[str] = []
        degraded: list[str] = []

        if prompt_size_pre <= budget:
            return EnforceResult(
                final_sections=working,
                dropped_sections=dropped,
                degraded_sections=degraded,
                prompt_size_pre=prompt_size_pre,
                prompt_size_post=prompt_size_pre,
                budget=budget,
            )

        # 按优先级从大到小（数字大先丢），跳过 protected
        sorted_keys = sorted(
            working.keys(),
            key=lambda k: priority.get(k, 999),
            reverse=True,
        )
        for name in sorted_keys:
            if name in protected:
                continue
            if sum(_estimate_tokens(v) for v in working.values()) <= budget:
                break
            working.pop(name)
            dropped.append(name)

        # 仍超 budget → 截断 recent_memory / available_skills
        if sum(_estimate_tokens(v) for v in working.values()) > budget:
            for name in ["recent_memory", "available_skills"]:
                if name in working:
                    target_chars = int(budget * 0.3 * 4)  # 粗估：1 token ≈ 4 ASCII char
                    if len(working[name]) > target_chars:
                        working[name] = working[name][:target_chars] + "\n...(截断)"
                        degraded.append(name)

        prompt_size_post = sum(_estimate_tokens(v) for v in working.values())
        return EnforceResult(
            final_sections=working,
            dropped_sections=dropped,
            degraded_sections=degraded,
            prompt_size_pre=prompt_size_pre,
            prompt_size_post=prompt_size_post,
            budget=budget,
        )


# ── PromptBuilder ───────────────────────────────────────────────────────────────


class PromptBuilder:
    """运行时 prompt 组装器；每个 agent 持有独立实例（spec 017 FR-X5/X6）。

    构造时加载并校验 config/agents/<agent_id>.md；
    build() 是唯一对外方法，返回 (SystemMessage, HumanMessage) 供 LLM 调用。
    """

    def __init__(
        self,
        agent_id: str,
        config_dir: Path,
        memory_provider: MemoryProvider,
        skill_provider: SkillProvider,
        model: str = "",
    ) -> None:
        self._agent_id = agent_id
        self._memory_provider = memory_provider
        self._skill_provider = skill_provider
        self._model = model
        self._enforcer = TokenBudgetEnforcer()

        config_path = config_dir / f"{agent_id}.md"
        self.config: AgentConfig = ConfigLoader.load(config_path)

    def build(
        self,
        snapshot: dict,
        portfolio: dict,
        agent_analyses: dict | None = None,
    ) -> tuple[SystemMessage, HumanMessage]:
        """组装 LLM messages — 唯一对外入口（spec 017 FR-X6）。

        返回 (SystemMessage, HumanMessage) tuple 供 agent 直接传给 LLM。
        telemetry 8 字段挂当前 active span 或 structured log（FR-X18/X19）。
        """
        t0 = time.monotonic()

        # 1. 获取记忆（异常 → 占位）
        try:
            recent_memory = self._memory_provider.get_recent_memory(self._agent_id, snapshot)
        except Exception:
            logger.warning("MemoryProvider 异常，降级为占位", exc_info=True)
            recent_memory = "暂无历史记忆"

        # 2. 获取 skills（异常 → 空列表）
        try:
            skills = self._skill_provider.get_available_skills(self._agent_id, snapshot)
        except Exception:
            logger.warning("SkillProvider 异常，降级为空列表", exc_info=True)
            skills = []

        # 3. 渲染 skills → markdown
        available_skills_text = self._render_skills(skills)

        # 4. 渲染 snapshot / portfolio / agent_analyses → str
        snapshot_text = self._render_snapshot(snapshot)
        portfolio_text = self._render_portfolio(portfolio)
        agent_analyses_text = self._render_agent_analyses(agent_analyses)

        # 5. 组装 sections dict（运行时注入覆盖 config body 中的占位内容）
        sections: dict[str, str] = {}
        for sec_name, sec_body in self.config.body_sections.items():
            sections[sec_name] = sec_body
        # 运行时 Provider 注入（覆盖 config 中的占位段）
        sections["available_skills"] = available_skills_text
        sections["recent_memory"] = recent_memory
        # 动态 section（不在 body，直接注入）
        sections["snapshot"] = snapshot_text
        sections["portfolio"] = portfolio_text
        if agent_analyses_text:
            sections["agent_analyses"] = agent_analyses_text

        # 6. Token budget 检查
        result = self._enforcer.enforce(sections, self.config.budget, self.config.priority)

        # 7. 按 slot 分配组装 SystemMessage + HumanMessage
        sys_msg, usr_msg = self._assemble_messages(result.final_sections)

        # 8. Telemetry（8 字段）
        duration_ms = (time.monotonic() - t0) * 1000
        self._emit_telemetry(result, duration_ms)

        return sys_msg, usr_msg

    def _render_skills(self, skills: list[Skill]) -> str:
        """把 list[Skill] 渲染为 markdown bullet list。"""
        if not skills:
            return "暂无可用技能"
        lines = []
        for sk in skills:
            lines.append(f"- **{sk.skill_id}**: {sk.description}")
            for step in sk.steps:
                lines.append(f"  - {step}")
        return "\n".join(lines)

    def _render_snapshot(self, snapshot: dict) -> str:
        """把 snapshot dict 渲染为 key: value 文本；缺失字段走占位。"""
        if not snapshot:
            return "<missing>"
        lines = []
        for k, v in snapshot.items():
            val = v if v is not None else "<missing>"
            lines.append(f"{k}: {val}")
        return "\n".join(lines)

    def _render_portfolio(self, portfolio: dict) -> str:
        """把 portfolio dict 渲染为文本；缺失走占位。"""
        if not portfolio:
            return "<missing>"
        lines = []
        for k, v in portfolio.items():
            val = v if v is not None else "<missing>"
            lines.append(f"{k}: {val}")
        return "\n".join(lines)

    def _render_agent_analyses(self, agent_analyses: dict | None) -> str:
        """把其他 agent 的分析结果渲染为文本（仅 verdict-style 用）。"""
        if not agent_analyses:
            return ""
        lines = []
        for agent_id, analysis in agent_analyses.items():
            if isinstance(analysis, dict):
                direction = analysis.get("direction", "?")
                confidence = analysis.get("confidence", "?")
                reasoning = analysis.get("reasoning", "")
                lines.append(f"- {agent_id}: {direction} (confidence={confidence}) — {reasoning}")
            else:
                lines.append(f"- {agent_id}: {analysis}")
        return "\n".join(lines)

    def _assemble_messages(self, final_sections: dict[str, str]) -> tuple[SystemMessage, HumanMessage]:
        """按 slot_overrides 或默认分配组装 SystemMessage + HumanMessage。"""
        sys_parts = []
        for sec in self.config.system_slot:
            if final_sections.get(sec):
                sys_parts.append(final_sections[sec])

        usr_parts = []
        for sec in self.config.user_slot:
            if final_sections.get(sec):
                usr_parts.append(final_sections[sec])

        sys_content = "\n\n".join(sys_parts)
        usr_content = "\n\n".join(usr_parts)

        return SystemMessage(content=sys_content), HumanMessage(content=usr_content)

    def _emit_telemetry(self, result: EnforceResult, duration_ms: float) -> None:
        """写入 8 个 telemetry 字段到当前 active OpenTelemetry span 或 structured log。"""
        attrs = {
            "prompt.builder.agent_id": self._agent_id,
            "prompt.builder.sections_included": list(result.final_sections.keys()),
            "prompt.builder.dropped_sections": result.dropped_sections,
            "prompt.builder.degraded_sections": result.degraded_sections,
            "prompt.builder.prompt_size_pre": result.prompt_size_pre,
            "prompt.builder.prompt_size_post": result.prompt_size_post,
            "prompt.builder.budget": result.budget,
            "prompt.builder.duration_ms": round(duration_ms, 2),
        }

        # 尝试挂到 OpenTelemetry active span（spec 010 基础设施）
        span_attached = False
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is not None and span.is_recording():
                for key, val in attrs.items():
                    if isinstance(val, list):
                        span.set_attribute(key, str(val))
                    else:
                        span.set_attribute(key, val)
                span_attached = True
        except Exception:
            pass  # opentelemetry 未安装或无 active span → fallback 到 log

        if not span_attached:
            logger.info(
                "prompt_builder_telemetry agent_id=%s sections_included=%s "
                "dropped=%s degraded=%s size_pre=%d size_post=%d budget=%d duration_ms=%.2f",
                self._agent_id,
                result.final_sections.keys(),
                result.dropped_sections,
                result.degraded_sections,
                result.prompt_size_pre,
                result.prompt_size_post,
                result.budget,
                duration_ms,
            )
