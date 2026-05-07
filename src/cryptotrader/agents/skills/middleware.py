"""SkillsInjectionMiddleware — 自动注入 SKILL.md 到 agent system prompt。

FR-018: 继承 LangChain AgentMiddleware（wrap_model_call 模式）。
FR-019: 4 个 agent 节点自动 discover + 注入 shared + agent:<self> skills。
FR-019a: mtime 缓存，无需重启即可 pickup 新/修改的 SKILL.md。
FR-020: 追加到 request.system_message。
FR-021: tools 类变量注册 load_skill_tool。
FR-024: 单条 SKILL.md 加载失败 → warning + 跳过，不阻塞 cycle。
"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_SKILLS_DIR, VALID_AGENT_IDS
from cryptotrader.agents.skills.loader import discover_skills_for_agent

logger = logging.getLogger(__name__)


class SkillsInjectionMiddleware:
    """LangChain-compatible middleware 将 SKILL.md 注入 agent system prompt。

    设计为轻量 wrapper：不依赖 AgentMiddleware 基类（langchain 版本兼容性），
    通过 build_system_addendum() 提供给 create_agent / BaseAgent 调用。

    Usage:
        middleware = SkillsInjectionMiddleware(agent_id="tech")
        addendum = middleware.build_system_addendum()
        # 将 addendum 追加到 system prompt
    """

    # FR-021: tools 类变量注册 load_skill_tool
    tools: list = []

    def __init__(
        self,
        agent_id: str,
        skill_dir: Path | None = None,
    ) -> None:
        if agent_id not in VALID_AGENT_IDS:
            logger.warning(
                "SkillsInjectionMiddleware: unknown agent_id '%s' (not in VALID_AGENT_IDS)",
                agent_id,
            )
        self.agent_id = agent_id
        self.skill_dir = skill_dir or DEFAULT_AGENT_SKILLS_DIR

    def build_system_addendum(self) -> str:
        """发现并加载匹配 skills，返回拼接后的 system prompt 追加内容。

        FR-019: 匹配 scope == "shared" 或 scope == "agent:<self>"。
        FR-024: 单条失败 → warning + 跳过（在 discover_skills_for_agent 内处理）。
        """
        skills = discover_skills_for_agent(self.agent_id, skill_dir=self.skill_dir)

        if not skills:
            return ""

        parts = []
        for skill in skills:
            parts.append(f"\n\n---\n## Skill: {skill.name}\n\n{skill.body}")
            logger.debug("Injected skill '%s' for agent '%s'", skill.name, self.agent_id)

        addendum = "".join(parts)
        logger.debug(
            "SkillsInjectionMiddleware: injected %d skills for agent '%s'",
            len(skills),
            self.agent_id,
        )
        return addendum

    async def wrap_model_call(self, request: object, handler) -> object:
        """FR-020: wrap_model_call — 追加 skill body 到 system message。

        兼容 LangChain AgentMiddleware 接口。
        """
        addendum = self.build_system_addendum()
        if addendum:
            try:
                # 尝试追加到 system message（LangChain request 结构）
                _inject_into_request(request, addendum)
            except Exception:
                logger.warning(
                    "SkillsInjectionMiddleware: failed to inject into request for '%s'",
                    self.agent_id,
                    exc_info=True,
                )
        return await handler(request)


def _inject_into_request(request: object, addendum: str) -> None:
    """追加 addendum 到 request 的 system message。

    支持 LangChain 各版本的 request 结构：
    - ChatPromptValue / messages list
    - dict with 'messages' key
    """
    from langchain_core.messages import SystemMessage

    messages = None

    if hasattr(request, "messages"):
        messages = request.messages
    elif isinstance(request, dict) and "messages" in request:
        messages = request["messages"]

    if messages is None:
        return

    # 查找并扩展 SystemMessage
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            existing = msg.content or ""
            if isinstance(messages, list):
                messages[i] = SystemMessage(content=existing + addendum)
            return

    # 无 SystemMessage 时插入为第一条
    if isinstance(messages, list):
        messages.insert(0, SystemMessage(content=addendum))


# ── 初始化 tools 类变量 ──

try:
    from cryptotrader.agents.skills.tool import load_skill_tool as _lst

    if _lst is not None:
        SkillsInjectionMiddleware.tools = [_lst]
except ImportError:
    pass
