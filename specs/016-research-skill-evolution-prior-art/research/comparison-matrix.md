# 跨项目对比矩阵

**状态**：Phase 1 部分完成 —— 仅"提示组装"和"记忆 ↔ 技能"两列已填。其余 6 列保留 `_Phase 2_` 占位。

每格格式：`<一句话发现>` + `path/to/file.ext`（指向源码引用）。N/A 格用 "N/A — <理由>"。

| 项目 | 进化算法 | 技能数据结构 | 检索机制 | **记忆 ↔ 技能** ✓ | **提示组装** ✓ | Evaluation | Agent ↔ Skill 边界 | 工程实现 |
|---|---|---|---|---|---|---|---|---|
| **SkillClaw** | _Phase 2_ | _Phase 2_ | _Phase 2_ | 因果链会话 DB 喂给技能演化；技能以 Markdown 文件本地+S3 存储；Task Ledger 跨会话保留目标 · `api_proxy/skill_router.py` | 两阶段：起初 system prompt 仅注入 YAML name+description 目录，命中时追加完整 SKILL.md；`push_min_injections=5` 质量门 · `api_proxy/proxy.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **MetaClaw** | _Phase 2_ | _Phase 2_ | _Phase 2_ | 6 类 MemoryUnit 含 importance + access_count；分组渲染到单一 system 消息 via `render_for_prompt()` · `memory/units.py` | 3 模式（Synergy / Memory-only / Skills-only）；20k token 贪心截断；分段 system 消息 · `api_server.py:build_prompt()` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **OpenClaw-RL** | _Phase 2_ | _Phase 2_ | _Phase 2_ | in-memory dict（`_session_conversations` / `_session_experience`）；无持久化；OEL 用 `loop.create_task()` 异步抽取经验 · `openclaw_rl/oel.py` | tokenizer-native via `tokenizer.apply_chat_template()` (Jinja2)；经验**仅注入最后一条 user 消息** via `_inject_experience_to_messages()` · `openclaw_rl/utils/template.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **Hermes** | _Phase 2_ | _Phase 2_ | _Phase 2_ | `hermes_state.py` 中 `SessionDB` 双用途：喂演化数据集 + 把记忆块注入固定 system prompt 段 · `agent/hermes_state.py` | 命名段（`MEMORY_GUIDANCE` / `SKILLS_GUIDANCE` / `SESSION_SEARCH_GUIDANCE`）；部分 DSPy 可演进、部分固定；SKILL.md 在 session 开始作 user msg 注入，**会话中绝不热替换**，单文件 ≤15KB · `agent/prompt_builder.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **EvoSkill** | _Phase 2_ | _Phase 2_ | _Phase 2_ | `feedback_history.md` 即记忆；**成功改进后清空**；`related_iterations` 字段反向追溯 · `evoskill/proposer.py` | Proposer LLM 接收 失败样本 + 历史 + 技能 的有序 prompt；输出 `prompt|skill` 二元路由后 Generator 写入 `.claude/skills/` · `evoskill/proposer.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **EvoSkills** | _Phase 2_ | _Phase 2_ | _Phase 2_ | 双存储：M_I（方向级）+ M_E（策略级）；top-k 余弦相似度注入当前 prompt · `memory/store.py` | 13 个 SKILL.md 含 `allowed_tools` 白名单 + `should-trigger` 声明式路由；SKILL.md 即 prompt body · `skills/*.md` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **skill-evolution** | _Phase 2_ | _Phase 2_ | _Phase 2_ | 无独立记忆模块；反思按 7 步流程把学到的写回 SKILL.md 或脚本；"确定性阶梯"原则划分 LLM vs 脚本职责 · `docs/protocol.md` | 三层：L1 元数据始终在 context、L2 SKILL.md 触发时载入、L3 `references/` 文件按需拉取 · `loader/progressive.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **microsoft/autogen** | N/A —— 纯编排框架，无技能演化概念 | N/A —— 工具静态注册，无技能抽象 | _Phase 2_ | `Memory.update_context()` 是官方注入钩子；对话历史走 `ChatCompletionContext`（变体：buffered / head-tail / token-trimmed）· `python/packages/autogen-core/src/autogen_core/memory/`、`.../model_context/` | **四槽位分离**：`_system_messages`（静态角色）+ `_tools`（schema）+ `Memory.update_context()` 注入 + `ChatCompletionContext` 历史；装配顺序 = SystemMessage → memory 注入 → 历史+本轮；system prompt 不进入 `ChatCompletionContext`（缓存友好）· `python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |

---

## 状态图例

- ✓ —— 该列 Phase 1 完成（提示组装 + 记忆 ↔ 技能）
- _Phase 2_ —— 推迟到 Phase 2 研究
- N/A — <理由> —— 该项目不具备此属性，附带原因

## 备注

- **autoresearch 替换**：autoresearch 按 D-PROC-01 退出活跃研究集（仅 2 / 8 列产生信号）；改为 **microsoft/autogen** Tier 2 深度研究。原 `autoresearch.md` 重命名为 `_deferred-autoresearch.md` 保留供参考。
- **autogen 贡献边界**：autogen 没有技能演化概念（纯编排框架），其价值集中在提示组装 + 记忆接线（spec 017 相关）。Phase 2 的进化算法研究将继续依赖 SkillClaw / EvoSkills / OpenClaw-RL / EvoSkill / skill-evolution。
- **License 覆盖**：8 个活跃项目全部 permissive（5 MIT + 3 Apache-2.0），spec 018 fork-borrow 建议无 GPL 污染风险。
- **Phase 1 范围确认**：2 列 × 8 项目 = 16 格已填。剩余 48 格（6 列 × 8 项目）待 Phase 2。
