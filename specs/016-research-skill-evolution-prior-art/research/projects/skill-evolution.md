---
name: skill-evolution
url: https://github.com/hao-cyber/skill-evolution
license: MIT
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# skill-evolution — hao-cyber

## 架构概览

skill-evolution 是一个面向 AI 编程 agent（主要是 Claude Code）的元框架，让 agent 的技能（skill）能够自主完成完整生命周期：创建 → 反思修复 → 评测 → 成熟度判断 → 发布 → 检索安装 → Fork/合并 → 卸载。

**关键约束**：
- 无 UI、无需人工介入：设计者是 AI，使用者也是 AI
- 零依赖：所有脚本仅使用 Python 标准库
- 离线优先：核心的创建/反思能力不依赖网络；公共注册表作为可选扩展层
- 后端存储：Supabase（PostgreSQL + 全文搜索）；anon key 无法直接写库，全部通过安全 RPC

**技术栈**：Python 81.7%，PLpgSQL 18.3%，143 stars，18 forks，首发 2026-02-25

**目录结构**：
```
skill-name/
├── SKILL.md          # YAML frontmatter + Markdown 指令（必须）
├── scripts/          # 确定性执行代码（可选）
├── references/       # 按需加载的深度文档（可选）
└── assets/           # 输出用素材，不注入上下文（可选）
```

---

## 提示组装（Phase 1）

### 渐进式三层加载（Progressive Disclosure）

这是 skill-evolution 最核心的提示组装机制，目标是将上下文开销压缩到接近零：

| 层级 | 内容 | 加载时机 | 大小预算 |
|------|------|----------|----------|
| L1 元数据 | `name` + `description`（YAML frontmatter） | 始终在上下文 | ~100 词 |
| L2 SKILL.md 正文 | 路由规则 + 共享约束 + 参考索引 | skill 触发后立即加载 | ≤150 行（≤300 行可接受） |
| L3 references/ | 特定场景的完整工作流、查找表、边缘案例 | agent 用 Read tool 按需拉取 | 无硬性上限，分文件隔离 |

**关键设计思路**：安装 50 个 skill 和安装 1 个，L1 的上下文开销完全相同——直到某个 skill 真正被触发，L2 才进入上下文；如果该触发场景需要深度文档，才进一步加载 L3。

### SKILL.md 结构规范

```yaml
---
name: my-skill
description: "同时描述功能和触发条件。所有 when-to-use 信息放这里，不放 body。"
depends_on:
  - web-read
---
```

Body 规范：
- 用祈使语气/不定式
- 只写 Claude 不已知的内容，避免重复通用知识
- 最多 150 行（路由层）；超出必须拆到 `references/`
- **拆分原则**：按使用场景拆，而非按内容类型拆

### 路由层模式（SKILL.md body）

```markdown
## Routing
- 创建新技能 → 读取 references/structure.md
- 执行失败反思 → 读取 references/reflect-mode.md
- 成熟度判断 → 读取 references/maturity.md
- 发布到注册表 → 读取 references/publish.md
```

每个 reference 文件只有在匹配场景时才被 Read 进上下文，彻底消除"一次加载全部"的上下文浪费。

### 评测集成（evals.yaml）

每个 skill 目录下可放 `evals.yaml`，与 promptfoo test case 格式兼容。SKILL.md 正文作为 system prompt 注入，`user_input` 变量包裹用户请求。测试断言分两类：
- 确定性断言：`contains`、`not-contains`、`regex`
- 语义断言：`llm-rubric`（配置 threshold，如 0.7）

评测运行时机：修改 SKILL.md 路由/触发规则后、用户主动要求回归测试时、reflect 修复后验证。5-15 个测试用例覆盖正面+负面触发。

---

## 记忆 ↔ 技能（Phase 1 lite）

### 反思循环（Reflect Loop）作为隐式记忆写入

skill-evolution 没有独立的"记忆模块"，而是将执行后的反思直接写入 SKILL.md 或 scripts/，形成持久化的隐式记忆：

**触发信号**：
- 执行失败（错误/错误输出/意外行为）
- 用户纠正行为（"不对，应该这样…"）
- 用户提供绕过 skill 限制的变通方法
- 临时修复了 skill 脚本/提示
- skill 应触发但未触发（静默漏触发）

**反思七步流程**：
1. 定位根因（指令缺失/脚本 bug/文档过时/触发描述太窄或太宽）
2. 重新读取相关 SKILL.md 和脚本（不依赖记忆）
3. 影响扫描（grep 共享概念，检查跨 skill 联动）
4. 确定修复层级（"确定性阶梯"）
5. 提出具体 diff 级别的变更方案
6. 获得用户明确批准
7. 应用修改，commit，push

**确定性阶梯（Determinism Ladder）**是关键设计：
```
能内置到 scripts/ 自动执行？→ 改脚本，删掉 SKILL.md 的手动指令
必须在运行前/后检查？     → 用 hooks 确定性拦截
需要 LLM 判断力？         → 才写 SKILL.md 指令
```
反模式：SKILL.md 写了"必须做 X"，但 X 是确定性操作——应内置到脚本。

**升级策略**：
- 同一问题连续 2 次 reflect → 停止打补丁，重新审视根本方法
- 3+ 次跨不同问题的 reflect → 可能需要重新设计 skill 本身

### 成熟度信号（发布门控）

成熟度判断是 skill 自主进化中的关键状态机节点，所有条件满足时 agent 主动建议发布：

1. **production-tested**：成功执行 ≥3 次（非 `--help` 测试）
2. **stable**：过去 3 天（或最近 5 次成功执行）无 reflect 修复
3. **well-structured**：frontmatter 有效，SKILL.md ≤300 行
4. **clean**：publish.py 预览无硬编码路径/泄露密钥警告
5. **self-contained**：脚本 `--help` 均返回 0，`depends_on` 已声明

### 模型特定记忆路由

当根因是模型缺陷而非 skill bug 时，修复路由到 `model_guides/{family}.md`：
- 先查 `MODEL` 环境变量识别当前模型（如 `glm-4.7`、`kimi-k2.5`）
- 问：同样问题在 Claude 上会出现吗？是 → skill bug；否 → 模型缺陷
- 追加经验到 `model_guides/{family}.md`（按模型族：glm、kimi、minimax、volc）

### 变体系统（Variant System）

```
web-scraper (base)
├── web-scraper@alice   ← 加了代理轮换
├── web-scraper@bob     ← 加了并发执行
└── web-scraper@merged  ← agent 合并最优版
```

发布同名 skill 时自动 Fork 为变体，无语义版本号。Agent 选择策略：审计通过 > 描述匹配 > 安装量 > 评分。合并时 agent 处理语义合并，`merge.py` 处理管道工作。

---

## Phase 2 占位符

（待 Phase 2 深度研究填充）

---

## 借鉴建议（仅 Phase 1）

1. **渐进式三层加载**：CryptoTrader 的 experience memory 目前一次性注入，可借鉴 L1/L2/L3 分层——元数据始终在上下文，具体经验案例按相关性按需拉取，与现有 GSSC pipeline 兼容。

2. **确定性阶梯原则**：在 spec 016 中区分"应写入 experience rules（LLM 判断）"和"应写入脚本/hooks（确定性执行）"，避免 experience memory 承担不适合它的角色。

3. **成熟度门控机制**：可类比设计 experience rule 的"发布门控"——rule 需经过 N 次成功验证且稳定期无修改才提升成熟度等级（对应现有 `maturity` 字段），形成完整状态机。

4. **反思触发信号**：参考其触发信号列表，为 CryptoTrader 的反思入口增加"静默漏触发"检测——当某个 pattern 应被 verbal reinforcement 命中但实际未命中时，触发 reflect 写入新经验。

5. **evals.yaml 对照**：可为 verbal_reinforcement 的 experience injection 建立类似的回归测试集（正面触发 + 负面触发），用 `llm-rubric` 断言注入内容的相关性质量。

---

## 备注 / 待解问题

- **默认注册表托管**：README 声称内置公共注册表，但 `.env.example` 包含 `SUPABASE_URL`/`SUPABASE_ANON_KEY`，实际"内置"可能是硬编码了作者自己的 Supabase 实例——需 Phase 2 确认 `scripts/lib/` 中注册表 URL 的实际来源
- **scripts/lib/ 目录**：Phase 1 未深入读取，可能包含共享的上下文注入辅助函数，留 Phase 2 精读
- **prompt-eval skill 依赖**：eval-mode 依赖独立安装的 `prompt-eval` skill，该 skill 的具体实现未研究
- **模型特定记忆**：`model_guides/` 目录在根目录文件列表中未出现，可能是约定的用户侧目录，非仓库内置
- **hooks 集成深度**：YAML frontmatter 支持声明 `hooks`（PreToolUse），但 settings.json 层面的 hook 配置与 skill frontmatter hook 的优先级关系未明
