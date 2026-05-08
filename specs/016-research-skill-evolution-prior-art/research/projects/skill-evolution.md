---
name: skill-evolution
url: https://github.com/hao-cyber/skill-evolution
license: MIT
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: true
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

## Phase 2：进化算法——7 步反思流程精解

### 触发信号（何时进入 reflect 模式）

以下任一条件满足时，agent 自动进入 reflect 模式（来源：`references/reflect-mode.md`）：

| 触发信号 | 典型场景 |
|---------|---------|
| 执行失败 | 错误输出、意外行为、脚本非零退出 |
| 用户纠正 | "不对，应该这样…" 式反馈 |
| 用户提供绕过方法 | 用户手动实现了 skill 应该做的事 |
| 临时修复 | agent 在执行中对脚本/提示做了即兴修补 |
| 绕过了另一个 skill 的守卫 | 如未经审查就编辑了 CLAUDE.md |
| 静默漏触发 | skill 本应触发但未触发，用户手动处理——触发描述过窄 |

**关键设计**：第六条"静默漏触发"是大多数框架缺失的。EvoSkill 的 Proposer-Evaluator 流水线只处理已触发技能的修复，而 skill-evolution 的 reflect 还捕获"本该触发但没触发"的未命中场景，直接拓宽了触发描述。

### 7 步流程（含精确输入/输出/约束）

**Step 1 — Identify（根因定位）**

- 输入：失败上下文、错误信息、用户反馈
- 任务：将根因归类到以下四类之一
  1. SKILL.md 指令缺失或歧义
  2. 脚本 bug 或边缘情况未覆盖
  3. references/ 文档过时或缺失
  4. 触发描述太窄或太宽
  5. 绕过了另一个 skill 的守卫（跨 skill 影响）
- 输出：带标签的根因描述

**Step 2 — Read（重新读取，不依赖记忆）**

- 输入：Step 1 确定的根因所在文件路径
- 约束：**必须**重新读取相关 SKILL.md 和脚本，不能依赖上下文中残留的记忆
- 输出：确认 gap 确实存在的原文引用

**Step 3 — Impact Scan（跨 skill 影响扫描）**

这一步是 skill-evolution 相对于其他框架的独特贡献——横向检查修改波及范围：

```bash
# 提取要修改的关键词
grep -rl "<关键词>" .claude/skills/
grep -l "<关键词>" .claude/CLAUDE.md
```

- 输入：要修改的关键词/概念列表
- 输出：引用了该概念的所有文件列表 + 是否需联动修改的判断
- 约束：在 Step 5 的提案中必须列出扫描结果（"影响 N 个文件，其中 M 个需联动修改"）

**Step 4 — Determinism Ladder（确定性阶梯判定）**

这是 skill-evolution 最核心的设计决策点，三级判断树：

```
能内置到 scripts/ 自动执行？
  YES → 改脚本，同时删掉 SKILL.md 中对应的手动指令
  NO ↓
必须在运行前/后检查（可确定性拦截）？
  YES → 用 hooks（PreToolUse / PostToolUse）
  NO ↓
需要 LLM 判断力（意图理解、权衡取舍）？
  YES → 才写 SKILL.md 指令
```

**反模式**：SKILL.md 写了"必须做 X"，但 X 是确定性操作（如检查文件是否存在、格式化输出）——应内置到脚本，否则 LLM 可能"忘记"执行。

**Step 5 — Propose（具体提案，diff 级别）**

- 输出必须包含：
  1. 要改哪个文件（SKILL.md / scripts/ / references/ / model_guides/）
  2. 增/删/改的具体内容（可以是 diff 格式）
  3. 为什么这样改能防止问题复现
  4. Impact scan 结果摘要（N 个文件受影响，M 个需联动）

**Step 6 — Confirm（明确用户批准）**

- 约束：**硬性门控**——未获用户明确批准前不得编辑任何文件
- 这是与 EvoSkill 完全自动化策略的关键区别：skill-evolution 保留人类在修改循环中的控制权

**Step 7 — Apply（应用并 commit/push）**

- 操作：make the change, commit, push
- 触发后置检查：**立即检查成熟度信号**（自动进入 Maturity Check 流程）

### 升级策略（Escalation）

reflect 不是无限迭代的，两条硬性升级规则：

1. **同一问题连续 2 次 reflect** → 停止打补丁，用新视角重新审视 skill 根本设计
2. **不同问题 3+ 次 reflect** → 标记为结构性问题，向用户报告："This skill may need a redesign, not another patch."

这两条规则防止 reflect 陷入局部最优补丁循环——与 EvoSkill 的 Proposer-Evaluator 相比，后者缺乏此类"元收敛"判断。

### 模型特定路由（非 skill bug 的特殊分支）

当 Step 1 发现根因是**模型能力缺陷**而非 skill 设计问题时，修复路由到 `model_guides/{family}.md`：

1. 读取 `MODEL` 环境变量（如 `glm-4.7`、`kimi-k2.5`、`claude-sonnet-4-6`）
2. 判断：同样问题在 Claude 上会出现吗？
   - 是 → skill bug，走正常 reflect 路径
   - 否 → 模型缺陷，路由到 `model_guides/`
3. 确定模型族：`glm`、`kimi`、`minimax`、`volc`
4. 追加经验到 `model_guides/{family}.md`（文件不存在则创建）

常见模型缺陷类型：幻觉 API、遗漏 `await`、缩进错误、发明不存在的文件路径、错误的 import 名称。

**注意**：`model_guides/` 是约定的用户侧目录，不在仓库内置——仓库只提供约定，用户在自己项目里维护。

---

## Phase 2：技能数据结构——三层 Schema 精确定义

### 完整 SKILL.md frontmatter schema

来源：`references/structure.md` + `setup.sql` 的 skills 表定义

```yaml
---
# 必填字段
name: my-skill           # 格式：^[a-z0-9][a-z0-9\-]{0,62}$（DB 约束）
description: "..."       # 最大 1000 字符（DB 约束）；同时描述功能和触发条件

# 可选字段
depends_on:              # 安装时自动递归安装依赖
  - web-read
  - llm-gateway

hooks:                   # 强制执行守卫（PreToolUse / PostToolUse）
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/skills/my-skill/scripts/validate.sh"
---
```

### L1 / L2 / L3 各层字段约束汇总

| 层 | 载体 | 大小限制 | 始终在上下文？ | 关键约束 |
|----|------|---------|--------------|---------|
| L1 | frontmatter `name` + `description` | ~100 词 | 是 | description ≤ 1000 chars；是主要触发机制 |
| L2 | SKILL.md body（frontmatter 之后） | 理想 ≤150 行；可接受 ≤300 行；>300 行必须拆分 | skill 触发后加载 | 只写路由逻辑 + 跨场景共享规则 |
| L3 | references/ 各文件 | 无硬性上限；文件隔离 | 按需 Read | 按使用场景拆分，非按内容类型拆分 |

### 数据库层 schema（Supabase PostgreSQL）

`setup.sql` 中 `skills` 表的关键字段：

```sql
create table skills (
  id          uuid primary key,
  name        text   check (name ~ '^[a-z0-9][a-z0-9\-]{0,62}$'),
  variant     text   default 'base',   -- 变体名，格式同 name
  parent_id   uuid   references skills(id),  -- 指向 base 变体
  description text   check (length(description) <= 1000),
  tags        text[] default '{}',     -- 最多 15 个
  author      text   not null,
  skill_md    text   not null,         -- SKILL.md 全文
  file_tree   jsonb  default '{}',     -- {rel_path: content}，≤500KB（jsonb text repr）
  requires_env     text[] default '{}',  -- 运行时必须存在的环境变量
  requires_tools   text[] default '{}',
  requires_runtime text[] default '{}',  -- uv / node 等
  depends_on       text[] default '{}',  -- 依赖的其他 skill 名称列表
  installs    int    default 0,
  forks       int    default 0,
  audited_at  timestamptz default null,  -- null=未审计；非null=已通过审计
  unique(name, variant)
);
```

`file_tree` 的 JSONB 结构是 flat dict：键为相对路径（如 `"scripts/run.py"`、`"references/reflect-mode.md"`），值为文件文本内容（二进制文件存 `"[binary file, N bytes]"` 占位符）。总体积上限约 500 KB（以 jsonb 文本表示计算）。

### file_tree 尺寸约束与文件数限制

来自 `scripts/audit.py` 的审计规则（非仅限 DB 约束）：

- `file_tree` 总文件内容 ≤ 500,000 字节（`FAIL` 级别）
- 文件总数 ≤ 50 个（`FAIL` 级别）
- SKILL.md 行数 ≤ 500 行（`WARN` 级别，建议 ≤300）
- description ≤ 1000 字符（`FAIL` 级别）

### 依赖声明与自动安装

`depends_on` 字段在 `scripts/install.py` 中实现递归安装，并用 `_visited` set 防止循环依赖：

```python
def install_skill(name, variant, skills_dir, force=False, _visited=None):
    dep_key = f"{name}@{variant}"
    if dep_key in _visited:
        return {"status": "skipped", "reason": "circular dependency"}
    _visited.add(dep_key)
    # ... 安装逻辑 ...
    # 递归安装 depends_on
```

---

## Phase 2：检索机制——L1→L2→L3 递进触发规则

### 触发递进的完整规则

**L1 → L2 触发**：由 `description` 字段驱动。Claude Code 在每次对话开始时将所有已安装 skill 的 `name + description` 注入上下文（L1 层）。当用户请求语义上匹配某个 description 时，agent 加载对应 SKILL.md 正文（L2 层）。

`description` 字段的设计要点：
- 同时描述**功能**和**触发条件**（"when to use"信息全部放这里，不放 body）
- 是唯一的触发机制——body 里不再重复触发条件
- 示例格式：`"Skill 全生命周期管理：创建 → 反思优化 → ... 触发场景：(1) ... (2) ..."`

**L2 → L3 触发**：由 SKILL.md body 中的**路由表**驱动。L2 的核心职责是路由，而非包含完整知识：

```markdown
## Routing
- **Creating/structuring a skill** → 读取 `references/structure.md`
- **Reflecting after skill failure** → 读取 `references/reflect-mode.md`
- **评测 skill prompt** → 读取 `references/eval-mode.md`
- **Checking skill maturity** → 读取 `references/maturity.md`
- **Publishing a skill** → 读取 `references/publish.md`
- **Searching/installing** → 读取 `references/search.md`
- **Merging skill variants** → 读取 `references/merge.md`
```

agent 读取路由表后，用 Read tool 按需加载对应 reference 文件——每次调用加载 ~1/7 的知识，而非全部。

### 注册表检索机制（search.py）

公共注册表通过 Supabase PostgreSQL 的全文搜索实现：

```sql
-- FTS index（自动维护）
alter table skills add column fts tsvector
  generated always as (
    to_tsvector('english', name || ' ' || description || ' ' || array_to_string(tags, ' '))
  ) stored;
create index skills_fts_idx on skills using gin(fts);
```

`search.py` 的核心调用路径：

```bash
# 全文搜索
python3 scripts/search.py --query "web scraper" --limit 10

# 按标签过滤
python3 scripts/search.py --tag csv --sort installs

# 查看某 skill 所有变体
python3 scripts/search.py --detail web-scraper

# 列出全部（含未审计）
python3 scripts/search.py --list-all --include-unaudited
```

### 变体选择决策树（Agent 执行）

安装时多变体存在时，agent 按以下顺序评估（来源：`references/search.md`）：

1. **审计状态过滤**：优先已审计变体（`audited_at IS NOT NULL`）；若仅有未审计变体，向用户警告风险后让其决定
2. **description 匹配**：读取每个变体的描述和优化重点，选最匹配当前任务上下文的
3. **安装量打分**：同等情况下选安装量更高的（社区验证）
4. **评分打分**：调用 `review.py stats` 比较各变体平均分
5. **互补合并建议**：若变体 A 优化 X、变体 B 优化 Y，当前任务需要 X+Y，建议 merge（进入 merge 流程）

agent 必须向用户解释选择理由（不沉默选择）。

### 懒加载的实现机制（零基础设施）

L1 层的注入由 Claude Code 原生 skill 发现机制处理——扫描 `.claude/skills/*/SKILL.md`，提取 frontmatter 的 `name` 和 `description`，自动注入上下文。这是 Claude Code 平台能力，非 skill-evolution 自行实现。

L2、L3 的懒加载完全依赖 agent 的 Read tool 主动调用——SKILL.md 路由表指示 agent "当场景 X 时读取 references/Y.md"，agent 自主执行 Read。无需任何运行时基础设施，纯文本约定驱动。

---

## Phase 2：评估与成熟度——5 条信号精确定义

### 5 条必要成熟度信号（全部满足才建议发布）

来源：`references/maturity.md`，经 Phase 2 原文核实：

**信号 1：production-tested（生产验证）**
- 判定：skill 在**真实任务**中成功执行 ≥ 3 次
- 排除：`--help` 冒烟测试、预演测试不计入
- 数据来源：agent 自身的执行记录（无自动追踪，由 agent 在上下文中计数）

**信号 2：stable（稳定期）**
- 判定：最近 **3 天**（或最近 5 次成功执行，取先满足者）**无 reflect 修复**
- 设计意图：时间维度（3 天）防止快速连续测试绕过稳定期；执行次数维度（5 次）防止低频使用 skill 永远等不到 3 天

**信号 3：well-structured（结构良好）**
- 判定：frontmatter 中 `name` 和 `description` 字段均有效，且 SKILL.md **≤300 行**
- 这是唯一的纯静态约束（不依赖执行历史）

**信号 4：clean（干净，无安全警告）**
- 判定：运行 `publish.py` 预览模式无硬编码路径、无泄露密钥警告
- 自动检查规则（来自 `scripts/publish.py` 的 `sanitize_check()`）：
  - 正则匹配 `(?:sk|api|token|key|secret|password)[-_]?\w*\s*[:=]\s*["\'][A-Za-z0-9_\-]{20,}` → possible API key
  - 匹配 `/home/\w+/` → hardcoded home path
  - 排除 `SUPABASE`、`TASKPOOL`、`$HOME`、`${HOME}` 等已知误报

**信号 5：self-contained（自包含）**
- 判定：若有脚本，所有脚本 `--help` 均退出 0；若有依赖，`depends_on` 在 frontmatter 中已声明
- 设计意图：防止发布后其他用户安装时因隐式依赖缺失而失败

### 可选加分信号（强化发布理由）

- 有通过测试的 `evals.yaml`（需独立安装 `prompt-eval` skill）
- 被多个用户使用（非仅作者自测）
- 同行评价平均分 ≥ 4.0

### 建议发布的触发时机

每次成功执行后或每次 reflect 修复（Step 7）后，agent 自动检查 5 条信号。全部满足且 skill 尚未发布时，呈现用户选择：

```
这个 skill 已经稳定运行了，要发布到社区市场吗？

[OPTIONS]
A: 发布到市场
B: 再观察一段时间
[/OPTIONS]
```

同时附：执行次数、距上次修复天数、文件数、任何警告。

### 不建议发布的情形

- skill 含组织内部逻辑、内部 API key、专有工作流
- skill 只是对单个 API 的薄封装
- 用户在本次会话中已拒绝过发布建议（不重复询问）

### 发布门控的硬性约束（Publish Gate）

`references/publish.md` 中的硬性规则：**Step 6（用户确认）是不可绕过的门控**。

具体实现：`publish.py` 默认以预览模式运行，只有传入 `--yes` 标志才真正上传。agent 被明确指示"不得在用户明确确认前传 `--yes`"——这是代码级约束（默认行为）加上 prompt 级约束（明确禁止）的双重保障。

### 失败回滚机制

skill-evolution 没有自动回滚——修改通过 git commit 实现持久化，回滚即 `git revert`。reflect 流程本身通过"2 次连续反思同一问题 → 停止打补丁"规则阻止错误修复的叠加，而非事后回滚。

注册表层面：更新已发布 skill 会清空 `audited_at`（`audited_at = null`），需重新过审计。这意味着 CI 中止审计 = 实际上阻止"带安全问题的更新"进入已审计列表，间接实现了发布级别的质量门控。

---

## Phase 2：Agent ↔ Skill 边界

### skill-evolution 作为元框架的定位

skill-evolution 是一个**元 skill**（meta-skill）：它不执行具体业务任务，而是管理其他所有 skill 的生命周期。其 `SKILL.md` 的 `description` 枚举了所有触发场景：

```
Skill 全生命周期管理：创建 → 反思优化 → 评测 → 成熟度判断 → 发布到市场
→ 检索多版本 → 选择/安装 → 融合迭代 → 卸载。
触发场景：
(1) 用户要求创建/修改 skill
(2) 发现可提取为 skill 的重复模式
(3) skill 执行出错或用户纠正后需要反思改进
(4) 用户要求发布/搜索/安装/合并社区 skill
(5) 反思后自动检查成熟度并建议发布
```

**架构关系**：

```
Agent (Claude Code)
  └── skill-dev (SKILL.md) — 元框架，管理 skill 生命周期
        ├── create → 创建其他 skill 目录
        ├── reflect → 修改其他 skill 的文件
        ├── publish → 调用 scripts/publish.py
        ├── search/install → 调用 scripts/install.py
        └── merge → 调用 scripts/merge.py
              └── Supabase (PostgreSQL) — 注册表存储层
```

所有复杂决策（选哪个变体、如何合并、质量评估）由 agent 做；基础设施只管存和查。

### 技能所有权模型

- **本地所有权**：`.claude/skills/<name>/` 目录物理上属于安装它的项目——agent 可自由修改、删除
- **注册表所有权**：`publishers` 表绑定 `author ↔ api_key`（UUID），只有匹配的 publisher key 才能更新同名变体
- **变体所有权**：同名 skill 不同 author → 自动 fork 为新变体（`variant = author_name`）；原作者保留 `base` 变体控制权
- **fork 策略**：发布已存在名称时自动触发，记录 `parent_id` 指向原始变体，`forks` 计数递增（10 分钟去重）

### 共享模型（跨项目共享）

skill-evolution 的共享通过**公共注册表**（Supabase）实现：

1. 作者在本地开发 → `publish.py` 序列化 `file_tree`（完整目录树）上传
2. 其他用户 `install.py` → 从注册表下载 `file_tree`，重建本地目录
3. 变体合并 → `merge.py prepare` 下载两个变体，agent 语义合并，`merge.py publish` 上传 merged 变体

无需 git submodule 或包管理器——注册表是独立的分发层，技能以**文件树快照**形式存储和分发。

### 何时不应创建 skill（边界判断）

以下情形下不应创建 skill（来源：`SKILL.md` body 的 "When NOT to Create a Skill"）：

| 条件 | 替代方案 |
|------|---------|
| 仅用一次 | 内联执行，无需抽象 |
| 一行 CLAUDE.md 规则即可覆盖 | 直接编辑 CLAUDE.md |
| 无可复用脚本且无非显而易见的知识 | Claude 本身已知道，无需 skill |
| 现有 skill 覆盖 80%+ 场景 | 扩展现有 skill，不新建 |

---

## Phase 2：工程实现细节

### 7 步反思的实现方式（纯 prompt 驱动）

skill-evolution 的 reflect 流程是**纯 prompt 驱动**，无 hook script，无 Python 调度器。`references/reflect-mode.md` 就是全部实现——agent 在上下文中读取这个文档并遵循其中的指令执行 7 步。

这是一个重要的架构选择：相比 EvoSkill 需要 Proposer agent + Evaluator agent + 调度器基础设施，skill-evolution 的 reflect 只需 agent 能读取文件（Read tool）和写文件（Edit/Write tool）。零基础设施，完全离线可用。

### 三层文件懒加载的实现机制

```
Claude Code SDK
  → 启动时扫描 .claude/skills/*/SKILL.md
  → 提取 frontmatter name + description → 注入上下文（L1，平台原生能力）

用户请求 → description 匹配 → agent 主动 Read SKILL.md 正文（L2）
  → 路由表指向 references/X.md → agent 主动 Read references/X.md（L3）
```

L1 注入是 Claude Code 平台行为；L2/L3 是 agent 的主动 Read tool 调用，由 SKILL.md 中的路由表指令驱动。无需任何中间件或运行时框架。

### 注册表客户端实现（scripts/lib/）

来源：`scripts/lib/supabase.py`（4,450 字节）和 `scripts/lib/__init__.py`（1,897 字节）

**关键设计**：
- 公共注册表 URL 和 anon key **硬编码**在 `scripts/lib/supabase.py` 中，用户无需任何配置即可使用：
  ```python
  _DEFAULT_URL = "https://ptwosnmrcfwmfnluufww.supabase.co"
  _DEFAULT_ANON_KEY = "sb_publishable_<...REDACTED-FOR-SECRETS-SCAN...>"  # pragma: allowlist secret
  ```
  这解答了 Phase 1 的待解问题：作者托管了自己的 Supabase 实例，并将 anon key 公开在代码中——这是有意设计（anon key 设计为可公开，通过 RLS 和 security-definer RPC 保护写入权限）
- `_load_dotenv()` 在 `lib/__init__.py` 导入时自动执行，从 cwd 向上遍历查找 `.env`——私有注册表通过环境变量 `SUPABASE_URL` + `SUPABASE_ANON_KEY` 覆盖默认值
- Publisher key 存储在 `.publisher_key` 文件中（skill 目录下），并通过 `PUBLISHER_KEY` 环境变量备份

**SSL 处理**：`supabase.py` 实现了三级 CA 解析策略（系统默认 → certifi → 跳过验证 + 警告），专门解决 macOS 上 Python OpenSSL 找不到系统 CA 的问题。

### 安全审计机制（scripts/audit.py）

audit.py（7,649 字节）由管理员通过 service_role key 定期运行，扫描规则分三类：

**FAIL 级别（直接拒绝）**：
- `eval()`、`exec()`、`compile()` 调用
- `os.system()`、`subprocess` with `shell=True`、`os.popen()`
- `../../` 路径遍历
- 硬编码密钥/Bearer token（正则匹配 ≥20 字符随机串）
- file_tree 超过 500,000 字节
- 文件数超过 50 个

**WARN 级别（通过但记录）**：
- f-string URL 构造（检查上下文中是否包含已知 API 域名，若是则豁免）
- 硬编码 `/home/username/` 路径

**通过/失败标准**：有任何 `FAIL` → `audited_at = null`（清除审计状态）；仅 `WARN` → 审计通过（`audited_at = now()`）

更新已通过审计的 skill 时，`publish_skill` RPC 会强制 `audited_at = null`，触发重新审计流程。

### 发布流程的实现（scripts/publish.py）

`publish.py`（11,459 字节）的完整流程：

1. `collect_file_tree()` — 递归遍历 skill 目录，读取所有文件为 flat dict；跳过 `__pycache__`、`.pyc`
2. `sanitize_check()` — 正则扫描 API key 和硬编码路径
3. `parse_frontmatter()` — 轻量级 YAML 解析器（无外部依赖，支持 `key: value` 和 `key: [list]` 两种格式）
4. `extract_tags()` — 从 name 和 description 自动提取标签
5. `extract_requires()` — 扫描 scripts/ 中的 `os.environ`/`os.getenv` 提取 requires_env；识别 shebang 推断 requires_runtime
6. `ensure_publisher_key()` — 自动注册发布者（首次）或验证已有 key
7. 调用 `publish_skill` RPC — server-side 验证 + upsert

Upsert 逻辑（在 PostgreSQL server 端）：
- 同名 + 同变体 + 同作者 → **更新**，`audited_at` 清空
- 同名 + 同变体 + 不同作者 → **报错**（不允许覆盖他人变体）
- 新名字或新变体 → **新建**

---

## Phase 2 借鉴建议（完整版）

### 对 spec 018（cryptotrader-ai 技能进化）的直接借鉴点

**1. 确定性阶梯在 ExperienceRule 写入中的应用**

当 reflect 触发时，判断修复应写入何处：
- 能被 `verbal_reinforcement()` 规则引擎自动处理？→ 写 `experience_rule`
- 需要在每次交易决策前确定性检查？→ 写 risk gate 规则（非 LLM 路径）
- 需要 agent 的意图理解和权衡？→ 才写 experience memory 的 `strategic_insights`

当前 `learning/reflect.py` 直接写入所有经验到同一 `experience_json`，未区分"确定性可执行规则"和"需 LLM 判断的策略洞见"——引入确定性阶梯可降低 experience injection 的 token 消耗。

**2. 5 条成熟度信号 → experience rule 的 maturity FSM 改造**

现有 `ExperienceRule.maturity`（字段已存在）缺乏精确的状态机触发条件。可借鉴 skill-evolution 的五信号模型：

| skill-evolution 信号 | cryptotrader-ai 对应条件 |
|---------------------|------------------------|
| production-tested ≥3 | rule 在真实交易中命中并产生正向 PnL ≥3 次 |
| stable（5 次无修复） | 最近 5 个包含该 pattern 的 cycle 无 reflect 覆盖此 rule |
| well-structured | rule 字段完整（conditions / action / rate 均非空），description ≤200 字 |
| clean | rule 不含硬编码 symbol 或时间戳（避免过拟合单次事件） |
| self-contained | rule 的 `conditions.regime_tags` 已声明，`depends_on` pattern 已存在于 history |

现有 maturity 字段是 float（0.0~1.0），可改为五状态 FSM：`draft → tested → stable → clean → mature`，发布门控在 `mature` 状态。

**3. 静默漏触发检测**

skill-evolution 的"skill 应触发但未触发"信号对 cryptotrader-ai 的 verbal reinforcement 尤为重要：当某个 experience rule 的 `conditions` 理论上应在当前 snapshot 上命中，但 `search_by_regime()` 返回空集时，应触发一次轻量 reflect——检查 pattern 描述是否过窄。

**4. Impact scan 的跨 skill 检查 → 跨规则影响检查**

当修改一个 experience rule 时，skill-evolution 的影响扫描思路可转化为：
```python
# 检查修改此 rule 是否影响依赖相同 regime_tags 的其他 rules
affected_rules = [r for r in all_rules if set(r.regime_tags) & set(modified_rule.regime_tags)]
```
防止修改一条规则导致相关规则的 rate 计算偏差（现有 `_verify_rules` 已按 regime 过滤，可扩展）。

**5. 注册表模型 → experience memory 共享**

skill-evolution 的 file_tree 快照发布模式可用于 experience memory 的团队共享——将 `experience_json` 序列化后通过类似 Supabase RPC 方案发布到团队注册表，其他实例通过 `install.py` 类似机制安装。与现有 `learning/context.py` 的 GSSC 结构兼容（L1=rule 元数据，L2=完整 rule，L3=支撑该 rule 的历史 case）。

**6. 审计规则 → experience rule 质量门控**

audit.py 的静态扫描思路可用于 experience rule 的发布前检查：
- 硬编码 symbol（如 `"BTC/USDT"` 字面量出现在 conditions 中）→ FAIL
- rate 值 = 0.0 或 1.0（极值，可能过拟合）→ WARN
- 缺少 `regime_tags`（无法按市场状态检索）→ WARN
- maturity < 0.3 且 `source = "reflect"` → WARN（过早外发未成熟规则）

---

## 备注 / 已解决问题（Phase 2 更新）

- **默认注册表托管** ✅ 已解决：`scripts/lib/supabase.py` 中硬编码了作者自己的 Supabase 实例 URL 和 anon key（`_DEFAULT_URL`/`_DEFAULT_ANON_KEY`）；anon key 可公开，写保护通过 RLS + security-definer RPC 实现。私有注册表通过环境变量覆盖。
- **scripts/lib/ 目录** ✅ 已精读：`supabase.py`（HTTP 客户端，懒初始化 SSL 上下文）+ `__init__.py`（dotenv 自动加载，publisher key 管理）。无上下文注入辅助函数，仅为网络和身份管理。
- **prompt-eval skill 依赖** ⚠️ 未深入：eval-mode 依赖独立安装的 `prompt-eval` skill，该 skill 未在本仓库内。评测功能需另行研究。
- **模型特定记忆** ✅ 已确认：`model_guides/` 是约定的用户侧目录，仓库不内置，由 agent 在 reflect 过程中按需创建。
- **hooks 集成深度** ✅ 已理清：skill frontmatter 中的 `hooks` 声明会被 Claude Code 直接读取执行（平台原生）；与 `.claude/settings.json` 中的 project-wide hooks 并列，无优先级冲突——两者均会执行。
- **hooks 集成深度**：YAML frontmatter 支持声明 `hooks`（PreToolUse），但 settings.json 层面的 hook 配置与 skill frontmatter hook 的优先级关系未明
