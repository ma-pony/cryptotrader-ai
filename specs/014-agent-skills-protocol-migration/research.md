# Phase 0 Research — 014 Agent Skills 协议迁移

解决 spec review 推迟的 5 个 gap + plan 阶段新引入的 1 个决策。

---

## R1：Reflection job 触发机制（resolves REVIEW-2）

**Decision**：在 graph 中作为新 node `nodes/reflection.py`，由 `nodes/data.py:verbal_reinforcement` 按 `[experience] every_n_cycles` 配置在每 N 个 trading cycle 后调用。

**Rationale**：
- 单进程同 process，单写者语义无需跨进程文件锁
- 与现状最少差异——`every_n_cycles` 配置已存在
- 失败仅捕获在节点本身，不阻塞下游 trading cycle（FR-018）

**Alternatives considered**：
- (b) 独立 cron / launchd timer：解耦但多一层运维（独立进程需要文件锁、需要独立日志、需要独立 health check）
- (c) Scheduler post-cycle hook：介于 (a)/(b)，无明显优势

---

## R2：性能基线与 cycle latency SC（resolves REVIEW-5）

**Decision**：
- 测一次基线：当前系统 trading cycle wall-clock 中位数（用 `metrics.cycle_duration_ms` 已有指标）
- 实现 `load_agent_skills` 后 microbench：~50 文件读 + YAML parse 单进程 SSD 上目标 < 50ms
- 引入 plan-level 新 SC（不回写 spec）：
  - `load_agent_skills(agent, regime)` p95 ≤ 100ms
  - trading cycle 总耗时较基线增量 ≤ 5%

**Rationale**：
- 50 文件的本地 SSD 读 + parse 估算 ~10-30ms；100ms 留 3x 余量给慢盘 / Linux container
- 5% cycle 增量 ≈ 当前 60s cycle 上加 3s——人不易察觉，metrics 报警阈值合理

**Alternatives considered**：
- 不设 cycle latency SC：spec SC-008 已说"启动时间不变或下降"，运行时性能其实需要单独 SC，否则只在异常情况才被观察到
- LRU 缓存：本期不引入。理由：每 cycle 4 次调用，缓存命中节省的时间 < 100ms 成本；只有 reflection 写入后失效——简单但收益有限。如基线测试显示 latency 超 100ms 再加

---

## R3：跨 agent 命名消歧（resolves REVIEW-6 + 完善 FR-008b）

**Decision**：
- `load_skill(name)` tool：未加 `agent::` 前缀且全局有同名 → 返回 `{"error": "ambiguous_name", "candidates": [...]}`
- `applied: <name>` reflection 解析：同名跨 agent → 跳过该引用 + logger.warning（spec FR-022 已部分覆盖）
- middleware 渲染 description 列表时**强制加 `agent::` 前缀**——避免 agent 用简短形式

**Rationale**：
- prompt 里看到的就是 `tech::funding_squeeze_long` 全称，agent 学到的命名习惯就避开了 ambiguity
- 即便 agent hallucinate 了短名字，tool 也能 fail-fast（不静默猜）

**Alternatives considered**：
- (a) "短名按调用方 agent 自身的目录解析"：对 4 个 analysis agent 工作，但 verdict 节点是 across-agent，没有"自身" → 不通用
- (b) "短名匹配第一个找到的"：随机性（取决于文件遍历顺序）→ 不可重现

---

## R4："短暂时间窗"具体值（resolves AMBI-1）

**Decision**：
- 单写者 + 多读者并发用 `fcntl.flock` 实现
- reflection job 写入：`with open(file_path, "w") as f: fcntl.flock(f, LOCK_EX); ...; fcntl.flock(f, LOCK_UN)` 全程独占
- agent 节点读：`with open(file_path, "r") as f: fcntl.flock(f, LOCK_SH); ...; fcntl.flock(f, LOCK_UN)` shared 锁
- agent 一次 cycle 读多个文件：在 `load_agent_skills` 函数级别一次性获取目录的 shared 锁（即对 `agent_skills/{agent}/` 目录加锁），减少锁开销

**Rationale**：
- POSIX flock 跨进程安全；本期虽单进程，未来 reflection 走独立 cron 也无需改代码
- macOS / Linux 都原生支持，无第三方依赖
- 等待 window 由 OS 调度——通常 < 1ms，最坏 reflection 写一文件 < 50ms 内 agent 读侧自动等待

**Alternatives considered**：
- 纯 Python `threading.Lock`：仅同进程有效，扩展性差
- 文件命名版本号 + 原子 rename（无锁方案）：实现复杂度高（需协议处理 list 时的版本不一致）

---

## R5：启动时间基线（resolves AMBI-2）

**Decision**：
- 迁移前在干净环境测 5 次完整启动流程（`git clone → docker compose up → arena migrate → arena run BTC/USDT 完成`）取中位数，记录于 implementation 阶段 PR 描述
- 实现完成后再测一次对比，差值在噪声内（±10s）即视为 SC-008 通过

**Predicted impact**：
- 新增 `agent_skills/` 17 markdown + 8 .gitkeep ≈ 50KB → git clone +< 1s
- 无新依赖 → uv sync 不变
- 无新 DB 表 → arena migrate 不变（仅 drop 一列；无前置数据迁移）
- 综合：增量 < 2s，远小于噪声

**Alternatives considered**：
- 不设硬性数值：成本忽略不计，留主观判断即可

---

## R6：`load_skill` 双接口（plan 阶段新引入）

**Problem**：spec FR-008a 说 4 个 analysis agent + verdict 节点都能调 `load_skill`。但 verdict 节点目前**不走 `create_agent`**（直接 LLM `ainvoke`），无法享受 middleware 注入与 tool 注册。

**Decision**：
- `load_skill` 实现为**单一 Python 函数**：`def load_skill(name: str, requesting_agent: str | None = None) -> dict`
- 同时**包装为 LangChain tool**：通过 `langchain.tools.tool` 装饰器或 `BaseTool` 子类，在 `SkillsInjectionMiddleware.tools = [load_skill_tool]` 注册
- 4 个 analysis agent 通过 middleware 自动获得 tool 调用能力
- verdict 节点直接调 Python 函数（同代码路径）；本期**不强求** verdict 也走 create_agent

**Rationale**：
- 双接口共享同一实现 = 行为一致 + 单点测试
- verdict 改造走 `create_agent` 范围太大（现 prompt template 也要重做）——超出本 feature
- 结果：verdict 应用 skill 时通过 reasoning 文本中的 `applied:` 字段被反思识别；不要求 verdict 主动调 `load_skill`

**Alternatives considered**：
- 强制 verdict 改 create_agent：scope creep，且 verdict 已用 weighted/AI 双 verdict 不易插中间件
- verdict 不能调 `load_skill`：会限制反思的精度，因为 verdict 应用 skill 是最关键的归因点

---

## 决策对 spec / plan 影响摘要

| Research | 影响 spec？ | 影响 plan？ | 影响 implementation？ |
|---|---|---|---|
| R1 reflection 触发 | 否 | ✅ 在 plan.md 项目结构中明确 `nodes/reflection.py` | ✅ 新文件 |
| R2 性能 SC | 否 | ✅ plan-level 新 SC | ✅ 加 microbench / 基线测试 |
| R3 命名消歧 | 否（FR-008b 已覆盖）| ✅ middleware 强制 `agent::` 前缀 | ✅ render 逻辑 |
| R4 文件锁 | 否 | ✅ contracts/middleware 提到 fcntl | ✅ loader / writer 加锁 |
| R5 启动时间基线 | 否 | ✅ 写入 PR 描述 | ✅ 测前 5 次 |
| R6 双接口 | 否 | ✅ contracts/load_skill 同时是函数 + tool | ✅ skill/tool.py 双导出 |

**所有 spec NEEDS CLARIFICATION 已 0**：confirmed by re-grep `specs/014-*/spec.md`，无 `[NEEDS CLARIFICATION]` 标记。
