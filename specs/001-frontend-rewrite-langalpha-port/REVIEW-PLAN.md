# Review Guide: 前端重写 — LangAlpha 移植 + Crypto 化

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-04-16

---

## What This Spec Does

把项目当前的 Streamlit dashboard（5 个业务页面、运行在 `:8501`）整体替换为一套基于 LangAlpha 架构的 React 19 + Vite 7 + TypeScript 5.9 SPA，并把所有股票语义迁移成加密货币语义（资金费率、OI、清算热图，删 SEC/EDGAR）。Streamlit 不是渐进迁移而是**单 PR 物理删除**，由 4 条 ripgrep 命令 0 命中作为合并硬门槛。

**In scope:** 7 个前端路由（5 个 P1 + 2 个 P2）+ ~12 个 FastAPI endpoint（其中 6~7 个新增）+ Streamlit 全栈基础设施物理删除（代码 / 测试 / pyproject 依赖 / docker-compose service / CLI 命令 / 文档）+ 单条 `docker compose up -d` 部署能力。

**Out of scope:** SSR / Next.js / PWA / 移动端适配、多用户 / OAuth、Sentry 等错误上报后端、Streamlit 删除前的灰度过渡、`:8501 → :5173` 自动重定向（详见 [spec §Assumptions](spec.md#assumptions) A-3/A-4 与 [EC-9](spec.md#edge-cases)）。

## Bigger Picture

CryptoTrader-AI 长期问题是 Streamlit 既当监控 UI 又被当成"操作面板"，但它的状态模型（session_state、整页 rerun）与一个**异步多代理 + 多轮辩论 + 后台调度器**的系统天然冲突。本特性的真正动机不是技术栈升级，而是把"前端"从决策管线副产物升级为**一类公民**：决策有完整可复盘的详情面板（节点时间线 / Agent 网格 / 辩论 / 裁决 / 风控门 / 执行结果），回测进度可取消，断路器可二次确认重置，AI 对话从此有了 SSE 流式入口。

LangAlpha (https://github.com/ginlix-ai/LangAlpha) 是公开的多代理交易研究前端，其 [contracts/sse-events.md](contracts/sse-events.md) 里的 5 类事件协议（`message_chunk` / `tool_call` / `tool_result` / `inline_widget` / `verdict`）和 InlineWidget iframe sandbox 模式经过实战验证，本特性 selective port 而非 git clone。**没有借鉴的部分**包括 LangAlpha 的多模型路由、collaboration 抽象、186KB 的 `useChatMessages` —— 这些是反面教材，spec 在 [NFR-M-007](spec.md#nfr-m-001) 把 `useChatMessages` 硬限到 ≤ 500 行。

后端 FastAPI 不动业务逻辑（[A-5](spec.md#assumptions)），只按 [contracts/http-endpoints.md](contracts/http-endpoints.md) 在 `src/api/routes/` 里增量补 endpoint。这意味着 Phase 2 后端任务（[T056-T071](tasks.md#phase-2-foundational)）必须先于任何 P1 故事完成 —— 这条依赖容易在并行排期时被忽略。

---

## Spec Review Guide (30 minutes)

> 本指南帮你把 30 分钟集中投在最需要人类判断的部分。每节都指向具体位置，并把审阅框架成问题。

### 理解整体方向（8 分钟）

读 [spec.md User Stories](spec.md#user-scenarios--testing)（7 个故事）+ [§11 阶段实施顺序](quickstart.md#4-实施阶段顺序来自-spec-11-阶段)。带着以下问题：

- **5 个 P1 故事 1:1 替换 Streamlit 5 大页面** —— 这个"零功能回退"目标是不是**过度承诺**？比如 [Decisions 页](spec.md#user-story-2--决策复盘实时决策页priority-p1)的 8 节详情（节点时间线 / Agent 网格 / 辩论 / 经验记忆 / 裁决 / 风控门 / 执行 / OTel 链接），Streamlit 现版本是否真的全部就绪？还是说"1:1"在某些 section 只是"对应位置有占位"？
- [US3 Backtest](spec.md#user-story-3--回测新会话回测页priority-p1) 提到 [FR-302](spec.md#4-回测页backtestp1) 用 5 秒轮询而不是 SSE；[research §9](research.md) 论证了"轮询足够 + SSE 增加无谓复杂度"。**长任务（30 分钟+ 见 [SC-006](spec.md#measurable-outcomes)）下 5 秒轮询的总请求数（~360 次）后端是否已加 cache？** 还是应该退化为渐进 backoff（5s → 10s → 30s）？
- [Phase 8 Streamlit 删除](tasks.md#phase-8-e2e-全绿--streamlit-一次性删除fr-900915pr-合并硬门槛) 的"硬门槛 = 4 条 rg 命令 0 命中"。**注释里的历史引用算不算命中？** [spec FR-915](spec.md#10-streamlit-完全弃用终态约束) 说"注释也不允许"，但 [tasks T159](tasks.md#83-终态校验pr-合并硬门槛-fr-915) 给 `:8501` 加了 brainstorm/ 例外。这两处口径需要对齐。

### 需要你眼睛盯的关键决策（12 分钟）

**D-1：SSE 客户端用 streamFetch 而不是 EventSource**（[contracts/sse-events.md §1](contracts/sse-events.md#1-sse-帧格式) + [research §6](research.md)）

EventSource 不能注入自定义 header（`X-API-Key` 只能走 query string，会被 nginx access log 记录 → 安全审计风险）。streamFetch 用 fetch + ReadableStream 解决这点，但需要手写 SSE 解析、reconnect、429 退避（[FR-602](spec.md#7-ai-对话页chatagentp2)、[contracts/sse-events.md §4](contracts/sse-events.md#4-错误处理)）。
- **问题**：自手写 SSE 解析比 EventSource 多了 ~150 行代码 + 边界 case（chunk 跨包 / `data:` 多行）。如果 X-API-Key 改放 cookie / Bearer 是否就可以用 EventSource？是否值得为安全审计省一份代码？

**D-2：状态管理 React Query + Zustand 而不是 Redux/Jotai**（[research §3](research.md) + [NFR-M-008](spec.md#non-functional-requirements)）

React Query 管远程数据（自动缓存、refetch、retry），Zustand 管 UI/Settings/Chat 本地状态。spec 明确禁止 monolithic store，按领域拆 3 个（[T029-T031](tasks.md#24-状态zustand)）。
- **问题**：`useChatStore` 与 `useChatMessages` hook 的边界在哪？前者管会话列表，后者管单会话消息流？还是说 messages 也存 store？这个分工对后续维护影响很大。

**D-3：Phase 8 Streamlit 删除作为单 PR**（[A-8](spec.md#assumptions) + [FR-907](spec.md#10-streamlit-完全弃用终态约束)）

不做半路灰度的好处是避免"新旧两套都要维护"的状态；坏处是 PR 巨大（~30 个删除任务 [T143-T155](tasks.md#82-物理删除按-fr-900913-顺序)），review 难。
- **问题**：Phase 8 是否应该拆成两个 commit（"删除 Streamlit 代码" + "新增 docker-compose web service"），便于 reviewer 对照？还是为了 atomic 强制单 commit？

**D-4：Backend 后端先按需补 endpoint，前端先行**（[A-7](spec.md#assumptions) + [tasks Phase 2.8](tasks.md#28-后端-p1-endpoint-全部就绪contractshttp-endpointsmd-10-优先级表)）

Phase 2 必须把 [contracts/http-endpoints.md §10 P1](contracts/http-endpoints.md#10-实施优先级) 的 9 个 endpoint 全部就绪，前端集成才能并行。
- **问题**：后端 endpoint 工作量被低估了吗？例如 [FR-805 backtest 异步任务](spec.md#9-后端-fastapi-扩展) 涉及 TaskRegistry + 进度上报 + 取消信号，看起来比一个简单 endpoint 复杂得多。Phase 2 的 1 人天预估是否激进？

**D-5：前端测试少 mock，e2e 走真实 docker compose**（[A-9](spec.md#assumptions) + [research §9](research.md)）

Playwright 启动前 `docker compose up -d`，跑真实 postgres/redis/api。优点是测的是真实集成，缺点是 CI 启动慢（~2 分钟）+ 数据准备（fixtures vs 真实 seed）成本高。
- **问题**：CI 是否应该并行跑 5 个 P1 spec？还是串行？fixture 准备是 SQL seed 还是 fixture-as-code？这部分 [tasks T140](tasks.md#81-e2e-终态校验) 没展开。

### 我不太确定的地方（5 分钟）

- [spec FR-009](spec.md#1-项目脚手架与基础设施) 默认 `VITE_API_BASE_URL=http://localhost:8003` —— 这是 `arena serve` 的 CLI 默认（`src/cli/main.py:384`），不是 docker-compose 暴露端口。**用户在 dev 模式启动后端的方式有两种**：`uv run arena serve`（:8003）vs `docker compose up api`（看 docker-compose.yml）。我假设两者端口对齐，但 [docker-compose.yml](../../docker-compose.yml) 实际暴露的端口我没核对。
- [FR-910](spec.md#10-streamlit-完全弃用终态约束) 要求清理 `CLAUDE.md`、`.kiro/**`、`brainstorm/**`、`CHANGELOG.md` 中的 streamlit 引用。我在 [tasks T152](tasks.md#82-物理删除按-fr-900913-顺序) 把它列成单任务，但这些文件其实是不同性质的：CLAUDE.md 是项目记忆（**当前架构描述应改写**），brainstorm/ 是历史会话（**应保留作为决策溯源**）。spec 没明确区分。我倾向于**只清当前架构描述、保留历史溯源**，但这与 FR-915 的"4 条 rg 0 命中"严格口径冲突。
- [research §11 ChatAgent iframe sandbox](research.md) 注入 24+ CSS 变量 —— 这个数字来自 LangAlpha 的具体实现，但我没读源码确认到底是哪 24 个。Phase 9 实施时 [T170](tasks.md#前端-2) 可能需要回查。
- [tasks T053 EquityChart 5k 点降级蜡烛图](tasks.md#27-共享组件图表决策详情web-vitals)：这个降级阈值是 [EC-11](spec.md#edge-cases) 写死的 5000。但 lightweight-charts v4 的实际性能拐点可能不是 5k，需要 Phase 11 [T196](tasks.md#phase-11-部署文档与-polishnfr-d--cross-cutting) 实测。

### 风险与开放问题（5 分钟）

- 如果 [FR-905 web service](spec.md#10-streamlit-完全弃用终态约束) 在 docker-compose 里依赖 api service，但 api 本身依赖 postgres + redis，**`docker compose up -d` 第一次拉起的 cold start 时序是否会让 web 在 api 还没 healthy 时就 ready**？是否需要 `depends_on.condition: service_healthy` + healthcheck？[SC-009](spec.md#measurable-outcomes) 只说"一条命令拉起全栈"没说健康检查策略。
- [Phase 8](tasks.md#phase-8-e2e-全绿--streamlit-一次性删除fr-900915pr-合并硬门槛) 一次性删除 streamlit 后，**如果 web 服务有任何 P0 bug 必须紧急回滚**，是回滚整个 PR（含 docker-compose 改动 + 依赖删除 + 文档改动）还是只回 web 相关？这个回滚预案 spec 没写。
- [NFR-M-007](spec.md#non-functional-requirements) `useChatMessages ≤ 500 行` 是硬限。如果 Phase 9 实施时发现确实需要 600 行才能正确处理 5 类 SSE 事件 + 错误恢复 + IndexedDB 持久化，**是降低质量硬塞 500 行还是修改 NFR**？quickstart §8 Q4 给的拆分方案（拆 `lib/chat-utils.ts`）是否真能压到 500 行内？
- [TradingView Widget](spec.md#user-story-7--加密货币市场看板市场看板页priority-p2) 的免费版有 attribution 要求 + 嵌入域限制。**生产部署如果走自建域名 + nginx 反代，是否会被 TradingView 的 referer 检查拒绝**？[EC-12](spec.md#edge-cases) 给了降级方案但没说判定标准。
- 5 个 P1 故事的 e2e（[T075/T088/T105/T118/T129](tasks.md#phase-3-user-story-1--自动交易监控dashboardp1--mvp)）走真实 docker compose 全栈，但**真实回测需要 30 分钟**（SC-006），e2e 怎么测完整 backtest 流程？mock 一个 5 秒回测？还是用极短的回测窗口（1 天数据）？

---

*Full context in linked [spec](spec.md) and [plan](plan.md).*
