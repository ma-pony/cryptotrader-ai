# Brainstorm: 深度分析借鉴 LangAlpha

**日期**: 2026-04-17
**仓库**: https://github.com/ginlix-ai/LangAlpha (803 stars, Apache 2.0)
**定位**: "Claude Code for Finance" — 金融领域 AI Agent 平台
**分析方法**: 16 个并行研究 Agent (3 轮: 4+6+6)，覆盖 SSE/中间件/Skills/Automations/前端/MCP/PTC/Steering/HITL/模型韧性/Redis/线程/记忆/测试/部署/配置/安全/可观测性/金融领域/API 设计/LangGraph 图构建/子 Agent 编排/DB Schema/性能优化

---

## 〇、执行摘要

**LangAlpha 本质**：面向金融分析师的通用 AI 助手平台（Agent 决定做什么）。
**CryptoTrader 本质**：自动化加密货币交易管道（系统控制流程）。

**两者根本差异在图架构**：LangAlpha 是单 ReAct 节点 + 24 层中间件；CryptoTrader 是多节点确定性 DAG + 条件边。CryptoTrader 的架构更适合交易场景（确定性 > 灵活性），不建议改。

**最大差距不在前端而在 Agent 基础设施层**：
- 🔴 缺失：价格触发器、图表视觉分析、断线重连、Live Steering
- 🟡 薄弱：模型韧性（单级 fallback）、SSE 事件（3 种 vs 15 种）、Agent 可配置性
- 🟢 已有优势：确定性交易管道、经验记忆 + regime 感知、辩论机制、risk gate、fail-closed 安全策略

**推荐首批 Spec**：价格触发器 (Spec A) → 模型韧性 (Spec C) → 图表分析 (Spec B)

---

## 一、LangAlpha 架构全景

### 技术栈
| 层 | 技术 |
|---|---|
| 前端 | React 19 + Vite + TypeScript + Tailwind + Radix UI + Framer Motion |
| 后端 | FastAPI + psycopg3 AsyncConnectionPool (无 ORM, 原始 SQL) + Alembic |
| Agent | Claude (主) + OpenAI/Gemini/DeepSeek/Qwen (备) + 24 层中间件栈 |
| 沙箱 | Daytona VM (PTC 代码执行) + SHA-256 manifest 增量同步 |
| 工具 | 9 个 FastMCP Server + 27 个声明式 Skills |
| 流式 | POST SSE + Redis List 事件缓冲 + Pub/Sub 跨进程通知 |
| 认证 | Supabase JWT (RS256/ES256) + JWKS 缓存 (300s) + BYOK 加密存储 |
| 部署 | Docker Compose 4 服务 (profile 分离基础设施) |

### 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  streamFetch POST SSE ─── watchThread Pub/Sub ─── InlineWidget  │
└──────────────┬───────────────────┬──────────────────────────────┘
               │                   │
               ▼                   ▼
┌──────────────────────┐  ┌─────────────────────┐
│  BackgroundTaskManager│  │  Redis               │
│  asyncio.shield()    │  │  - List: 事件缓冲     │
│  live_queues (mem)   │──│  - Pub/Sub: wake 通知 │
│  soft interrupt (ESC)│  │  - Steering: 实时引导  │
└──────────┬───────────┘  │  - Burst guard: 限流  │
           │              └─────────────────────┘
           ▼
┌──────────────────────────────────────────────────┐
│              PTCAgent (LangGraph)                  │
│                                                    │
│  Middleware Stack (按 prompt cache 优化排序):        │
│  ┌─ Static (cached prefix) ─────────────────────┐ │
│  │ SubAgent defs → Tools → Skills → Summarize   │ │
│  │ → Model Resilience → Prompt Cache breakpoint  │ │
│  └──────────────────────────────────────────────┘ │
│  ┌─ Dynamic (not cached) ───────────────────────┐ │
│  │ Steering → Workspace (agent.md) → Runtime    │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  SubagentOrchestrator → 并行子 Agent (独立工具集)    │
│              ↓                                     │
│  MCPRegistry → 9 MCP Server → Daytona 沙箱        │
└──────────────────────────────────────────────────┘
```

---

## 二、十五大核心模式深度分析

---

### 模式 1: BackgroundTaskManager — 工作流与 SSE 解耦 ⭐⭐⭐⭐⭐

**原理**: 将 Agent 工作流的执行与 HTTP SSE 连接完全解耦。

**实现细节**:
- 单例管理器，`TaskInfo` dataclass 跟踪每 thread 的 asyncio task + 状态 (QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED/SOFT_INTERRUPTED)
- `asyncio.shield()` 包裹生成器消费循环 → 客户端断连不取消工作流
- 事件双路广播：
  - **内存 `asyncio.Queue`** (`live_queues`) — 同进程实时 viewer，零延迟
  - **Redis List** (`workflow:events:{thread_id}`, RPUSH + LTRIM) — 跨进程 + 断线重连回放
- 断线重连：`GET /{thread_id}/messages/stream` 先 `LRANGE` Redis List 回放历史，再 attach 到 `live_queues`
- Watch 端点：`GET /{thread_id}/watch` 订阅 Redis pub/sub `thread:wake:{thread_id}`，45s keepalive ping，30min 最大时长
- 后台 dispatch：`X-Dispatch: background` header → 返回 `{"status": "dispatched"}` 立即响应
- `pre_register()` 关闭 dispatch → start_workflow 之间的时间窗口竞态

**CryptoTrader 现状**: `/api/chat/stream` 直接绑定 `StreamingResponse` → 断连即丢失分析结果

**借鉴价值**: ★★★★★ — 交易决策分析 30-60s，断连代价极高

**实施路径**:
```
Phase 1: Redis List 事件缓冲 + 断线重连端点 (3天, 含测试)
Phase 2: asyncio.shield() + live_queues 双路广播 (2天)
Phase 3: Watch 端点 + 后台 dispatch (1天)
Phase 4: 前端重连逻辑 + 集成测试 (2天)
```

---

### 模式 2: Soft Interrupt + Checkpoint Flush ⭐⭐⭐⭐

**原理**: 用户按 ESC 不是强制 cancel，而是协作中断 — 主 Agent 停止，子 Agent 继续，状态完整保存。

**实现细节**:
- `BackgroundTaskManager` 持有 `soft_interrupt_event: asyncio.Event` per TaskInfo
- `consume_workflow()` 每 yield 一个事件后检查 `soft_interrupt_event.is_set()`
- 触发时：
  1. `aclose()` 关闭生成器，抛出 `SoftInterruptError`
  2. `_flush_checkpoint()` — 调用 `graph.aupdate_state()` 强制写入当前状态
  3. 发送 `None` sentinel 到所有 `live_queues` 结束 SSE 流
  4. 如有后台子 Agent，spawn collector task 等待 + 持久化结果
  5. `thread_id` 立即可用于发送下一条消息（从 checkpoint 继续）

**CryptoTrader 现状**: 无中断机制，只能等 Agent 跑完或整个请求超时

**借鉴价值**: ★★★★☆ — 用户发现市场突变时，需要立即停止当前分析并重新开始

**实施路径**: 与模式 1 合并实施，在 BackgroundTaskManager 中增加 soft interrupt 支持

---

### 模式 3: Live Steering — 运行中实时引导 ⭐⭐⭐⭐⭐

**原理**: 用户在 Agent 运行过程中发送消息，不打断工作流，而是在下一次 LLM 调用前注入。

**实现细节**:
- `SteeringMiddleware` hook 到 LangChain `AgentMiddleware.abefore_model()`
- 用户发送消息 → 服务端推入 Redis List `workflow:steering:{thread_id}`
- 每次 LLM 调用前，middleware 原子 `LRANGE + DEL` 取出所有 pending 消息
- 批量合并为一条 `HumanMessage`，前缀 `[Steering from User]`
- 发出 `steering_delivered` SSE 事件通知前端
- 子 Agent 有独立的 `SubagentSteeringMiddleware`，使用 `subagent:steering:{tool_call_id}` key

**CryptoTrader 现状**: 无 live steering — 分析开始后只能等待完成

**借鉴价值**: ★★★★★ — 场景举例：
- 分析进行中突发重大新闻 → 用户输入 "注意刚才 CZ 的推文"
- 回测运行中发现参数不对 → 用户输入 "止损改为 3%"
- Agent 分析方向偏了 → 用户纠正 "重点看资金费率，不是 RSI"

**实施路径**:
```
Phase 1: Redis List steering 队列 + 注入 middleware (2天)
Phase 2: 前端 steering 输入 UI + delivery 确认 (1天)
Phase 3: 子 Agent steering 支持 (1天)
```

---

### 模式 4: HITL (Human-in-the-Loop) 中断 ⭐⭐⭐⭐

**原理**: Agent 在关键决策点暂停，请求人类审批。

**三种中断机制**:
1. **Plan approval** (`PlanModeMiddleware`): Agent 提交 `SubmitPlan(description=...)` → LangGraph `interrupt()` → 用户 approve/reject → 恢复
2. **AskUserQuestion** (`AskUserMiddleware`): Agent 调用 `AskUserQuestion(question, options)` → interrupt → 用户选择 → 返回答案
3. **Proposal-based**: 类型化中断（create_workspace / delete_thread 等），前端通过 `PROPOSAL_DATA_KEY_MAP` 路由到不同 UI 卡片

**前端模式** (`useChatMessages.ts`):
```typescript
const PROPOSAL_DATA_KEY_MAP = {
  create_workspace: 'workspaceProposals',
  start_question: 'questionProposals',
  ptc_agent: 'ptcAgentProposals',
  // ...
};
// SSE interrupt 事件 → 按类型路由到 AssistantMessage 对应字段 → 渲染不同审批卡片
```

**CryptoTrader 现状**: 完全自动化，无人工审批节点

**借鉴价值**: ★★★★☆ — 关键场景：
- 大额交易前要求人工确认（>5% 仓位）
- Agent 分歧严重时请求人工裁决
- 风控熔断重置需要人工审批

**实施路径**:
```
Phase 1: LangGraph interrupt() 在 risk gate 前的确认点 (2天)
Phase 2: 前端审批卡片 + sendHitlResponse (1天)
Phase 3: Telegram bot 作为远程 HITL 通道 (2天)
```

---

### 模式 5: Model Resilience — 多 Provider + 自动降级 ⭐⭐⭐⭐⭐

**原理**: 两层中间件 (retry → fallback) + 两文件 manifest (models.json + providers.json)。

**实现细节**:

**Manifest 系统**:
- `models.json`: `{model_id, provider, visible, input_modalities, parameters, system_provider}`
- `providers.json`: `{sdk, env_key, access_type, byok_eligible, base_url, variants}`
- 支持 6 个 SDK: anthropic / openai / codex / gemini / deepseek / qwq
- Provider variants 共享 SDK 指向不同 base_url（Doubao, Moonshot 等）

**三种 access_type**:
- `api_key` — BYOK: `pgp_sym_encrypt` 加密存储，调用时解密传入 `api_key=...`
- `oauth` — Claude/Codex OAuth 流: `ChatAnthropicOAuth` 使用 `Authorization: Bearer`
- `local` — Ollama/vLLM 等本地模型

**Key 解析优先级**: BYOK override → 环境变量 → 平台代理 (system key)

**Resilience 两层中间件**:
```
Model → ModelRetryMiddleware (3次, 指数退避 1s→2s→4s, jitter)
      → ModelFallbackMiddleware (切换预解析的 fallback 模型列表)
```

**Agent 配置**:
```yaml
llm:
  name: "claude-sonnet-4-6"       # 主模型
  flash: "claude-haiku-4-5"       # 轻量子模型
  summarization: "claude-haiku-4-5" # 摘要专用
  fallback: ["gpt-5.4-mini"]      # 降级链
```

**CryptoTrader 现状**: `create_llm()` 有 `.with_fallbacks()` 但只支持单一 fallback，无重试中间件，无 BYOK

**借鉴价值**: ★★★★★ — 交易系统 LLM 不可用 = 错过交易机会

**实施路径**:
```
Phase 1: models.toml manifest (model_id → provider mapping) (1天)
Phase 2: 指数退避重试 + 多级 fallback 链 (1天)
Phase 3: 分角色模型配置 (analysis/flash/summarization/fallback) (1天)
```

---

### 模式 6: Prompt Caching 优化 ⭐⭐⭐⭐

**原理**: 中间件栈按内容稳定性排序，静态内容在前（被缓存），动态内容在后（不影响缓存命中）。

**实现细节**:
- `AnthropicPromptCachingMiddleware`（来自 `langchain_anthropic.middleware`）在 retry/fallback 之后运行
- 在最后一个 system message block 上放置 `cache_control` breakpoint
- 排序原则：
  ```
  ┌─ Cached Prefix ─────────────────────────┐
  │ Skills + Tools + SubAgent definitions    │  ← 稳定，每会话不变
  ├──────────────────────────────────────────┤
  │ cache_control breakpoint                 │
  ├──────────────────────────────────────────┤
  │ agent.md (workspace memory)              │  ← 偶尔变化
  │ Runtime context (time, user profile)     │  ← 每次变化
  └──────────────────────────────────────────┘
  ```
- `models.json` 中 `parameters.enable_caching=true` per-model opt-in

**CryptoTrader 现状**: 使用 `SQLiteCache` 缓存完整响应，但无 prompt prefix 缓存

**借鉴价值**: ★★★★☆ — 4 个 Agent 的系统 prompt + 工具定义是稳定的，每轮对话只有市场数据变化

**实施路径**: 在 `create_llm()` 中对 Anthropic 模型启用 prompt caching，重排 system message 顺序

---

### 模式 7: Auto-Summarization 两级压缩 ⭐⭐⭐⭐

**原理**: 长对话自动压缩，避免 context window 溢出。

**实现细节**:
- `SummarizationMiddleware` 通过 `awrap_model_call()` 包裹每次模型调用
- **Tier 1 (无 LLM)**: 消息数 > `truncate_args_trigger_messages` → 截断旧消息的大型 tool arguments，保留最近 20 条完整。去重陈旧 Read 结果
- **Tier 2 (LLM 摘要)**: `total_tokens >= 120,000` → 保留最近 5 条消息，其余用 `gpt-5-nano` 摘要。Base64 blob 预剥离，原始消息离线到沙箱文件系统
- 链式摘要支持（summarization of summarization）
- 非破坏性：完整 checkpoint 历史保留

**CryptoTrader 现状**: 无对话压缩机制（单次分析不会溢出，但 ChatAgent 页长对话可能）

**借鉴价值**: ★★★★☆ — ChatAgent 页的 SSE 对话会积累上下文，需要压缩

**实施路径**: 在 `/api/chat/stream` 中实现两级压缩（参数截断 + LLM 摘要）

---

### 模式 8: agent.md 工作区记忆 ⭐⭐⭐⭐

**原理**: 可读的 Markdown 文件作为 Agent 持久记忆，每次 LLM 调用前自动注入系统 prompt。

**实现细节**:
- 位于沙箱文件系统 `/agent.md`
- 内容：workspace 用途、目标、Thread Index、Key Findings、File Index + YAML front matter
- `WorkspaceContextMiddleware.wrap_model_call()` 每次调用前读取并追加为系统 prompt 的最后一个 content block
- Session 级缓存 + `_agent_md_dirty` flag → Agent 修改后通过 `on_agent_md_write` 回调失效缓存
- YAML front matter 变化时 fire-and-forget 同步回 Postgres
- 硬截断 8192 字符

**CryptoTrader 现状**: `experience_memory` 是 JSON 格式，存在 SQLite 中，通过 GSSC 引擎注入 prompt

**借鉴价值**: ★★★★☆ — agent.md 比 JSON 更可读、可手动编辑、可 git 跟踪

**建议**: 不替换 experience_memory（那是结构化数据），但为 ChatAgent 对话引入 agent.md 风格的 per-workspace 记忆文件

---

### 模式 9: Automations — 定时/价格触发 ⭐⭐⭐⭐⭐

**实现细节**:

**AutomationScheduler**:
- 30s 轮询循环，查询 `next_run_at` 已到期的 automations
- `claim_due_automations()` 原子锁（server_id）防多实例重复执行
- 每个 claimed automation 作为独立 `asyncio.Task` dispatch
- Cron: `croniter` 时区感知重计算 `next_run_at`; 一次性: 设 `next_run_at = NULL`
- 优雅关闭：60s drain → force-cancel

**AutomationExecutor** (7 步):
1. 标记状态 → 2. 解析 workspace → 3. 线程策略 (new/continue/fresh) → 4. 构建 `ChatRequest` → 5. 调用 Agent workflow → 6. 消费 async generator → 7. 持久化结果
- Webhook 通知 (开始/完成/失败)
- 价格触发器自动恢复失败; 一次性标记完成防重跑

**触发类型**:
- Cron 表达式（"每天 9:00 分析 BTC"）
- ISO datetime（一次性）
- 价格条件：阈值 / 百分比变化 + 冷却期

**Agent-facing tools**: `check_automations`, `create_automation`, `manage_automation` — Agent 自己可以创建 automation

**前端**: 模板卡片 (preset cron) + 内联表单 (AnimatePresence 滑入) + 暂停/恢复/触发/删除

**CryptoTrader 现状**: APScheduler 3.x，只有 interval + cron 两种，无价格触发，无 webhook，无前端管理

**借鉴价值**: ★★★★★ — 价格触发器是交易系统的核心能力缺失

**实施路径**:
```
Phase 1: 价格触发器 (Binance WS → 条件匹配 → 触发分析) (4天, WS 稳定性 + 重连 + 测试)
Phase 2: Webhook (Telegram bot) (2天)
Phase 3: 前端调度器页面 (模板卡 + cron 编辑器 + 管理表) (3天)
Phase 4: Agent self-scheduling (Agent 自建 automation) (2天)
```

---

### 模式 10: MCP Server 数据层 ⭐⭐⭐⭐

**实现细节**:
- 9 个 FastMCP Server: price_data / macro / fundamentals / options / yf_analysis (4 variants)
- 双传输: stdio (沙箱内进程间) + HTTP/SSE (生产)
- `MCPRegistry` 会话初始化时通过 JSON-RPC `tools/list` 发现所有工具 → 存为 `MCPToolInfo`
- 数据源回退: ginlix-data → FMP (Financial Modeling Prep)
- `ToolFunctionGenerator` 将 MCP JSON Schema → 带类型签名的 Python 函数 (用于 PTC 沙箱)
- Dispatch: `MCPRegistry.call_tool(server, tool, args)` → connector → transport

**CryptoTrader 现状**: 硬编码 Python 模块 (binance.py / sosovalue.py / rss_news.py)

**借鉴价值**: ★★★★☆ — MCP 标准化后可接入 Claude Code / Cursor 生态

**实施路径**:
```
Phase 1: binance.py → FastMCP Server (stdio) (2天)
Phase 2: OKX / macro data MCP Server (2天)
Phase 3: MCPRegistry 动态发现 + Agent 自动选择 (2天)
```

---

### 模式 11: Subagent Registry + Compiler ⭐⭐⭐⭐

**实现细节**:
- `SubagentRegistry`: 两层加载 — 内置定义 → YAML 用户覆盖（同名覆盖）
- `SubagentCompiler`: 编译 `SubagentDefinition` → 可执行 `SubAgent` TypedDict `{name, prompt, tools}`
- 三层 prompt 解析: 自定义字符串 → 模板 → 默认 base + role
- 子 Agent 共享中间件但无 HITL、无后台任务（安全隔离）
- `BackgroundSubagentOrchestrator` 管理并行执行 + 超时 + 结果收集

**CryptoTrader 现状**: 4 个硬编码 Agent，`asyncio.gather()` 并行，LLM 有 timeout 配置但无 per-agent 超时管理

**借鉴价值**: ★★★★☆

**实施路径**:
```
Phase 1: AgentRegistry (TOML 定义 name/model/prompt_template/tools/timeout) (2天)
Phase 2: 用户可通过配置 enable/disable/override Agent (1天)
Phase 3: Per-agent 超时 + 降级 (保守默认值 or skip) (1天)
```

---

### 模式 12: Skills 声明式系统 ⭐⭐⭐⭐

**实现细节**:
- 27 个 Skills，每个是目录 + `SKILL.md`
- SKILL.md 声明: 能力、可用工具、参数、交付格式
- `SKILL_REGISTRY` 运行时加载，`SubagentCompiler._resolve_tools()` 引用
- Skills 不是代码类，是声明式规范

**CryptoTrader 借鉴**: 将交易策略模式化为 Skills:
```
strategies/
  trend-following.md    # 趋势跟随: MA crossover, breakout
  mean-reversion.md     # 均值回归: Bollinger, RSI extremes
  momentum.md           # 动量: volume spike, OI surge
  funding-arb.md        # 资金费率套利: funding > threshold
```
Agent 根据 `regime_tags` 自动选择合适的策略 Skill。

---

### 模式 13: 自适应轮询 + WS-first 双层实时 ⭐⭐⭐⭐

**LangAlpha Dashboard 数据策略**:
| 数据 | 策略 | staleTime |
|------|------|-----------|
| Market Status | 60s 固定轮询 | 30s |
| Market Indices | 市场开盘 30s / 收盘 60s (动态切换) | 10s |
| News Feed | 单次加载 | 5min |
| Stock Prices | WS connected → 禁用 REST 轮询; WS down → 60s REST fallback | - |

**MarketView WS-first 模式**:
- `MarketDataWSContext` → `useMarketDataWS` → `subscribe(symbol)/unsubscribe(symbol)`
- WS 新 bar → 直接 `setState`（绕过 React Query 缓存）
- `realTimePriceMatch` 守卫防止切换股票时陈旧数据闪烁
- 优先级: `wsPrices.get(symbol)` > `realTimePrice` (REST)

**CryptoTrader 现状**: 所有数据 React Query 固定 10s 轮询

**借鉴价值**: ★★★★☆ — Dashboard 可采用自适应轮询 + Binance WS 实时价格

**实施路径**:
```
Phase 1: Binance WS 实时价格 (useMarketDataWS hook) (2天)
Phase 2: 自适应轮询 (交易时段更频繁) (1天)
Phase 3: WS-first + REST fallback 模式 (1天)
```

---

### 模式 14: 图表截图 → LLM 分析 ⭐⭐⭐⭐⭐

**LangAlpha MarketView 创新**:
- `chartRef.captureChartAsDataUrl()` 获取图表 base64 截图
- 同时构建文字描述 (symbol / 区间 / MA / RSI / OHLCV / ATH-ATL)
- 以 `{ type: 'image', data: base64, description: text }` 发给 chat API 作为 `additional_context`
- 双模式:
  - **Fast**: 直接调用 flash API，在 MarketView 内显示
  - **Deep**: 跳转 `/chat` 页面，携带 `state.additionalContext`

**CryptoTrader 现状**: Agent 只处理数值数据，无法"看到"图表

**借鉴价值**: ★★★★★ — 让 Agent 真正"看图说话"，比纯数值分析更接近人类交易者

**实施路径**:
```
Phase 1: lightweight-charts captureChartAsDataUrl API (已有图表组件) (1天, 含多时间周期截图 + 测试)
Phase 2: 图表截图 + 文字描述 → /api/chat/stream additional_context (2天, 含后端 multimodal 解析)
Phase 3: MarketView 内 "AI 分析此图" 按钮 (Fast/Deep 双模式) (1天)
```

---

### 模式 15: 结构化错误 + 429 智能处理 ⭐⭐⭐

**LangAlpha 三层错误栈**:
1. **结构化 HTTPException.detail**: `{message, type, link}` — 前端按 `type` 分支 (`"no_provider"` → 显示设置链接)
2. **SSE typed error events**: `{type: "timeout_error", elapsed, timeout, thread_id}` — 带 sequence ID 支持重连
3. **Re-raise guard**: `except HTTPException: raise` before `except Exception: raise HTTPException(500)` — 防止吞掉有意的 HTTP 错误

**前端 streamFetch 处理**:
- 429 → 从 `response.data.detail` 解构限速信息 + `Retry-After` header
- 413 → 提示消息过大
- 返回 `{ disconnected: boolean }` 支持上层重连逻辑

**CryptoTrader 现状**: 基本 HTTPException，前端只处理 status code

**借鉴价值**: ★★★☆☆ — 改善用户体验，但不是核心交易功能

---

## 三、认证与安全体系对比

| 维度 | LangAlpha | CryptoTrader | 差距 |
|------|-----------|-------------|------|
| 认证 | Supabase JWT (RS256) + JWKS 缓存 300s | X-API-Key 静态 header | 大 |
| BYOK | pgcrypto 加密存储，per-user key 注入 | 全局 config 配置 | 大 |
| 限流 | Redis INCR burst guard (10 并发/300s) + credit quota | 无 | 大 |
| SSRF | `_check_ssrf` 拦截私有/内网地址 | 无 | 中 |
| Sandbox | Daytona VM 隔离 + SHA-256 执行追踪 | 不适用 | - |
| OAuth | Claude OAuth + Codex OAuth (Bearer token) | 不适用 | - |
| Fail policy | Auth 不可达 → fail-open (放行) | Redis 不可达 → 保守拒绝 | CryptoTrader 更优 |

**CryptoTrader 保守策略更合理** — 金融交易场景应 fail-closed，LangAlpha 的 fail-open 在交易中不可接受。

---

## 四、数据库与基础设施对比

| 维度 | LangAlpha | CryptoTrader |
|------|-----------|-------------|
| DB 驱动 | psycopg3 AsyncConnectionPool (min=1, max=20) | SQLAlchemy async + asyncpg |
| ORM | 无 (原始 SQL + dict_row) | 无 (原始 SQL + dict results) |
| 迁移 | Alembic 手写 SQL | Alembic |
| Docker | 4 服务, profile 分离基础设施 | docker-compose + 独立服务 |
| DI | FastAPI Depends + typed aliases | 无 DI 层 |
| 测试 DB | 单一 `get_db_connection` mock 点 | 多处 mock |
| API 版本 | `/api/v1/` 前缀，预留多版本共存 | `/api/` 无版本前缀 |

**借鉴**:
- `单一 mock 点` 模式值得采用 — 收敛所有 DB mock 到一个入口
- API 版本前缀 — CryptoTrader 当前无版本号，后续 breaking change 需要并行版本时成本高；建议新端点使用 `/api/v1/` 前缀

---

## 五、测试架构对比

**LangAlpha 测试模式**:
```
tests/
  unit/          # 按模块镜像 src/，mock get_db_connection 单一点
  integration/   # 真实 DB + 可切换 sandbox provider (memory/docker/daytona)
```
- 单元: `pytest-asyncio` + `AsyncClient(ASGITransport)` 不启动真实服务器
- 集成: `SANDBOX_TEST_PROVIDER` 环境变量切换测试提供者
- **不直接测试 LLM 调用**，专注基础设施层
- `Makefile`: `test` / `test-sandbox PROVIDER=memory` / `test-web` / `test-all`

**CryptoTrader**: 单层 tests/ 目录，742 tests，mock 较多但正在减少

**借鉴**: 分离 unit/integration 目录 + sandbox provider 切换模式

---

## 六、不建议采用的模式（含理由）

| 模式 | 原因 |
|------|------|
| **PTC 代码执行** | 交易系统需要确定性路径；Agent 自由写代码引入不可预测风险；Daytona 增加 ~3s 延迟 |
| **Daytona 沙箱** | 运维复杂度高，每用户一个 VM；交易 Agent 的工具调用是确定性的，不需要沙箱 |
| **24 层中间件栈** | LangGraph DAG 已经是更好的编排抽象；中间件栈适合线性 pipeline，不适合分支 DAG |
| **LRU 多聊天缓存** (5 ChatView keep-alive) | CryptoTrader 不是多聊天场景 |
| **Workspace 文件持久化** | 交易系统无需文件系统级持久化，experience_memory 已满足 |
| **Supabase JWT** | 单用户交易系统不需要多租户认证；X-API-Key 简单够用 |
| **BYOK** | 单用户场景，环境变量配置 API Key 更简单 |
| **Credit quota billing** | 自用系统不需要计费 |
| **OSS host mode** (dev bypass) | 已有 test fixtures 覆盖 |
| **AnimatePresence 页面过渡** | 交易界面需要信息密度和即时响应；过渡动画增加感知延迟，且 Framer Motion (+30KB gzipped) 不值得为此引入 |

---

## 七、CryptoTrader 已有优势（LangAlpha 不具备）

报告不应只单向分析差距，CryptoTrader 在以下方面优于 LangAlpha：

| 优势 | CryptoTrader | LangAlpha |
|------|-------------|-----------|
| **确定性交易管道** | 显式 DAG: Data→Agent→Debate→Verdict→Risk→Execute，每步可审计 | 单 ReAct 节点，LLM 自主决定路径，不可预测 |
| **多 Agent 辩论机制** | 4 Agent 并行分析 → 辩论门控 → 共识/分歧检测 → 加权或 AI 裁决 | 无辩论机制，单 Agent 或顺序子 Agent |
| **经验记忆 + Regime 感知** | GSSC 结构化经验注入，regime_tags 匹配，防过拟合五层防线 | agent.md 自由文本，无结构化经验系统 |
| **Risk Gate** | 独立 risk gate node：仓位限制、日损上限、频率限制、熔断器 | 无独立风控层，依赖 Agent 自我约束 |
| **Fail-closed 安全** | Redis 不可达 → 保守拒绝交易 | Auth 不可达 → fail-open 放行 |
| **回测引擎** | 完整 backtest engine + session storage + 决策时间线 | 无回测能力 |
| **Crypto 原生** | 资金费率、OI、清算数据、链上指标 | 股票导向 (SEC/EDGAR/期权链) |
| **Progressive Filtering** | 共识跳过辩论 (13→4-5 LLM calls)，加权降级 (0 LLM) | 无 LLM 调用优化机制 |
| **i18n 已完成** | zh-CN 默认 + en-US，8 个命名空间，Zustand 持久化 | 英文单语 |

---

## 八、可催生的 Spec 候选 (按优先级)

### Spec A: 「价格触发器 + 智能调度」
- Binance WebSocket 价格流 → 条件匹配引擎
- 触发类型: 绝对阈值 / 百分比变化 / 连续 K 线模式 / 资金费率异常
- 冷却期防重复 + 优雅关闭
- Webhook → Telegram bot 通知
- 前端: 模板卡片 + cron 编辑器 + 管理表 + 触发历史
- Agent self-scheduling (Agent 自己创建 automation)

### Spec B: 「图表视觉分析 — Chart-to-LLM Pipeline」
- lightweight-charts `captureChartAsDataUrl()` 截图
- 结构化文字描述 (OHLCV / indicator / regime)
- `/api/chat/stream` 支持 `additional_context` multimodal 输入
- MarketView "AI 分析此图" 按钮 (Fast / Deep 双模式)
- 支持多时间周期叠加分析

### Spec C: 「LLM 韧性工程」
- models.toml manifest (model_id → provider → parameters)
- 指数退避重试 (3次, 1→2→4s, jitter)
- 多级 fallback 链 (Claude → GPT → Gemini)
- 分角色模型配置 (analysis / flash / summarization / fallback)
- Prompt cache 优化 (静态前缀 + 动态后缀排序)
- 结构化 JSON 解析重试 (5次 + schema 提示)

### Spec D: 「分析防丢失 + Live Steering」
- BackgroundTaskManager: asyncio.shield + Redis List 事件缓冲
- 断线重连: 回放 Redis 历史 → attach live_queues
- Soft interrupt (ESC): checkpoint flush → 子 Agent 继续
- Live steering: Redis List 队列 + before_model 注入
- Watch 端点: Pub/Sub 新工作流通知

### Spec E: 「HITL 人工审批 + Telegram 远程控制」
- LangGraph interrupt() 在 risk gate 前
- 大额交易确认 (>5% 仓位)
- Agent 分歧严重时人工裁决
- 前端审批卡片 + PROPOSAL_DATA_KEY_MAP
- Telegram bot 作为远程 HITL 通道

### Spec F: 「Agent 可配置化 — Registry + Skills」
- AgentRegistry: TOML 定义 (name / model / prompt / tools / timeout)
- 用户自定义 Agent (enable / disable / override)
- 策略 Skills: Markdown 定义 (entry / exit / risk rules)
- Regime 自动匹配策略
- Per-agent 超时 + 降级

### Spec G: 「MCP 标准化数据层」
- Binance → FastMCP Server (stdio)
- OKX / macro / onchain MCP Server
- MCPRegistry 动态发现
- Agent 自动选择数据源
- 兼容 Claude Code / Cursor 生态

### Spec H: 「WS 实时行情 + 自适应轮询」
- Binance WebSocket 实时价格流 → React Context
- useMarketDataWS hook (subscribe / unsubscribe / connection status)
- WS-first + REST fallback (WS 断开自动降级为轮询)
- 自适应轮询 (交易活跃时段 10s / 非活跃 60s)
- Dashboard metric 卡实时更新

---

## 九、关键洞察总结

1. **最大的架构差距不在前端而在 Agent 基础设施层** — LangAlpha 的 BackgroundTaskManager + Live Steering + HITL + Soft Interrupt 形成了完整的 Agent 生命周期管理，CryptoTrader 的 Agent 是"fire and forget"的一次性管道。

2. **图表视觉分析是最被低估的能力** — LangAlpha 的 chart screenshot → LLM 模式让 Agent 能"看到"技术形态，这比纯数值指标分析是质变。CryptoTrader 已有 lightweight-charts，只差一个 capture → multimodal 的 pipeline。

3. **价格触发器不是功能增强，是核心缺失** — 交易系统没有价格触发器就像闹钟不能设时间。LangAlpha 的 Automation 系统（cron + 价格条件 + Agent self-scheduling）是最值得 1:1 移植的。

4. **Model Resilience 是生产必需品** — LangAlpha 的 retry(3×) → fallback chain 模式在 API 不稳定的现实中至关重要。CryptoTrader 的 `.with_fallbacks()` 只是最基础的单级降级。

5. **Live Steering 改变了人机协作范式** — 从"下单 → 等结果"变为"下单 → 过程中纠偏 → 结果"。对于 30-60s 的多 Agent 分析，这是用户体验的质变。

6. **Prompt Cache 是免费午餐** — 只需要重排 system message 顺序（稳定内容在前，动态内容在后），就能降低 ~30% Anthropic 调用成本。零风险，一天实现。

7. **Skills 系统让交易策略从"硬编码"变为"可配置"** — 用户可以像写文档一样定义策略，Agent 根据市场 regime 自动选择。这是系统从"工具"进化为"平台"的关键一步。

8. **LangAlpha 的"不做"同样值得学习** — 无 ORM（原始 SQL）、无 OTel（只有 request-id correlation）、单一 DB mock 点、不测试 LLM 调用只测试基础设施。这些"减法"让他们保持了快速迭代。

---

## 十、深层实现细节 — 第三轮深度挖掘

### 10.1 LangGraph 图架构：根本性设计差异

**核心发现**：LangAlpha **不构建传统 StateGraph**。它使用 LangChain 1.2+ 的 `create_agent()` 创建单个 ReAct 循环节点，外层包裹 `BackgroundSubagentOrchestrator`：

```python
agent = create_agent(
    model, system_prompt=system_prompt, tools=tools,
    middleware=deepagent_middleware, checkpointer=checkpointer,
    store=store,
).with_config({"recursion_limit": 2000})

return BackgroundSubagentOrchestrator(agent=agent, middleware=background_middleware)
```

**拓扑对比**：

```
LangAlpha:   START → [单 ReAct 节点: model ↔ tool 循环] → END
             （子 Agent 通过 Task 工具内联调用，非独立图节点）

CryptoTrader: START → Data → Agents(并行) → DebateGate → Debate(条件)
              → Verdict → Risk → Execute → Journal → END
             （显式多节点 StateGraph + 条件边）
```

**设计哲学差异**：
- LangAlpha：图是**通用 AI 助手基础设施**，LLM 决定做什么
- CryptoTrader：图是**确定性交易管道**，系统控制流程

**状态 Schema**：LangAlpha 最小化 — `messages` + `todos` + `structured_response`。CryptoTrader 丰富得多 — `AgentAnalysis`, `DebateRound`, `RiskState`, `TradeOrder`, `regime_tags` 等领域对象。

**Checkpoint**：LangAlpha 用 `AsyncPostgresSaver`（每对话持久化），子 Agent 通过 `checkpoint_ns=f"task:{task_id}"` 命名空间隔离。`fork_from_turn` 不是函数，而是通过 checkpoint 命名空间 + `current_background_tool_call_id` ContextVar 实现。

**对 CryptoTrader 的启示**：不建议改架构（确定性管道更适合交易），但可以借鉴：
- `AsyncPostgresSaver` checkpoint 用于回测分支和历史复现
- `recursion_limit: 2000` 的设置（CryptoTrader 默认可能过低）
- 子 Agent checkpoint 命名空间隔离模式

---

### 10.2 完整 24 层中间件栈

**工具子栈（7 层）**：
| # | 中间件 | 作用 |
|---|--------|------|
| 1 | `ToolArgumentParsingMiddleware` | JSON 字符串参数 → Python 对象 |
| 2 | `ToolErrorHandlingMiddleware` | 异常 → 截断 800 字符 → `ToolMessage(status="error")`；`GraphBubbleUp` 不拦截 |
| 3 | `ToolResultNormalizationMiddleware` | `None→[]`, `dict/list→JSON`, 其他→`str()` |
| 4 | `LeakDetectionMiddleware` | 检测工具结果中的凭证/密钥泄露 |
| 5 | `ProtectedPathMiddleware` | 阻止对系统保护路径的文件操作 |
| 6 | `CodeValidationMiddleware` | 代码工具执行前语法校验 |
| 7 | `EmptyToolCallRetryMiddleware` | 空工具调用自动重试 |

**功能性中间件（10 层）**：
| # | 中间件 | 作用 |
|---|--------|------|
| 8 | `ToolResultCacheMiddleware` | 工具结果缓存 + SSE 命中事件 |
| 9 | `FileOperationMiddleware` | 文件读写 SSE 事件发射 |
| 10 | `MultimodalMiddleware` | 图片/PDF → 视觉内容块转换 |
| 11 | `TodoWriteMiddleware` | Todo 状态持久化 |
| 12 | `LargeResultEvictionMiddleware` | 过大结果驱逐防 context 溢出 |
| 13 | `SummarizationMiddleware` | token 超限摘要 (120K 阈值) |
| 14 | `SkillsMiddleware` | 动态技能注册 + 工具锁定 |
| 15 | `PlanModeMiddleware` | `SubmitPlan` → HITL 中断 |
| 16 | `AskUserMiddleware` | `AskUserQuestion` → interrupt() |
| 17 | `ToolCallCounterMiddleware` | 工具调用计数/上限 |

**上下文注入（3 层, main-only）**：
| # | 中间件 | 作用 |
|---|--------|------|
| 18 | `SteeringMiddleware` | Redis 轮询 → 注入 `[Steering from User]` |
| 19 | `WorkspaceContextMiddleware` | agent.md → 系统 prompt 最后 content block |
| 20 | `RuntimeContextMiddleware` | `<time_awareness>` + `<user_profile>` (缓存断点之后) |

**后台子代理（4 层）**：
| # | 中间件 | 作用 |
|---|--------|------|
| 21 | `BackgroundSubagentMiddleware` | Task 工具 → 后台 asyncio.Task，立即返回任务 ID |
| 22 | `BackgroundSubagentOrchestrator` | 任务生命周期管理 + 状态机 + 结果收集 |
| 23 | `SubAgentMiddleware` | 注入 Task 工具 (init/update/resume) + 编译子 Agent |
| 24 | `SubagentSteeringMiddleware` | 子 Agent 专用 Redis steering |

**工具执行管道**：
```
LLM tool_call
  → [1] 参数解析 → [5] 路径保护 → [6] 语法校验 → [8] 缓存检查
  → 实际执行
  → [3] 结果规范化 → [4] 泄露检测 → [12] 大结果驱逐 → [2] 异常包装
  → 返回 LLM
```

**主 Agent vs 子 Agent 中间件差异**：
| 层 | 主 Agent | 子 Agent |
|----|---------|---------|
| Steering | ✅ (Redis `workflow:steering:`) | SubagentSteering (Redis `subagent:steering:`) |
| Workspace/Runtime Context | ✅ | ❌ |
| Background/SubAgent | ✅ | ❌ (防无限递归) |
| 工具子栈 7 层 | ✅ | ✅ (显式传入) |

**对 CryptoTrader 的启示**：
- `LeakDetectionMiddleware` — 检测 API key 泄露到 LLM 响应中，交易系统必须有
- `LargeResultEvictionMiddleware` — 防止大量市场数据撑爆 context
- `ToolResultCacheMiddleware` — 技术指标数据缓存，同 prompt 不重复计算
- `ToolCallCounterMiddleware` — 防止 Agent 循环调用（交易系统成本敏感）

---

### 10.3 SSE 事件类型谱 (15 种) — 扩展模式 1 的事件缓冲机制

| 事件类型 | 含义 | 载荷关键字段 |
|----------|------|-------------|
| `reasoning_signal` | 推理开始/结束 | `content: 'start' \| 'complete'` |
| `reasoning_content` | 推理内容流式块 | `content: string` |
| `message_chunk` | 正文文本流式块 | `content, finish_reason?` |
| `tool_calls` | 完整工具调用声明 | `id, name, args` |
| `tool_call_chunks` | 工具调用参数预告 (LLM 还在生成) | `id, name, partial_args` |
| `tool_call_result` | 工具执行结果 | `content, content_type` |
| `artifact` | 结构化产物 | `artifact_type: todo_update \| html_widget` |
| `interrupt` | HITL 中断 | `action_requests[]` (plan/question/workspace) |
| `user_message` | 用户消息 (仅回放) | `content, timestamp` |
| `workflow_status` | 工作流状态变更 | `status` |
| `thread_created` | 新线程 ID | `thread_id` |
| `error` | 错误 | `error_type, message` |
| `steering_delivered` | Steering 已注入 | — |
| `task_steering_accepted` | 子 Agent 接受 follow-up | `task_id` |
| `finish` | 流结束 | `finish_reason` |

**ContentSegment 模型** — 有序判别联合：
- `TextSegment` — 每个 `message_chunk` 追加，`content` 直接嵌入
- `ReasoningSegment` — 引用 `reasoningId`，实体存在 `reasoningProcesses` map
- `ToolCallSegment` — 引用 `toolCallId`，实体存在 `toolCallProcesses` map (request + result)
- 渲染层按 `order` 排序所有 segment 混合展示

**三阶段重连协议**：
```
1. watchThread() 监听 Redis pub/sub → 发现 workflow_started → 关闭 watch
2. getWorkflowStatus() 查询 GET /threads/{id}/status → can_reconnect?
3. reconnectToWorkflowStream() → GET /messages/stream?last_event_id=N
   → 服务端回放 last_event_id 之后的缓冲事件 → 接入实时流
```

**Event ID 机制**：每个 SSE 事件带整数 `id: N`（`_eventId`），客户端记录最后收到的 ID，重连时作为 `last_event_id` 传递，服务端据此去重。

**对 CryptoTrader 的启示**：
- CryptoTrader 当前 SSE 只有 3-4 种事件类型，应扩展为类似的结构化分类
- `reasoning_signal` + `reasoning_content` 让用户看到 Agent "思考过程"
- `tool_call_result` 让用户看到每个工具的返回值（如技术指标数据）
- Event ID 重连机制几乎可以 1:1 移植

---

### 10.4 数据库 Schema (23 张表) — 扩展模式 9 的 Automation 表结构

**应用层 17 张表**：

```
users ──┬── workspaces ──── workspace_files
        │                 ├── workspace_vault_secrets
        │                 └── conversation_threads ──┬── conversation_queries
        │                                            ├── conversation_responses (含 sse_events JSONB)
        │                                            └── conversation_usages (无 FK, 独立保留)
        ├── user_preferences (JSONB: risk/investment/agent preference)
        ├── user_api_keys (BYTEA 加密)
        ├── user_oauth_tokens (BYTEA 加密)
        ├── watchlists ──── watchlist_items
        ├── user_portfolios
        └── market_insights

automations ──── automation_executions
```

**LangGraph 基础设施 6 张表**：`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`, `store` (JSONB KV, TTL), `store_migrations`

**关键设计模式**：
1. **JSONB 广泛使用**：preferences / alert_settings / trigger_config / sse_events / token_usage / artifacts / config — 扩展字段全走 JSONB
2. **加密存储**：api_key / access_token / refresh_token / vault secrets 全用 BYTEA (应用层加密)
3. **无软删除**：ON DELETE CASCADE 硬删；workspace 用 status='deleted' 逻辑删除
4. **sse_events JSONB**：完整流式事件序列归档入库，支持回放和 debug
5. **conversation_usages 故意无 FK**：response 删除后账单记录独立保留
6. **Automation 三触发器**：cron (croniter) / once (next_run_at=NULL) / price (JSONB trigger_config)
7. **Partial Index**：`market_insights WHERE status='completed' AND user_id IS NULL` vs `IS NOT NULL` 分离全局和个人

**对 CryptoTrader 的启示**：
- `sse_events JSONB` 归档模式值得采用 — 将分析过程完整存储用于复盘
- Automation 表结构可直接参考 — trigger_type / cron_expression / trigger_config / next_run_at / server_id 锁
- `conversation_usages` 无 FK 模式 — token 消耗独立记账
- Partial Index 模式 — 按状态分离索引路径

---

### 10.5 子 Agent 编排完整机制

**定义 → 注册 → 编译 → 执行 → 结果回流** 五步：

**Step 1: 定义** (`SubagentDefinition` dataclass)
```python
SubagentDefinition(
    name="research",
    mode="ptc",                  # ptc 或 flash
    tools=["web_search", "filesystem"],  # 工具集标识符
    role_prompt="You are a research agent...",
    max_iterations=15,
    stateful=True,               # 是否需要 sandbox + MCP
)
```

**Step 2: 注册** (`SubagentRegistry`, 两层加载)
1. 内置 5 个: `research`, `general-purpose`, `data-prep`, `equity-analyst`, `report-builder`
2. YAML 用户覆盖: `agent_config.yaml` → `SubagentConfig` → 同名覆盖内置

**Step 3: 编译** (`SubagentCompiler`, 四级 prompt 优先级)
1. `custom_prompt` — 原始字符串
2. `custom_prompt_template` — 独立 Jinja2
3. `role_prompt` + `subagent_base.md.j2` — 模板 + 角色注入
4. 默认 base + mode defaults

工具隔离在此步完成：
```python
subagent_tool_sets = {
    "execute_code": [execute_code_tool],
    "filesystem": [read_file, write_file, edit_file, glob, grep],
    "web_search": [web_search_tool, web_fetch_tool],
    "finance": [get_sec_filing, get_stock_daily_prices, ...],
}
# research 子 Agent 只得到 ["web_search", "filesystem"]
```

**Step 4: 执行** (`BackgroundSubagentMiddleware`)
- 拦截 `Task(action="init")` → `asyncio.create_task()` 后台执行
- 立即返回伪结果: "Background subagent deployed: Task-k7Xm2p"
- 主 Agent 继续不阻塞

**Step 5: 结果回流** (通知/等待室模式)
1. 主 Agent 一轮结束后，`check_and_get_notification()` 收集已完成任务
2. 注入 `HumanMessage`: "Your background tasks have completed: Task-k7Xm2p"
3. Agent 调用 `TaskOutput(task_id=...)` 读取缓存结果
4. 最多 `max_iterations=3` 轮

**孤儿恢复** (服务重启场景)：
`_hydrate_from_checkpoint()` 查询 LangGraph checkpointer `checkpoint_ns=f"task:{task_id}"`，重建最小 `BackgroundTask`，标记 `completed=True`，防止 `update/resume` 失败。

**对 CryptoTrader 的启示**：
- 工具集隔离模式直接可用 — TechnicalAgent 只给技术指标工具，SentimentAgent 只给新闻/社交工具
- 后台执行 + 结果回流模式 — 替代当前的 `asyncio.gather()` 全等待，允许快的 Agent 先返回
- YAML 用户覆盖 — 用户在 TOML 中定义自定义 Agent 覆盖默认

---

### 10.6 性能优化模式（Prompt Cache 之外）

**1. Session 缓存** (`SessionService` 单例)
- 按 `workspace_id` 键缓存 `Session` 对象
- 30 分钟空闲超时，每 5 分钟后台扫描清理
- 每 workspace 独立 `asyncio.Lock`（通过二级 `_lock_registry_mu` 保护注册表）
- 锁超时 60s → `RuntimeError`（防死锁）
- `needs_init` 在锁内判断，`session.initialize()` 在锁外执行（最小化锁持有时间）

**2. Sync Cooldown** (`WorkspaceManager`)
- `_SYNC_COOLDOWN_SECONDS = 30`：30s 内请求直接返回缓存 session
- 跳过 `ensure_sandbox_ready()` + `sync_sandbox_assets()` 两次 Daytona API 调用
- `_user_data_synced` set：用户数据每 workspace 只同步一次
- `_workspace_to_user` / `_user_to_workspaces` 双向映射精准失效

**3. SHA-256 Manifest Diffing** (`sync_sandbox_assets()`)
- 6 个模块各计算本地 SHA-256 版本哈希
- 三路并发: `asyncio.gather(prune, compute_manifest, read_remote_manifest)`
- 只上传 `local != remote` 的模块
- 上传阶段再次 `asyncio.gather` 所有独立模块
- 性能日志: `[ASSET_SYNC] total=Xms (manifest=Xms uploads=Xms)`

**4. Snapshot 版本控制** (`DaytonaProvider`)
- `SHA-256[:8]` 截断哈希命名: `ptc-base-{hash}`
- 命中已有 snapshot → 直接复用；miss → 重建
- `sandbox_config_hash` JSONB 持久化进 workspace DB
- `_maybe_migrate_sandbox()`: 哈希匹配 → fast path 跳过；不匹配 → backup-destroy-restore

**5. asyncio.gather 并行启动**
- mint_token + manifest sync + vault secrets 同时推送
- MCP 文件 + 内部包 + tokens 首次上传并行
- 三路并发 Step 0/1/2 + 两路并发 Step 5/6

**对 CryptoTrader 的启示**：
- `_SYNC_COOLDOWN_SECONDS` 模式 → 数据层缓存（30s 内不重复拉 Binance API）
- Per-resource `asyncio.Lock` → 替代全局锁（不同币对不互相阻塞）
- `asyncio.gather` 并行数据加载 → 技术指标 + 链上数据 + 新闻同时拉取
- 性能日志模式 → 每个 node 记录耗时，识别瓶颈

---

## 十一、CryptoTrader vs LangAlpha 终极对比矩阵

| 维度 | LangAlpha | CryptoTrader | 差距评级 |
|------|-----------|-------------|----------|
| **图拓扑** | 单 ReAct 节点 + 中间件 | 多节点确定性 DAG | 不同范式，各有优势 |
| **子 Agent** | Registry + Compiler + 后台执行 | 4 个硬编码 + gather | ★★★★ 大 |
| **工具隔离** | 编译时按名称集隔离 | 无隔离，所有 Agent 共享 | ★★★ 中 |
| **SSE 事件** | 15 种结构化类型 | 3-4 种基本类型 | ★★★★ 大 |
| **重连** | 三阶段 (watch → status → replay) | 无 | ★★★★★ 极大 |
| **Live Steering** | Redis 注入 + before_model | 无 | ★★★★★ 极大 |
| **HITL** | 3 种中断 + 审批 UI | 无 | ★★★★ 大 |
| **Soft Interrupt** | ESC + checkpoint flush + 子 Agent 继续 | 无 | ★★★★ 大 |
| **中间件** | 24 层可组合 | node-level wrappers | ★★★ 中 |
| **模型韧性** | retry(3×) → fallback chain + manifest | `.with_fallbacks()` 单级 | ★★★★ 大 |
| **Prompt Cache** | 静态/动态分离 + breakpoint | SQLiteCache 全响应缓存 | ★★★ 中 |
| **自动摘要** | 两级 (截断 + LLM 摘要 120K) | 无 | ★★★★ 大 |
| **Session 缓存** | 30min idle + per-workspace lock | 无 | ★★★ 中 |
| **DB Schema** | 23 表 + JSONB + 加密 + partial index | SQLite + Postgres 混合 | ★★★ 中 |
| **Checkpoint** | AsyncPostgresSaver + 命名空间隔离 | 可选，非主要 | ★★★ 中 |
| **Automation** | cron + once + price + Agent self-schedule | APScheduler interval + cron | ★★★★ 大 |
| **安全** | 泄露检测 + 路径保护 + SSRF guard | 基本 | ★★★ 中 |
| **图表 → LLM** | 截图 + 文字描述 → multimodal | 无 | ★★★★★ 极大 |
| **实时数据** | WS-first + REST fallback + 自适应轮询 | 固定 10s 轮询 | ★★★★ 大 |
| **错误处理** | 结构化 detail + typed SSE error | 基本 HTTPException | ★★★ 中 |

---

## 十二、最终行动建议

### 必须做 (Tier 0) — 不做则系统存在硬伤

1. **价格触发器** — 无此能力的交易系统是不完整的
2. **模型韧性** (retry + fallback chain) — LLM 不可用 = 错过交易
3. **图表截图 → LLM** — 从"看数据"到"看图表"的质变

### 应该做 (Tier 1) — 显著提升竞争力

4. **BackgroundTaskManager + 断线重连** — 30-60s 分析不因断连丢失
5. **Live Steering** — 分析中纠偏，人机协作质变
6. **SSE 事件扩展** (15 种) — 用户看到 Agent 思考过程
7. **HITL 大额交易审批** — 安全兜底
8. **WS 实时价格** — Dashboard 数据鲜活度

### 值得做 (Tier 2) — 系统进化为平台

9. **AgentRegistry + YAML 覆盖** — 用户自定义 Agent
10. **工具集隔离** — 每个 Agent 只能访问其负责的数据
11. **Skills 策略系统** — 交易策略可配置化
12. **MCP Server 数据层** — 生态兼容
13. **Auto-Summarization** — ChatAgent 长对话不溢出

### 顺手做 (Tier 3) — 低成本高回报

14. **Prompt Cache 优化** — 重排 system message，~30% 成本降低
15. **LeakDetectionMiddleware** — 防 API Key 泄露到 LLM 响应
16. **LargeResultEvictionMiddleware** — 防市场数据撑爆 context
17. **ToolCallCounterMiddleware** — 防 Agent 循环调用
18. **Per-resource asyncio.Lock** — 不同币对不互相阻塞
19. **性能日志** — 每个 node 记录耗时
