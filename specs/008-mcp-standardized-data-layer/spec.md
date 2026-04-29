# 功能规格说明：MCP 标准化数据层

**Feature Branch**: `008-mcp-standardized-data-layer`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与动机

CryptoTrader-AI 当前数据层由分散的硬编码 Python 模块构成：

- `data/providers/binance.py` — Binance Futures 衍生品数据（OI、多空比、清算）
- `data/providers/sosovalue.py` — ETF 资金流入 / 加密新闻
- `data/providers/rss_news.py` — RSS 聚合新闻（CoinDesk / CoinTelegraph / Decrypt）
- `data/providers/coinglass.py` — 持仓量、清算数据（付费 API）
- `data/providers/defillama.py` — DeFi TVL（免费）
- `data/providers/cryptoquant.py` — 交易所净流量（付费 API）
- `data/providers/whale_alert.py` — 大额转账预警（付费 API）
- `data/macro.py` — 宏观数据（FRED / CoinGecko / Alternative.me）
- `data/onchain.py` — 链上数据聚合
- `data/market.py` — 行情数据（ccxt，OHLCV + Ticker + 订单簿）

这些模块通过 `SnapshotAggregator` 并行调用，结果注入 `ArenaState`，最终以硬编码方式传入 LangGraph 节点 `nodes/data.py`。

**核心问题**：数据提供者与 Agent 逻辑紧耦合，新增数据源需修改多处 Python 代码；Agent 无法在运行时动态发现或选择最优数据源；整个数据层对 Claude Code / Cursor 等外部 AI 工具不可见、不可复用。

MCP（Model Context Protocol）是 Anthropic 主导的开放协议，可将数据和工具以标准化方式暴露给 LLM Agent。将数据层改造为 MCP Server，能使 Agent 在运行时动态发现工具、让外部 AI 生态（Claude Code、Cursor、Continue 等）直接消费相同数据，并为后续解耦奠定基础。

---

## 用户场景与验收测试

### 用户故事 1 — 交易 Agent 通过 MCP 工具获取衍生品数据 (Priority: P0)

**场景描述**：交易系统在分析 BTC/USDT 时，Agent 通过 MCP 工具调用获取 Binance 衍生品数据（OI、多空比、清算量），不直接依赖硬编码的 Python 函数。

**Why this priority**: 衍生品数据是 Agent 做多空判断的核心依据，P0 级别数据必须优先迁移，其他数据源以此为模板复制。

**Independent Test**: 在不修改任何 Agent 代码的前提下，替换底层数据提供者实现，Agent 输出应不变。

**验收场景**:

1. **Given** Binance MCP Server 已启动（stdio 传输），**When** Agent 通过 MCP 客户端调用 `get_binance_derivatives(symbol="BTC")` 工具，**Then** 返回包含 `open_interest`、`long_short_ratio`、`taker_buy_sell_ratio` 的结构化字典，格式与现有 `fetch_derivatives_binance()` 返回值一致。

2. **Given** Binance API 超时或返回错误，**When** MCP 工具被调用，**Then** 返回带默认零值的完整结构，不抛出异常，错误信息记录至 MCP Server 日志。

3. **Given** 同一 Agent 在 5 分钟内对同一 symbol 多次调用同一 MCP 工具，**When** TTL 缓存未过期，**Then** 实际外部 API 调用次数为 1 次（命中现有 SQLite 缓存）。

---

### 用户故事 2 — MCPRegistry 动态发现数据源 (Priority: P0)

**场景描述**：`SnapshotAggregator` 不再硬编码导入各 Provider 类，而是通过 `MCPRegistry` 查询当前可用的 MCP Server 列表，动态选择数据源。

**Why this priority**: 动态发现是 MCP 标准化的核心价值——无需修改 Agent 代码即可新增、替换或禁用数据源。

**Independent Test**: 在 Registry 中注册一个 mock MCP Server，替换真实的 Binance Server，`SnapshotAggregator` 应自动使用 mock 数据。

**验收场景**:

1. **Given** `MCPRegistry` 初始化时读取配置（TOML 或 JSON），**When** 调用 `registry.list_tools()` 方法，**Then** 返回当前已注册的全部 MCP 工具名列表（如 `["get_binance_derivatives", "get_fear_greed", "get_sosovalue_etf", ...]`）。

2. **Given** 某个 MCP Server 被标记为 `enabled=false`，**When** `SnapshotAggregator` 请求该类别的数据，**Then** 自动跳过该 Server 并使用对应的零值兜底，行为与现有 `enabled=false` provider 配置一致。

3. **Given** 新增一个第三方 MCP Server（如 Glassnode），**When** 仅修改配置文件并重启 Registry，**Then** Agent 在下一个交易周期自动发现并调用新工具，无需修改任何 Python 业务代码。

---

### 用户故事 3 — 兼容 Claude Code / Cursor 外部 AI 工具 (Priority: P1)

**场景描述**：开发者在 Claude Code 或 Cursor 中使用 MCP 客户端，能直接调用 CryptoTrader-AI 的数据工具查询实时市场数据，用于调试或临时分析。

**Why this priority**: 外部 AI 生态兼容性是 MCP 的关键差异化价值，但不影响核心交易流程，定为 P1。

**Independent Test**: 在 Claude Code 中通过 `.mcp.json` 配置连接 MCP Server，执行工具调用并验证返回格式。

**验收场景**:

1. **Given** 项目根目录存在 `.mcp.json` 配置文件，**When** Claude Code 或 Cursor 通过 MCP 客户端连接，**Then** 可发现并成功调用至少以下工具：`get_binance_derivatives`、`get_fear_greed`、`get_rss_news`、`get_sosovalue_etf`。

2. **Given** 外部 AI 工具调用 MCP Server，**When** 当前 SQLite 缓存中存在有效数据，**Then** 直接返回缓存数据，不发起外部 API 请求（保护 API 配额）。

3. **Given** MCP Server 以 HTTP/SSE 传输运行，**When** 外部客户端并发调用多个工具，**Then** 工具调用互不阻塞，响应时间 p95 < 2000ms（缓存命中）或 < 10000ms（外部 API）。

---

### 用户故事 4 — 渐进式迁移：现有数据路径不中断 (Priority: P0)

**场景描述**：MCP 迁移分阶段进行。第一阶段仅将现有 Provider 函数包装为 MCP 工具，Agent 内部通过适配器调用 MCP，而非直接导入 Provider 模块。第二阶段才解耦 Provider 实现。整个过程中，现有回测和实盘路径保持完全可用。

**Why this priority**: 渐进式迁移是避免引入回归缺陷的关键约束，尤其是对回测引擎（`use_llm=True` 模式）影响不可接受。

**Independent Test**: 在迁移任意单个 Provider 后，运行完整回测套件（`pytest tests/`），所有测试保持通过。

**验收场景**:

1. **Given** Binance Provider 已完成 MCP 包装，**When** 执行 `SnapshotAggregator.collect()` 路径，**Then** 数据流经 MCP 适配器后与原始调用结果在业务字段上等价（允许非关键字段缺失）。

2. **Given** MCP Server 未启动（进程不存在），**When** `MCPRegistry` 请求对应工具，**Then** 自动降级到直接 Python 调用（fallback 路径），交易流程不中断，日志记录 `[MCP fallback]` 警告。

3. **Given** 回测模式 (`backtest_mode=True`) 激活，**When** `collect_snapshot` 节点运行，**Then** 不通过 MCP 网络调用获取数据（防止回测模式下的延迟和不确定性），直接从 SQLite 历史数据读取，与迁移前行为完全一致。

---

### 用户故事 5 — OKX / 宏观 / 链上 MCP Server (Priority: P1)

**场景描述**：在 Binance MCP Server 验证成功后，将 OKX 实时行情、宏观数据（FRED / Fear & Greed）、链上数据（DefiLlama / CryptoQuant / WhaleAlert）分别包装为独立 MCP Server。

**Why this priority**: P1 级别，依赖 P0 MCP 基础设施验证完成后再推进。

**Independent Test**: 对每个新 MCP Server 独立运行集成测试，验证工具发现、调用、缓存命中三个路径。

**验收场景**:

1. **Given** 宏观 MCP Server 已注册，**When** `MacroCollector` 通过适配器调用 `get_fear_greed` 工具，**Then** 返回 `{"value": int, "classification": str}` 格式数据，与现有 `MacroData.fear_greed_index` 语义一致。

2. **Given** 链上 MCP Server 集成 DefiLlama，**When** 调用 `get_defi_tvl(chain="Ethereum")` 工具，**Then** 返回 `{"defi_tvl": float, "defi_tvl_change_7d": float}` 结构。

3. **Given** 付费 API（CoinGlass / CryptoQuant / WhaleAlert）的 API Key 未配置，**When** 对应 MCP 工具被调用，**Then** 返回对应的零值兜底结构，且在 MCP 工具响应元数据中标记 `"data_available": false`。

---

### 边界条件

- **缓存一致性**：MCP 工具调用必须复用现有 `store.py` 的 SQLite 缓存和 TTL 限速逻辑（`_RATE_LIMITS` 表），不引入第二套缓存机制。
- **回测隔离**：回测模式下，MCP 适配器不得发起任何网络请求，必须完全依赖 SQLite 历史数据（防止 look-ahead bias）。
- **stdio 传输并发**：stdio 传输的 MCP Server 为单进程，需支持 asyncio 并发处理同一进程内的多个工具调用请求。
- **API Key 泄漏防护**：MCP Server 工具描述和返回值中不得包含原始 API Key 字符串。
- **工具名冲突**：不同 MCP Server 注册的工具名必须全局唯一，命名规范为 `{source}_{data_type}`（如 `binance_derivatives`、`sosovalue_etf_metrics`）。
- **大数据量截断**：单次 MCP 工具调用返回的数据量不超过 50KB，超过时截断并在响应中标注 `"truncated": true`。

---

## 功能需求

### 核心实体

- **MCPServer**：一个独立的 FastMCP 服务实例，暴露一组相关工具（如 BinanceMCPServer、MacroMCPServer）。可通过 stdio 传输（同进程内嵌）或 HTTP/SSE 传输（独立进程）运行。
- **MCPTool**：MCP Server 中定义的单个可调用工具，对应现有 Provider 中的一个异步函数。每个工具有唯一名称、输入 schema（Pydantic 验证）和返回 schema。
- **MCPRegistry**：工具注册表，维护 `server_name → MCPServer 连接` 的映射，支持工具发现、路由和健康检查。
- **MCPAdapter**：现有 `SnapshotAggregator` 调用 MCP 工具的适配层，负责请求转发、fallback 到直接 Python 调用、以及回测模式短路。
- **MCPConfig**：TOML 配置项（嵌入现有 `config.py`），描述每个 MCP Server 的传输类型、启用状态和工具列表。

### 功能需求列表

**FR-001**：项目提供 `BinanceMCPServer`，暴露以下 3 个工具：
- 衍生品数据查询（OI、多空比、清算量）
- 资金费率查询
- K 线数据查询（支持指定时间周期和数量）

**FR-002**：项目提供 `MacroMCPServer`，暴露以下 4 个工具：
- Fear & Greed 指数查询
- BTC 市值占比查询
- FRED 宏观经济序列查询（支持指定序列 ID）
- ETF 资金流查询（支持指定 ETF 类型）

**FR-003**：项目提供 `OnchainMCPServer`，暴露以下 4 个工具：
- DeFi TVL 查询（支持指定链）
- 衍生品持仓量和清算量查询
- 交易所净流量查询
- 大额转账监控查询

**FR-004**：项目提供 `NewsMCPServer`，暴露以下 2 个工具：
- RSS 新闻聚合查询（支持限制每个来源的条目数）
- SoSoValue 特色新闻查询（支持分页）

**FR-005**：所有 MCP 工具必须复用现有数据缓存层（SQLite 缓存 + 限速规则），不绕过缓存机制。

**FR-006**：`MCPRegistry` 提供工具列表查询、工具调用路由、Server 健康检查三个核心能力。

**FR-007**：`MCPAdapter` 在以下情况自动降级到直接 Python 调用（fallback）：
  - MCPRegistry 中对应工具不可用
  - MCP 工具调用超时（默认超时 5 秒）
  - MCP Server 返回错误响应

**FR-008**：`MCPAdapter` 检测到 `backtest_mode=True` 时，绕过 MCP 调用，直接调用现有 Python 函数（与当前行为一致，防止 look-ahead bias）。

**FR-009**：`config.py` 新增 `MCPConfig` dataclass，对应 TOML `[mcp]` 节，支持以下配置项：
  - `enabled: bool`（全局开关，默认 `false`，迁移期渐进启用）
  - `transport: str`（`"stdio"` 或 `"http"`，默认 `"stdio"`）
  - `servers: list[MCPServerConfig]`（每个 Server 的名称、传输地址、工具列表）
  - `fallback_on_error: bool`（默认 `true`，降级到直接调用）

**FR-010**：项目根目录提供 `.mcp.json` 文件（MCP 客户端发现入口），兼容 Claude Code / Cursor 的 MCP 配置格式，描述所有可用 Server 的连接方式。

**FR-011**：新增 `arena mcp list` CLI 命令，输出当前已注册的全部 MCP 工具列表及其健康状态。

**FR-012**：新增 `arena mcp call <tool_name> [--args '{}']` CLI 命令，允许开发者从命令行直接调用任意 MCP 工具，用于调试和验证。

**FR-013**：所有 MCP 工具的返回 schema 与现有 Provider 函数的返回类型保持向后兼容，Agent 侧无需修改数据使用代码。

**FR-014**：`SnapshotAggregator` 在 `mcp.enabled=true` 时通过 `MCPAdapter` 调用数据工具；在 `mcp.enabled=false`（默认）时保持现有直接调用路径，两条路径的外部行为一致。

**FR-015**：MCP Server 实现不引入新的外部依赖，仅新增 `fastmcp` 包（通过 `pyproject.toml` 声明）。

---

## 成功标准

### 可量化的验收指标

**SC-001**：全部 4 个 MCP Server（Binance / Macro / Onchain / News）成功注册，`arena mcp list` 命令输出的工具数量 ≥ 13 个。

**SC-002**：在 `mcp.enabled=true` 模式下，现有全套测试套件（`pytest tests/`）通过率保持 100%，0 个新增回归失败。

**SC-003**：MCP 工具的缓存命中响应时间 p95 < 50ms（同进程 stdio 传输，SQLite 缓存命中），对比现有直接调用的 p95 < 30ms，性能劣化不超过 100%。

**SC-004**：在 `mcp.enabled=false`（默认）时，所有现有代码路径行为与引入 MCP 前完全一致，零改动量。

**SC-005**：Claude Code 通过 `.mcp.json` 配置成功连接 MCP Server，并能调用至少 4 个工具（每个 Server 各一个），无需额外安装或配置步骤。

**SC-006**：`MCPAdapter` 的 fallback 机制经测试验证：当 MCP Server 不可用时，`SnapshotAggregator` 自动降级，回测和实盘的快照采集功能不中断。

**SC-007**：回测模式下，通过断言验证零次 MCP 网络调用（`MCPAdapter` 被短路），所有数据来自 SQLite 本地存储。

**SC-008**：代码覆盖率：MCP Server 工具层（仅包装函数）在 `pyproject.toml` 的 `coverage omit` 列表中（与现有 provider 一致），其余 `MCPRegistry` / `MCPAdapter` / `MCPConfig` 代码覆盖率 ≥ 80%。

---

## 迁移策略与阶段规划

本 spec 描述"做什么"，不描述"怎么做"。以下迁移阶段仅作为范围边界参考：

**阶段 A（包装）**：将现有 Provider 函数一对一包装为 MCP 工具，不修改 Provider 内部实现，不改变 `SnapshotAggregator` 调用路径。

**阶段 B（接入）**：`SnapshotAggregator` 中引入 `MCPAdapter`，在 `mcp.enabled=true` 时经由 MCP 调用数据，`mcp.enabled=false` 时保持原路径（默认关闭）。

**阶段 C（外部兼容）**：提供 `.mcp.json` 配置，验证 Claude Code / Cursor 连接，添加 HTTP/SSE 传输选项支持跨进程调用。

**阶段 D（解耦）**：（超出本 spec 范围）逐步将 Provider 实现迁出主进程，作为独立 MCP Server 部署。

---

## 假设

- 项目 Python 运行时版本 ≥ 3.12（与现有要求一致），FastMCP 库与 asyncio 兼容。
- `fastmcp` 包在 PyPI 上可用且提供稳定 API，版本锁定在 `pyproject.toml`。
- stdio 传输模式下，MCP Server 与 Agent 在同一进程内运行，不存在跨进程序列化开销。
- 现有 SQLite 缓存（`~/.cryptotrader/market_data.db`）在 MCP 工具调用路径中保持可访问。
- `config.py` 的 `load_config()` 缓存机制已存在，`MCPConfig` 作为其中一个子配置 dataclass 加入即可。
- 本 spec 不覆盖 MCP 身份认证（API Key 权限控制）——外部调用安全由部署环境（Docker 网络隔离）负责。
- 不移除或重构现有 Provider 模块，第一阶段只新增包装层。
