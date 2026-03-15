# 需求文档

## 项目描述（输入）

审查优化整个项目架构和规范，保证符合最佳实践和原则

## 简介

本规格定义了对 CryptoTrader AI 多智能体加密货币交易系统的全面架构审查与优化目标。系统以 LangGraph `StateGraph` 为骨架，串联数据采集、多 Agent 并行分析、辩论门控、AI 裁决、风控门、订单执行等核心流程，并集成经验记忆（GSSC 引擎）、APScheduler 定时调度、FastAPI 服务层与 Streamlit 仪表盘。

审查范围覆盖：代码架构与模块边界、错误处理一致性、配置管理、测试覆盖率、异步并发安全、交易所抽象与组合管理、安全实践（密钥/API Key/输入校验）、性能优化（LLM 成本与延迟）、可观测性（日志/追踪/指标）以及部署与基础设施（Docker、CI/CD）。目标是确保各层均符合最佳实践与领域驱动分层原则，同时保持零 lint 错误、零静默异常、配置文件单一来源等硬性要求不退化。

---

## 需求

### 需求 1：代码架构与模块边界

**目标**：作为系统架构师，我希望各模块职责清晰、依赖方向单向可控，以便在不影响其他层的前提下独立修改或替换任意领域模块。

#### 验收标准

1. The CryptoTrader AI 系统 shall 保持 `nodes/` 层对领域层（`agents/`、`debate/`、`execution/`、`risk/`、`learning/`）的单向依赖，领域层不得反向导入 `nodes/` 或 `graph.py` 中的任何符号。
2. The CryptoTrader AI 系统 shall 确保 `graph.py` 仅包含拓扑声明（`add_node`、`add_edge`、`add_conditional_edges`），不包含任何业务逻辑或数据转换代码。
3. When 新增跨模块共享工具函数时，the CryptoTrader AI 系统 shall 将其下沉到 `db.py`、`agents/base.py` 或 `state.py` 等共享基础设施文件，而非在调用方重复实现。
4. The CryptoTrader AI 系统 shall 保证 `ArenaState` 是所有 LangGraph 节点之间唯一的数据契约，禁止在节点间通过全局变量或模块级可变状态传递数据。
5. If 检测到循环导入（circular import），the CryptoTrader AI 系统 shall 通过重构模块边界或使用 `TYPE_CHECKING` 保护块消除该循环，而非使用延迟导入绕过。
6. The CryptoTrader AI 系统 shall 对所有替代性代码路径（如 `graph_supervisor.py`、`langchain_agents.py`）明确标注其状态（实验性/未启用），并在文档中说明与主路径的关系。

---

### 需求 2：错误处理一致性

**目标**：作为运维工程师，我希望系统在任何异常场景下均能产生结构化日志并执行安全降级，以便快速定位问题根因而不丢失上下文。

#### 验收标准

1. The CryptoTrader AI 系统 shall 确保所有 `except Exception` 块均携带 `logger.warning(exc_info=True)` 或 `logger.debug(exc_info=True)`，禁止存在静默吞异常的空 `except` 块。
2. When LLM 调用因网络超时或 API 限速失败时，the CryptoTrader AI 系统 shall 触发 fallback LLM 链（`.with_fallbacks()`），并在日志中记录原始异常和 fallback 触发原因。
3. If 交易所 API 返回错误状态码或抛出 CCXT 异常，the CryptoTrader AI 系统 shall 在 `OrderManager` 或 `LiveExchange` 层捕获并以结构化字段（`exchange_id`、`symbol`、`error_code`、`retry_count`）记录，而非将原始异常向上层传播。
4. While 风控门（Risk Gate）执行检查期间，the CryptoTrader AI 系统 shall 对每项独立检查的异常单独捕获，确保单项检查失败不会中断其他检查项的执行。
5. If Redis 连接不可用且已在配置中启用 Redis，the CryptoTrader AI 系统 shall 保守拒绝交易并在日志中记录 `redis_unavailable` 事件，而非静默跳过风控状态检查。
6. The CryptoTrader AI 系统 shall 为所有后台异步任务（`loop.create_task()` 触发的协程）添加异常回调或 `try/except` 包裹，防止后台任务静默失败。
7. When APScheduler 调度任务抛出未捕获异常时，the CryptoTrader AI 系统 shall 记录完整堆栈并继续执行后续调度周期，不得因单次任务失败导致调度器停止。

---

### 需求 3：配置管理最佳实践

**目标**：作为系统管理员，我希望所有可调参数均集中于 `config/default.toml`，以便在不修改代码的情况下调整系统行为。

#### 验收标准

1. The CryptoTrader AI 系统 shall 确保源码中不存在任何硬编码的数值阈值、模型名称字符串或外部服务地址，所有此类值均须通过 `load_config()` 从 `config/default.toml` 获取。
2. The CryptoTrader AI 系统 shall 保证 `load_config()` 在进程生命周期内仅初始化一次（单例缓存），后续调用直接返回缓存对象，不重复解析 TOML 文件。
3. When 配置文件中某个必填字段缺失或类型不匹配时，the CryptoTrader AI 系统 shall 在启动阶段（`load_config()` 调用时）抛出明确的 `ConfigurationError`，包含字段路径和期望类型信息。
4. If 用户未提供 `[exchanges.*]` 凭证配置，the CryptoTrader AI 系统 shall 在 `arena live-check` 时输出清晰的缺失凭证提示，而非在运行时抛出 `KeyError`。
5. The CryptoTrader AI 系统 shall 确保所有 Pydantic 配置 dataclass 使用 `model_validator` 或 `field_validator` 对关键约束（如阈值范围、非空字符串）进行声明式校验，而非在运行时散落检查。
6. Where 用户通过环境变量覆盖配置时，the CryptoTrader AI 系统 shall 支持通过 `CRYPTOTRADER_*` 前缀环境变量覆盖对应 TOML 字段，且优先级高于文件配置。

---

### 需求 4：测试覆盖率与测试模式

**目标**：作为开发工程师，我希望核心业务逻辑均有对应的自动化测试，以便在重构时快速验证行为不变性。

#### 验收标准

1. The CryptoTrader AI 系统 shall 对 `nodes/`、`agents/`、`debate/`、`risk/`、`learning/` 下的每个公开函数至少提供一个 pytest 单元测试，覆盖正常路径和主要异常路径。
2. When 测试需要调用 LLM 时，the CryptoTrader AI 系统 shall 使用 `patch("langchain_openai.ChatOpenAI.ainvoke", return_value=AIMessage(...))` mock LLM 调用，禁止在单元测试中产生真实 API 请求。
3. The CryptoTrader AI 系统 shall 为 `build_trading_graph()`、`build_lite_graph()`、`build_debate_graph()` 提供集成测试，验证图的拓扑结构（节点名称、条件边路由）符合设计文档。
4. If 测试文件与被测模块不对应（即缺少 `test_{module}.py`），the CryptoTrader AI 系统 shall 在 CI 检查中产生警告，提示覆盖率缺口。
5. The CryptoTrader AI 系统 shall 在 `pyproject.toml` 中配置 `pytest-cov` 并设置最低分支覆盖率阈值（不低于 70%），CI 流水线在覆盖率低于阈值时失败。
6. While 执行回测集成测试时，the CryptoTrader AI 系统 shall mock 外部数据源（CCXT、HTTP 接口）并使用 SQLite 内存数据库，确保测试可在无网络环境下完整运行。
7. The CryptoTrader AI 系统 shall 确保所有异步测试函数使用 `asyncio_mode = "auto"` 配置，无需在每个测试函数上手动添加 `@pytest.mark.asyncio` 装饰器。

---

### 需求 5：异步模式与并发安全

**目标**：作为系统工程师，我希望所有异步并发路径均无竞态条件和死锁风险，以便系统在高并发场景下稳定运行。

#### 验收标准

1. The CryptoTrader AI 系统 shall 确保所有并行 Agent 调用和辩论轮次使用 `asyncio.gather(*tasks, return_exceptions=True)` 实现，并对 `return_exceptions=True` 返回的异常逐项检查和记录。
2. When 后台反思任务（`loop.create_task()`）被创建时，the CryptoTrader AI 系统 shall 持有对 Task 对象的引用（存入集合或列表），防止 Task 对象被垃圾回收器提前回收导致任务静默终止。
3. The CryptoTrader AI 系统 shall 禁止在 `async def` 函数内部调用阻塞式 I/O 操作（如同步文件读写、`requests.get()`、`sqlite3.connect()` 的同步方法），此类操作须使用 `asyncio.to_thread()` 或异步等价库。
4. If `PaperExchange` 或 `OrderManager` 的内部状态被多个并发协程同时写入，the CryptoTrader AI 系统 shall 使用 `asyncio.Lock` 或不可变数据结构保护共享状态，确保线程安全。
5. The CryptoTrader AI 系统 shall 为所有 `asyncio.gather()` 调用设置合理的超时（通过 `asyncio.wait_for()` 包裹），防止单个 LLM 调用无限挂起导致整个流水线阻塞。
6. While APScheduler 调度器运行期间，the CryptoTrader AI 系统 shall 确保每次调度任务启动前检查上次任务是否仍在运行，避免重叠执行（misfire 策略设为 `'drop'` 或 `'queue'`）。

---

### 需求 6：交易所抽象与组合管理

**目标**：作为量化交易员，我希望 Paper 和 Live 两套执行引擎对上层调用透明，以便在不修改业务逻辑的情况下切换执行环境。

#### 验收标准

1. The CryptoTrader AI 系统 shall 确保 `PaperExchange` 和 `LiveExchange` 实现相同的抽象接口（`place_order`、`get_position`、`get_balance`、`cancel_order`），节点层通过接口调用而非具体类型调用。
2. When 切换 `config.engine` 从 `"paper"` 到 `"live"` 时，the CryptoTrader AI 系统 shall 无需修改 `nodes/execution.py` 或业务逻辑代码，仅通过配置文件变更即可完成引擎切换。
3. The CryptoTrader AI 系统 shall 以交易所实时查询结果（仓位、余额）为组合状态的权威来源，数据库仅存储历史快照（PnL、回撤、净值曲线），禁止以数据库记录覆盖实时交易所数据。
4. If `LiveExchange` 在执行下单后未能获取订单确认，the CryptoTrader AI 系统 shall 通过轮询（带退避重试）确认订单状态，并在超过重试上限后发出告警通知。
5. The CryptoTrader AI 系统 shall 为 `OrderManager` 的订单状态机定义明确的状态转换（`pending` → `open` → `filled`/`cancelled`/`failed`），禁止在状态机之外直接修改订单状态字段。
6. Where 用户配置了多个交易所（`[exchanges.binance]`、`[exchanges.okx]`），the CryptoTrader AI 系统 shall 支持在同一调度周期内对不同交易对选择不同交易所执行，交易所选择逻辑集中在 `_get_exchange()` 工厂函数中。

---

### 需求 7：安全实践

**目标**：作为安全工程师，我希望系统对密钥、API Key 和外部输入实施严格保护，以便防止凭证泄露和注入攻击。

#### 验收标准

1. The CryptoTrader AI 系统 shall 确保交易所 API Key、Secret 和 Passphrase 仅从 `config/default.toml` 的 `[exchanges.*]` 节读取，禁止硬编码于源码或以明文写入 Git 仓库。
2. The CryptoTrader AI 系统 shall 在 pre-commit hook 中运行 `detect-secrets` 和 `detect-private-key` 检查，阻止含有高熵字符串或私钥格式内容的提交进入版本库。
3. When 处理来自外部数据源（新闻 API、链上数据 API）的响应时，the CryptoTrader AI 系统 shall 对响应数据进行 schema 校验（Pydantic 模型或等价校验），拒绝不符合预期结构的响应并记录告警。
4. If API 路由（FastAPI）接收到格式异常或超长的请求体，the CryptoTrader AI 系统 shall 返回 `422 Unprocessable Entity` 并记录请求摘要（不含敏感字段），不得将原始异常信息暴露给调用方。
5. The CryptoTrader AI 系统 shall 对所有向 LLM 发送的 prompt 中包含的用户输入或外部数据进行长度限制和特殊字符过滤，防止 prompt 注入攻击。
6. Where 系统部署于生产环境时，the CryptoTrader AI 系统 shall 禁用 FastAPI 的 `/docs` 和 `/redoc` 端点（或通过认证保护），防止 API schema 信息泄露。
7. The CryptoTrader AI 系统 shall 确保 `verify=False` 的 HTTPS 调用仅出现在已知第三方数据源的适配器层，并在代码注释中标注跳过验证的业务原因及对应 issue 追踪编号。

---

### 需求 8：性能优化（LLM 成本与延迟）

**目标**：作为产品负责人，我希望系统在保证决策质量的前提下尽量减少 LLM 调用次数和 API 费用，以便控制运营成本。

#### 验收标准

1. The CryptoTrader AI 系统 shall 在高共识场景（`consensus_strength >= consensus_skip_threshold`）下通过辩论门控跳过辩论阶段，将 LLM 调用从最多 13 次降低至 4-5 次。
2. When 裁决节点检测到持平仓位且无电路熔断器激活时，the CryptoTrader AI 系统 shall 降级为加权裁决（0 次 LLM 调用），并在日志中记录 `verdict_downgraded_to_weighted` 事件。
3. The CryptoTrader AI 系统 shall 对所有 LLM 响应启用 `SQLiteCache`（位于 `~/.cryptotrader/llm_cache.db`），对相同 prompt hash 的重复调用直接返回缓存结果。
4. If Agent 分析的输入快照数据与上一周期完全相同，the CryptoTrader AI 系统 shall 复用上一周期的 Agent 分析结果，跳过重复 LLM 调用。
5. The CryptoTrader AI 系统 shall 对每次 LLM 调用记录 token 消耗（input tokens、output tokens、model name）到结构化日志，支持按时间窗口汇总成本报告。
6. While 回测模式运行时，the CryptoTrader AI 系统 shall 默认使用规则裁决而非 AI 裁决（`use_llm=False` 或 `backtest_mode=True`），并跳过经验记忆注入，以节省 LLM 调用成本。
7. The CryptoTrader AI 系统 shall 为所有 `asyncio.gather()` 并行 LLM 调用设置单次调用超时上限（可通过 `config.models.timeout_seconds` 配置），超时后以空结果降级而非阻塞整个流水线。

---

### 需求 9：可观测性（日志、追踪与指标）

**目标**：作为 SRE 工程师，我希望系统产生结构化、可查询的日志和追踪数据，以便在生产环境中快速诊断交易决策链路中的异常。

#### 验收标准

1. The CryptoTrader AI 系统 shall 使用 `structlog` 统一所有日志输出，日志条目包含固定字段：`timestamp`、`level`、`module`、`trade_id`（如适用）、`symbol`（如适用）。
2. When 一次完整交易流水线执行（从数据采集到订单执行）时，the CryptoTrader AI 系统 shall 为该次执行生成唯一的 `trace_id`，并将其传播至所有子节点的日志条目中。
3. The CryptoTrader AI 系统 shall 在每个 LangGraph 节点的入口和出口记录结构化日志，包含节点名称、耗时（毫秒）和关键输出摘要（如 Agent 评分、裁决结果、风控检查项状态）。
4. If 风控门拒绝交易，the CryptoTrader AI 系统 shall 在日志中记录每项失败检查的名称、当前值和阈值，以及最终拒绝原因摘要。
5. The CryptoTrader AI 系统 shall 暴露 `/metrics` 端点（Prometheus 格式），包含以下指标：LLM 调用次数（按模型分组）、辩论跳过率、裁决分布（buy/sell/hold）、风控拒绝率、交易执行延迟（P50/P95/P99）。
6. Where OpenTelemetry 已集成时，the CryptoTrader AI 系统 shall 将 LLM 调用 span 和 LangGraph 节点 span 导出至配置的 OTLP 端点，支持分布式追踪可视化。
7. The CryptoTrader AI 系统 shall 对 API 路由（FastAPI）的每个请求记录 `method`、`path`、`status_code`、`response_time_ms` 和 `client_ip`（已脱敏），支持请求级别的性能分析。

---

### 需求 10：部署与基础设施

**目标**：作为 DevOps 工程师，我希望系统通过 Docker Compose 一键部署所有服务，并具备完整的 CI/CD 流水线，以便实现可重复的生产部署。

#### 验收标准

1. The CryptoTrader AI 系统 shall 通过 Docker Compose 提供包含以下服务的完整部署配置：`api`（FastAPI）、`scheduler`（APScheduler）、`dashboard`（Streamlit）、`postgres`（可选）、`redis`（可选）。
2. When 构建 Docker 镜像时，the CryptoTrader AI 系统 shall 使用多阶段构建（multi-stage build）分离依赖安装阶段和运行阶段，确保生产镜像不包含开发依赖（`uv`、`ruff`、测试库）。
3. The CryptoTrader AI 系统 shall 在 CI 流水线中依次执行：`ruff check`（lint）、`ruff format --check`（格式验证）、`pytest`（测试 + 覆盖率）、`docker build`（镜像构建验证），任意步骤失败则阻断后续步骤。
4. If Docker 健康检查（`HEALTHCHECK`）失败，the CryptoTrader AI 系统 shall 通过 `/health` 端点返回详细的组件状态（数据库连通性、Redis 连通性、LLM API 可达性），Docker 编排器据此决定是否重启容器。
5. The CryptoTrader AI 系统 shall 在 `docker-compose.yml` 中为所有服务配置资源限制（`mem_limit`、`cpus`），防止单服务资源耗尽影响其他服务。
6. Where 生产环境需要持久化数据时，the CryptoTrader AI 系统 shall 将 SQLite 数据库文件（`~/.cryptotrader/`）和 PostgreSQL 数据目录挂载到命名卷（named volumes），而非容器内部文件系统。
7. The CryptoTrader AI 系统 shall 在 `pyproject.toml` 中维护 `[project.optional-dependencies]` 分组（如 `dev`、`test`），CI 环境仅安装 `test` 组依赖，生产镜像仅安装核心依赖。
