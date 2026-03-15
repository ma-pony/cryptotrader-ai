# 实施计划

- [x] 1. 强化配置管理层
- [x] 1.1 (P) 实现环境变量覆盖机制
  - 新增 `apply_env_overrides()` 函数，解析 `CRYPTOTRADER_*` 前缀环境变量并按双下划线路径合并至 TOML 数据字典
  - 覆盖优先级：环境变量 > `local.toml` > `default.toml`，在 `load_config()` 的 `local.toml` 合并之后、`_build_config()` 之前调用
  - 类型转换规则：`"true"/"false"` → bool，纯数字字符串 → int/float，其余保留 str；类型转换失败时记录 `logger.warning` 并跳过该键
  - _Requirements: 3.6_

- [x] 1.2 (P) 实现启动阶段声明式配置校验
  - 新建 `ConfigurationError(ValueError)` 异常类，包含 `field_path: str` 和 `expected: str` 属性
  - 实现 `validate_config()` 函数，对关键字段执行范围断言：`risk.loss.max_daily_loss_pct` ∈ (0,1)、`risk.position.max_single_pct` ∈ (0,1)、`debate.consensus_skip_threshold` ∈ (0,1)、`models.fallback` 非空
  - 在 `load_config()` 构建 `AppConfig` 后立即调用校验，不符合则以非零退出码中断启动
  - 确认 `config/default.toml` 中各默认值本身已满足约束范围，否则同步修正
  - _Requirements: 3.3, 3.5_

- [x] 1.3 (P) 完善缺失凭证检测与 ModelConfig 超时字段
  - 在 `ModelConfig` dataclass 中新增 `timeout_seconds: int = 60` 字段，同步更新 `config/default.toml`
  - 改造 `arena live-check` 命令，在遍历已配置交易所时检测 `api_key`/`secret` 是否为空，缺失时输出清晰提示信息而非运行时 `KeyError`
  - _Requirements: 3.4, 8.7_

- [x] 1.4 为新配置能力补充单元测试
  - 测试 `apply_env_overrides()`：路径解析（双下划线嵌套）、类型转换（bool/int/float/str）、优先级覆盖
  - 测试 `validate_config()`：越界值触发 `ConfigurationError` 并携带正确的 `field_path`
  - 端到端测试：设置 `CRYPTOTRADER_RISK__LOSS__MAX_DAILY_LOSS_PCT=0.01` 后调用 `load_config()` 验证字段值被覆盖
  - _Requirements: 3.3, 3.5, 3.6_

- [x] 2. 建立后台任务注册表与错误处理加固
- [x] 2.1 (P) 实现 TaskRegistry 防 GC 任务注册
  - 新建 `task_registry.py` 基础设施模块，提供模块级 `_background_tasks: set[asyncio.Task]` 单例
  - 实现 `add_background_task(coro, name)` 函数：调用 `loop.create_task()`、将 Task 存入集合、注册完成回调
  - 完成回调 `_on_task_done()` 负责从集合移除 Task；若 Task 有未捕获异常则 `logger.warning(exc_info=True)`
  - 将 `nodes/data.py` 中 `verbal_reinforcement()` 的 `loop.create_task(reflect_coro)` 替换为 `add_background_task()`
  - _Requirements: 2.6, 5.2_

- [x] 2.2 (P) 改造 RiskGate 使每项检查独立捕获异常
  - 重构 `risk/gate.py` 的 `check()` 方法，将 `await c.evaluate()` 包裹在独立 `try/except` 中
  - 单项检查异常时记录 `logger.warning(exc_info=True)` 并将该检查标记为 `check_error`（视为未通过），继续执行下一项
  - 最终门控结果仍为"任一失败则拒绝"，但保证所有检查项均有执行机会
  - _Requirements: 2.4_

- [x] 2.3 (P) 加固调度器单次任务异常隔离
  - 确认 `scheduler.py` 的 `_run_cycle()` 已有完整 `try/except` 包裹，若缺失则补全；确保异常被记录为 `logger.warning(exc_info=True)` 后调度器继续运行
  - 为 APScheduler `add_job()` 补充 `max_instances=1` 和 `misfire_grace_time=0` 参数，防止上次周期未完成时重叠触发
  - _Requirements: 2.7, 5.6_

- [x] 2.4 为后台任务和错误处理补充测试
  - 测试 `add_background_task()`：验证 Task 引用被持有、Task 完成后从集合移除、异常触发 `logger.warning`
  - 测试改造后的 `RiskGate.check()`：单项检查抛出异常时其余检查继续执行，最终结果为拒绝
  - _Requirements: 2.4, 2.6, 5.2_

- [x] 3. 实现 LLM 并行调用超时保护与 PaperExchange 并发安全
- [x] 3.1 (P) 为并行 LLM 调用添加 asyncio.wait_for 超时
  - 在 `nodes/agents.py` 和 `nodes/debate.py` 的 `asyncio.gather()` 调用中，用 `asyncio.wait_for(coro, timeout=cfg.models.timeout_seconds)` 包裹每个协程
  - 超时时记录 `logger.warning("LLM timeout")` 并将该 Agent/辩论方的结果降级为空/mock 结果，不阻塞整体 gather
  - 确保 `return_exceptions=True` 返回的所有异常均逐项检查并记录，不静默忽略
  - _Requirements: 5.1, 5.5, 8.7_

- [x] 3.2 (P) 为 PaperExchange 共享状态添加 asyncio.Lock 保护
  - 在 `PaperExchange.__init__()` 中创建 `self._lock = asyncio.Lock()`
  - 所有写入内部状态的方法（`place_order()`、`_update_balance()`）使用 `async with self._lock:` 保护，只读方法同样通过 Lock 快照读
  - _Requirements: 5.4_

- [x] 3.3 为超时保护与并发安全补充测试
  - 测试 LLM 超时降级：mock `ChatOpenAI.ainvoke` 延迟超过 `timeout_seconds`，验证 `asyncio.wait_for` 超时并以空结果降级，不阻塞流水线
  - 测试 `PaperExchange` 并发写入：并发调用 `place_order()`，验证余额最终一致性
  - _Requirements: 5.1, 5.4, 5.5_

- [x] 4. 实现快照 Hash 复用与 LLM 用量日志
- [x] 4.1 (P) 为相同快照实现分析结果复用
  - 在 `nodes/data.py` 的 `collect_snapshot()` 中，对 `price`、`funding_rate`、`volatility`、`orderbook_imbalance` 关键字段计算 SHA256 hash，存入 `state["data"]["snapshot_hash"]`
  - 在 `nodes/agents.py` 中，将当前 hash 与 `state["data"].get("prev_snapshot_hash")` 对比；相同时复用 `state["data"]["prev_analyses"]` 并跳过 LLM 调用，不同时正常执行后更新 `prev_snapshot_hash`
  - 此优化仅在调度器连续周期场景生效，单次调用不触发
  - _Requirements: 8.4_

- [x] 4.2 (P) 实现 LLM 调用 token 消耗结构化日志
  - 在 `agents/base.py` 的 `create_llm()` 或 LLM 调用后，从响应的 `usage_metadata` 中提取 `input_tokens`、`output_tokens`、`model_name` 并通过 structlog 记录
  - 日志字段使用 `llm_usage` 命名空间，支持后续按时间窗口汇总成本报告
  - _Requirements: 8.5_

- [x] 5. 建立可观测性基础设施层
- [x] 5.1 (P) 实现 MetricsCollector 与 Prometheus 指标端点
  - 新建 `metrics.py` 基础设施模块，使用 `prometheus-client` 注册系统核心指标：`ct_llm_calls_total`（Counter，标签 `model/node`）、`ct_debate_skipped_total`（Counter）、`ct_verdict_total`（Counter，标签 `action`）、`ct_risk_rejected_total`（Counter，标签 `check_name`）、`ct_trade_executed_total`（Counter，标签 `engine/side`）、`ct_execution_latency_ms`（Histogram，标签 `engine`）、`ct_pipeline_duration_ms`（Histogram）
  - 实现模块级单例 `get_metrics_collector()` 函数
  - 新建 `api/routes/metrics.py`，暴露 `GET /metrics` 端点返回 Prometheus 文本格式，并在 `api/main.py` 中注册路由
  - 在 `prometheus-client` 加入 `pyproject.toml` 生产依赖
  - _Requirements: 9.5_

- [x] 5.2 (P) 实现 NodeLogger 节点耗时装饰器
  - 在 `tracing.py` 中扩展实现 `@node_logger()` 装饰器工厂，包裹异步节点函数
  - 入口记录 `node_entry` 事件（含 `node`、`trace_id`），出口记录 `node_exit` 事件（含 `node`、`duration_ms`、`trace_id`）
  - 使用 `functools.wraps` 保留函数元数据，确保 LangGraph `get_type_hints()` 正常解析
  - 为 `nodes/` 下所有公开节点函数添加 `@node_logger()` 装饰器
  - _Requirements: 9.3_

- [x] 5.3 (P) 规范化 structlog 字段集与 trace_id 传播
  - 在 `tracing.py` 中确认 `set_trace_id()` 通过 `structlog.contextvars` 绑定，使日志字段 `trace_id` 在整个流水线的所有子节点日志中自动传播
  - 确认所有节点日志条目包含标准字段：`timestamp`、`level`、`module`、`trace_id`（流水线执行期间）、`symbol`（适用时）、`node`（节点执行期间）、`duration_ms`（节点 exit 时）
  - 在 `api/routes/` 的请求中间件（`trace_middleware`）中补全缺失字段：`method`、`path`、`status_code`、`response_time_ms`、`client_ip`（已脱敏）
  - _Requirements: 9.1, 9.2, 9.7_

- [x] 5.4 (P) 实现 OpenTelemetry 可选追踪集成
  - 新建 `otel.py` 基础设施模块，提供 `setup_otel(service_name)` 和 `get_tracer()` 接口
  - 仅在环境变量 `OTLP_ENDPOINT` 非空时激活，否则返回 `NoOpTracer`（静默降级）
  - 将 `opentelemetry-sdk` 和 `opentelemetry-exporter-otlp-proto-grpc` 加入 `pyproject.toml` 的 `otel` 可选依赖组
  - 在 `api/main.py` 的 `lifespan()` 和 CLI 入口的 `setup_logging()` 之后初始化 OTel
  - 在 `@node_logger()` 装饰器内同时创建 OTel span，实现日志与追踪关联
  - _Requirements: 9.6_

- [x] 5.5 完善风控拒绝日志字段与指标埋点
  - 在 `nodes/verdict.py` 的 `risk_check()` 中，对拒绝事件补充结构化日志字段：`check_name`、`current_value`、`threshold`、最终拒绝原因摘要
  - 将 MetricsCollector 埋点接入关键调用点：`create_llm()` 记录 `ct_llm_calls_total`、`debate_gate()` 跳过时记录 `ct_debate_skipped_total`、`make_verdict()` 记录 `ct_verdict_total`、风控拒绝时记录 `ct_risk_rejected_total`
  - _Requirements: 9.4, 9.5_

- [x] 5.6 为可观测性组件补充测试
  - 测试 `MetricsCollector`：各计数器递增正确、`generate_latest()` 输出包含预期指标名
  - 测试 `GET /metrics` 端点：使用 FastAPI `TestClient` 验证响应含 `ct_llm_calls_total`
  - 测试 `@node_logger()` 装饰器：验证 `node_entry`/`node_exit` 事件被记录且 `duration_ms` 字段存在
  - _Requirements: 9.3, 9.5_

- [x] 6. 实施安全实践加固
- [x] 6.1 (P) 实现 PromptSanitizer 防注入清洗
  - 新建 `security.py` 基础设施模块，实现 `sanitize_input(text, max_chars=2000)` 函数
  - 清洗规则：截断至 `max_chars`、移除 Unicode 控制字符（`\x00`–`\x1f` 除 `\n\t`）、检测并过滤常见注入模式（如连续多换行 + 指令前缀）
  - 在 `agents/base.py` 的 prompt 构建逻辑中，对 `snapshot.news.headlines` 等外部来源字段逐项调用 `sanitize_input()`
  - 不对 Agent 内部系统 prompt（`role_description`、`ANALYSIS_FRAMEWORK`）应用清洗
  - _Requirements: 7.5_

- [x] 6.2 (P) 实施 FastAPI 生产加固
  - 在 `api/main.py` 的 `FastAPI()` 初始化中，读取环境变量 `DOCS_ENABLED`（默认 `"false"`），为 false 时传入 `docs_url=None, redoc_url=None` 关闭文档端点
  - 确认 `RequestValidationError` 全局处理器返回 422 并记录请求摘要日志（不含请求体原文和敏感字段）
  - _Requirements: 7.4, 7.6_

- [x] 6.3 (P) 为 verify=False 调用添加规范注释
  - 在 `data/sync.py` 和 `data/providers/sosovalue.py` 中，对所有 `verify=False` 的 HTTPS 调用同行添加注释：`# nosec S501 — 第三方数据源 {名称} 使用自签名证书，已确认无敏感数据传输`
  - 确认相关 `S501` 已在 ruff 忽略配置中，不产生 lint 错误
  - _Requirements: 7.7_

- [x] 6.4 (P) 实现外部 API 响应 schema 校验
  - 在 `models.py` 中新增 `NewsHeadlineResponse` 和 `OnchainMetricResponse` Pydantic 模型，含 `field_validator` 约束（非空 title 等）
  - 在数据采集层适配器中，对新闻 API 和链上数据 API 的响应使用对应模型进行 schema 校验；校验失败时记录 `logger.warning` 并跳过该条数据
  - _Requirements: 7.3_

- [x] 6.5 为安全组件补充测试
  - 测试 `sanitize_input()`：超长截断、控制字符过滤、常见注入模式被移除、合法特殊字符（如 Token 名含 `&`）不被误过滤
  - _Requirements: 7.5_

- [x] 7. 加固测试基础设施与覆盖率门控
- [x] 7.1 (P) 配置 pytest-cov 分支覆盖率阈值
  - 在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中，`addopts` 补充 `--cov=src --cov-report=term-missing --cov-fail-under=70 --cov-branch`
  - 将 `pytest-cov>=5.0` 加入 `[project.optional-dependencies]` 的 `test` 组
  - 将 `test` 组与 `dev` 组拆分，`dev` 组仅含 `ruff`、`pre-commit` 等开发工具；新增 `otel` 可选组
  - _Requirements: 4.5, 10.7_

- [x] 7.2 (P) 补齐图拓扑集成测试
  - 为 `build_trading_graph()`、`build_lite_graph()`、`build_debate_graph()` 补充集成测试，验证节点名称（`collect_data`、`debate_gate`、`risk_gate` 等）和条件边路由（`debate_gate_router` 返回 `"skip"` 时连接 `enrich_context`）
  - _Requirements: 4.3_

- [x] 7.3 (P) 补齐回测集成测试（无网络模式）
  - 补充 `test_backtest.py`，mock CCXT 和外部 HTTP 接口，使用 SQLite 内存数据库执行完整回测流程
  - 验证结果包含 `total_pnl`、`win_rate` 等关键字段，确保测试可在无网络环境下完整运行
  - _Requirements: 4.6_

- [x] 7.4 (P) 完善调度器防重叠测试
  - 补充 `test_scheduler_misfire.py`，模拟上次任务未完成时触发下次调度，验证 `max_instances=1` 阻止重叠执行
  - _Requirements: 5.6_

- [x] 8. 完善代码架构边界约束
- [x] 8.1 (P) 消除 nodes 层同层依赖并标注替代路径
  - 将 `nodes/verdict.py` 中对 `nodes/execution.py.read_portfolio_from_exchange` 的引用，通过将该函数提升至 `portfolio/manager.py` 来消除同层依赖
  - 在 `graph_supervisor.py` 和 `agents/langchain_agents.py` 文件顶部添加标准化状态注释，说明其为"实验性（未启用于主路径）"及与主路径的关系
  - _Requirements: 1.1, 1.6_

- [x] 8.2 (P) 配置 ruff 模块边界静态检查
  - 在 `pyproject.toml` 的 `[tool.ruff.lint]` 中，通过 `TID` 规则配置禁止 `nodes/` 或 `graph.py` 被领域层（`agents/`、`debate/`、`execution/`、`risk/`、`learning/`）反向导入
  - 确认现有 `TCH`（循环导入保护）规则已启用，对检测到的循环导入通过重构模块边界或 `TYPE_CHECKING` 保护块消除
  - _Requirements: 1.1, 1.5_

- [x] 9. 完善 CI 流水线与 Docker 部署配置
- [x] 9.1 (P) 更新 CI 流水线步骤
  - 更新 `.github/workflows/ci.yml`，将安装目标从 `.[dev]` 改为 `.[test]`
  - 确保 CI 步骤按序执行：`ruff check`（lint）→ `ruff format --check`（格式验证）→ `pytest`（测试 + 覆盖率，由 `addopts` 的 `--cov-fail-under=70` 自动门控）→ `docker build`（仅 main 分支）
  - 任意步骤失败则阻断后续步骤
  - _Requirements: 4.4, 10.3_

- [x] 9.2 (P) 完善 Docker Compose 资源限制与命名卷
  - 为 `api`、`scheduler`、`dashboard` 服务添加 `deploy.resources.limits`（`memory: 512m`、`cpus: "1.0"`）
  - 新增 `ctdata` 命名卷，挂载至 `api`/`scheduler`/`dashboard` 服务的 `/home/appuser/.cryptotrader`，确保 SQLite 数据文件跨容器重启持久化
  - 确认服务命名与需求一致（`api`、`scheduler`、`dashboard`、`redis`、`postgres`），`api` 服务通过 `DOCS_ENABLED=false` 环境变量关闭文档端点
  - _Requirements: 10.1, 10.5, 10.6_

- [x] 9.3 (P) 完善健康检查端点组件状态
  - 补全 `api/routes/health.py` 的 `/health` 端点，返回详细的组件状态：数据库连通性、Redis 连通性、LLM API 可达性
  - 确保 Docker 编排器可依据 `/health` 响应决定是否重启容器（HEALTHCHECK 配置已存在或补充）
  - _Requirements: 10.4_
