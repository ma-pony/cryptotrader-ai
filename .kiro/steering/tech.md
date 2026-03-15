# 技术栈

## 架构

领域驱动的分层架构，以 LangGraph `StateGraph` 为骨架串联各层：

```
数据层 → 语言强化 → Agent 并行分析 → 辩论门控 → 裁决 → 风控 → 执行
```

`ArenaState`（TypedDict + Annotated reducer）是唯一的跨节点数据载体。三种图变体共享同一套节点函数，在 `graph.py` 中组装为不同拓扑。

## 核心技术

- **语言**：Python 3.12+，uv 包管理器
- **编排框架**：LangGraph 1.x（`StateGraph`、条件路由、`asyncio.gather` 并行节点）
- **LLM 接入**：LangChain 1.2+（`langchain-openai`、`langchain-anthropic`、`langchain-google-genai`）；统一网关模式，所有模型通过同一 `base_url` 路由
- **交易所**：CCXT 4.5+，`PaperExchange`（模拟）/ `LiveExchange`（真实）双引擎
- **数据存储**：SQLite（本地市场数据缓存 + LLM 响应缓存 + 经验记忆）；PostgreSQL（可选，通过 Docker）；Redis（可选，风控状态持久化）
- **API 服务**：FastAPI + Uvicorn
- **定时调度**：APScheduler 3.x（`AsyncIOScheduler`，`IntervalTrigger` + `CronTrigger`）
- **CLI**：Typer + Rich（入口命令 `arena`）
- **Dashboard**：Streamlit

## 关键库约定

| 库 | 用法约定 |
|---|---|
| `langchain` | 使用 `create_agent()`，禁止 `create_react_agent()`（已废弃）|
| `ChatOpenAI` | 使用 `max_completion_tokens`，不用 `max_tokens` |
| `SQLiteCache` | 位于 `~/.cryptotrader/llm_cache.db`，自动缓存 LLM 调用 |
| `create_llm()` | 统一工厂，空模型字符串自动解析为 `config.models.analysis` 或 `config.models.fallback` |
| `structlog` | 结构化日志；`structlog.contextvars` 自动传播 `trace_id` 至所有子节点日志 |
| `prometheus_client` | `MetricsCollector` 单例收集 7 个指标，`/metrics` 端点以 Prometheus 文本格式暴露 |

## 开发标准

### 类型安全

Pydantic 2.x 数据模型用于配置和 API 响应；LangGraph State 使用 `Annotated[T, reducer]` 声明合并策略；`TYPE_CHECKING` 保护纯类型导入（但 LangGraph 运行时需 `get_type_hints()` 的文件例外，已在 ruff 中配置忽略 `TCH001/002/003`）。

### 代码质量

#### Pre-commit Hooks

项目使用 pre-commit 在提交前自动检查，配置于 `.pre-commit-config.yaml`：

| Hook | 来源 | 说明 |
|---|---|---|
| `ruff-check --fix` | astral-sh/ruff-pre-commit | Lint 检查（自动修复可修复项）|
| `ruff-format` | astral-sh/ruff-pre-commit | 代码格式化 |
| `trailing-whitespace` | pre-commit-hooks | 删除行尾空白 |
| `end-of-file-fixer` | pre-commit-hooks | 确保文件末尾有换行 |
| `check-yaml` / `check-toml` | pre-commit-hooks | 校验 YAML/TOML 语法 |
| `check-merge-conflict` | pre-commit-hooks | 检测未解决的合并冲突标记 |
| `check-added-large-files` | pre-commit-hooks | 阻止 >1MB 大文件 |
| `detect-private-key` | pre-commit-hooks | 检测提交中的私钥 |
| `detect-secrets` | Yelp/detect-secrets | 基于 `.secrets.baseline` 的密钥泄漏检测 |

#### Ruff 配置

配置位于 `pyproject.toml [tool.ruff]`：

- **行宽**：120
- **目标版本**：Python 3.12
- **启用规则集**：`E`/`W`（pycodestyle）、`F`（pyflakes）、`I`（isort）、`C`（comprehensions）、`B`（bugbear）、`T20`（print 检测）、`ASYNC`（异步检查）、`UP`（pyupgrade）、`RET`（return 风格）、`SIM`（简化）、`PERF`（性能）、`PIE`、`RSE`（raise 风格）、`RUF`（ruff 特有）、`DTZ`（时区感知 datetime）、`TID`（import 禁令）、`PT`（pytest 风格）、`TCH`（TYPE_CHECKING）、`S`（安全 bandit）、`N`（命名规范）
- **忽略项**：`S101`（assert）、`S301`（pickle）、`S501`（verify=False）、`S110`（try-except-pass）、`S105/S106`（密码误报）、`RUF012`（ClassVar）、`PT011`（pytest.raises match）
- **per-file 豁免**：`state.py` 豁免 `TCH002/003`（LangGraph `get_type_hints()`）、`nodes/*.py` 豁免 `TCH001`（同理）、`reflect.py` 豁免 `RUF001`（中文全角标点）
- **isort**：`known-first-party = ["cryptotrader", "api", "cli"]`

#### 硬性要求

- **零 `noqa` 注释**，零 lint 错误
- 所有 `except Exception` 必须有 `logger.debug/warning(exc_info=True)`，禁止静默吞异常
- 魔法数字提升为命名常量或 `config/default.toml` 配置项

### 异步模式

节点函数全部为 `async def`；并行任务用 `asyncio.gather()`；后台任务用 `loop.create_task()`（禁止 `asyncio.ensure_future`）；LangChain 调用统一 `await llm.ainvoke()`。

### 测试

pytest + pytest-asyncio（`asyncio_mode = "auto"`）；覆盖率门槛 `--cov-fail-under=70`（分支覆盖）；LLM 调用 mock 路径：`patch("langchain_openai.ChatOpenAI.ainvoke") -> AIMessage`；测试路径 `tests/`，`pythonpath = ["src"]`。

依赖组分离（`pyproject.toml`）：`[project.optional-dependencies]` 下 `test`（pytest 套件）、`dev`（pre-commit + ruff）、`otel`（OpenTelemetry SDK）三组独立安装。

## 开发环境

### 必要工具

- Python 3.12+，uv
- Docker（Postgres / Redis 唯一运行方式，禁止 Homebrew 安装）
- `config/default.toml` 驱动所有配置；`CRYPTOTRADER_*` 环境变量可覆盖任意配置项（`__` 分隔嵌套路径，优先级最高）

### 常用命令

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest

# lint 检查
uv run ruff check src/ tests/

# 启动完整交易循环（模拟盘）
uv run arena run BTC/USDT

# 启动 API 服务
uv run arena serve

# 启动 Streamlit Dashboard
uv run streamlit run src/dashboard/app.py

# 回测
uv run arena backtest BTC/USDT --days 30

# 查看实盘就绪状态
uv run arena live-check
```

## 关键技术决策

**配置文件单一来源**：所有阈值、模型选择、交易对均在 `config/default.toml` 中，`load_config()` 首次调用后缓存，禁止散落的硬编码值。`validate_config()` 在启动时校验关键约束，违反则抛出 `ConfigurationError`。环境变量 `CRYPTOTRADER_<SECTION>__<KEY>` 以最高优先级覆盖 TOML 值。

**交易所为组合状态权威来源**：仓位、余额从交易所实时查询；DB 只存历史快照（PnL、回撤、净值曲线）和 Paper 引擎冷启动数据。

**渐进过滤降低 LLM 成本**：辩论门控（共识跳过）+ 裁决降级（持平仓位 + 无熔断 → 加权裁决，0 次 LLM 调用）将高共识场景的 LLM 调用从 13 次降至 4-5 次。

**Redis 可选但保守**：Redis 不可用时，如已配置则保守拒绝交易（而非忽略风控状态）。

**可观测性分层**：`log_config.py`（`setup_logging()`）配置结构化日志输出（`LOG_FORMAT=json|console`，`LOG_LEVEL` 可调）；`tracing.py`（`@node_logger()`）装饰所有节点函数，emit `node_entry`/`node_exit` 事件并含 `duration_ms`；`otel.py` 提供 OpenTelemetry 可选集成（`OTLP_ENDPOINT` 未设置时降级为 no-op）；`metrics.py` 提供 Prometheus 指标单例（7 个计数器/直方图），通过 `/metrics` 端点暴露。

**安全：外部数据消毒**：`security.py`（`sanitize_input()`）在所有外部数据（新闻标题、链上文本等）嵌入 LLM prompt 前调用，防止 prompt 注入；内部系统提示（`role_description`、`ANALYSIS_FRAMEWORK`）不得过滤。

**模块边界强制**：`pyproject.toml` 通过 ruff `TID251`（`flake8-tidy-imports.banned-api`）在 lint 阶段强制禁止领域层（`agents/`、`debate/`、`execution/`、`risk/`、`learning/`）反向导入 `nodes/` 或 `graph.py`；入口层（`api/routes/`、`cli/main.py`、`scheduler.py`、`backtest/engine.py`）豁免。

**CI 流水线**：`.github/workflows/ci.yml` 按 lint → format → test → docker build 顺序执行；docker 构建仅在 main 分支且测试通过后运行。

---
_记录标准与模式，而非所有依赖项_
