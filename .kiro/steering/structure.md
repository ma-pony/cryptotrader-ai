# 项目结构

## 组织哲学

**领域分层 + 节点解耦**：核心业务逻辑按领域划分为独立子包（`agents/`、`debate/`、`execution/`、`risk/`、`learning/` 等），LangGraph 节点函数统一放在 `nodes/` 层，`graph.py` 只负责拓扑组装。这样节点可测试、可替换，图结构与业务逻辑分离。

## 目录模式

### 主包 (`src/cryptotrader/`)

核心系统，所有交易逻辑的归属地。子目录按职责单一原则划分：

- `nodes/` — LangGraph 节点函数（`data.py`、`agents.py`、`debate.py`、`verdict.py`、`execution.py`、`journal.py`）。每个文件对应流水线中的一个阶段。
- `agents/` — Agent 实现（`base.py` 含 `create_llm()` 工厂和共享常量；`tech.py`、`chain.py`、`news.py`、`macro.py` 为四个专业 Agent）。
- `data/` — 数据采集与缓存（`sync.py` 全量同步，`store.py` SQLite 缓存，`snapshot.py` 快照聚合，`providers/` 第三方数据源适配器）。
- `debate/` — 辩论逻辑（`convergence.py` 共识强度计算，`verdict.py` 辩论裁决）。
- `execution/` — 订单执行（`exchange.py` PaperExchange/LiveExchange，`simulator.py` 回测模拟器，`order.py` OrderManager）。
- `risk/` — 风控检查（`checks/` 各类检查器，`state.py` 风控状态，`gate.py` 门控聚合）。
- `learning/` — 经验记忆（`reflect.py` 规则提炼，`context.py` GSSC 上下文引擎，`regime.py` 市场状态标签，`verbal.py` 语言强化）。
- `backtest/` — 回测引擎（`engine.py`、`result.py`、`session.py`、`historical_data.py`）。
- `journal/` — 交易日志（`store.py` 写入，`search.py` 检索）。
- `portfolio/` — 组合管理（`manager.py` 持仓跟踪与通知）。

### 入口层 (`src/api/`、`src/cli/`)

- `api/routes/` — FastAPI 路由，每个资源一个文件（`analyze.py`、`health.py`、`metrics.py`、`journal.py`、`portfolio.py`）；`metrics.py` 暴露 `GET /metrics` Prometheus 端点
- `cli/main.py` — Typer CLI，`arena` 命令的唯一入口
- API 文档由 `DOCS_ENABLED` 环境变量控制（生产环境可关闭）；`RequestValidationError` 统一处理返回 422

### 图组装 (`src/cryptotrader/graph.py`)

只做拓扑声明：导入 `nodes/*` 的函数，调用 `add_node` / `add_edge` / `add_conditional_edges`，导出 `build_trading_graph()`、`build_lite_graph()`、`build_debate_graph()`、`build_backtest_graph()`。不含业务逻辑。

### 状态定义 (`src/cryptotrader/state.py`)

`ArenaState` TypedDict 是所有节点的唯一数据契约；`build_initial_state()` 工厂替代散落的内联初始化；`merge_dicts` 等 reducer 与 State 共处一文件。

### 配置与模型

- `config/default.toml` — 所有可调参数的唯一来源；`config/local.toml`（gitignored）用于本地密钥覆盖
- `src/cryptotrader/config.py` — dataclass 解析层，`load_config()` 单例缓存；`validate_config()` 启动校验；`apply_env_overrides()` 处理 `CRYPTOTRADER_*` 环境变量
- `src/cryptotrader/models.py` — Pydantic 业务数据模型（`ExperienceRule`、`ExperienceMemory` 等）
- `src/cryptotrader/db.py` — 共享异步 DB session 工厂，URL-keyed 引擎缓存

### 可观测性与基础设施模块

核心包根目录下的基础设施单例，按职责分文件：

- `log_config.py` — `setup_logging()`：在应用启动时统一配置 stdlib logging + structlog（`LOG_FORMAT=json|console`，`LOG_LEVEL` 可调）
- `tracing.py` — `@node_logger()` 装饰器（emit `node_entry`/`node_exit` + `duration_ms`）、`set_trace_id()`/`get_trace_id()`（基于 structlog contextvars）、`run_graph_traced()`
- `otel.py` — OpenTelemetry 可选集成；`setup_otel()` 仅在 `OTLP_ENDPOINT` 非空且已安装 `otel` 依赖组时激活，否则静默降级为 no-op tracer
- `metrics.py` — `MetricsCollector` 单例（`get_metrics_collector()`），封装 7 个 prometheus-client 计数器/直方图（`ct_llm_calls_total`、`ct_debate_skipped_total`、`ct_verdict_total`、`ct_risk_rejected_total`、`ct_trade_executed_total`、`ct_execution_latency_ms`、`ct_pipeline_duration_ms`）
- `security.py` — `sanitize_input()`：外部数据消毒（截断 + 控制字符过滤 + prompt 注入模式移除）；仅用于外部数据，内部系统提示不过滤
- `task_registry.py` — `add_background_task()`：防止 asyncio Task 被 GC 回收的模块级单例集合，含完成回调和异常日志

## 命名约定

- **文件**：`snake_case.py`，与领域名称一致（`backtest_engine.py` → 不用；直接 `engine.py` 放在 `backtest/` 下）
- **类**：PascalCase（`LiveExchange`、`ArenaState`、`ExperienceRule`）
- **函数/变量**：`snake_case`，动词开头表动作（`collect_snapshot`、`make_verdict`、`build_trading_graph`）
- **常量**：`UPPER_SNAKE_CASE`，定义在使用它的模块或 `agents/base.py`（跨模块常量）
- **私有辅助函数**：`_single_underscore` 前缀

## 导入组织

```python
# 1. 标准库
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

# 2. 第三方库
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# 3. 本项目（绝对路径）
from cryptotrader.config import load_config
from cryptotrader.state import ArenaState

# 4. 类型专用导入（延迟，避免运行时开销）
if TYPE_CHECKING:
    from cryptotrader.models import ExperienceMemory
```

LangGraph 通过 `get_type_hints()` 在运行时解析 State 和节点函数签名，`nodes/*.py` 和 `state.py` 中的类型导入不能放在 `TYPE_CHECKING` 块内（已在 ruff 配置中豁免 `TCH001/002/003`）。

## 代码组织原则

**单向依赖**：`nodes/` 依赖 `agents/`、`debate/`、`execution/` 等领域层；领域层不得反向依赖 `nodes/` 或 `graph.py`。该规则由 ruff `TID251` 在 lint 阶段强制执行。

**共享基础设施下沉**：重复的 DB 会话逻辑放 `db.py`；重复的 LLM 创建逻辑放 `agents/base.py`；重复的状态构造放 `state.py`；可观测性基础设施放 `log_config.py`/`tracing.py`/`otel.py`/`metrics.py`/`security.py`/`task_registry.py`。

**测试镜像结构**：`tests/` 下文件名与被测模块对应（`test_nodes.py`、`test_risk_checks.py`、`test_integration.py`）；mock 路径使用被测模块的完整路径；覆盖率门槛 70%（分支覆盖）由 CI 强制。

**实验性代码隔离**：`graph_supervisor.py` 和 `langchain_agents.py` 是备选的 Supervisor 模式实现，不在主流水线路径上，标记为实验性。主路径为 `graph.py`。

**从交易所读取组合数据**：`read_portfolio_from_exchange()` 归属 `portfolio/manager.py`（不在 nodes 层），由需要初始化组合状态的入口层调用。

**规格与引导分离**：开发规格放 `.kiro/specs/{feature}/`；项目记忆放 `.kiro/steering/`；两者独立演化，互不包含对方的内容。

---
_记录模式，而非文件树。遵循既有模式的新文件无需更新本文档_
