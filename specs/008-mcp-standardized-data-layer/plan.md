# 技术实施方案：MCP 标准化数据层

## 技术上下文

### 现有数据层架构分析

数据层采用分散式 Python 模块设计，各 Provider 模块直接被 `OnchainCollector`、`MacroCollector` 等聚合器通过硬编码 `import` 调用：

```
nodes/data.py → SnapshotAggregator → OnchainCollector / MacroCollector
                                          ↓
                    providers/binance.py     data/macro.py
                    providers/coinglass.py   data/onchain.py
                    providers/defillama.py   data/market.py
                    providers/cryptoquant.py
                    providers/whale_alert.py
                    providers/sosovalue.py
                    providers/rss_news.py
```

**核心模式**：每个 Provider 函数均为 `async def`，接受简单参数（`symbol: str`、`api_key: str`），返回结构化 `dict`，失败时返回零值兜底 dict（不抛出异常）。这种模式与 MCP 工具签名天然对齐。

### 缓存基础设施分析（`data/store.py`）

现有缓存层通过 SQLite（`~/.cryptotrader/market_data.db`）提供以下核心能力：

- **`get_cached_or_none(source, date)`**：TTL 检查（实时模式）或精确日期查找（回测模式）
- **`cache_result(source, data, date)`**：存储结果并记录 fetch 时间戳
- **`_RATE_LIMITS`**：每个数据源的最小 fetch 间隔（5 分钟～1 小时）
- **`_should_fetch(source)`**：基于 `fetch_log` 表判断是否需要重新拉取

关键约束：`get_cached_or_none(source, date=None)` 在 TTL 未过期时直接返回缓存，**不发起 HTTP 请求**。MCP 工具层必须复用此机制，不能绕过。

### Provider 函数模式（以 `binance.py` 为例）

```python
async def fetch_derivatives_binance(symbol: str = "BTC") -> dict:
    result = {"open_interest": 0.0, "long_short_ratio": 1.0, ...}  # 零值兜底
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            ...
    except Exception:
        logger.warning("...", exc_info=True)
    return result  # 永远不抛出
```

**特征**：
1. 纯异步函数，无状态
2. 内置零值兜底，调用方无需处理异常
3. 参数简单（`symbol`、`api_key`、`chain` 等字符串）
4. 返回 `dict`，字段与 `ArenaState` 中的数据字段一一对应

### 宏观数据模式（`data/macro.py`）

宏观 Provider 在函数内部直接调用 `get_cached_or_none()` / `cache_result()`，而非让上层聚合器管理缓存。MCP 工具包装时只需直接调用这些函数，缓存行为自动继承。

### 回测短路模式（`data/macro.py`、`nodes/data.py`）

`collect_snapshot` 节点在 `state["data"]["snapshot"]` 已存在时直接复用（回测注入路径），跳过整个 `SnapshotAggregator.collect()` 调用。`MCPAdapter` 的回测短路须在更早的位置介入，确保 `backtest_mode=True` 时根本不进入 MCP 调用栈。

### 配置系统（`config.py`）

`config.py` 采用 `@dataclass` 模式，`load_config()` 缓存首次加载结果。新增 `MCPConfig` 只需：
1. 定义 `MCPServerConfig` 和 `MCPConfig` dataclass
2. 在顶级 `Config` dataclass 中增加 `mcp: MCPConfig` 字段
3. 在 `config/default.toml` 中增加 `[mcp]` 段（`enabled = false`）

### 现有依赖（`pyproject.toml`）

```toml
dependencies = [
    "httpx>=0.28",
    "fastapi>=0.135",
    "pydantic>=2.12",
    # fastmcp 尚未引入
]

[tool.coverage.run]
omit = [
    "src/cryptotrader/data/providers/binance.py",
    "src/cryptotrader/data/providers/rss_news.py",
    "src/cryptotrader/data/providers/sosovalue.py",
    # 新增 MCP Server 工具层也应 omit
]
```

---

## 架构决策

### 决策 1：FastMCP 作为 MCP 框架（不自实现协议）

**选择**：使用 `fastmcp` 库（`pyproject.toml` 新增唯一外部依赖）作为 MCP Server 实现框架。

**理由**：`fastmcp` 提供 `@mcp.tool()` 装饰器、stdio/HTTP 传输内置支持、Pydantic 参数验证，与现有 Python 3.12+ / asyncio / Pydantic v2 技术栈完全兼容。自实现 JSON-RPC 2.0 协议层（MCP 底层协议）约需 2000 行代码，且需维护协议兼容性，不值当。

**版本锁定**：在 `pyproject.toml` 声明 `fastmcp>=2.0`（当前最新稳定系列），避免破坏性升级。

### 决策 2：stdio 传输作为默认模式，HTTP/SSE 作为可选模式

**选择**：`MCPConfig.transport` 默认为 `"stdio"`（同进程内嵌运行），`"http"` 为跨进程/外部 AI 工具模式（按需配置）。

**理由**：
- `stdio` 模式下 MCP Server 与 Agent 同进程，无网络序列化开销，p95 延迟 < 50ms（缓存命中），满足 SC-003
- `http` 模式启用 Claude Code / Cursor 外部连接，但不应成为实盘交易默认路径（增加不确定性）
- `fastmcp` 同时支持两种传输，切换仅需修改配置，无需修改工具实现代码

**并发处理**：stdio 模式下 `fastmcp` 基于 asyncio 协程，单进程内多个工具调用通过事件循环并发执行，无需额外线程池。

### 决策 3：MCPRegistry — 轻量内存注册表，不引入服务发现中间件

**选择**：`MCPRegistry` 是一个 Python 类，持有 `dict[str, MCPServerHandle]` 内存映射，在进程启动时从 `MCPConfig` 初始化。不使用 Redis、etcd 或任何外部服务发现组件。

**理由**：
- CryptoTrader-AI 是单进程应用（含 scheduler 服务），不需要分布式服务发现
- 内存注册表与现有 `load_config()` 生命周期一致，测试简单（mock 注入即可）
- 支持通过 `registry.register_server(name, handle)` 动态注册，满足 US-2（无需重启即可新增 Server 的设计扩展点）

**健康检查**：`MCPRegistry.health_check()` 遍历所有注册 Server，通过调用空参数 `list_tools()` 验证连通性（非破坏性）。

### 决策 4：MCPAdapter — 三级路由（MCP → Python fallback → 零值兜底）

**选择**：`MCPAdapter` 实现三级调用路由：

```
调用请求
  ↓
backtest_mode=True? → 直接调用 Python Provider 函数（零 MCP 调用）
  ↓ (False)
mcp.enabled=True? 且 Registry 中工具存在?
  → MCP 工具调用（超时 5s）
    ↓ 成功 → 返回结果
    ↓ 失败/超时 → fallback_on_error=True? → Python Provider 函数调用
                                           → 失败 → 零值兜底 dict
  ↓ (mcp.enabled=False 或工具不存在)
直接调用 Python Provider 函数（当前行为）
```

**理由**：
- 回测短路必须在最高层（防止 look-ahead bias），任何 MCP 网络 I/O 在回测模式下均不可接受
- Python fallback 确保 MCP Server 故障不中断实盘（FR-007，SC-006）
- `mcp.enabled=false`（默认）时完全绕过 MCP 代码路径，满足 SC-004 零改动量要求

### 决策 5：缓存复用策略 — MCP 工具层不持有任何缓存状态

**选择**：MCP 工具函数体仅包装现有 Provider 函数调用（1-3 行代码），缓存逻辑完全由 Provider 内部的 `get_cached_or_none()` / `cache_result()` 管理。

**理由**：
- 现有 Provider 已在函数内部调用 `data/store.py` 的缓存 API（`macro.py`、`onchain.py` 等均如此）
- MCP 工具层引入第二套缓存会导致缓存失效逻辑分裂，同一数据可能在两处过期不同步
- 不引入缓存状态的 MCP 工具层可完全无状态，便于测试（无需 mock SQLite）

**截断策略**：工具调用后检查返回 `dict` 的 JSON 序列化大小，超过 50KB 时截断列表字段并追加 `"truncated": true`（边界条件 BC-006）。

### 决策 6：`.mcp.json` 格式 — 兼容 Claude Code 官方规范

**选择**：项目根目录提供 `.mcp.json`，格式遵循 Claude Code MCP 客户端配置规范：

```json
{
  "mcpServers": {
    "cryptotrader-binance": {
      "command": "python",
      "args": ["-m", "cryptotrader.mcp.servers.binance"],
      "env": {}
    },
    "cryptotrader-macro": { ... },
    "cryptotrader-onchain": { ... },
    "cryptotrader-news": { ... }
  }
}
```

**理由**：Claude Code 和 Cursor 均使用此格式发现 MCP Server（标准 `mcpServers` 键名）。`command: "python" + args: ["-m", "..."]` 模式利用项目虚拟环境，无需额外安装步骤（满足 SC-005）。

### 决策 7：回测模式短路 — 在 MCPAdapter 层，不在 SnapshotAggregator 层

**选择**：`MCPAdapter.call(tool_name, args, *, backtest_mode)` 接收 `backtest_mode` 参数，在适配器入口处判断并短路，而非在 `SnapshotAggregator` 或 `collect_snapshot` 节点层处理。

**理由**：
- `SnapshotAggregator` 已有自己的回测路径（注入 `snapshot` 时跳过 `collect()`），两者职责不同
- `MCPAdapter` 集中管理 MCP 调用决策，保持 `SnapshotAggregator` 对 MCP 的感知最小化
- `backtest_mode` 来自 `state["metadata"]["backtest_mode"]`，在 `collect_snapshot` 节点传递给适配器

### 决策 8：工具命名规范 — `{server_prefix}_{data_type}` 全局唯一

**选择**：所有 MCP 工具名使用小写 snake_case，格式为 `{server_prefix}_{data_type}`：
- `binance_derivatives`、`binance_funding_rate`、`binance_klines`
- `macro_fear_greed`、`macro_btc_dominance`、`macro_fred_series`、`macro_etf_flow`
- `onchain_defi_tvl`、`onchain_derivatives`、`onchain_exchange_netflow`、`onchain_whale_transfers`
- `news_rss`、`news_sosovalue`

**理由**：前缀即 Server 所属，全局唯一，`MCPRegistry.route(tool_name)` 可 O(1) 查找。满足边界条件 BC-005（工具名冲突防护）。

---

## 文件结构（新增 / 修改）

```
config/
  default.toml                    # 修改：新增 [mcp] 段，enabled=false

src/cryptotrader/
  mcp/                            # 新增子包
    __init__.py
    config.py                     # MCPServerConfig, MCPConfig dataclasses
    registry.py                   # MCPRegistry 类，工具发现与路由
    adapter.py                    # MCPAdapter 类，三级路由（MCP / Python / 零值）
    servers/                      # 新增子包
      __init__.py
      binance.py                  # BinanceMCPServer（3 工具）
      macro.py                    # MacroMCPServer（4 工具）
      onchain.py                  # OnchainMCPServer（4 工具）
      news.py                     # NewsMCPServer（2 工具）
    utils.py                      # truncate_response()，API Key 过滤

  config.py                       # 修改：顶级 Config 新增 mcp: MCPConfig 字段

  data/
    snapshot.py                   # 修改：SnapshotAggregator.collect() 接受 adapter 参数

  nodes/
    data.py                       # 修改：collect_snapshot 节点传递 backtest_mode 给 MCPAdapter

.mcp.json                         # 新增：Claude Code / Cursor MCP 发现入口

src/cli/
  main.py                         # 修改：新增 arena mcp list / arena mcp call 子命令

tests/
  test_mcp_config.py              # 新增：MCPConfig 加载、默认值验证
  test_mcp_registry.py            # 新增：Registry 注册、路由、健康检查
  test_mcp_adapter.py             # 新增：三级路由、回测短路、fallback、超时
  test_mcp_servers.py             # 新增：4 个 Server 工具发现 + 调用 + 缓存命中
  test_mcp_cli.py                 # 新增：arena mcp list / mcp call CLI 命令

pyproject.toml                    # 修改：新增 fastmcp>=2.0 依赖；coverage omit 增加 mcp/servers/
```

---

## 数据模型

### `MCPServerConfig` 和 `MCPConfig` dataclasses

```python
# src/cryptotrader/mcp/config.py

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MCPServerConfig:
    name: str                                  # 全局唯一，如 "cryptotrader-binance"
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "localhost"                    # HTTP 模式用
    port: int = 8080                           # HTTP 模式用
    enabled: bool = True
    tools: list[str] = field(default_factory=list)  # 空列表 = 自动发现


@dataclass
class MCPConfig:
    enabled: bool = False                      # 全局开关，默认关闭（渐进迁移）
    transport: Literal["stdio", "http"] = "stdio"
    fallback_on_error: bool = True
    call_timeout_s: float = 5.0
    servers: list[MCPServerConfig] = field(default_factory=list)
```

### 顶级 `AppConfig` dataclass 扩展（`config.py`）

```python
@dataclass
class AppConfig:
    # ... 现有字段 ...
    mcp: MCPConfig = field(default_factory=MCPConfig)  # 新增
```

### `config/default.toml` 新增段

```toml
[mcp]
enabled = false
transport = "stdio"
fallback_on_error = true
call_timeout_s = 5.0

[[mcp.servers]]
name = "cryptotrader-binance"
transport = "stdio"
enabled = true
tools = ["binance_derivatives", "binance_funding_rate", "binance_klines"]

[[mcp.servers]]
name = "cryptotrader-macro"
transport = "stdio"
enabled = true
tools = ["macro_fear_greed", "macro_btc_dominance", "macro_fred_series", "macro_etf_flow"]

[[mcp.servers]]
name = "cryptotrader-onchain"
transport = "stdio"
enabled = true
tools = ["onchain_defi_tvl", "onchain_derivatives", "onchain_exchange_netflow", "onchain_whale_transfers"]

[[mcp.servers]]
name = "cryptotrader-news"
transport = "stdio"
enabled = true
tools = ["news_rss", "news_sosovalue"]
```

### MCP 工具 Schema（以 BinanceMCPServer 为例）

> **注意**：`providers/binance.py` 当前仅有 `fetch_derivatives_binance()` 一个函数。
> `binance_funding_rate` 工具需在 T007 之前于 `providers/binance.py` 中新增
> `fetch_funding_rate_binance()` Provider 函数；`binance_klines` 工具的 K 线数据
> 来源为 `data/market.py` 中现有 ccxt 接口的 `fetch_ohlcv()`（需新增
> `fetch_klines_binance()` 包装函数），而非 `providers/binance.py`。

```python
# src/cryptotrader/mcp/servers/binance.py
from fastmcp import FastMCP

mcp = FastMCP("cryptotrader-binance")

@mcp.tool()
async def binance_derivatives(symbol: str = "BTC") -> dict:
    """查询 Binance 衍生品数据：OI、多空比、吃单比率。

    Returns:
        open_interest: 当前未平仓合约量（BTC）
        open_interest_value: 未平仓合约价值（USDT）
        long_short_ratio: 多空账户比
        top_trader_ratio: 大户多空比
        taker_buy_sell_ratio: 主动买卖比
        liquidations_24h: 24h 清算量汇总
    """
    from cryptotrader.data.providers.binance import fetch_derivatives_binance
    return await fetch_derivatives_binance(symbol)

@mcp.tool()
async def binance_funding_rate(symbol: str = "BTC") -> dict:
    """查询 Binance 合约资金费率。（需先在 providers/binance.py 新增 fetch_funding_rate_binance()）"""
    from cryptotrader.data.providers.binance import fetch_funding_rate_binance
    return await fetch_funding_rate_binance(symbol)

@mcp.tool()
async def binance_klines(symbol: str = "BTC", interval: str = "1h", limit: int = 100) -> dict:
    """查询 Binance K 线数据。（使用 data/market.py ccxt fetch_ohlcv() 包装的 fetch_klines_binance()）"""
    from cryptotrader.data.market import fetch_klines_binance
    return await fetch_klines_binance(symbol, interval, limit)
```

### `MCPRegistry` 类接口

```python
# src/cryptotrader/mcp/registry.py

class MCPRegistry:
    def __init__(self, config: MCPConfig) -> None: ...

    def register_server(self, name: str, server: FastMCP) -> None: ...

    def list_tools(self) -> list[str]: ...
    """返回所有已注册工具名，如 ["binance_derivatives", "macro_fear_greed", ...]"""

    async def call_tool(self, tool_name: str, args: dict) -> dict: ...
    """路由工具调用到对应 Server，超时 call_timeout_s 秒"""

    async def health_check(self) -> dict[str, bool]: ...
    """返回 {"cryptotrader-binance": True, "cryptotrader-macro": False, ...}"""

    def find_server(self, tool_name: str) -> FastMCP | None: ...
```

### `MCPAdapter` 类接口

```python
# src/cryptotrader/mcp/adapter.py

class MCPAdapter:
    def __init__(self, registry: MCPRegistry, config: MCPConfig) -> None: ...

    async def call(
        self,
        tool_name: str,
        args: dict,
        *,
        backtest_mode: bool = False,
        python_fallback: Callable[..., Awaitable[dict]] | None = None,
        fallback_args: dict | None = None,
        zero_value: dict | None = None,
    ) -> dict: ...
```

---

## 向后兼容性保证

| 变更点 | 现有调用方 | 保证 |
|--------|----------|------|
| `config.py` 新增 `mcp: MCPConfig` 字段 | `load_config()` 所有消费方 | `field(default_factory=MCPConfig)` 默认值，TOML 无 `[mcp]` 段时 `enabled=false`，行为零变化 |
| `SnapshotAggregator.collect()` 签名 | `nodes/data.py`、`backtest/engine.py` | 新增 `adapter: MCPAdapter | None = None` 可选参数，默认 `None` 时走现有代码路径 |
| `collect_snapshot` 节点 | `graph.py` 拓扑结构 | 仅在 `mcp.enabled=true` 时创建 `MCPAdapter`，`false`（默认）时节点行为与当前完全一致 |
| `mcp.enabled=false`（默认） | 全部现有测试（742+） | MCP 代码路径完全不执行，`MCPAdapter` 不实例化，零侵入 |
| `pyproject.toml` 新增 `fastmcp` | CI / Docker 构建 | `fastmcp` 仅在 `mcp.enabled=true` 时被 import（懒加载），即使包缺失也不影响 `mcp.enabled=false` 模式 |

---

## 依赖变更

```toml
# pyproject.toml [project.dependencies] 新增：
"fastmcp>=2.0",

# [tool.coverage.run] omit 新增：
"src/cryptotrader/mcp/servers/binance.py",
"src/cryptotrader/mcp/servers/macro.py",
"src/cryptotrader/mcp/servers/onchain.py",
"src/cryptotrader/mcp/servers/news.py",
```

**理由**：MCP Server 工具层是纯包装函数（调用现有 Provider），与现有 Provider 模块同等性质，均需实际外部 API 进行集成测试，不适合单元测试覆盖率要求。`MCPRegistry`、`MCPAdapter`、`MCPConfig` 属于业务逻辑，目标覆盖率 ≥ 80%（SC-008）。

---

## 风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| `fastmcp` API 在 2.x 发生破坏性变更 | 中 | 版本锁定 `>=2.0,<3`；MCP Server 工具层代码极薄（仅装饰器 + 一行调用），迁移成本低 |
| stdio 模式下 MCP Server 与 Provider 同进程，Provider 异常可能影响 Server 稳定性 | 低 | Provider 函数均有内置 `try/except` 零值兜底（现有设计），MCP 工具层透传即可；Server 崩溃由 `MCPAdapter` fallback 兜底 |
| `mcp.enabled=true` 时引入额外延迟（MCP 协议序列化开销） | 中 | stdio 模式无网络 I/O，序列化开销 < 1ms；缓存命中路径 p95 目标 < 50ms（SC-003），比当前 < 30ms 劣化不超过 100% |
| 回测模式被错误启用 MCP 导致 look-ahead bias | 高 | `MCPAdapter.call()` 的 `backtest_mode` 判断在方法入口第一行（防御优先），并增加断言测试（SC-007）；`backtest_mode` 来自不可变的 `state["metadata"]` |
| `.mcp.json` 中包含 API Key 敏感信息 | 中 | `.mcp.json` 只配置启动命令（`python -m ...`），API Key 通过环境变量传递（`env` 字段引用 `os.environ` 变量名而非值）；`.mcp.json` 加入 `.gitignore` 公开示例（`.mcp.json.example` 提交到仓库） |
| 工具名冲突（多个 Server 注册同名工具） | 低 | `MCPRegistry.register_server()` 在注册时检查工具名全局唯一性，冲突时抛出 `ValueError` 并记录 ERROR 日志，启动失败 |
| `fastmcp` 包在 PyPI 暂时不可用导致 CI 失败 | 低 | `fastmcp` 仅在运行时懒加载（`from fastmcp import FastMCP` 在 Server 模块顶层，非 `config.py` 顶层）；CI 离线缓存 wheel；`mcp.enabled=false` 测试套件无需安装 `fastmcp` |
| HTTP/SSE 模式并发调用超过 SQLite WAL 写锁容量 | 低 | 现有 `store.py` 使用 `WAL` 模式（`PRAGMA journal_mode=WAL`）支持并发读；MCP 工具主要是读操作（`get_cached_or_none`），写操作（`cache_result`）极少，不构成瓶颈 |
