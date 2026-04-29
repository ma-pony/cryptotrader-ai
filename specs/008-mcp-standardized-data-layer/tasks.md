# 实施任务清单：MCP 标准化数据层

**Spec**: `specs/008-mcp-standardized-data-layer/`
**总任务数**: 36
**预计阶段**: 8 个阶段

---

## Phase 1：MCPConfig 基础设施

> 目标：建立配置数据模型和 `pyproject.toml` 依赖，不修改任何现有业务代码路径。

- [x] T001 [P] `pyproject.toml` 新增 `fastmcp>=2.0,<3` 依赖；在 `[tool.coverage.run] omit` 中追加 `src/cryptotrader/mcp/servers/*.py`
- [x] T002 [P] 新建 `src/cryptotrader/mcp/__init__.py` 和 `src/cryptotrader/mcp/servers/__init__.py`（空包初始化文件）
- [x] T003 新建 `src/cryptotrader/mcp/config.py`：定义 `MCPServerConfig` 和 `MCPConfig` dataclass，字段参考 `plan.md` 数据模型节；`MCPConfig.enabled` 默认 `False`，`call_timeout_s` 默认 `5.0`
- [x] T004 修改 `src/cryptotrader/config.py`：顶级 `AppConfig` dataclass 新增 `mcp: MCPConfig = field(default_factory=MCPConfig)`；在文件头部 import `MCPConfig`；确保 `load_config()` 不需改动（TOML 反序列化自动处理新字段）
- [x] T005 修改 `config/default.toml`：新增 `[mcp]` 段（`enabled = false`、`transport = "stdio"`、`fallback_on_error = true`、`call_timeout_s = 5.0`）及四个 `[[mcp.servers]]` 条目（名称与工具列表参考 `plan.md`）
- [x] T006 新建 `src/cryptotrader/mcp/utils.py`：实现 `truncate_response(data: dict, max_bytes: int = 51200) -> dict`（超过 50KB 时截断列表字段并追加 `"truncated": true`）；实现 `redact_api_key(text: str) -> str`（过滤 key 相关字符串）

---

## Phase 2：BinanceMCPServer

> 目标：首个 MCP Server 实现，验证 fastmcp 包装模式，作为后续 Server 的模板。

- [x] T006b [S] **新增底层 Provider 函数（T007 前置依赖）**：在 `src/cryptotrader/data/providers/binance.py` 中新增 `fetch_funding_rate_binance(symbol: str = "BTC") -> dict` 异步函数（通过 Binance REST API 获取合约资金费率，零值兜底 `{"funding_rate": 0.0, "next_funding_time": 0}`）；在 `src/cryptotrader/data/market.py` 中新增 `fetch_klines_binance(symbol: str = "BTC", interval: str = "1h", limit: int = 100) -> dict` 异步函数（利用现有 ccxt 接口的 `fetch_ohlcv()` 获取 K 线数据，返回 `{"klines": list[dict]}`，零值兜底 `{"klines": []}`）
- [x] T007 新建 `src/cryptotrader/mcp/servers/binance.py`：使用 `FastMCP("cryptotrader-binance")` 创建 Server 实例；实现 `binance_derivatives(symbol: str = "BTC") -> dict` 工具（包装 `fetch_derivatives_binance`）；实现 `binance_funding_rate(symbol: str = "BTC") -> dict` 工具（包装 `providers/binance.py` 的 `fetch_funding_rate_binance`）；实现 `binance_klines(symbol: str = "BTC", interval: str = "1h", limit: int = 100) -> dict` 工具（包装 `data/market.py` 的 `fetch_klines_binance`）；每个工具加 `truncate_response()` 后处理
- [x] T008 验证 `BinanceMCPServer` 工具可通过 `mcp.list_tools()` 发现，返回 3 个工具；在本 task 增加 `if __name__ == "__main__": mcp.run()` 入口供 stdio 运行

---

## Phase 3：MacroMCPServer / OnchainMCPServer / NewsMCPServer

> 目标：完成剩余三个 MCP Server，达到 SC-001 要求的 ≥ 13 个工具。

- [x] T009 [P] 新建 `src/cryptotrader/mcp/servers/macro.py`：实现 `macro_fear_greed() -> dict`（包装 `_fetch_fear_greed`；注意该函数返回 `tuple[int, list[int]]`，MCP 工具层须按阈值映射转换为 `{"value": int, "classification": str}`，映射规则：0-24→"Extreme Fear"，25-49→"Fear"，50-74→"Greed"，75-100→"Extreme Greed"）；实现 `macro_btc_dominance() -> dict`（包装 `_fetch_btc_dominance`）；实现 `macro_fred_series(series_id: str = "DFF") -> dict`（包装 `_fetch_fred`，从 `load_config()` 读取 `fred_api_key`）；实现 `macro_etf_flow(etf_type: str = "btc") -> dict`（包装 `sosovalue.fetch_etf_metrics`，需从 `load_config()` 注入 `api_key`，`api_key` 为空时返回零值兜底 `{"net_flow": 0.0, "data_available": false}`）
- [x] T010 [P] 新建 `src/cryptotrader/mcp/servers/onchain.py`：实现 `onchain_defi_tvl(chain: str = "Ethereum") -> dict`（包装 `defillama.fetch_tvl`）；实现 `onchain_derivatives(symbol: str = "BTC") -> dict`（包装 `coinglass.fetch_derivatives`，API Key 从 `load_config().providers.coinglass_api_key` 读取）；实现 `onchain_exchange_netflow() -> dict`（包装 `cryptoquant.fetch_exchange_netflow`；注意该函数返回 `float`，MCP 工具层须包装为 `{"exchange_netflow": float}` 结构后返回）；实现 `onchain_whale_transfers() -> dict`（包装 `whale_alert.fetch_whale_transfers`；注意该函数返回 `list[dict]`，MCP 工具层须包装为 `{"transfers": list[dict], "count": int}` 结构后返回）；付费 API Key 未配置时工具返回 `{"data_available": false, ...零值字段...}`
- [x] T011 [P] 新建 `src/cryptotrader/mcp/servers/news.py`：实现 `news_rss(max_per_source: int = 5) -> dict`（包装 `rss_news.fetch_crypto_news`；注意该函数为**同步函数**，MCP 工具层须通过 `asyncio.to_thread(rss_news.fetch_crypto_news, max_per_source)` 包装为 async 调用）；实现 `news_sosovalue(page: int = 1) -> dict`（包装 `sosovalue.fetch_news`，需从 `load_config()` 注入 `api_key`，`api_key` 为空时返回零值兜底 `{"articles": [], "count": 0}`）；两个工具均加 `truncate_response()` 后处理

---

## Phase 4：MCPRegistry

> 目标：实现工具注册表，支持工具发现、O(1) 路由和健康检查。

- [x] T012 新建 `src/cryptotrader/mcp/registry.py`：定义 `MCPRegistry` 类，持有 `_servers: dict[str, FastMCP]`（server_name → Server 实例）和 `_tool_index: dict[str, str]`（tool_name → server_name）内存映射
- [x] T013 实现 `MCPRegistry.register_server(name: str, server: FastMCP) -> None`：遍历 `server.list_tools()` 构建 `_tool_index`；工具名冲突时抛出 `ValueError` 并记录 `ERROR` 日志
- [x] T014 实现 `MCPRegistry.list_tools() -> list[str]`：返回 `_tool_index` 所有 key，有序排列；实现 `MCPRegistry.find_server(tool_name: str) -> FastMCP | None`：O(1) 查找
- [x] T015 实现 `MCPRegistry.call_tool(tool_name: str, args: dict) -> dict`：通过 `find_server()` 路由，调用 `server.call_tool()`；超时 `config.call_timeout_s` 秒（`asyncio.wait_for`），超时抛出 `MCPToolTimeoutError`
- [x] T016 实现 `MCPRegistry.health_check() -> dict[str, bool]`：对每个已注册 Server 调用 `list_tools()`，成功返回 `True`，失败/超时返回 `False`；实现 `MCPRegistry.from_config(config: MCPConfig) -> MCPRegistry` 工厂方法，自动实例化并注册四个内置 Server
- [x] T017a 在 `src/cryptotrader/mcp/__init__.py` 导出 `MCPRegistry`、`MCPConfig`、`MCPServerConfig`；定义 `MCPToolTimeoutError`、`MCPToolNotFoundError` 自定义异常类（Phase 4 末尾，MCPAdapter 尚未实现，暂不导出）
- [x] T017b 在 `src/cryptotrader/mcp/__init__.py` 追加导出 `MCPAdapter`（Phase 5 末尾，T019 完成后执行）

---

## Phase 5：MCPAdapter + SnapshotAggregator 集成

> 目标：将 MCPAdapter 接入 SnapshotAggregator，实现三级路由，mcp.enabled=false 时行为零变化。

- [x] T018 新建 `src/cryptotrader/mcp/adapter.py`：定义 `MCPAdapter` 类，构造函数接受 `registry: MCPRegistry` 和 `config: MCPConfig`
- [x] T019 实现 `MCPAdapter.call(tool_name, args, *, backtest_mode, python_fallback, fallback_args, zero_value) -> dict`：第一级 `backtest_mode=True` → 直接调用 `python_fallback(**fallback_args)`；第二级 `mcp.enabled=True` → `registry.call_tool()`；失败/超时且 `fallback_on_error=True` → 第三级 `python_fallback`；所有路径失败 → 返回 `zero_value`；记录 `[MCP fallback]` 警告日志（包含 tool_name 和失败原因）
- [x] T020 修改 `src/cryptotrader/data/snapshot.py` 中的 `SnapshotAggregator`：`collect()` 方法新增 `adapter: MCPAdapter | None = None` 可选参数（默认 `None`）；`adapter is None` 时代码路径与当前完全一致；`adapter is not None` 时通过 `adapter.call()` 获取衍生品数据、宏观数据、链上数据，`python_fallback` 指向原始 Provider 函数
- [x] T021 修改 `src/cryptotrader/nodes/data.py` 中的 `collect_snapshot` 节点：在 `mcp.enabled=true` 时从 `load_config().mcp` 创建 `MCPRegistry` 和 `MCPAdapter`，传入 `SnapshotAggregator.collect()`；`backtest_mode` 从 `state["metadata"].get("backtest_mode", False)` 读取并透传至 `adapter.call()`

---

## Phase 6：`.mcp.json` + Claude Code 兼容

> 目标：生成外部 AI 工具发现文件，满足 SC-005（Claude Code 零配置连接）。

- [x] T022 新建项目根目录 `.mcp.json`：四个 Server 条目，格式参考 `plan.md` 架构决策 6；`command` 使用 `"python"`，`args` 使用 `["-m", "cryptotrader.mcp.servers.binance"]` 等；`env` 字段为空对象（API Key 由用户在运行环境设置）
- [x] T023 新建 `.mcp.json.example`（提交到仓库的示例文件，注释说明各 Server 用途和所需环境变量）；确认 `.mcp.json` 本身已在 `.gitignore`（若尚未添加则追加）
- [x] T024 验证 MCP Server 可作为独立进程运行：在各 Server 文件中确认 `if __name__ == "__main__": mcp.run(transport="stdio")` 入口；通过 `python -m cryptotrader.mcp.servers.binance` 可启动并接受 stdio 输入

---

## Phase 7：CLI 命令

> 目标：实现 `arena mcp list` 和 `arena mcp call`，供开发者调试和验证（FR-011、FR-012）。

- [x] T025 修改 `src/cli/main.py`：新增 `mcp` Typer 子命令组（`app_mcp = typer.Typer()`，挂载为 `app.add_typer(app_mcp, name="mcp")`）
- [x] T026 实现 `arena mcp list` 命令：从 `load_config().mcp` 初始化 `MCPRegistry`；调用 `registry.list_tools()` 和 `registry.health_check()`；以 Rich 表格输出工具名、所属 Server、健康状态；`mcp.enabled=false` 时输出提示并以 Server 配置列表展示（不实际连接）
- [x] T027 实现 `arena mcp call <tool_name> [--args '{}']` 命令：解析 `--args` JSON 字符串为 `dict`；通过 `registry.call_tool(tool_name, args)` 调用；以 Rich JSON 格式美化输出结果；工具不存在时输出 `MCPToolNotFoundError` 错误信息并以非零退出码退出

---

## Phase 8：测试

> 目标：验证所有 FR 和 SC，覆盖率 MCPRegistry / MCPAdapter / MCPConfig ≥ 80%。

- [x] T028 [P] 新建 `tests/test_mcp_config.py`：测试 `MCPConfig` 默认值（`enabled=False`、`call_timeout_s=5.0`）；测试 `load_config()` 在无 `[mcp]` TOML 段时返回默认 `MCPConfig`；测试 `MCPServerConfig` 字段解析；测试 `config/default.toml` 中四个 Server 条目可正确反序列化（验证 SC-004）
- [x] T029 [P] 新建 `tests/test_mcp_registry.py`：测试 `register_server()` 构建工具索引；测试工具名冲突时抛出 `ValueError`；测试 `list_tools()` 返回有序工具名列表（≥ 13 个，验证 SC-001）；测试 `find_server()` O(1) 路由；测试 `health_check()` 对不可用 Server 返回 `False`；使用 mock `FastMCP` 实例，不依赖真实 Server 进程
- [x] T030 [P] 新建 `tests/test_mcp_adapter.py`：测试 `backtest_mode=True` 时直接调用 `python_fallback`，不触发任何 MCP 相关调用（验证 SC-007、FR-008）；测试 `mcp.enabled=False` 时直接调用 `python_fallback`（验证 SC-004）；测试 MCP 调用成功路径返回正确数据；测试 MCP 超时时 fallback 到 `python_fallback`（验证 FR-007）；测试 MCP 返回错误时 fallback 到 `python_fallback`；测试 `fallback_on_error=False` 时不 fallback，直接返回 `zero_value`；测试 fallback 路径记录 `[MCP fallback]` 日志（验证 US-4 场景 2）
- [x] T031 [P] 新建 `tests/test_mcp_servers.py`：使用 `pytest-asyncio` + mock Provider 函数测试四个 Server；测试 `BinanceMCPServer` 可通过 `mcp.list_tools()` 发现 3 个工具名（`binance_derivatives`、`binance_funding_rate`、`binance_klines`）；测试 `MacroMCPServer` 发现 4 个工具；测试 `OnchainMCPServer` 发现 4 个工具；测试 `NewsMCPServer` 发现 2 个工具；测试 `binance_derivatives` 调用 mock Provider，返回结构包含 `open_interest`、`long_short_ratio` 字段（验证 FR-013）；测试付费 API Key 未配置时 `onchain_derivatives` 返回 `{"data_available": false, ...}`（验证 US-5 场景 3）；测试 `truncate_response()` 对超过 50KB 响应追加 `"truncated": true`
- [x] T032 [P] 新建 `tests/test_mcp_cli.py`：测试 `arena mcp list` 在 `mcp.enabled=false` 时正常输出（不崩溃）；测试 `arena mcp call` 对不存在工具名以非零退出码退出；使用 Typer `CliRunner` 运行命令，mock `MCPRegistry`
- [x] T033 [P] 新建 `tests/test_mcp_cache.py`：测试 MCP 工具调用命中 SQLite 缓存时（`get_cached_or_none` 返回非 None），Provider 底层 HTTP 函数不被调用（验证 FR-005、US-1 场景 3）；mock `get_cached_or_none` 返回预设数据，断言 `httpx.AsyncClient` 未调用
- [x] T034 [P] 新建 `tests/test_mcp_snapshot_integration.py`：测试 `SnapshotAggregator.collect(adapter=None)` 与引入 MCP 前行为一致（验证 SC-002、SC-004）；测试 `SnapshotAggregator.collect(adapter=mock_adapter)` 在 `mcp.enabled=True` 时通过 adapter 获取数据；使用最小化 mock，不依赖真实 Provider 或网络
- [x] T035 运行完整测试套件 `pytest tests/`，确认通过率 100%，0 新增回归失败（验证 SC-002）；检查 `MCPRegistry`、`MCPAdapter`、`MCPConfig` 代码覆盖率 ≥ 80%（验证 SC-008）
- [x] T036 执行端到端验证：启动 `BinanceMCPServer`（stdio 模式），通过 `arena mcp call binance_derivatives --args '{"symbol": "BTC"}'` 调用，确认返回包含 `open_interest` 字段；验证 `arena mcp list` 输出工具总数 ≥ 13（验证 SC-001）；验证 `.mcp.json` 语法合法（`python -c "import json; json.load(open('.mcp.json'))"`）

---

## 任务依赖关系

```
T001, T002 (并行，基础设施)
  ↓
T003 (MCPConfig dataclass)
  ↓
T004, T005, T006 (并行：config.py 修改 / default.toml / utils)
  ↓
T006b (新增 Provider 函数：fetch_funding_rate_binance / fetch_klines_binance)
  ↓
T007, T008 (BinanceMCPServer，Phase 2 模板验证)
  ↓
T009, T010, T011 (并行：Macro / Onchain / News Server)
  ↓
T012 → T013 → T014 → T015 → T016 → T017a (MCPRegistry，串行)
  ↓
T018 → T019 → T017b (MCPAdapter，串行)
  ↓
T020, T021 (并行：SnapshotAggregator / collect_snapshot 节点)
  ↓
T022, T023, T024 (并行：.mcp.json + 外部兼容)
  ↓
T025 → T026 → T027 (CLI 命令，串行)
  ↓
T028, T029, T030, T031, T032, T033, T034 (并行：各测试文件)
  ↓
T035 → T036 (全套验证，串行)
```

---

## 标注说明

- `[P]` — 与同阶段其他 `[P]` 任务可并行执行
- 串行任务按编号顺序执行（存在数据依赖）
- Phase 1～3 完成后可立即进行 Phase 8 中 T028、T031 的部分用例编写（TDD 可选）
