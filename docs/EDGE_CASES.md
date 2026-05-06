# CryptoTrader AI — 边缘情况与开发规范

---

## 1. 容错处理

| 场景 | 处理方式 |
|------|----------|
| LLM 调用失败 | Agent 返回 `is_mock=True, confidence=0.1, direction=neutral` |
| 全部 Agent mock | Verdict 强制 `hold`，不交易 |
| Redis 不可用 | 若 Redis 已配置但不可用，保守拒绝交易（而非跳过）；若从未配置，降级到内存状态管理。拒单 reason 携带 `[ConnectionError: ...]` 等 root cause |
| 辩论门控跳过 | `debate_skipped=True` 写入 state，Verdict 可能降级为 weighted 投票 |
| PostgreSQL 不可用 | 降级到内存日志（最多 10,000 条） |
| 交易所 API 超时 | LiveExchange 3 次重试，指数退避 |
| 余额不足 | PaperExchange / LiveExchange 抛 `ValueError`；`OrderManager.place` 把 `error_type` / `error_msg` 写入 result，`place_order` 透出到 `state.data.execution_status` |
| JSON 解析失败 | `_extract_json` 平衡括号提取，处理 markdown fence |
| 数据源 API 失败 | 各 collector 独立 try/catch，返回默认值 |
| 熔断器触发 | Verdict 强制 `hold`，需手动 `arena risk reset-breaker` |
| **OKX `instType=FUTURES` / `SPOT` 端点偶发 5xx**（2026-05-06 实测） | ccxt 默认 `load_markets` 用 `asyncio.gather` 并发拉 4 类 instType，任一失败 → wrap 成 `ExchangeNotAvailable` → 整个 cycle 挂掉。所有 ccxt 实例统一限定 `options.fetchMarkets = ["spot", "swap"]` 排除 future / option |
| **Spot 现货账户尝试做空且无库存** | `_build_entry_order` 早拦：返回 None + 写 `state.data.execution_error = "spot_short_no_inventory: ..."` → `execution_status` 透到 /decisions reject_reason，不再走完整下单流程产生 `ValueError` |
| **`set_leverage` 在已开仓 symbol 上失败** | 警告吞掉，`_leverage_attempts` 计数；连续 `_LEVERAGE_RETRY_LIMIT = 3` 次失败才进缓存。让用户改 `[exchanges.okx] leverage = N` 后下次仓位释放后自动生效，不需重启 |
| **`portfolio_unknown` / `redis_unavailable` 拒单事件** | reason 字段携带 `[error_type: msg]` 后缀，下次再触发可直接定位（不再需要 grep traceback） |

## 2. 关键设计决策

| 决策 | 原因 |
|------|------|
| 固定 2 轮辩论（非动态收敛） | 动态收敛会人为趋同，2 轮允许真实分歧 |
| 辩论门控：强共识或共同困惑时跳过辩论 | 将 LLM 调用从 13 次减少到 4-5 次，无质量损失 |
| 困惑 vs 分歧：离散度阈值区分 | 低均值 + 低离散度 = 困惑（跳过）；低均值 + 高离散度 = 分歧（辩论） |
| 辩论跳过 + 置信度平坦 + 无熔断 → Verdict 降级为加权投票 | 可安全节省 1 次 LLM 调用 |
| 辩论轮次内各 Agent 并行执行（asyncio.gather） | 每轮延迟降低 4 倍 |
| `close` 动作豁免全部风控 | 减仓是降低风险，不应被风控阻断 |
| position_scale 连续映射 | 三档离散映射丢失 AI 的精细判断 |
| ToolAgent 回测跳过工具 | 避免前视偏差 + 减少无效 HTTP 超时 |
| FnG limit 动态计算 | 固定 400 天不够覆盖远期回测 |
| 每个 FRED 序列独立速率限制 | 共享 key 导致第二个序列永远跳过 |
| Config 首次加载后缓存 | 避免 mid-run 配置变更导致不一致 |
| 信号生成在 bar[i]，执行在 bar[i+1] 开盘 | 消除前视偏差 |
| **Equity = cash + spot×price + perp.unrealized_pnl** （非名义） | 衍生品名义不是资产，margin 已在 cash 里。`amount × price`（abs/signed）都会 ±notional 误差几万美金。详见 commit 175089c |
| **execution_status 与 risk_gate 解耦** | 执行层失败（spot_short_no_inventory / Insufficient ETH 等）不再覆写 `risk_gate`；前者污染风控通过率分析。`/decisions` reject_reason 优先 gate 真拒单，再 fallback `execution_status` |
| **配置驱动 perp leverage** | `[exchanges.okx] leverage / margin_mode` → `LiveExchange._ensure_leverage` 在首次下单时调 `set_leverage(N, symbol, posSide=long+short)`，long_short_mode 必须两边都设 |
| **`total_return` baseline 可配** | `[portfolio] initial_capital` > 0 时优先；否则用首次 portfolio_snapshots。中途充值 / 提现的用户应当 pin 本金 |

## 3. 已知限制

- CorrelationCheck 使用硬编码 14 组，非动态计算
- `verify=False` 存在于约 15 个外部 API 调用中（部分 provider）
- `pm.get_portfolio()`（DB 路径）只能给出最近一次 sync 时的 `unrealized_pnl`，不实时；live API 不可达时显示的 equity 略滞后
- `total_return_pct` baseline 是首次 snapshot（除非 `[portfolio] initial_capital` > 0），不感知中途充值 / 提现 cash flow
- spot pair 买入持仓后又决策 short → 走"卖现货"路径（实际是减仓 / 平仓），并不是真做空（推荐统一切到 perp pair）

> 历史备注：`graph_supervisor.py` / `langchain_agents.py` / `agents/skills.py` / `agents/tools.py` 等实验性 supervisor 架构已于 2026-04-28 删除，仅保留 `build_trading_graph()` / `build_lite_graph()` / `build_debate_graph()` 主路径。

---

## 4. 开发规范

### 4.1 代码质量

```bash
make lint          # uv run ruff check src/ tests/  → 必须零错误
make test          # uv run pytest tests/ -v        → 2003 pass, 2 skip
```

- **禁止 `noqa` 注释** — 遇到 C901 必须重构，遇到 F401 必须删除或 `__all__`
- **C901 阈值 = 10** — 超过时拆分辅助函数
- **异步测试**：`asyncio_mode = "auto"`，无需 `@pytest.mark.asyncio`
- **Mock LLM**：`patch("langchain_openai.ChatOpenAI.ainvoke")` → `AIMessage(content=...)`
- **导入路径**：`from cryptotrader.*`（非 `src.cryptotrader.*`）
- **必须用 `uv run pytest`**（Python 3.12 venv），非裸 `pytest`
- **禁止 `logger.debug(..., exc_info=True)` 吞异常** — 详见 [logging-conventions.md](./logging-conventions.md)；`tests/test_logging_conventions.py` CI 守护此规则

### 4.2 依赖管理

```bash
uv add <package>              # 添加到 pyproject.toml
uv add --group dev <package>  # 添加到 dev 组
uv sync                       # 从 lockfile 安装
uv lock                       # 重新生成 lockfile
```

**永远不要** `pip install`。

### 4.3 基础设施

```bash
docker compose up -d           # PostgreSQL 16 + Redis 7
arena migrate                  # 创建数据库表
arena sync                     # 同步历史数据
```
