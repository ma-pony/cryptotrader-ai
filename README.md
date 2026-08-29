# CryptoTrader AI

CryptoTrader AI 是一个可插拔的加密资产信号融合与多平台执行系统。Kronos、LLM 四智能体委员会和自定义组件都是独立信号来源；它们以网页可调整的信任权重融合为一个目标仓位，再由各执行资金池独立风控、审批和执行。

## 运行时模型

运行配置唯一保存在数据库 `runtime_config` 中。进程只接受两个启动参数：`DATABASE_URL` 和 `CONFIG_MASTER_KEY`；后者用于加密平台凭据。没有 TOML、`.env` 合并或单平台模式开关。

每次网页保存都是整份严格校验的配置替换，并递增全局 revision。周期与 HITL 提案会冻结其 revision；保存失败时页面保留当前运行状态并显示错误，revision 变化会使待审批计划失效。

```text
市场数据 → Kronos / 四智能体内部辩论 / 自定义组件 → 信号融合 → 目标仓位
       → 每个执行资金池的风控、可选 HITL、连接分配 → 周期审计
```

`target_position` 使用 `side: long | short | flat` 与 `size_ratio: 0..1`。它是每个资金池在风控上限内的目标比例，不是杠杆或直接下单量；连接只执行当前仓位到目标仓位的差额。

## 首次启动

要求 Python 3.12+、Node.js 20+、uv 和 pnpm。准备 PostgreSQL，并生成一个 32 字节 AES-GCM 主密钥后启动 API：

```bash
uv sync --all-extras
export DATABASE_URL='postgresql+asyncpg://<db-user>:<db-password>@localhost:5432/cryptotrader'
export CONFIG_MASTER_KEY='base64 编码的 32 字节密钥'
uv run trader serve --port 8003
```

另一个终端启动网页：

```bash
cd web
pnpm install
pnpm dev
```

打开 `http://localhost:5173`。未配置时网页会自动进入初始化向导，按顺序配置：LLM、信号组件、市场数据、平台连接、执行资金池、风控、调度和通知；全部通过校验后才能激活运行时。

平台连接可同时使用 Paper、OKX、Bybit 或以后安装的适配器。Paper、Demo、Testnet 只能分配给 `simulated` 资金池；Live 连接只能分配给 `real` 资金池。一个连接最多属于一个启用资金池。每个资金池可分别打开 HITL，审批的是含保护价格和配置 revision 的完整计划。

## 容器启动

Compose 只向唯一的 API 运行时 owner 传入两个运行时变量：

```bash
export CONFIG_MASTER_KEY='base64 编码的 32 字节密钥'
docker compose up --build
```

PostgreSQL 地址由 Compose 组装后作为 `DATABASE_URL` 传入应用。首次访问网页完成配置；容器不会读取本地配置文件。

## 常用命令

```bash
uv run trader run --pair BTC/USDT
uv run trader backtest --pair BTC/USDT --start 2025-01-01 --end 2025-03-01
uv run trader journal log
uv run trader journal show <cycle-id>
```

回测只使用临时 Paper 资金池，不连接 Demo、Testnet 或 Live。真实模型与模拟盘验证应在网页完成配置后，使用已启用的测试环境连接；实盘连接只允许只读检查，禁止自动化真实资金订单。

## 验收金丝雀

完成网页配置后，可对数据库中已启用的 Paper、Demo 或 Testnet 连接运行一次最小仓位闭环。脚本不会接收或输出平台凭据；它只取消带有本次 canary 标记的订单，并在独立新进程重新连接审计零残留。任何无法确认订单归属、实际成交量或清理结果的失败都会返回非零和 `requires_attention`，不会猜测性修改账户状态。

```bash
uv run python scripts/venue_canary.py --connection bybit-testnet --pair BTC/USDT:USDT
uv run python scripts/signal_canary.py --pair BTC/USDT
```

`signal_canary.py` 只收集真实行情，运行 Kronos 和四智能体内部辩论、融合及目标仓位；它不进入任何执行资金池。Live 连接仅可显式作只读检查：

```bash
uv run python scripts/venue_canary.py --connection bybit-live --pair BTC/USDT:USDT --live-read-only
```

## 验证

```bash
uv run pytest --no-cov -q
uv run ruff check src tests scripts
uv run python scripts/import_smoke.py
cd web && pnpm test && pnpm typecheck && pnpm lint
```

架构约束与数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)。
