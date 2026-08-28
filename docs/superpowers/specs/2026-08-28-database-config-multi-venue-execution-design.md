# 数据库配置与多交易平台执行架构设计

日期：2026-08-28
状态：已在讨论中确认，等待最终审阅

## 1. 背景

当前信号层已经把 Kronos、LLM 四智能体委员会和自定义组件统一为可插拔信号组件，融合结果表达平台无关的目标风险敞口。但是执行层仍围绕单一 `exchange_id`、单一账户持仓和混合了 OKX 特有行为的 `LiveExchange` 组织。配置同时来自 TOML、环境变量和数据库，存在多套来源和优先级。

这导致四个根本问题：

- 信号组件可插拔，交易平台却仍是单例。
- OKX 模拟盘、OKX 实盘和其他平台无法同时作为独立执行目标。
- 模拟资金与真实资金缺少明确隔离，不能安全聚合。
- 网页动态配置无法成为运行时唯一事实来源。

本次重构直接替换单平台执行和多来源配置架构，不保留旧 TOML、旧环境变量覆盖、旧 `exchange_id` 主链或兼容适配层。

## 2. 目标

- 交易平台实现统一适配器协议，可新增 OKX、Bybit 或自定义平台。
- 同一平台允许存在多个独立连接，例如 OKX Demo 和 OKX Live。
- 一个信号周期可以同时驱动模拟资金池和实盘资金池。
- 一个资金池可以按固定权重同时执行多个平台连接。
- 模拟权益与真实权益严格隔离，不能进入同一仓位计算。
- 数据库成为唯一运行配置源，网页修改从下一周期生效。
- 平台凭据加密存库，前端、API、日志和审计均不返回明文。
- 保留可配置 HITL，并让审批绑定精确的多平台计划。
- 多平台部分失败必须被准确记录，不伪装成整体成功。
- 删除平台名称硬编码和旧配置链，完成彻底重构。

## 3. 非目标

第一版不实现：

- 根据价格、手续费或深度自动选择平台。
- 平台失败后的自动权重重新分配。
- 平台间自动转账、套利或跨平台对冲。
- 多账户层级、子账户管理和凭据自动轮换。
- 多套策略草稿、发布流程或历史配置回滚。
- 旧 TOML、旧环境变量、旧 API 或旧领域模型兼容。
- 旧周期记录向多平台格式转换。
- 大量交易所特定错误码分支和罕见恢复流程。
- 使用真实资金完成自动化验收。

## 4. 核心领域模型

### 4.1 平台适配器

`VenueAdapter` 表示交易所协议实现，而不是具体账户：

```python
class VenueAdapter(Protocol):
    adapter_id: str

    def capabilities(self, environment: ConnectionEnvironment) -> VenueCapabilities:
        ...

    async def connect(self, connection: VenueConnection, credentials: SecretValue) -> VenueSession:
        ...
```

首批适配器：

```text
VenueAdapter
├── OkxVenueAdapter
├── BybitVenueAdapter
└── PaperVenueAdapter
```

OKX 和 Bybit 的公共 CCXT 行为放入轻量 `CcxtVenueBase`。合约面值、持仓模式、环境地址、参数和保护单接口留在各自适配器中。业务层不得判断平台名称。

### 4.2 平台连接

`VenueConnection` 表示一个实际可执行账户：

```python
@dataclass(frozen=True)
class VenueConnection:
    id: str
    label: str
    adapter_id: str
    environment: Literal["paper", "demo", "testnet", "live"]
    enabled: bool
    credential_ref: str | None
    leverage: int
    margin_mode: Literal["isolated", "cross"]
```

示例：

```text
okx-demo
okx-live-main
bybit-testnet
bybit-live-main
paper-local
```

同一个 `OkxVenueAdapter` 可以服务 `okx-demo` 和 `okx-live-main`。两个连接共享平台实现，但拥有独立凭据、余额、持仓、挂单、健康状态和审计记录。

连接的 `environment` 创建后不可切换。模拟盘转实盘通过创建新的实盘连接完成，避免同一身份在审计历史中改变资金性质。

### 4.3 平台能力

适配器通过能力声明替代平台名称判断：

```python
@dataclass(frozen=True)
class VenueCapabilities:
    market_types: frozenset[str]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: frozenset[str]
```

第一版规则：

- 衍生品连接必须支持平台侧保护单，才允许增加仓位。
- 不支持保护单的连接可以读取资产、执行减仓或清仓，但不能开新仓。
- 能力检查发生在生成执行计划之前。

### 4.4 执行资金池

`ExecutionBook` 是独立的资金、仓位、风控和审批范围：

```python
@dataclass(frozen=True)
class ExecutionBook:
    id: str
    label: str
    capital_scope: Literal["simulated", "real"]
    enabled: bool
    hitl_required: bool
    allocations: tuple[ConnectionAllocation, ...]
```

```python
@dataclass(frozen=True)
class ConnectionAllocation:
    connection_id: str
    enabled: bool
    weight: float
```

核心约束：

- 一个资金池只能包含相同 `capital_scope` 的连接。
- `paper`、`demo` 和 `testnet` 属于 `simulated`。
- `live` 属于 `real`。
- 每个启用资金池内，启用连接的权重必须显式合计为 `1.0`。
- 一个连接最多属于一个启用资金池，避免同一账户权益被重复分配。
- 只启用一个权重为 `1.0` 的连接就是单平台执行，不需要单独策略。

推荐默认资金池：

```text
simulation
├── okx-demo       40%
└── bybit-testnet  60%

live
├── okx-live-main     30%
└── bybit-live-main   70%
```

## 5. 平台无关信号与行情来源

信号行情来源与执行平台彻底分离：

```text
MarketDataSource
      ↓ SignalContext
Kronos / LLM Committee / Custom Components
      ↓ FusedSignal
TargetPosition
      ↓
Execution Books
```

`SignalContext` 不再包含执行 `exchange_id`、当前执行仓位、账户权益或投资组合。它只记录独立的 `market_data_source_id`、统一参考价格和信号组件所需的市场证据。信号可以使用 Binance 行情，同时在 OKX 和 Bybit 执行。当前仓位只在后续资金池分配和执行阶段读取。

各平台连接自己的最新价格只用于：

- 目标名义价值到订单数量的转换。
- 合约面值和精度处理。
- 平台级余额、保证金和滑点检查。
- 保护单价格精度处理。

执行价格不能反向改变已经生成的融合信号。

## 6. 仓位聚合与固定权重分配

`PortfolioAggregator` 并行读取资金池内所有连接，形成聚合视图并保留连接明细：

```python
@dataclass(frozen=True)
class BookPortfolioSnapshot:
    book_id: str
    capital_scope: str
    total_equity: float
    total_signed_notional: float
    connections: tuple[ConnectionPortfolioSnapshot, ...]
```

第一版只实现确定性的固定权重分配：

```text
连接目标名义仓位
= 资金池聚合可管理权益
× target_exposure
× 连接权重
```

示例：模拟资金池权益为 100,000 USDT，目标敞口为 `+0.5`：

```text
OKX Demo      40% → +20,000 USDT
Bybit Testnet 60% → +30,000 USDT
```

平台余额不足或保证金不足时，由平台级风控拒绝该连接计划。系统不自动把失败额度移动到其他平台。

分配接口保持可插拔：

```python
class AllocationPolicy(Protocol):
    id: str

    def allocate(
        self,
        target: TargetPosition,
        book: ExecutionBook,
        portfolio: BookPortfolioSnapshot,
    ) -> tuple[ConnectionTarget, ...]:
        ...
```

第一版 Registry 只注册 `WeightedAllocationPolicy`。智能路由属于后续独立实现。

## 7. 配置系统

### 7.1 单一事实来源

数据库是唯一运行配置源。运行时删除：

- `config/default.toml`
- `config/local.toml`
- `load_config()`
- TOML 递归合并
- `CRYPTOTRADER_*` 动态覆盖
- `AppConfig` TOML 构建链
- TOML 中的自定义组件 factory

应用只保留两个外部引导参数：

```text
DATABASE_URL
CONFIG_MASTER_KEY
```

数据库连接必须在读取其他配置前可用。`CONFIG_MASTER_KEY` 只用于加密数据库中的凭据，不能与密文保存在同一数据库。

### 7.2 简化存储

配置只使用两个核心表：

```text
runtime_config
├── id = "global"
├── revision BIGINT
├── document JSONB
└── updated_at TIMESTAMPTZ

venue_credentials
├── credential_ref TEXT PRIMARY KEY
├── encrypted_payload BYTEA
└── updated_at TIMESTAMPTZ
```

`runtime_config.document` 保存：

- LLM 网关与模型角色。
- 信号组件和信任权重。
- 行情来源。
- 风控、退出策略和调度器。
- 平台连接的非秘密配置。
- 模拟/实盘资金池与连接权重。
- HITL、通知和触发器。
- Redis 等启动后基础设施配置。

整个文档由严格类型的 `RuntimeConfigSnapshot` 校验并在一个事务中替换。任意保存都会递增全局 `revision`。

凭据使用 `CONFIG_MASTER_KEY` 通过 AES-GCM 加密。API 只返回是否已配置，不返回明文、密文或可逆内容。凭据更新同样递增配置 revision。

### 7.3 首次启动

```text
连接数据库
→ 执行 schema migration
→ 写入代码内置的最小默认文档
→ 标记 setup_required
→ 网页初始化向导
→ 配置 LLM、信号、平台、资金池、风控和调度
→ 测试连接
→ 激活系统
```

旧 TOML 不导入，不提供迁移器，不保留运行时 fallback。现有本机配置通过网页重新录入。

### 7.4 自定义组件发现

自定义信号组件和平台适配器通过 Python entry points 发现：

```text
cryptotrader.signal_components
cryptotrader.venue_adapters
```

代码安装仍需要重启，启用状态和参数进入数据库。运行时不从数据库加载任意 Python 路径，也不执行用户提交的代码字符串。

## 8. 周期数据流

一个信号周期只运行一次信号主链：

```text
1. 读取 RuntimeConfigSnapshot(revision)
2. 构建统一行情与 SignalContext
3. 运行 Kronos、LLM 委员会和自定义组件
4. 融合得到全局 target_exposure
5. 为每个启用 ExecutionBook 并行读取投资组合
6. 按连接权重生成 ConnectionTarget
7. 执行资金池级和连接级风控
8. 生成 BookExecutionProposal
9. 根据资金池配置进入 HITL 或直接执行
10. 各连接并行执行和安装保护单
11. 对账并记录每个资金池和连接结果
```

同一融合结果可以同时驱动两个资金池：

```text
FusedSignal
    ├── simulation → OKX Demo + Bybit Testnet
    └── live       → OKX Live + Bybit Live
```

模拟盘失败不阻止实盘，实盘失败也不修改模拟盘结果。两个资金池共享信号证据，但分别计算权益、目标仓位、风控、审批和执行结果。

## 9. 风控

风控分为两层：

```text
BookRiskGate
├── 总净敞口
├── 总毛敞口
├── 资金池回撤
└── 连接集中度

ConnectionRiskGate
├── 余额和可用保证金
├── 杠杆和保证金模式
├── 最小订单和精度
├── 平台能力
└── 当前持仓与保护状态
```

执行规则：

- 增加资金池风险时，所有目标连接必须先通过预检，才开始并行下单。
- 减少风险或清仓时，能连接的平台立即执行，不因其他连接不可用而阻止减仓。
- 风控可以拒绝连接计划或降低整个资金池目标，但不能静默修改连接权重。
- 模拟资金池和实盘资金池分别风控，模拟权益永远不进入实盘约束。

## 10. HITL

HITL 按资金池配置。推荐默认：

```text
simulation.hitl_required = false
live.hitl_required = true
```

审批展示一份资金池完整计划：

```text
资金池：live
全局目标：+50%
配置版本：42

OKX Live
当前：+8,000 USDT
目标：+20,000 USDT
增量：+12,000 USDT

Bybit Live
当前：0 USDT
目标：+30,000 USDT
增量：+30,000 USDT
```

审批约束：

- 审批绑定资金池、连接目标、保护价格和配置 revision。
- 配置 revision 变化后，未执行审批失效。
- 审批通过后执行原计划，不自动调整平台权重。
- 用户只能批准或拒绝完整资金池计划；修改权重需要保存新配置并重新运行周期。
- 模拟资金池可以已经完成，而实盘资金池仍处于 `awaiting_approval`。

资金池在执行前可以处于：

```text
awaiting_approval
approval_rejected
ready
```

进入执行后才产生 `completed`、`partial` 或 `failed` 终态。若 Simulation 已完成而 Live 仍待审批，周期状态保持 `awaiting_approval`，同时保留已经完成的 Simulation 结果。

## 11. 多平台执行

`ExecutionCoordinator` 管理资金池级并行执行：

```python
class ExecutionCoordinator:
    async def execute(self, proposal: BookExecutionProposal) -> BookExecutionResult:
        ...
```

每个连接由独立 `VenueExecutionService` 执行：

```text
读取连接当前状态
→ 计算目标差值
→ 下单
→ 安装新的平台侧保护单
→ 取消旧保护单
→ 对账
```

保护单替换遵循先确保新仓位受保护、再退休旧保护的原则。开仓成功但新保护单失败时，立即在同一连接尝试补偿平仓。

跨平台不存在原子事务。执行终态保持简单：

```text
completed  所有目标连接完成并受保护
partial    部分连接完成，部分连接失败
failed     没有连接完成目标
```

若补偿平仓失败或留下未保护仓位：

```text
requires_attention = true
```

系统不自动扩张其他平台仓位，也不实现复杂恢复状态机。下一周期根据平台真实持仓重新计算差值。

## 12. 平台环境实现

### 12.1 OKX

```text
demo → 模拟交易凭据 + x-simulated-trading: 1
live → 实盘凭据，无模拟头
```

两种环境使用同一个 `OkxVenueAdapter`，但连接、客户端、余额和审计完全独立。

### 12.2 Bybit

```text
testnet → api-testnet.bybit.com
demo    → api-demo.bybit.com
live    → api.bybit.com 或账户所属官方区域域名
```

第一版验收使用 Bybit Testnet。Demo 和 Live 共享 `BybitVenueAdapter` 的订单、持仓和保护单实现。

### 12.3 Paper

`PaperVenueAdapter` 使用内部模拟撮合，不需要凭据。它作为 `simulated` 连接参与相同资金池、分配、风控、HITL 和审计流程，不保留独立 paper 主链。

回测在启动时固定一份配置 revision，并使用只包含 Paper 连接的临时模拟资金池。历史行情提供者替换实时 `MarketDataSource`，其余信号、融合、分配、风控和执行契约保持一致。回测永远不会装配 Demo、Testnet 或 Live 连接。

## 13. 网页

### 13.1 初始化向导

首次启动通过网页依次配置：

```text
LLM
→ 信号组件
→ 行情来源
→ 平台连接
→ 模拟/实盘资金池
→ 风控与 HITL
→ 调度器
→ 连接测试与激活
```

### 13.2 平台连接页面

网页按平台品牌分组展示连接：

```text
OKX
├── OKX Demo       [模拟盘] [已连接]
└── OKX Live Main  [实盘]   [已连接]

Bybit
├── Bybit Testnet  [模拟盘] [已连接]
└── Bybit Live     [实盘]   [未配置]
```

支持：

- 新建和停用连接。
- 录入或更新凭据。
- 选择环境、杠杆和保证金模式。
- 测试连接。
- 查看余额、持仓、能力和最近健康状态。

凭据查询只显示 `configured`，保存后输入框清空。

### 13.3 执行资金池页面

```text
Simulation
├── OKX Demo       40%
├── Bybit Testnet  60%
└── HITL：关闭

Live
├── OKX Live Main  30%
├── Bybit Live     70%
└── HITL：开启
```

保存前验证：

- 连接环境与资金池类型一致。
- 启用连接权重合计为 100%。
- 连接凭据和连接测试有效。
- 实盘与模拟盘没有混入同一资金池。
- 用于增加衍生品仓位的连接支持原生保护单。

页面显示配置 revision，并提供基于当前权益和示例目标敞口的分配预览。

### 13.4 周期结果页面

周期详情按资金池和连接展示：

```text
信号证据与融合结果
├── Simulation
│   ├── OKX Demo
│   └── Bybit Testnet
└── Live
    ├── OKX Live Main
    └── Bybit Live
```

页面明确区分模拟和实盘，不把模拟收益合并到真实投资组合指标中。

## 14. API

第一版接口：

```text
GET  /api/config
PUT  /api/config

POST   /api/venue-connections
PUT    /api/venue-connections/{connection_id}
PUT    /api/venue-connections/{connection_id}/credentials
POST   /api/venue-connections/{connection_id}/test

GET /api/portfolio/books
GET /api/portfolio/books/{book_id}
```

`PUT /api/config` 完整替换配置文档并携带期望 revision。版本冲突返回 `409`，前端重新加载，不静默覆盖。

第一版不删除连接，只允许停用。停用前若仍被启用资金池引用，返回校验错误。这样 Journal 中的连接身份始终可解析，也不需要级联修复。

## 15. Journal

新执行使用新的多平台周期记录，不转换旧记录：

```text
multi_venue_cycles
├── cycle_id
├── config_revision
├── market_data_source_id
├── component_signals JSONB
├── fused_signal JSONB
├── target_position JSONB
├── book_results JSONB
├── cycle_status
├── execution_status
├── requires_attention
└── created_at
```

`book_results` 保存：

```text
book_id
capital_scope
portfolio_before
targets
risk_result
hitl_result
connection_executions
portfolio_after
status
```

每个连接执行记录保存连接 ID、环境、目标、订单、保护单、错误和最终持仓。凭据、Authorization 头和密钥永不进入 Journal。

旧周期表保留在数据库中但新代码不读取。删除旧表属于独立运维动作，不在本次重构范围。

## 16. 代码布局

```text
src/cryptotrader/
├── runtime_config/
│   ├── models.py
│   ├── repository.py
│   ├── defaults.py
│   └── secrets.py
├── market_sources/
│   ├── protocol.py
│   └── registry.py
├── venues/
│   ├── protocol.py
│   ├── models.py
│   ├── registry.py
│   ├── ccxt_base.py
│   ├── okx.py
│   ├── bybit.py
│   └── paper.py
├── portfolio/
│   └── aggregator.py
├── execution/
│   ├── allocation.py
│   ├── coordinator.py
│   ├── service.py
│   └── models.py
└── trading_cycle.py

src/api/routes/
├── config.py
├── venues.py
└── portfolio_books.py

web/src/pages/
├── setup/
├── settings/
│   ├── venues/
│   └── execution-books/
└── cycles/
```

## 17. 删除范围

重构完成后删除：

- TOML 运行配置和所有 TOML 配置加载代码。
- `AppConfig`、`ExchangeCredentials` 和 `ExchangesConfig` 旧配置模型。
- `CRYPTOTRADER_*` 配置覆盖及其测试。
- 全局 `exchange_id` 和 scheduler/请求/context 中的执行平台字段。
- 混合平台特例的 `LiveExchange`。
- `supports_protection_orders() == exchange_id == "okx"` 等平台名称判断。
- 单一账户装配和单平台 `ExecutionService` 入口。
- 独立 paper 执行主链。
- TOML 自定义组件 factory。
- 旧配置 API 和兼容 wrapper。

旧 Journal 表和生产历史数据不主动删除，但新运行时不读取它们。

## 18. 测试策略

不为旧配置和罕见平台错误编写兼容测试。自动化测试聚焦核心行为。

### 18.1 领域测试

- 模拟与实盘连接不能混入同一资金池。
- 一个连接不能同时属于两个启用资金池。
- 启用连接权重必须合计为 `1.0`。
- 一个 100% 连接等价于单平台。
- 固定权重目标名义仓位计算正确。
- 模拟权益不进入实盘资金池。
- 配置保存递增 revision，周期固定使用快照。
- 凭据加密后可由后端解密，API 模型不包含明文。

### 18.2 适配器契约测试

同一套契约分别验证 OKX 和 Bybit：

- 余额、仓位和订单标准化。
- 现货/合约数量与精度转换。
- 开仓、减仓、平仓和 `reduce_only`。
- 平台侧止损止盈。
- 保护单替换和对账。
- Demo/Testnet/Live 环境映射。

### 18.3 多平台集成测试

- 两个连接按 40/60 完成执行，结果为 `completed`。
- 一个连接成功、一个失败，结果为 `partial`，不重新分配。
- 保护单失败触发同连接补偿平仓。
- 补偿失败设置 `requires_attention=true`。
- Simulation 可以完成而 Live 等待 HITL。
- Live 审批后执行原计划并写入同一周期。
- 网页保存新权重后下一周期读取新 revision。
- API 和 Journal 不包含凭据。

### 18.4 真实验证

- 使用 Bybit Testnet 完成最小开仓。
- 安装并查询平台侧保护单。
- 完整平仓。
- 独立进程确认零持仓、零挂单和零保护单残留。
- OKX Demo 恢复后运行相同契约 canary。
- 模型网关恢复后运行真实四智能体分析、内部辩论和信号融合。

实盘代码路径通过适配器环境切换和只读连接检查验证，自动化验收不下真实资金订单。

## 19. 完成标准

- 运行时代码不读取 TOML，不存在配置优先级合并。
- 除数据库引导和配置加密密钥外，运行配置全部来自数据库。
- 业务层搜索不到 OKX、Bybit 等平台名称判断。
- `SignalContext` 和 `TargetPosition` 不依赖执行平台。
- 网页可以创建同一平台的模拟连接和实盘连接。
- 网页可以分别配置 Simulation 与 Live 资金池。
- 同一信号周期可以同时生成两个资金池的计划和结果。
- 模拟与实盘权益、风控、HITL 和审计严格隔离。
- 一个资金池可以按权重执行多个连接。
- 任何衍生品新增仓位都有平台侧保护单。
- 部分失败准确记录为 `partial`，不自动转移额度。
- Python 全量测试、Ruff、前端测试、类型检查和生产构建通过。
- Bybit Testnet 完成真实开仓、保护、平仓和零残留闭环。
- 真实 LLM 调用完成四智能体辩论与融合。

## 20. 与前置信号融合设计的关系

本设计保留前置规格已经实现的以下边界：

- Kronos、LLM 委员会和自定义组件使用统一信号协议。
- 组件权重确定性融合。
- 融合结果映射为目标仓位，而不是命令式买卖动作。
- LLM 四智能体保留内部辩论。
- HITL、硬风控和 Journal 保持独立层。

本设计替代前置规格中的：

- TOML 配置与 Profile 默认值。
- 单一当前仓位上下文。
- 单一 Executor 和单一实盘交易所。
- `SignalContext.exchange_id`。
- paper/live/backtest 仅通过单 Executor 注入的结构。

最终边界是：信号组件决定市场观点，融合层决定全局目标敞口，执行资金池隔离模拟与真实资本，分配策略把目标敞口映射到多个平台连接，各平台适配器独立完成受保护的实际执行。
