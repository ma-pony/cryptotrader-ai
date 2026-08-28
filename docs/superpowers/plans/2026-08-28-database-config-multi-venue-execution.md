# 数据库配置与多交易平台执行重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用数据库唯一配置源和可插拔交易平台层彻底替换 TOML、多环境变量覆盖、单一交易所与独立 Paper 主链，让同一次 Kronos/LLM/自定义组件融合结果可以分别驱动多个模拟盘与实盘资金池，并在网页动态管理连接、权重、HITL 和运行状态。

**Architecture:** 信号层只读取独立行情并生成一份平台无关 `TargetPosition`；`PortfolioAggregator`、`WeightedAllocationPolicy`、两级风控和 `ExecutionCoordinator` 按 `ExecutionBook` 将同一目标拆分到多个 `VenueConnection`。平台差异由 `VenueAdapter` 能力和 `VenueSession` 契约封装。`runtime_config` 是配置唯一事实来源，`venue_credentials` 使用 AES-GCM 加密，周期固定配置 revision；模拟与真实资金严格隔离，跨平台结果允许准确表达 `completed|partial|failed`。

**Tech Stack:** Python 3.12、Pydantic v2、dataclasses、asyncio、SQLAlchemy async、AES-GCM/cryptography、CCXT、FastAPI、PostgreSQL/SQLite 测试、React 19、TypeScript、TanStack Query、Zod、Vitest、pytest、Ruff。

**Spec:** `docs/superpowers/specs/2026-08-28-database-config-multi-venue-execution-design.md`

## Global Constraints

- 不保留 `AppConfig`、`load_config()`、`ExchangeCredentials`、`ExchangesConfig`、`LiveExchange`、TOML 加载、`CRYPTOTRADER_*` 覆盖、旧字段 fallback、alias、wrapper 或双装配入口。
- 运行时外部引导参数只有 `DATABASE_URL` 和 `CONFIG_MASTER_KEY`；其余配置从 `runtime_config` 读取。测试可以直接注入 repository、snapshot 和 fake session，不读取进程环境。
- `runtime_config.document` 是一次整体校验、一次事务替换的严格类型文档；任意文档或凭据写入都递增全局 revision。
- 凭据使用 32 字节 AES-GCM 主密钥加密；API、日志、异常、Journal、测试快照和前端状态不得出现明文、密文或 Authorization 头。
- `SignalContext` 只包含行情证据，不包含执行平台、账户权益、当前执行仓位或投资组合。
- 一个周期只运行一次 Kronos、LLM 四智能体内部辩论和自定义组件融合；Simulation 与 Live 共享该结果，但独立读取权益、风控、HITL 和执行。
- `paper|demo|testnet` 只能进入 `simulated` 资金池，`live` 只能进入 `real` 资金池；一个连接最多属于一个启用资金池。
- 第一版只实现固定权重分配，不做价格择优、失败重分配、跨平台转账、套利或复杂恢复状态机。
- 增加衍生品风险前，全部目标连接必须支持平台侧保护单并通过预检；减仓和清仓不因其他连接失败而被阻塞。
- 跨平台执行没有伪原子事务。部分成功必须返回 `partial`；未保护仓位或补偿平仓失败必须设置 `requires_attention=true`。
- HITL 绑定资金池完整计划、保护价格和配置 revision；revision 变化使待审批计划失效。
- Backtest 只装配临时 Paper 资金池，不连接 Demo、Testnet 或 Live。
- 不迁移旧 TOML，不转换旧周期记录。旧数据库表可物理保留，但新运行时不得读取。
- 每个任务结束都必须可导入、可启动、可测试；删除旧代码产生的错误通过迁移调用方解决。
- 自动化测试不调用真实 LLM，不连接真实交易所。真实验收只在 Bybit Testnet、OKX Demo 和真实模型网关执行；实盘环境只做只读连接检查，禁止自动化真实资金订单。
- 执行本文所有 shell 命令时都按项目 `RTK.md` 加 `rtk` 前缀；代码块中的命令保留原生命令形态，执行器不得绕过此前缀。
- 所有项目 Markdown 使用简体中文。

## 设计覆盖矩阵

| 设计章节 | 实施任务 | 验收证据 |
|---|---|---|
| 4 核心领域模型 | Task 1、4、7 | RuntimeConfig、VenueConnection、ExecutionBook、组合与分配领域测试 |
| 5 平台无关信号 | Task 3、11 | market-only `SignalContext`、一次信号运行、多资金池复用测试 |
| 6 仓位与固定权重 | Task 7 | 100% 单连接、40/60、多空与零目标精确数学测试 |
| 7 数据库配置 | Task 1–3、12–13 | AES-GCM、revision CAS、setup_required、entry points、API 脱敏测试 |
| 8 周期数据流 | Task 7–11 | 同一 FusedSignal 驱动 Simulation/Live、互不阻塞集成测试 |
| 9 两级风控 | Task 8 | 全连接预检、减仓优先、风险 cap 不改权重测试 |
| 10 HITL | Task 10–12 | per-book 审批、revision 失效、原计划恢复执行测试 |
| 11 多平台执行 | Task 8–9 | completed/partial/failed、保护失败补偿、attention 测试 |
| 12 平台环境 | Task 4–6 | OKX Demo/Live、Bybit Testnet/Demo/Live、Paper 契约测试 |
| 13 网页 | Task 16–17 | 初始化向导、连接、资金池、分配预览、周期详情 UI 测试 |
| 14 API | Task 12 | revision 409、停用约束、凭据写入和连接测试接口 |
| 15 Journal | Task 10、17 | `multi_venue_cycles` round-trip、连接明细和秘密零泄漏 |
| 16–17 布局与删除 | Task 13–15、18 | 所有入口迁移、旧符号与旧路径零命中 |
| 18 测试策略 | Task 1–19 | 领域、契约、集成、前端、容器、导入和真实 canary |
| 19 完成标准 | Task 19–20 | 全量绿灯、真实模型/模拟交易闭环、零残留后才允许合并推送 |

---

## 文件结构锁定

重构后的核心职责固定如下。任务中不得临时新增第二套配置模型、平台抽象或执行入口。

```text
src/cryptotrader/
├── bootstrap.py                         # 从 DB snapshot 组装唯一 Runtime
├── runtime.py                           # Runtime 生命周期与连接关闭
├── trading_cycle.py                     # 一次信号、多资金池 application service
├── runtime_config/
│   ├── __init__.py
│   ├── models.py                        # RuntimeConfigDocument/Snapshot 严格模型
│   ├── defaults.py                      # 代码内最小 setup_required 文档
│   ├── repository.py                    # runtime_config CAS 与 revision
│   └── secrets.py                       # AES-GCM CredentialVault
├── market_sources/
│   ├── __init__.py
│   ├── protocol.py                      # MarketDataSource
│   ├── default.py                       # 当前内置行情聚合实现
│   └── registry.py                      # 内置与 entry point 发现
├── signals/
│   ├── models.py                        # market-only SignalContext
│   └── registry.py                      # 内置与 entry point 发现
├── venues/
│   ├── __init__.py
│   ├── models.py                        # connection/capability/session DTO
│   ├── protocol.py                      # VenueAdapter/VenueSession
│   ├── registry.py                      # 内置与 entry point 发现
│   ├── ccxt_base.py                     # 纯 CCXT 公共机制
│   ├── okx.py                           # OKX 环境、数量、保护单
│   ├── bybit.py                         # Bybit 环境、数量、保护单
│   └── paper.py                         # 内部模拟撮合 session
├── portfolio/
│   ├── models.py                        # connection/book snapshot
│   └── aggregator.py                    # 资金池并行读取与隔离聚合
├── execution/
│   ├── models.py                        # targets/proposals/results
│   ├── allocation.py                    # WeightedAllocationPolicy
│   ├── planner.py                       # connection target → order delta
│   ├── service.py                       # VenueExecutionService
│   └── coordinator.py                   # book 并行执行与结果聚合
├── risk/
│   ├── models.py                        # book/connection risk DTO
│   └── gate.py                          # BookRiskGate/ConnectionRiskGate
├── hitl/
│   ├── models.py                        # BookApproval
│   └── store.py                         # revision-bound approval persistence
└── journal/
    ├── models.py                        # MultiVenueCycleRecord
    └── store.py                         # multi_venue_cycles only

src/api/routes/
├── config.py                            # GET/PUT /api/config
├── venues.py                            # connection/credential/test endpoints
├── portfolio_books.py                   # per-book portfolio endpoints
└── cycles.py                            # multi-venue cycle list/detail

web/src/
├── hooks/
│   ├── use-runtime-config.ts
│   ├── use-venue-connections.ts
│   ├── use-portfolio-books.ts
│   └── use-multi-venue-cycles.ts
├── pages/
│   ├── setup/
│   ├── settings/venues/
│   ├── settings/execution-books/
│   └── cycles/
└── types/api.schema.ts                  # 新 API 唯一手写 schema

scripts/
├── import_smoke.py                      # 生产入口导入门禁
├── venue_canary.py                      # DB connection 驱动模拟盘闭环
└── signal_canary.py                     # 真实四智能体辩论与融合
```

固定依赖方向：`domain models → pure policies/services → infrastructure adapters/repositories → runtime/bootstrap → API/CLI/Scheduler/Web`。领域层不得导入 FastAPI、SQLAlchemy、CCXT 或 React。

## 固定领域与接口契约

后续任务只能实现这些契约，不能用字典在层间临时传递核心业务状态：

```python
@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    revision: int
    document: RuntimeConfigDocument
    updated_at: datetime

    @property
    def setup_required(self) -> bool:
        return not self.document.system.active


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


@dataclass(frozen=True)
class ExecutionBook:
    id: str
    label: str
    capital_scope: Literal["simulated", "real"]
    enabled: bool
    hitl_required: bool
    allocations: tuple[ConnectionAllocation, ...]


class VenueAdapter(Protocol):
    adapter_id: str

    def capabilities(self, environment: ConnectionEnvironment) -> VenueCapabilities: ...

    async def connect(
        self,
        connection: VenueConnection,
        credentials: CredentialPayload | None,
    ) -> VenueSession: ...


class VenueSession(Protocol):
    connection_id: str

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot: ...
    async def fetch_quote(self, pair: Pair) -> VenueQuote: ...
    async def place_order(self, intent: OrderIntent) -> NormalizedOrder: ...
    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState: ...
    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None: ...
    async def list_open_state(self, pair: Pair) -> OpenVenueState: ...
    async def close(self) -> None: ...


class AllocationPolicy(Protocol):
    id: str

    def allocate(
        self,
        target: TargetPosition,
        book: ExecutionBook,
        portfolio: BookPortfolioSnapshot,
    ) -> tuple[ConnectionTarget, ...]: ...


class ExecutionCoordinator:
    async def execute(self, proposal: BookExecutionProposal) -> BookExecutionResult: ...
```

状态语义固定：执行前只有 `awaiting_approval|approval_rejected|ready`；执行后只有 `completed|partial|failed`。周期可以在 Simulation 已完成时仍是 `awaiting_approval`，不得丢弃已完成结果。

---

### Task 1: 建立 RuntimeConfig 严格领域模型与最小默认文档

**Files:**
- Create: `src/cryptotrader/runtime_config/__init__.py`
- Create: `src/cryptotrader/runtime_config/models.py`
- Create: `src/cryptotrader/runtime_config/defaults.py`
- Create: `src/cryptotrader/venues/__init__.py`
- Create: `src/cryptotrader/venues/models.py`
- Create: `src/cryptotrader/execution/models.py`
- Create: `tests/factories/runtime_config.py`
- Test: `tests/test_runtime_config_models.py`
- Modify: `tests/factories/__init__.py`

**Interfaces:**
- Produces: `RuntimeConfigDocument`、`RuntimeConfigSnapshot`、`SystemConfig`、`MarketDataConfig`、`SignalConfig`、`ExecutionConfig`、`SchedulerConfig`。
- Produces: `venues.models.VenueConnection` 与 `execution.models.ExecutionBook/ConnectionAllocation` 的基础不可变模型；后续 Task 4、7 只增加平台能力和执行状态，不移动或复制这些类型。
- Produces: `minimal_runtime_document()`，只含内置 market source、Kronos/LLM 组件元数据、空连接/资金池并且 `system.active=False`。
- Consumes: 现有 `SignalProfile` 的融合阈值和权重语义；不读取 `config.py`。

- [ ] **Step 1: 写失败测试锁定完整文档和跨字段约束**

```python
def test_minimal_document_requires_setup_and_contains_no_connection():
    snapshot = RuntimeConfigSnapshot(1, minimal_runtime_document(), NOW)
    assert snapshot.setup_required is True
    assert snapshot.document.execution.connections == ()
    assert snapshot.document.execution.books == ()


def test_document_rejects_connection_in_two_enabled_books():
    document = runtime_document(
        connections=(connection("paper-local", "paper"),),
        books=(
            book("simulation-a", "simulated", allocation("paper-local", 1.0)),
            book("simulation-b", "simulated", allocation("paper-local", 1.0)),
        ),
    )
    with pytest.raises(ValueError, match="one enabled book"):
        validate_runtime_document(document, installed_signal_ids={"kronos", "llm_committee"}, installed_adapter_ids={"paper"})


@pytest.mark.parametrize(
    ("environment", "scope"),
    [("paper", "real"), ("demo", "real"), ("testnet", "real"), ("live", "simulated")],
)
def test_document_rejects_environment_scope_mismatch(environment, scope):
    with pytest.raises(ValueError, match="capital_scope"):
        validate_runtime_document(
            runtime_document(
                connections=(connection("venue", environment),),
                books=(book("book", scope, allocation("venue", 1.0)),),
            ),
            installed_signal_ids={"kronos", "llm_committee"},
            installed_adapter_ids={"paper", "okx", "bybit"},
        )


def test_document_requires_enabled_allocation_weights_to_equal_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_runtime_document(runtime_document_with_weights(0.4, 0.5), INSTALLED_SIGNALS, INSTALLED_ADAPTERS)
```

- [ ] **Step 2: 运行测试确认新包不存在**

Run: `uv run pytest tests/test_runtime_config_models.py --no-cov -q`

Expected: FAIL，包含 `ModuleNotFoundError: cryptotrader.runtime_config`。

- [ ] **Step 3: 实现严格文档，不从旧配置模型继承**

`RuntimeConfigDocument` 使用 `ConfigDict(extra="forbid", frozen=True)`。顶层字段一次锁定为：

```python
class RuntimeConfigDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: SystemConfig
    market_data: MarketDataConfig
    llm: LlmConfig
    signals: SignalConfig
    risk: RiskConfig
    execution: ExecutionConfig
    hitl: HitlConfig
    scheduler: SchedulerConfig
    triggers: TriggerConfig
    notifications: NotificationConfig
    infrastructure: InfrastructureConfig
```

将仍被使用的 LLM、风险、调度、通知和触发字段从 `config.py` 搬到该文件并改成上述组合关系；不复制 `AppConfig`、`engine`、`exchange_id`、`exchanges`、TOML alias 或环境变量元数据。`SignalConfig` 直接包含组件权重、`neutral_threshold`、`max_target_ratio`、ATR 退出参数；`ExecutionConfig` 直接包含 connections、books 和 `allocation_policy="weighted"`。

- [ ] **Step 4: 实现一次性全局校验**

```python
def validate_runtime_document(document, installed_signal_ids, installed_adapter_ids) -> None:
    validate_signal_profile(document.signals.to_profile(revision=0), installed_signal_ids)
    _validate_connection_ids(document.execution.connections, installed_adapter_ids)
    _validate_book_weights(document.execution.books)
    _validate_capital_scopes(document.execution.connections, document.execution.books)
    _validate_unique_enabled_membership(document.execution.books)
    _validate_active_document(document)
```

`system.active=True` 时额外要求 market source 可解析、至少一个启用信号组件、至少一个启用资金池、每个启用非 Paper 连接具有 `credential_ref`；最小默认文档因 `active=False` 可以合法保存。

- [ ] **Step 5: 运行领域测试**

Run: `uv run pytest tests/test_runtime_config_models.py tests/test_signal_profile.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/cryptotrader/runtime_config src/cryptotrader/venues src/cryptotrader/execution/models.py tests/factories tests/test_runtime_config_models.py
git commit -m "feat: add strict runtime configuration domain"
```

---

### Task 2: 实现 AES-GCM 凭据库与 revision CAS Repository

**Files:**
- Create: `src/cryptotrader/runtime_config/secrets.py`
- Create: `src/cryptotrader/runtime_config/repository.py`
- Test: `tests/test_runtime_config_secrets.py`
- Test: `tests/test_runtime_config_repository.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `CredentialPayload`、`CredentialState`、`CredentialVault`。
- Produces: `RuntimeConfigRepository.get_or_create()`、`replace(expected_revision, document)`、`put_credentials(expected_revision, credential_ref, payload)`、`credential_state()`、`reveal_credentials()`。
- Produces: `RevisionConflict(expected, actual)` 和 `CredentialNotConfigured`。
- Consumes: Task 1 `RuntimeConfigDocument/Snapshot`。

- [ ] **Step 1: 先把 cryptography 声明为直接依赖**

Run: `uv add 'cryptography>=46.0'`

Expected: `pyproject.toml` 的直接依赖出现 `cryptography`，lockfile 更新。

- [ ] **Step 2: 写加密与泄漏失败测试**

```python
def test_vault_round_trip_uses_unique_nonce_and_reference_as_aad():
    vault = CredentialVault(base64.urlsafe_b64encode(b"k" * 32).decode())
    payload = CredentialPayload(api_key="key", secret="secret", passphrase="phrase")
    first = vault.seal("okx-demo", payload)
    second = vault.seal("okx-demo", payload)
    assert first != second
    assert b"secret" not in first
    assert vault.open("okx-demo", first) == payload
    with pytest.raises(InvalidTag):
        vault.open("other-ref", first)


def test_invalid_master_key_fails_closed():
    with pytest.raises(ValueError, match="32-byte"):
        CredentialVault("not-a-valid-key")
```

- [ ] **Step 3: 写 repository revision 和事务失败测试**

```python
async def test_replace_is_compare_and_swap_and_increments_revision(repository):
    current = await repository.get_or_create()
    saved = await repository.replace(current.revision, active_document())
    assert saved.revision == current.revision + 1
    with pytest.raises(RevisionConflict) as error:
        await repository.replace(current.revision, active_document())
    assert (error.value.expected, error.value.actual) == (current.revision, saved.revision)


async def test_credential_update_and_revision_change_are_one_transaction(repository):
    before = await repository.get_or_create()
    after = await repository.put_credentials(before.revision, "okx-demo", credential_payload())
    assert after.revision == before.revision + 1
    assert (await repository.credential_state("okx-demo")).configured is True
    assert await repository.reveal_credentials("okx-demo") == credential_payload()


async def test_repository_api_objects_never_serialize_secrets(repository):
    await seed_credential(repository, "bybit-testnet", "visible-marker")
    state = await repository.credential_state("bybit-testnet")
    assert "visible-marker" not in json.dumps(asdict(state))
```

- [ ] **Step 4: 运行测试确认失败**

Run: `uv run pytest tests/test_runtime_config_secrets.py tests/test_runtime_config_repository.py --no-cov -q`

Expected: FAIL，缺少 vault 和 repository。

- [ ] **Step 5: 实现固定密文 envelope**

```python
class CredentialVault:
    VERSION = b"\x01"

    def __init__(self, encoded_key: str) -> None:
        key = base64.urlsafe_b64decode(encoded_key.encode())
        if len(key) != 32:
            raise ValueError("CONFIG_MASTER_KEY must encode a 32-byte key")
        self._cipher = AESGCM(key)

    def seal(self, credential_ref: str, payload: CredentialPayload) -> bytes:
        nonce = os.urandom(12)
        plaintext = payload.model_dump_json().encode()
        ciphertext = self._cipher.encrypt(nonce, plaintext, credential_ref.encode())
        return self.VERSION + nonce + ciphertext

    def open(self, credential_ref: str, envelope: bytes) -> CredentialPayload:
        if envelope[:1] != self.VERSION or len(envelope) < 30:
            raise ValueError("unsupported credential envelope")
        plaintext = self._cipher.decrypt(envelope[1:13], envelope[13:], credential_ref.encode())
        return CredentialPayload.model_validate_json(plaintext)
```

`CredentialPayload.__repr__` 固定返回 `CredentialPayload(**redacted**)`，日志不得对它调用 `model_dump()`。

- [ ] **Step 6: 实现两个表和事务 CAS**

`runtime_config` 固定一行 `id="global"`；SQLite 测试用 JSON，PostgreSQL 用 JSONB variant。`replace()` 在同一个 `UPDATE ... WHERE revision=:expected` 中写 document、`revision=revision+1`、`updated_at`，`rowcount=0` 时查询实际 revision 并抛出 `RevisionConflict`。`put_credentials()` 在同一事务 upsert 密文并 CAS 递增全局 revision；CAS 失败整体回滚。

- [ ] **Step 7: 运行测试并检查日志表示**

Run: `uv run pytest tests/test_runtime_config_secrets.py tests/test_runtime_config_repository.py --no-cov -q`

Expected: PASS；输出不含测试使用的 `visible-marker`。

- [ ] **Step 8: 提交**

```bash
git add pyproject.toml uv.lock src/cryptotrader/runtime_config tests/test_runtime_config_secrets.py tests/test_runtime_config_repository.py
git commit -m "feat: persist encrypted runtime configuration"
```

---

### Task 3: 用 Python entry points 发现信号组件和行情来源

**Files:**
- Create: `src/cryptotrader/market_sources/__init__.py`
- Create: `src/cryptotrader/market_sources/protocol.py`
- Create: `src/cryptotrader/market_sources/default.py`
- Create: `src/cryptotrader/market_sources/registry.py`
- Modify: `src/cryptotrader/signals/models.py`
- Rewrite: `src/cryptotrader/signals/registry.py`
- Modify: `src/cryptotrader/signals/components/kronos.py`
- Modify: `src/cryptotrader/signals/components/llm_committee.py`
- Modify: `pyproject.toml`
- Create: `tests/factories/fake_market_source_plugin.py`
- Modify: `tests/factories/fake_signal_plugin.py`
- Test: `tests/test_plugin_entry_points.py`
- Test: `tests/test_market_only_signal_context.py`

**Interfaces:**
- Produces: `MarketDataSource.collect(pair, as_of, requirements) -> SignalContext`。
- Produces: `MarketSourceRegistry.discover()` 和 `SignalComponentRegistry.discover(config, events)`。
- Consumes: Task 1 `RuntimeConfigSnapshot` 的 signal/market source 配置。
- Removes from domain: `SignalContext.exchange_id/equity/current_position/portfolio` 和 TOML factory path。

- [ ] **Step 1: 写 market-only context 失败测试**

```python
def test_signal_context_contains_market_evidence_only():
    fields = {field.name for field in dataclasses.fields(SignalContext)}
    assert fields == {
        "pair", "as_of", "market_data_source_id", "market_type",
        "current_price", "atr", "snapshots",
    }
    assert not fields & {"exchange_id", "equity", "current_position", "portfolio"}
```

- [ ] **Step 2: 写 entry point 发现与重复 ID 测试**

```python
def test_registry_discovers_installed_component_entry_points(monkeypatch):
    monkeypatch.setattr(metadata, "entry_points", fake_entry_points("cryptotrader.signal_components"))
    registry = SignalComponentRegistry.discover(signal_config(), sink=NullCycleEventSink())
    assert registry.ids() == {"kronos", "llm_committee", "fixture_signal"}


def test_duplicate_plugin_id_fails_startup(monkeypatch):
    monkeypatch.setattr(metadata, "entry_points", duplicate_signal_entry_points())
    with pytest.raises(ValueError, match="duplicate signal component id"):
        SignalComponentRegistry.discover(signal_config(), sink=NullCycleEventSink())


def test_market_source_entry_point_returns_configured_source(monkeypatch):
    registry = MarketSourceRegistry.discover(market_config(), entry_points=fake_market_entry_points())
    assert registry.require("fixture-market").id == "fixture-market"
```

- [ ] **Step 3: 运行测试确认旧字段和旧 registry 失败**

Run: `uv run pytest tests/test_plugin_entry_points.py tests/test_market_only_signal_context.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现协议与 entry point 组**

```python
class MarketDataSource(Protocol):
    id: str

    def requirements(self) -> DataRequirements: ...

    async def collect(
        self,
        pair: Pair,
        as_of: datetime,
        requirements: DataRequirements,
    ) -> SignalContext: ...
```

`SignalComponentRegistry.discover()` 先注册 `kronos` 与 `llm_committee` 的内置 factory，再读取 `importlib.metadata.entry_points(group="cryptotrader.signal_components")`。每个 entry point 必须加载一个 `factory(config, sink) -> SignalComponent`；DB 只能选择已安装 ID 并传递参数，不能保存或执行 Python 路径。

- [ ] **Step 5: 在 pyproject 声明内置 entry point ID**

```toml
[project.entry-points."cryptotrader.signal_components"]
kronos = "cryptotrader.signals.components.kronos:create_component"
llm_committee = "cryptotrader.signals.components.llm_committee:create_component"

[project.entry-points."cryptotrader.market_sources"]
default = "cryptotrader.market_sources.default:create_source"
```

内置 registry 与安装后的 entry points 使用同一 factory 契约；editable test 环境中 registry 显式合并内置 factory，按 ID 去重，不依赖 wheel 安装状态。

- [ ] **Step 6: 迁移 Kronos 和 LLM 组件只读 market-only context**

Kronos 使用 snapshots/current_price/atr；LLM prompt 和四智能体内部辩论只使用市场证据。删除 prompt 中账户权益、当前仓位和平台 ID，不改变内部 researcher/challenge/verdict 图。

- [ ] **Step 7: 运行组件与内部辩论回归测试**

Run: `uv run pytest tests/test_plugin_entry_points.py tests/test_market_only_signal_context.py tests/test_kronos_component.py tests/test_llm_committee_component.py tests/test_debate_parallel.py tests/test_debate_turn_capture.py --no-cov -q`

Expected: PASS。

- [ ] **Step 8: 提交**

```bash
git add pyproject.toml src/cryptotrader/market_sources src/cryptotrader/signals tests/factories tests/test_plugin_entry_points.py tests/test_market_only_signal_context.py
git commit -m "refactor: discover market and signal plugins"
```

---

### Task 4: 建立 VenueAdapter、VenueSession 与连接能力模型

**Files:**
- Modify: `src/cryptotrader/venues/__init__.py`
- Modify: `src/cryptotrader/venues/models.py`
- Create: `src/cryptotrader/venues/protocol.py`
- Create: `src/cryptotrader/venues/registry.py`
- Test: `tests/test_venue_domain.py`
- Test: `tests/test_venue_registry.py`
- Create: `tests/contracts/venue_adapter.py`

**Interfaces:**
- Produces: `VenueConnection`、`VenueCapabilities`、`VenueQuote`、`OrderIntent`、`NormalizedOrder`、`ProtectionSpec/State`、`OpenVenueState`。
- Produces: `VenueAdapter`、`VenueSession` 和 `VenueAdapterRegistry.discover()`。
- Produces: `assert_venue_contract(adapter_factory, environments)` 供 OKX/Bybit/Paper 共用。
- Consumes: Task 2 `CredentialPayload`。

- [ ] **Step 1: 写环境、能力与协议失败测试**

```python
@pytest.mark.parametrize("environment", ["paper", "demo", "testnet", "live"])
def test_connection_accepts_only_declared_environments(environment):
    assert connection("c", environment).environment == environment


def test_live_connection_requires_credential_reference():
    with pytest.raises(ValueError, match="credential_ref"):
        VenueConnection("live", "Live", "okx", "live", True, None, 1, "isolated")


def test_registry_resolves_by_adapter_id_without_brand_conditionals():
    registry = VenueAdapterRegistry((FakeAdapter("alpha"), FakeAdapter("beta")))
    assert registry.require("beta").adapter_id == "beta"
    with pytest.raises(KeyError, match="missing"):
        registry.require("missing")
```

- [ ] **Step 2: 运行测试确认新领域不存在**

Run: `uv run pytest tests/test_venue_domain.py tests/test_venue_registry.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 3: 实现不可变 DTO 与能力规则**

```python
@dataclass(frozen=True)
class VenueCapabilities:
    market_types: frozenset[MarketType]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: frozenset[str]


@dataclass(frozen=True)
class OpenVenueState:
    position: ConnectionPosition
    open_orders: tuple[NormalizedOrder, ...]
    protections: tuple[ProtectionState, ...]
```

所有 money/amount/price 领域值使用 `Decimal`，只在 API JSON 边界转字符串，禁止二进制浮点进入订单数量和价格精度逻辑。

- [ ] **Step 4: 实现 registry discovery 契约**

registry 读取 `cryptotrader.venue_adapters` entry point group，校验 factory 返回对象 ID 与 entry point name 一致；重复 ID 或配置引用未安装 ID 时启动失败。本任务用 monkeypatch entry points 测试 discovery，不在 OKX/Bybit/Paper 模块尚未完成时写入会断裂的 pyproject entry point。

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/test_venue_domain.py tests/test_venue_registry.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/cryptotrader/venues tests/contracts tests/test_venue_domain.py tests/test_venue_registry.py
git commit -m "feat: define pluggable venue contracts"
```

---

### Task 5: 拆分 CCXT 公共层并实现 OKX、Bybit 适配器

**Files:**
- Create: `src/cryptotrader/venues/ccxt_base.py`
- Create: `src/cryptotrader/venues/okx.py`
- Create: `src/cryptotrader/venues/bybit.py`
- Create: `tests/fakes/ccxt_client.py`
- Test: `tests/test_okx_venue_adapter.py`
- Test: `tests/test_bybit_venue_adapter.py`
- Test: `tests/test_ccxt_venue_contract.py`
- Modify: `tests/contracts/venue_adapter.py`

**Interfaces:**
- Produces: `OkxVenueAdapter` environments `demo|live`。
- Produces: `BybitVenueAdapter` environments `testnet|demo|live`。
- Produces: `CcxtVenueBase` 的连接、market metadata、精度、标准化与关闭机制。
- Consumes: Task 4 契约；不导入 `execution.exchange.LiveExchange`。

- [ ] **Step 1: 写环境映射和请求参数失败测试**

```python
async def test_okx_demo_sets_simulated_header_only_for_demo():
    client = await connect_okx("demo")
    assert client.headers["x-simulated-trading"] == "1"
    live = await connect_okx("live")
    assert "x-simulated-trading" not in live.headers


@pytest.mark.parametrize(
    ("environment", "hostname"),
    [("testnet", "api-testnet.bybit.com"), ("demo", "api-demo.bybit.com"), ("live", "api.bybit.com")],
)
async def test_bybit_environment_maps_to_official_endpoint(environment, hostname):
    session = await connect_bybit(environment)
    assert hostname in session.client.urls["api"]["public"]


async def test_reduce_only_and_protection_are_normalized_for_both_adapters(adapter):
    order = await adapter.session.place_order(order_intent(reduce_only=True))
    protection = await adapter.session.replace_protection(protection_spec())
    assert order.reduce_only is True
    assert protection.active is True
```

- [ ] **Step 2: 写同一套 adapter contract 测试**

契约必须对两个 adapter 分别执行：余额/持仓标准化、spot/swap 数量、`reduce_only`、开仓、减仓、平仓、止损止盈、保护替换、挂单查询、session close。fake CCXT client 只记录调用并返回平台真实字段形状，不在生产 adapter 中添加测试分支。

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_okx_venue_adapter.py tests/test_bybit_venue_adapter.py tests/test_ccxt_venue_contract.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现 `CcxtVenueBase` 的纯公共职责**

公共层只负责：创建 async CCXT client、load markets 缓存、`Decimal` 精度、统一异常 `VenueOperationError`、订单/余额/持仓标准化和幂等 close。环境 URL、headers、合约面值、持仓模式、下单 params、保护单 API 全由具体 adapter 覆盖。

- [ ] **Step 5: 移植 OKX 平台行为**

从旧 `LiveExchange` 只提取已经验证的 OKX 行为：Demo header、passphrase、swap contract size、`tdMode`、`posSide`、algo OCO 创建/查询/取消。新代码只能通过 `OkxVenueSession` 暴露 Task 4 契约，业务层不得接触 OKX raw response。

- [ ] **Step 6: 实现 Bybit 平台行为**

使用 Bybit unified account/contract API 的 CCXT 参数映射；Testnet/Demo/Live 客户端分别配置。保护单创建后必须通过 `list_open_state()` 查询并标准化，不能把下单响应本身当成保护已安装证据。

- [ ] **Step 7: 运行 adapter 契约与原 OKX 行为回归**

Run: `uv run pytest tests/test_okx_venue_adapter.py tests/test_bybit_venue_adapter.py tests/test_ccxt_venue_contract.py tests/test_exchange_algo_oco.py tests/test_exchange_hardened.py --no-cov -q`

Expected: PASS。旧测试此时改为直接调用 `OkxVenueAdapter`，不得再导入 `LiveExchange`。

- [ ] **Step 8: 提交**

```bash
git add src/cryptotrader/venues tests/fakes tests/contracts tests/test_okx_venue_adapter.py tests/test_bybit_venue_adapter.py tests/test_ccxt_venue_contract.py tests/test_exchange_algo_oco.py tests/test_exchange_hardened.py
git commit -m "feat: add OKX and Bybit venue adapters"
```

---

### Task 6: 将 Paper 撮合实现成同一 VenueAdapter

**Files:**
- Create: `src/cryptotrader/venues/paper.py`
- Modify: `pyproject.toml`
- Test: `tests/test_paper_venue_adapter.py`
- Rewrite: `tests/test_paper_exchange_concurrency.py`
- Rewrite: `tests/test_paper_exchange_protection.py`
- Modify: `tests/contracts/venue_adapter.py`

**Interfaces:**
- Produces: `PaperVenueAdapter/PaperVenueSession`，environment 只允许 `paper`，不接受凭据。
- Consumes: Task 4 session 契约；复用旧 simulator 已验证的撮合和保护触发数学，不复用旧类名。

- [ ] **Step 1: 写 Paper contract、并发和保护测试**

```python
async def test_paper_connection_rejects_credentials():
    with pytest.raises(ValueError, match="does not accept credentials"):
        await PaperVenueAdapter().connect(paper_connection(), credential_payload())


async def test_paper_session_uses_same_portfolio_and_order_contract():
    session = await PaperVenueAdapter(initial_equity=Decimal("10000")).connect(paper_connection(), None)
    await session.set_quote(pair(), Decimal("50000"))
    order = await session.place_order(order_intent(amount="0.1"))
    snapshot = await session.fetch_portfolio(pair())
    assert order.status == "filled"
    assert snapshot.position.signed_notional == Decimal("5000")


async def test_paper_protection_triggers_once_under_concurrent_reads():
    session = protected_long_session()
    await session.set_quote(pair(), Decimal("48000"))
    states = await asyncio.gather(*(session.list_open_state(pair()) for _ in range(8)))
    assert sum(bool(state.triggered_protections) for state in states) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_paper_venue_adapter.py tests/test_paper_exchange_concurrency.py tests/test_paper_exchange_protection.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 3: 实现内存 session**

每个 connection ID 拥有独立余额、仓位、订单和保护状态；每个 pair 使用 `asyncio.Lock` 串行化撮合与保护触发。`fetch_portfolio()` 和 `list_open_state()` 返回与真实 adapter 相同 DTO。初始化本金来自 Paper connection 的数据库参数，不再来自 `backtest.initial_capital` 或独立 mode。

- [ ] **Step 4: 运行 Paper 与通用契约测试**

Run: `uv run pytest tests/test_paper_venue_adapter.py tests/test_paper_exchange_concurrency.py tests/test_paper_exchange_protection.py tests/test_ccxt_venue_contract.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 在三个 adapter 均可导入后声明 entry points**

```toml
[project.entry-points."cryptotrader.venue_adapters"]
okx = "cryptotrader.venues.okx:create_adapter"
bybit = "cryptotrader.venues.bybit:create_adapter"
paper = "cryptotrader.venues.paper:create_adapter"
```

Run: `uv run pytest tests/test_venue_registry.py tests/test_okx_venue_adapter.py tests/test_bybit_venue_adapter.py tests/test_paper_venue_adapter.py --no-cov -q`

Expected: PASS，三个 production factory 均能由 metadata 加载。

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml src/cryptotrader/venues/paper.py tests/contracts tests/test_paper_venue_adapter.py tests/test_paper_exchange_concurrency.py tests/test_paper_exchange_protection.py
git commit -m "feat: implement paper as a venue adapter"
```

---

### Task 7: 实现资金池组合聚合与固定权重分配

**Files:**
- Create: `src/cryptotrader/portfolio/models.py`
- Create: `src/cryptotrader/portfolio/aggregator.py`
- Modify: `src/cryptotrader/execution/models.py`
- Create: `src/cryptotrader/execution/allocation.py`
- Test: `tests/test_portfolio_aggregator.py`
- Test: `tests/test_weighted_allocation.py`

**Interfaces:**
- Produces: `ConnectionPortfolioSnapshot`、`BookPortfolioSnapshot`、`ConnectionTarget`。
- Produces: `PortfolioAggregator.read(book, sessions, pair)`。
- Produces: `WeightedAllocationPolicy.allocate(target, book, portfolio)`。
- Consumes: `TargetPosition.signed_ratio` 作为平台无关敞口；不修改信号结果。

- [ ] **Step 1: 写隔离聚合失败测试**

```python
async def test_aggregator_sums_only_connections_in_the_requested_book():
    snapshots = {
        "okx-demo": portfolio("10000", "2000"),
        "bybit-testnet": portfolio("20000", "-1000"),
        "bybit-live": portfolio("999999", "999999"),
    }
    result = await aggregator(snapshots).read(simulation_book(), sessions(snapshots), pair())
    assert result.total_equity == Decimal("30000")
    assert result.total_signed_notional == Decimal("1000")
    assert {item.connection_id for item in result.connections} == {"okx-demo", "bybit-testnet"}
```

- [ ] **Step 2: 写固定权重数学失败测试**

```python
@pytest.mark.parametrize(
    ("side", "ratio", "expected"),
    [
        ("long", "0.5", ("20000", "30000")),
        ("short", "0.5", ("-20000", "-30000")),
        ("flat", "0", ("0", "0")),
    ],
)
def test_weighted_allocation_uses_book_equity_and_explicit_weights(side, ratio, expected):
    targets = WeightedAllocationPolicy().allocate(
        target(side, ratio),
        book_with_weights("0.4", "0.6"),
        book_portfolio(total_equity="100000"),
    )
    assert tuple(str(item.target_signed_notional) for item in targets) == expected


def test_single_connection_at_one_hundred_percent_is_normal_case():
    targets = WeightedAllocationPolicy().allocate(target("long", "0.25"), one_connection_book(), book_portfolio("8000"))
    assert targets == (connection_target("paper-local", "2000"),)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_portfolio_aggregator.py tests/test_weighted_allocation.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现聚合器**

并行 `fetch_portfolio()`，保持配置 allocations 顺序输出明细；任何读取异常转成带 connection ID 的 `PortfolioReadError`。聚合器本身不吞掉失败，也不把不可用连接权益设成零。

- [ ] **Step 5: 实现确定性分配**

```python
target_signed_notional = (
    portfolio.total_equity
    * Decimal(str(target.signed_ratio))
    * allocation.weight
)
```

每个 `ConnectionTarget` 同时保存 `book_id`、`connection_id`、`weight`、`book_equity`、`target_exposure`、`target_signed_notional`。禁用 allocation 不输出 target；不做余额择优和失败重分配。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_portfolio_aggregator.py tests/test_weighted_allocation.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add src/cryptotrader/portfolio src/cryptotrader/execution/models.py src/cryptotrader/execution/allocation.py tests/test_portfolio_aggregator.py tests/test_weighted_allocation.py
git commit -m "feat: aggregate books and allocate weighted targets"
```

---

### Task 8: 实现两级风控与资金池执行提案

**Files:**
- Rewrite: `src/cryptotrader/risk/models.py`
- Rewrite: `src/cryptotrader/risk/gate.py`
- Rewrite: `src/cryptotrader/execution/planner.py`
- Modify: `src/cryptotrader/execution/models.py`
- Test: `tests/test_book_risk_gate.py`
- Test: `tests/test_connection_risk_gate.py`
- Rewrite: `tests/test_execution_planner.py`

**Interfaces:**
- Produces: `BookRiskDecision`、`ConnectionRiskDecision`、`BookExecutionProposal`、`ConnectionExecutionPlan`。
- Produces: `BookRiskGate.evaluate()`、`ConnectionRiskGate.evaluate()`、`ExecutionPlanner.propose()`。
- Consumes: Task 7 snapshot/targets；Task 4 capabilities/quotes。

- [ ] **Step 1: 写增仓全连接预检与减仓优先测试**

```python
async def test_increase_requires_every_connection_to_pass_before_any_plan_is_ready():
    proposal = await planner.propose(longer_targets(), portfolios(), sessions(one_unavailable=True))
    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert proposal.risk.rejected_by == "connection_preflight"


async def test_risk_reduction_keeps_reachable_connection_plans():
    proposal = await planner.propose(flat_targets(), open_portfolios(), sessions(one_unavailable=True))
    assert proposal.ready is True
    assert [plan.connection_id for plan in proposal.connection_plans] == ["reachable"]
    assert proposal.unavailable_connections == ("unreachable",)
```

- [ ] **Step 2: 写能力、资金池 cap 与权重不变测试**

```python
def test_derivative_risk_increase_requires_native_protection():
    result = ConnectionRiskGate(config()).evaluate(increase_request(capabilities(native_protection=False)))
    assert result.passed is False
    assert result.reason == "native protection required for derivative risk increase"


def test_book_cap_scales_whole_target_without_reweighting_connections():
    result = BookRiskGate(config(max_net="0.4")).evaluate(book_request(target_exposure="0.7"))
    assert result.capped_target_exposure == Decimal("0.4")
    assert result.connection_weights == (Decimal("0.4"), Decimal("0.6"))
```

- [ ] **Step 3: 运行测试确认旧单账户 gate 不符合契约**

Run: `uv run pytest tests/test_book_risk_gate.py tests/test_connection_risk_gate.py tests/test_execution_planner.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现资金池级 gate**

检查总净敞口、总毛敞口、资金池回撤和连接集中度。若需要 cap，先得到新的资金池 target exposure，再用原始 allocation 权重重新调用 Task 7 policy；不得单独缩放某一连接。

- [ ] **Step 5: 实现连接级 gate 与订单差值计划**

每个 plan 固定包含：current signed notional、target signed notional、delta、quote、amount、reduce_only、market type、stop/take-profit、旧 protection IDs 和 capabilities。planner 通过 session 的 market metadata/quote 完成数量精度转换，业务层不判断 adapter ID。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_book_risk_gate.py tests/test_connection_risk_gate.py tests/test_execution_planner.py tests/test_target_risk_gate.py --no-cov -q`

Expected: PASS；`test_target_risk_gate.py` 改为新 Book/Connection 请求，不保留旧参数 wrapper。

- [ ] **Step 7: 提交**

```bash
git add src/cryptotrader/risk src/cryptotrader/execution tests/test_book_risk_gate.py tests/test_connection_risk_gate.py tests/test_execution_planner.py tests/test_target_risk_gate.py
git commit -m "refactor: enforce book and connection risk gates"
```

---

### Task 9: 实现连接执行服务与多平台协调器

**Files:**
- Rewrite: `src/cryptotrader/execution/service.py`
- Create: `src/cryptotrader/execution/coordinator.py`
- Modify: `src/cryptotrader/execution/models.py`
- Rewrite: `tests/test_execution_service.py`
- Create: `tests/test_execution_coordinator.py`
- Create: `tests/test_execution_compensation.py`

**Interfaces:**
- Produces: `VenueExecutionService.execute(plan)` 和 `ExecutionCoordinator.execute(proposal)`。
- Produces: `ConnectionExecutionResult`、`BookExecutionResult`。
- Consumes: Task 8 proposal；Task 4 session。

- [ ] **Step 1: 写保护替换顺序与补偿测试**

```python
async def test_service_installs_new_protection_before_cancelling_old_protection():
    result = await service(recording_session()).execute(increase_plan(old_protection_ids=("old",)))
    assert result.status == "completed"
    assert result.trace == ("place_order", "replace_protection", "reconcile", "cancel_old_protection")


async def test_open_success_protection_failure_compensates_on_same_connection():
    result = await service(session(protection_error=True)).execute(increase_plan())
    assert result.status == "failed"
    assert result.compensation.attempted is True
    assert result.compensation.succeeded is True
    assert result.requires_attention is False


async def test_failed_compensation_marks_attention():
    result = await service(session(protection_error=True, close_error=True)).execute(increase_plan())
    assert result.requires_attention is True
    assert result.final_position.protected is False
```

- [ ] **Step 2: 写 completed/partial/failed 聚合测试**

```python
@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("completed", "completed"), "completed"),
        (("completed", "failed"), "partial"),
        (("failed", "failed"), "failed"),
    ],
)
async def test_coordinator_aggregates_connection_outcomes(statuses, expected):
    result = await coordinator_with(statuses).execute(proposal())
    assert result.status == expected
    assert result.reallocated is False
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_execution_service.py tests/test_execution_coordinator.py tests/test_execution_compensation.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现单连接执行顺序**

执行服务流程固定为：重新读取 open state → 根据目标差值下单 → 安装新保护 → 查询保护与最终持仓对账 → 取消旧保护。减仓/清仓不创建多余保护；开仓成功但保护失败只在该连接反向 `reduce_only` 补偿，不触碰其他连接。

- [ ] **Step 5: 实现 coordinator 并行与精确终态**

只并行执行 proposal 中已有计划；连接异常转为该连接 `failed` 结果，不让 `gather` 丢失其他成功结果。`requires_attention` 是任意连接标记的 OR。结果保留目标权重，明确 `reallocated=False`。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_execution_service.py tests/test_execution_coordinator.py tests/test_execution_compensation.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add src/cryptotrader/execution tests/test_execution_service.py tests/test_execution_coordinator.py tests/test_execution_compensation.py
git commit -m "feat: coordinate protected multi-venue execution"
```

---

### Task 10: 重建 per-book HITL 与多平台 Journal

**Files:**
- Create: `src/cryptotrader/hitl/models.py`
- Rewrite: `src/cryptotrader/hitl/store.py`
- Rewrite: `src/cryptotrader/hitl/gate.py`
- Rewrite: `src/cryptotrader/journal/models.py`
- Rewrite: `src/cryptotrader/journal/store.py`
- Test: `tests/test_book_hitl_store.py`
- Rewrite: `tests/test_hitl_gate.py`
- Create: `tests/test_multi_venue_journal.py`
- Rewrite: `tests/test_cycle_journal_store.py`

**Interfaces:**
- Produces: `BookApproval` 与状态 `pending|approved|rejected|invalidated|executed`。
- Produces: `ApprovalStore.create/approve/reject/claim_for_execution`，claim 必须校验 revision 并且只成功一次。
- Produces: `MultiVenueCycleRecord` 与 `MultiVenueCycleStore`，只访问 `multi_venue_cycles`。
- Consumes: Task 8 proposal、Task 9 result、Task 2 current revision。

- [ ] **Step 1: 写审批绑定与并发领取失败测试**

```python
async def test_revision_change_invalidates_pending_approval(store):
    approval = await store.create(book_proposal(config_revision=7))
    with pytest.raises(ApprovalInvalidated):
        await store.claim_for_execution(approval.id, current_revision=8)
    assert (await store.get(approval.id)).status == "invalidated"


async def test_approval_executes_exact_saved_proposal_once(store):
    approval = await store.create(book_proposal(weights=("0.4", "0.6")))
    await store.approve(approval.id)
    first, second = await asyncio.gather(
        store.claim_for_execution(approval.id, current_revision=7),
        store.claim_for_execution(approval.id, current_revision=7),
        return_exceptions=True,
    )
    claims = [item for item in (first, second) if isinstance(item, BookExecutionProposal)]
    assert len(claims) == 1
    assert claims[0].weights == (Decimal("0.4"), Decimal("0.6"))
```

- [ ] **Step 2: 写新 Journal round-trip 和零秘密测试**

```python
async def test_multi_venue_cycle_round_trip_preserves_book_and_connection_results(store):
    record = multi_venue_record(simulation="completed", live="partial", requires_attention=True)
    await store.save(record)
    loaded = await store.get(record.cycle_id)
    assert loaded == record


async def test_journal_serialization_rejects_secret_shaped_keys(store):
    record = multi_venue_record(raw={"api_key": "marker"})
    with pytest.raises(ValueError, match="secret field"):
        await store.save(record)
```

- [ ] **Step 3: 运行测试确认旧 schema 不符合契约**

Run: `uv run pytest tests/test_book_hitl_store.py tests/test_hitl_gate.py tests/test_multi_venue_journal.py tests/test_cycle_journal_store.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现 approval schema 与原子 claim**

审批行保存 `approval_id/cycle_id/book_id/config_revision/proposal_json/status/created_at/decided_at/claimed_at`。批准/拒绝只能从 pending 转换；claim 使用带 status/revision 条件的单条 UPDATE，失败后返回明确的 rejected/invalidated/already claimed 错误。

- [ ] **Step 5: 实现 `multi_venue_cycles` 新表**

字段严格对应 spec：cycle_id、config_revision、market_data_source_id、component_signals、fused_signal、target_position、book_results、cycle_status、execution_status、requires_attention、created_at。新 store 不声明、不查询旧 `trading_cycles` table；旧表物理存在不影响。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_book_hitl_store.py tests/test_hitl_gate.py tests/test_multi_venue_journal.py tests/test_cycle_journal_store.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add src/cryptotrader/hitl src/cryptotrader/journal tests/test_book_hitl_store.py tests/test_hitl_gate.py tests/test_multi_venue_journal.py tests/test_cycle_journal_store.py
git commit -m "refactor: persist book approvals and multi-venue cycles"
```

---

### Task 11: 重写 TradingCycle 为一次信号、多资金池主链

这是唯一刻意保持较大的原子任务：`CycleRequest/TradingCycle` 签名一旦切换，所有生产入口必须在同一提交迁移，不能留下依赖下一任务修复的断裂提交。

**Files:**
- Create: `src/cryptotrader/runtime.py`
- Rewrite: `src/cryptotrader/trading_cycle.py`
- Rewrite: `src/cryptotrader/bootstrap.py`
- Rewrite: `src/cryptotrader/decision/models.py`
- Rewrite: `src/cryptotrader/cycle_serialization.py`
- Modify: `src/cryptotrader/cycle_events.py`
- Rewrite: `src/cryptotrader/scheduler.py`
- Modify: `src/cryptotrader/triggers/engine.py`
- Modify: `src/cryptotrader/backtest/engine.py`
- Modify: `src/cryptotrader/backtest/session.py`
- Rewrite: `src/cryptotrader/chat/analysis_runner.py`
- Rewrite: `src/cli/main.py`
- Rewrite: `src/api/main.py`
- Rewrite: `src/api/routes/chat.py`
- Rewrite: `src/api/routes/scheduler.py`
- Modify: `src/api/routes/health.py`
- Modify: `src/api/routes/portfolio_v2.py`
- Modify: `tests/factories/signal_fusion.py`
- Rewrite: `tests/test_trading_cycle.py`
- Create: `tests/test_multi_book_cycle.py`
- Rewrite: `tests/test_signal_fusion_e2e.py`
- Rewrite: `tests/test_live_backtest_decision_parity.py`
- Rewrite: `tests/test_bootstrap.py`
- Create: `tests/test_runtime_entrypoints.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_chat_cycle_cancellation.py`
- Modify: every existing test returned by the old-constructor inventory in Step 6。

**Interfaces:**
- Produces: 新 `CycleRequest(pair)`，不含 mode/exchange ID。
- Produces: 新 `CycleOutcome(cycle_id, config_revision, target_position, books, status, execution_status, requires_attention)`。
- Produces: 最小可启动 `Runtime/build_runtime()`，使 setup_required API 和 active cycle 在切换提交中都可启动。
- Consumes: snapshot、market source、signal registry/runner/fusion/decision、portfolio/allocation/risk/planner/HITL/coordinator/journal。
- Migrates atomically: API、CLI、Scheduler、Triggers 与 Chat 对新 CycleRequest/Runtime 的构造调用；Task 13–14 只完善生命周期和操作体验，不再承担旧签名迁移。

- [ ] **Step 1: 写信号只运行一次的核心失败测试**

```python
async def test_one_signal_pass_drives_simulation_and_live_books():
    cycle, runner = cycle_with_books(simulation_book(), live_book(hitl=False))
    outcome = await cycle.run(CycleRequest(pair()))
    assert runner.call_count == 1
    assert outcome.target_position is runner.expected_target
    assert {result.book_id for result in outcome.books} == {"simulation", "live"}


async def test_book_failures_do_not_block_other_book():
    cycle = cycle_with_book_outcomes(simulation="failed", live="completed")
    outcome = await cycle.run(CycleRequest(pair()))
    assert outcome.book("simulation").status == "failed"
    assert outcome.book("live").status == "completed"
    assert outcome.execution_status == "partial"


async def test_completed_simulation_is_preserved_while_live_waits_for_approval():
    outcome = await cycle_with_books(simulation_book(), live_book(hitl=True)).run(CycleRequest(pair()))
    assert outcome.status == "awaiting_approval"
    assert outcome.book("simulation").status == "completed"
    assert outcome.book("live").status == "awaiting_approval"
```

- [ ] **Step 2: 写周期 snapshot 固定测试**

```python
async def test_cycle_keeps_initial_revision_when_config_changes_mid_run():
    repository = repository_that_changes_from(12, to=13, after_first_get=True)
    outcome = await cycle(repository=repository).run(CycleRequest(pair()))
    assert outcome.config_revision == 12
    assert all(book.config_revision == 12 for book in outcome.books)
```

- [ ] **Step 3: 运行测试确认旧单执行主链失败**

Run: `uv run pytest tests/test_trading_cycle.py tests/test_multi_book_cycle.py tests/test_signal_fusion_e2e.py tests/test_live_backtest_decision_parity.py tests/test_bootstrap.py tests/test_runtime_entrypoints.py tests/test_scheduler.py tests/test_chat_cycle_cancellation.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现固定周期流程**

`run()` 顺序严格为：读取一次 snapshot → 校验 active → 收集一次 market context → 运行一次组件与融合 → 决策/退出价格 → 对启用 books 并行 `_prepare_book()` → 对无需审批的 ready proposals 并行执行 → 持久化整周期。每个 `_prepare_book()` 独立捕获并记录 book failure；信号或融合失败时所有 books 都不执行。

- [ ] **Step 5: 实现审批恢复入口**

`TradingCycle.execute_approved(approval_id)` 从 store 原子 claim 原 proposal，校验当前 revision，通过 coordinator 执行后更新同一 cycle 的对应 book result；不得重新运行信号、重新读取分配权重或创建新 cycle。

- [ ] **Step 6: 枚举并在同一改动中迁移最小 Runtime 与全部构造调用方**

Run:

```bash
rg -l 'CycleRequest\(|build_trading_cycle\(|initialize_trading_cycle\(|TradingCycle\(' src tests
```

`build_runtime()` 从 Task 2 repository 获取 snapshot，setup_required 时不创建 sessions，active 时组装 Task 3–10 服务和新 `TradingCycle`。API lifespan、CLI、Scheduler、Trigger callback、Chat、Backtest、portfolio route、测试 factory 和上述搜索返回的所有测试全部改传 `CycleRequest(pair)` 或新 Runtime；删除它们对 mode、exchange ID 和旧 `TradingCycle` constructor 的引用。Backtest 在本任务先使用 100% Paper book 保持可运行，Task 15 再完成历史 session 和 API parity。本步骤完成前不得提交。

- [ ] **Step 7: 统一序列化与事件**

事件新增 `book_proposed/book_awaiting_approval/book_execution_started/connection_execution_completed/book_execution_completed`，所有事件携带 cycle/book/connection ID 和 config revision，不含凭据或 raw auth。删除旧单 `execution_result` payload。

- [ ] **Step 8: 运行主链、启动与入口测试**

Run: `uv run pytest tests/test_trading_cycle.py tests/test_multi_book_cycle.py tests/test_signal_fusion_e2e.py tests/test_live_backtest_decision_parity.py tests/test_cycle_events.py tests/test_bootstrap.py tests/test_runtime_entrypoints.py tests/test_scheduler.py tests/test_chat_cycle_cancellation.py tests/test_api.py tests/test_health_endpoint.py --no-cov -q`

Run: `uv run pytest -q`

Expected: targeted 与当时存在的全量测试全部 PASS，证明原子签名切换没有把错误留给后续任务。

- [ ] **Step 9: 提交原子切换**

```bash
git add src/cryptotrader/runtime.py src/cryptotrader/trading_cycle.py src/cryptotrader/bootstrap.py src/cryptotrader/decision/models.py src/cryptotrader/cycle_serialization.py src/cryptotrader/cycle_events.py src/cryptotrader/scheduler.py src/cryptotrader/triggers/engine.py src/cryptotrader/chat/analysis_runner.py src/cli/main.py src/api/main.py src/api/routes/chat.py src/api/routes/scheduler.py src/api/routes/health.py tests
git commit -m "refactor: run one signal across execution books"
```

---

### Task 12: 交付数据库配置、平台、资金池和周期 API

**Files:**
- Create: `src/api/routes/config.py`
- Create: `src/api/routes/venues.py`
- Create: `src/api/routes/portfolio_books.py`
- Create: `src/api/routes/cycles.py`
- Rewrite: `src/api/routes/hitl.py`
- Rewrite: `src/api/routes/decisions.py`
- Modify: `src/api/routes/__init__.py`
- Modify: `src/api/main.py`
- Test: `tests/test_runtime_config_api.py`
- Test: `tests/test_venue_connections_api.py`
- Test: `tests/test_portfolio_books_api.py`
- Rewrite: `tests/test_hitl_api.py`
- Rewrite: `tests/test_api_decisions_list.py`
- Rewrite: `tests/test_api_decisions_detail.py`
- Modify: `tests/test_api_security_hardening.py`

**Interfaces:**
- Produces exactly: `GET/PUT /api/config`、connection create/update/credentials/test、portfolio book list/detail、multi-venue cycle list/detail。
- Consumes: Task 2 repository/vault、Task 4 registry、Task 7 aggregator、Task 10 stores、Task 11 cycle。
- Removes route: `/api/signal-profile`，signal settings live inside `/api/config`。

- [ ] **Step 1: 写 config CAS、setup 和脱敏失败测试**

```python
async def test_put_config_requires_expected_revision(client):
    current = await client.get("/api/config")
    saved = await client.put("/api/config", json={"expected_revision": current.json()["revision"], "document": active_payload()})
    assert saved.status_code == 200
    stale = await client.put("/api/config", json={"expected_revision": current.json()["revision"], "document": active_payload()})
    assert stale.status_code == 409


async def test_config_and_connection_responses_never_return_credentials(client):
    marker = "credential-redaction-sentinel"
    await put_fixture_credentials(client, "okx-demo", marker=marker)
    body = json.dumps((await client.get("/api/config")).json()) + json.dumps((await client.get("/api/portfolio/books")).json())
    assert marker not in body
    assert '"configured": true' in body
```

- [ ] **Step 2: 写 connection 生命周期与测试连接失败测试**

```python
async def test_environment_is_immutable_after_creation(client):
    await create_connection(client, id="okx-demo", environment="demo")
    response = await client.put("/api/venue-connections/okx-demo", json=update(environment="live"))
    assert response.status_code == 422


async def test_connection_cannot_be_disabled_while_enabled_book_references_it(client):
    response = await client.put("/api/venue-connections/okx-demo", json=update(enabled=False))
    assert response.status_code == 422


async def test_connection_test_uses_adapter_and_returns_normalized_health(client):
    response = await client.post("/api/venue-connections/bybit-testnet/test")
    assert response.json() == {"connection_id": "bybit-testnet", "healthy": True, "environment": "testnet", "capabilities": ANY, "credential_configured": True}
```

- [ ] **Step 3: 运行 API 测试确认路由不存在**

Run: `uv run pytest tests/test_runtime_config_api.py tests/test_venue_connections_api.py tests/test_portfolio_books_api.py tests/test_hitl_api.py tests/test_api_decisions_list.py tests/test_api_decisions_detail.py --no-cov -q`

Expected: FAIL/404。

- [ ] **Step 4: 实现 Pydantic API 模型与错误映射**

所有 request model `extra="forbid"`。`RevisionConflict → 409`，领域校验 → 422，未初始化/未配置凭据 → 503，adapter 连接失败 → 502。API response 使用显式 DTO，不直接 `model_dump()` repository 或 adapter 对象。

- [ ] **Step 5: 实现连接写入为完整文档 CAS**

connection create/update 先读取 snapshot，在内存替换 `execution.connections`，调用完整 `validate_runtime_document()` 后 CAS 保存。credential endpoint 只传 `CredentialPayload` 给 vault 并返回 `CredentialState`；输入对象离开 handler 后不进入 app state。

- [ ] **Step 6: 实现 portfolio 和 cycle read endpoints**

portfolio 响应按 book 分组并保留 connection 明细；模拟与 real 使用独立数组和 totals，不生成跨 scope aggregate。Decisions 路由改读 `multi_venue_cycles` 并返回 book results。

- [ ] **Step 7: 运行 API 测试和 OpenAPI 秘密字段检查**

Run: `uv run pytest tests/test_runtime_config_api.py tests/test_venue_connections_api.py tests/test_portfolio_books_api.py tests/test_hitl_api.py tests/test_api_decisions_list.py tests/test_api_decisions_detail.py tests/test_api_security_hardening.py --no-cov -q`

Expected: PASS；OpenAPI response schemas 无 `api_key/secret/passphrase/encrypted_payload`。

- [ ] **Step 8: 提交**

```bash
git add src/api tests/test_runtime_config_api.py tests/test_venue_connections_api.py tests/test_portfolio_books_api.py tests/test_hitl_api.py tests/test_api_decisions_list.py tests/test_api_decisions_detail.py tests/test_api_security_hardening.py
git commit -m "feat: expose database configuration and venue APIs"
```

---

### Task 13: 完善 Runtime 生命周期、连接复用与首次启动边界

**Files:**
- Modify: `src/cryptotrader/runtime.py`
- Modify: `src/cryptotrader/bootstrap.py`
- Modify: `src/api/main.py`
- Rewrite: `tests/test_bootstrap.py`
- Create: `tests/test_runtime_lifecycle.py`
- Rewrite: `tests/test_health_endpoint.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: `BootstrapSettings.from_environment()`，只读取 `DATABASE_URL/CONFIG_MASTER_KEY`。
- Hardens: Task 11 已可启动的 `build_runtime(settings, event_sink=None) -> Runtime`。
- Produces: `Runtime.reload_for_cycle()`、`Runtime.close()` 和 app.state 唯一 `runtime`。
- Consumes: Tasks 1–12 全部基础服务。

- [ ] **Step 1: 写 bootstrap 外部参数和 setup_required 测试**

```python
def test_bootstrap_settings_read_exactly_two_environment_variables(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CONFIG_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("CRYPTOTRADER_EXCHANGE_ID", "must-be-ignored")
    settings = BootstrapSettings.from_environment()
    assert settings == BootstrapSettings("sqlite+aiosqlite:///test.db", MASTER_KEY)


async def test_first_start_seeds_setup_document_without_opening_venue_session(settings):
    runtime = await build_runtime(settings, adapter_registry=recording_registry())
    assert runtime.snapshot.setup_required is True
    assert runtime.adapter_registry.connect_calls == []
```

- [ ] **Step 2: 写 active runtime 组装和关闭测试**

```python
async def test_active_runtime_opens_each_enabled_connection_once_and_closes_all(settings):
    runtime = await build_runtime(settings, snapshot=active_snapshot(two_connections=True))
    assert set(runtime.sessions) == {"okx-demo", "bybit-testnet"}
    await runtime.close()
    assert all(session.closed for session in runtime.sessions.values())


async def test_cycle_reads_new_revision_without_reopening_unchanged_connections(runtime):
    first = runtime.sessions["okx-demo"]
    await runtime.repository.replace(runtime.snapshot.revision, snapshot_with_new_weight().document)
    cycle = await runtime.reload_for_cycle()
    assert cycle.config_revision == runtime.snapshot.revision + 1
    assert runtime.sessions["okx-demo"] is first
```

- [ ] **Step 3: 运行测试确认最小 Runtime 尚未满足完整生命周期语义**

Run: `uv run pytest tests/test_bootstrap.py tests/test_runtime_lifecycle.py tests/test_health_endpoint.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现唯一运行时装配**

顺序固定：读取 bootstrap settings → repository.ensure schema/get_or_create → discovery registries → 校验 snapshot → setup_required 时只装配配置 API → active 时解密启用连接凭据并创建 sessions → 组装 market source、signal components、books、cycle、scheduler dependencies。连接 cache key 为 `(connection.id, credential revision-relevant fingerprint, non-secret connection fields)`；revision 只因权重变化不重连。

- [ ] **Step 5: 改造 FastAPI lifespan**

lifespan 只调用一次 `build_runtime()`，将其放入 `app.state.runtime`。setup_required 时 health 返回 `status="setup_required"` 且不启动 Scheduler/Triggers；配置激活后由下一次进程启动完整装配。shutdown 只 `await runtime.close()`，不得沿 `cycle.executor.exchange` 反射关闭。

- [ ] **Step 6: 运行 bootstrap、API 启动和关闭测试**

Run: `uv run pytest tests/test_bootstrap.py tests/test_runtime_lifecycle.py tests/test_health_endpoint.py tests/test_api.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add src/cryptotrader/runtime.py src/cryptotrader/bootstrap.py src/api/main.py tests/conftest.py tests/test_bootstrap.py tests/test_runtime_lifecycle.py tests/test_health_endpoint.py tests/test_api.py
git commit -m "refactor: bootstrap runtime from database snapshots"
```

---

### Task 14: 完成 Scheduler、Triggers、CLI 与 Chat 的多资金池操作语义

**Files:**
- Modify: `src/cryptotrader/scheduler.py`
- Modify: `src/cryptotrader/triggers/engine.py`
- Modify: `src/cryptotrader/triggers/store.py`
- Modify: `src/cli/main.py`
- Modify: `src/cryptotrader/chat/analysis_runner.py`
- Modify: `src/api/routes/chat.py`
- Modify: `src/api/routes/scheduler.py`
- Modify: `src/api/routes/health.py`
- Rewrite: `tests/test_scheduler.py`
- Rewrite: `tests/test_scheduler_endpoint.py`
- Rewrite: `tests/test_cli_agent_list.py`
- Rewrite: `tests/test_cli_backtest.py`
- Rewrite: `tests/test_chat_cycle_cancellation.py`
- Modify: `tests/test_trigger_engine.py`
- Modify: `tests/test_runtime_entrypoints.py`

**Interfaces:**
- Verifies and completes: Task 11 已迁移的所有入口只消费 `Runtime`/`RuntimeConfigRepository`，不调用配置全局函数。
- Scheduler generates `CycleRequest(pair)`；不保存 mode/exchange ID。
- CLI `arena run` 使用数据库启用 books；移除 `--mode` 和 `--exchange`。
- Chat 启动同一个 cycle，并按 book 报告结果；保留取消和内部辩论事件。

- [ ] **Step 1: 写入口禁用旧字段测试**

```python
def test_cli_run_has_no_mode_or_exchange_options(runner):
    help_text = runner.invoke(app, ["run", "--help"]).stdout
    assert "--mode" not in help_text
    assert "--exchange" not in help_text


async def test_scheduler_passes_platform_independent_request(recording_cycle):
    scheduler = Scheduler(runtime_config().scheduler, cycle=recording_cycle)
    await scheduler.run_once()
    assert recording_cycle.requests == [CycleRequest(pair("BTC/USDT"))]


async def test_chat_reports_each_book_without_rerunning_signal(chat_runner):
    events = await chat_runner.run("分析 BTC/USDT")
    assert chat_runner.cycle.runner.call_count == 1
    assert {event.book_id for event in events if event.kind == "book_result"} == {"simulation", "live"}
```

- [ ] **Step 2: 运行入口测试确认最小切换尚缺完整状态展示与操作行为**

Run: `uv run pytest tests/test_runtime_entrypoints.py tests/test_scheduler.py tests/test_scheduler_endpoint.py tests/test_chat_cycle_cancellation.py tests/test_trigger_engine.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 3: 迁移 Scheduler 和 Trigger**

Scheduler 构造参数只接收 `SchedulerConfig`、cycle 和可选 trigger engine。轮询频率、pairs、Redis、funding 配置从固定 snapshot 构造，不在循环内读环境/TOML。Trigger callback 只触发 `CycleRequest(pair)`；不注入平台。

- [ ] **Step 4: 迁移 CLI 与 Chat**

CLI 每个命令创建并关闭一个 `Runtime`，`arena run --pair` 执行数据库中所有启用 books。journal 子命令改读 `MultiVenueCycleStore`。Chat 复用 API app.state runtime，取消只取消尚未下单的 task；若执行已开始，返回精确 attention 提示并等待 coordinator 收集结果。

- [ ] **Step 5: 运行所有入口测试**

Run: `uv run pytest tests/test_runtime_entrypoints.py tests/test_scheduler.py tests/test_scheduler_endpoint.py tests/test_api_scheduler_v2.py tests/test_chat_cycle_cancellation.py tests/test_chat_event_buffer.py tests/test_chat_task_manager.py tests/test_trigger_engine.py tests/test_trigger_engine_completeness.py tests/test_cli_agent_list.py tests/test_cli_backtest.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/cryptotrader/scheduler.py src/cryptotrader/triggers src/cryptotrader/chat src/cli/main.py src/api/routes/chat.py src/api/routes/scheduler.py src/api/routes/health.py tests
git commit -m "refactor: migrate runtime entry points to execution books"
```

---

### Task 15: 让 Backtest 只使用临时 Paper 资金池

**Files:**
- Rewrite: `src/cryptotrader/backtest/engine.py`
- Rewrite: `src/cryptotrader/backtest/session.py`
- Modify: `src/cryptotrader/backtest/historical_data.py`
- Rewrite: `src/api/routes/backtest.py`
- Rewrite: `tests/test_backtest.py`
- Rewrite: `tests/test_live_backtest_decision_parity.py`
- Modify: `tests/test_api_backtest_run.py`
- Modify: `tests/test_api_backtest_sessions.py`
- Modify: `tests/test_api_backtest_status.py`

**Interfaces:**
- Backtest consumes one fixed `RuntimeConfigSnapshot` and historical `MarketDataSource`。
- It constructs an ephemeral `ExecutionBook(id="backtest", capital_scope="simulated")` with one Paper allocation at 1.0。
- It never resolves credentials or connects any configured DB venue connection。

- [ ] **Step 1: 写无外部连接和 revision 固定测试**

```python
async def test_backtest_never_connects_configured_demo_testnet_or_live_adapters():
    registry = recording_venue_registry()
    result = await backtest(snapshot_with_all_environments(), registry=registry).run()
    assert result.completed is True
    assert registry.connect_calls == [("paper", "backtest-paper")]


async def test_backtest_uses_one_snapshot_revision_for_every_bar():
    result = await backtest(snapshot_revision=44, repository=changing_repository()).run()
    assert {cycle.config_revision for cycle in result.cycles} == {44}
```

- [ ] **Step 2: 写 live/paper 决策 parity 测试**

相同 market context、component signals、signal config 和 book equity 必须产生相同 `TargetPosition` 与 100% connection target；只允许 quote/精度导致 order amount 差异。

- [ ] **Step 3: 运行测试确认旧独立 simulator 主链失败**

Run: `uv run pytest tests/test_backtest.py tests/test_live_backtest_decision_parity.py tests/test_api_backtest_run.py tests/test_api_backtest_sessions.py tests/test_api_backtest_status.py --no-cov -q`

Expected: FAIL。

- [ ] **Step 4: 实现临时 Paper 装配**

历史 source 替换 runtime market source；Paper session 初始权益使用请求 capital；其余 registry、runner、fusion、decision、allocation、risk、execution、journal DTO 与生产一致。Backtest cycle records 保存在 backtest session 结果中，不写生产 `multi_venue_cycles`。

- [ ] **Step 5: 运行 Backtest 契约测试**

Run: `uv run pytest tests/test_backtest.py tests/test_live_backtest_decision_parity.py tests/test_api_backtest_run.py tests/test_api_backtest_sessions.py tests/test_api_backtest_status.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/cryptotrader/backtest src/api/routes/backtest.py tests/test_backtest.py tests/test_live_backtest_decision_parity.py tests/test_api_backtest_run.py tests/test_api_backtest_sessions.py tests/test_api_backtest_status.py
git commit -m "refactor: run backtests through a paper execution book"
```

---

### Task 16: 实现网页初始化向导、平台连接和资金池配置

**Files:**
- Create: `web/src/hooks/use-runtime-config.ts`
- Create: `web/src/hooks/use-venue-connections.ts`
- Create: `web/src/pages/setup/index.tsx`
- Create: `web/src/pages/setup/setup-page.test.tsx`
- Create: `web/src/pages/settings/venues/index.tsx`
- Create: `web/src/pages/settings/venues/venue-form.tsx`
- Create: `web/src/pages/settings/venues/venues-page.test.tsx`
- Create: `web/src/pages/settings/execution-books/index.tsx`
- Create: `web/src/pages/settings/execution-books/book-form.tsx`
- Create: `web/src/pages/settings/execution-books/allocation-preview.tsx`
- Create: `web/src/pages/settings/execution-books/execution-books-page.test.tsx`
- Rewrite: `web/src/pages/strategy/index.tsx`
- Rewrite: `web/src/hooks/use-signal-profile.ts`
- Modify: `web/src/types/api.schema.ts`
- Modify: `web/src/types/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/layout/sidebar.tsx`
- Modify: `web/src/lib/i18n.ts`

**Interfaces:**
- Frontend only calls `/api/config` and `/api/venue-connections*` for configuration。
- `useRuntimeConfig` exposes `revision/document/setupRequired/replace()` and converts 409 into explicit reload state。
- Credential form never hydrates an existing secret; success clears input and renders `configured` badge。

- [ ] **Step 1: 写 setup wizard 失败测试**

```tsx
it("routes setup_required users into the ordered setup flow", async () => {
  server.use(getConfig({ setup_required: true, revision: 1 }))
  renderApp("/")
  expect(await screen.findByRole("heading", { name: "初始化交易系统" })).toBeInTheDocument()
  expect(stepLabels()).toEqual(["LLM", "信号组件", "行情来源", "平台连接", "执行资金池", "风控与审批", "调度器", "测试并激活"])
})
```

- [ ] **Step 2: 写连接脱敏、环境不可改与测试连接 UI 测试**

```tsx
it("shows configured state without ever rendering saved credential values", async () => {
  server.use(getConfigWithCredentialState("okx-demo", true))
  renderApp("/settings/venues")
  expect(await screen.findByText("凭据已配置")).toBeInTheDocument()
  expect(screen.getByLabelText("API Key")).toHaveValue("")
  expect(document.body.textContent).not.toContain("secret-marker")
})
```

- [ ] **Step 3: 写资金池验证与预览测试**

```tsx
it("blocks mixed capital scopes and previews exact weighted notionals", async () => {
  renderBookForm({ equity: 100000, targetExposure: 0.5, weights: [40, 60] })
  expect(screen.getByText("20,000 USDT")).toBeInTheDocument()
  expect(screen.getByText("30,000 USDT")).toBeInTheDocument()
  await addLiveConnectionToSimulation()
  expect(screen.getByText("模拟资金池不能包含实盘连接")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "保存" })).toBeDisabled()
})
```

- [ ] **Step 4: 运行前端测试确认页面不存在**

Run: `cd web && npm test -- --run src/pages/setup/setup-page.test.tsx src/pages/settings/venues/venues-page.test.tsx src/pages/settings/execution-books/execution-books-page.test.tsx`

Expected: FAIL。

- [ ] **Step 5: 实现 schema、hooks 和初始化路由保护**

`api.schema.ts` 明确声明 RuntimeConfig、Connection、Book、CredentialState；不存在 secret response 字段。App 首次 config load 时，`setup_required=true` 只允许 setup 页面和 health；active 后进入正常路由。

- [ ] **Step 6: 实现平台和资金池页面**

按 adapter 品牌分组连接，environment 使用明确 badge；编辑已有连接时 environment disabled。资金池页面按 simulated/real 分区、权重总和实时校验、HITL toggle、revision 展示和示例 exposure 预览。保存使用完整 config CAS；409 显示“配置已被其他操作更新，请重新加载”，不自动覆盖。

- [ ] **Step 7: 合并 Strategy 页面到 RuntimeConfig**

保留现有组件信任权重、阈值、ATR 和内部辩论配置 UI，但读写 `document.signals/document.llm`。删除 `/api/signal-profile` hook 请求语义；文件名可在 Task 18 删除后由 `use-runtime-config` 直接替代。

- [ ] **Step 8: 运行页面、类型和 lint 测试**

Run: `cd web && npm test -- --run src/pages/setup src/pages/settings src/pages/strategy`

Run: `cd web && npm run typecheck && npm run lint`

Expected: PASS。

- [ ] **Step 9: 提交**

```bash
git add web/src
git commit -m "feat: add web setup and execution book settings"
```

---

### Task 17: 改造网页投资组合、HITL 与周期结果为多资金池视图

**Files:**
- Create: `web/src/hooks/use-portfolio-books.ts`
- Create: `web/src/hooks/use-multi-venue-cycles.ts`
- Create: `web/src/pages/cycles/index.tsx`
- Create: `web/src/pages/cycles/cycle-detail.tsx`
- Create: `web/src/pages/cycles/cycles-page.test.tsx`
- Rewrite: `web/src/hooks/use-hitl-approvals.ts`
- Rewrite: `web/src/hooks/use-portfolio-snapshot.ts`
- Rewrite: `web/src/pages/dashboard/components/positions-table.tsx`
- Rewrite: `web/src/pages/dashboard/components/metric-cards-row.tsx`
- Rewrite: `web/src/pages/risk/components/approval-item.tsx`
- Rewrite: `web/src/pages/risk/components/approval-queue-card.tsx`
- Rewrite: `web/src/pages/decisions/index.tsx`
- Rewrite: `web/src/pages/decisions/components/decisions-table.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/layout/sidebar.tsx`
- Modify: `web/src/types/api.schema.ts`
- Test: `web/src/pages/risk/components/approval-book-plan.test.tsx`
- Test: `web/src/pages/dashboard/dashboard-books.test.tsx`

**Interfaces:**
- Dashboard never computes one total across simulated and real scopes。
- Approval UI approves/rejects complete Book proposal and shows config revision。
- Cycle detail renders signal evidence once, then book → connection hierarchy。

- [ ] **Step 1: 写 scope 隔离 UI 失败测试**

```tsx
it("never merges simulated equity or pnl into real metrics", async () => {
  server.use(getPortfolioBooks({ simulatedEquity: 100000, realEquity: 12000 }))
  renderApp("/")
  expect(await screen.findByText("模拟资金 100,000 USDT")).toBeInTheDocument()
  expect(screen.getByText("真实资金 12,000 USDT")).toBeInTheDocument()
  expect(screen.queryByText("总资金 112,000 USDT")).not.toBeInTheDocument()
})
```

- [ ] **Step 2: 写完整资金池审批测试**

```tsx
it("shows the immutable connection plan and revision before approval", async () => {
  renderApproval(bookApproval({ revision: 42, targets: [["OKX Live", 12000], ["Bybit Live", 30000]] }))
  expect(screen.getByText("配置版本 42")).toBeInTheDocument()
  expect(screen.getByText("OKX Live")).toBeInTheDocument()
  expect(screen.getByText("Bybit Live")).toBeInTheDocument()
  expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument()
})
```

- [ ] **Step 3: 写周期层级和 attention 测试**

```tsx
it("renders shared signals once and partial connection failure under its book", async () => {
  renderCycleDetail(partialCycle())
  expect(screen.getAllByText("LLM 四智能体委员会")).toHaveLength(1)
  expect(screen.getByText("Simulation · 部分完成")).toBeInTheDocument()
  expect(screen.getByText("Bybit Testnet · 失败")).toBeInTheDocument()
  expect(screen.getByText("需要人工处理")).toBeInTheDocument()
})
```

- [ ] **Step 4: 运行前端测试确认旧单组合视图失败**

Run: `cd web && npm test -- --run src/pages/cycles src/pages/risk/components/approval-book-plan.test.tsx src/pages/dashboard/dashboard-books.test.tsx src/pages/decisions`

Expected: FAIL。

- [ ] **Step 5: 实现 hooks 和三层视图**

portfolio hook 返回 `simulatedBooks/realBooks`；cycle hook 返回共享 signal section 和 ordered book results。所有 status 使用统一 formatter，`partial` 和 `requires_attention` 不得映射成普通失败 toast 后丢失明细。

- [ ] **Step 6: 运行前端全量测试和构建**

Run: `cd web && npm test -- --run`

Run: `cd web && npm run typecheck && npm run lint && npm run build`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add web/src
git commit -m "refactor: show portfolios and cycles by execution book"
```

---

### Task 18: 硬删除旧配置、单平台执行与全部残留调用方

**Files:**
- Delete: `src/cryptotrader/config.py`
- Delete: `src/cryptotrader/execution/exchange.py`
- Delete: `src/cryptotrader/execution/simulator.py`
- Delete: `src/cryptotrader/portfolio/exchange_reader.py`
- Delete: `src/cryptotrader/profiles/repository.py`
- Delete: `src/api/routes/signal_profile.py`
- Delete: `web/src/hooks/use-signal-profile.ts`
- Delete: `config/default.toml`
- Delete: `config/local.toml`
- Modify: `src/cryptotrader/profiles/models.py`
- Modify: `src/api/dependencies.py`
- Modify: `src/cryptotrader/security.py`
- Modify: `src/cryptotrader/log_config.py`
- Modify: `src/cryptotrader/otel.py`
- Modify: `src/cryptotrader/notifications.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `web/README.md`
- Modify: `pyproject.toml`
- Modify: `scripts/staging_validate.py`
- Create: `scripts/import_smoke.py`
- Delete/Rewrite: all tests importing deleted symbols, especially `tests/test_config_loader.py`, `tests/test_config_validation.py`, `tests/test_unified_config.py`, `tests/test_config_pair_object_form.py`, `tests/test_env_override.py`, `tests/test_credentials_and_model_timeout.py`, `tests/test_agent_config.py`, `tests/test_live_exchange_pair.py`, `tests/test_read_portfolio_spot_merge.py`。
- Create: `tests/test_hard_cutover.py`
- Rewrite: `tests/test_docker_compose.py`
- Rewrite: `tests/test_staging_validate.py`
- Rewrite: `tests/test_pyproject_config.py`

**Interfaces:**
- Final runtime has one config repository, one venue abstraction and one execution-book cycle。
- Docker/CLI/API health start from `DATABASE_URL/CONFIG_MASTER_KEY` only。
- Removed test behavior is replaced by RuntimeConfig/adapter/book behavior tests, not renamed compatibility tests。

- [ ] **Step 1: 写硬切换静态门禁**

```python
FORBIDDEN_RUNTIME_PATTERNS = (
    r"\bload_config\b",
    r"\bAppConfig\b",
    r"\bExchangeCredentials\b",
    r"\bExchangesConfig\b",
    r"\bCRYPTOTRADER_",
    r"config/(default|local)\.toml",
    r"\bexchange_id\b",
    r"\bLiveExchange\b",
    r"\bsupports_protection_orders\b",
)


def test_forbidden_symbols_are_absent_from_runtime_tests_docker_and_web():
    violations = scan_paths(("src", "tests", "scripts", "web/src", "Dockerfile", "docker-compose.yml"), FORBIDDEN_RUNTIME_PATTERNS)
    assert violations == []


def test_config_directory_contains_no_runtime_toml():
    assert not Path("config/default.toml").exists()
    assert not Path("config/local.toml").exists()
```

设计与实施 Markdown 允许解释旧符号，因此门禁不扫描 `docs/superpowers`。

- [ ] **Step 2: 写生产模块导入 smoke**

`scripts/import_smoke.py` 使用 `pkgutil.walk_packages` 导入 `cryptotrader`、`api`、`cli` 下所有生产模块，排除仅在调用时需要网络的 provider side effects；模块导入不得读取除两个 bootstrap 变量外的配置，不得连接交易所。

Run: `uv run python scripts/import_smoke.py`

Expected before deletion: PASS；若当前模块有 import side effect，先在同一步把 side effect 移到显式 factory。

- [ ] **Step 3: 删除旧文件并迁移安全、日志、OTel、通知调用方**

这些模块改为构造函数接受对应 `RuntimeConfigDocument` 子配置。API dependency 从 app.state runtime 读取鉴权配置；setup_required 只允许初始化配置流程。删除 `python-dotenv`、`tomli` 和 `.env` 自动加载依赖。任何旧字段错误直接改调用方，不新增兼容属性。

- [ ] **Step 4: 迁移 Docker、Compose、脚本与文档**

容器只要求 `DATABASE_URL`、`CONFIG_MASTER_KEY`，首次启动文档指向网页向导。删除 volume/command 中 TOML 路径、exchange/mode flags 和旧 `arena` 示例。staging validation 先检查 DB schema/config revision，再检查 runtime health 和每个启用 connection health。

- [ ] **Step 5: 重写或删除旧测试**

删除只验证 TOML merge/env precedence/旧 dataclass 的测试；仍有业务价值的测试迁移到 Task 1–17 新接口。禁止通过 test-only alias 让旧测试继续运行。

- [ ] **Step 6: 运行硬切换门禁与所有入口 smoke**

Run: `uv run pytest tests/test_hard_cutover.py tests/test_runtime_entrypoints.py tests/test_docker_compose.py tests/test_staging_validate.py tests/test_pyproject_config.py --no-cov -q`

Run: `uv run python scripts/import_smoke.py`

Expected: PASS。

- [ ] **Step 7: 手工零命中复核**

Run:

```bash
rg -n --glob '!docs/superpowers/**' --glob '!*.lock' 'load_config|AppConfig|ExchangeCredentials|ExchangesConfig|CRYPTOTRADER_|config/(default|local)\.toml|exchange_id|LiveExchange|supports_protection_orders' src tests scripts web/src Dockerfile docker-compose.yml pyproject.toml
```

Expected: exit 1，零输出。若命中，修改消费者；不得增加 ignore。

- [ ] **Step 8: 提交原子硬切换**

```bash
git add -A
git commit -m "refactor: remove legacy config and single-venue runtime"
```

---

### Task 19: 全量自动化验证与独立代码审查

**Files:**
- Modify only when verification exposes a root-cause defect; do not weaken assertions or add compatibility。

**Interfaces:**
- Verifies: backend、frontend、coverage、lint、build、容器配置、导入、秘密扫描、hard-cutover zero search。
- Consumes: Tasks 1–18 完整实现。

- [ ] **Step 1: 后端全量测试和覆盖率**

Run: `uv run pytest -q`

Expected: PASS，branch coverage 不低于项目门槛 70%。记录 passed 数和 coverage。

- [ ] **Step 2: Ruff 与格式检查**

Run: `uv run ruff check src tests scripts`

Run: `uv run ruff format --check src tests scripts`

Expected: PASS。

- [ ] **Step 3: 前端全量验证**

Run: `cd web && npm test -- --run`

Run: `cd web && npm run typecheck && npm run lint && npm run build`

Expected: PASS。

- [ ] **Step 4: 启动与容器静态验证**

Run: `uv run python scripts/import_smoke.py`

Run: `docker compose config`

Run: `uv run pytest tests/test_hard_cutover.py tests/test_docker_compose.py tests/test_health_endpoint.py --no-cov -q`

Expected: PASS，Compose 输出只暴露两个 runtime bootstrap 变量名，不包含凭据值。

- [ ] **Step 5: 工作树、diff 和秘密扫描**

Run: `git diff --check`

Run: `git status --short`

Run: `rg -n 'api-marker|secret-marker|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|x-simulated-trading.*[A-Za-z0-9]{20}' src tests scripts web docs Dockerfile docker-compose.yml`

Expected: diff check PASS；秘密扫描零输出；status 只含预期实现文件。

- [ ] **Step 6: 使用 `superpowers:requesting-code-review` 做独立审查**

审查必须逐项验证：spec 覆盖、平台名称是否泄漏到业务层、模拟/真实隔离、revision/HITL 绑定、部分失败、凭据泄漏、所有入口硬切换、真实验收脚本安全边界。修复 P0/P1/P2 后重跑本任务全部命令。

- [ ] **Step 7: 使用 `superpowers:verification-before-completion` 复核证据**

只有读取本轮最新命令输出后才能声称自动化完成。

- [ ] **Step 8: 提交验证修复**

若有修改：

```bash
git add <only-files-changed-by-review>
git commit -m "fix: address multi-venue review findings"
```

若无修改，不创建空提交。

---

### Task 20: 真实模型与模拟交易闭环验收，之后才合并推送主分支

**Files:**
- Create: `scripts/venue_canary.py`
- Create: `scripts/signal_canary.py`
- Create: `tests/test_canary_safety.py`
- Modify: `README.md`

**Interfaces:**
- `venue_canary.py --connection <id> --pair <pair>` 只允许 `paper|demo|testnet`，从数据库读取连接和密文凭据，不接受命令行 secret。
- `signal_canary.py --pair <pair>` 使用真实 market source、Kronos 和真实 LLM 四智能体内部辩论，默认不进入 execution books。
- 两个脚本输出结构化 cycle/connection/model IDs，不输出 prompts 中的凭据或平台 raw auth。

- [ ] **Step 1: 写 canary 安全门禁**

```python
def test_venue_canary_refuses_live_environment():
    result = runner.invoke(venue_canary, ["--connection", "bybit-live", "--pair", "BTC/USDT"])
    assert result.exit_code == 2
    assert "live connections are read-only in canary" in result.stdout


def test_signal_canary_defaults_to_no_execution():
    options = parse_signal_canary_args(["--pair", "BTC/USDT"])
    assert options.execute is False
```

- [ ] **Step 2: 实现 venue canary 的强制清理流程**

流程固定为：读取 snapshot/connection → 拒绝 live → 连接健康与余额读取 → 确认初始零目标 pair 仓位/挂单/保护单 → 以平台最小合法名义开仓 → 安装并查询平台侧止损止盈 → reduce-only 完整平仓 → 取消 canary 保护/挂单 → 关闭 session → 新进程重新连接验证零仓位、零挂单、零保护单。任一步失败都进入 `finally` 清理并在结束时打印 residual state；有残留返回非零且 `requires_attention=true`。

- [ ] **Step 3: 实现真实 signal canary**

脚本读取当前 active snapshot，执行一次真实行情收集、Kronos、四智能体分析、辩论门控、多轮交叉辩论、verdict 和融合；打印每个组件 direction/confidence、模型实际 ID、辩论轮次、fused score、target position 和 config revision。任何模型调用失败整次失败，禁止 mock fallback。

- [ ] **Step 4: 运行 canary 单元安全测试**

Run: `uv run pytest tests/test_canary_safety.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 真实 Bybit Testnet 闭环**

前置：网页创建并测试 `bybit-testnet`，凭据保存数据库，connection 位于 simulated book 或仅为 canary 启用。

Run: `uv run python scripts/venue_canary.py --connection bybit-testnet --pair BTC/USDT`

Expected: health PASS；最小开仓 PASS；平台侧保护查询 PASS；完整平仓 PASS；独立重连后 position/orders/protections 全部为零。

- [ ] **Step 6: OKX Demo 服务恢复后执行相同闭环**

Run: `uv run python scripts/venue_canary.py --connection okx-demo --pair BTC/USDT`

Expected: 与 Step 5 相同。若平台仍返回服务级故障，保存原始 request ID/错误码和零残留证据，不把该平台标为验证通过。

- [ ] **Step 7: 真实 Kronos + LLM 四智能体内部辩论验证**

Run: `uv run python scripts/signal_canary.py --pair BTC/USDT`

Expected: Kronos 和 LLM committee 均为真实调用；LLM 输出包含四个角色、内部辩论轮次和实际 model ID；融合与 target 生成成功；无 execution。

- [ ] **Step 8: 实盘只读能力验证**

对已配置 live connection 只调用 test connection、fetch portfolio、fetch capabilities 和 list open state。不得调用 `place_order` 或 protection write；记录 environment=live 和 adapter 能力即可。

- [ ] **Step 9: 重新运行 Task 19 全量验证**

真实 canary 不替代自动化。完成真实验证后再次运行 backend、Ruff、frontend、build、import smoke、hard-cutover search 和 `git diff --check`。

- [ ] **Step 10: 使用 `superpowers:finishing-a-development-branch` 完成合并推送**

只有以下证据同时成立才允许执行：全量自动化绿色、Bybit Testnet 完整闭环且零残留、真实 LLM 四智能体辩论成功、工作树干净、独立 review 无未解决 P0/P1/P2。然后：

```bash
git switch main
git pull --ff-only
git merge --ff-only codex/pluggable-signal-fusion
git push origin main
```

若主分支不能 fast-forward，停止并审查差异；不得自动 rebase/merge 远端新提交后直接推送。若真实模型或模拟盘仍不可用，保留 feature branch 并准确报告 blocker，不提前合并。

---

## 最终验收清单

- [ ] 网页可以从空数据库完成首次初始化并激活系统。
- [ ] 数据库是除两个 bootstrap 参数外的唯一运行配置源。
- [ ] 凭据密文使用 AES-GCM，所有公开面只有 `configured`。
- [ ] Kronos、LLM 四智能体内部辩论和自定义组件仍是可调信任权重的信号组件。
- [ ] 自定义信号和平台实现通过 Python entry points 安装，网页只配置已安装插件。
- [ ] 同一信号周期只运行一次，并同时驱动 Simulation 与 Live。
- [ ] OKX、Bybit、Paper 都实现相同 VenueAdapter 契约。
- [ ] 一个或多个连接都通过同一 ExecutionBook/WeightedAllocationPolicy 工作。
- [ ] 模拟与真实权益、持仓、PnL、风控和审批完全隔离。
- [ ] HITL 按资金池配置并绑定完整计划和 revision。
- [ ] 多平台部分成功、保护失败补偿和 `requires_attention` 可从 API/UI/Journal 准确看到。
- [ ] Backtest 只使用临时 Paper 资金池且与生产决策契约一致。
- [ ] API、CLI、Scheduler、Chat、Triggers、Docker、脚本、Web 和测试全部迁移新入口。
- [ ] 旧 TOML、旧配置模型、旧单平台执行和旧符号在运行时范围零命中。
- [ ] Bybit Testnet 开仓/保护/平仓/零残留真实通过。
- [ ] OKX Demo 服务可用时同样通过；不可用时有明确 blocker 和零残留证据。
- [ ] 真实 Kronos 与 LLM 四智能体内部辩论、融合和目标仓位通过。
- [ ] 全量 backend/frontend/lint/build/import/container 验证通过。
- [ ] 只有全部强制门槛满足后才合并并推送 `main`。
