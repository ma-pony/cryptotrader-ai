# 第一批：扩展与决策 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增平台和组件只需后端代码注册，网页完成配置并能查看持久化分析结果。

**Architecture:** 目录和运行工厂共用代码注册表；凭据按平台模型校验后加密。抽出无账户依赖的分析服务，交易周期调用同一分析服务；历史查询直接读取 journal。

**Tech Stack:** 现有 Python 3.12+／FastAPI／Pydantic 2／SQLAlchemy、React／TanStack Query／Zod；不新增依赖。

**Spec:** [设计规格](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/specs/2026-08-31-trading-workbench-design.md)，执行约束与命名见[总计划](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-trading-workbench.md)。

## Global Constraints

- Kronos、LLM 四智能体和后续自定义组件是同级信号来源。LLM 内部辩论保留。
- 新增信号类型或平台类型通过后端代码实现和注册。本期不做插件安装、市场、包发现、热加载或浏览器上传代码。
- 凭据只写不回显，不进入普通配置草稿、历史快照或浏览器持久化缓存。
- 直接替换旧流程，不长期维护双套页面、双套业务规则或兼容路由。
- 本批只修改代码并使用隔离测试数据；不更改运行库、不启用交易、不调用外部模型、不提交推送。

---

## 文件责任图

路径相对仓库根目录。Create 表示拟新增；其余路径已经存在。F1–F5 涉及的配置／凭据文件已有未提交修改，先阅读 diff。

| 文件 | 操作及责任 |
| --- | --- |
| `src/cryptotrader/configuration/registry.py` | Create；代码注册条目及惰性工厂 |
| `src/cryptotrader/configuration/catalog.py`、`fields.py`、`parameters.py` | Modify；声明、参数与凭据校验；移除包发现 |
| `src/cryptotrader/signals/registry.py`、`venues/registry.py`、`market_sources/registry.py` | Modify；实例化只消费统一注册表 |
| `src/cryptotrader/runtime_config/secrets.py`、`repository.py`、`models.py` | Modify；动态密文载荷、版本、能力配置 |
| `src/cryptotrader/venues/models.py`、`protocol.py`、`paper.py`、`okx.py`、`bybit.py` | Modify；环境声明、凭据消费 |
| `src/cryptotrader/migrations/__init__.py`、`workbench.py` | Create；显式一次性迁移，不从 HTTP 请求隐式迁移 |
| `src/cryptotrader/signals/presentation.py` | Create；结果块和评估参考模型 |
| `src/cryptotrader/signals/models.py`、`components/kronos.py`、`components/llm_committee.py` | Modify；产生当次展示数据 |
| `src/cryptotrader/decision/analysis.py`、`read_service.py`、`service.py` | Create；分析、读记录、统一运行入口 |
| `src/cryptotrader/journal/models.py`、`store.py` | Modify；运行元数据、快照、展示块及状态持久化 |
| `src/cryptotrader/trading_cycle.py`、`runtime.py` | Modify；拆分分析与账户执行生命周期 |
| `src/cryptotrader/tasks.py` | Create，迁入并简化现有 `chat/task_manager.py` 的任务管理，不另造通用任务系统 |
| `src/api/routes/analyses.py`、`trading_runs.py`、`runtime_status.py` | Create；明确运行模式与就绪接口 |
| `src/api/routes/config.py`、`venues.py`、`decisions.py`、`hitl.py`、`src/api/dependencies.py`、`main.py` | Modify；契约、访问边界和接线 |
| `src/cryptotrader/scheduler.py`、`src/cli/main.py` | Modify；调度／CLI 使用新运行服务 |
| `web/src/types/api.schema.ts`、`api.ts` | Modify；严格响应模型与类型 |
| `web/src/components/configuration/credential-panel.tsx`、`parameter-fields.tsx` | Modify；通用字段和凭据体验 |
| `web/src/hooks/use-venue-connections.ts`、`use-connection-checks.ts` | Modify；完整保存与版本关联的检查 |
| `web/src/pages/accounts/connection-form.tsx`、`connections-page.tsx` | Create；迁入旧 venues 表单和列表的有效逻辑 |
| `web/src/components/signals/result-blocks.tsx`、`web/src/pages/engine/component-detail.tsx`、`index.tsx` | Create；通用组件结果、历史和配置入口 |
| `web/src/hooks/use-decisions.ts`、`use-runtime-status.ts`、`use-trading-runs.ts` | Create；新接口查询和操作 |
| `web/src/pages/decisions/index.tsx`、`detail.tsx` | Modify 列表，Create 统一详情 |
| `web/src/locales/zh-CN/configuration.json`、`common.json`、`cycles.json` | Modify；中文状态，不新增平台 ID 翻译白名单 |

## F1：统一后端注册与动态凭据契约

**Files:** 上表 registry／catalog／fields、三个运行注册表、secrets／repository、venues、config／venues API、迁移文件；Modify `pyproject.toml`。Test：Modify `tests/test_configuration_catalog.py`、`test_signal_registry.py`、`test_venue_registry.py`、`test_runtime_config_secrets.py`、`test_api_security_hardening.py`；Create `tests/factories/workbench_extensions.py`、`tests/test_workbench_migration.py`。

**Interfaces:**

- Consumes：`configuration_fields(model)`、`CredentialVault.seal/open`、当前 CAS 配置替换。
- Produces：`ExtensionRegistry(components, venues, market_sources)`，值为 `ExtensionRegistration(configuration, factory)`；`get_extension_registry() -> ExtensionRegistry` 是目录和运行时唯一入口。
- `ComponentFactoryContext(document, events, llm_gateway_key, llm_factory_builder)` 封装现有依赖；每个组件工厂接受一个 context，运行注册器不再判断 `component_id == 'llm_committee'`。
- `CredentialField(key, label, description, required)`；`EnvironmentDefinition(id, label, capital_scope)`；`PluginConfiguration` 增加 `credential_model` 和环境描述，原 `credential_fields: tuple[str]` 删除。
- `ComponentDependency(kind, key, label, configuration_path)`，kind 为 market/model_service/local_artifact/context；`PluginConfiguration.dependencies(parameters) -> tuple[ComponentDependency, ...]` 只根据参数声明所需依赖，不构造组件。F5 用它生成组件级缺项。
- `venue_definition(adapter_id: str, environment: str) -> VenueDefinitionOut` 返回当前环境适用的 fields、credential_fields、margin_modes、杠杆与账户／退出能力。目录列全部类型／环境，选中后调用 `/api/config/catalog/venues/{adapter_id}?environment=…` 获取适用定义，仍不创建会话。
- `CredentialPayload(values: dict[str, SecretStr])`；`validate_venue_credentials(adapter_id: str, environment: str, values: dict[str, str]) -> CredentialPayload`。凭据字段来自对应模型，未知字段拒绝；验证错误仅含字段名／代码，不含原输入。

CredentialVault.seal 在加密边界显式取 `get_secret_value()` 构成明文 bytes，立即加密；不能继续直接 `model_dump_json()` 把 SecretStr 的掩码存进去。普通序列化／日志始终保留掩码，TokenPayload 的独立流程保持不变。

- [ ] 建立后端测试类型 `sample_venue`：普通字段 `account_code`，凭据 `access_token` 必填、`tenant_pin` 可选，环境 `sandbox` 属于 simulation。建立 `sample_signal`：窗口参数、指标／表格输出。工厂调用次数由测试 spy 记录。

```python
def test_catalog_does_not_instantiate_extensions(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import configuration_catalog
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    definition = configuration_catalog().require_venue("sample_venue")
    assert definition.environments[0].capital_scope == "simulation"
    assert [field.key for field in definition.credential_fields] == ["access_token", "tenant_pin"]
    assert calls == []
```

- [ ] 运行红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_configuration_catalog.py tests/test_runtime_config_secrets.py -q`。目标失败是目录／载荷契约不满足，不接受意外网络或依赖缺失作为红测。
- [ ] 实现代码注册条目。内置类型声明只引入轻量参数模型；工厂函数体内部才导入组件／适配器并构造实例。移除 `metadata.entry_points` 和运行 `load_factory(path)` 入口；行情源同样改代码注册，避免留下另一条安装发现链。

```python
@dataclass(frozen=True)
class ExtensionRegistration:
    configuration: PluginConfiguration
    factory: Callable

def configuration_catalog() -> ConfigurationCatalog:
    registry = get_extension_registry()
    return ConfigurationCatalog(
        components={key: item.configuration for key, item in registry.components.items()},
        venues={key: item.configuration for key, item in registry.venues.items()},
        market_sources={key: item.configuration for key, item in registry.market_sources.items()},
    )
```

- [ ] 将环境合法性与资金性质判断改为查询平台声明；删除固定环境枚举造成的新平台阻断。实现环境适用定义查询和组件依赖纯计算，支持现有保证金／杠杆能力声明。凭据模型在适配器边界取 `values`，不得再截取三字段；加密绑定仍用连接 credential_ref。
- [ ] 完成临时库密文往返、错误脱敏和迁移测试：解密后的 `values['access_token'].get_secret_value()` 必须精确等于测试原值，不能是掩码；自定义字段可取给适配器。响应、日志、普通配置／历史均无原值；可选字段不要求填写。旧密文显式迁移后值与关联不变，普通运行没有双格式分支。移除 pyproject 三组 entry points，将 `tests/test_plugin_entry_points.py` 的有效断言改写为代码注册测试。
- [ ] 重跑本任务全部测试及 `tests/test_runtime_config_models.py`。通过标准：目录零构造、所有内置类型可创建、自定义环境可保存、凭据不丢字段；保留 diff，未获授权不提交。

## F2：网页动态配置、凭据状态与自动检查

**Files:** 上表账户表单／列表、credential-panel、parameter-fields、两个连接 hooks、API schemas、config／venues API；Modify `web/src/test/configuration-catalog-fixture.ts`、`configuration-workflow.tsx`。Test：Create `web/src/pages/accounts/connection-form.test.tsx`；Modify `web/src/hooks/use-connection-checks.test.tsx`、`web/src/pages/settings/credential-state.test.tsx`、`tests/test_venue_connections_api.py`。

**Interfaces:** Consumes F1 目录、环境适用定义查询及 `PUT /api/venue-connections/{id}/credentials {expected_revision, values}`。Produces `ConnectionForm({connectionId?: string, onSaved: (id: string) => void})`、普通保存状态和凭据状态；`ConnectionHealthOut` 的 fingerprint 绑定当前参数与凭据版本。环境变化只按服务端定义显示字段和能力，不查前端平台 ID 表。

- [ ] 先写表单行为测试：目录返回 `sample_venue` 时出现“账户编码／访问令牌”；填写并保存后非三字段载荷完整，刷新只显示“凭据已配置”。复用现有 workflow 渲染工具，新增 fixture 数据不能在生产前端导入。

```tsx
expect(credentialsRequest).toEqual({
  expected_revision: savedRevision,
  values: { access_token: 'fixture-token' },
});
expect(screen.getByText('凭据已配置')).toBeVisible();
expect(screen.queryByDisplayValue('fixture-token')).not.toBeInTheDocument();
expect(checkRequests).toHaveLength(1);
```

- [ ] 红测：在 `web/` 执行总计划的 Vitest 命令，目标 `src/pages/accounts/connection-form.test.tsx src/hooks/use-connection-checks.test.tsx`。
- [ ] 从旧 `settings/venues/venue-form.tsx` 迁入有效逻辑，用字段描述构成载荷，按 required 决定完整性。凭据初始只显示状态／时间和“更换凭据”；成功清空内存、失败保留当前输入，无 localStorage。无凭据的平台不渲染该区域。

```tsx
const values = Object.fromEntries(
  definition.credential_fields
    .filter((field) => credentialDraft[field.key]?.length)
    .map((field) => [field.key, credentialDraft[field.key]]),
);
const complete = definition.credential_fields
  .filter((field) => field.required)
  .every((field) => Boolean(values[field.key]));
```

- [ ] 一个“保存并检查”动作顺序保存连接和本次凭据，使用上一写入返回的新 revision；全部需要的保存成功才调用一次检查。已配置凭据不更换时复用服务端状态。缺必填允许保存基本信息并列缺项；凭据失败显示“连接已保存，凭据保存失败”。普通表单 dirty 状态不冒用旧检查成功。
- [ ] 自动检查由这条保存流程统一触发；保留 GET 已保存检查和手动重新检查。页面 mount 不无条件检测。检查成功／失败都保存匹配 fingerprint，过期结果不得覆盖新配置状态。清除整套凭据用独立 DELETE 与确认，不把空输入解释成清除。
- [ ] 加测试覆盖无凭据平台、可选字段、参数修改后一次再检查、两阶段部分失败和返回页面；重跑前端目标、现有配置草稿与 credential-state 测试、`tests/test_venue_connections_api.py`。浏览测试替身断言没有 place_order 调用。

## F3：通用结果块、Kronos 曲线与组件详情

**Files:** 上表 presentation、signals/models、两个内置组件、journal 编解码、result-blocks、engine 页面；Modify `web/src/components/charts/trend-chart.tsx` 仅当现有图表需要预测边界能力。Test：Create `tests/test_signal_presentation.py`、`web/src/components/signals/result-blocks.test.tsx`、`web/src/pages/engine/component-detail.test.tsx`；Modify `tests/test_multi_venue_journal.py`。

**Interfaces:**

- Consumes F1 注册目录、`ComponentSignal`、现有 Kronos DataFrame 与 LLM 辩论结果。
- Produces `ResultBlock = TextBlock | MetricsBlock | SeriesBlock | TableBlock | TimelineBlock`，用 `kind` 做判别联合；每块 `title` 为中文。文本 body；指标 key/value/unit/note；序列 name/unit/points(time,value)、forecast_start；表格 columns 和 `rows: list[TableRow]`，每行 `cells: list[TableCell]`，单元格固定 column_key/value 字段、value 为标量；时间线 time/actor/body。所有模型禁止多余字段、HTML 和脚本，不用任意响应对象承载表格。
- `EvaluationReference(reference_time, reference_price, due_at, interval, market_source_id)`；`ComponentSignal` 新增 blocks、evaluation_reference、duration_ms 和已有证据可提供的 usage/cost（不可得为 null）。`status` 明确 `completed/skipped/failed`，保留 direction 中立语义。
- `ResultBlocks({blocks: ResultBlock[]})`、`ComponentDetail({componentId: string})`，不分支组件 ID。

- [ ] 红测先覆盖序列与时间线落库后读取相等、skipped 无预测序列、错误原因脱敏及测试组件表格渲染；用 fixture predictor，调用记录在历史查询前后不增加。

```python
assert restored.component_signals[0].blocks == original.component_signals[0].blocks
assert restored.component_signals[0].evaluation_reference == original.component_signals[0].evaluation_reference
assert predictor.call_count == 1
assert skipped.status == "skipped"
assert not any(block.kind == "series" and block.forecast_start for block in skipped.blocks)
```

- [ ] 运行 `rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_signal_presentation.py tests/test_multi_venue_journal.py -q`。
- [ ] 定义严格展示块模型，并让 journal 序列化支持这些确定结构。Kronos 在实际预测时保存历史采样、完整预测时间和值、起点及已有置信度分项；门控跳过返回说明块。弱信号过滤保留本次实际预测，并在原因中解释过滤。LLM 保存四智能体意见与完整辩论时间线，融合输入仍只有一个 llm_committee。

```python
class SeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    time: datetime
    value: Decimal | None

class EvaluationReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_time: datetime
    reference_price: Decimal
    due_at: datetime
    interval: str
    market_source_id: str
```

- [ ] 在 `runtime_config/models.py`、`api/routes/config.py` 和引擎表单增加 `market_data.timeframe: str = '1h'` 作为全局参考周期，以及 `signals.evaluation_interval: str | None`；null 表示使用前者。行情采集合并全局参考周期与各组件需求，不能假定组件的第一种周期就是全局周期。产生信号时解析实际周期并冻结 due_at，用参考周期最新已收盘价／时间，不用浏览器时间生成历史。
- [ ] 写通用 renderer 的有限 switch(kind)，组件页显示概览／历史／配置，权重百分比合计提示由后端验证。未知费用显示“费用未知”，零目标显示“目标持仓为零”。Kronos 历史页从记录显示图表，LLM 同页显示辩论，不拉起推理。
- [ ] 运行本任务后端与三个新增前端测试；验证空历史下一步、预测区标注、窄屏表格滚动、键盘展开，新增测试组件无需翻译 ID 映射。测试使用保存的确定样本，禁止伪造生产预测。

## F4：独立分析与统一持久化决策记录

**Files:** 上表 decision 三文件、journal、trading_cycle、runtime、tasks、analyses／decisions API、dependencies／main、前端 decisions 页面／hook／schemas。Test：Create `tests/test_analysis_isolation.py`、`tests/test_decision_read_api.py`、`web/src/pages/decisions/detail.test.tsx`；Modify `tests/test_runtime_entrypoints.py`、`test_runtime_lifecycle.py`、`test_trading_cycle.py`、迁移测试。

**Interfaces:**

- `AnalysisResult(context, component_signals, fused_signal, target_position, failure)` 保留输入和失败阶段；failure 为结构化 code/stage/message 或 null。
- `SignalAnalysisService.analyze(pair: Pair, snapshot: RuntimeConfigSnapshot, as_of: datetime) -> AnalysisResult` 只注入行情源、registry、runner、fusion、decision engine、clock；构造函数不接受 venue registry、account service、approval store 或 execution service。
- `DecisionReadService.get(decision_id: str) -> DecisionOut | None`；`list(*, pair=None, mode=None, origin=None, revision=None, started_at=None, ended_at=None, limit=50, offset=0) -> DecisionListOut`。mode 为 analysis/trading/backtest；origin 为 manual/scheduled/trigger/backtest。
- `DecisionOut` 包含 decision_id、pair、mode、origin、config_revision、config_snapshot、created_at、finished_at、status、components、fusion、target、books、failure。analysis 的 books 固定空，执行状态不伪造 completed；未知旧快照列 incomplete_fields。
- `RunService.start_analysis(pair: Pair, expected_revision: int) -> str` 持久化 queued 后返回 ID；交易入口在 F5 增加。迁用现有任务管理器调度协程，终态和历史以数据库为准。

- [ ] 新增隔离测试，运行配置里保留真实授权和启用资金池，给账户工厂、读取、下单和审批注册禁止调用的 spy；执行真实分析服务和 fake signal，而非整段 mock 分析。

```python
decision_id = await run_service.start_analysis(pair, snapshot.revision)
await task_manager.drain()
record = await read_service.get(decision_id)
assert record.mode == "analysis"
assert record.books == []
venue_factory.assert_not_called()
account_reader.assert_not_called()
order_writer.assert_not_called()
approval_writer.assert_not_called()
```

- [ ] 红测运行 `rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_analysis_isolation.py tests/test_decision_read_api.py -q`；将测试 fixture 所需依赖在同文件构造，禁止调用 build_runtime 时偷建真实账户会话。
- [ ] 从 `TradingCycle.run` 提取行情→组件→融合→目标过程。Runtime 的只读／分析服务独立构建，账户会话延迟到明确交易／账户同步职责；配置目录读取也不能加载推理模型。交易周期消费 AnalysisResult 后执行原逐池过程。

```python
result = await self.analysis.analyze(request.pair, snapshot, created_at)
if result.failure is not None:
    return await self.save_failed_analysis(result, request, created_at)
return await self.prepare_and_execute_books(result, request, created_at)
```

这里 `TradingCycle.save_failed_analysis(result, request, created_at)` 返回失败的 `CycleOutcome`，`prepare_and_execute_books(result, request, created_at)` 返回原交易 `CycleOutcome`；两者由本任务从现有 run 内代码抽出，禁止另建交易链。

- [ ] journal 补 queued/running/terminal 元数据与不可变配置摘要，开始时写记录、结束时更新合法状态。安全摘要从允许字段生成，排除凭据与 provider tokens；展示与模型身份按当次保存。扩展 store.list 筛选，不依赖活动 runtime.cycle。重启时尚在 queued/running 的记录标 interrupted，不做自动恢复订单执行。
- [ ] 新增 `POST /api/analyses` 和 `/api/decisions` 查询。认证仍有效；取消读接口因 inactive 被拒绝。前端详情统一呈现组件、目标、逐池结果；analysis 到目标处结束。状态轮询通过 query，关闭页不取消后台运行，完成后详情仍可读。
- [ ] 通过隔离、暂停后历史、改权重不改旧记录、失败阶段、重启读取与凭据脱敏测试；运行现有交易主链回归。写迁移夹具证明旧 ID 保留、缺失曲线不补造。本任务不向运行库应用迁移。

## F5：交易范围确认、就绪与自动运行总开关

**Files:** 上表 runtime_status／trading_runs、RunService、runtime、scheduler、main、CLI、hitl、runtime_config/models、三个前端 hooks；Create `web/src/components/trading/run-dialog.tsx`。Test：Create `tests/test_run_controls.py`、`web/src/components/trading/run-dialog.test.tsx`；Modify `tests/test_multi_book_cycle.py`、`test_runtime_entrypoints.py`。

**Interfaces:** Consumes F4 AnalysisService／ReadService、当前逐池审批。Produces `RunService.trading_scope(pair) -> TradingScopeOut`、`start_trading(pair, expected_revision, confirmed_book_ids) -> str`、`set_automation(enabled, expected_revision) -> RuntimeConfigSnapshot`。

`ReadinessOut` 分 `analysis`、`trading` 能力，各含 ready 与 `reasons[{code,message,path}]`；同时返回每个组件的依赖就绪及缺项、saved_revision/applied_revision/apply_error、automation_enabled、latest_run_at。就绪由 F1 的依赖声明和当前配置／文件可用性计算，不为检查就绪而调用模型。`TradingScopeOut` 每池包含连接 ID、资金性质、enabled、eligible、原因与 hitl_required；B3 再接入实时品种／完整风险事实。

- [ ] 先写全局范围和暂停测试：选一品种，确认所有 eligible books；少确认一个或配置 revision 已变返回冲突。停用资金池被排除。暂停后 scheduled 与 trigger 均不调用运行，手动明确发起不受自动开关代替授权。

```python
await run_service.set_automation(False, snapshot.revision)
await scheduled_callback(pair)
await trigger_callback(pair, {"source": "price_change"})
assert cycle.run.call_count == 0
assert (await read_service.get(saved_id)).decision_id == saved_id
```

- [ ] 红测 `rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_run_controls.py tests/test_runtime_entrypoints.py -q`。
- [ ] 在 DB scheduler 配置增加 `automation_enabled` 一个总开关；原 scheduler.enabled／triggers.enabled 只控制对应规则来源，不能绕过总开关。旧 scheduler.pairs 迁到 `execution.pairs: tuple[str, ...]`，以已有 `Pair.parse` 区分现货／合约；定时、触发和手动交易共用该范围，分析按行情源可读品种选择。入口检查与触发回调使用同一快照；暂停只阻止新运行。应用配置后更新两个 owner，不停止独立读服务，startup／保存不得隐式打开开关。

```python
if source in {"scheduled", "trigger"} and not snapshot.document.scheduler.automation_enabled:
    return None
enabled_books = tuple(book for book in snapshot.document.execution.books if book.enabled)
```

- [ ] 交易请求先核对 expected_revision、execution.pairs 和 confirmed_book_ids，再按原执行屏障运行。删除旧 system.active／setup_required 总激活门禁：Runtime 读服务常驻，分析按依赖就绪，交易依赖配置应用及资金池／连接启用和真实授权。修改 `_validate_active_document` 为按所需能力验证，允许保存完整的纯分析配置和未补齐凭据的账户草稿。显式迁移将旧 inactive 下的资金池设为停用，automation_enabled=false，保留原规则，不自动恢复执行。
- [ ] 保留逐资金池 HITL；删除未接主链的 `signals.hitl_required` 及 `profiles/models.py` 对应字段、UI、序列化和 fixtures。CLI、scheduler 和 main 的 trigger callback 全部调用 RunService，不能留绕过暂停／停用的 cycle.run 入口。相应变更同时修改 `runtime_config/defaults.py`、配置 API DTO、所有运行入口 fixture。
- [ ] 运行对话框先选品种，列模拟／真实池、账户及是否审批；真实授权缺失给具体原因，不在对话框顺手授权。页面“配置已保存”“已应用”“自动运行中”分开显示；仅分析按钮只依据分析依赖。连接已检查但未分配时显示“已连接，未参与执行”。
- [ ] 重跑本任务、runtime 生命周期与多池回归，补按钮确认测试和 inactive 可浏览测试。人工批准的最新全账户风险复核在 B3 完成前，不宣称 A7 全部通过。记录本批 F1–F5 的实际测试结果，再进入第二批。
