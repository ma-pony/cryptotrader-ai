# 交易工作台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可配置、可追溯的多组件、多账户交易工作台，完成配置、分析、模拟执行、复盘及人工退出流程。

**Architecture:** 保留全局信号融合与逐资金池执行架构，统一后端类型注册和持久化事实。每个领域同时交付接口与页面；最后整合导航、移除旧入口并验证完整流程。

**Tech Stack:** Python 3.12+、FastAPI、Pydantic 2、SQLAlchemy async、现有 SQLite/PostgreSQL 支持、pytest；React 19、React Router 7、TanStack Query 5、TypeScript、Vite、Vitest、Playwright。沿用现有依赖与锁文件。

**Spec:** [交易工作台与可扩展配置设计规格](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/specs/2026-08-31-trading-workbench-design.md)

## Global Constraints

- Kronos、LLM 四智能体和后续自定义组件是同级信号来源。LLM 内部辩论保留。
- 组件信任权重、平台连接、资金分配、人工审批及运行策略在网页配置，持久化到数据库。
- 新增信号类型或平台类型通过后端代码实现和注册。本期不做插件安装、市场、包发现、热加载或浏览器上传代码。
- 模拟账户和真实账户可以同时存在，不设置全局互斥的“模拟／实盘模式”。真实下单另有明确授权。
- 沿用一套全局决策引擎，向多个资金池分发结果。本期不引入多策略工作区、每池独立信号组合或可视化节点编排。
- 直接替换旧流程，不长期维护双套页面、双套业务规则或兼容路由。
- 凭据只写不回显，不进入普通配置草稿、历史快照或浏览器持久化缓存。
- 保存配置、连接检查、浏览历史不得触发交易。仅分析入口不访问交易账户。

---

## 当前状态与执行边界

日期：2026-08-31。用户已确认设计并要求继续制定实施计划。本文及四份子计划尚未执行，复选框不代表已经完成的代码。

本轮只写计划。后续代码实施默认限于当前工作区与隔离测试数据库，不自动提交、合并、推送、重启服务、改变运行配置或连接外部模型／交易账户。每批验收后保留本地改动，报告实际运行的测试与尚未验证的内容。提交由用户另行授权，不能直接执行技能模板中的自动提交步骤。

工作区存在 37 个已修改的受跟踪文件以及新凭据／检查组件，涉及配置目录、凭据、连接检查和中文界面；实施前逐项阅读重叠 diff。保留 `.pnpm-store/` 等无关文件，不整目录暂存，不覆盖已有改动。

## 分批计划与依赖

| 顺序 | 子计划 | 可独立验收的交付物 | 依赖 |
| --- | --- | --- | --- |
| 1 | [扩展与决策](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-workbench-01-extensions-decisions.md) | 后端代码注册、通用凭据、组件详情、独立分析和统一记录 | 当前源码及规格 |
| 2 | [账户与风险](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-workbench-02-accounts-risk.md) | 全账户同步、成交收益、整池风险、审批复核和人工退出 | F1–F5 |
| 3 | [研究与告警](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-workbench-03-research-alerts.md) | 组件评估、可靠回测、比较页面、持久化告警 | F3–F5、B1–B3 |
| 4 | [页面整合与验收](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-workbench-04-ui-integration.md) | 六入口、完整首次／次日流程、旧入口删除及扩展证明 | 前三批 |

执行采用 F1 → F2 → F3 → F4 → F5 → B1 → B2 → B3 → B4 → R1 → R2 → R3 → R4 → U1 → U2 → U3。文档拆分用于控制审查范围，不要求并行开发。各领域页面随对应任务落地，U1 负责导航和跨领域汇总。

## 共享接口与责任

以下是新契约的唯一命名来源；子计划可以增加私有辅助函数，不另起同义公共类型。

| 接口／对象 | 定义及责任 | 生产任务 | 消费任务 |
| --- | --- | --- | --- |
| `ExtensionRegistry` | `configuration/registry.py`；同一类型条目绑定声明和惰性工厂，包含 components/venues/market_sources | F1 | F2、F3、B1、U3 |
| `CredentialField`、`EnvironmentDefinition` | `configuration/fields.py`；凭据中文元数据；环境 ID、名称和 `capital_scope: simulated/real` | F1 | F2、B1、B2 |
| `ComponentDependency` | `configuration/fields.py`；行情／模型服务／本地模型文件／上下文的依赖声明，按参数纯计算；不加载模型 | F1 | F3、F5、U1 |
| `CredentialPayload` | `runtime_config/secrets.py`；平台校验后的 `values: dict[str, SecretStr]`，只写、加密，禁入响应 | F1 | F2、B1 |
| `ResultBlock`、`EvaluationReference` | `signals/presentation.py`；有限展示块，以及冻结的参考价、时间、期限 | F3 | F4、R1、R3 |
| `SignalAnalysisService.analyze(pair, snapshot, as_of)` | `decision/analysis.py`；返回 `AnalysisResult`，无账户会话依赖 | F4 | F5、R2 |
| `MultiVenueCycleRecord` | 沿用 `journal/models.py` 并补齐 pair、mode、origin、安全快照、运行状态；内部 `cycle_id` 对外映射为 `decision_id` | F4 | F5、B3、R1–R4 |
| `DecisionReadService.get/list` | `decision/read_service.py`；只依赖 journal store，返回 `DecisionOut` | F4 | 全部新读页面 |
| `ReadinessOut`、`TradingScopeOut` | `api/routes/runtime_status.py`；后端给出能力与缺项；交易范围包含所选品种、配置版本及每池能否参与 | F5 | B3、U1 |
| `AccountSnapshot`、`Fill`、`FundingEntry`、`Instrument` | `accounts/models.py`；带连接、币种、时间和来源的标准事实；金额使用 Decimal，DTO 输出字符串 | B1 | B2–B4、R2–R4 |
| `AccountSyncService.sync(connection_id)` | `accounts/sync.py`；只读采集并提交账户快照和账本，返回 `AccountSnapshot` | B2 | B3、B4、R4、U1 |
| `BookRiskState`、`BookRiskStateStore` | `risk/book_state.py`；整池当前／峰值权益、全品种及待成交占用、估值时间 | B3 | B4、R2、U1 |
| `ExecutionOwnership.book(book_id)` | `execution_ownership.py`；同一资金池的刷新、风控及写订单串行边界；保留现有配置应用屏障 | B3 | B4、HITL、所有策略来源 |
| `AccountOperationService.prepare/execute` | `accounts/operations.py`；停用确认、冻结退出计划、复核执行，独立于策略分析 | B4 | 账户页面、R4 |
| `EvaluationService.evaluate_due(now)` | `signals/evaluation.py`；用后续已收盘行情评估冻结输出 | R1 | 组件复盘、U1 |
| `BacktestStore` | `backtest/store.py`；完整运行状态、快照、成交、权益曲线和比较条件 | R3 | 研究页面、U3 |
| `AlertService.record(event)` | `alerts/service.py`；业务事件去重保存、状态关联；投递与业务处理分离 | R4 | U1、系统设置 |

### HTTP 契约

各响应使用明确 Pydantic DTO；前端同步更新 `web/src/types/api.schema.ts` 的 Zod schema 和 `web/src/types/api.ts`，不加入响应端任意对象逃生口。普通请求仍使用 JSON 传输，但界面只提供字段表单。

| 接口 | 请求关键字段 | 响应／副作用 |
| --- | --- | --- |
| `GET /api/config/catalog` | 无 | 注册类型、字段、凭据及环境能力声明；不实例化模型或连接 |
| `GET /api/config/catalog/venues/{adapter_id}?environment=…` | 平台 ID 与环境 | 该环境适用的参数、凭据、保证金／杠杆及读写能力声明；无网络请求 |
| `PUT /api/venue-connections/{id}/credentials` | `expected_revision, values` | 仅配置状态／时间；字段按平台校验 |
| `POST /api/analyses` | `pair, expected_revision` | 202，`decision_id`；纯信号分析 |
| `GET /api/trading-runs/scope?pair=…` | 选择的规范化品种 | 配置版本、参与／跳过资金池、账户及资金性质 |
| `POST /api/trading-runs` | `pair, expected_revision, confirmed_book_ids` | 202，`decision_id`；确认范围必须精确匹配当时可参与集合 |
| `GET /api/decisions`、`/{id}` | pair、mode、origin、revision、日期、limit/offset | 持久化记录；历史组件和执行数据来自当次快照 |
| `GET /api/runtime/status` | 无 | 保存／应用版本、分析与交易就绪、自动运行状态及原因 |
| `GET /api/accounts`、`/{id}` | 无 | 全账户、快照时间、同步失败、资金性质；停用账户仍出现 |
| `GET /api/accounts/{id}/fills`、`/income` | 品种、时间、分页 | 成交、费用及明确口径收益 |
| `POST /api/accounts/{id}/sync` | 无 | 只读刷新；不启用账户或交易 |
| `POST /api/accounts/{id}/operations/prepare` | `kind: cancel_orders/flatten, pair, expected_revision, confirm_stop` | 202，`operation_id`；先持久化停用范围，再等待在途执行并生成可确认计划 |
| `GET /api/account-operations/{id}` | 无 | `preparing/awaiting_confirmation/executing/completed/failed/invalidated`、实际计划与结果 |
| `POST /api/account-operations/{id}/execute` | `plan_version` | 202；复核账户后执行冻结计划，失效需重新确认 |
| `GET /api/components/{id}/evaluations` | pair、mode、revision、周期、时间 | 样本分母、状态计数、指标和对照序列 |
| `POST /api/backtest/runs`、`GET /api/backtest/runs/{id}` | 参数或记录 ID | 隔离回测与持久化状态；详情读取不调用模型 |
| `GET /api/backtest/runs`、`/compare` | 分页，或 `left,right` | 历史及条件差异；不同比较条件先说明差异 |
| `GET /api/alerts`、`POST /api/alerts/{id}/read` | 筛选，或已读操作 | 待处理业务事项；已读不改变审批／风险状态 |
| `POST /api/alert-deliveries/{id}/retry` | 无 | 仅重试既有告警投递 |

已有配置、连接检查、资金池及审批写接口继续使用乐观版本检查。清除凭据增加独立 DELETE 入口；普通空凭据保存不表示删除。账户、信号和研究读接口均独立于 `runtime.cycle`。

配置字段统一为：`market_data.timeframe` 保存全局参考行情周期，初始默认 `1h`；`signals.evaluation_interval` 为空时使用该周期；`execution.pairs` 保存规范化可交易范围（`Pair` 已区分现货／合约），自动运行消费此范围；`scheduler.automation_enabled` 为自动运行总开关，现有 scheduler.enabled 和 triggers.enabled 保留为各自规则来源选择。只分析不要求交易范围或账户齐全。

旧 `system.active`／setup_required 不再作为业务总门禁，移除该字段和总激活操作；分析按依赖就绪，交易按配置已应用、资金池／连接启用、平台能力和真实授权判断。原来总停用的配置迁移后将资金池保持停用，自动运行关闭；不因去掉旧开关而恢复执行。

## 数据演进与迁移

当前项目使用各 store 的 SQLAlchemy 元数据建表，没有可接续的 Alembic revision 链。沿用该组织方式，新增 `src/cryptotrader/migrations/workbench.py` 和 `tests/test_workbench_migration.py`，显式执行一次性迁移，不引入第二套迁移框架。

- F1：保留现有加密算法和连接绑定；旧三字段凭据在显式迁移时解密、按对应平台注册模型校验并重封装。迁移不得输出明文。已保存凭据标记、检查时间保留；因格式／版本变化失效的连接检查显示重新检查原因。
- F4：扩展现有多平台 journal 表并迁移记录。旧数据缺失的 pair／配置摘要／曲线只从原记录能证明的信息回填；其余标记历史资料不完整，不拿当前配置或重新推理补造。新的写路径只使用新结构。
- F5：旧 scheduler.pairs 迁至 execution.pairs；删除 system.active 时保留原来的停用效果。首次引入 automation_enabled 统一设 false，规则定义保留；所有已配置账户、凭据和真实授权值保留，但不能隐式开始新运行。
- B2–B4：新增账户快照、订单事实、成交、资金费、同步游标、账户分配有效期、资金池风险状态、人工操作表；数据库唯一键保证重复同步不重复记账。
- R1、R3、R4：新增评估、回测及告警／投递表。旧文件回测按明确源目录一次性导入，缺曲线的记录保留缺项标记。保留原文件作为备份；不维持文件与数据库双写。
- 迁移入口要求显式数据库地址、备份位置；不默认读取生产 URL。用临时数据库复制已脱敏旧格式夹具验证数量、ID、凭据状态、不可逆数据缺项及重复执行无重复记录。
- 运行库迁移必须另获授权、确认目标与备份。不能为让页面启动而自动清库、覆盖账户或删除旧历史。

## 验证命令约定

仓库根目录记为 `/Users/rccpony/Projects/cryptotrader-ai`。2026-08-31 只验证了工具入口：`.venv/bin/python` 为 3.12.9，pytest 9.0.3，FastAPI 0.136.0；可用 Node 为 `/Users/rccpony/.nvm/versions/node/v24.19.0/bin/node`。本轮未运行业务测试。

当前 shell 中 `pnpm --version` 遇到管理器临时文件权限／Node PATH 问题。计划使用已安装的 Node 和包内 CLI，不安装依赖、不改锁文件。执行时若环境已正常，可以使用 `web/package.json` 等价脚本。

后端，在仓库根目录执行；移除进程传入的运行库地址，让测试 conftest 使用临时 SQLite 和测试密钥：

```sh
rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_configuration_catalog.py tests/test_venue_connections_api.py tests/test_runtime_config_secrets.py -q
```

前端，在 `web/` 工作目录执行；子计划只列测试文件，统一套用此命令：

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/components/configuration/configuration-forms.test.tsx
```

完整验证在 U3 执行。pytest 默认 70% 分支覆盖率门槛保留；窄测试使用 `--no-cov` 避免把未运行文件视作新增覆盖失败。测试夹具禁止外部模型、交易所和 Webhook 网络请求；已有测试不满足隔离时先修测试边界，不连接真实平台凑通过。

Playwright 运行地址不得复用 `localhost:5173`，也不得驱动运行数据库。U3 实施时发现原计划的 4173 已由无关 `geo-workbench` 进程占用；经主控安全裁定保留该进程并将本项目隔离前端固定为 4174，API 仍为 8011。`web/playwright.workbench.config.ts` 对两端均使用 `reuseExistingServer: false`，前端使用 `strictPort`，浏览器请求 allowlist 只允许 4174／8011。退出后 8011／4174 均无监听，5173／8000／4173 未操作。

## 验收覆盖索引

| 规格条件 | 实施任务 | 关键证明 |
| --- | --- | --- |
| A1 | F1、F2、B1、U3 | 新后端平台注册后，原前端构建完成配置、非三字段凭据、持仓和退出 |
| A2 | F1、F3、U3 | 自定义组件字段、权重、指标和表格，无前端 ID 分支 |
| A3 | F3、F4、R1 | 当次 Kronos 曲线持久化，跳过／错误区分，历史读取零推理 |
| A4 | F3、F4 | 四智能体辩论留在单一组件下 |
| A5 | F4、F5、U3 | 真实授权测试配置中，分析路径的账户／审批／订单 spy 调用数为零 |
| A6 | F1、F2、B2 | 同平台多连接；环境分组，资金池禁止混资 |
| A7 | F5、B3 | 确认真实范围；按池审批；最新风险复核使旧计划失效 |
| A8 | F5、U1 | 暂停两种自动来源，保留读服务和在途订单语义 |
| A9 | F1、F2 | 非三字段凭据完整往返密文库；刷新显示状态，响应无明文 |
| A10 | F2 | 完整保存一次自动只读检查；版本匹配、参数变更失效 |
| A11 | B3 | 集成权益 100→80、峰值100，执行与页面同源 |
| A12 | F2、F5、U1、U3 | 初次、缺项、失败、重访均有下一步，读页面不被启动状态阻断 |
| A13 | F2、F3、B2、B4、R1–R4、U1、U3 | 六入口、中文无 JSON、键盘／窄屏／主题 |
| A14 | F3、F4 | 修改当前配置不改变历史；分析记录无虚构执行 |
| A15 | B1、B2 | 多账户多品种账本、币种、成本口径、重复同步及未知值 |
| A16 | B2、B4 | 停用仍可读；有资产／挂单时不可移出或删除 |
| A17 | B3、B4、U3 | 停用后独立退出、二次确认、保留保护、禁止恢复开仓 |
| A18 | B3 | 100权益、其他70、上限80、新目标40；待成交与同池并发 |
| A19 | F3、R1 | 冻结参考与期限、正确分母、原预测不变 |
| A20 | R2、R3 | 实际模拟成交、成本、历史曲线、保护触发、重启、可比条件 |
| A21 | R4 | 持久化业务告警和投递，失败可重试，已读不批准 |
| A22 | U3 | 隔离环境网页完成次日复盘与退出，另一池不受误操作 |

## 分批完成定义

- [x] 每个任务先有会因目标行为缺失而失败的测试，再实现和运行对应回归；不以导入错误代替行为断言。
- [x] 每批完成独立审查，记录测试命令、结果、页面状态和未验证部分，不把测试夹具输出说成真实模型／平台结果。
- [x] 任一规格条件无法映射到测试或页面步骤时，补齐任务后再进入最终验收。
- [ ] 运行库迁移、真实模型与官方模拟交易验证尚未获得单独授权，因此按交付边界未执行；这不是本轮隔离验收失败，也不触碰真实资金。

计划自检范围：规格 A1–A22、接口名称、文件是否现有／拟新增、测试隔离、旧入口替换以及交付权限。执行方式可选当前任务内分批实施或经用户明确选择后使用子代理；默认不主动派发。

2026-08-31 计划自检结果：5份计划、16项任务，A1–A22均有任务映射；文档链接无缺失、无行尾空白。人工核对并修正了风险状态新文件与旧文件重名、全局行情周期／交易范围归属、旧停用状态迁移及 SecretStr 加密序列化约束。文案最终扫描76条规则，0命中；技术文档所需表格和步骤保留。本轮只验证计划与工具入口，没有运行业务测试，也没有实施上述业务改动。

## 2026-09-01 U3 隔离执行结果

U3 使用专用临时 SQLite、固定测试密钥、固定行情／模型、fake Webhook 与一致状态的 fake/Paper 账户完成隔离回归。审查发现首次 E2E 曾连接无凭据的 Binance 公共行情 WebSocket，因此首次“未连接外部行情”结论无效；Fix1 已用 build-time 关闭公共行情流并增加 HTTP／WebSocket fail-closed guard，修后全部外联尝试为零。它没有读取运行库、`.env`、`local.toml`、运行密钥或缓存，也没有连接外部模型、Redis、Webhook 或交易平台。首次最终后端全量为 `2305 passed, 1 skipped, 1 warning`，branch coverage `81.71%`；Fix1 定向后端为 `58 passed`。修后前端无 exclude Vitest 为 `55 files / 326 tests passed`；TypeScript、全 ESLint、固定 8011/4174 构建和 same-dist baseline／extension 自动证明均通过。

| 条件 | U3 结果 | 证据归类 |
| --- | --- | --- |
| A1 | 通过 | baseline／extension 共用完全相同的固定行情基础 registry；同一 `dist` 自动发现 extension 仅新增的 `sample_venue` 环境、参数和两项自定义凭据，并由真实 fake/Paper 账本完成买入、Alpha 独立卖出归零、Beta 保留仓位 |
| A2 | 通过 | extension 相对 baseline 唯一另一项新增为 `sample_signal`；同一 `dist` 自动发现窗口字段、权重、指标与表格，无前端组件 ID 分支 |
| A3–A4 | 通过 | 后端／前端全量回归覆盖冻结组件结果、历史零推理与单组件辩论证据 |
| A5 | 通过 | 黄金流只分析前后账户读取与订单写入均为 0 |
| A6–A12 | 通过 | 两连接、两池、全局范围确认、逐池 HITL、凭据写入后自动只读检查、首次／失败／重访均在黄金流或全量合同覆盖 |
| A13 | 自动化通过；手工阻塞 | 六入口、中文无参数 JSON、键盘保存／批准／返回焦点、1440／390、深浅主题均通过；Fix1 用真实 iPhone 13 `isMobile` context 证明 390、drawer dialog、保存栏与无横溢；browser 插件 bootstrap 故障导致要求的手工交互验收未完成 |
| A14–A21 | 通过 | 各责任任务测试保留在最终全量；U3 额外直接显示分析结果与后续冻结评估，不调用外部能力 |
| A22 | 通过 | 首次收益标题证据已降级；Fix1 逐账户核对 `/income` 金额／币种／methodology 与 Paper fill fee／realized，并证明页面规范化显示；网页继续完成评估、Alpha 退出与 Beta 保留 |

Fix1 的可复跑 runner 在运行前、baseline 后、extension 后三次硬断言同一前端不变：`web/src + web/index.html` SHA-256 均为 `bc6742bf7b7d68a6bfc5266e025b248db3273de8f852829519636e9261140806`，`web/dist` 均为 `f02c3dd05bd91fb110c91ebe6f8e39511ae8a08c794689bb5654f4c21c3cee46`。最终失败账本、修前／修后截图和外联纠正见 `task-16-fix1-report.md`。

## 2026-09-01 整项最终审查与验收

U3 后继续进行了跨任务整项审查。审查要求生产启动不得顺手建表，后端与前端不得用任意对象掩盖合同差异，也不得保留旧页面树和旧数据模型作为兼容层。修复后，Workbench schema 只由 `migrations/workbench.py` 的显式入口创建；启动 preflight 只读检查完整 schema，缺失或部分迁移时进入 degraded，健康接口可读，受保护业务接口返回 503。回测、Scheduler 四类规则、Trigger 行情快照、API 错误详情和分析 SSE 均使用同形严格 DTO/Zod；旧 Chat、CycleJournal、旧 portfolio/risk API 与无用展示类型已物理删除。

最终独立限定复审结果为 0 Critical、0 Important、0 Minor。根代理按最终源码重新运行完整验证：后端 `2308 passed, 1 skipped, 1 warning`，总覆盖率 `81.93%`；前端 `58 files / 332 tests passed`；Ruff、433 个 Python 文件格式检查、TypeScript、全 ESLint、`git diff --check` 与 Vite 2049 模块构建全部通过。

同一当前 `dist` 的自动 E2E 先运行基础注册表，再只增加 `sample_signal`／`sample_venue` 的扩展注册表：baseline `1/1`、extension `3/3` 通过，包含 390px 真实移动上下文及首次配置到次日复盘、逐池审批、单池退出且另一池保留的黄金流。runner 返回 `sameFrontendBuild=true`，源码哈希 `104e362c852654feb3a4be3770479050e75e72658e798db3ae2c59a712eaea3c`，dist 哈希 `899eb89cfa06ab5e410b27725f9f06b0a8074cb10b2d6508b84ae2be51646898`。HTTP 与 WebSocket 均 fail-closed，本轮没有外联尝试；首次 U3 错连 Binance 公共行情 WebSocket 的事实与修前截图继续保留。

边界不变：本轮未读取或迁移运行数据库，未读取 `.env`／`local.toml`／运行密钥，未连接真实 Kronos／LLM、官方 Demo／Paper、Redis、Webhook 或 Docker。browser 插件本机缓存入口故障，因此没有把 Playwright 冒充手工浏览器验收。隔离验证服务 `8011/4174` 已退出，`main` HEAD 仍为 `cfe5ac40759b814c7d181275ba7b9abafc3b4079`，Git 索引和锁文件无变化，未 commit 或 push。
