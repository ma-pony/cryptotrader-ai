# Review Guide: 引入 Pair 值对象统一交易对类型语义

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-04-30

---

## What This Spec Does

把项目里所有"交易对"概念从字符串升级为 `Pair` 值对象。当前项目用 `"BTC/USDT"` 这种字符串到处传，但 OKX 永续合约账户在 ccxt 里返回的是 `"BTC/USDT:USDT"`（带 `:SETTLE` 后缀），两种字符串不匹配 → 平仓静默失败、加仓可能叠满仓。本 spec 新增 `Pair` 类把 ccxt 的 unified symbol 作为唯一真实来源，从配置、state、journal、API、前端 5 个层面收敛。

**In scope:** Pair 值对象（[pair.py](contracts/pair_api.md)）、`[scheduler].pairs` TOML schema 升级、LangGraph state schema bump、DB 加 `market_type` 列、API 加 `pair_display`/`market_type` 字段、前端 `<PortfolioPositions>` + `<DecisionDetail>` 加徽章。

**Out of scope:** 其他前端视图（market view / chat / TradingView widget）继续用字符串、跨 exchange 同 base/quote 歧义、`MarketData.pair` `OnchainData.pair` 等数据领域 pair 字段（[Out of Scope 节](spec.md#out-of-scope)）。

## Bigger Picture

这个 spec 是 2026-04-30 排查 "close on flat → 0 真实交易" 事件的治本方案。当时已合并一个 band-aid（commit `d05a0bf`，`LiveExchange._canonical_pair`）解决症状；本 spec 的最后一个 task（[T039](tasks.md)）才会撤掉那个 band-aid。

更长远地看，项目当前是单 exchange (OKX sandbox) 单账户类型，但 spec 留了"多 exchange / spot+perp 混账户"的扩展位（D4 选了 per-pair market_type 而不是 per-exchange）。如果未来加 Binance、加 spot 现货账户、加期权，`Pair` 类的设计能覆盖（FR-007 `market_type` 已支持 `"future"`，`"option"` 显式 NotImplementedError 留口子）。

ccxt 4.x 的 unified symbol 约定是这次设计的基石（[research.md R1](research.md)）。如果 ccxt 5.x 改了 symbol 格式，所有依赖 `Pair.canonical()` ≡ ccxt symbol 的假设需要回顾。

## Spec Review Guide (30 minutes)

> 30 分钟里聚焦这 4 个最需要人来判断的部分。

### 理解整体方向 (8 min)

读 [spec.md Design Decisions 表 (D1-D7)](spec.md#design-decisions-2026-04-30-brainstorm) + [Phased Delivery 表](spec.md#phased-delivery)。一边读一边问：

- D1 "Pair 把 ccxt symbol 作为唯一真实来源"和 D3 "Order.pair 保持 str = pair.canonical()"组合是否清晰？两者意味着 Order 序列化用 ccxt 形式，整个项目的"trading 语言"由 ccxt 定义 — 这跟项目当前"自有领域模型"的风格是一致还是冲突？
- D7 把 Phase 3 拆成 3a/3b/3c 是为了降风险，但 3 个 phase 同 PR 走完才不会留 "半升级状态"。tasks.md [T011-T023](tasks.md) 都在 US1 里 — 这个粒度合适，还是应该分 PR？
- spec.md 列了 4 个 user story 全是 P1 + P2，没有 P3。其中 [US2 "消除散点"](spec.md#user-story-2--内部代码不再有字符串拼接--拆分-pair-的散点逻辑-priority-p1) 跟 US1 是相辅相成的（治本必须靠 US2 收口），还是可以独立交付？

### 需要你重点看的关键决策 (12 min)

**D2 — state.metadata.pair 一刀切升级** ([spec.md#design-decisions](spec.md#design-decisions-2026-04-30-brainstorm))

LangGraph state schema bump 要求停 scheduler、清空 in-flight checkpoint、单 PR 切换。tasks.md [T018-T023](tasks.md) 是这一步。
- Question: research.md [R2](research.md) 提到"项目当前 LangGraph 是否启用 persistent checkpoint 待验证"。如果未启用（state 只活在内存），停服务那一步能省掉。这个不确定性会否影响 Phase 3c 的 deploy plan？

**D4 — `[scheduler].pairs` 升级为对象数组** ([contracts/scheduler_pairs_config.md](contracts/scheduler_pairs_config.md))

旧 `pairs = ["BTC/USDT"]` 视作全 spot 兼容；新形式 `[[scheduler.pairs]]` 加 `market` + `settle`。
- Question: 这个混合策略 5 年后会不会让运维只看 config 一眼分不清是 spot 还是 perp？要不要加个 deprecation warning 推用户全量迁移到新形式？

**D5 — DB 只加列不改值** ([spec.md FR-400~405](spec.md#storage-fr-400--419))

`market_type` 列默认 `'spot'`；存量 row 不动；新 commit 双写。
- Question: 如果将来要按 `market_type` 做查询（比如 "本月所有 perp commit 的 PnL"），spot 历史会污染结果（看起来有很多 spot commit 但其实是数据缺失）。是否需要在 README 或 ops doc 里明示"`market_type='spot'` 在 2026-04-30 之前的数据等同于'未知'"？

**D6 — 前端最小切片** ([spec.md FR-500~506](spec.md#api--frontend-fr-500--519))

只在 `<PortfolioPositions>` 和 `<DecisionDetail>` 加徽章，其他 30+ 处用 `pair_display` 字符串兜底。
- Question: chat agent 页和 TradingView widget 里 pair 现在还可能渲染成 ccxt 形式（带 `:USDT`），用户体验 OK 吗？还是应该把 `pair_display` 用到所有渲染位？

### Areas where I'm less certain (5 min)

诚实记录我对 spec 的几个不确定点：

- [spec.md FR-204](spec.md#state--nodes-fr-200--219) "checkpoint 反序列化时通过 `Pair.parse(saved_str)` 重建"：写得是"WARNING log"，但如果 checkpoint 里的 pair 解析失败（比如以前写错了），fallback 行为不明确。是 raise 还是 default 到 spot？data-model.md [Validation Rules](data-model.md#validation-rules) 说"fallback 为 spot + warning"，但 spec 里描述跟 data-model 略有歧义。
- [tasks.md T019](tasks.md) 把 4 个 nodes 文件的 Pair 化合并成一个 task。每个 node 的修改其实是独立的，合并成一个 task 可能导致 PR 太大、code review 不方便。要不要拆？
- [contracts/pair_api.md](contracts/pair_api.md) 的 `Pair.from_ccxt(exchange, symbol)` 签名里的 `exchange: Any`。能否把 `exchange` 限定为某个 protocol（避免误传非 ccxt 对象）？还是 `Any` 就行，因为 ccxt 的类型系统本身也比较松？
- [research.md R5](research.md) 性能测得 `~0.6μs`，但用的是 M1 Mac。在 Linux server (项目部署目标) 上是否有差异需要单独跑一次基准？还是 Linux 性能通常更好，可以忽略？

### Risks and open questions (5 min)

把风险变成问题：

- 如果 OKX sandbox 升级了 ccxt market metadata 结构（比如 `m['swap']` 这个 flag 被换成了 `m['type'] == 'swap'`），[Pair.from_ccxt](contracts/pair_api.md) 会不会全军覆没？要不要给 `from_ccxt` 加个集成测试覆盖真实 ccxt 4.x 的 OKX market 字典 fixture？
- [spec.md User Story 1 acceptance scenario 2](spec.md#user-story-1--永续合约用户能正确平仓-priority-p1) "long scale=0.5 已有 0.02 BTC perp → 下单 amount = target - 0.02"。如果当前持仓是 short（amount < 0），target - existing 的符号怎么处理？spec 里没显式回答。是不是要加一个 [User Story 1 acceptance scenario 4]：covering short position adjustment？
- [tasks.md T028 alembic migration](tasks.md) 没指定 `XXXX` 的 migration 编号。运行时按 alembic 自动序号，但本 spec 走 ship pipeline 后多个 PR 并行可能撞号。要不要在 task 里明确"运行 `alembic revision --autogenerate -m 'add market_type column'` 然后人工编辑生成的文件"，减少冲突？
- 如果 [Phase 3c (T018-T023)](tasks.md) 部署到生产时 scheduler 正在跑，并发 cycle 写入旧 schema 的 checkpoint，会触发 `Pair.parse` 错误。tasks 里有提到 T022 加 compat shim，但部署 runbook 没要求 "先停 scheduler"。是否需要在 [quickstart.md](quickstart.md) 加一个 deployment checklist？

---
*Full context in linked [spec](spec.md) and [plan](plan.md).*
