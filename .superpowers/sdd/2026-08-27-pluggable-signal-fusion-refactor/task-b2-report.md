# Task B2 报告：实盘与 Paper 保护单安全

日期：2026-08-28

## 交付范围

- 基线提交：`0edfbc6a91b003f050fc835ba564092c305d1550`
- 交付提交：本报告所在的单一 B2 提交；最终提交哈希在提交完成后的任务交接消息中记录。Git 提交无法在自身内容中可靠自包含最终哈希。
- 未调用真实交易所；LiveExchange 测试全部使用 ccxt fake。
- 未修改 BacktestExecutor 的保护单实现或触发顺序。

## 实现结果

1. `ExchangeAdapter.supports_protection_orders()` 显式声明保护能力：Paper 和 OKX 返回支持，当前 Binance 路径返回不支持。
2. `ExecutionService` 在首笔订单前计算最终有符号仓位；非空仓必须具备受支持的交易所保护能力，以及有限、正数、围绕当前价方向正确的止损和止盈。
3. 执行前读取旧 OCO，但不取消；目标差值全部成交后先安装新 OCO，确认成功后才取消旧 OCO。成功空仓只取消旧保护，不创建替换。
4. 目标差值中途失败、替换 OCO 创建失败、旧 OCO 取消失败都会按已成交意图的逆序提交补偿订单。旧 OCO 取消失败时先删除新 OCO，再恢复原始仓位。
5. 失败结果通过 `retained_algo_ids` 报告保留的旧保护 ID；该字段进入 `TradingCycle` 事件、Journal 和 Decisions API。
6. OKX 取消接口的顶层 `code` 和单腿 `sCode` 业务失败都会向上抛出，避免把取消失败误判为成功。
7. Paper OCO 在新收集的最新闭合 K 线上、组件运行前触发；多空止损/止盈均支持，同一根 K 线同时命中时止损优先。触发会更新模拟仓位、已实现盈亏和 USDT 余额，并把 OCO 标记为一次性终态。
8. Paper 保护触发后刷新执行上下文；即使随后组件失败，组件看到的仍是刷新后的仓位，保护成交不会回滚。

## TDD 证据

### RED

- 首轮命令：`uv run pytest tests/test_execution_service.py tests/test_paper_exchange_protection.py tests/test_trading_cycle.py --no-cov -q`
- 结果：`15 failed, 21 passed`。
- 失败准确覆盖：不支持能力仍下单、无效退出价仍下单、旧 OCO 先取消、无补偿、Paper 无触发入口、组件先于保护运行。
- 部分成交补充 RED：反向计划第二腿失败后仅出现已成交平仓腿和失败开仓腿，未出现恢复原仓位的补偿单，`1 failed`。
- OKX 取消补充 RED：顶层业务错误与单腿业务错误均未抛出，`2 failed`。

### GREEN

- 聚焦安全套件：`38 passed`。
- 邻接执行、OCO、OrderManager、Paper、TradingCycle、Decisions API、Backtest 和 live/backtest parity：`96 passed`。
- OKX OCO 聚焦套件：`18 passed`。
- Protocol、HITL fresh ticker、执行与 OCO 回归：`33 passed, 1 warning`。
- 全量后端：`1809 passed, 36 warnings`，分支覆盖率 `74.17%`，通过 `70%` 门槛。
- 质量门禁：`ruff check .` 通过；`ruff format --check .` 报告 `357 files already formatted`；`git diff --check` 通过。

## 变更文件

- `src/cryptotrader/execution/exchange.py`
- `src/cryptotrader/execution/service.py`
- `src/cryptotrader/execution/simulator.py`
- `src/cryptotrader/trading_cycle.py`
- `tests/test_exchange_algo_oco.py`
- `tests/test_execution_service.py`
- `tests/test_paper_exchange_protection.py`
- `tests/test_trading_cycle.py`
- `tests/test_api_decisions_detail.py`
- `tests/test_bootstrap.py`

## 假设

- 同一交易对正常状态下只有一个待处理保护 OCO；执行层仍会捕获并依次处理交易所返回的全部待处理 OCO。
- 交易所成交回报中的 `filled/closed` 是补偿和保护替换继续推进的唯一依据；未成交或部分成交不会被当作成功。
- Paper 使用上下文快照中时间最新的闭合 K 线；跨周期的真实细粒度成交路径不可知时，沿用回测的保守止损优先规则。

## 剩余关注点

- 若交易所网络错误使取消结果未知，`retained_algo_ids` 表示最后可确认的旧保护状态；下一轮仍需依赖交易所查询进行事实对账。
- 若补偿订单本身被交易所拒绝，系统会显式返回 `position compensation failed` 并保留全部订单审计，不会宣称原仓位已恢复；此时需要人工处理真实仓位。
- Paper 只模拟闭合 K 线触发，不模拟 K 线内部价格路径；同 K 线双触发固定采用止损优先。

## 独立审查修正（2026-08-28）

- 修正基线：`c7b6b429d960707ef10bffa31694ca48bf7e88f5`；修正提交哈希在任务交接消息中记录。
- `OrderManager.place` 的普通 `Exception` 现在会转成明确失败结果；若此前已有可确认成交，会先逆序补偿。补偿下单自身的普通异常被收敛为 `position compensation failed`，订单和保留保护单 ID 不会因异常逃逸而丢失。
- 每笔执行回报新增 `filled_amount`。`PARTIALLY_FILLED` 及其他非 `FILLED` 状态只要原始回报包含正数实际成交量，当前腿也按实际成交量进入逆序补偿，不再按完整 intent 假定成交。
- 旧 OCO 逐笔取消时，`retained_algo_ids` 只保留从首个未确认取消项开始的 ID；若旧 OCO 取消失败且新 OCO 清理也失败，则结构化报告旧、新两组仍活动或状态未知的 ID。
- Paper OCO 保存创建时的 `bar_watermark`，只处理创建后新闭合的 K 线。跨 timeframe 以 `open time + timeframe interval` 计算闭合时点，并选择最新的新闭合 bar；同 bar 双命中继续止损优先。
- Paper 保护触发返回结构化的 `algo_id`、`trigger_reason`、`trigger_price`、`order_id`。TradingCycle 在刷新前发布触发事件，并把事实带入 Journal/API 的 `execution_result`；刷新失败直接写入单一 `execution_failed` 终态，后续组件失败也保留触发事实。
- 前端 `CycleExecutionResultSchema` 新增 `retained_algo_ids` 和结构化保护触发；订单 schema 同步 `filled_amount`，避免审计字段在解析边界被剥离。

### 修正 TDD 证据

#### RED

- 后端：`uv run pytest tests/test_execution_service.py tests/test_paper_exchange_protection.py tests/test_trading_cycle.py --no-cov -q` → `15 failed, 32 passed`。失败逐项覆盖 direct place/compensation exception、当前腿实际 partial fill、旧/新 OCO 未决 ID、多旧 OCO 部分取消、Paper refresh 终态审计、创建 watermark 与多 timeframe 闭合时点。
- 前端：`pnpm test -- tests/unit/schema-contract.test.ts` → `1 failed, 99 passed`，失败为 `retained_algo_ids` 被 Zod schema 剥离。

#### GREEN

- 后端聚焦：同一命令 → `47 passed`。
- 后端邻接：Decisions API、Bootstrap、Backtest CLI、Cycle Event/Journal、Live OCO、HITL、live/backtest parity、Paper concurrency、Scheduler、E2E → `95 passed, 1 warning`。
- 后端全量：`uv run pytest -q` → `1818 passed, 36 warnings`，覆盖率 `74.31%`，通过 `70%` 门槛。
- 前端测试：`pnpm test -- tests/unit/schema-contract.test.ts` 实际执行完整 Vitest 单测集 → `16 files passed, 100 tests passed`。
- 前端类型：`pnpm typecheck` → `TypeScript: No errors found`。
- 质量门禁：`ruff check .`、`ruff format --check .`（`357 files already formatted`）、`git diff --check` 全部通过。

### 修正后的假设与剩余关注点

- 实际成交量以交易所/OrderManager 原始回报的 `filled` 为准；若下单调用在返回回报前异常，只能补偿此前已确认的成交，不能虚构当前请求的成交事实。
- 普通取消异常表示当前及后续 ID 的取消状态未确认，因此均进入 `retained_algo_ids`；`CancelledError`、`KeyboardInterrupt` 等基础异常继续向上层传播，不被安全结果吞掉。
- Paper watermark 是简单的每单闭合时点水位，不是撮合或 reconciliation 子系统；它避免历史 bar 回放，但仍不模拟 bar 内价格路径。
- 若补偿本身被拒绝、部分成交或抛出异常，结果会明确报告 `position compensation failed` 并保留已获得的订单审计；真实仓位仍需人工/后续交易所事实对账。

## 最终独立审查修正（2026-08-28）

- 修正基线：`b306981b26f2e5df34de2133633f2fcd211a1978`；本轮最终提交哈希在任务交接消息中记录，避免提交自引用。
- LiveExchange 现在在 swap 适配器边界把 ccxt 合约张数回报统一为基础币单位：`filled_contracts` 保留原始 `filled`，`filled` 改为 `filled_contracts * contractSize`，原始 `info` 不变。ExecutionService 因此只消费统一的基础币 `filled`。
- 下单、持仓读取、OCO 创建共用同一简单 `contractSize` 解析合同：缺失、非数字或非正数继续回退为 `1.0`；Spot 不归一化且保持原始基础币回报。
- `contractSize=0.01` 的完整 0.1 BTC 成交只发送 10 张并成功安装保护；4 张 partial/cancelled fill 归一化为 0.04 BTC，逆序补偿只发送 4 张，不再放大为 400 张。
- 所有 `replace_journal=True` 的最终写入会在当前执行结果缺少 trigger 时，从同周期旧 Journal 合并结构化 `protection_trigger`。该最小合并覆盖 HITL 批准完成、拒绝、批准后 refresh 失败、批准后取消，无需修改审批表或新增工作流状态。
- Decisions API 与前端 schema contract 测试明确断言最终 `protection_trigger` 仍可见。

### 最终修正 TDD 证据

#### RED

- `uv run pytest tests/test_exchange_algo_oco.py tests/test_trading_cycle.py --no-cov -q` → `10 failed, 40 passed`。
- OKX 完整成交被误判失败；partial/cancelled 的 4 张实际成交产生 400 张补偿；无效/缺失 contract size 缺少原始合约张数审计。HITL approve 把 trigger 改为 `None`，reject/refresh failure/cancellation 把整个 `execution_result` 改为 `None`。

#### GREEN

- 初始聚焦：同一命令 → `50 passed`。
- B2 聚焦：ExecutionService、Paper protection、TradingCycle、Decisions detail、OKX OCO、OrderManager → `89 passed`。
- 邻接：perp close、live pair、exchange protocol、HITL API/gate、Cycle Journal、Decisions list、Bootstrap、live/backtest parity → `53 passed, 1 warning`。
- 后端全量：`uv run pytest -q` → `1829 passed, 36 warnings`，覆盖率 `74.51%`。
- 前端：Vitest `16 files passed, 100 tests passed`；`pnpm typecheck` → `TypeScript: No errors found`。
- 质量门禁：`ruff check .`、`ruff format --check .`（`357 files already formatted`）、`git diff --check` 全部通过。

### 最终修正假设与剩余关注点

- ccxt swap 的订单 `amount` 与 `filled` 均为合约张数；`info` 和新增 `filled_contracts` 提供原始交易所审计，内部 `filled` 坚持基础币单位。
- 对缺失或无效 `contractSize` 继续使用既有 1:1 回退，避免引入新的市场元数据兼容层；交易所元数据本身错误仍属于外部事实风险。
- HITL 合并只恢复不可逆的保护触发事实；当前终态已有 trigger 时以当前值为准，不合并或重写其他旧 execution 字段。
