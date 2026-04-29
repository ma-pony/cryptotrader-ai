# 需求文档

## 项目描述（输入）

完善 Dashboard 功能，要求可以展示实盘记录和回测记录，每一次触发的时间点，每一个 Agent 输入的数据，Agent 的思考过程，Agent 的决策结果，整个的决策流程，每一个节点等等，需要做到全流程可观测。

## 需求

### 需求 1：实盘决策历史全流程展示

**目标：** 作为交易员，我希望能在 Dashboard 中查看每一次实盘交易决策的完整执行流程，以便我能审查 AI 的决策过程和最终结果。

#### 验收标准

1. The Dashboard shall 展示所有历史实盘决策记录列表，按触发时间倒序排列，支持按交易对筛选。
2. When 用户选择某条实盘决策记录, the Dashboard shall 展示该次决策的触发时间点（精确到秒的 UTC 时间戳）、交易对及当时市场价格。
3. The Dashboard shall 在决策详情页中展示完整的节点执行流水线：数据采集 → 语言强化 → Agent 并行分析 → 辩论门控 → （可选）辩论轮次 → 裁决 → 风控 → 执行，每个节点显示名称与耗时（毫秒）。
4. When 辩论被门控跳过, the Dashboard shall 在节点流水线中标注"辩论已跳过"并显示跳过原因（共识/混乱）。
5. The Dashboard shall 展示每条实盘记录对应的 `trace_id`，用于与外部日志系统关联追踪。

---

### 需求 2：Agent 输入数据与分析过程展示

**目标：** 作为交易员，我希望能查看每个 Agent 在本次决策中接收到的输入数据和分析结论，以便我能评估 AI 判断的数据依据是否充分。

#### 验收标准

1. When 用户查看某次决策详情, the Dashboard shall 展示四个 Agent（技术/链上/新闻/宏观）各自的分析结论，包括方向（bullish/bearish/neutral）、置信度评分及数据充分性评级。
2. The Dashboard shall 对每个 Agent 的完整推理过程（`reasoning` 字段）提供可展开/折叠查看功能。
3. The Dashboard shall 展示 Agent 分析时使用的市场快照摘要，包括当时的价格、波动率及市场状态标签（regime tags）。
4. Where 历史案例经验被注入上下文, the Dashboard shall 显示本次决策注入了哪些经验规则（成功模式/禁止区域/战略洞察）。
5. If Agent 数据不足（`data_sufficiency` 为 low）, the Dashboard shall 以视觉标识（警告图标或颜色）突出显示该 Agent 的数据充分性状态。

---

### 需求 3：辩论过程详细展示

**目标：** 作为研究员，我希望能查看多 Agent 辩论的每一轮交锋内容，以便我能理解最终裁决是如何在辩论中形成的。

#### 验收标准

1. The Dashboard shall 展示本次决策发生了几轮辩论（`debate_rounds` 字段）以及整体分歧度（`divergence`）。
2. When 辩论轮次大于零, the Dashboard shall 展示每轮辩论中各参与方的挑战论点（`challenges` 字段），按轮次和发言方分组展示。
3. The Dashboard shall 计算并展示各 Agent 评分的共识强度（均值与标准差），以可视化方式呈现分歧收敛过程。
4. If 辩论被门控跳过, the Dashboard shall 显示触发跳过的共识强度阈值与实际值对比。
5. The Dashboard shall 将辩论结果（辩论型裁决 vs 加权降级裁决）在裁决区域中加以标注区分。

---

### 需求 4：裁决与风控结果展示

**目标：** 作为风控负责人，我希望能查看每次决策的裁决逻辑和风控门控结果，以便我能审计系统是否正确执行了风控规则。

#### 验收标准

1. The Dashboard shall 展示裁决结论（action、confidence、position_scale、thesis、reasoning），区分 AI 裁决与加权降级裁决两种来源。
2. When 风控门通过, the Dashboard shall 以绿色标识展示通过状态及参与检查的风控项列表。
3. When 风控门拒绝, the Dashboard shall 显示拒绝原因（`rejected_by` 和 `reason` 字段），以红色警告标识展示。
4. The Dashboard shall 展示本次决策后的实际执行动作（买入/卖出/平仓/持仓），以及止损触发状态。
5. The Dashboard shall 展示执行后的仓位变化（执行前 → 执行后）和权益变化。

---

### 需求 5：回测记录全流程回放

**目标：** 作为策略研究员，我希望能在 Dashboard 中查看回测运行的全部决策序列及其详细过程，以便我能逐条分析策略在历史行情中的表现。

#### 验收标准

1. The Dashboard shall 在回测结果页面展示汇总指标（总收益率、夏普比率、最大回撤、胜率、交易次数）以及权益曲线图表。
2. When 回测结果包含 `decisions` 序列, the Dashboard shall 以时间轴形式展示每个决策点的价格、仓位、动作、风控状态及置信度。
3. When 用户选中时间轴上的某个决策点, the Dashboard shall 以与实盘相同的格式展示该回测决策的完整 Agent 分析、辩论过程与裁决详情。
4. The Dashboard shall 展示回测过程中节点执行追踪（`node_trace`），包括每个节点名称、耗时和输出摘要。
5. If 回测决策列表为空（纯规则模式）, the Dashboard shall 降级展示权益曲线和交易列表，不显示 Agent 分析区域。
6. The Dashboard shall 支持将回测会话结果持久化展示，允许用户在多次回测之间切换对比。

---

### 需求 6：实时运行状态监控

**目标：** 作为运维人员，我希望能在 Dashboard 中实时查看系统当前的运行状态，以便我能快速发现和响应系统异常。

#### 验收标准

1. The Dashboard shall 每 10 秒自动刷新组合概览数据，展示当前总权益、现金余额、日内盈亏和当前回撤。
2. The Dashboard shall 展示当前风控状态：每小时/每日交易次数、电路熔断器状态（活跃/非活跃）及关键风控阈值参数。
3. When 电路熔断器处于活跃状态, the Dashboard shall 以红色警告突出显示，并提供手动重置按钮。
4. If Redis 不可连接, the Dashboard shall 显示"风控状态不可用"警告，而非静默隐藏该模块。
5. The Dashboard shall 展示当前持仓列表，包括交易对、方向（多/空）、数量、均价和当前市值。
6. While 调度器（scheduler）正在运行, the Dashboard shall 展示下一次计划触发的交易对和预计时间。

---

### 需求 7：Prometheus 指标与可观测性集成

**目标：** 作为 SRE 工程师，我希望 Dashboard 能与已有的 Prometheus 指标体系对接，以便我能在统一的可观测性平台中查看系统性能数据。

#### 验收标准

1. The Dashboard shall 展示来自 `MetricsCollector` 的关键指标：LLM 调用总数、辩论跳过次数、裁决分布、风控拒绝次数、交易执行次数及流水线平均耗时。
2. The Dashboard shall 提供指向 `/metrics` Prometheus 端点的跳转链接，供外部监控系统直接抓取。
3. When OpenTelemetry 集成已激活（`OTLP_ENDPOINT` 非空）, the Dashboard shall 在决策详情中显示对应的 OTel trace 链接。
4. The Dashboard shall 以图表形式展示流水线耗时（`ct_pipeline_duration_ms`）和执行延迟（`ct_execution_latency_ms`）的历史趋势。
5. If `/metrics` 端点不可访问, the Dashboard shall 显示"指标端点不可用"提示，不影响其他页面功能。

---

### 需求 8：Dashboard 导航与用户体验

**目标：** 作为系统用户，我希望 Dashboard 具备清晰的页面导航和响应式布局，以便我能高效地在不同功能区域之间切换。

#### 验收标准

1. The Dashboard shall 提供统一侧边栏导航，包含：概览（Overview）、实盘决策（Live Decisions）、回测（Backtest）、风控状态（Risk Status）、指标（Metrics）五个主页面。
2. The Dashboard shall 通过 URL query 参数（`?page=...`）保存当前页面状态，支持浏览器前进/后退和链接分享。
3. When 数据库连接失败, the Dashboard shall 显示明确的错误提示并停止页面渲染，而非展示空白或崩溃。
4. The Dashboard shall 对决策详情中的长文本字段（reasoning、thesis、挑战论点）提供截断展示和"展开查看全文"功能。
5. The Dashboard shall 在宽屏布局下以多列网格展示 Agent 分析卡片，在窄屏布局下自动降级为单列展示。
6. Where 决策记录支持分页, the Dashboard shall 提供分页控件，默认每页展示 20 条记录，支持用户调整。
