# 实施计划

- [x] 1. 扩展 DecisionCommit 数据模型，新增五个可观测性字段
- [x] 1.1 新增 ConsensusMetrics 和 NodeTraceEntry 值对象到业务模型层
  - 在现有模型文件中定义 `ConsensusMetrics` dataclass，包含 strength、mean_score、dispersion、skip_threshold、confusion_threshold 五个字段
  - 定义 `NodeTraceEntry` dataclass，包含 node、duration_ms、summary 字段
  - 所有新字段设置合理默认值，确保向后兼容
  - _Requirements: 1.3, 1.4, 3.3, 3.4_

- [x] 1.2 在 DecisionCommit 主模型上追加五个可观测性字段
  - 追加 `consensus_metrics: ConsensusMetrics | None = None`
  - 追加 `verdict_source: Literal["ai", "weighted", "hold_all_mock"] = "ai"`
  - 追加 `experience_memory: dict[str, Any] = field(default_factory=dict)`
  - 追加 `node_trace: list[NodeTraceEntry] = field(default_factory=list)`
  - 追加 `debate_skip_reason: str = ""`
  - 确保所有字段均有默认值，旧代码构造 DecisionCommit 时无需改动
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 1.3 扩展数据库存储层，支持新字段序列化与反序列化
  - 在 `DecisionCommitRow` ORM 模型中新增五列：consensus_metrics（JSONB）、verdict_source（VARCHAR(20) DEFAULT 'ai'）、experience_memory（JSONB）、node_trace（JSONB DEFAULT '[]'）、debate_skip_reason（VARCHAR(500) DEFAULT ''）
  - 扩展 `_dc_to_row_dict()` 将新字段序列化为 JSON
  - 扩展 `_row_to_dc()` 实现 None 安全的反序列化，旧记录（NULL 值）不报错
  - 扩展内存 fallback 的 `_serialize()` / `_deserialize()` 路径，保持与 DB 路径行为一致
  - `_ensure_tables()` 使用 `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ADD COLUMN IF NOT EXISTS` 实现零停机迁移
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 2. 扩展节点层，将可观测性数据写入 ArenaState
- [x] 2.1 扩展 debate_gate 节点，将共识指标写入 state
  - 在 `debate_gate` 节点现有共识强度计算之后，将完整的 ConsensusMetrics 字典和 debate_skip_reason 写入 `state["data"]`
  - 无论辩论是否跳过，均写入 consensus_metrics（反映跳过决策的依据）
  - 写入 `debate_skip_reason`：共识跳过写 "consensus"、混乱跳过写 "confusion"、未跳过写空字符串
  - 从 config 读取 skip_threshold 和 confusion_threshold 并包含在指标中
  - _Requirements: 3.3, 3.4_

- [x] 2.2 扩展 make_verdict 节点，在裁决结果中注入 verdict_source 字段
  - 在 `make_verdict_llm` 分支返回值中追加 `verdict_source: "ai"`
  - 在 `make_verdict_weighted` 分支返回值中追加 `verdict_source: "weighted"`
  - 在全 mock hold 分支返回值中追加 `verdict_source: "hold_all_mock"`
  - 确保三个分支均覆盖，不遗漏任何执行路径
  - _Requirements: 3.5, 4.1_

- [x] 2.3 扩展 journal_trade 和 journal_rejection 节点，持久化五个新字段
  - 在 `build_commit()` 调用中追加五个参数，从 `state["data"]` 安全读取（使用 `.get()` 防止 KeyError）
  - `node_trace` 从 `state["data"].get("node_trace", [])` 读取（由 `run_graph_traced()` 注入）
  - `experience_memory` 从 `state["data"].get("experience_memory", {})` 读取（verbal_reinforcement 节点写入）
  - 确保两个 journal 节点（trade 路径和 rejection 路径）均同步扩展
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 3. (P) 新增 FastAPI `/scheduler/status` 端点
  - 新建 `src/api/routes/scheduler.py`，注册 `APIRouter(prefix="/scheduler")`
  - 定义 `SchedulerJobStatus` 和 `SchedulerStatusResponse` Pydantic 模型
  - 实现 `GET /scheduler/status`：Scheduler 运行时返回 jobs、cycle_count、interval_minutes、pairs；未启动时返回 `running=false` 和空 jobs，不返回 503
  - 在 FastAPI 主应用中 include 新路由
  - 独立于任务 4，不共享文件，可并行开发
  - _Requirements: 6.6_

- [x] 4. (P) 新增 FastAPI `/metrics/summary` 端点
  - 在现有 `src/api/routes/metrics.py` 中追加 `GET /metrics/summary` 路由（保持现有 `GET /metrics` Prometheus 文本格式端点不变）
  - 定义 `MetricsSummaryResponse` Pydantic 模型，包含 llm_calls_total、debate_skipped_total、verdict_distribution、risk_rejected_total、risk_rejected_by_check、trade_executed_total、pipeline_duration_p50_ms、pipeline_duration_p95_ms、execution_latency_p50_ms、execution_latency_p95_ms、snapshot_time 字段
  - 从 `get_metrics_collector()` 读取 prometheus-client 内部计数器当前值
  - 独立于任务 3，不共享文件，可并行开发
  - _Requirements: 7.1, 7.4_

- [x] 5. 重构 Dashboard 基础架构为 pages/ 子模块
  - 创建 `src/dashboard/pages/__init__.py`，建立模块骨架
  - 重构 `app.py`：仅保留路由逻辑、配置加载（`@st.cache_resource`）和侧边栏导航渲染，删除所有内联页面代码
  - 实现五页侧边栏导航，使用 `st.query_params` 读写 `?page=` URL 参数，支持浏览器前进/后退
  - 将 `_run()` 辅助函数从 `app.py` 迁移到 `data_loader.py`，暴露为 `run_async()`
  - 确保配置加载失败时 `st.error() + st.stop()` 阻止渲染
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 6. (P) 实现 Dashboard 数据加载层和共享渲染组件
- [x] 6.1 (P) 实现 DashboardDataLoader 数据加载层
  - 新建 `src/dashboard/data_loader.py`，集中所有 `@st.cache_data(ttl=N)` 装饰函数
  - 实现 `load_portfolio(db_url)`（TTL=10s）、`load_journal(db_url, limit, pair, offset)`（TTL=10s）、`load_commit_detail(db_url, commit_hash)`（TTL=10s）、`load_risk_status(redis_url)`（TTL=10s）
  - 实现 `load_scheduler_status(api_base_url)`（TTL=30s）和 `load_metrics_summary(api_base_url)`（TTL=30s），使用 `httpx.get()` 同步调用，超时 5s，异常时返回 `None`
  - 实现 `list_backtest_sessions()`（TTL=300s）和 `load_backtest_session(session_id)`（TTL=300s）
  - 数据库异常透传，由页面层捕获；HTTP 异常吞噬并返回 None
  - 此子任务不依赖 6.2，可同步开发（不同文件）
  - _Requirements: 1.1, 5.6, 6.1, 6.2, 6.6, 7.1, 7.4_

- [x] 6.2 (P) 实现共享渲染组件库
  - 新建 `src/dashboard/components.py`，实现所有页面复用的渲染函数
  - `render_agent_analysis_grid(analyses, columns)`：data_sufficiency=="low" 时在卡片标题旁显示警告图标；自动按屏宽决定列数（最多 4 列）
  - `render_node_trace_pipeline(node_trace)`：横向流水线，每节点显示名称和耗时，辩论跳过节点标注为灰色虚线框
  - `render_verdict_section(verdict, verdict_source)`：区分 AI 裁决与加权降级裁决的视觉标识
  - `render_risk_gate_section(risk_gate)`：通过时绿色展示检查列表，拒绝时红色展示 rejected_by 和 reason
  - `render_consensus_metrics_chart(consensus_metrics, analyses)`：st.bar_chart 展示各 Agent 评分，st.caption 显示均值/标准差/强度
  - `render_debate_section(debate_rounds, challenges, debate_skip_reason, consensus_metrics)`：辩论跳过时显示阈值与实际值对比
  - `render_experience_memory_section(experience_memory)`：展示注入的成功模式/禁止区域/战略洞察
  - `render_expandable_text(label, text, preview_chars=200)`：前 200 字符直接展示，超出部分折叠在 st.expander 内
  - `render_pagination_controls(total, page_size=20, key)`：返回 (offset, limit) 元组
  - 此子任务不依赖 6.1，可同步开发（不同文件）
  - _Requirements: 1.3, 1.4, 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 8.4, 8.5, 8.6_

- [x] 7. (P) 实现 LiveDecisionsPage，展示实盘决策历史与全流程详情
  - 新建 `src/dashboard/pages/live_decisions.py`
  - 筛选栏：交易对下拉（None = 全部）+ 分页控件，通过 `load_journal(pair, offset, limit)` 拉取数据
  - 决策列表：`st.dataframe(on_select="rerun")` 捕获行点击；版本不支持时降级为 `st.selectbox` 选择 hash
  - 决策详情（点击后条件渲染）：
    - 决策头部：触发时间（UTC 精确到秒）、交易对、市场价格、trace_id；OTLP_ENDPOINT 非空时 trace_id 渲染为 OTel trace 链接
    - 节点执行流水线（复用 `render_node_trace_pipeline`）
    - Agent 分析网格（复用 `render_agent_analysis_grid`）
    - 经验记忆注入（复用 `render_experience_memory_section`）
    - 辩论区域（复用 `render_debate_section`）
    - 裁决区域（复用 `render_verdict_section`）
    - 风控与执行区域（复用 `render_risk_gate_section`）+ 执行动作、止损状态、仓位权益变化
  - 数据库异常最顶层捕获，`st.error() + st.stop()`
  - 可与任务 8、9、10 并行（不同页面文件，依赖任务 5 和 6 完成后启动）
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 7.3, 8.3, 8.4, 8.5, 8.6_

- [x] 8. (P) 实现 BacktestPage，展示回测决策序列与会话持久化对比
  - 新建 `src/dashboard/pages/backtest.py`
  - 顶层标签页：Run New Backtest（参数表单 + Run 按钮）| Load Session（会话列表下拉 `list_sessions()`）
  - Run New 完成后调用 `save_commits() + save_result()` 获取 session_id，自动切换到加载视图
  - 回测汇总指标卡片：总收益率、夏普比率、最大回撤、胜率、交易次数（5 项）
  - 权益曲线：`st.line_chart`
  - 决策时间轴：DataFrame 展示每个决策点的价格、仓位、动作、风控状态、置信度
  - 时间轴点选后展示完整决策详情，复用 components.py 全套组件（与实盘格式一致）
  - 纯规则模式（decisions=[]）时隐藏 Agent 分析区域，仅显示权益曲线和交易列表
  - 可与任务 7、9、10 并行（不同页面文件，依赖任务 5 和 6 完成后启动）
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 9. (P) 实现 OverviewPage 和 RiskStatusPage
- [x] 9.1 (P) 实现 OverviewPage，展示实时组合概览与调度器状态
  - 新建 `src/dashboard/pages/overview.py`
  - 使用 `load_portfolio(db_url)` 展示总权益、现金余额、日内盈亏、当前回撤（TTL=10s）
  - 权益曲线 `st.line_chart`，页面末尾 `st.empty()` 占位块 + `time.sleep(10)` + `st.rerun()` 实现 10s 自动刷新
  - 持仓列表：`st.table()`，方向标识为 Long/Short
  - Scheduler 状态：`load_scheduler_status(api_base_url)` 获取下次触发交易对和时间；返回 None 时显示"调度器状态不可用"，不崩溃
  - _Requirements: 6.1, 6.5, 6.6_

- [x] 9.2 (P) 实现 RiskStatusPage，展示风控状态与熔断器管理
  - 新建 `src/dashboard/pages/risk_status.py`
  - `load_risk_status(redis_url)` 返回 None 时（Redis 不可达）渲染 `st.warning("风控状态不可用 — Redis 未连接")`，提前返回，不调用 `st.stop()`
  - 展示每小时/每日交易次数、电路熔断器状态（活跃/非活跃）及关键风控阈值参数
  - 熔断器 ACTIVE 时整个卡片使用 `st.error()` 容器渲染（红色警告）
  - 重置按钮：调用 `run_async(rsm.reset_circuit_breaker())` 后 `st.rerun()`
  - _Requirements: 6.2, 6.3, 6.4_

- [x] 10. (P) 实现 MetricsPage，展示 Prometheus 指标与延迟趋势
  - 新建 `src/dashboard/pages/metrics.py`
  - `load_metrics_summary(api_base_url)` 返回 None 时渲染 `st.warning("指标端点不可用")` 并 return，不影响其他页面
  - 展示关键指标（LLM 调用总数、辩论跳过次数、裁决分布、风控拒绝次数、交易执行次数）
  - pipeline_duration_p50_ms / p95_ms 和 execution_latency_p50_ms / p95_ms 使用 `st.metric()` 展示
  - 历史趋势图：在 `st.session_state` 中累积多次采样（快照时间戳 + 值），用 `st.line_chart` 展示
  - `/metrics` Prometheus 端点通过 `st.link_button()` 提供跳转链接
  - 可与任务 7、8、9 并行（不同页面文件，依赖任务 5 和 6 完成后启动）
  - _Requirements: 7.1, 7.2, 7.4, 7.5_

- [x] 11. 数据库迁移验证与端到端集成测试
- [x] 11.1 编写数据模型和序列化单元测试
  - 验证 `DecisionCommit` 新字段默认值：所有五个字段在旧代码路径构造时保持合法默认值
  - 验证 `ConsensusMetrics` 和 `NodeTraceEntry` dataclass 约束（类型校验、序列化往返）
  - 测试 `_dc_to_row_dict()` / `_row_to_dc()` 对五个新字段的 JSON 序列化往返，覆盖 None 安全处理
  - 测试内存 fallback 路径 `_deserialize()` 与 DB 路径行为一致
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 11.2 编写节点层单元测试
  - 验证 `debate_gate` 节点返回值包含 `consensus_metrics` 字典和 `debate_skip_reason` 字段，覆盖跳过和不跳过两条路径
  - 验证 `make_verdict` 三个分支（AI/weighted/hold_all_mock）各自写入正确的 `verdict_source` 值
  - 验证 `journal_trade` 和 `journal_rejection` 从 state 正确读取并传递五个新字段给 `build_commit()`
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 11.3 编写 API 端点单元测试
  - mock `Scheduler` 对象，验证 `GET /scheduler/status` 响应结构和字段类型；验证 Scheduler 未启动时返回 `running=false`
  - mock `MetricsCollector` 单例，验证 `GET /metrics/summary` 响应结构和计数器值读取
  - _Requirements: 6.6, 7.1, 7.4_

- [x] 11.4 编写数据库迁移集成测试
  - 准备不含新列的旧版数据库，调用 `_ensure_tables()` 后验证五列均已添加
  - 验证旧记录（新列为 NULL）通过 `_row_to_dc()` 读取时不报错，新字段返回默认值
  - 验证新记录完整写入并能正确读回
  - _Requirements: 1.3, 1.4, 2.3, 2.4, 3.3, 3.5_

- [x] 11.5 编写 DashboardDataLoader 集成测试
  - mock `JournalStore` 和 httpx，验证 `load_scheduler_status()` 和 `load_metrics_summary()` 的 None 降级行为
  - 验证 TTL 缓存键包含所有参数，不同参数组合不共享缓存
  - 验证 HTTP 超时（5s）时返回 None 而非抛出异常
  - _Requirements: 6.6, 7.1, 7.5_

- [x] 11.6 可选：LiveDecisionsPage 渲染验收测试
  - 验证辩论跳过场景下"辩论已跳过"标注和跳过原因正确渲染（需求 1.4 / 3.4）
  - 验证 data_sufficiency="low" 时 Agent 卡片显示警告图标（需求 2.5）
  - 验证裁决类型标注（AI vs 加权降级）区分渲染（需求 3.5 / 4.1）
  - _Requirements: 1.4, 2.5, 3.4, 3.5, 4.1_
