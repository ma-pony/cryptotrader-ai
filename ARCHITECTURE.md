# CryptoTrader AI 架构

运行配置以数据库 `runtime_config` 为唯一来源，凭据保存在加密 vault。`BootstrapSettings` 只读取 `DATABASE_URL` 和 `CONFIG_MASTER_KEY`。`build_runtime()` 装配运行时；未启用时只发现已安装插件元数据，不打开交易连接或构造模型。

## 配置与运行时

`configuration/catalog.py` 从内置工厂和 Python package entry points 读取 `PluginConfiguration`。同一份 Pydantic 参数模型生成字段、默认值、选项和服务端校验。`GET /api/config/catalog` 可在未启用时使用，受实际 API 访问认证约束。

网页八个配置分区共享 draft 和服务端 baseline。保存一个分区不会覆盖其他分区的未保存内容。`PUT /api/config` 使用 `expected_revision` 做 compare-and-swap；凭据单独写入，不放入配置草稿或可读响应。

```text
类型化表单 → 参数/领域校验 → 准备候选运行时
         → 应用屏障内 CAS 保存 desired revision
         → 发布候选 → applied revision
                      ↘ 失败：保留 desired，标记 failed，关闭执行入口
```

准备失败不会提交新 revision；发布失败可能发生在持久化之后。不能把 HTTP 保存错误解释成旧运行时继续交易。相关实现位于 `src/api/routes/config.py`、`src/cryptotrader/runtime.py` 和 `src/cryptotrader/runtime_config/`。

## 信号与执行边界

`TradingCycle` 是业务主链。Kronos、四智能体 LLM 委员会和自定义 `SignalComponent` 提供方向、置信度与证据，不产生订单、仓位比例、杠杆或退出价格。

```text
冻结配置 revision → 合并数据需求 → point-in-time SignalContext
 → 全部启用组件成功 → 加权融合 → TargetPosition → ATR 退出价格
 → 每个资金池的风控/审批/连接分配 → 差值执行 → Journal
```

启用权重之和必须为 1。融合内部以 long=+1、short=-1、neutral=0 计算 `Σ(weight × direction × confidence)`；下游只接收 `side: long | short | flat` 和非负 `size_ratio`。单一组件失败会终止本次融合。

`profiles/models.py` 保留信号领域模型；持久化和 revision 属于整份运行配置，不存在独立的网页 Profile 存储。执行资金池区分 simulated 与 real；Paper/Demo/Testnet 不能混入真实资金池，禁用或 canary-only 连接不能参与正常分配。每个连接使用自己的 venue session。

信号要求审批或资金池要求审批，任一成立即需要 HITL。审批携带配置 revision、冻结计划和有效期；过期、revision 变化、重复执行等条件受运行时检查。真实下单还需要独立的默认关闭开关，审批不能绕过它。

实际风险配置包含资金池总敞口、连接集中度、连接名义敞口和最大回撤四项；审批有效期单独由 HITL 消费。回测使用历史上下文和临时 Paper 执行，不等待人工审批。

## 页面与扩展

`/setup` 是同一配置中心的检查清单。八个入口及保存语义见 [配置指南](docs/CONFIGURATION.md)。`/strategy` 编辑信号；`/risk` 展示运行风险；`/scheduler` 管理运行规则与历史；`/backtest` 使用独立的参数表单。周期和审批事实通过现有周期 API 与审计页面读取。

插件安装示例及开发约束见 [类型化插件](docs/CONFIGURATION.md#类型化插件)。验收环境不构造真实 Runtime，详见 [验证边界](docs/verification/configuration-center/README.md)。
