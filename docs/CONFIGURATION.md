# 配置中心

运行配置保存在数据库，密钥只写入加密 vault。初始化与日常修改使用相同表单，系统未启用时也能访问所有分区。

| 分区 | 路径 |
| --- | --- |
| 模型与网关 | `/settings/models` |
| 信号、行情、风险与自动化 | `/engine` |
| 平台连接与资金池 | `/accounts` |
| 研究与回测 | `/research` |
| 模型 | `/settings/models` |
| 通知 | `/settings/notifications` |
| 安全 | `/settings/security` |

工作台显示配置准备情况和下一步。访问过某节不代表该节有效；保存配置不会启用交易。

## 保存与检查

每节独立保存，其他节的未保存编辑保留。空数字字段保持为空并显示关联错误，不自动变成零。风控比例、信号权重和资金池分配等比例字段在界面以百分数显示，例如 7.5%，在配置中保存为 0.075。调度规则的涨跌幅、资金费率使用百分数值本身；例如资金费率 0.1% 的参数仍是 0.1，不除以 100。

出现版本冲突时，先重新加载最新 revision；本地非秘密编辑保留供检查和再次保存。放弃修改需要确认。凭据输入与配置草稿分开，写入后清空，响应只显示是否配置与更新时间。

保存成功与运行时应用成功分开显示。准备候选失败不会提交新 revision；发布失败时 desired revision 可能已经保存，`apply_status=failed`，运行时关闭执行入口。重新加载可读取准确状态，不能根据 HTTP 错误推断旧配置仍在运行。

Paper 新连接显示 10000 USDT 默认资金，保存前可修改，不需要 API Key。OKX 只提供 Demo/Live，需要 Key、Secret 和创建 Key 时设置的 Passphrase；Bybit 提供 Demo/Testnet/Live，需要 Key 和 Secret。连接环境保存后不可改。

只读检查使用已保存配置和凭据查询账户，显示检查时间，不验证下单权限、不提交订单。改动配置或轮换凭据会使旧结果失效。保存凭据本身不代表认证成功。

资金池的 ID 和资金作用域保存后固定。只有启用且作用域匹配的非 canary 连接可分配。权重总计必须为 100%；计算示例清楚标注为示例，不使用真实余额。真实下单开关默认关闭，HITL 不能绕过它。

## 类型化代码注册

代码注册示例位于 [`examples/configuration_plugin/configuration_example.py`](../examples/configuration_plugin/configuration_example.py)。它始终返回中性信号，不调用模型。应用启动代码显式调用示例的注册函数：

```python
from cryptotrader.configuration.registry import get_extension_registry
from examples.configuration_plugin.configuration_example import register_example

register_example(get_extension_registry())
```

本期不扫描已安装包、entry point 或运行时目录。注册 ID、声明 ID 与组件 ID 必须相同；注册与导入都不应联系模型或交易所。运行配置仍保存在数据库。

参数模型使用 `BaseModel`、`ConfigDict(extra="forbid", hide_input_in_errors=True)` 和有默认值的字段。`Field` 的 `json_schema_extra` 支持双语 label/description、unit、step、advanced 及 choice options；数值约束使用 Pydantic 的范围规则。嵌套模型生成分组字段，示例的 `diagnostics.enabled` 是高级开关。

支持文本、整数/数值、布尔、枚举选择与字符串列表。不支持任意对象、秘密参数字段或 JSON 编辑兜底。缺少声明、ID 不匹配、重复注册和不支持的字段定义会在发现阶段失败；未知键和非法参数在保存前被拒绝。工厂应再次从同一参数模型读取类型化值。

市场源和平台使用同一个应用代码注册表。平台声明还包含支持的 environments 和 credential_fields；凭据通过 vault 接口写入，不能作为普通配置字段暴露。

## 已移除字段与部署前清理

本次验收没有修改现有用户数据库。旧库若包含下列字段，严格模型会拒绝加载；部署前需要另行确认目标数据库并授权一次性清理。不要用兼容层重新引入字段，也不要删除数据库重建。

- `llm.vision_models`、`llm.max_image_bytes`
- `notifications.telegram`、`execution.allocation_policy`
- `risk.max_stop_loss_pct`、`risk.token_tax_threshold`
- `risk.position.max_correlated_positions`、`risk.position.max_same_direction_positions`
- `risk.loss.max_daily_loss_pct`、`risk.loss.max_cvar_95`、`risk.loss.cvar_min_returns`
- 整个 `risk.cooldown`、`risk.volatility`、`risk.exchange`、`risk.rate_limit`
- `llm_committee` 组件参数中的 `debate.divergence_hold_threshold`

`notifications.events` 只保留 `daily_summary`；原本为空就保持为空。已停用事件没有运行时消费者。新闻供应商 Key 使用 write-only vault；若旧配置不存在明文 `market_data.parameters.coindesk_api_key`，保持不存在，不要凭空补值。

授权清理时先保存可恢复备份，暂停唯一运行时 owner，以最新 revision 为基准移除精确路径，再用当前模型验证。保留凭据引用、加密 vault、稳定的 `CONFIG_MASTER_KEY` 和所有其他用户值。不得改变 `system.active` 或真实下单开关；已关闭的开关必须继续关闭。清理与重启需要单独验证，本验收不代表部署已完成。
