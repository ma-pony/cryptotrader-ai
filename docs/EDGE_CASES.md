# CryptoTrader AI — 边缘情况与开发规范

---

## 1. 容错处理

| 场景 | 处理方式 |
|------|----------|
| LLM 调用失败 | Agent 返回 `is_mock=True, confidence=0.1, direction=neutral` |
| 全部 Agent mock | Verdict 强制 `hold`，不交易 |
| Redis 不可用 | 降级到内存状态管理，跳过 Redis 依赖检查 |
| PostgreSQL 不可用 | 降级到内存日志（最多 10,000 条） |
| 交易所 API 超时 | LiveExchange 3 次重试，指数退避 |
| 余额不足 | PaperExchange 返回 `failed`，不抛异常 |
| JSON 解析失败 | `_extract_json` 平衡括号提取，处理 markdown fence |
| 数据源 API 失败 | 各 collector 独立 try/catch，返回默认值 |
| 熔断器触发 | Verdict 强制 `hold`，需手动 `arena risk reset-breaker` |

## 2. 关键设计决策

| 决策 | 原因 |
|------|------|
| 固定 2 轮辩论（非动态收敛） | 动态收敛会人为趋同，2 轮允许真实分歧 |
| `close` 动作豁免全部风控 | 减仓是降低风险，不应被风控阻断 |
| position_scale 连续映射 | 三档离散映射丢失 AI 的精细判断 |
| ToolAgent 回测跳过工具 | 避免前视偏差 + 减少无效 HTTP 超时 |
| FnG limit 动态计算 | 固定 400 天不够覆盖远期回测 |
| 每个 FRED 序列独立速率限制 | 共享 key 导致第二个序列永远跳过 |
| Config 首次加载后缓存 | 避免 mid-run 配置变更导致不一致 |
| 信号生成在 bar[i]，执行在 bar[i+1] 开盘 | 消除前视偏差 |

## 3. 已知限制

- `langchain_agents.py` 绕过统一配置（不读 `base_url`/`api_key`）
- CorrelationCheck 使用硬编码 14 组，非动态计算
- `verify=False` 存在于约 15 个外部 API 调用中
- `graph_supervisor.py` 是备选架构，非主路径

---

## 4. 开发规范

### 4.1 代码质量

```bash
make lint          # uv run ruff check src/ tests/  → 必须零错误
make test          # uv run pytest tests/ -v        → 288 pass, 1 skip
```

- **禁止 `noqa` 注释** — 遇到 C901 必须重构，遇到 F401 必须删除或 `__all__`
- **C901 阈值 = 10** — 超过时拆分辅助函数
- **异步测试**：`asyncio_mode = "auto"`，无需 `@pytest.mark.asyncio`
- **Mock LLM**：`patch("langchain_openai.ChatOpenAI.ainvoke")` → `AIMessage(content=...)`
- **导入路径**：`from cryptotrader.*`（非 `src.cryptotrader.*`）
- **必须用 `uv run pytest`**（Python 3.12 venv），非裸 `pytest`

### 4.2 依赖管理

```bash
uv add <package>              # 添加到 pyproject.toml
uv add --group dev <package>  # 添加到 dev 组
uv sync                       # 从 lockfile 安装
uv lock                       # 重新生成 lockfile
```

**永远不要** `pip install`。

### 4.3 基础设施

```bash
docker compose up -d           # PostgreSQL 16 + Redis 7
arena migrate                  # 创建数据库表
arena sync                     # 同步历史数据
```
