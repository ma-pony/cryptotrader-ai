# 日志与异常记录规范

> 适用范围：`src/` 下全部业务代码（含 `cryptotrader/`、`api/`、`cli/`）。

## 核心原则

**异常一定要看得见至少一次。** 被吞掉 = 静默丢失行为正确性，下次出问题无法溯源。

历史遗留问题：曾有 85 处 `except → logger.debug(..., exc_info=True)`，
默认 `LOG_LEVEL=INFO` 下 `debug` 日志完全不输出，导致 `portfolio_unknown` /
`redis_unavailable` 等拒单事件触发后无任何 root cause 线索。本规范固化整改后的约定。

## 4 条规范

### 规范 1 — 异常分级映射

| 级别 | 含义 | 适用场景 |
|---|---|---|
| `logger.error(..., exc_info=True)` | 影响功能正确性，需要立即告警 | 配置缺失、不可恢复的初始化失败 |
| `logger.warning(..., exc_info=True)` | **关键路径**异常被吞，必须留痕 | 余额读取失败、风控数据缺失、订单状态丢失、止损读取失败、Redis ping 失败 |
| `logger.info(..., exc_info=True)` | 副作用/降级路径失败，功能仍可用 | 通知/UI 推送失败、dashboard 渲染降级、缓存不可用、心跳写入失败 |
| `logger.debug(...)`（**不带 `exc_info`**） | 调用追踪、采样诊断 | 进入函数、参数 dump、命中缓存等 |
| `logger.debug(..., exc_info=True)` | **❌ 禁止**用于吞异常 | — |
| `pass`（无日志） | **❌ 零容忍**——禁止 | — |

### 规范 2 — 关键路径硬规则

下列路径里的 `except` 块**至少**用 `logger.warning(..., exc_info=True)`，
不允许 `logger.debug` 或 `pass` 吞异常：

- `src/cryptotrader/nodes/execution.py`
- `src/cryptotrader/nodes/verdict.py`
- `src/cryptotrader/nodes/data.py`（仅 portfolio/risk-input 相关分支）
- `src/cryptotrader/portfolio/`
- `src/cryptotrader/risk/`
- `src/cryptotrader/execution/`
- `src/cryptotrader/journal/store.py`
- `src/cryptotrader/hitl/gate.py`
- `src/cryptotrader/data/market.py`（funding/OI 是风控输入）

### 规范 3 — 拒单/降级事件结构化记录

用户可见的拒单/降级事件（如 `portfolio_unknown`、`redis_unavailable`、
`exchange_unhealthy`）**事件本体必须携带原始 exception**，而不是只在 logger 里记一笔
traceback：

```python
# ❌ 反例 —— 出问题时事件里看不到任何 root cause
except Exception:
    logger.debug("Failed to read portfolio from exchange", exc_info=True)
    return None

# ✅ 正例 —— 事件本身携带 error_type，下次拒单查 log 直接定位
except Exception as e:
    logger.warning("read_portfolio_from_exchange failed: %s", e, exc_info=True)
    self._last_error = {"error_type": type(e).__name__, "error": str(e)}
    return None
```

下游的拒单 reason 可以拼接 `error_type`：

```python
return GateResult(
    passed=False,
    rejected_by="portfolio_unknown",
    reason=f"Exchange returned 0 balance — {error_type}: {error_msg}",
)
```

### 规范 4 — CI 强制检查

`pyproject.toml` 加 ruff 规则禁止 `except → logger.debug(..., exc_info=True)`
的反模式；新提交 grep 命中即拒绝合并。

```bash
# pre-commit / CI 扫描脚本
! git diff --cached --name-only -z | xargs -0 grep -l 'logger\.debug.*exc_info=True'
```

## 决策树（写代码时如何选 level）

```
异常被 except 接住后：
├─ 是否影响交易/订单/风控/portfolio 正确性？
│   └─ 是 → logger.warning(..., exc_info=True)
│           + 若是用户可见的拒单事件，把 error_type 塞进 reason/event
├─ 是否只是通知/UI/dashboard/缓存/心跳等副作用？
│   └─ 是 → logger.info(..., exc_info=True)
└─ 其他 → 默认 logger.info(..., exc_info=True)
         （不确定时倾向 info，绝不用 debug 吞）
```

## FAQ

**Q：为什么不全部用 warning？**
A：会刷屏。通知失败、UI 刷新失败、心跳偶发抖动是预期范围内事件，warning 会
让真正的关键告警淹没在噪声里。

**Q：能不能用 `pass`？**
A：不能。哪怕一行 `logger.info("dust amount filtered: %s", amt)` 也行——
保留语义意图、保留排查窗口。

**Q：什么时候用 `logger.error`？**
A：服务无法继续工作时。一般是启动期配置错误、关键依赖永久不可达。**绝大多数业务异常
都是 warning，不是 error。**

**Q：`logger.debug` 还能用吗？**
A：可以，**但不能带 exc_info=True 用来吞异常**。`logger.debug` 适合：
进入/退出函数追踪、参数 dump、命中缓存、跳过 cooldown 之类的采样诊断。
