# 技术实施方案：LLM 韧性工程

## 技术上下文

### 现有 `create_llm()` 实现分析

`create_llm()` 位于 `src/cryptotrader/agents/base.py`，是整个项目的唯一 LLM 创建入口。当前实现：

```
create_llm(model, temperature, timeout, json_mode, *, with_fallback)
  → _build_llm_kwargs()  # 始终生成 ChatOpenAI kwargs
  → ChatOpenAI(**kwargs)
  → llm.with_fallbacks([ChatOpenAI(**fallback_kwargs)])  # 单级，仅当 fallback_model != model
```

**关键问题**：
- `_build_llm_kwargs()` 硬编码返回 `ChatOpenAI` 参数，无法支持 Anthropic/Google provider
- 单级 fallback，且两个 LLM 实例均通过同一 `llm_cfg.base_url` 和 `llm_cfg.api_key`，本质上是同一 API 端点
- 无重试中间件：失败直接抛出异常，上层 `except Exception` 捕获后返回 `is_mock=True` 的结果
- `log_llm_usage()` 已有基础实现（记录 input/output tokens），但未记录 cache 相关字段

### 现有 fallback 机制

```python
if with_fallback:
    fallback_model = cfg.models.fallback
    if fallback_model and fallback_model != model:
        fallback_kwargs = _build_llm_kwargs(...)
        llm = llm.with_fallbacks([ChatOpenAI(**fallback_kwargs)])
```

LangChain 的 `.with_fallbacks()` 接口支持传入多个 fallback runnable，可扩展为多级链。

### 现有 JSON 解析链

`debate/verdict.py` 中的 `_extract_json(text)` 使用平衡括号提取法，失败时抛出 `ValueError`。
`agents/base.py` 的 `_parse_response()` 捕获 `ValueError` 后调用 `_regex_fallback()`，最终失败时返回 `is_mock=True`。

**缺口**：无"携带 schema 错误重问 LLM"的机制，regex fallback 是当前最后一道防线。

### 现有配置结构

- `LLMConfig`：`api_key`, `base_url`, `default_temperature`, `timeout`, `streaming_models`（全局单一 provider 参数）
- `ModelConfig`：8 个角色字段（`analysis`, `debate`, `verdict`, `tech_agent`, `chain_agent`, `news_agent`, `macro_agent`, `fallback`）——仅保存 model 名称字符串，无 provider 类型信息
- 所有角色均走同一 `ChatOpenAI` + 同一 `base_url`

### 已有依赖

`pyproject.toml` 中已包含：
- `langchain-openai>=1.1.14`（`ChatOpenAI`）
- `langchain-anthropic>=1.3`（`ChatAnthropic`）
- `langchain-google-genai>=4.2`（`ChatGoogleGenerativeAI`）

三个 provider 的 LangChain 绑定均已安装，无需新增依赖。

---

## 架构决策

### 决策 1：`models.toml` 作为独立文件，缺失时透明降级

**选择**：新增 `config/models.toml`，通过 `config/local.toml` 中的 `models_path` 字段可覆盖路径。`create_llm()` 在加载时检查文件是否存在：存在则启用新逻辑；不存在则完全回退到现有行为。

**理由**：现有 742+ 个测试不依赖 `models.toml`，零回归风险。部署时 `models.toml` 可选，用户可渐进迁移。

### 决策 2：`ProviderRegistry`（轻量 dataclass，非重量级注册表）

**选择**：新增 `src/cryptotrader/llm/` 子包，包含：
- `registry.py`：`ProviderEntry`、`ModelRoleConfig`、`ModelsManifest` dataclass + `load_manifest()` 函数
- `factory.py`：`_build_provider_llm()`（按 provider_type 实例化正确的 LangChain 类）、扩展 `create_llm()` 的角色参数处理

**理由**：provider 分发逻辑与配置加载解耦，便于单元测试。`agents/base.py` 的 `create_llm()` 签名向后兼容不变（`role` 参数默认为 `""`）。

### 决策 3：重试中间件使用 LangChain `RunnableRetry` + 自定义 `retry_if_exception`

**选择**：使用 LangChain `langchain_core.runnables.RunnableRetry`（或 `with_retry()`）包装 LLM runnable，配置指数退避 + jitter。对于 LangChain 未暴露完整 retry 控制的情况，降级为 `tenacity` 库手动封装 `ainvoke`。

**理由**：`tenacity` 已是 LangChain 的传递依赖（langchain-core 内部使用），无需新增依赖。`RunnableRetry` API 更简洁，但对 exception 类型的精细控制需要自定义 `retry_if_exception` 函数。

### 决策 4：JSON 解析重试在 `BaseAgent._parse_response()` 和 `_extract_json()` 扩展点

**选择**：新增 `src/cryptotrader/llm/json_retry.py`，暴露 `extract_json_with_retry(text, llm, schema_hint, max_retries)` 函数。在 `debate/verdict.py` 和 `nodes/debate.py` 中替换对 `_extract_json()` 的调用；在 `BaseAgent._parse_response()` 中替换现有的 `json.loads`（注意：`_parse_response()` 不调用 `_extract_json()`，使用 `json.loads` + `_regex_fallback()` 路径，改造时封装 `extract_json_with_retry()` 替代 `json.loads`）。

**理由**：集中管理重试逻辑，schema_hint 可按角色定制。原始 `_extract_json()` 保留供 `extract_json_with_retry` 内部调用。

### 决策 5：Prompt Cache 优化通过消息结构分离，不修改 LLM 实例创建

**选择**：在各 node 的 prompt 构建函数中，将静态前缀（system prompt 的角色定义、分析框架、输出 schema）与动态后缀（市场数据）拆分到不同的消息块或通过 `cache_control` 标记（Anthropic）。`log_llm_usage()` 新增 `cache_read_input_tokens` 字段记录。

**理由**：Prompt Cache 是 provider 层面的优化，对 LangChain 接口透明。结构分离本身不影响输出质量，对不支持 cache 的 provider 零影响。

### 决策 6：`llm_all_providers_exhausted` 事件通过结构化异常传递，不引入新的回调机制

**选择**：定义 `LLMProvidersExhaustedError(Exception)` 自定义异常，在 `create_llm()` 构建的 fallback 链耗尽后由 `ResilientLLM.ainvoke()` 抛出。上层调用方（`BaseAgent.analyze()`、`make_verdict_ai()`）在 `except` 块中检测此异常类型并执行降级逻辑（mock 分析 / `make_verdict_weighted()`），同时通过 structlog 发出 `llm_all_providers_exhausted` 警告事件。

**理由**：保持现有 `try/except` 模式，无需引入 event bus 或 callback 机制，最小化改动范围。

---

## 文件结构（新增/修改）

```
config/
  models.toml                     # 新增：provider manifest 文件
  default.toml                    # 修改：[llm.retry] 段
  local.toml.example              # 修改：添加 models_path 示例

src/cryptotrader/
  llm/                            # 新增子包
    __init__.py
    registry.py                   # ProviderEntry, ModelRoleConfig, ModelsManifest, load_manifest()
    factory.py                    # _build_provider_llm(), _build_resilient_llm()
    json_retry.py                 # extract_json_with_retry(), JsonParseRetryContext
    errors.py                     # LLMProvidersExhaustedError

  agents/
    base.py                       # 修改：create_llm() 新增 role 参数；log_llm_usage() 新增 cache 字段

  config.py                       # 修改：LLMConfig 新增 RetryConfig 嵌套；ModelConfig 新增 models_path

  debate/
    verdict.py                    # 修改：_extract_json() 调用改为 extract_json_with_retry()

  nodes/
    debate.py                     # 修改：_debate_one_agent() JSON 解析使用重试
    agents.py                     # 修改：_run_agent() 异常处理感知 LLMProvidersExhaustedError

tests/
  test_llm_registry.py            # 新增：ModelsManifest 加载、ProviderEntry 解析
  test_llm_factory.py             # 新增：create_llm() provider 分发、multi-fallback 链
  test_llm_retry.py               # 新增：重试中间件 RateLimitError/不可重试错误/耗尽
  test_json_retry.py              # 新增：JSON 解析重试链路
  test_prompt_cache.py            # 新增：log_llm_usage() cache 字段记录
  test_models_toml_missing.py     # 新增：models.toml 缺失时现有行为不变
```

---

## 数据模型

### `config/models.toml` Schema

```toml
# Provider 条目：定义一个具体的 model + provider 组合
[[providers]]
model_id     = "anthropic/claude-opus-4"     # 唯一标识（provider/model 命名风格）
provider_type = "anthropic"                   # openai | anthropic | google | openai_compatible
base_url     = ""                             # 空 = 使用官方端点
api_key_env  = "ANTHROPIC_API_KEY"           # 读取 API Key 的环境变量名

[[providers]]
model_id     = "openai/gpt-5.4"
provider_type = "openai"
base_url     = ""
api_key_env  = "OPENAI_API_KEY"

[[providers]]
model_id     = "google/gemini-2.5-pro"
provider_type = "google"
base_url     = ""
api_key_env  = "GOOGLE_API_KEY"

[[providers]]
model_id     = "custom/local-model"
provider_type = "openai_compatible"
base_url     = "http://localhost:8080/v1"
api_key_env  = "LOCAL_API_KEY"

# 角色配置：每个角色映射到有序的 provider 链
[roles.analysis]
primary_model  = "anthropic/claude-opus-4"
provider_chain = ["anthropic/claude-opus-4", "openai/gpt-5.4", "google/gemini-2.5-pro"]

[roles.verdict]
primary_model  = "openai/gpt-5.4"
provider_chain = ["openai/gpt-5.4", "anthropic/claude-opus-4"]

[roles.debate]
primary_model  = "anthropic/claude-opus-4"
provider_chain = ["anthropic/claude-opus-4", "openai/gpt-5.4"]

[roles.flash]
primary_model  = "google/gemini-2.5-pro"
provider_chain = ["google/gemini-2.5-pro", "openai/gpt-5.4"]

[roles.summarization]
primary_model  = "google/deepseek-v4-flash"
provider_chain = ["google/deepseek-v4-flash", "anthropic/claude-opus-4"]

[roles.fallback]
primary_model  = "openai/gpt-5.4"
provider_chain = ["openai/gpt-5.4"]
```

### Python Dataclass 模型

```python
# src/cryptotrader/llm/registry.py

@dataclass
class ProviderEntry:
    model_id: str
    provider_type: Literal["openai", "anthropic", "google", "openai_compatible"]
    base_url: str = ""
    api_key_env: str = ""

@dataclass
class ModelRoleConfig:
    primary_model: str
    provider_chain: list[str]  # 长度 1~3，ordered by priority

@dataclass
class ModelsManifest:
    providers: dict[str, ProviderEntry]      # model_id -> ProviderEntry
    roles: dict[str, ModelRoleConfig]         # role_name -> ModelRoleConfig

    def get_role(self, role: str) -> ModelRoleConfig | None: ...
    def get_provider(self, model_id: str) -> ProviderEntry | None: ...
```

### `LLMConfig` 扩展（`config.py`）

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    backoff_factor: float = 2.0
    jitter: bool = True

@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    streaming_models: list[str] = field(default_factory=list)
    default_temperature: float = 0.2
    timeout: int = 120
    retry: RetryConfig = field(default_factory=RetryConfig)  # 新增
```

### `ModelConfig` 扩展（`config.py`）

```python
@dataclass
class ModelConfig:
    # 现有 8 个字段保持不变（向后兼容）
    analysis: str = "gemini-3-flash"
    debate: str = "gemini-3-flash"
    verdict: str = "gpt-5.4"
    tech_agent: str = "gemini-3-flash"
    chain_agent: str = "gemini-3.1-pro"
    news_agent: str = "gemini-3.1-pro"
    macro_agent: str = "gemini-3.1-pro"
    fallback: str = "deepseek-chat"
    timeout_seconds: int = 90
    models_path: str = ""  # 新增：指向 models.toml 的路径
```

---

## `create_llm()` 新签名与行为

```python
def create_llm(
    model: str = "",
    temperature: float | None = None,
    timeout: int | None = None,
    json_mode: bool = False,
    *,
    with_fallback: bool = True,
    role: str = "",                  # 新增：角色名，用于查找 models.toml
) -> BaseChatModel:
```

**行为逻辑**：
1. 若 `role` 非空且 `models.toml` 已加载：从 `ModelsManifest` 获取角色的 `provider_chain`，构建多级 fallback LLM
2. 若 `role` 为空或 `models.toml` 不存在：行为与当前完全相同（`model` 参数 + `ChatOpenAI`）
3. 无论走哪条路径，均包裹指数退避重试中间件

---

## 向后兼容性保证

| 变更点 | 现有调用方 | 保证 |
|--------|-----------|------|
| `create_llm(model=...)` 签名 | 所有 agents、debate、verdict、reflect | `role=""` 默认值，现有调用不受影响 |
| `LLMConfig` 新增 `retry` | `_build_config()` | `RetryConfig` 有 `field(default_factory=...)` 默认值，TOML 无此段时使用默认值 |
| `ModelConfig` 新增 `models_path` | `load_config()` | `models_path=""` 默认值，空字符串时系统搜索 `config/models.toml` |
| `models.toml` 缺失 | `load_manifest()` | 返回 `None`，`create_llm()` 走现有路径 |
| `_extract_json()` 调用方 | `verdict.py`、`nodes/debate.py` | `extract_json_with_retry()` 内部仍调用 `_extract_json()`；不传 `llm` 参数时退化为原始行为。注意：`agents/base.py` 的 `_parse_response()` 使用 `json.loads` + `_regex_fallback()` 实现，不调用 `_extract_json()`，改造方案见 T018 |

---

## 依赖变更

无需新增外部依赖：
- `langchain-anthropic>=1.3`：已在 `pyproject.toml`
- `langchain-google-genai>=4.2`：已在 `pyproject.toml`
- `tenacity`：langchain-core 的传递依赖，已可用

---

## 风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| 重试放大 rate limit 压力 | 中 | jitter=True 引入 ±50% 随机延迟，分散并行请求；重试仅针对可重试错误类型 |
| Fallback provider 延迟更高导致交易周期超时 | 中 | `execution.graph_timeout_s`（300s）远大于最大重试总延迟（7s + fallback 切换开销），有足够 buffer |
| `models.toml` 语法错误导致系统无法启动 | 高 | `load_manifest()` 捕获 TOML 解析异常并 fallback 到 `None`（而非抛出），记录 WARNING 日志，系统继续以现有模式运行 |
| Anthropic/Google API key 未配置导致 fallback 链全部失败 | 中 | `LLMProvidersExhaustedError` 触发 `make_verdict_weighted()` 兜底，系统不崩溃 |
| JSON 重试中的 schema 错误信息包含 LLM 原始输出（注入风险） | 低 | `JsonParseRetryContext` 中的错误说明仅包含字段名和期望类型，不将原始 LLM 输出内容嵌入重试 prompt |
| Backtest 模式下重试增加 LLM 调用次数 | 低 | `_cache_disabled` 标志已在 `base.py` 管理，重试机制透明通过 `ainvoke`，SQLiteCache 的 miss 行为不变 |
| Prompt Cache 结构重排导致现有测试 snapshot 失效 | 低 | 仅修改 `ANALYSIS_FRAMEWORK` 等常量的结构注释，不改变实际内容；mock 测试不依赖 system prompt 内容 |
