# Feature Specification: LLM 韧性工程

**Feature Branch**: `004-llm-resilience-engineering`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与动机

CryptoTrader-AI 是全自动加密货币交易系统，所有 LLM 调用通过 `create_llm()` 工厂统一创建。当前架构存在以下韧性薄弱点：

1. **单点降级**：`create_llm()` 仅通过 `.with_fallbacks([fallback_llm])` 提供单级降级，主 provider 失败后仅有一次救援机会，降级失败则交易机会完全丢失。
2. **无重试中间件**：LLM 调用失败（超时、速率限制、网络抖动）直接抛出异常，由上层 `except Exception` 捕获并返回 mock 分析，造成数据质量下降。
3. **模型配置扁平化**：`[models]` 段仅有 8 个字段（`analysis`、`debate`、`verdict` 等），无法表达 provider 类型、API 端点、每秒请求上限等参数，多 provider 切换依赖手动修改 TOML。
4. **全局单一 provider**：`_build_llm_kwargs()` 始终生成 `ChatOpenAI` 实例，其他 provider（Anthropic Claude、Google Gemini）虽在架构评审中被列为支持目标，但无统一的 manifest 管理机制。
5. **JSON 解析脆弱性**：`_extract_json()` 在平衡括号提取失败后直接抛出 `ValueError`，`_regex_fallback()` 为最后兜底，但无结构化重试机制（携带 schema 提示重问 LLM）。
6. **Prompt cache 未优化**：各 node 的 system prompt 为静态前缀，但未利用 Anthropic/OpenAI 的 Prompt Caching 特性；动态内容（agent 分析结果）混在静态 system 中，降低 cache 命中率。

**LLM 不可用 = 交易信号缺失**，这是生产关键路径。韧性工程的目标是在 LLM 基础设施出现故障时最大限度保留交易决策质量，同时降低推理成本。

---

## 用户场景与验收测试 *(必填)*

### User Story 1 — 主 provider 瞬时故障时系统自动重试并恢复 (Priority: P0)

**用户故事**：作为交易系统运营者，当主 LLM provider（如 OpenAI）返回 HTTP 429 速率限制或 HTTP 503 短暂不可用时，系统应在不中断当前交易周期的情况下自动重试，无需人工干预。

**Why this priority**：速率限制和网络抖动是生产中最常见的 LLM 故障类型。现有代码在此场景下直接返回 `is_mock=True` 的低质量分析，导致交易决策降级。这是 P0 级韧性缺口。

**Independent Test**：模拟 LLM provider 前两次调用返回 `RateLimitError`，第三次成功，验证 `create_llm()` 最终返回有效响应而非 mock 分析。

**Acceptance Scenarios**:
1. **Given** 主 LLM provider 在同一个请求的前两次尝试中返回 `RateLimitError`，**When** `BaseAgent.analyze()` 或 `make_verdict_ai()` 调用 LLM，**Then** 系统在第三次尝试后成功获得有效响应，`AgentAnalysis.is_mock` 为 `False`，日志中记录重试次数。
2. **Given** 重试间隔配置为 `1s → 2s → 4s`（指数退避），**When** 连续三次重试均失败，**Then** 系统触发多级 fallback 链，切换到下一个 provider，而非直接返回 mock。
3. **Given** 系统正在执行完整交易周期（collect_snapshot → agents → verdict），**When** 任意一个 agent 的 LLM 调用发生 429，**Then** 整体 pipeline 延迟增加但不中断，不触发 circuit breaker。

---

### User Story 2 — 多 provider 有序 fallback：Claude → GPT → Gemini (Priority: P0)

**用户故事**：作为交易系统运营者，当主 provider 在所有重试耗尽后仍不可用时，系统应按照预设的优先级顺序自动切换到下一个 provider，而非立即返回 mock 分析或 hold 决策。

**Why this priority**：`create_llm()` 当前最多一级降级（主 → 单个 fallback），且两者均为 `ChatOpenAI` 实例，本质上仍依赖同一 API 端点。多 provider 链是真正意义上的韧性保障。

**Independent Test**：配置三级 fallback 链（provider_a → provider_b → provider_c），模拟前两个 provider 不可用，验证第三个 provider 成功响应。

**Acceptance Scenarios**:
1. **Given** `models.toml` 中为 `verdict` 角色配置了 `[provider_chain] = ["anthropic/claude-opus-4", "openai/gpt-5.4", "google/gemini-2.5-pro"]`，**When** 前两个 provider 均不可用，**Then** 系统使用第三个 provider 完成 verdict，`TradeVerdict.action` 不为 mock hold。
2. **Given** fallback 链的某个 provider 已切换，**When** 该调用结束后系统回到下一个交易周期，**Then** 系统仍优先尝试链中第一个 provider（不粘滞在降级 provider）。
3. **Given** fallback 链所有 provider 均不可用，**When** 系统尝试所有选项后，**Then** 系统返回加权规则 verdict（`make_verdict_weighted()`），并发出 `llm_all_providers_exhausted` 警告事件。

---

### User Story 3 — 分角色模型配置：不同任务使用不同能力级别的模型 (Priority: P1)

**用户故事**：作为系统配置者，我希望能够为不同角色（分析、辩论、总结、fallback）指定不同的模型，而不必为所有任务使用相同的高成本模型，以便在精度和成本之间做出合理权衡。

**Why this priority**：当前 `[models]` 段有 8 个字段但缺乏表达能力（无 provider 类型、无 API 参数）。分角色配置能让 `analysis` 使用强模型、`summarization` 使用便宜快速模型，在保持决策质量的同时降低运营成本。

**Independent Test**：配置 `analysis` 角色使用 `claude-opus-4`，`summarization` 角色使用 `deepseek-v4-flash`，分别调用对应角色的 LLM 创建路径，验证各自实例化的模型名称正确。

**Acceptance Scenarios**:
1. **Given** `models.toml` 中 `[roles.analysis]` 指向 `claude-opus-4`，`[roles.summarization]` 指向 `deepseek-v4-flash`，**When** `experience/reflect.py` 调用 summarization 模型，**Then** 使用 `deepseek-v4-flash` 而非 `claude-opus-4`，token 成本更低。
2. **Given** `[roles.flash]` 指向快速响应模型，**When** debate_round 中每个 agent 的挑战响应调用 flash 模型，**Then** 辩论轮次延迟相比使用 verdict 级模型降低可量化比例。
3. **Given** 某角色的模型字段为空（`""`），**When** `create_llm()` 被调用，**Then** 自动回退到 `roles.analysis` 配置，行为与当前一致（不破坏向后兼容）。

---

### User Story 4 — 结构化 JSON 解析重试：附 schema 提示重问 LLM (Priority: P1)

**用户故事**：作为系统开发者，当 LLM 返回格式不符合预期 JSON schema 的响应时，系统应携带具体的 schema 错误说明重新调用 LLM（最多 N 次），而非直接降级到 regex fallback 或返回 mock 分析。

**Why this priority**：`_extract_json()` → `_regex_fallback()` 的当前链路在 LLM 返回 markdown 代码块、带前缀说明文字等情况时会导致 `is_mock=True`，降低分析质量。结构化重试是在不丢失上下文的情况下修复格式错误的最有效手段。

**Independent Test**：模拟 LLM 前两次返回包含 markdown 代码块的 JSON，第三次返回纯净 JSON，验证解析成功且未触发 regex fallback。

**Acceptance Scenarios**:
1. **Given** LLM 首次返回 `\`\`\`json\n{...}\n\`\`\`` 格式，**When** JSON 解析管道处理该响应，**Then** 系统剥离 markdown 标记后成功解析，不触发重试（视为合法输出变体）。
2. **Given** LLM 返回的 JSON 缺少 `direction` 字段（schema 违规），**When** schema 验证失败，**Then** 系统在下一次重试的 prompt 中附加具体报错（`"Missing required field: direction"`）并重新调用 LLM。
3. **Given** 经过 5 次重试 JSON 仍无法满足 schema，**When** 所有重试耗尽，**Then** 系统使用 `_regex_fallback()` 提取方向和置信度，`is_mock=True`，记录 `json_parse_exhausted` 警告。

---

### User Story 5 — Prompt Cache 优化：降低重复 token 成本 (Priority: P2)

**用户故事**：作为系统运营者，在同一个交易周期内（或跨连续周期的相同市场状态下），重复使用相同 system prompt 的 LLM 调用应能利用 provider 的 Prompt Caching 特性，减少 token 消耗和延迟。

**Why this priority**：`VERDICT_PROMPT`、`ANALYSIS_FRAMEWORK`、`DEBATE_SYSTEM` 等 system prompt 内容在每次调用时均相同，而动态的 agent 分析结果在 user message 中。合理排布消息结构可大幅提升 cache 命中率，降低成本。

**Independent Test**：连续两次调用同一 system prompt 的 LLM 实例，验证第二次调用的 `response.usage_metadata` 中出现 `cache_read_input_tokens > 0`（Anthropic）或 `cached_tokens > 0`（OpenAI）。

**Acceptance Scenarios**:
1. **Given** 同一交易周期内 4 个 agent 使用相同的 `ANALYSIS_FRAMEWORK` system prompt，**When** 每个 agent 调用 LLM，**Then** 第 2~4 个 agent 的调用产生可量化的 cache hit（通过 `log_llm_usage()` 记录 `cache_read_input_tokens`）。
2. **Given** `VERDICT_PROMPT` 为静态内容，**When** 连续两个交易周期执行 verdict，**Then** 第二个周期的 verdict LLM 调用的 input token 成本低于第一个周期（cache 命中）。
3. **Given** Prompt Cache 特性仅在特定 provider/模型上可用，**When** 当前模型不支持 cache，**Then** 系统正常完成调用，不报错，`log_llm_usage()` 记录 `cache_read_input_tokens=0`。

---

### 边界条件

- **所有 provider 均不可用**：系统应最终降级到规则 verdict（`make_verdict_weighted()`），不崩溃，记录 `CRITICAL` 级别日志。
- **Fallback provider 比主 provider 更慢**：重试 + fallback 总耗时超过单次调用 timeout，系统应在超时边界返回已有结果而非无限等待。
- **重试放大 rate limit 压力**：对同一 provider 的指数退避不应引入 thundering herd；多个 agent 并行重试时 jitter 机制应分散请求。
- **Prompt cache 内容变更**：当 system prompt 模板更新后，cache 自动失效，首次调用产生 cache miss，不应触发错误。
- **Backtest 模式下的 LLM 缓存**：重试和 fallback 逻辑在 backtest 模式下（`disable_llm_cache()` 已激活）应正常工作，但不应额外增加 backtest 的 token 消耗。
- **JSON 重试中包含敏感数据**：重试 prompt 附加的 schema 错误说明不应包含原始 LLM 输出的用户数据（防止二次注入）。
- **空 fallback 链**：若 `models.toml` 中某角色的 `provider_chain` 只有一个 provider，系统行为应与当前 `.with_fallbacks([])` 等价，不报错。

---

## 需求规格 *(必填)*

### 功能需求

**FR-001**：系统应提供 `models.toml` 文件（位于 `config/models.toml`），支持将 model_id 映射到 provider 类型（`openai` / `anthropic` / `google` / `openai_compatible`）、API 端点、角色归属（`analysis` / `flash` / `summarization` / `fallback`）及每角色的 provider 优先级链。

**FR-002**：`create_llm()` 工厂应根据角色参数（`role: str`）查找 `models.toml` 中对应的 provider 配置，并实例化对应 provider 类型的 LangChain chat model 实例，保持现有调用方不感知 provider 类型变更。

**FR-003**：`create_llm()` 应支持多级 fallback 链构建——按 `models.toml` 中 `provider_chain` 的顺序，将第 2～N 个 provider 构建为 `.with_fallbacks()` 链，而非仅使用单一 fallback 模型。

**FR-004**：系统应在 `create_llm()` 层面实现指数退避重试中间件，默认参数为：最大重试 3 次、初始等待 1 秒、倍增系数 2、全随机 jitter（±50%），可通过 `[llm.retry]` 配置段覆盖。

**FR-005**：重试中间件应仅对可重试错误类型触发（`RateLimitError`、`APIConnectionError`、`APITimeoutError`、HTTP 5xx），对 `AuthenticationError`、`InvalidRequestError` 等不可重试错误立即抛出，不消耗重试预算。

**FR-006**：JSON 解析管道应在 `_extract_json()` 失败后（非 markdown 代码块剥离可修复的情况），携带具体的 schema 错误说明构造追加 prompt，最多重试 5 次调用 LLM 重新生成合规格式，仅在所有重试耗尽后降级到 `_regex_fallback()`。

**FR-007**：JSON 解析重试应使用与原始调用相同的 LLM 实例（含 fallback 链），不额外创建新的 LLM 实例，避免重试引入多余的初始化开销。

**FR-008**：`create_llm()` 应支持新增的角色参数 `role: str = ""`，当 `role` 非空时从 `models.toml` 解析配置；当 `role` 为空时，行为与当前相同（使用 `model` 参数 + `config.models.analysis` 解析逻辑），确保向后兼容。

**FR-009**：系统应在 `log_llm_usage()` 中增加对 `cache_read_input_tokens`（Anthropic）和 `cached_tokens`（OpenAI）字段的记录，通过 structlog 以 `prompt_cache_hit` 字段标注，支持后续成本分析。

**FR-010**：各 node 的 system prompt 构建应将静态前缀（角色定义、分析框架、输出 schema 约束）与动态后缀（市场数据、agent 分析结果）明确分离，静态前缀长度应满足目标 provider 的最低 cache 触发阈值（Anthropic: ≥1024 tokens；OpenAI: ≥1024 tokens）。

**FR-011**：`models.toml` 应支持独立于 `config/default.toml` 存在，通过 `config/local.toml` 中的 `models_path` 字段指定路径，默认为 `config/models.toml`；首次加载后与 `AppConfig` 一起缓存。

**FR-012**：当 fallback 链中所有 provider 均调用失败时，系统应发出结构化警告事件（`llm_all_providers_exhausted`，包含角色、错误摘要），上层调用方（如 `BaseAgent.analyze()`、`make_verdict_ai()`）应能感知并选择降级策略（mock 分析或规则 verdict），而非由异常冒泡决定行为。

**FR-013**：`config.py` 中的 `LLMConfig` 应新增 `retry_max_attempts: int`、`retry_base_delay_s: float`、`retry_backoff_factor: float`、`retry_jitter: bool` 字段，对应 `[llm.retry]` TOML 段，默认值分别为 `3`、`1.0`、`2.0`、`true`。

**FR-014**：`config.py` 中的 `ModelConfig` 应保留现有 8 个字段（向后兼容），新增 `models_path: str = ""` 字段，指向 `models.toml` 的文件路径；`models.toml` 缺失时系统应 fallback 到现有 `ModelConfig` 字段行为，不报错。

---

### 关键实体

**`models.toml`（新增文件）**：provider manifest 文件，按 model_id 为键，描述 provider 类型、API 基础 URL、对应角色、provider_chain 顺序。与 `config/default.toml` 同目录，通过 `local.toml` 可覆盖路径。

**`ResilientLLM`（概念实体）**：由 `create_llm()` 构建，封装指数退避重试中间件 + 多级 fallback 链的 LangChain runnable 组合体，对外表现为标准的 LangChain `BaseChatModel` 接口。

**`RetryConfig`（配置实体）**：对应 `[llm.retry]` TOML 段，控制重试次数、延迟、退避系数、jitter 开关。嵌套于 `LLMConfig` dataclass。

**`ModelRoleConfig`（配置实体）**：对应 `models.toml` 中单个角色的配置块，包含 `primary_model`（主模型）和 `provider_chain`（有序 fallback 列表）。

**`ProviderEntry`（配置实体）**：对应 `models.toml` 中单个 provider 的描述，包含 `model_id`、`provider_type`（`openai` / `anthropic` / `google` / `openai_compatible`）、`base_url`、`api_key_env`（读取 API Key 的环境变量名）。

**`JsonParseRetryContext`（运行时实体）**：JSON 解析重试时携带的上下文，包含原始 LLM 响应、失败原因、schema 期望字段、重试次数，用于构造修正 prompt。

---

## 成功标准 *(必填)*

### 可量化的验收指标

**SC-001**：在 LLM provider 返回 `RateLimitError` 的情况下，`BaseAgent.analyze()` 的 `AgentAnalysis.is_mock` 比率（相比当前）降低 ≥80%（即：只有所有重试和所有 fallback 均耗尽时才返回 mock）。

**SC-002**：在主 provider 完全不可用的场景下（模拟测试），`make_verdict_ai()` 的 `TradeVerdict.action` 为有效决策（非降级 hold）的比率 ≥ 当前通过单级 fallback 的比率，且至少支持 2 个可用 fallback provider。

**SC-003**：重试中间件的单元测试覆盖以下场景：仅可重试错误触发重试、不可重试错误立即穿透、最大重试耗尽后触发 fallback 链。

**SC-004**：JSON 解析重试使 `AgentAnalysis.is_mock=True`（由格式错误引起）的比率在压测中降低 ≥60%（相比当前仅有 `_regex_fallback()` 时的失败率）。

**SC-005**：`log_llm_usage()` 在 Anthropic provider 下正确记录 `cache_read_input_tokens` 字段，且连续两次相同 system prompt 的调用中第二次的 `cache_read_input_tokens > 0`（集成测试验证）。

**SC-006**：`models.toml` 缺失时（仅存在 `default.toml`），系统行为与当前完全一致，现有 742 个测试全部通过，无回归。

**SC-007**：新增的 `[llm.retry]` 配置段支持通过 `CRYPTOTRADER_LLM__RETRY__MAX_ATTEMPTS` 等环境变量覆盖（遵循现有 `apply_env_overrides()` 机制），且配置覆盖测试通过。

**SC-008**：在 backtest 模式下，LLM 缓存禁用（`disable_llm_cache()`）与重试/fallback 逻辑相互独立，backtest 运行不因重试机制引入额外 LLM 调用（重试不绕过 `_cache_disabled` 标志）。

---

## 假设

- 系统为单用户自托管部署，不需要考虑多租户 API key 隔离。
- `create_llm()` 是唯一的 LLM 创建入口，所有 LLM 调用（agents、debate、verdict、reflect）均经过此工厂，不存在绕过该工厂的直接实例化。
- 支持的 LangChain chat model 类型为：`ChatOpenAI`（OpenAI 及 OpenAI-compatible 接口）、`ChatAnthropic`（Anthropic Claude）、`ChatGoogleGenerativeAI`（Google Gemini），其他 provider 通过 `langchain-community` 扩展支持。
- `models.toml` 中 `api_key_env` 字段引用的环境变量由用户在部署时手动配置，系统不负责密钥轮换或安全存储。
- Prompt Cache 优化依赖 provider 支持（Anthropic Claude 3+ 和 OpenAI GPT-4o+），对不支持 cache 的 provider 无性能收益，但也无副作用。
- `models.toml` 中每个角色的 `provider_chain` 长度最多为 3（primary + 2 fallback），超过此数量的配置视为无效。
- 结构化 JSON 重试的最大次数（默认 5）适用于所有需要 JSON 输出的角色（agents、debate、verdict、reflect），由统一的解析重试管道处理。
- 指数退避重试的总最大等待时间（1 + 2 + 4 = 7 秒）加上 fallback 切换开销，不超过 `execution.graph_timeout_s`（默认 300 秒）的 5%，对交易周期整体延迟影响可接受。
- 本 spec 不涉及 LLM 调用的分布式追踪（OpenTelemetry spans），该功能已在架构评审（Completed 2026-03-15）中独立实现。
- 本 spec 不修改 `disable_llm_cache()` / `restore_llm_cache()` 的 backtest cache 控制逻辑，重试机制在 backtest 模式下对 cache 状态透明。
