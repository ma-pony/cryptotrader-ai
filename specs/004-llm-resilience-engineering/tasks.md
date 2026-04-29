# 实施任务：LLM 韧性工程

## Phase 1：基础设施 — 配置层扩展

- [X] T001 在 `src/cryptotrader/config.py` 新增 `RetryConfig` dataclass，字段：`max_attempts: int = 3`、`retry_base_delay_s: float = 1.0`、`retry_backoff_factor: float = 2.0`、`retry_jitter: bool = True`；将其嵌套至 `LLMConfig` 的 `retry` 字段；在 `_build_config()` 中从 `llm.retry` TOML 段解析
  - 关联：FR-013
  - 验收：`LLMConfig().retry.max_attempts == 3`；`default.toml` 无 `[llm.retry]` 段时使用默认值

- [X] T002 在 `src/cryptotrader/config.py` 的 `ModelConfig` 中新增 `models_path: str = ""` 字段；在 `_build_config()` 中从 `models.models_path` TOML 读取；`apply_env_overrides()` 支持 `CRYPTOTRADER_MODELS__MODELS_PATH` 覆盖
  - 关联：FR-014、SC-007
  - 验收：`ModelConfig().models_path == ""`；环境变量覆盖生效

- [X] T003 [P] 在 `config/default.toml` 新增 `[llm.retry]` 配置段（`max_attempts = 3`、`retry_base_delay_s = 1.0`、`retry_backoff_factor = 2.0`、`retry_jitter = true`）
  - 关联：FR-013
  - 验收：`load_config()` 读取后 `cfg.llm.retry.max_attempts == 3`

- [X] T004 [P] 创建 `config/models.toml` manifest 文件：定义示例 providers（anthropic/claude-opus-4、openai/gpt-5.4、google/gemini-2.5-pro、google/deepseek-v4-flash）及 roles（analysis、verdict、debate、flash、summarization、fallback）；每个 role 的 `provider_chain` 长度为 1~3
  - 关联：FR-001、US2
  - 验收：TOML 语法合法，可被 `tomllib.load()` 解析；至少包含 3 个 providers 和 4 个 roles

---

## Phase 2：LLM 子包 — Provider 注册表

- [X] T005 创建 `src/cryptotrader/llm/__init__.py`（空文件，声明子包）和 `src/cryptotrader/llm/errors.py`：定义 `LLMProvidersExhaustedError(RuntimeError)`，含 `role: str`、`providers_tried: list[str]`、`last_error: Exception` 属性
  - 关联：FR-012
  - 验收：`raise LLMProvidersExhaustedError(role="verdict", ...)` 可被 `except LLMProvidersExhaustedError as e:` 捕获，`e.role` 属性正确

- [X] T006 创建 `src/cryptotrader/llm/registry.py`：实现 `ProviderEntry`、`ModelRoleConfig`、`ModelsManifest` dataclass；实现 `load_manifest(path: Path | None = None) -> ModelsManifest | None`，处理文件不存在（返回 `None`）和 TOML 语法错误（记录 WARNING 后返回 `None`）两种 fallback 场景；manifest 对象在进程内缓存（模块级变量，线程安全）
  - 关联：FR-001、FR-011、SC-006
  - 验收：`load_manifest(Path("config/models.toml"))` 返回正确的 `ModelsManifest`；`load_manifest(Path("不存在.toml"))` 返回 `None` 而非抛出异常

- [X] T007 [P] 在 `src/cryptotrader/llm/registry.py` 中实现 `ModelsManifest.get_role(role)` 和 `ModelsManifest.get_provider(model_id)` 方法；对 `provider_chain` 超过 3 个条目的角色记录 WARNING 并截断至前 3 个；对 `provider_chain` 中引用但 providers 表中不存在的 model_id 记录 WARNING 并跳过该条目
  - 关联：FR-001（`provider_chain` 长度约束）
  - 验收：超过 3 个的链被截断，缺失 provider 不导致异常

- [X] T008 创建 `src/cryptotrader/llm/factory.py`：实现 `_build_provider_llm(entry: ProviderEntry, temperature, timeout, json_mode) -> BaseChatModel`，按 `provider_type` 分发：
  - `"openai"` / `"openai_compatible"` → `ChatOpenAI`
  - `"anthropic"` → `ChatAnthropic`
  - `"google"` → `ChatGoogleGenerativeAI`
  - `api_key_env` 非空时从 `os.environ` 读取 API key（不覆盖已有 key）
  - `base_url` 非空时传入（仅 openai/openai_compatible 支持）
  - 关联：FR-002
  - 验收：分别传入 4 种 provider_type，返回对应 LangChain 类的实例；未知 provider_type 抛出 `ValueError`

---

## Phase 3：核心功能 — 指数退避重试中间件

- [X] T009 [US1] 在 `src/cryptotrader/llm/factory.py` 实现 `_is_retryable(exc: Exception) -> bool`：可重试 → `RateLimitError`、`APIConnectionError`、`APITimeoutError`、HTTP 5xx（状态码 500~599）；不可重试 → `AuthenticationError`、`BadRequestError`（含 `InvalidRequestError`）
  - 关联：FR-005、SC-003
  - 验收：单元测试覆盖可重试/不可重试错误类型；`AuthenticationError` 立即穿透

- [X] T010 [US1] 在 `src/cryptotrader/llm/factory.py` 实现 `_wrap_with_retry(llm: BaseChatModel, retry_cfg: RetryConfig) -> BaseChatModel`：使用 `tenacity` 库的 `retry`/`wait_exponential`/`wait_random`/`stop_after_attempt`；`retry=retry_if_exception(_is_retryable)`；等待策略：`wait_exponential(multiplier=retry_cfg.retry_backoff_factor, min=retry_cfg.retry_base_delay_s)` + 若 `retry_jitter` 则叠加 `wait_random(0, retry_cfg.retry_base_delay_s * 0.5)`；通过 `before_sleep_log` 在 structlog 中记录重试次数
  - 关联：FR-004、SC-003
  - 验收：前 2 次 `RateLimitError` 后第 3 次成功时，`is_mock=False`，日志含重试次数；`AuthenticationError` 不触发重试

- [X] T011 [US1, P] 实现 `_build_resilient_llm(role_cfg: ModelRoleConfig, manifest: ModelsManifest, temperature, timeout, json_mode, retry_cfg: RetryConfig) -> BaseChatModel`：按 `provider_chain` 顺序构建 LLM 列表；每个 LLM 先 `_wrap_with_retry()`；首个 LLM 调用 `.with_fallbacks(rest, exceptions_to_handle=(Exception,))`；所有 provider 均失败时，`with_fallbacks` 的最终异常被 `_catch_exhausted_wrapper()` 捕获后抛出 `LLMProvidersExhaustedError`
  - 关联：FR-003、FR-012、US2
  - 验收：配置 3 provider 链，前 2 个不可用，第 3 个成功；全部不可用时抛出 `LLMProvidersExhaustedError`

---

## Phase 4：核心功能 — `create_llm()` 扩展

- [X] T012 [US2, US3] 修改 `src/cryptotrader/agents/base.py` 中的 `create_llm()`：新增 `role: str = ""` 关键字参数；`role` 非空时调用 `load_manifest()` 尝试加载 `models.toml`（路径来自 `cfg.models.models_path` 或默认 `config/models.toml`）；manifest 存在时走 `_build_resilient_llm()` 路径；manifest 不存在或角色未找到时降级到现有 `ChatOpenAI` 路径（确保向后兼容）
  - 关联：FR-002、FR-008、FR-014、SC-006
  - 验收：`create_llm(model="gpt-4o")` 行为与当前完全一致（无 `role` 参数时）；`create_llm(role="verdict")` 在 manifest 存在时返回多级 fallback chain

- [X] T013 [US2] 修改 `create_llm()` 中已有的单级 fallback 路径（`role=""` 时）：现有 `with_fallbacks([ChatOpenAI(...)])` 之前先 `_wrap_with_retry()` 包装主 LLM；若 `with_fallback=True`，fallback LLM 同样包裹重试中间件
  - 关联：FR-004、US1
  - 验收：`role=""` 路径下，`RateLimitError` 前 2 次失败后第 3 次成功，测试通过

- [X] T014 [US2] 修改 `src/cryptotrader/agents/base.py` 的 `BaseAgent.analyze()` 异常处理：新增对 `LLMProvidersExhaustedError` 的捕获，发出 structlog `llm_all_providers_exhausted` 警告事件（含 `role`、`providers_tried`、`agent_id` 字段），然后返回 `is_mock=True` 的分析结果
  - 关联：FR-012、SC-001
  - 验收：所有 provider 耗尽时，`analyze()` 返回 mock 分析而非抛出异常，structlog 包含 `llm_all_providers_exhausted` 事件

- [X] T015 [US2, P] 修改 `src/cryptotrader/debate/verdict.py` 的 `make_verdict_ai()` 异常处理：新增对 `LLMProvidersExhaustedError` 的捕获，调用 `make_verdict_weighted(analyses)` 作为降级；发出 structlog `llm_all_providers_exhausted` 警告；返回 `TradeVerdict(action="hold", ...)` 但通过 `make_verdict_weighted()` 可能得到有效 action
  - 关联：FR-012、US2 验收场景 3
  - 验收：所有 provider 耗尽时，`make_verdict_ai()` 不崩溃，返回 `make_verdict_weighted()` 的结果

---

## Phase 5：核心功能 — 结构化 JSON 解析重试

- [X] T016 [US4] 创建 `src/cryptotrader/llm/json_retry.py`：定义 `JsonParseRetryContext` dataclass（`raw_text: str`、`error_msg: str`、`schema_hint: str`、`attempt: int`）；实现 `_strip_markdown_fences(text: str) -> str`（剥离 ` ```json ` 等代码块标记，视为合法变体，不触发重试）
  - 关联：FR-006、US4 验收场景 1
  - 验收：包含 ` ```json\n{...}\n``` ` 的文本经剥离后能被 `json.loads()` 解析

- [X] T017 [US4] 在 `src/cryptotrader/llm/json_retry.py` 实现 `extract_json_with_retry(text: str, llm: BaseChatModel | None = None, schema_hint: str = "", max_retries: int = 5, original_messages: list | None = None) -> dict`：
  - 先尝试 `_strip_markdown_fences()` + `_extract_json()`（不计入重试次数）
  - 失败后：若 `llm is None` 或 `max_retries == 0`，直接调用 `_regex_fallback()`
  - 有 `llm` 时：构造 `JsonParseRetryContext`，追加修正 prompt（含 schema 期望字段，**不包含原始 LLM 输出内容**），调用 `await llm.ainvoke(fix_messages)`，递归重试
  - 所有重试耗尽后：调用 `_regex_fallback()`，发出 `json_parse_exhausted` structlog 警告
  - 关联：FR-006、FR-007、SC-004
  - 验收：前 2 次返回格式错误 JSON，第 3 次返回合法 JSON 时，函数成功解析且未触发 regex fallback；5 次全败时 `_regex_fallback()` 被调用

- [X] T018 [US4, P] 修改 `src/cryptotrader/agents/base.py` 的 `_parse_response()`：
  - **前置检查**：先运行 `grep -rn "_parse_response" src/` 确认 `_parse_response()` 的所有调用点；预期仅在 `BaseAgent.analyze()`（已为 `async def`）中被调用，若存在其他同步调用点需先重构
  - **注意**：`agents/base.py` 的 `_parse_response()` 当前使用 `json.loads` + `_regex_fallback()` 实现，**不调用 `_extract_json()`**；改造目标是将其中的 `json.loads` 替换为 `await extract_json_with_retry()`，使 JSON 解析路径具备重试能力
  - 将 `_parse_response()` 改为 `async def _parse_response()`，内部调用 `await extract_json_with_retry(response_text, llm=self._current_llm, schema_hint=AGENT_JSON_SCHEMA_HINT, max_retries=5, original_messages=self._last_messages)` 替代现有的 `json.loads` 调用；保留 `_regex_fallback()` 作为 `extract_json_with_retry` 的终极兜底（已在 T017 的实现中集成）
  - 在 `analyze()` 中保存 `self._current_llm` 和 `self._last_messages` 实例变量供 `_parse_response()` 使用；将 `_parse_response()` 的调用改为 `await self._parse_response()`
  - **测试更新**：同步更新所有 mock `_parse_response` 的测试（在 `tests/` 中搜索相关 mock），将同步 mock 改为 `AsyncMock`
  - 关联：FR-006、FR-007
  - 验收：`BaseAgent.analyze()` 集成测试中，LLM 前 2 次返回含 markdown 代码块的 JSON，第 3 次纯净 JSON，`is_mock=False`；相关测试中 `_parse_response` mock 均已更新为 `AsyncMock`

- [X] T019 [US4, P] 修改 `src/cryptotrader/nodes/debate.py` 的 `_debate_one_agent()`：将 `_extract_json(text)` 替换为 `await extract_json_with_retry(text, llm=llm, schema_hint=DEBATE_JSON_SCHEMA_HINT, max_retries=3)`
  - 关联：FR-006
  - 验收：debate 节点 JSON 解析失败后可携带 schema 提示重试

- [X] T020 [US4, P] 修改 `src/cryptotrader/debate/verdict.py` 的 `make_verdict_ai()`：将 `_extract_json(text)` 替换为 `await extract_json_with_retry(text, llm=llm, schema_hint=VERDICT_JSON_SCHEMA_HINT, max_retries=5)`
  - 关联：FR-006
  - 验收：verdict LLM 返回格式错误时不立即 fallback 到 hold

---

## Phase 6：核心功能 — Prompt Cache 优化

- [X] T021 [US5] 修改 `src/cryptotrader/agents/base.py` 的 `log_llm_usage()`：从 `response.usage_metadata` 中额外提取 `cache_read_input_tokens`（Anthropic 字段）和 `cached_tokens`（OpenAI `prompt_tokens_details` 中的字段）；通过 structlog 新增 `prompt_cache_hit: bool`（>0 时为 True）和 `cache_read_input_tokens: int` 字段记录
  - 关联：FR-009、SC-005
  - 验收：Anthropic provider mock 返回含 `cache_read_input_tokens=150` 的 usage_metadata，`log_llm_usage()` 日志中 `prompt_cache_hit=True`

- [X] T022 [US5, P] 在 `src/cryptotrader/agents/base.py` 中提取 `ANALYSIS_FRAMEWORK` 的静态前缀（角色定义 + 规则 + 输出 schema 约束）与动态后缀（数据内容）的分离注释；确认 `ANALYSIS_FRAMEWORK` 常量的 token 数 ≥ 1024（使用已有的 `_estimate_tokens()` 函数计算，该函数对 ASCII 字符按 `÷4`、CJK 字符按 `÷1.5` 估算，比 `len // 4` 更准确；如不足则通过扩充 schema 说明达到阈值）
  - 关联：FR-010
  - 验收：`_estimate_tokens(ANALYSIS_FRAMEWORK) >= 1024`，或添加 schema 详细说明后满足阈值

- [X] T023 [US5, P] 在 `src/cryptotrader/debate/verdict.py` 中确认 `VERDICT_PROMPT` 静态内容 token 数 ≥ 1024；使用 `_estimate_tokens(VERDICT_PROMPT)` 计算（而非 `len // 4`），必要时通过在 VERDICT_PROMPT 末尾添加更详细的 schema 字段说明来补足（不改变 prompt 语义）；分离静态前缀（VERDICT_PROMPT 常量）与动态内容（agent_reports、position_block 等）到不同消息
  - 关联：FR-010
  - 验收：`_estimate_tokens(VERDICT_PROMPT) >= 1024`

---

## Phase 7：测试覆盖

- [X] T024 [US1, SC-003] 新增 `tests/test_llm_retry.py`：
  - 用 `unittest.mock.patch` mock `tenacity` 的 sleep 函数（避免实际等待）
  - 测试 1：前 2 次 `RateLimitError`，第 3 次成功 → 返回有效响应，`is_mock=False`
  - 测试 2：`AuthenticationError` 不触发重试，立即抛出
  - 测试 3：连续 3 次 `RateLimitError`（超过 max_attempts=3）→ 触发 fallback 链
  - 测试 4：`max_attempts` 通过 `CRYPTOTRADER_LLM__RETRY__MAX_ATTEMPTS=1` 环境变量覆盖后仅重试 1 次
  - 关联：SC-003、SC-007

- [X] T025 [US2] 新增 `tests/test_llm_factory.py`：
  - 测试 `_build_provider_llm()` 按 `provider_type` 返回正确的 LangChain 类
  - 测试 3 provider 链：前 2 个不可用（mock），第 3 个成功
  - 测试全链耗尽后抛出 `LLMProvidersExhaustedError`，`e.providers_tried` 长度为 3
  - 测试 `create_llm(role="verdict")` 在 manifest 存在时返回 resilient LLM（via mock manifest）

- [X] T026 [US3] 新增 `tests/test_llm_registry.py`：
  - 测试 `load_manifest()` 正确解析 `config/models.toml` 中的 providers 和 roles
  - 测试 `load_manifest(Path("不存在"))` 返回 `None`
  - 测试 `provider_chain` 超过 3 条时截断并记录 WARNING
  - 测试 `models_path` 为空时走默认搜索路径

- [X] T027 [US4, SC-004] 新增 `tests/test_json_retry.py`：
  - 测试 1：` ```json\n{...}\n``` ` 格式直接解析，不触发重试
  - 测试 2：缺少 `direction` 字段，LLM 第 2 次返回完整 JSON → 解析成功，`is_mock=False`
  - 测试 3：5 次重试全败 → `_regex_fallback()` 被调用，发出 `json_parse_exhausted` 事件
  - 测试 4：`extract_json_with_retry(text, llm=None)` 降级为原始 `_extract_json()` + regex 行为

- [X] T028 [US5, SC-005] 新增 `tests/test_prompt_cache.py`：
  - mock `AIMessage.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 80}`
  - 验证 `log_llm_usage()` 输出中 `prompt_cache_hit=True`、`cache_read_input_tokens=80`
  - mock `cache_read_input_tokens=0` 时 `prompt_cache_hit=False`

- [X] T029 [SC-006] 新增 `tests/test_models_toml_missing.py`：
  - 设置 `models_path` 指向不存在的路径
  - 调用 `create_llm(role="analysis")` → 降级到 `ChatOpenAI`，不抛异常
  - 调用 `create_llm(model="gpt-4o")` → 与当前行为完全一致
  - 运行完整交易流程 mock（snapshot → agents → verdict），所有 742 个既有测试通过

- [X] T030 [SC-007] 在 `tests/test_credentials_and_model_timeout.py`（已存在）中追加测试：
  - `CRYPTOTRADER_LLM__RETRY__MAX_ATTEMPTS=5` 环境变量被正确解析为 `cfg.llm.retry.max_attempts == 5`
  - `CRYPTOTRADER_LLM__RETRY__RETRY_BASE_DELAY_S=2.0` 被正确解析为 `cfg.llm.retry.retry_base_delay_s == 2.0`
  - 关联：SC-007

- [X] T031 [SC-008, P] 在 `tests/test_nodes.py`（已存在）中追加 backtest 模式测试：
  - `disable_llm_cache()` 激活后，重试中间件的 sleep 不绕过 cache 标志
  - backtest 模式下 `create_llm()` 正常工作，重试逻辑不引入额外真实 LLM 调用（所有 LLM mock）
  - 关联：SC-008

---

## Phase 8：集成与 lint 验证

- [X] T032 [P] 运行 `ruff check src/ tests/` 确保零 lint 错误；重点检查新增 `src/cryptotrader/llm/` 子包是否符合 TID251 模块边界规则（`llm/` 是 domain 层，不得反向引用 `nodes/`）
  - 关联：项目约束（零 lint 错误）
  - 验收：`ruff check` 输出无任何错误

- [X] T033 运行完整测试套件，分两步验证覆盖率：
  - **步骤 1（新增子包）**：`pytest tests/ -x --cov=src/cryptotrader/llm --cov-fail-under=85`，确保新增 `src/cryptotrader/llm/` 子包覆盖率 ≥ 85%
  - **步骤 2（全量）**：`pytest tests/ -x --cov=src`，确保所有原有 742 个测试通过，新增测试（T024~T031）全部通过
  - 关联：SC-006
  - 验收：`src/cryptotrader/llm/` 子包覆盖率 ≥ 85%；全量测试套件 `pytest` 退出码为 0

- [X] T034 [P] 验证 `config/models.toml` 格式文档：在 `config/models.toml` 文件头部添加注释说明 `provider_chain` 长度约束（最多 3 个）和 `api_key_env` 使用说明；验证默认 `models.toml` 中引用的所有 `model_id` 均有对应的 `[[providers]]` 条目
  - 关联：FR-001
  - 验收：`load_manifest("config/models.toml")` 无 WARNING 日志
