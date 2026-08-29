import { RuntimeConfigSchema } from '@/types/api.schema';

export const runtimeConfigFixture = (overrides: Record<string, unknown> = {}) => RuntimeConfigSchema.parse({
  revision: 1, updated_at: '2026-08-28T00:00:00Z', setup_required: false, apply_status: 'applied', applied_revision: 1, apply_error: null,
  document: {
    system: { active: true }, security: { enabled: false, access_credential_configured: false, access_credential_updated_at: null }, market_data: { source_id: 'binance', parameters: [] },
    llm: { base_url: 'https://gateway.example', streaming_models: [], default_temperature: 0.2, timeout: 30, prompt_caching: false, retry: { max_attempts: 1, retry_base_delay_s: 1, retry_backoff_factor: 2, retry_jitter: false }, model_costs: [], models: { analysis: 'analysis', debate: 'debate', committee_summary: 'summary', tech_agent: 'tech', chain_agent: 'chain', news_agent: 'news', macro_agent: 'macro', fallback: 'fallback', timeout_seconds: 30 }, gateway_credential_configured: false, gateway_credential_updated_at: null },
    signals: { components: [], neutral_threshold: 0.2, max_target_ratio: 1, atr_stop_multiplier: 2, reward_ratio: 2, hitl_required: false },
    risk: { max_stop_loss_pct: 0.1, position: { max_single_pct: 0.2, max_total_exposure_pct: 0.8, max_margin_used_pct: 0.5, max_correlated_positions: 2, max_same_direction_positions: 2 }, loss: { max_daily_loss_pct: 0.05, max_drawdown_pct: 0.15, max_cvar_95: 0.1, cvar_min_returns: 20 }, cooldown: { same_pair_minutes: 30, post_loss_minutes: 60 }, volatility: { flash_crash_threshold: 0.1, funding_rate_threshold: 0.01, flash_crash_lookback: 10 }, exchange: { max_api_latency_ms: 1000, health_check_interval_s: 60 }, rate_limit: { max_trades_per_hour: 10, max_trades_per_day: 30 } },
    execution: { connections: [], books: [], allocation_policy: 'weighted', live_order_execution_enabled: false }, hitl: { approval_ttl_minutes: 15 }, scheduler: { enabled: false, pairs: ['BTC/USDT'], interval_minutes: 15, daily_summary_hour: 8 }, triggers: { enabled: false, max_rules: 10, ws_reconnect_max_s: 30, funding_rate_poll_interval_minutes: 15 }, notifications: { webhook_url: '', enabled: false, webhook_timeout: 10, events: [], telegram: { enabled: false, chat_id: '' } }, infrastructure: { redis_url: 'redis://localhost:6379/0' }, observability: { otlp_endpoint: '' },
  },
  ...overrides,
});
