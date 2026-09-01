import { toRuntimeDocument } from '@/hooks/use-runtime-config';
import { runtimeConfigFixture } from './runtime-config-fixture';
import type { BacktestRunStatus } from '@/types/api';

export function researchRun(overrides: Partial<BacktestRunStatus> = {}): BacktestRunStatus {
  const document = toRuntimeDocument(runtimeConfigFixture().document);
  const { base_url: _base, ...llm } = document.llm;
  return {
    run_id: 'run-first',
    status: 'completed',
    progress: 1,
    started_at: '2026-08-20T00:00:00Z',
    finished_at: '2026-08-20T01:00:00Z',
    error: null,
    incomplete_fields: [],
    params: {
      pair: 'BTC/USDT',
      start: '2024-01-01',
      end: '2024-01-02',
      interval: '1h',
      initial_equity: '1000',
      fee_rate: '0.001',
      slippage_bps: '0',
      funding_assumption: 'available_only',
      name: null,
      snapshot_run_id: null,
    },
    config_snapshot: {
      version: 1,
      revision: 7,
      updated_at: '2026-08-20T00:00:00Z',
      market_data: document.market_data,
      signals: {
        ...document.signals,
        components: document.signals.components.map((item) => ({ ...item, model_identity: {} })),
      },
      risk: document.risk,
      llm,
    },
    model_evidence: [
      {
        requested_model: 'requested-fixture',
        actual_model: null,
        actual_model_reason: 'response_identity_unavailable',
        prompt_hash: 'a'.repeat(64),
        prompt_version: 'messages-sha256-v1',
        status: 'completed',
      },
    ],
    result: {
      metrics: {
        total_return_pct: 0.00979,
        sharpe: 0.2,
        max_drawdown_pct: -0.001,
        win_rate: 1,
        fill_count: 2,
        closed_trade_count: 1,
      },
      equity_curve: [
        { ts: '2024-01-01T00:00:00Z', equity: 1000 },
        { ts: '2024-01-01T01:00:00Z', equity: 999.9 },
        { ts: '2024-01-01T02:00:00Z', equity: 1009.79 },
      ],
      decisions: [],
      decision_ids: ['decision-original'],
      fills: [
        {
          side: 'buy',
          price: '100',
          fee: { amount: '0.1', currency: 'USDT', unavailable_reason: null },
          venue_fill_id: 'fill-buy',
          occurred_at: '2024-01-01T01:00:00Z',
        },
        {
          side: 'sell',
          price: '110',
          fee: { amount: '0.11', currency: 'USDT', unavailable_reason: null },
          venue_fill_id: 'fill-sell',
          occurred_at: '2024-01-01T02:00:00Z',
        },
      ].map((fill) => ({
        ...fill,
        connection_id: 'backtest-paper',
        venue_order_id: fill.venue_fill_id,
        instrument: { venue_symbol: 'BTC/USDT', pair: 'BTC/USDT', market_type: 'spot', tradable: true, reason: null },
        amount: '1',
        realized_pnl: { amount: fill.side === 'sell' ? '10' : '0', currency: 'USDT', unavailable_reason: null },
        source: 'local_calculation',
        client_order_id: null,
      })),
      closed_trades: [
        {
          pair: 'BTC/USDT',
          opened_at: '2024-01-01T01:00:00Z',
          closed_at: '2024-01-01T02:00:00Z',
          side: 'long',
          gross_pnl: '10',
          fees: '0.21',
          funding: '0',
          net_pnl: '9.79',
          fill_ids: ['fill-buy', 'fill-sell'],
        },
      ],
      fees: '0.21',
      funding: '0',
      funding_entries: [],
      cost_assumptions: { fee_rate: '0.001', slippage_bps: '0' },
      unmodeled_costs: ['market impact and intrabar path'],
      data_coverage: { historical_news: 'unavailable', candles: { '1h': { expected: 24, available: 24, missing: 0 } } },
    },
    ...overrides,
  };
}
