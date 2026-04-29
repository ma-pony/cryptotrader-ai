/**
 * Debate UI-level types + re-exports of the shared agent palette.
 *
 * The canonical agent palette lives in ``@/lib/agents`` — this file only defines
 * types specific to the Debate page's rendering layer.
 */

import type { AgentKind } from '@/lib/agents';

export { AGENTS, normalizeAgentKind as normalizeKind, scoreToDirection } from '@/lib/agents';
export type { AgentKind, AgentMeta } from '@/lib/agents';

export type Direction = 'bullish' | 'bearish' | 'neutral' | 'long' | 'short' | 'hold' | 'close';

/** UI-level turn used by ``DebateTurnCard`` — narrower than the API shape. */
export interface DebateTurn {
  from: AgentKind;
  to: AgentKind | null;
  critique: string;
  dir: Direction;
  conf: number;
  move: string;
}

export interface DebateRound {
  n: number;
  turns: DebateTurn[];
}

export interface DebateScenario {
  id: string;
  pair: string;
  price: number;
  gate: { decision: 'debate' | 'skipped'; reason: string };
  initial: { kind: AgentKind; dir: Direction; conf: number }[];
  rounds: DebateRound[];
  convergence: { before: number; after: number; target: number };
  final_verdict: { action: Direction; confidence: number; scale: number; thesis: string };
}
