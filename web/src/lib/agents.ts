/**
 * Single source of truth for agent kind → palette + direction thresholds.
 *
 * Deep-review FE-I14 (2026-04-24): three independent AgentKind + AGENTS maps existed
 * in the codebase (debate/constants, debate/components/agent-badge, decision-detail/
 * agent-analysis-grid) — drift was already present (``agent-analysis-grid`` was
 * missing the ``verdict`` kind). This module consolidates them.
 *
 * Deep-review FE-I11: two components used different score→direction thresholds
 * (±0.1 in the Debate page fallback, ±0.3 in the detail grid). Both now call
 * ``scoreToDirection()`` here, so the same AgentAnalysis.score maps to the same
 * label everywhere.
 */

export type AgentKind = 'tech' | 'chain' | 'news' | 'macro' | 'verdict' | 'other';

export interface AgentMeta {
  zh: string;
  en: string;
  color: string;
  role: string;
}

/**
 * OKLCH literals are used here rather than CSS variables so SVG / inline
 * ``style`` consumers get predictable values under both themes. Consumers that
 * want theme-reactive colors should switch to the ``--agent-*`` CSS tokens from
 * ``globals.css``.
 */
export const AGENTS: Record<AgentKind, AgentMeta> = {
  tech: { zh: '技术面', en: 'Tech', color: 'oklch(74% 0.155 220)', role: '指标 · RSI/MACD/SMA/BB' },
  chain: { zh: '链上', en: 'Chain', color: 'oklch(72% 0.150 300)', role: 'OI · 资金费率 · 鲸鱼' },
  news: { zh: '新闻', en: 'News', color: 'oklch(78% 0.145 45)', role: 'RSS · 情绪 · 社交' },
  macro: { zh: '宏观', en: 'Macro', color: 'oklch(74% 0.150 150)', role: 'Fed · DXY · FnG · ETF' },
  verdict: { zh: 'AI 决策者', en: 'Verdict', color: 'oklch(78% 0.165 70)', role: '首席裁决者' },
  other: { zh: '其他', en: 'Other', color: 'oklch(62% 0.180 295)', role: '未分类 agent' },
};

const KNOWN_KINDS = new Set<string>(['tech', 'chain', 'news', 'macro', 'verdict', 'other']);

/** Normalise any agent id string (e.g. ``tech_agent``, ``whale_tracker``) to a kind. */
export const normalizeAgentKind = (raw: string): AgentKind => {
  const low = raw.toLowerCase();
  if (KNOWN_KINDS.has(low)) return low as AgentKind;
  if (/(tech|indicator)/.test(low)) return 'tech';
  if (/(chain|onchain|whale|funding)/.test(low)) return 'chain';
  if (/(news|sentiment|social)/.test(low)) return 'news';
  if (/(macro|fed|dxy|etf|fng)/.test(low)) return 'macro';
  if (/verdict|judge/.test(low)) return 'verdict';
  return 'other';
};

/**
 * Backend ``AgentAnalysis.score`` is derived from the agent's direction label:
 * ``_serialize_analyses`` in ``src/api/routes/decisions.py`` maps
 * ``bullish → 0.6 / bearish → -0.6 / neutral → 0.0``. Anything > 0.1 (half of
 * 0.6) is therefore a real bullish signal; ≤ 0.1 is either neutral or a score
 * from some other source we shouldn't over-interpret.
 *
 * A single threshold is used everywhere to eliminate the drift bug where the
 * Debate page displayed "bullish" for scores that the Detail grid showed as
 * "neutral".
 */
export const AGENT_SCORE_DIRECTION_THRESHOLD = 0.3;

export const scoreToDirection = (
  score: number,
  threshold: number = AGENT_SCORE_DIRECTION_THRESHOLD,
): 'bullish' | 'bearish' | 'neutral' => {
  if (score > threshold) return 'bullish';
  if (score < -threshold) return 'bearish';
  return 'neutral';
};
