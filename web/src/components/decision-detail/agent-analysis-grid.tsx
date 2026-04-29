import { Activity, Bot, Flame, Link2, Scale, Zap, type LucideIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Card, CardContent } from '@/components/ui/card';
import { DirChip } from '@/components/ui/dir-chip';
import { StatusPill } from '@/components/ui/status-pill';
import { AGENTS, type AgentKind, normalizeAgentKind, scoreToDirection } from '@/lib/agents';
import type { AgentAnalysis } from '@/types/api';

interface Props {
  analyses: AgentAnalysis[];
}

const AGENT_ICONS: Record<AgentKind, LucideIcon> = {
  tech: Activity,
  chain: Link2,
  news: Flame,
  macro: Scale,
  verdict: Zap,
  other: Bot,
};

const AgentCard = ({ analysis }: { analysis: AgentAnalysis }) => {
  const kind = normalizeAgentKind(analysis.name);
  const meta = AGENTS[kind];
  const Icon = AGENT_ICONS[kind];
  const dir = scoreToDirection(analysis.score);

  return (
    <Card className="overflow-hidden">
      <div
        className="flex items-center gap-2.5 border-b border-border p-3.5"
        style={{
          background: `linear-gradient(90deg, color-mix(in oklch, ${meta.color} 10%, transparent) 0%, transparent 80%)`,
        }}
      >
        <span
          className="inline-flex h-7 w-7 items-center justify-center rounded-md border"
          style={{
            color: meta.color,
            background: `color-mix(in oklch, ${meta.color} 18%, transparent)`,
            borderColor: `color-mix(in oklch, ${meta.color} 40%, transparent)`,
          }}
        >
          <Icon size={14} strokeWidth={1.8} />
        </span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold" style={{ color: meta.color }}>
            {meta.zh} · {analysis.name}
          </div>
          <div className="text-[10px] text-muted-foreground">{meta.role}</div>
        </div>
        <DirChip dir={dir} confidence={analysis.confidence} />
        {analysis.is_mock ? <StatusPill tone="default">mock</StatusPill> : null}
      </div>
      <CardContent className="flex flex-col gap-2.5 p-3.5">
        {analysis.reasoning ? (
          <p className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
            {analysis.reasoning}
          </p>
        ) : (
          <p className="text-xs italic text-muted-foreground">（无推理文本）</p>
        )}
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>
            倾向评分:{' '}
            <span
              className="font-mono font-medium"
              style={{
                color:
                  analysis.score > 0
                    ? 'var(--trade-long)'
                    : analysis.score < 0
                      ? 'var(--trade-short)'
                      : 'hsl(var(--muted-foreground))',
              }}
            >
              {analysis.score > 0 ? '+' : ''}
              {analysis.score.toFixed(2)}
            </span>
          </span>
          <span>置信度 {(analysis.confidence * 100).toFixed(0)}%</span>
        </div>
      </CardContent>
    </Card>
  );
};

export const AgentAnalysisGrid = ({ analyses }: Props) => {
  const { t } = useTranslation('decisions');
  if (analyses.length === 0) return null;

  return (
    <section aria-label={t('detail.agents', { defaultValue: '四方分析' })}>
      <div className="grid grid-cols-2 gap-3">
        {analyses.map((a) => (
          <AgentCard key={a.name} analysis={a} />
        ))}
      </div>
    </section>
  );
};
