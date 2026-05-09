/**
 * spec 018 — RulesGrid component
 * 4 agent × 5 maturity 状态 grid，显示 rule 数量 + 点击展开 list
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/cn';

import type { RuleItem } from '../queries';
import { useMemoryRules } from '../queries';

const AGENTS = ['tech', 'chain', 'news', 'macro'] as const;
const MATURITIES = ['observed', 'probationary', 'active', 'deprecated', 'archived'] as const;

type Agent = (typeof AGENTS)[number];
type Maturity = (typeof MATURITIES)[number];

const maturityColor: Record<Maturity, string> = {
  observed: 'bg-muted text-muted-foreground',
  probationary: 'bg-blue-500/15 text-blue-400',
  active: 'bg-trade-long/15 text-trade-long',
  deprecated: 'bg-amber-500/15 text-amber-400',
  archived: 'bg-muted/40 text-muted-foreground line-through',
};

interface CellProps {
  agent: Agent;
  maturity: Maturity;
  rules: RuleItem[];
}

const RuleCell = ({ agent, maturity, rules }: CellProps) => {
  const [open, setOpen] = useState(false);
  const filtered = rules.filter((r) => r.agent === agent && r.maturity === maturity);
  const count = filtered.length;

  return (
    <div
      className={cn(
        'rounded-md border border-border p-2 text-center cursor-pointer transition-colors',
        count > 0 ? 'hover:bg-muted/50' : 'opacity-40 cursor-default',
      )}
      onClick={() => count > 0 && setOpen((v) => !v)}
      role={count > 0 ? 'button' : undefined}
      aria-expanded={open}
    >
      <span
        className={cn(
          'inline-flex items-center justify-center rounded-full px-1.5 py-0.5 text-xs font-semibold tabular-nums min-w-[1.25rem]',
          maturityColor[maturity],
        )}
      >
        {count}
      </span>
      {open && count > 0 ? (
        <ul className="mt-2 space-y-1 text-left">
          {filtered.map((r) => (
            <li key={r.name} className="text-[10px] truncate text-foreground/80" title={r.description}>
              {r.name}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
};

export const RulesGrid = () => {
  const { t } = useTranslation('memory');
  const { data, isLoading } = useMemoryRules();
  const rules = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">
          {t('rules_grid.title', { defaultValue: '规则状态矩阵' })}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-xs text-muted-foreground">{t('loading', { defaultValue: '加载中…' })}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-left text-muted-foreground/70 uppercase text-[10px] tracking-wider">
                    {t('rules_grid.agent', { defaultValue: 'Agent' })}
                  </th>
                  {MATURITIES.map((m) => (
                    <th
                      key={m}
                      className="px-2 py-1 text-center text-muted-foreground/70 uppercase text-[10px] tracking-wider"
                    >
                      {t(`maturity.${m}`, { defaultValue: m })}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {AGENTS.map((agent) => (
                  <tr key={agent} className="border-t border-border">
                    <td className="px-2 py-1.5 font-medium text-foreground/80">{agent}</td>
                    {MATURITIES.map((maturity) => (
                      <td key={maturity} className="px-1 py-1">
                        <RuleCell agent={agent} maturity={maturity} rules={rules} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="mt-2 flex flex-wrap gap-2">
          {MATURITIES.map((m) => (
            <Badge
              key={m}
              variant="outline"
              className={cn('text-[10px] px-1.5 py-0', maturityColor[m])}
            >
              {t(`maturity.${m}`, { defaultValue: m })}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
