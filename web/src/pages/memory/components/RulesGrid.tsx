/**
 * spec 018 / 021 — RulesGrid component
 * 4 agent × 5 maturity 状态 grid，显示 rule 数量 + 点击跳转 /memory/rules?agent=X&status=Y
 */

import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';

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
  onOpen: (agent: Agent, maturity: Maturity) => void;
}

const RuleCell = ({ agent, maturity, rules, onOpen }: CellProps) => {
  const count = rules.filter((r) => r.agent === agent && r.maturity === maturity).length;
  const interactive = count > 0;

  return (
    <button
      type="button"
      onClick={() => interactive && onOpen(agent, maturity)}
      disabled={!interactive}
      className={cn(
        'w-full rounded-md border border-border p-2 text-center transition-colors',
        interactive
          ? 'cursor-pointer hover:bg-muted/50 focus-visible:bg-muted/50 focus-visible:outline-none'
          : 'cursor-default opacity-40',
      )}
      aria-label={`${agent} · ${maturity} · ${count} rules`}
    >
      <span
        className={cn(
          'inline-flex min-w-[1.25rem] items-center justify-center rounded-full px-1.5 py-0.5 text-xs font-semibold tabular-nums',
          maturityColor[maturity],
        )}
      >
        {count}
      </span>
    </button>
  );
};

export const RulesGrid = () => {
  const { t } = useTranslation('memory');
  const navigate = useNavigate();
  const { data, isLoading } = useMemoryRules();
  const rules = data?.items ?? [];

  const handleOpen = (agent: Agent, maturity: Maturity) => {
    void navigate(`/memory/rules?agent=${agent}&status=${maturity}`);
  };

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
                  <th className="px-2 py-1 text-left text-[10px] uppercase tracking-wider text-muted-foreground/70">
                    {t('rules_grid.agent', { defaultValue: 'Agent' })}
                  </th>
                  {MATURITIES.map((m) => (
                    <th
                      key={m}
                      className="px-2 py-1 text-center text-[10px] uppercase tracking-wider text-muted-foreground/70"
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
                        <RuleCell
                          agent={agent}
                          maturity={maturity}
                          rules={rules}
                          onOpen={handleOpen}
                        />
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
              className={cn('px-1.5 py-0 text-[10px]', maturityColor[m])}
            >
              {t(`maturity.${m}`, { defaultValue: m })}
            </Badge>
          ))}
        </div>
        <div className="mt-2 text-[10px] text-muted-foreground">
          {t('rules_grid.hint', { defaultValue: '点击数字查看该分类下的全部规则' })}
        </div>
      </CardContent>
    </Card>
  );
};
