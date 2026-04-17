import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatCurrency, formatPercent, pnlClass } from '@/lib/format';
import type { Position } from '@/types/api';

interface PositionsTableProps {
  positions: Position[] | undefined;
  isLoading: boolean;
}

export const PositionsTable = ({ positions, isLoading }: PositionsTableProps) => {
  const { t } = useTranslation('dashboard');

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  const sorted = [...(positions ?? [])].sort((a, b) => Math.abs(b.unrealized_pnl) - Math.abs(a.unrealized_pnl));

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{t('positions.title', { defaultValue: '持仓' })}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {sorted.length === 0 ? (
          <p className="p-4 text-sm text-muted-foreground">{t('positions.empty', { defaultValue: '暂无持仓' })}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" role="table">
              <thead>
                <tr className="border-b text-muted-foreground text-left">
                  <th className="px-4 py-2 font-medium">{t('positions.pair', { defaultValue: '币对' })}</th>
                  <th className="px-4 py-2 font-medium">{t('positions.side', { defaultValue: '方向' })}</th>
                  <th className="px-4 py-2 font-medium text-right">{t('positions.size', { defaultValue: '数量' })}</th>
                  <th className="px-4 py-2 font-medium text-right">{t('positions.avg_price', { defaultValue: '均价' })}</th>
                  <th className="px-4 py-2 font-medium text-right">{t('positions.pnl', { defaultValue: '未实现 PnL' })}</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((pos) => (
                  <tr key={pos.pair} className="border-b last:border-b-0 hover:bg-muted/50">
                    <td className="px-4 py-2 font-medium">{pos.pair}</td>
                    <td className="px-4 py-2">
                      <Badge variant={pos.side === 'long' ? 'default' : 'destructive'}>
                        {pos.side === 'long' ? '做多' : '做空'}
                      </Badge>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">{pos.size.toFixed(4)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{formatCurrency(pos.avg_price)}</td>
                    <td className={`px-4 py-2 text-right tabular-nums ${pnlClass(pos.unrealized_pnl)}`}>
                      {formatCurrency(pos.unrealized_pnl)} ({formatPercent(pos.unrealized_pnl_pct)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
