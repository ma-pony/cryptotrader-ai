import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCurrency } from '@/lib/format';
import type { Execution } from '@/types/api';

interface Props {
  execution: Execution | null;
}

export const ExecutionSection = ({ execution }: Props) => {
  const { t } = useTranslation('decisions');
  if (!execution) return null;

  const statusVariant = execution.status === 'filled' ? 'success' : execution.status === 'failed' ? 'destructive' : 'secondary';

  return (
    <section aria-label={t('detail.execution', { defaultValue: '执行' })}>
      <Card>
        <CardHeader className="p-3 pb-1">
          <CardTitle className="text-sm flex items-center gap-2">
            {t('detail.execution', { defaultValue: '执行' })}
            <Badge variant={statusVariant}>{execution.status}</Badge>
            <Badge variant="flat">{execution.exchange}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-1">
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">{t('detail.order_id', { defaultValue: '订单号' })}</span>
              <p className="font-mono text-[10px] truncate" title={execution.order_id}>{execution.order_id}</p>
            </div>
            <div>
              <span className="text-muted-foreground">{t('detail.fill_price', { defaultValue: '成交价' })}</span>
              <p className="font-medium tabular-nums">{formatCurrency(execution.fill_price)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">{t('detail.fill_size', { defaultValue: '成交量' })}</span>
              <p className="font-medium tabular-nums">{execution.fill_size}</p>
            </div>
            <div>
              <span className="text-muted-foreground">{t('detail.fee', { defaultValue: '手续费' })}</span>
              <p className="font-medium tabular-nums">{formatCurrency(execution.fee)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">{t('detail.slippage', { defaultValue: '滑点' })}</span>
              <p className="font-medium tabular-nums">{execution.slippage_bps} bps</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
};
