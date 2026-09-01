import type { z } from 'zod';
import type { AccountMoneySchema } from '@/types/api.schema';

function displayAmount(value: string) {
  return /^[+-]?0+(?:\.0+)?(?:e[+-]?\d+)?$/i.test(value) ? '0' : value;
}

export function FactReason({ reason }: { reason: string | null }) {
  return <>{reason && /[\u4e00-\u9fff]/.test(reason) ? reason : '平台未提供可核对数据，请检查历史覆盖与品种支持'}</>;
}

export function MoneyValues({ values }: { values: z.output<typeof AccountMoneySchema>[] }) {
  if (!values.length) return <span className="text-muted-foreground">暂无可核对金额</span>;
  return (
    <>
      {values.map((item, index) => (
        <span key={`${item.currency}-${index}`} className="block tabular-nums">
          {item.amount === null ? (
            <>
              未知 {item.currency === 'UNKNOWN' ? '（币种未知）' : item.currency} ·{' '}
              <FactReason reason={item.unavailable_reason} />
            </>
          ) : (
            `${displayAmount(item.amount)} ${item.currency}`
          )}
        </span>
      ))}
    </>
  );
}
