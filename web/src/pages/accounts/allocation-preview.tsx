import { useTranslation } from 'react-i18next';
export function AllocationPreview({
  equity,
  targetExposure,
  allocations,
}: {
  equity: number | '';
  targetExposure: number | '';
  allocations: { connectionId: string; label: string; weight: number | '' }[];
}) {
  const { i18n } = useTranslation();
  return (
    <dl className="configuration-allocation-preview">
      {allocations.map((item) => (
        <div key={item.connectionId}>
          <dt>{item.label}</dt>
          <dd className="font-mono tabular-nums">
            {equity === '' || targetExposure === '' || item.weight === ''
              ? '—'
              : new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 2 }).format(
                  (equity * targetExposure * item.weight) / 100,
                ) + ' USDT'}
          </dd>
        </div>
      ))}
    </dl>
  );
}
