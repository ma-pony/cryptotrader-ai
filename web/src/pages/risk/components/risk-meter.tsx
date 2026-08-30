import { useTranslation } from 'react-i18next';
import { Card, CardContent } from '@/components/ui/card';

interface Props {
  label: string;
  value: number | null | undefined;
  unit?: string;
  precision?: number;
}

/** Portfolio diagnostics have different scope from the connection-level risk gates. */
export const RiskMeter = ({ label, value, unit = '%', precision = 1 }: Props) => {
  const { t } = useTranslation('risk');
  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-4">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <div className="font-mono text-[22px] font-semibold leading-none tracking-tight tabular-nums">
          {value != null ? `${value.toFixed(precision)}${unit}` : '—'}
        </div>
        <span className="text-xs text-muted-foreground">{t('reporting_only')}</span>
      </CardContent>
    </Card>
  );
};
