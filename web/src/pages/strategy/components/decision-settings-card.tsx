import { Crosshair, ShieldCheck } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export interface DecisionSettingsDraft {
  neutral_threshold: number;
  max_target_ratio: number;
  atr_stop_multiplier: number;
  reward_ratio: number;
  hitl_required: boolean;
}

interface Props {
  settings: DecisionSettingsDraft;
  onChange: (settings: DecisionSettingsDraft) => void;
}

const inputClassName =
  'h-10 w-full rounded-lg border border-input bg-background px-3 text-right font-mono text-sm tabular-nums outline-none transition focus:border-amber-500 focus:ring-2 focus:ring-amber-500/15';

export const DecisionSettingsCard = ({ settings, onChange }: Props) => {
  const { t } = useTranslation('strategy');
  const update = (key: keyof DecisionSettingsDraft, value: number | boolean) =>
    onChange({ ...settings, [key]: value });

  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border/70 bg-muted/20">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500/10 text-amber-500">
            <Crosshair className="h-4 w-4" aria-hidden="true" />
          </span>
          <div>
            <CardTitle>{t('settings.title')}</CardTitle>
            <CardDescription className="mt-1">{t('settings.description')}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-5 pt-6 sm:grid-cols-2">
        <NumberSetting
          label={t('settings.neutral_threshold')}
          hint={t('settings.neutral_threshold_hint')}
          suffix="%"
          min={0}
          max={99}
          value={Math.round(settings.neutral_threshold * 100)}
          onChange={(value) => update('neutral_threshold', value / 100)}
        />
        <NumberSetting
          label={t('settings.max_target_ratio')}
          hint={t('settings.max_target_ratio_hint')}
          suffix="%"
          min={1}
          max={100}
          value={Math.round(settings.max_target_ratio * 100)}
          onChange={(value) => update('max_target_ratio', value / 100)}
        />
        <NumberSetting
          label={t('settings.atr_stop_multiplier')}
          hint={t('settings.atr_stop_multiplier_hint')}
          suffix="×"
          min={0.1}
          step={0.1}
          value={settings.atr_stop_multiplier}
          onChange={(value) => update('atr_stop_multiplier', value)}
        />
        <NumberSetting
          label={t('settings.reward_ratio')}
          hint={t('settings.reward_ratio_hint')}
          suffix="R"
          min={0.1}
          step={0.1}
          value={settings.reward_ratio}
          onChange={(value) => update('reward_ratio', value)}
        />

        <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-border bg-muted/20 p-4 sm:col-span-2">
          <span className="flex min-w-0 items-start gap-3">
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" aria-hidden="true" />
            <span>
              <span className="block text-sm font-semibold text-foreground">{t('settings.hitl')}</span>
              <span className="mt-1 block text-xs leading-5 text-muted-foreground">{t('settings.hitl_hint')}</span>
            </span>
          </span>
          <input
            type="checkbox"
            checked={settings.hitl_required}
            onChange={(event) => update('hitl_required', event.target.checked)}
            className="h-5 w-5 shrink-0 accent-amber-500"
            aria-label={t('settings.hitl')}
          />
        </label>
      </CardContent>
    </Card>
  );
};

interface NumberSettingProps {
  label: string;
  hint: string;
  suffix: string;
  value: number;
  min: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}

const NumberSetting = ({ label, hint, suffix, value, min, max, step = 1, onChange }: NumberSettingProps) => (
  <label className="grid gap-2 rounded-xl border border-border/70 p-4">
    <span className="text-sm font-medium text-foreground">{label}</span>
    <span className="text-xs leading-5 text-muted-foreground">{hint}</span>
    <span className="relative mt-1">
      <input
        type="number"
        value={value}
        min={min}
        {...(max !== undefined ? { max } : {})}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        className={inputClassName}
        aria-label={label}
      />
      <span className="pointer-events-none absolute left-3 top-2.5 text-xs font-medium text-muted-foreground">
        {suffix}
      </span>
    </span>
  </label>
);
