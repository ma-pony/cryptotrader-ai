import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { ScheduleRule } from '@/types/api';

import { useCreateRule, useUpdateRule } from '../hooks/use-rules';
import type { RuleFormValues } from './template-selector';

const schema = z.object({
  name: z.string().min(1),
  trigger_type: z.enum(['price_threshold', 'pct_change', 'candle_pattern', 'funding_rate']),
  pair: z.string().min(1),
  cooldown_minutes: z.number().int().min(1),
  price_direction: z.enum(['above', 'below']).optional(),
  price_target: z.number().optional(),
  pct_window_minutes: z.number().int().min(1).optional(),
  pct_threshold: z.number().min(0).optional(),
  candle_interval: z.string().optional(),
  candle_count: z.number().int().min(1).optional(),
  candle_direction: z.enum(['bearish', 'bullish']).optional(),
  fr_threshold: z.number().min(0).optional(),
});

type FormValues = z.infer<typeof schema>;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  rule: ScheduleRule | undefined;
  prefill: Partial<RuleFormValues> | undefined;
}

function ruleToForm(rule: ScheduleRule): FormValues {
  const p = rule.parameters;
  const base: FormValues = {
    name: rule.name,
    trigger_type: rule.trigger_type,
    pair: rule.pair,
    cooldown_minutes: rule.cooldown_minutes,
  };
  switch (rule.trigger_type) {
    case 'price_threshold':
      return {
        ...base,
        price_direction: (p['direction'] as 'above' | 'below') ?? 'below',
        price_target: (p['price'] as number) ?? 0,
      };
    case 'pct_change':
      return {
        ...base,
        pct_window_minutes: (p['window_minutes'] as number) ?? 15,
        pct_threshold: (p['threshold_pct'] as number) ?? 3,
      };
    case 'candle_pattern':
      return {
        ...base,
        candle_interval: (p['interval'] as string) ?? '1h',
        candle_count: (p['consecutive_count'] as number) ?? 3,
        candle_direction: (p['direction'] as 'bearish' | 'bullish') ?? 'bearish',
      };
    case 'funding_rate':
      return { ...base, fr_threshold: (p['threshold_pct'] as number) ?? 0.1 };
    default:
      return base;
  }
}

function prefillToForm(pf: Partial<RuleFormValues>): FormValues {
  const triggerType = pf.trigger_type ?? 'price_threshold';
  const p = pf.parameters ?? {};
  const base: FormValues = {
    name: pf.name ?? '',
    trigger_type: triggerType,
    pair: pf.pair ?? 'BTC/USDT',
    cooldown_minutes: pf.cooldown_minutes ?? 60,
  };
  switch (triggerType) {
    case 'price_threshold':
      return { ...base, price_direction: (p['direction'] as 'above' | 'below') ?? 'below', price_target: (p['price'] as number) ?? 60000 };
    case 'pct_change':
      return { ...base, pct_window_minutes: (p['window_minutes'] as number) ?? 15, pct_threshold: (p['threshold_pct'] as number) ?? 3 };
    case 'candle_pattern':
      return { ...base, candle_interval: (p['interval'] as string) ?? '1h', candle_count: (p['consecutive_count'] as number) ?? 3, candle_direction: (p['direction'] as 'bearish' | 'bullish') ?? 'bearish' };
    case 'funding_rate':
      return { ...base, fr_threshold: (p['threshold_pct'] as number) ?? 0.1 };
    default:
      return base;
  }
}

function formToPayload(values: FormValues): RuleFormValues {
  let parameters: Record<string, unknown> = {};
  switch (values.trigger_type) {
    case 'price_threshold':
      parameters = { direction: values.price_direction ?? 'below', price: values.price_target ?? 0 };
      break;
    case 'pct_change':
      parameters = { window_minutes: values.pct_window_minutes ?? 15, threshold_pct: values.pct_threshold ?? 3 };
      break;
    case 'candle_pattern':
      parameters = { interval: values.candle_interval ?? '1h', consecutive_count: values.candle_count ?? 3, direction: values.candle_direction ?? 'bearish' };
      break;
    case 'funding_rate':
      parameters = { threshold_pct: values.fr_threshold ?? 0.1 };
      break;
  }
  return { name: values.name, trigger_type: values.trigger_type, pair: values.pair, cooldown_minutes: values.cooldown_minutes, parameters };
}

const inputCls = 'block h-8 w-full rounded-md border border-input bg-background px-2 text-sm';
const labelCls = 'flex flex-col gap-1 text-xs';
const labelTextCls = 'text-muted-foreground';

export const RuleFormDialog = ({ open, onOpenChange, rule, prefill }: Props) => {
  const { t } = useTranslation('scheduler');
  const isEdit = !!rule;
  const createMutation = useCreateRule();
  const updateMutation = useUpdateRule();
  const busy = createMutation.isPending || updateMutation.isPending;

  const { register, handleSubmit, watch, reset, formState: { errors } } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { trigger_type: 'price_threshold', cooldown_minutes: 60, pair: 'BTC/USDT', price_direction: 'below' },
  });

  const triggerType = watch('trigger_type');

  useEffect(() => {
    if (!open) return;
    if (rule) {
      reset(ruleToForm(rule));
    } else if (prefill) {
      reset(prefillToForm(prefill));
    } else {
      reset({ trigger_type: 'price_threshold', cooldown_minutes: 60, pair: 'BTC/USDT', price_direction: 'below' });
    }
  }, [open, rule, prefill, reset]);

  const onSubmit = (values: FormValues) => {
    const payload = formToPayload(values);
    if (isEdit && rule) {
      updateMutation.mutate({ id: rule.id, ...payload }, { onSuccess: () => onOpenChange(false) });
    } else {
      createMutation.mutate(payload, { onSuccess: () => onOpenChange(false) });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{isEdit ? t('form.edit_title') : t('form.create_title')}</DialogTitle>
        </DialogHeader>

        <form id="rule-form" onSubmit={(e) => void handleSubmit(onSubmit)(e)} className="space-y-3 py-1">
          <label className={labelCls}>
            <span className={labelTextCls}>{t('form.name')}</span>
            <input {...register('name')} placeholder={t('form.name_placeholder')} className={inputCls} />
            {errors.name && <span className="text-xs text-destructive">{errors.name.message}</span>}
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className={labelCls}>
              <span className={labelTextCls}>{t('form.trigger_type')}</span>
              <select {...register('trigger_type')} className={inputCls}>
                {(['price_threshold', 'pct_change', 'candle_pattern', 'funding_rate'] as const).map((tt) => (
                  <option key={tt} value={tt}>{t(`trigger_type.${tt}`)}</option>
                ))}
              </select>
            </label>
            <label className={labelCls}>
              <span className={labelTextCls}>{t('form.pair')}</span>
              <input {...register('pair')} placeholder={t('form.pair_placeholder')} className={inputCls} />
            </label>
          </div>

          <label className={labelCls}>
            <span className={labelTextCls}>{t('form.cooldown_minutes')}</span>
            <input type="number" min={1} {...register('cooldown_minutes', { valueAsNumber: true })} className={inputCls} />
          </label>

          <div className="space-y-2 rounded-md border border-border p-3">
            <p className="text-xs font-medium text-muted-foreground">{t('form.parameters')}</p>

            {triggerType === 'price_threshold' && (
              <div className="grid grid-cols-2 gap-3">
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.price_threshold.direction')}</span>
                  <select {...register('price_direction')} className={inputCls}>
                    <option value="below">{t('form.price_threshold.below')}</option>
                    <option value="above">{t('form.price_threshold.above')}</option>
                  </select>
                </label>
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.price_threshold.price')}</span>
                  <input type="number" min={0} step={100} {...register('price_target', { valueAsNumber: true })} className={inputCls} />
                </label>
              </div>
            )}

            {triggerType === 'pct_change' && (
              <div className="grid grid-cols-2 gap-3">
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.pct_change.window_minutes')}</span>
                  <input type="number" min={1} {...register('pct_window_minutes', { valueAsNumber: true })} className={inputCls} />
                </label>
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.pct_change.threshold_pct')}</span>
                  <input type="number" min={0} step={0.1} {...register('pct_threshold', { valueAsNumber: true })} className={inputCls} />
                </label>
              </div>
            )}

            {triggerType === 'candle_pattern' && (
              <div className="grid grid-cols-3 gap-3">
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.candle_pattern.interval')}</span>
                  <select {...register('candle_interval')} className={inputCls}>
                    {['1m', '5m', '15m', '1h', '4h', '1d'].map((i) => <option key={i} value={i}>{i}</option>)}
                  </select>
                </label>
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.candle_pattern.consecutive_count')}</span>
                  <input type="number" min={1} max={10} {...register('candle_count', { valueAsNumber: true })} className={inputCls} />
                </label>
                <label className={labelCls}>
                  <span className={labelTextCls}>{t('form.candle_pattern.direction')}</span>
                  <select {...register('candle_direction')} className={inputCls}>
                    <option value="bearish">{t('form.candle_pattern.bearish')}</option>
                    <option value="bullish">{t('form.candle_pattern.bullish')}</option>
                  </select>
                </label>
              </div>
            )}

            {triggerType === 'funding_rate' && (
              <label className={labelCls}>
                <span className={labelTextCls}>{t('form.funding_rate.threshold_pct')}</span>
                <input type="number" min={0} step={0.01} {...register('fr_threshold', { valueAsNumber: true })} className="block h-8 w-full rounded-md border border-input bg-background px-2 text-sm" />
              </label>
            )}
          </div>
        </form>

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            {t('actions.cancel')}
          </Button>
          <Button type="submit" form="rule-form" disabled={busy}>
            {busy ? '…' : t('actions.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
