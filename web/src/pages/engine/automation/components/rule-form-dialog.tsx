import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ApiError } from '@/lib/api-client';
import type { ScheduleRule } from '@/types/api';

import { useCreateRule, useUpdateRule } from '../hooks/use-rules';
import type { RuleFormValues } from './template-selector';

const schema = z
  .object({
    name: z.string().trim().min(1).max(255),
    trigger_type: z.enum(['price_threshold', 'pct_change', 'candle_pattern', 'funding_rate']),
    pair: z.string().trim().min(3).max(20),
    cooldown_minutes: z.number().int().min(1).max(1440),
    price_direction: z.enum(['above', 'below']).optional(),
    price_target: z.number().positive().optional(),
    pct_window_minutes: z.number().int().min(1).optional(),
    pct_threshold: z.number().positive().optional(),
    candle_interval: z.string().optional(),
    candle_count: z.number().int().min(1).optional(),
    candle_direction: z.enum(['bearish', 'bullish']).optional(),
    fr_threshold: z.number().positive().optional(),
  })
  .superRefine((values, context) => {
    const required = {
      price_threshold: ['price_direction', 'price_target'],
      pct_change: ['pct_window_minutes', 'pct_threshold'],
      candle_pattern: ['candle_interval', 'candle_count', 'candle_direction'],
      funding_rate: ['fr_threshold'],
    } as const;
    for (const name of required[values.trigger_type]) {
      if (values[name] === undefined || values[name] === '')
        context.addIssue({ code: z.ZodIssueCode.custom, path: [name], message: 'Required' });
    }
  });

type FormValues = z.infer<typeof schema>;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  rule: ScheduleRule | undefined;
  prefill: Partial<RuleFormValues> | undefined;
}

function ruleToForm(rule: ScheduleRule): FormValues {
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
        price_direction: rule.parameters.direction,
        price_target: rule.parameters.price,
      };
    case 'pct_change':
      return {
        ...base,
        pct_window_minutes: rule.parameters.window_minutes,
        pct_threshold: rule.parameters.threshold_pct,
      };
    case 'candle_pattern':
      return {
        ...base,
        candle_interval: rule.parameters.interval,
        candle_count: rule.parameters.consecutive_count,
        candle_direction: rule.parameters.direction,
      };
    case 'funding_rate':
      return { ...base, fr_threshold: rule.parameters.threshold_pct };
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
      return {
        ...base,
        price_direction: (p['direction'] as 'above' | 'below') ?? 'below',
        price_target: (p['price'] as number) ?? 60000,
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

function formToPayload(values: FormValues): RuleFormValues {
  let parameters: Record<string, unknown> = {};
  switch (values.trigger_type) {
    case 'price_threshold':
      parameters = { direction: values.price_direction, price: values.price_target };
      break;
    case 'pct_change':
      parameters = { window_minutes: values.pct_window_minutes, threshold_pct: values.pct_threshold };
      break;
    case 'candle_pattern':
      parameters = {
        interval: values.candle_interval,
        consecutive_count: values.candle_count,
        direction: values.candle_direction,
      };
      break;
    case 'funding_rate':
      parameters = { threshold_pct: values.fr_threshold };
      break;
  }
  return {
    name: values.name,
    trigger_type: values.trigger_type,
    pair: values.pair,
    cooldown_minutes: values.cooldown_minutes,
    parameters,
  };
}

const inputCls = 'configuration-control';
const labelCls = 'flex flex-col gap-1 text-sm';
const labelTextCls = 'text-muted-foreground';

export const RuleFormDialog = ({ open, onOpenChange, rule, prefill }: Props) => {
  const { t } = useTranslation('scheduler');
  const isEdit = !!rule;
  const createMutation = useCreateRule();
  const updateMutation = useUpdateRule();
  const busy = createMutation.isPending || updateMutation.isPending;

  const {
    register,
    handleSubmit,
    watch,
    reset,
    setError,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    shouldUnregister: true,
    defaultValues: {
      trigger_type: 'price_threshold',
      cooldown_minutes: 60,
      pair: 'BTC/USDT',
      price_direction: 'below',
    },
  });

  const triggerType = watch('trigger_type');
  const resetCreate = createMutation.reset;
  const resetUpdate = updateMutation.reset;

  useEffect(() => {
    if (!open) return;
    resetCreate();
    resetUpdate();
    if (rule) {
      reset(ruleToForm(rule));
    } else if (prefill) {
      reset(prefillToForm(prefill));
    } else {
      reset({ trigger_type: 'price_threshold', cooldown_minutes: 60, pair: 'BTC/USDT', price_direction: 'below' });
    }
  }, [open, rule, prefill, reset, resetCreate, resetUpdate]);

  const fieldState = (name: keyof FormValues) => ({
    id: `rule-${name}`,
    'aria-invalid': Boolean(errors[name]),
    'aria-describedby': `rule-${name}-error`,
    'aria-required': true,
    required: true,
    disabled: busy,
  });
  const fieldError = (name: keyof FormValues) => (
    <span id={`rule-${name}-error`} className="configuration-help configuration-error">
      {errors[name] ? t('form.invalid') : ''}
    </span>
  );

  const onRequestError = (error: Error) => {
    if (!(error instanceof ApiError) || !error.details?.fieldErrors) return;
    const aliases: Record<string, keyof FormValues> = {
      'parameters.price': 'price_target',
      'parameters.direction': triggerType === 'price_threshold' ? 'price_direction' : 'candle_direction',
      'parameters.window_minutes': 'pct_window_minutes',
      'parameters.threshold_pct': triggerType === 'funding_rate' ? 'fr_threshold' : 'pct_threshold',
      'parameters.interval': 'candle_interval',
      'parameters.consecutive_count': 'candle_count',
      name: 'name',
      pair: 'pair',
      cooldown_minutes: 'cooldown_minutes',
    };
    let first = true;
    for (const path of Object.keys(error.details.fieldErrors)) {
      const name = aliases[path];
      if (name) {
        setError(name, { type: 'server', message: t('form.invalid') }, { shouldFocus: first });
        first = false;
      }
    }
  };
  const onSubmit = (values: FormValues) => {
    const payload = formToPayload(values);
    if (isEdit && rule) {
      updateMutation.mutate(
        { id: rule.id, ...payload },
        { onSuccess: () => onOpenChange(false), onError: onRequestError },
      );
    } else {
      createMutation.mutate(payload, { onSuccess: () => onOpenChange(false), onError: onRequestError });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{isEdit ? t('form.edit_title') : t('form.create_title')}</DialogTitle>
          <DialogDescription>{t('form.help')}</DialogDescription>
        </DialogHeader>

        <form noValidate id="rule-form" onSubmit={(e) => void handleSubmit(onSubmit)(e)} className="space-y-3 py-1">
          <div className={labelCls}>
            <label htmlFor="rule-name" className={labelTextCls}>
              {t('form.name')}
            </label>
            <input
              {...fieldState('name')}
              {...register('name')}
              placeholder={t('form.name_placeholder')}
              className={inputCls}
            />
            {fieldError('name')}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className={labelCls}>
              <label htmlFor="rule-trigger_type" className={labelTextCls}>
                {t('form.trigger_type')}
              </label>
              <select {...fieldState('trigger_type')} {...register('trigger_type')} className={inputCls}>
                {(['price_threshold', 'pct_change', 'candle_pattern', 'funding_rate'] as const).map((tt) => (
                  <option key={tt} value={tt}>
                    {t(`trigger_type.${tt}`)}
                  </option>
                ))}
              </select>
              {fieldError('trigger_type')}
            </div>
            <div className={labelCls}>
              <label htmlFor="rule-pair" className={labelTextCls}>
                {t('form.pair')}
              </label>
              <input
                {...fieldState('pair')}
                {...register('pair')}
                placeholder={t('form.pair_placeholder')}
                className={inputCls}
              />
              {fieldError('pair')}
            </div>
          </div>

          <div className={labelCls}>
            <label htmlFor="rule-cooldown_minutes" className={labelTextCls}>
              {t('form.cooldown_minutes')}
            </label>
            <input
              type="number"
              min={1}
              max={1440}
              {...fieldState('cooldown_minutes')}
              {...register('cooldown_minutes', { valueAsNumber: true })}
              className={inputCls}
            />
            {fieldError('cooldown_minutes')}
          </div>

          <div className="space-y-2 rounded-md border border-border p-3">
            <p className="text-xs font-medium text-muted-foreground">{t('form.parameters')}</p>

            {triggerType === 'price_threshold' && (
              <div className="grid grid-cols-2 gap-3">
                <div className={labelCls}>
                  <label htmlFor="rule-price_direction" className={labelTextCls}>
                    {t('form.price_threshold.direction')}
                  </label>
                  <select {...fieldState('price_direction')} {...register('price_direction')} className={inputCls}>
                    <option value="below">{t('form.price_threshold.below')}</option>
                    <option value="above">{t('form.price_threshold.above')}</option>
                  </select>
                  {fieldError('price_direction')}
                </div>
                <div className={labelCls}>
                  <label htmlFor="rule-price_target" className={labelTextCls}>
                    {t('form.price_threshold.price')}
                  </label>
                  <input
                    type="number"
                    min={0}
                    step={100}
                    {...fieldState('price_target')}
                    {...register('price_target', { valueAsNumber: true })}
                    className={inputCls}
                  />
                  {fieldError('price_target')}
                </div>
              </div>
            )}

            {triggerType === 'pct_change' && (
              <div className="grid grid-cols-2 gap-3">
                <div className={labelCls}>
                  <label htmlFor="rule-pct_window_minutes" className={labelTextCls}>
                    {t('form.pct_change.window_minutes')}
                  </label>
                  <input
                    type="number"
                    min={1}
                    {...fieldState('pct_window_minutes')}
                    {...register('pct_window_minutes', { valueAsNumber: true })}
                    className={inputCls}
                  />
                  {fieldError('pct_window_minutes')}
                </div>
                <div className={labelCls}>
                  <label htmlFor="rule-pct_threshold" className={labelTextCls}>
                    {t('form.pct_change.threshold_pct')}
                  </label>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    {...fieldState('pct_threshold')}
                    {...register('pct_threshold', { valueAsNumber: true })}
                    className={inputCls}
                  />
                  {fieldError('pct_threshold')}
                </div>
              </div>
            )}

            {triggerType === 'candle_pattern' && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className={labelCls}>
                  <label htmlFor="rule-candle_interval" className={labelTextCls}>
                    {t('form.candle_pattern.interval')}
                  </label>
                  <select {...fieldState('candle_interval')} {...register('candle_interval')} className={inputCls}>
                    {['1m', '5m', '15m', '1h', '4h', '1d'].map((i) => (
                      <option key={i} value={i}>
                        {i}
                      </option>
                    ))}
                  </select>
                  {fieldError('candle_interval')}
                </div>
                <div className={labelCls}>
                  <label htmlFor="rule-candle_count" className={labelTextCls}>
                    {t('form.candle_pattern.consecutive_count')}
                  </label>
                  <input
                    type="number"
                    min={1}
                    {...fieldState('candle_count')}
                    {...register('candle_count', { valueAsNumber: true })}
                    className={inputCls}
                  />
                  {fieldError('candle_count')}
                </div>
                <div className={labelCls}>
                  <label htmlFor="rule-candle_direction" className={labelTextCls}>
                    {t('form.candle_pattern.direction')}
                  </label>
                  <select {...fieldState('candle_direction')} {...register('candle_direction')} className={inputCls}>
                    <option value="bearish">{t('form.candle_pattern.bearish')}</option>
                    <option value="bullish">{t('form.candle_pattern.bullish')}</option>
                  </select>
                  {fieldError('candle_direction')}
                </div>
              </div>
            )}

            {triggerType === 'funding_rate' && (
              <div className={labelCls}>
                <label htmlFor="rule-fr_threshold" className={labelTextCls}>
                  {t('form.funding_rate.threshold_pct')}
                </label>
                <input
                  type="number"
                  min={0}
                  step={0.01}
                  {...fieldState('fr_threshold')}
                  {...register('fr_threshold', { valueAsNumber: true })}
                  className={inputCls}
                />
                {fieldError('fr_threshold')}
              </div>
            )}
          </div>
          {Object.keys(errors).length ? (
            <p role="alert" className="configuration-error">
              {t('form.invalid')}
            </p>
          ) : null}
          {createMutation.isError || updateMutation.isError ? (
            <p role="alert" className="configuration-error">
              {t('form.save_failed')}
            </p>
          ) : null}
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
