import { AlertTriangle, CheckCircle2, Save, SlidersHorizontal } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useSaveSignalProfile, useSignalProfile } from '@/hooks/use-signal-profile';
import type { SignalProfile, SignalProfileUpdate } from '@/types/api';

import {
  ComponentWeightCard,
  type ComponentWeightDraft,
} from './components/component-weight-card';
import {
  DecisionSettingsCard,
  type DecisionSettingsDraft,
} from './components/decision-settings-card';

const COMPONENT_ACCENTS = ['#f59e0b', '#38bdf8', '#a78bfa', '#34d399', '#fb7185'];

interface StrategyDraft extends DecisionSettingsDraft {
  components: ComponentWeightDraft[];
}

const draftFromProfile = (profile: SignalProfile): StrategyDraft => {
  const configured = new Map(profile.components.map((component) => [component.component_id, component]));
  return {
    components: profile.installed_components.map((metadata) =>
      configured.get(metadata.component_id) ?? {
        component_id: metadata.component_id,
        enabled: false,
        weight: 0,
      },
    ),
    neutral_threshold: profile.neutral_threshold,
    max_target_ratio: profile.max_target_ratio,
    atr_stop_multiplier: profile.atr_stop_multiplier,
    reward_ratio: profile.reward_ratio,
    hitl_required: profile.hitl_required,
  };
};

const StrategyEditor = ({ profile }: { profile: SignalProfile }) => {
  const { t } = useTranslation('strategy');
  const saveProfile = useSaveSignalProfile();
  const [draft, setDraft] = useState<StrategyDraft>(() => draftFromProfile(profile));
  const [savedRevision, setSavedRevision] = useState(profile.revision);
  const enabledComponents = draft.components.filter((component) => component.enabled);
  const totalWeight = enabledComponents.reduce((total, component) => total + component.weight, 0);
  const validWeight = enabledComponents.length > 0 && Math.abs(totalWeight - 1) <= 1e-9;
  const validSettings =
    draft.neutral_threshold >= 0 &&
    draft.neutral_threshold < 1 &&
    draft.max_target_ratio > 0 &&
    draft.max_target_ratio <= 1 &&
    draft.atr_stop_multiplier > 0 &&
    draft.reward_ratio > 0;
  const canSave = validWeight && validSettings && !saveProfile.isPending;
  const metadataById = useMemo(
    () => new Map(profile.installed_components.map((component) => [component.component_id, component])),
    [profile.installed_components],
  );

  const updateComponent = (next: ComponentWeightDraft) => {
    setDraft((current) => ({
      ...current,
      components: current.components.map((component) =>
        component.component_id === next.component_id ? next : component,
      ),
    }));
  };

  const submit = () => {
    const payload: SignalProfileUpdate = {
      revision: savedRevision,
      components: draft.components.map((component) => ({
        ...component,
        weight: component.enabled ? component.weight : 0,
      })),
      neutral_threshold: draft.neutral_threshold,
      max_target_ratio: draft.max_target_ratio,
      atr_stop_multiplier: draft.atr_stop_multiplier,
      reward_ratio: draft.reward_ratio,
      hitl_required: draft.hitl_required,
    };
    saveProfile.mutate(payload, {
      onSuccess: (saved) => {
        setSavedRevision(saved.revision);
        setDraft(draftFromProfile(saved));
      },
    });
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={t('eyebrow')}
        title={t('title')}
        subtitle={t('subtitle')}
        actions={
          <div className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 font-mono text-xs font-semibold text-amber-500">
            Revision {savedRevision}
          </div>
        }
      />

      <section
        className="relative overflow-hidden rounded-2xl border border-border bg-card p-6"
        style={{
          backgroundImage:
            'radial-gradient(circle at 8% 20%, color-mix(in oklch, var(--amber-500) 13%, transparent), transparent 35%), linear-gradient(135deg, transparent, color-mix(in oklch, hsl(var(--card)) 88%, black))',
        }}
      >
        <div className="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-amber-500">
              <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
              {t('mixer.label')}
            </div>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">{t('mixer.description')}</p>
          </div>
          <div className="shrink-0 text-left lg:text-right">
            <div className="font-mono text-4xl font-semibold tabular-nums text-foreground">
              {Math.round(totalWeight * 100)}%
            </div>
            <div className={validWeight ? 'text-xs text-trade-long' : 'text-xs text-amber-500'}>
              {t('mixer.total', { value: Math.round(totalWeight * 100) })}
            </div>
          </div>
        </div>
        <div className="relative mt-6 flex h-3 overflow-hidden rounded-full border border-border bg-muted/70">
          {enabledComponents.map((component, index) => (
            <span
              key={component.component_id}
              className="h-full transition-[width] duration-300"
              style={{
                width: `${component.weight * 100}%`,
                backgroundColor: COMPONENT_ACCENTS[index % COMPONENT_ACCENTS.length],
              }}
              title={`${component.component_id} ${Math.round(component.weight * 100)}%`}
            />
          ))}
        </div>
        {!validWeight ? (
          <div className="relative mt-3 flex items-center gap-2 text-xs text-amber-500" role="alert">
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
            {enabledComponents.length === 0 ? t('mixer.enable_one') : t('mixer.must_total')}
          </div>
        ) : null}
      </section>

      <section>
        <div className="mb-3 flex items-end justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-foreground">{t('components.title')}</h2>
            <p className="mt-1 text-xs text-muted-foreground">{t('components.description')}</p>
          </div>
          <span className="font-mono text-[11px] text-muted-foreground">
            {enabledComponents.length}/{draft.components.length} {t('components.active')}
          </span>
        </div>
        {draft.components.length > 0 ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {draft.components.map((component, index) => {
              const metadata = metadataById.get(component.component_id);
              return (
                <ComponentWeightCard
                  key={component.component_id}
                  component={component}
                  displayName={metadata?.display_name ?? component.component_id}
                  description={metadata?.description ?? t('components.custom_description')}
                  accent={COMPONENT_ACCENTS[index % COMPONENT_ACCENTS.length] ?? '#f59e0b'}
                  onChange={updateComponent}
                />
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
            {t('components.empty')}
          </div>
        )}
      </section>

      <DecisionSettingsCard
        settings={draft}
        onChange={(settings) => setDraft((current) => ({ ...current, ...settings }))}
      />

      <div className="sticky bottom-4 z-10 flex flex-col gap-3 rounded-xl border border-border bg-card/95 p-4 shadow-xl backdrop-blur sm:flex-row sm:items-center sm:justify-between">
        <div className="min-h-5 text-xs">
          {saveProfile.isSuccess ? (
            <span className="flex items-center gap-2 text-trade-long" role="status">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
              {t('save.success')}
            </span>
          ) : saveProfile.isError ? (
            <span className="flex items-center gap-2 text-trade-short" role="alert">
              <AlertTriangle className="h-4 w-4" aria-hidden="true" />
              {t('save.error')}
            </span>
          ) : (
            <span className="text-muted-foreground">{t('save.next_cycle')}</span>
          )}
        </div>
        <Button size="lg" onClick={submit} disabled={!canSave} className="shadow-glow-amber">
          <Save className="h-4 w-4" aria-hidden="true" />
          {saveProfile.isPending ? t('save.saving') : t('save.button')}
        </Button>
      </div>
    </div>
  );
};

const StrategyPage = () => {
  const { t } = useTranslation('strategy');
  const profile = useSignalProfile();

  return (
    <PageBoundary
      loading={profile.isLoading}
      isError={profile.isError}
      onRetry={() => void profile.refetch()}
      errorTitle={t('load.error_title')}
      errorDescription={t('load.error_description')}
    >
      {profile.data ? <StrategyEditor profile={profile.data} /> : null}
    </PageBoundary>
  );
};

export default StrategyPage;
