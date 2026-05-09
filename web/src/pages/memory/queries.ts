/**
 * spec 018 — Memory page React Query hooks
 *
 * 4 hooks:
 *   useMemoryRules       — GET /api/memory/rules
 *   useMemoryCases       — GET /api/memory/cases
 *   useRecentTransitions — GET /api/memory/transitions
 *   useArchivedRules     — GET /api/memory/archived
 */

import { useQuery } from '@tanstack/react-query';
import { z } from 'zod';

import { apiClient } from '@/lib/api-client';

// ── Zod Schemas ──────────────────────────────────────────────────────────────

const PnLTrackSchema = z.object({
  successes: z.number(),
  losses: z.number(),
  total_pnl: z.number(),
});

export const RuleItemSchema = z.object({
  name: z.string(),
  agent: z.string(),
  description: z.string(),
  maturity: z.string(),
  importance: z.number(),
  access_count: z.number(),
  last_accessed_at: z.string().nullable(),
  pnl_track: PnLTrackSchema,
  regime_tags: z.array(z.string()),
  fundamental_failure_streak: z.number(),
  version: z.number(),
  manually_edited: z.boolean(),
});
export type RuleItem = z.infer<typeof RuleItemSchema>;

export const RulesListSchema = z.object({
  items: z.array(RuleItemSchema),
  total: z.number(),
});
export type RulesList = z.infer<typeof RulesListSchema>;

const TradeExecutionSchema = z.object({
  entry_price: z.number().nullable().optional(),
  stop_loss: z.number().nullable().optional(),
  take_profit: z.number().nullable().optional(),
  actual_exit_price: z.number().nullable().optional(),
  fill_status: z.string().nullable().optional(),
  hit_sl: z.boolean().nullable().optional(),
});

const IVEClassificationSchema = z.object({
  failure_type: z.string(),
  confidence: z.number(),
  reasoning: z.string(),
});

export const CaseItemSchema = z.object({
  cycle_id: z.string(),
  timestamp: z.string(),
  pair: z.string(),
  verdict_action: z.string(),
  final_pnl: z.number().nullable(),
  trade_execution: TradeExecutionSchema.nullable().optional(),
  ive_classification: IVEClassificationSchema.nullable().optional(),
});
export type CaseItem = z.infer<typeof CaseItemSchema>;

export const CasesListSchema = z.object({
  items: z.array(CaseItemSchema),
  total: z.number(),
});
export type CasesList = z.infer<typeof CasesListSchema>;

export const TransitionItemSchema = z.object({
  rule_id: z.string(),
  agent_id: z.string(),
  old_state: z.string(),
  new_state: z.string(),
  triggered_by: z.string(),
  timestamp: z.string(),
});
export type TransitionItem = z.infer<typeof TransitionItemSchema>;

export const TransitionsListSchema = z.object({
  items: z.array(TransitionItemSchema),
  total: z.number(),
});
export type TransitionsList = z.infer<typeof TransitionsListSchema>;

export const ArchivedRuleItemSchema = z.object({
  name: z.string(),
  agent: z.string(),
  archived_at: z.string().nullable(),
  fundamental_failure_streak: z.number(),
  final_pnl_track: PnLTrackSchema,
});
export type ArchivedRuleItem = z.infer<typeof ArchivedRuleItemSchema>;

export const ArchivedListSchema = z.object({
  items: z.array(ArchivedRuleItemSchema),
  total: z.number(),
});
export type ArchivedList = z.infer<typeof ArchivedListSchema>;

// ── Query param types ─────────────────────────────────────────────────────────

export interface MemoryRulesParams {
  agent?: string;
  status?: string;
}

export interface MemoryCasesParams {
  from?: string;
  to?: string;
  agent?: string;
}

export interface RecentTransitionsParams {
  since?: string;
}

// ── Hooks ─────────────────────────────────────────────────────────────────────

export const useMemoryRules = (params: MemoryRulesParams = {}) => {
  const qs = new URLSearchParams();
  if (params.agent) qs.set('agent', params.agent);
  if (params.status) qs.set('status', params.status);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-rules', params],
    queryFn: () => apiClient.get(`/api/memory/rules${query}`, RulesListSchema),
    staleTime: 30_000,
  });
};

export const useMemoryCases = (params: MemoryCasesParams = {}) => {
  const qs = new URLSearchParams();
  if (params.from) qs.set('from', params.from);
  if (params.to) qs.set('to', params.to);
  if (params.agent) qs.set('agent', params.agent);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-cases', params],
    queryFn: () => apiClient.get(`/api/memory/cases${query}`, CasesListSchema),
    staleTime: 60_000,
  });
};

export const useRecentTransitions = (params: RecentTransitionsParams = {}) => {
  const qs = new URLSearchParams();
  if (params.since) qs.set('since', params.since);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-transitions', params],
    queryFn: () => apiClient.get(`/api/memory/transitions${query}`, TransitionsListSchema),
    staleTime: 30_000,
  });
};

export const useArchivedRules = () =>
  useQuery({
    queryKey: ['memory-archived'],
    queryFn: () => apiClient.get('/api/memory/archived', ArchivedListSchema),
    staleTime: 300_000,
  });

// ── spec 019 Skill hooks ──────────────────────────────────────────────────────

export const SkillItemSchema = z.object({
  name: z.string(),
  scope: z.string(),
  version: z.string(),
  regime_tags: z.array(z.string()),
  triggers_keywords: z.array(z.string()),
  importance: z.number(),
  confidence: z.number(),
  access_count: z.number(),
  last_accessed_at: z.string().nullable(),
  manually_edited: z.boolean(),
  description: z.string(),
  inference_failed: z.boolean().optional().default(false),
});
export type SkillItem = z.infer<typeof SkillItemSchema>;

export const SkillDetailSchema = SkillItemSchema.extend({ body: z.string() });
export type SkillDetail = z.infer<typeof SkillDetailSchema>;

export const SkillsListSchema = z.object({
  items: z.array(SkillItemSchema),
  total: z.number(),
});
export type SkillsList = z.infer<typeof SkillsListSchema>;

export const SkillAccessItemSchema = z.object({
  skill_name: z.string(),
  scope: z.string(),
  access_count: z.number(),
  last_accessed_at: z.string().nullable(),
});
export type SkillAccessItem = z.infer<typeof SkillAccessItemSchema>;

export const SkillAccessListSchema = z.object({
  items: z.array(SkillAccessItemSchema),
  total: z.number(),
});
export type SkillAccessList = z.infer<typeof SkillAccessListSchema>;

const SkillProposalMetadataSchema = z.object({
  regime_tags: z.array(z.string()),
  triggers_keywords: z.array(z.string()),
  importance: z.number(),
  confidence: z.number(),
});

export const SkillProposalItemSchema = z.object({
  name: z.string(),
  draft_path: z.string(),
  created_at: z.string(),
  llm_inferred_metadata: SkillProposalMetadataSchema,
  llm_call_failed: z.boolean(),
  user_saved: z.boolean(),
});
export type SkillProposalItem = z.infer<typeof SkillProposalItemSchema>;

export const SkillProposalsListSchema = z.object({
  items: z.array(SkillProposalItemSchema),
  total: z.number(),
});
export type SkillProposalsList = z.infer<typeof SkillProposalsListSchema>;

export interface SkillsParams {
  agent?: string;
}

export interface SkillAccessParams {
  since?: string;
  agent?: string;
}

export interface SkillProposalsParams {
  since?: string;
}

export const useSkills = (params: SkillsParams = {}) => {
  const qs = new URLSearchParams();
  if (params.agent) qs.set('agent', params.agent);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-skills', params],
    queryFn: () => apiClient.get(`/api/memory/skills${query}`, SkillsListSchema),
    staleTime: 30_000,
  });
};

export const useSkillByName = (name: string) =>
  useQuery({
    queryKey: ['memory-skill', name],
    queryFn: () => apiClient.get(`/api/memory/skills/${encodeURIComponent(name)}`, SkillDetailSchema),
    staleTime: 30_000,
    enabled: Boolean(name),
  });

export const useSkillAccess = (params: SkillAccessParams = {}) => {
  const qs = new URLSearchParams();
  if (params.since) qs.set('since', params.since);
  if (params.agent) qs.set('agent', params.agent);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-skill-access', params],
    queryFn: () => apiClient.get(`/api/memory/skill-access${query}`, SkillAccessListSchema),
    staleTime: 30_000,
  });
};

export const useSkillProposals = (params: SkillProposalsParams = {}) => {
  const qs = new URLSearchParams();
  if (params.since) qs.set('since', params.since);
  const query = qs.toString() ? `?${qs.toString()}` : '';

  return useQuery({
    queryKey: ['memory-skill-proposals', params],
    queryFn: () => apiClient.get(`/api/memory/skill-proposals${query}`, SkillProposalsListSchema),
    staleTime: 300_000,
  });
};
