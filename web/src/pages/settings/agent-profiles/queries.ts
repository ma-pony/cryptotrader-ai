/**
 * Memory / Skill page React Query hooks — skill side only after 2026-05-13.
 *
 * Active hooks:
 *   useSkills          — GET /api/memory/skills
 *   useSkillByName     — GET /api/memory/skills/:name
 *   useSkillAccess     — GET /api/memory/skill-access
 *   useSkillProposals  — GET /api/memory/skill-proposals
 */

import { useQuery } from '@tanstack/react-query';
import { z } from 'zod';

import { apiClient } from '@/lib/api-client';

// ── Skill schemas ────────────────────────────────────────────────────────────

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
