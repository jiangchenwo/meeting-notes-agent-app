import { apiFetch } from './client';
import type { Domain, Template } from './types';

// ── Domains ──────────────────────────────────────────────────────────────────

export const getDomains = () => apiFetch<Domain[]>('/domains');

export const createDomain = (body: { name: string; description?: string }) =>
  apiFetch<Domain>('/domains', { method: 'POST', body: JSON.stringify(body) });

export const updateDomain = (
  id: number,
  body: Partial<{ name: string; description: string; color: string | null; sort_order: number }>,
) => apiFetch<Domain>(`/domains/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

export const deleteDomain = (id: number) =>
  apiFetch<{ ok: boolean }>(`/domains/${id}`, { method: 'DELETE' });

// ── Templates ─────────────────────────────────────────────────────────────────

export const getTemplates = (domainId?: number) =>
  apiFetch<Template[]>(`/templates${domainId !== undefined ? `?domain_id=${domainId}` : ''}`);

export const getTemplate = (id: number) => apiFetch<Template>(`/templates/${id}`);

export const createTemplate = (body: {
  name: string;
  description?: string;
  domain_id?: number | null;
  prompt_template?: string;
  output_sections?: string[];
  workflow_config?: string | null;
}) => apiFetch<Template>('/templates', { method: 'POST', body: JSON.stringify(body) });

export const updateTemplate = (
  id: number,
  body: Partial<{
    name: string;
    description: string;
    domain_id: number | null;
    prompt_template: string;
    output_sections: string[];
    workflow_config: string | null;
  }>,
) => apiFetch<Template>(`/templates/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

export const deleteTemplate = (id: number) =>
  apiFetch<{ ok: boolean }>(`/templates/${id}`, { method: 'DELETE' });
