import { apiFetch } from './client';
import type { Summary, LMStudioStatus, LMConfig, AsrStatus, AsrConfig, ActionItem } from './types';

export const startSummarization = (noteId: number) =>
  apiFetch<{ status: string }>(`/notes/${noteId}/summarize`, { method: 'POST' });

export const getSummary = (noteId: number) =>
  apiFetch<Summary>(`/notes/${noteId}/summary`);

export const updateSummary = (
  noteId: number,
  data: { summary_text?: string; action_items?: ActionItem[]; suggestions_text?: string },
) => apiFetch<Summary>(`/notes/${noteId}/summary`, { method: 'PATCH', body: JSON.stringify(data) });

export const getPromptPreview = (noteId: number) =>
  apiFetch<{ system: string; user: string }>(`/notes/${noteId}/prompt-preview`);

export const getLMStudioStatus = () =>
  apiFetch<LMStudioStatus>('/settings/lm-studio/status');

export const getLMConfig = () =>
  apiFetch<LMConfig>('/settings/llm');

export const updateLMConfig = (cfg: Partial<LMConfig>) =>
  apiFetch<LMConfig>('/settings/llm', {
    method: 'PUT',
    body: JSON.stringify(cfg),
  });

export const getAsrStatus = () =>
  apiFetch<AsrStatus>('/settings/asr/status');

export const getAsrConfig = () =>
  apiFetch<AsrConfig>('/settings/asr');

export const updateAsrConfig = (cfg: Partial<AsrConfig>) =>
  apiFetch<AsrConfig>('/settings/asr', {
    method: 'PUT',
    body: JSON.stringify(cfg),
  });
