import { apiFetch } from './client';
import type { Summary, LMStudioStatus, LMConfig, WhisperConfig, ActionItem } from './types';

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

export const getWhisperConfig = () =>
  apiFetch<WhisperConfig>('/settings/whisper');

export const updateWhisperConfig = (cfg: Partial<Pick<WhisperConfig, 'binary_path' | 'model' | 'model_path'>>) =>
  apiFetch<WhisperConfig>('/settings/whisper', {
    method: 'PUT',
    body: JSON.stringify(cfg),
  });
