import { apiFetch } from './client';
import type { ProjectSpeaker } from './types';

export const getProjectSpeakers = (projectId: number) =>
  apiFetch<ProjectSpeaker[]>(`/projects/${projectId}/speakers`);

export const createProjectSpeaker = (
  projectId: number,
  body: { name: string; color?: string | null },
) =>
  apiFetch<ProjectSpeaker>(`/projects/${projectId}/speakers`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const updateProjectSpeaker = (
  projectId: number,
  speakerId: number,
  body: Partial<{ name: string; color: string | null }>,
) =>
  apiFetch<ProjectSpeaker>(`/projects/${projectId}/speakers/${speakerId}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  });

export const deleteProjectSpeaker = (projectId: number, speakerId: number) =>
  apiFetch<{ ok: boolean }>(`/projects/${projectId}/speakers/${speakerId}`, {
    method: 'DELETE',
  });
