import { apiFetch } from './client';
import type { Project } from './types';

export const getProjects = () => apiFetch<Project[]>('/projects');

export const getProject = (id: number) => apiFetch<Project>(`/projects/${id}`);

export const createProject = (body: { name: string; description?: string }) =>
  apiFetch<Project>('/projects', { method: 'POST', body: JSON.stringify(body) });

export const updateProject = (
  id: number,
  body: Partial<{ name: string; description: string; custom_system_prompt: string; knowledge_base: string; color: string | null; icon: string | null }>,
) => apiFetch<Project>(`/projects/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

export const deleteProject = (id: number) =>
  apiFetch<{ ok: boolean }>(`/projects/${id}`, { method: 'DELETE' });
