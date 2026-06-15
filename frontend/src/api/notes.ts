import { apiFetch } from './client';
import type { NoteBlock } from './types';

export const getNotes = (projectId?: number) =>
  apiFetch<NoteBlock[]>(`/notes${projectId !== undefined ? `?project_id=${projectId}` : ''}`);

export const getNote = (id: number) =>
  apiFetch<NoteBlock>(`/notes/${id}`);

export const updateNote = (
  id: number,
  body: Partial<{
    display_name: string;
    project_id: number | null;
    domain_id: number | null;
    template_id: number | null;
    color: string | null;
    sort_order: number;
  }>,
) => apiFetch<NoteBlock>(`/notes/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

export const deleteNote = (id: number) =>
  apiFetch<{ ok: boolean }>(`/notes/${id}`, { method: 'DELETE' });

export const transcribeNote = (id: number) =>
  apiFetch<{ ok: boolean; note_id: number }>(`/notes/${id}/transcribe`, { method: 'POST' });

export const searchNotes = (params: {
  q?: string;
  project_id?: number | null;
  domain_id?: number | null;
  status?: string;
}) => {
  const sp = new URLSearchParams();
  if (params.q) sp.set('q', params.q);
  if (params.project_id) sp.set('project_id', String(params.project_id));
  if (params.domain_id) sp.set('domain_id', String(params.domain_id));
  if (params.status) sp.set('status', params.status);
  return apiFetch<NoteBlock[]>(`/notes/search?${sp.toString()}`);
};

export const bulkDeleteNotes = (ids: number[]) =>
  apiFetch<{ ok: boolean; deleted: number }>('/notes/bulk-delete', {
    method: 'POST',
    body: JSON.stringify({ ids }),
  });

export const bulkUpdateNotes = (ids: number[], data: { project_id?: number | null }) =>
  apiFetch<{ ok: boolean; updated: number }>('/notes/bulk-update', {
    method: 'PATCH',
    body: JSON.stringify({ ids, ...data }),
  });
