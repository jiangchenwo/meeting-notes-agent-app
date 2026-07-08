import { apiFetch } from './client';
import type { WorkflowRunInfo, WorkflowStep } from './types';

export const startWorkflow = (noteId: number) =>
  apiFetch<{ status: string }>(`/notes/${noteId}/run-workflow`, { method: 'POST' });

export const getWorkflowRun = (noteId: number) =>
  apiFetch<{ run: WorkflowRunInfo | null }>(`/notes/${noteId}/workflow-run`);

export const getWorkflowSteps = (noteId: number) =>
  apiFetch<{ run_id: number | null; steps: WorkflowStep[] }>(`/notes/${noteId}/workflow-run/steps`);
