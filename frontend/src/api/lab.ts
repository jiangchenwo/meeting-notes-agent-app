import { apiFetch } from './client';

export interface LabDomain {
  domain: string;
  transcript_length: number;
}

export interface LabFixture {
  domain: string;
  transcript: string;
  ground_truth: Record<string, unknown>;
}

export interface CritiqueDimensions {
  coverage: number;
  accuracy: number;
  specificity: number;
  structure: number;
}

export interface WorkflowStep {
  name: string;
  phase: 'extraction' | 'critique';
  status: 'done' | 'error';
  duration_ms: number;
  output: Record<string, unknown>;
  tokens?: { input: number; output: number };
  prompt?: { system: string; user: string };
  critique_score?: number;
  attempt: number;
  error?: string;
}

export interface SchemaCheckDetail {
  pass: boolean;
  missing: string[];
  type_errors: string[];
}

export interface RiskClassification {
  needs_review: boolean;
  risk_flags: string[];
  domain: string;
}

export interface ActionItem {
  task: string;
  owner: string;
  deadline: string | null;
  priority: 'high' | 'medium' | 'low';
}

export interface RougeScores {
  rouge1: number;
  rouge2: number;
  rougeL: number;
}

export interface BertScore {
  precision: number;
  recall: number;
  f1: number;
}

export interface EvalMetrics {
  agent_coverage: number;
  naive_coverage: number;
  coverage_delta: number;
  action_recall: number | null;
  hallucinations: string[];
  hallucination_count: number;
  ground_truth_facts: number;
  schema_check: Record<string, boolean>;
  rouge_vs_transcript: RougeScores | null;
  rouge_naive_vs_transcript: RougeScores | null;
  bertscore_vs_transcript: BertScore | null;
  bertscore_naive_vs_transcript: BertScore | null;
  rouge_vs_gold?: RougeScores | null;
  rouge_naive_vs_gold?: RougeScores | null;
  bertscore_vs_gold?: BertScore | null;
  bertscore_naive_vs_gold?: BertScore | null;
}

export interface WorkflowPlan {
  steps: string[];
  critique_steps: string[];
  critique_threshold: number;
  max_retries: number;
}

export interface LabRunResult {
  domain: string;
  workflow_plan: WorkflowPlan;
  steps: WorkflowStep[];
  results: Record<string, Record<string, unknown>>;
  summary_text: string;
  action_items: ActionItem[];
  suggestions_text: string;
  confidence_score: number | null;
  total_ms: number;
  naive_summary: string;
  eval: EvalMetrics;
  schema_checks?: Record<string, SchemaCheckDetail>;
  risk_classification?: RiskClassification;
  run_label?: string;
  chunked?: boolean;
  chunk_count?: number;
  case_id?: string;
  ground_truth?: Record<string, unknown>;
  // history
  history_id?: string;
  // batch-item extras
  filename?: string;
  transcript?: string;
  transcript_chars?: number;
  whisper_model?: string;
  segments?: Array<{ start: number; end: number; text: string }>;
}

export interface WorkflowOverride {
  steps: string[];
  critique_steps?: string[];
  critique_threshold?: number;
  max_retries?: number;
}

export interface EvalCaseMeta {
  id: string;
  domain: string;
  title: string;
  source: string;
  transcript_length: number;
  fact_count: number;
}

export interface EvalCaseFull extends EvalCaseMeta {
  transcript: string;
  ground_truth: Record<string, unknown>;
}

export interface LabHistoryEntry {
  id: string;
  saved_at: string;
  domain: string;
  run_label?: string;
  case_id?: string;
  confidence_score: number | null;
  total_ms: number;
  step_count: number;
  chunked: boolean;
  eval_coverage?: number;
  eval_coverage_delta?: number;
}

export interface LabTranscribeResult {
  transcript: string;
  segments: Array<{ start: number; end: number; text: string }>;
  model_used: string;
}

export interface AudioLibraryFile {
  filename: string;
  domain: string;
  size_bytes: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export const getLabDomains = () =>
  apiFetch<LabDomain[]>('/lab/domains');

export const getLabFixture = (domain: string) =>
  apiFetch<LabFixture>(`/lab/fixture/${encodeURIComponent(domain)}`);

export const runLabPipeline = (body: {
  domain: string;
  transcript?: string;
  case_id?: string;
  knowledge_base?: string;
  system_prompt?: string;
  template_prompt?: string;
  workflow_override?: WorkflowOverride;
  run_label?: string;
}) =>
  apiFetch<LabRunResult>('/lab/run', {
    method: 'POST',
    body: JSON.stringify(body),
  });

/** Transcribe an audio file in the lab (no DB writes). Uses raw fetch — FormData must not have Content-Type overridden. */
export async function labTranscribeAudio(file: File): Promise<LabTranscribeResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/lab/transcribe', { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<LabTranscribeResult>;
}

export const getAudioLibrary = () =>
  apiFetch<Record<string, AudioLibraryFile[]>>('/lab/audio-library');

export const getLabDatasets = (domain?: string) =>
  apiFetch<EvalCaseMeta[]>(`/lab/datasets${domain ? `?domain=${encodeURIComponent(domain)}` : ''}`);

export const getLabCase = (caseId: string) =>
  apiFetch<EvalCaseFull>(`/lab/datasets/${encodeURIComponent(caseId)}`);

export const getLabHistory = () =>
  apiFetch<LabHistoryEntry[]>('/lab/history');

export const getLabHistoryRun = (runId: string) =>
  apiFetch<LabRunResult>(`/lab/history/${encodeURIComponent(runId)}`);

export const deleteLabHistoryRun = (runId: string) =>
  apiFetch<void>(`/lab/history/${encodeURIComponent(runId)}`, { method: 'DELETE' });

export const runBatchItem = (body: {
  domain: string;
  filename: string;
  workflow_override?: WorkflowOverride;
}) =>
  apiFetch<LabRunResult>('/lab/batch-item', {
    method: 'POST',
    body: JSON.stringify(body),
  });
