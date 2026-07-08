export interface Domain {
  id: number;
  name: string;
  description: string;
  is_builtin: boolean;
  color: string | null;
  sort_order: number;
}

export interface Template {
  id: number;
  name: string;
  description: string;
  domain_id: number | null;
  prompt_template: string;
  output_sections: string[];
  workflow_config: string | null;
  is_builtin: boolean;
}

export type NoteStatus =
  | 'pending'
  | 'transcribing'
  | 'transcribed'
  | 'summarizing'
  | 'done'
  | 'error';

export interface NoteBlock {
  id: number;
  display_name: string;
  audio_file_name: string | null;
  audio_file_size: number | null;
  audio_url: string | null;
  project_id: number | null;
  project_name: string | null;
  project_color: string | null;
  audio_duration_ms: number | null;
  domain_id: number | null;
  domain_name: string | null;
  template_id: number | null;
  template_name: string | null;
  status: NoteStatus;
  color: string | null;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

export interface Transcription {
  note_id: number;
  full_text: string | null;
  segments: TranscriptionSegment[];
  model_used: string | null;
  language: string | null;
  diarized?: boolean;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  custom_system_prompt: string;
  knowledge_base: string;
  color: string | null;
  icon: string | null;
  note_count: number;
  top_domains: string[];
  total_size: number;
  created_at: string;
  updated_at: string;
}

export interface ProjectSpeaker {
  id: number;
  project_id: number;
  name: string;
  color: string | null;
  created_at: string;
}

export interface ActionItem {
  task: string;
  owner: string;
  deadline: string;
}

export interface Summary {
  note_id: number;
  summary_text: string | null;
  action_items: ActionItem[];
  suggestions_text: string | null;
  llm_model_used: string | null;
  generated_at: string | null;
}

export interface WorkflowRunInfo {
  id: number;
  status: string;
  current_step: string | null;
  workflow_plan: { steps: { agent: string; prompt_override: string | null }[] } | null;
  error_message: string | null;
  total_input_tokens: number | null;
  total_output_tokens: number | null;
  model_name: string | null;
  trace_id: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface WorkflowStep {
  id: number;
  step_name: string;
  status: 'pending' | 'running' | 'done' | 'error';
  duration_ms: number | null;
  critique_score: number | null;
  attempt: number;
  input_tokens: number | null;
  output_tokens: number | null;
  model_name: string | null;
  result: Record<string, unknown> | null;
  created_at: string | null;
}

export interface LMStudioStatus {
  connected: boolean;
  models: string[];
}

export interface LMConfig {
  base_url: string;
  model: string;
  max_tokens: number;
  max_response_tokens: number;
  global_system_prompt: string;
  output_mode: 'native' | 'prompted';
}

export interface TelemetryConfig {
  enabled: boolean;
  endpoint: string;
  capture_content: boolean;
}

export interface AsrStatus {
  base_url: string;
  connected: boolean;
  models_loaded: boolean;
}

export interface AsrConfig {
  base_url: string;
}
