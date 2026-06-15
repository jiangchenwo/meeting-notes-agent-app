import { apiFetch } from './client';
import type { Transcription, TranscriptionSegment } from './types';

export const startTranscription = (noteId: number) =>
  apiFetch<{ status: string }>(`/notes/${noteId}/transcribe`, { method: 'POST' });

export const getTranscription = (noteId: number) =>
  apiFetch<Transcription>(`/notes/${noteId}/transcription`);

export const updateTranscription = (noteId: number, fullText: string) =>
  apiFetch<Transcription>(`/notes/${noteId}/transcription`, {
    method: 'PATCH',
    body: JSON.stringify({ full_text: fullText }),
  });

export const updateSegments = (noteId: number, segments: TranscriptionSegment[]) =>
  apiFetch<Transcription>(`/notes/${noteId}/transcription`, {
    method: 'PATCH',
    body: JSON.stringify({ segments }),
  });
