import { useEffect, useRef, useState } from 'react';
import { getProjects } from '../api/projects';
import { getTemplates } from '../api/domains';
import { uploadAudio } from '../api/upload';
import { updateNote, transcribeNote } from '../api/notes';
import type { NoteBlock, Project, Template } from '../api/types';

interface Props {
  initialFiles?: File[];
  onClose: () => void;
  onUploaded: (notes: NoteBlock[]) => void;
}

function formatMB(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadModal({ initialFiles = [], onClose, onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>(initialFiles);
  const [projects, setProjects] = useState<Project[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [templateId, setTemplateId] = useState<number | null>(null);
  const [transcribe, setTranscribe] = useState(true);
  const [summarize, setSummarize] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    Promise.all([getProjects(), getTemplates()]).then(([p, t]) => {
      setProjects(p);
      setTemplates(t);
    });
  }, []);

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size));
      return [...prev, ...Array.from(incoming).filter((f) => !seen.has(f.name + f.size))];
    });
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    const uploaded: NoteBlock[] = [];
    for (const file of files) {
      try {
        let note = await uploadAudio(file);
        if (projectId || templateId) {
          const patched = await updateNote(note.id, {
            ...(projectId ? { project_id: projectId } : {}),
            ...(templateId ? { template_id: templateId } : {}),
          }).catch(() => null);
          if (patched) note = patched;
        }
        if (transcribe) {
          transcribeNote(note.id).catch(() => {});
        }
        uploaded.push(note);
      } catch (e) {
        console.error('Upload failed:', e);
      }
    }
    setUploading(false);
    onUploaded(uploaded);
  };

  return (
    <div className="fixed inset-0 bg-inverse-surface/40 backdrop-blur-[2px] z-[60] flex items-center justify-center p-space-4">
      <div className="bg-surface-container-lowest w-full max-w-xl rounded-xl shadow-xl overflow-hidden flex flex-col">

        {/* Header */}
        <div className="px-space-6 py-space-4 border-b border-outline-variant flex items-center justify-between">
          <h2 className="font-headline-md text-headline-md text-on-surface">Upload Recording</h2>
          <button onClick={onClose} className="text-on-surface-variant hover:text-primary transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* Body */}
        <div className="p-space-6 flex flex-col gap-space-4 overflow-y-auto max-h-[70vh]">

          {/* Drop zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-space-8 flex flex-col items-center justify-center gap-space-3 cursor-pointer transition-all ${
              dragging
                ? 'border-primary bg-primary-fixed/10'
                : 'border-outline-variant bg-surface-container-low/30 hover:bg-surface-container-low hover:border-primary'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
            onClick={() => fileRef.current?.click()}
          >
            <div className="w-12 h-12 rounded-full bg-primary-container/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-[28px] text-primary">cloud_upload</span>
            </div>
            <div className="text-center">
              <p className="font-body-md text-on-surface">
                Drag and drop recordings or{' '}
                <span className="text-primary font-bold">Browse</span>
              </p>
              <p className="font-body-sm text-on-surface-variant mt-1">MP3, WAV, M4A, OGG, WEBM — up to 2 GB</p>
            </div>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".mp3,.wav,.m4a,.ogg,.webm"
            multiple
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />

          {/* Selected files */}
          {files.length > 0 && (
            <div className="flex flex-col gap-2">
              {files.map((f, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between px-3 py-2 bg-surface-container rounded-lg border border-outline-variant/30"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="material-symbols-outlined text-[16px] text-on-surface-variant shrink-0">audio_file</span>
                    <span className="font-body-sm text-body-sm text-on-surface truncate">{f.name}</span>
                    <span className="font-label-sm text-label-sm text-on-surface-variant shrink-0">{formatMB(f.size)}</span>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setFiles((prev) => prev.filter((_, j) => j !== i)); }}
                    className="text-on-surface-variant hover:text-error transition-colors shrink-0 ml-2"
                  >
                    <span className="material-symbols-outlined text-[16px]">close</span>
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Form controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-space-4">
            {/* Project */}
            <div className="flex flex-col gap-1.5">
              <label className="font-label-md text-label-md text-on-surface-variant px-1">Project</label>
              <select
                className="bg-surface border border-outline-variant rounded-DEFAULT px-3 py-2 font-body-sm text-body-sm focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                value={projectId ?? ''}
                onChange={(e) => setProjectId(e.target.value ? Number(e.target.value) : null)}
              >
                <option value="">No Project</option>
                {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </div>

            {/* Template */}
            <div className="flex flex-col gap-1.5">
              <label className="font-label-md text-label-md text-on-surface-variant px-1">Summary Template</label>
              <select
                className="bg-surface border border-outline-variant rounded-DEFAULT px-3 py-2 font-body-sm text-body-sm focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                value={templateId ?? ''}
                onChange={(e) => setTemplateId(e.target.value ? Number(e.target.value) : null)}
              >
                <option value="">No Template</option>
                {templates.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </div>

            {/* Transcription toggle */}
            <div className="flex items-center justify-between p-3 bg-surface-container rounded-lg border border-outline-variant/30">
              <div>
                <p className="font-label-md text-label-md text-on-surface">Enable Transcription</p>
                <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">Generate full text transcript</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer shrink-0">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={transcribe}
                  onChange={(e) => setTranscribe(e.target.checked)}
                />
                <div className="relative w-9 h-5 bg-outline-variant rounded-full transition-colors peer-checked:bg-primary
                  after:content-[''] after:absolute after:top-0.5 after:left-0.5
                  after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all
                  peer-checked:after:translate-x-full" />
              </label>
            </div>

            {/* Summarization toggle */}
            <div className="flex items-center justify-between p-3 bg-surface-container rounded-lg border border-outline-variant/30">
              <div>
                <p className="font-label-md text-label-md text-on-surface">Enable Summarization</p>
                <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">Generate structured summary after transcription</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer shrink-0">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={summarize}
                  onChange={(e) => setSummarize(e.target.checked)}
                />
                <div className="relative w-9 h-5 bg-outline-variant rounded-full transition-colors peer-checked:bg-primary
                  after:content-[''] after:absolute after:top-0.5 after:left-0.5
                  after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all
                  peer-checked:after:translate-x-full" />
              </label>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-space-6 py-space-4 bg-surface-container-low border-t border-outline-variant flex justify-end gap-space-3">
          <button
            onClick={onClose}
            className="px-space-4 py-2 font-label-md text-label-md text-on-surface-variant hover:bg-surface-variant rounded-DEFAULT transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            className="px-space-6 py-2 font-label-md text-label-md bg-primary text-on-primary rounded-DEFAULT hover:opacity-90 transition-opacity shadow-sm flex items-center gap-2 disabled:opacity-50"
          >
            {uploading ? (
              <>
                <span className="material-symbols-outlined text-[18px] animate-spin">sync</span>
                Uploading…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[18px]">play_arrow</span>
                Start Processing
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
