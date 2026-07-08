import { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import StatusBadge from '../components/StatusBadge';
import { getNote, updateNote } from '../api/notes';
import { getProjects } from '../api/projects';
import { getProjectSpeakers, createProjectSpeaker } from '../api/speakers';
import { getDomains, getTemplates } from '../api/domains';
import { startTranscription, getTranscription, updateTranscription, updateSegments } from '../api/transcribe';
import { getSummary, getPromptPreview, getLMStudioStatus, updateSummary } from '../api/summarize';
import { startWorkflow, getWorkflowRun, getWorkflowSteps } from '../api/workflow';
import { exportNote, copyText } from '../api/export';
import Breadcrumb from '../components/Breadcrumb';
import Select from '../components/Select';
import type { NoteBlock, Project, ProjectSpeaker, Domain, Template, Transcription, Summary, LMStudioStatus, WorkflowRunInfo, WorkflowStep } from '../api/types';
import { speakerColor } from '../lib/speakerColor';

function fmtSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function fmtFileSize(bytes: number | null): string {
  if (!bytes) return '—';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

type MainTab = 'transcription' | 'full' | 'summary' | 'action_items' | 'suggestions';

function CopyButton({ text, title = 'Copy' }: { text: string; title?: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await copyText(text).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      onClick={handleCopy}
      title={title}
      className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors"
    >
      <span className="material-symbols-outlined text-[14px]">{copied ? 'check' : 'content_copy'}</span>
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

export default function AudioManagement() {
  const { noteId } = useParams<{ noteId: string }>();
  const id = Number(noteId);

  const [note, setNote] = useState<NoteBlock | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [domains, setDomains] = useState<Domain[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [activeTab, setActiveTab] = useState<MainTab>('transcription');
  const [loading, setLoading] = useState(true);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [workflowRun, setWorkflowRun] = useState<WorkflowRunInfo | null>(null);
  const [workflowSteps, setWorkflowSteps] = useState<WorkflowStep[]>([]);
  const [lmStatus, setLmStatus] = useState<LMStudioStatus | null>(null);
  const [doneTasks, setDoneTasks] = useState<Record<number, boolean>>({});
  const [promptPreview, setPromptPreview] = useState<{ system: string; user: string } | null>(null);
  const [promptPreviewLoading, setPromptPreviewLoading] = useState(false);

  // Inline transcription editing
  const [editingTranscript, setEditingTranscript] = useState(false);
  const [draftTranscript, setDraftTranscript] = useState('');
  const [savingTranscript, setSavingTranscript] = useState(false);

  // Segment editing
  const [editingSegments, setEditingSegments] = useState(false);
  const [draftSegments, setDraftSegments] = useState<{ start: number; end: number; text: string; speaker?: string | null }[]>([]);
  const [savingSegments, setSavingSegments] = useState(false);
  // Bulk-rename rows for the Speakers panel: id is the label at edit-start (stable key)
  const [speakerRows, setSpeakerRows] = useState<{ id: string; name: string }[]>([]);

  // Per-project speaker roster (shared vocabulary for consistent labels)
  const [roster, setRoster] = useState<ProjectSpeaker[]>([]);

  // Summary field editing
  const [editingSummary, setEditingSummary] = useState(false);
  const [draftSummaryText, setDraftSummaryText] = useState('');
  const [editingSuggestions, setEditingSuggestions] = useState(false);
  const [draftSuggestionsText, setDraftSuggestionsText] = useState('');
  const [editingActionItems, setEditingActionItems] = useState(false);
  const [draftActionItems, setDraftActionItems] = useState<{ task: string; owner: string; deadline: string }[]>([]);
  const [savingSummary, setSavingSummary] = useState(false);

  // audio player
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!id) return;
    Promise.all([getNote(id), getProjects(), getDomains(), getTemplates()])
      .then(([n, p, d, t]) => {
        setNote(n);
        setProjects(p);
        setDomains(d);
        setTemplates(t);
        const loads: Promise<unknown>[] = [];
        if (n.status === 'transcribed' || n.status === 'done') {
          loads.push(getTranscription(id).then(setTranscription));
        }
        if (n.status === 'done') {
          loads.push(getSummary(id).then(setSummary));
        }
        return Promise.all(loads);
      })
      .finally(() => setLoading(false));
    getLMStudioStatus().then(setLmStatus).catch(() => setLmStatus({ connected: false, models: [] }));
  }, [id]);

  useEffect(() => {
    if (!note?.domain_id) return;
    getTemplates(note.domain_id).then(setTemplates);
  }, [note?.domain_id]);

  // Load the project's speaker roster so rename/reassign fields can autocomplete
  useEffect(() => {
    const pid = note?.project_id;
    if (!pid) { setRoster([]); return; }
    getProjectSpeakers(pid).then(setRoster).catch(() => setRoster([]));
  }, [note?.project_id]);

  // Poll while transcribing
  useEffect(() => {
    if (note?.status !== 'transcribing') return;
    const interval = setInterval(async () => {
      const updated = await getNote(id).catch(() => null);
      if (!updated) return;
      setNote(updated);
      if (updated.status !== 'transcribing') {
        clearInterval(interval);
        if (updated.status === 'transcribed' || updated.status === 'done') {
          getTranscription(id).then(setTranscription);
        }
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [note?.status, id]);

  // Poll while summarizing: note status + per-step workflow progress
  useEffect(() => {
    if (note?.status !== 'summarizing') return;
    const interval = setInterval(async () => {
      const [updated, runResp, stepsResp] = await Promise.all([
        getNote(id).catch(() => null),
        getWorkflowRun(id).catch(() => null),
        getWorkflowSteps(id).catch(() => null),
      ]);
      if (runResp) setWorkflowRun(runResp.run);
      if (stepsResp) setWorkflowSteps(stepsResp.steps);
      if (!updated) return;
      setNote(updated);
      if (updated.status !== 'summarizing') {
        clearInterval(interval);
        if (updated.status === 'done') {
          getSummary(id).then((s) => {
            setSummary(s);
            setActiveTab('summary');
          });
        } else if (updated.status === 'error') {
          setSummarizeError(runResp?.run?.error_message || 'Workflow failed — check that LM Studio is running.');
        }
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [note?.status, id]);

  const handleTranscribe = async () => {
    if (!note) return;
    setTranscribeError(null);
    try {
      await startTranscription(id, true);
      setNote((prev) => prev ? { ...prev, status: 'transcribing' } : prev);
    } catch (err) {
      setTranscribeError(err instanceof Error ? err.message : 'Failed to start transcription');
    }
  };

  const handleSummarize = async () => {
    if (!note) return;
    setSummarizeError(null);
    setWorkflowRun(null);
    setWorkflowSteps([]);
    try {
      await startWorkflow(id);
      setNote((prev) => prev ? { ...prev, status: 'summarizing' } : prev);
      setActiveTab('summary');
    } catch (err) {
      setSummarizeError(err instanceof Error ? err.message : 'Failed to start summarization');
    }
  };

  const handleShowPromptPreview = async () => {
    setPromptPreviewLoading(true);
    try {
      const preview = await getPromptPreview(id);
      setPromptPreview(preview);
    } catch {
      // no transcription yet
    } finally {
      setPromptPreviewLoading(false);
    }
  };

  const handleConfigChange = async (field: 'project_id' | 'domain_id' | 'template_id', value: number | null) => {
    if (!note) return;
    const updated = await updateNote(id, { [field]: value }).catch(() => null);
    if (updated) setNote(updated);
  };

  const startEditName = () => {
    if (!note) return;
    setDraftName(note.display_name);
    setEditingName(true);
  };

  const commitName = async () => {
    setEditingName(false);
    const trimmed = draftName.trim();
    if (!trimmed || trimmed === note?.display_name) return;
    const updated = await updateNote(id, { display_name: trimmed }).catch(() => null);
    if (updated) setNote(updated);
  };

  const handleNameKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.currentTarget.blur(); }
    else if (e.key === 'Escape') { setEditingName(false); }
  };

  const startEditTranscript = () => {
    setDraftTranscript(transcription?.full_text ?? '');
    setEditingTranscript(true);
  };

  const cancelEditTranscript = () => {
    setEditingTranscript(false);
    setDraftTranscript('');
  };

  const commitEditTranscript = async () => {
    setSavingTranscript(true);
    try {
      const updated = await updateTranscription(id, draftTranscript);
      setTranscription(updated);
      setEditingTranscript(false);
    } catch {
      // keep editing open on error
    } finally {
      setSavingTranscript(false);
    }
  };

  const startEditSegments = () => {
    const segs = transcription?.segments.map((s) => ({ ...s })) ?? [];
    setDraftSegments(segs);
    const distinct = Array.from(new Set(segs.map((s) => s.speaker).filter((x): x is string => !!x)));
    setSpeakerRows(distinct.map((name) => ({ id: name, name })));
    setEditingSegments(true);
  };

  const cancelEditSegments = () => { setEditingSegments(false); setDraftSegments([]); setSpeakerRows([]); };

  // Bulk rename: change every segment that currently carries this speaker's name
  const renameSpeakerRow = (rowId: string, newName: string) => {
    const row = speakerRows.find((r) => r.id === rowId);
    const oldName = row ? row.name : '';
    setSpeakerRows((rows) => rows.map((r) => (r.id === rowId ? { ...r, name: newName } : r)));
    setDraftSegments((segs) => segs.map((s) => (s.speaker === oldName ? { ...s, speaker: newName } : s)));
  };

  // Per-segment reassignment: fix a single mis-attributed line
  const reassignSegmentSpeaker = (index: number, value: string) => {
    setDraftSegments((segs) => segs.map((s, i) => (i === index ? { ...s, speaker: value === '' ? null : value } : s)));
  };

  // Autocomplete pool: project roster names + speakers already present in this transcript
  const speakerSuggestions = Array.from(new Set([
    ...roster.map((r) => r.name),
    ...draftSegments.map((s) => s.speaker).filter((x): x is string => !!x),
  ]));

  const commitEditSegments = async () => {
    setSavingSegments(true);
    try {
      const updated = await updateSegments(id, draftSegments);
      setTranscription(updated);
      // Persist any newly-typed names to the project roster for cross-meeting consistency
      const pid = note?.project_id;
      if (pid) {
        const names = Array.from(new Set(draftSegments.map((s) => s.speaker).filter((x): x is string => !!x)));
        const known = new Set(roster.map((r) => r.name.toLowerCase()));
        const toAdd = names.filter((n) => !known.has(n.toLowerCase()));
        if (toAdd.length) {
          await Promise.all(toAdd.map((name) => createProjectSpeaker(pid, { name }).catch(() => null)));
          getProjectSpeakers(pid).then(setRoster).catch(() => {});
        }
      }
      setEditingSegments(false);
    } catch {} finally { setSavingSegments(false); }
  };

  const startEditSummary = () => { setDraftSummaryText(summary?.summary_text ?? ''); setEditingSummary(true); };
  const cancelEditSummary = () => setEditingSummary(false);
  const commitEditSummary = async () => {
    setSavingSummary(true);
    try {
      const updated = await updateSummary(id, { summary_text: draftSummaryText });
      setSummary(updated);
      setEditingSummary(false);
    } catch {} finally { setSavingSummary(false); }
  };

  const startEditSuggestions = () => { setDraftSuggestionsText(summary?.suggestions_text ?? ''); setEditingSuggestions(true); };
  const cancelEditSuggestions = () => setEditingSuggestions(false);
  const commitEditSuggestions = async () => {
    setSavingSummary(true);
    try {
      const updated = await updateSummary(id, { suggestions_text: draftSuggestionsText });
      setSummary(updated);
      setEditingSuggestions(false);
    } catch {} finally { setSavingSummary(false); }
  };

  const startEditActionItems = () => {
    setDraftActionItems(summary?.action_items.map((a) => ({ ...a })) ?? []);
    setEditingActionItems(true);
  };
  const cancelEditActionItems = () => setEditingActionItems(false);
  const commitEditActionItems = async () => {
    setSavingSummary(true);
    try {
      const updated = await updateSummary(id, { action_items: draftActionItems });
      setSummary(updated);
      setEditingActionItems(false);
    } catch {} finally { setSavingSummary(false); }
  };

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) { el.pause(); } else { el.play().catch(() => {}); }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = audioRef.current;
    if (!el || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    el.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-on-surface-variant font-body-md text-body-md">
        Loading…
      </div>
    );
  }

  if (!note) {
    return (
      <div className="flex-1 flex items-center justify-center text-on-surface-variant font-body-md text-body-md">
        Note not found.{' '}
        <Link to="/" className="text-primary ml-1 hover:underline">Back to home</Link>
      </div>
    );
  }

  const canTranscribe = note.status !== 'transcribing' && note.status !== 'summarizing';
  const isTranscribing = note.status === 'transcribing';
  const hasTranscript = note.status === 'transcribed' || note.status === 'done' || note.status === 'summarizing';
  const isSummarizing = note.status === 'summarizing';
  const hasSummary = !!summary?.summary_text;
  const canSummarize = hasTranscript && !isSummarizing;

  const filteredTemplates = note.domain_id
    ? templates.filter((t) => t.domain_id === note.domain_id || t.domain_id === null)
    : templates;

  const TABS: { key: MainTab; label: string; show: boolean }[] = hasSummary
    ? [
        { key: 'summary', label: 'Summary', show: true },
        { key: 'action_items', label: 'Action Items', show: true },
        { key: 'suggestions', label: 'Suggestions', show: true },
        { key: 'transcription', label: 'Segments', show: true },
        { key: 'full', label: 'Full Text', show: true },
      ]
    : [
        { key: 'transcription', label: 'Segments', show: true },
        { key: 'full', label: 'Full Text', show: true },
        { key: 'summary', label: 'Summary', show: isSummarizing },
      ];

  const summaryText = summary?.summary_text ?? '';
  const actionItemsText = summary?.action_items.map((a) => `- [ ] ${a.task}${a.owner ? ` (${a.owner}` : ''}${a.deadline ? `, ${a.deadline}` : ''}${a.owner || a.deadline ? ')' : ''}`).join('\n') ?? '';
  const suggestionsText = summary?.suggestions_text ?? '';

  return (
    <div className="flex-1 overflow-auto p-margin-mobile md:px-margin-desktop md:py-space-6">
      <Breadcrumb items={[
        { label: 'Home', to: '/' },
        { label: 'Projects', to: '/projects' },
        ...(note.project_name && note.project_id
          ? [{ label: note.project_name, to: `/projects/${note.project_id}` }]
          : []),
        { label: note.display_name },
      ]} />

      <div className="max-w-container-max mx-auto flex flex-col md:flex-row gap-space-4">

        {/* Left: main content panel */}
        <div className="w-full md:w-2/3 flex flex-col bg-surface-container-lowest rounded flex-shrink-0 min-h-[600px] overflow-hidden">
          {/* Header */}
          <div className="p-space-4 border-b border-outline-variant flex justify-between items-start bg-surface-container-lowest">
            <div className="min-w-0 pr-4 flex-1">
              {editingName ? (
                <input
                  autoFocus
                  className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface w-full bg-surface-container-low border border-primary rounded px-2 py-0.5 outline-none focus:ring-1 focus:ring-primary"
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  onBlur={commitName}
                  onKeyDown={handleNameKeyDown}
                />
              ) : (
                <h2
                  className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface truncate cursor-text hover:bg-surface-container-low rounded px-2 py-0.5 -mx-2 -my-0.5 transition-colors group flex items-center gap-2"
                  onClick={startEditName}
                  title="Click to rename"
                >
                  {note.display_name}
                  <span className="material-symbols-outlined text-[16px] text-outline-variant opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">edit</span>
                </h2>
              )}
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-1 flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">audio_file</span>
                {note.audio_file_name ?? 'audio'} · {fmtFileSize(note.audio_file_size)}
              </p>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <StatusBadge status={note.status} />
            </div>
          </div>

          {/* Audio Player */}
          {note.audio_url && (
            <div className="px-space-4 py-space-3 bg-surface-container-low/50 border-b border-outline-variant flex items-center gap-space-4">
              <audio
                ref={audioRef}
                src={note.audio_url}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime ?? 0)}
                onLoadedMetadata={() => setDuration(audioRef.current?.duration ?? 0)}
                onEnded={() => setPlaying(false)}
              />
              <button
                onClick={togglePlay}
                className="w-10 h-10 rounded-full bg-primary-container text-on-primary-container flex items-center justify-center flex-shrink-0 hover:bg-primary hover:text-on-primary transition-colors"
              >
                <span className="material-symbols-outlined icon-fill">
                  {playing ? 'pause' : 'play_arrow'}
                </span>
              </button>
              <div className="flex-1 h-2 bg-outline-variant/30 rounded-full overflow-hidden cursor-pointer" onClick={handleSeek}>
                <div
                  className="h-full bg-primary-container transition-all"
                  style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }}
                />
              </div>
              <span className="font-label-sm text-label-sm text-on-surface-variant whitespace-nowrap">
                {fmtSeconds(currentTime)} / {duration ? fmtSeconds(duration) : '--:--'}
              </span>
            </div>
          )}

          {/* Tabs */}
          <div className="flex border-b border-outline-variant px-space-4 bg-surface-container-lowest font-label-md text-label-md overflow-x-auto">
            {TABS.filter((t) => t.show).map((t) => (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                className={`py-space-3 px-space-4 border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === t.key
                    ? 'border-primary text-primary font-bold'
                    : 'border-transparent text-on-surface-variant hover:text-primary'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto bg-surface-container-lowest p-space-4">

            {/* Transcribing spinner */}
            {isTranscribing && (activeTab === 'transcription' || activeTab === 'full') && (
              <div className="flex flex-col items-center justify-center h-full gap-space-4 text-on-surface-variant">
                <span className="material-symbols-outlined text-[48px] animate-spin text-primary">sync</span>
                <p className="font-body-md text-body-md">Transcribing with Whisper…</p>
                <p className="font-body-sm text-body-sm">This may take a minute depending on file length.</p>
              </div>
            )}

            {/* No transcript yet */}
            {!isTranscribing && !hasTranscript && (activeTab === 'transcription' || activeTab === 'full') && (
              <div className="flex flex-col items-center justify-center h-full gap-space-3 text-on-surface-variant">
                <span className="material-symbols-outlined text-[48px]">subtitles_off</span>
                <p className="font-body-md text-body-md">No transcript yet.</p>
                <p className="font-body-sm text-body-sm">Click <strong>Transcribe</strong> in the panel to begin.</p>
              </div>
            )}

            {/* Segments tab */}
            {hasTranscript && activeTab === 'transcription' && (
              <div>
                <div className="flex items-center justify-between mb-space-3 gap-2">
                  <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Segments</p>
                  <div className="flex items-center gap-2">
                    {!editingSegments ? (
                      <button
                        onClick={startEditSegments}
                        className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors"
                      >
                        <span className="material-symbols-outlined text-[14px]">edit</span>
                        Edit
                      </button>
                    ) : (
                      <div className="flex gap-2">
                        <button onClick={cancelEditSegments} className="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">Cancel</button>
                        <button onClick={commitEditSegments} disabled={savingSegments} className="font-label-sm text-label-sm bg-primary text-on-primary rounded px-3 py-1 hover:opacity-90 disabled:opacity-50">
                          {savingSegments ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                {/* Speakers panel — rename a speaker once to relabel every line they said */}
                {editingSegments && speakerRows.length > 0 && (
                  <div className="mb-space-3 p-space-3 rounded border border-outline-variant/50 bg-surface-container-low/30">
                    <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-space-2">Speakers</p>
                    <div className="space-y-space-2">
                      {speakerRows.map((row) => (
                        <div key={row.id} className="flex items-center gap-space-2">
                          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: speakerColor(row.name) }} />
                          <input
                            list="speaker-suggestions"
                            className="flex-1 font-body-sm text-body-sm text-on-surface bg-surface-container-low border border-primary/50 rounded px-2 py-1 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                            value={row.name}
                            onChange={(e) => renameSpeakerRow(row.id, e.target.value)}
                          />
                        </div>
                      ))}
                    </div>
                    <p className="font-body-sm text-body-sm text-on-surface-variant mt-space-2">
                      Renaming updates every line by that speaker.{note?.project_id ? ' New names are saved to the project’s people for reuse across meetings.' : ''}
                    </p>
                  </div>
                )}
                <div className="space-y-space-2">
                  <datalist id="speaker-suggestions">
                    {speakerSuggestions.map((name) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                  {!transcription?.segments.length && (
                    <p className="text-on-surface-variant font-body-sm text-body-sm">No segments found in transcript.</p>
                  )}
                  {editingSegments
                    ? draftSegments.map((seg, i) => (
                        <div key={i} className={`p-space-3 rounded flex gap-space-4 border border-outline-variant/50 ${i % 2 === 1 ? 'bg-surface-container-low/50' : ''}`}>
                          <div className="w-16 flex-shrink-0 text-right pt-2">
                            <span className="font-label-sm text-label-sm text-on-surface-variant">{fmtSeconds(seg.start)}</span>
                          </div>
                          <div className="flex-1 flex flex-col gap-1">
                            <input
                              list="speaker-suggestions"
                              placeholder="Speaker"
                              className="w-44 font-label-sm text-label-sm bg-surface-container-low border border-outline-variant rounded px-2 py-1 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                              style={seg.speaker ? { color: speakerColor(seg.speaker) } : undefined}
                              value={seg.speaker ?? ''}
                              onChange={(e) => reassignSegmentSpeaker(i, e.target.value)}
                            />
                            <input
                              className="font-body-md text-body-md text-on-surface bg-surface-container-low border border-primary/50 rounded px-2 py-1 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                              value={seg.text}
                              onChange={(e) => setDraftSegments((prev) => prev.map((s, idx) => idx === i ? { ...s, text: e.target.value } : s))}
                            />
                          </div>
                        </div>
                      ))
                    : transcription?.segments.map((seg, i) => (
                        <div
                          key={i}
                          className={`p-space-3 rounded hover:bg-surface-container-low/30 transition-colors flex gap-space-4 border border-transparent hover:border-outline-variant/50 ${
                            i % 2 === 1 ? 'bg-surface-container-low/50' : ''
                          }`}
                        >
                          <div className="w-16 flex-shrink-0 text-right pt-1">
                            <span className="font-label-sm text-label-sm text-on-surface-variant">{fmtSeconds(seg.start)}</span>
                          </div>
                          <div className="flex-1">
                            {seg.speaker && (
                              <span className="block font-label-sm text-label-sm mb-1" style={{ color: speakerColor(seg.speaker) }}>{seg.speaker}</span>
                            )}
                            <p className="font-body-md text-body-md text-on-surface leading-relaxed">{seg.text}</p>
                          </div>
                        </div>
                      ))
                  }
                </div>
              </div>
            )}

            {/* Full text tab */}
            {hasTranscript && activeTab === 'full' && (
              <div>
                <div className="flex items-center justify-between mb-space-3 gap-2">
                  <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Full Transcript</p>
                  <div className="flex items-center gap-2">
                    {!editingTranscript && transcription?.full_text && (
                      <CopyButton text={transcription.full_text} title="Copy transcript" />
                    )}
                    {!editingTranscript ? (
                      <button
                        onClick={startEditTranscript}
                        className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors"
                      >
                        <span className="material-symbols-outlined text-[14px]">edit</span>
                        Edit
                      </button>
                    ) : (
                      <div className="flex gap-2">
                        <button
                          onClick={cancelEditTranscript}
                          className="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={commitEditTranscript}
                          disabled={savingTranscript}
                          className="font-label-sm text-label-sm bg-primary text-on-primary rounded px-3 py-1 hover:opacity-90 transition-opacity disabled:opacity-50"
                        >
                          {savingTranscript ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                {editingTranscript ? (
                  <textarea
                    className="w-full font-body-md text-body-md text-on-surface bg-surface-container-low border border-primary rounded p-space-3 outline-none focus:ring-1 focus:ring-primary resize-y leading-relaxed"
                    rows={20}
                    value={draftTranscript}
                    onChange={(e) => setDraftTranscript(e.target.value)}
                    autoFocus
                  />
                ) : (
                  <div className="font-body-md text-body-md text-on-surface leading-relaxed whitespace-pre-wrap">
                    {transcription?.full_text ?? 'No text available.'}
                  </div>
                )}
              </div>
            )}

            {/* Summary tab */}
            {activeTab === 'summary' && (
              <>
                {isSummarizing && (
                  <div className="flex flex-col items-center justify-center h-full gap-space-4 text-on-surface-variant">
                    <span className="material-symbols-outlined text-[48px] animate-spin text-primary">auto_awesome</span>
                    <p className="font-body-md text-body-md">
                      {workflowRun?.status === 'chunking' ? 'Condensing long transcript…'
                        : workflowRun?.status === 'critiquing' ? 'Reviewing quality…'
                        : workflowRun?.status === 'assembling' ? 'Assembling notes…'
                        : 'Running agent workflow…'}
                    </p>
                    {workflowSteps.length > 0 && (
                      <div className="w-full max-w-md bg-surface-container-lowest border border-outline-variant/50 rounded-lg divide-y divide-outline-variant/30">
                        {workflowSteps.map((s) => (
                          <div key={s.id} className="flex items-center gap-3 px-4 py-2">
                            <span className={`material-symbols-outlined text-[18px] shrink-0 ${
                              s.status === 'done' ? 'text-primary'
                                : s.status === 'error' ? 'text-error'
                                : 'text-on-surface-variant animate-spin'
                            }`}>
                              {s.status === 'done' ? 'check_circle' : s.status === 'error' ? 'error' : 'progress_activity'}
                            </span>
                            <span className="font-body-sm text-body-sm text-on-surface flex-1 truncate">
                              {s.step_name}
                              {s.attempt > 1 && (
                                <span className="ml-2 font-label-sm text-[10px] text-on-surface-variant bg-surface-container rounded px-1.5 py-0.5">
                                  retry {s.attempt - 1}
                                </span>
                              )}
                            </span>
                            <span className="font-label-sm text-[11px] text-on-surface-variant shrink-0">
                              {s.critique_score != null && `score ${s.critique_score.toFixed(1)}`}
                              {s.critique_score == null && s.status === 'done' && s.duration_ms != null && `${(s.duration_ms / 1000).toFixed(1)}s`}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                    {(workflowRun?.total_input_tokens ?? 0) > 0 && (
                      <p className="font-body-sm text-[11px] text-on-surface-variant">
                        {workflowRun?.total_input_tokens} in / {workflowRun?.total_output_tokens ?? 0} out tokens
                        {workflowRun?.model_name ? ` · ${workflowRun.model_name}` : ''}
                      </p>
                    )}
                  </div>
                )}
                {!isSummarizing && summary?.summary_text && (
                  <>
                    <div className="flex items-center justify-between mb-space-3 gap-2">
                      <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Summary</p>
                      <div className="flex items-center gap-2">
                        {!editingSummary && <CopyButton text={summaryText} title="Copy summary" />}
                        {!editingSummary ? (
                          <button onClick={startEditSummary} className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">
                            <span className="material-symbols-outlined text-[14px]">edit</span>Edit
                          </button>
                        ) : (
                          <div className="flex gap-2">
                            <button onClick={cancelEditSummary} className="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">Cancel</button>
                            <button onClick={commitEditSummary} disabled={savingSummary} className="font-label-sm text-label-sm bg-primary text-on-primary rounded px-3 py-1 hover:opacity-90 disabled:opacity-50">
                              {savingSummary ? 'Saving…' : 'Save'}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                    {editingSummary ? (
                      <textarea
                        className="w-full font-body-md text-body-md text-on-surface bg-surface-container-low border border-primary rounded p-space-3 outline-none focus:ring-1 focus:ring-primary resize-y leading-relaxed"
                        rows={16}
                        value={draftSummaryText}
                        onChange={(e) => setDraftSummaryText(e.target.value)}
                        autoFocus
                      />
                    ) : (
                      <div className="prose prose-sm max-w-none text-on-surface prose-p:leading-relaxed prose-p:mt-0 prose-p:mb-[1.6em] prose-headings:mt-8 prose-headings:mb-3 prose-ul:my-3 prose-ol:my-3 prose-li:my-1">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary.summary_text}</ReactMarkdown>
                      </div>
                    )}
                  </>
                )}
              </>
            )}

            {/* Action Items tab */}
            {activeTab === 'action_items' && hasSummary && (
              <div>
                <div className="flex items-center justify-between mb-space-3 gap-2">
                  <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Action Items</p>
                  <div className="flex items-center gap-2">
                    {!editingActionItems && <CopyButton text={actionItemsText} title="Copy action items" />}
                    {!editingActionItems ? (
                      <button onClick={startEditActionItems} className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">
                        <span className="material-symbols-outlined text-[14px]">edit</span>Edit
                      </button>
                    ) : (
                      <div className="flex gap-2">
                        <button onClick={cancelEditActionItems} className="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">Cancel</button>
                        <button onClick={commitEditActionItems} disabled={savingSummary} className="font-label-sm text-label-sm bg-primary text-on-primary rounded px-3 py-1 hover:opacity-90 disabled:opacity-50">
                          {savingSummary ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {editingActionItems ? (
                  <div className="space-y-space-2">
                    {draftActionItems.map((item, i) => (
                      <div key={i} className="flex items-start gap-2 p-space-3 rounded border border-outline-variant/50 bg-surface-container-lowest">
                        <div className="flex-1 min-w-0 space-y-1.5">
                          <input
                            className="w-full font-body-md text-body-md text-on-surface bg-surface-container-low border border-outline-variant rounded px-2 py-1 focus:outline-none focus:border-primary"
                            placeholder="Task…"
                            value={item.task}
                            onChange={(e) => setDraftActionItems((prev) => prev.map((a, idx) => idx === i ? { ...a, task: e.target.value } : a))}
                          />
                          <div className="flex gap-2">
                            <input
                              className="flex-1 font-body-sm text-body-sm text-on-surface bg-surface-container-low border border-outline-variant rounded px-2 py-1 focus:outline-none focus:border-primary"
                              placeholder="Owner…"
                              value={item.owner}
                              onChange={(e) => setDraftActionItems((prev) => prev.map((a, idx) => idx === i ? { ...a, owner: e.target.value } : a))}
                            />
                            <input
                              className="flex-1 font-body-sm text-body-sm text-on-surface bg-surface-container-low border border-outline-variant rounded px-2 py-1 focus:outline-none focus:border-primary"
                              placeholder="Deadline…"
                              value={item.deadline}
                              onChange={(e) => setDraftActionItems((prev) => prev.map((a, idx) => idx === i ? { ...a, deadline: e.target.value } : a))}
                            />
                          </div>
                        </div>
                        <button
                          onClick={() => setDraftActionItems((prev) => prev.filter((_, idx) => idx !== i))}
                          className="mt-1 p-1 text-outline hover:text-error hover:bg-error/10 rounded transition-colors flex-shrink-0"
                        >
                          <span className="material-symbols-outlined text-[18px]">delete</span>
                        </button>
                      </div>
                    ))}
                    <button
                      onClick={() => setDraftActionItems((prev) => [...prev, { task: '', owner: '', deadline: '' }])}
                      className="w-full flex items-center justify-center gap-1.5 font-label-sm text-label-sm text-on-surface-variant border border-dashed border-outline-variant rounded py-2 hover:bg-surface-container-low transition-colors"
                    >
                      <span className="material-symbols-outlined text-[16px]">add</span>
                      Add item
                    </button>
                  </div>
                ) : (
                  <div className="space-y-space-3">
                    {summary!.action_items.length === 0 && (
                      <p className="text-on-surface-variant font-body-sm text-body-sm">No action items found.</p>
                    )}
                    {summary!.action_items.map((item, i) => (
                      <div
                        key={i}
                        className={`flex items-start gap-space-3 p-space-3 rounded border border-outline-variant/50 transition-colors ${
                          doneTasks[i] ? 'bg-surface-container opacity-60' : 'bg-surface-container-lowest'
                        }`}
                      >
                        <button
                          onClick={() => setDoneTasks((prev) => ({ ...prev, [i]: !prev[i] }))}
                          className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors ${
                            doneTasks[i]
                              ? 'bg-primary border-primary text-on-primary'
                              : 'border-outline-variant hover:border-primary'
                          }`}
                        >
                          {doneTasks[i] && <span className="material-symbols-outlined text-[14px]">check</span>}
                        </button>
                        <div className="flex-1 min-w-0">
                          <p className={`font-body-md text-body-md text-on-surface ${doneTasks[i] ? 'line-through' : ''}`}>
                            {item.task}
                          </p>
                          <div className="flex flex-wrap gap-space-3 mt-space-1">
                            {item.owner && (
                              <span className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant">
                                <span className="material-symbols-outlined text-[14px]">person</span>
                                {item.owner}
                              </span>
                            )}
                            {item.deadline && (
                              <span className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant">
                                <span className="material-symbols-outlined text-[14px]">schedule</span>
                                {item.deadline}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Suggestions tab */}
            {activeTab === 'suggestions' && hasSummary && (
              <>
                <div className="flex items-center justify-between mb-space-3 gap-2">
                  <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Suggestions</p>
                  <div className="flex items-center gap-2">
                    {!editingSuggestions && <CopyButton text={suggestionsText} title="Copy suggestions" />}
                    {!editingSuggestions ? (
                      <button onClick={startEditSuggestions} className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">
                        <span className="material-symbols-outlined text-[14px]">edit</span>Edit
                      </button>
                    ) : (
                      <div className="flex gap-2">
                        <button onClick={cancelEditSuggestions} className="font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-2 py-1 hover:bg-surface-container-low transition-colors">Cancel</button>
                        <button onClick={commitEditSuggestions} disabled={savingSummary} className="font-label-sm text-label-sm bg-primary text-on-primary rounded px-3 py-1 hover:opacity-90 disabled:opacity-50">
                          {savingSummary ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                {editingSuggestions ? (
                  <textarea
                    className="w-full font-body-md text-body-md text-on-surface bg-surface-container-low border border-primary rounded p-space-3 outline-none focus:ring-1 focus:ring-primary resize-y leading-relaxed"
                    rows={14}
                    value={draftSuggestionsText}
                    onChange={(e) => setDraftSuggestionsText(e.target.value)}
                    autoFocus
                  />
                ) : (
                  <div className="prose prose-sm max-w-none text-on-surface prose-p:leading-relaxed prose-p:mt-0 prose-p:mb-[1.6em] prose-headings:mt-8 prose-headings:mb-3 prose-ul:my-3 prose-ol:my-3 prose-li:my-1">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary!.suggestions_text || 'No suggestions generated.'}</ReactMarkdown>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Right: config panels */}
        <div className="w-full md:w-1/3 flex flex-col gap-space-4 flex-shrink-0">

          {/* Transcription panel */}
          <div className="bg-surface-container-lowest rounded p-space-4">
            <h3 className="font-headline-md text-headline-md font-bold text-on-surface mb-space-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px]">mic</span>
              Transcription
            </h3>
            {transcribeError && (
              <p className="font-body-sm text-body-sm text-error mb-space-3">{transcribeError}</p>
            )}
            <button
              disabled={!canTranscribe && !isTranscribing}
              onClick={handleTranscribe}
              className="w-full bg-surface-container-lowest border border-outline-variant text-on-surface font-label-md text-label-md rounded py-2 hover:bg-surface-container-low transition-colors flex justify-center items-center gap-2 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span className={`material-symbols-outlined text-[18px] ${isTranscribing ? 'animate-spin' : ''}`}>
                {isTranscribing ? 'sync' : 'subtitles'}
              </span>
              {isTranscribing ? 'Transcribing…' : hasTranscript ? 'Re-transcribe' : 'Transcribe'}
            </button>
          </div>

          {/* Project panel */}
          <div className="bg-surface-container-lowest rounded p-space-4">
            <h3 className="font-headline-md text-headline-md font-bold text-on-surface mb-space-3 flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px]">folder</span>
              Project
            </h3>
            <Select
              value={String(note.project_id ?? '')}
              onChange={(v) => handleConfigChange('project_id', v ? Number(v) : null)}
              options={[
                { value: '', label: 'No Project' },
                ...projects.map((p) => ({ value: String(p.id), label: p.name })),
              ]}
              size="md"
            />
          </div>

          {/* Summarization panel */}
          <div className="bg-surface-container-lowest rounded p-space-4">
            <div className="flex items-center justify-between mb-space-4">
              <h3 className="font-headline-md text-headline-md font-bold text-on-surface flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px]">auto_awesome</span>
                Summarization
              </h3>
              {lmStatus && (
                <span className={`flex items-center gap-1 font-label-sm text-label-sm px-2 py-0.5 rounded-full ${
                  lmStatus.connected
                    ? 'bg-primary/10 text-primary'
                    : 'bg-error/10 text-error'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${lmStatus.connected ? 'bg-primary' : 'bg-error'}`} />
                  {lmStatus.connected ? 'LM Studio' : 'Offline'}
                </span>
              )}
            </div>
            <div className="space-y-space-4">
              <div>
                <label className="block font-label-sm text-label-sm text-on-surface-variant mb-space-2 uppercase tracking-wider">Domain</label>
                <Select
                  value={String(note.domain_id ?? '')}
                  onChange={(v) => handleConfigChange('domain_id', v ? Number(v) : null)}
                  options={[
                    { value: '', label: 'No Domain' },
                    ...domains.map((d) => ({ value: String(d.id), label: d.name })),
                  ]}
                  size="md"
                />
              </div>

              <div>
                <label className="block font-label-sm text-label-sm text-on-surface-variant mb-space-2 uppercase tracking-wider">Template</label>
                <Select
                  value={String(note.template_id ?? '')}
                  onChange={(v) => handleConfigChange('template_id', v ? Number(v) : null)}
                  options={[
                    { value: '', label: 'No Template' },
                    ...filteredTemplates.map((t) => ({ value: String(t.id), label: t.name })),
                  ]}
                  size="md"
                />
              </div>

              {summarizeError && (
                <p className="font-body-sm text-body-sm text-error">{summarizeError}</p>
              )}

              <button
                disabled={!canSummarize}
                onClick={handleSummarize}
                title={canSummarize ? undefined : 'Transcribe first to enable summarization'}
                className="w-full bg-primary-container text-on-primary-container font-label-md text-label-md rounded py-2 flex justify-center items-center gap-2 shadow-sm hover:bg-primary hover:text-on-primary transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <span className={`material-symbols-outlined icon-fill text-[18px] ${isSummarizing ? 'animate-spin' : ''}`}>
                  {isSummarizing ? 'sync' : 'auto_awesome'}
                </span>
                {isSummarizing ? 'Summarizing…' : hasSummary ? 'Re-summarize' : 'Summarize'}
              </button>

              {hasTranscript && (
                <button
                  onClick={handleShowPromptPreview}
                  disabled={promptPreviewLoading}
                  className="w-full border border-outline-variant text-on-surface-variant font-label-sm text-label-sm rounded py-1.5 flex justify-center items-center gap-1.5 hover:bg-surface-container-low transition-colors disabled:opacity-40"
                >
                  <span className="material-symbols-outlined text-[16px]">preview</span>
                  {promptPreviewLoading ? 'Loading…' : 'Preview Prompt'}
                </button>
              )}
            </div>
          </div>

          {/* Export panel */}
          <div className="bg-surface-container-lowest rounded p-space-4">
            <h3 className="font-headline-md text-headline-md font-bold text-on-surface mb-space-3 flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px]">download</span>
              Export
            </h3>
            <div className="flex gap-2">
              <button
                onClick={() => exportNote(id, 'markdown')}
                className="flex-1 flex items-center justify-center gap-1.5 font-label-sm text-label-sm text-on-surface border border-outline-variant rounded py-2 hover:bg-surface-container-low transition-colors"
              >
                <span className="material-symbols-outlined text-[16px]">article</span>
                Markdown
              </button>
              <button
                onClick={() => exportNote(id, 'text')}
                className="flex-1 flex items-center justify-center gap-1.5 font-label-sm text-label-sm text-on-surface border border-outline-variant rounded py-2 hover:bg-surface-container-low transition-colors"
              >
                <span className="material-symbols-outlined text-[16px]">text_snippet</span>
                Plain Text
              </button>
            </div>
          </div>

          {/* File Info */}
          <div className="bg-surface-container-lowest rounded p-space-4">
            <h3 className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-space-3">File Info</h3>
            <div className="space-y-space-2 font-body-sm text-body-sm text-on-surface">
              <div className="flex justify-between gap-2">
                <span className="text-on-surface-variant shrink-0">Uploaded</span>
                <span className="text-right">{fmtDate(note.created_at)}</span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-on-surface-variant shrink-0">File</span>
                <span className="text-right truncate max-w-[160px]">{note.audio_file_name ?? '—'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-on-surface-variant">Size</span>
                <span>{fmtFileSize(note.audio_file_size)}</span>
              </div>
              {hasTranscript && (
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant shrink-0">Transcribed</span>
                  <span className="text-right">{fmtDate(note.updated_at)}</span>
                </div>
              )}
              {transcription?.model_used && (
                <>
                  <div className="border-t border-outline-variant/50 my-space-2" />
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Whisper</span>
                    <span>{transcription.model_used}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Segments</span>
                    <span>{transcription.segments.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Words</span>
                    <span>{transcription.full_text?.split(/\s+/).filter(Boolean).length ?? 0}</span>
                  </div>
                </>
              )}
              {summary?.llm_model_used && (
                <>
                  <div className="border-t border-outline-variant/50 my-space-2" />
                  <div className="flex justify-between gap-2">
                    <span className="text-on-surface-variant shrink-0">LLM</span>
                    <span className="text-right truncate max-w-[130px]">{summary.llm_model_used}</span>
                  </div>
                  {summary.generated_at && (
                    <div className="flex justify-between gap-2">
                      <span className="text-on-surface-variant shrink-0">Summarized</span>
                      <span className="text-right">{fmtDate(summary.generated_at)}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Actions</span>
                    <span>{summary.action_items.length}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Prompt Preview Modal */}
      {promptPreview && (
        <div
          className="fixed inset-0 bg-on-surface/40 backdrop-blur-[2px] z-[60] flex items-center justify-center p-4"
          onClick={() => setPromptPreview(null)}
        >
          <div
            className="bg-surface-container-lowest w-full max-w-2xl rounded-xl shadow-2xl flex flex-col overflow-hidden max-h-[80vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-space-6 py-space-4 border-b border-outline-variant flex items-center justify-between flex-shrink-0">
              <h2 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px] text-primary">preview</span>
                Prompt Preview
              </h2>
              <button
                onClick={() => setPromptPreview(null)}
                className="material-symbols-outlined text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer"
              >
                close
              </button>
            </div>
            <div className="overflow-y-auto p-space-6 space-y-space-4">
              <div>
                <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-space-2">System</p>
                <pre className="font-label-md text-label-md text-on-surface bg-surface-container-low rounded p-space-3 whitespace-pre-wrap text-xs leading-relaxed">
                  {promptPreview.system}
                </pre>
              </div>
              <div>
                <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-space-2">User</p>
                <pre className="font-label-md text-label-md text-on-surface bg-surface-container-low rounded p-space-3 whitespace-pre-wrap text-xs leading-relaxed">
                  {promptPreview.user}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
