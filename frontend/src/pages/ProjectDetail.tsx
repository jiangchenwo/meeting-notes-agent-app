import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import StatusBadge from '../components/StatusBadge';
import Breadcrumb from '../components/Breadcrumb';
import Select from '../components/Select';
import { getProject, updateProject } from '../api/projects';
import { getProjectSpeakers, createProjectSpeaker, updateProjectSpeaker, deleteProjectSpeaker } from '../api/speakers';
import { getNotes } from '../api/notes';
import { resolveColor, colorSwatchClass, projectIconClass, COLORS } from '../lib/domains';
import { speakerColor } from '../lib/speakerColor';
import type { NoteBlock, Project, ProjectSpeaker } from '../api/types';

const PROJECT_ICONS = [
  'folder', 'work', 'business_center', 'groups', 'code', 'science',
  'school', 'local_hospital', 'rocket_launch', 'analytics',
  'campaign', 'engineering', 'build', 'bug_report', 'chat',
  'calendar_today', 'public', 'hub', 'bolt', 'star',
  'home', 'laptop', 'sports_esports', 'account_balance',
];

const KB_TEMPLATE = `## Team
- [Name] — [Role]

## Glossary
- [term]: [definition]

## Context
[What this project/team does, recurring topics, key systems.]
`;

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = (now.getTime() - d.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatSize(bytes: number | null): string {
  if (!bytes) return '—';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);

  const [project, setProject] = useState<Project | null>(null);
  const [notes, setNotes] = useState<NoteBlock[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<'overview' | 'system_prompt' | 'knowledge_base' | 'people'>('overview');
  const [statusFilter, setStatusFilter] = useState('All Status');
  const [search, setSearch] = useState('');

  // People (speaker roster) tab state
  const [speakers, setSpeakers] = useState<ProjectSpeaker[]>([]);
  const [newSpeaker, setNewSpeaker] = useState('');
  const [addingSpeaker, setAddingSpeaker] = useState(false);

  // Inline edit state
  const [inlineEditing, setInlineEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editColor, setEditColor] = useState<string | null>(null);
  const [editIcon, setEditIcon] = useState<string | null>(null);
  const [savingEdit, setSavingEdit] = useState(false);

  // System prompt tab state
  const [spDraft, setSpDraft] = useState('');
  const [spSaving, setSpSaving] = useState(false);
  const [spSaved, setSpSaved] = useState(false);

  // Knowledge base tab state
  const [kbDraft, setKbDraft] = useState('');
  const [kbSaving, setKbSaving] = useState(false);
  const [kbSaved, setKbSaved] = useState(false);

  useEffect(() => {
    Promise.all([getProject(projectId), getNotes(projectId)])
      .then(([p, n]) => {
        setProject(p);
        setNotes(n);
        setKbDraft(p.knowledge_base);
        setSpDraft(p.custom_system_prompt);
      })
      .finally(() => setLoading(false));
    getProjectSpeakers(projectId).then(setSpeakers).catch(() => setSpeakers([]));
  }, [projectId]);

  const setSpeakerNameLocal = (sid: number, name: string) =>
    setSpeakers((prev) => prev.map((s) => (s.id === sid ? { ...s, name } : s)));

  const handleAddSpeaker = async () => {
    const name = newSpeaker.trim();
    if (!name) return;
    setAddingSpeaker(true);
    const created = await createProjectSpeaker(projectId, { name }).catch(() => null);
    setAddingSpeaker(false);
    if (created) {
      setSpeakers((prev) =>
        prev.some((s) => s.id === created.id)
          ? prev
          : [...prev, created].sort((a, b) => a.name.localeCompare(b.name)),
      );
      setNewSpeaker('');
    }
  };

  const handleRenameSpeaker = async (sid: number, name: string) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    const updated = await updateProjectSpeaker(projectId, sid, { name: trimmed }).catch(() => null);
    if (updated) setSpeakers((prev) => prev.map((s) => (s.id === sid ? updated : s)));
  };

  const handleDeleteSpeaker = async (sid: number) => {
    const ok = await deleteProjectSpeaker(projectId, sid).then(() => true).catch(() => false);
    if (ok) setSpeakers((prev) => prev.filter((s) => s.id !== sid));
  };

  const startInlineEdit = (p: Project) => {
    setEditName(p.name);
    setEditDescription(p.description);
    setEditColor(p.color);
    setEditIcon(p.icon);
    setInlineEditing(true);
  };

  const handleInlineSave = async () => {
    if (!project) return;
    setSavingEdit(true);
    const updated = await updateProject(project.id, {
      name: editName.trim() || project.name,
      description: editDescription,
      color: editColor,
      icon: editIcon,
    }).catch(() => null);
    setSavingEdit(false);
    if (updated) {
      setProject(updated);
      setInlineEditing(false);
    }
  };

  const handleSaveSp = async () => {
    if (!project) return;
    setSpSaving(true);
    const updated = await updateProject(project.id, { custom_system_prompt: spDraft }).catch(() => null);
    setSpSaving(false);
    if (updated) {
      setProject(updated);
      setSpSaved(true);
      setTimeout(() => setSpSaved(false), 2000);
    }
  };

  const handleSaveKb = async () => {
    if (!project) return;
    setKbSaving(true);
    const updated = await updateProject(project.id, { knowledge_base: kbDraft }).catch(() => null);
    setKbSaving(false);
    if (updated) {
      setProject(updated);
      setKbSaved(true);
      setTimeout(() => setKbSaved(false), 2000);
    }
  };

  const insertKbTemplate = () => {
    if (kbDraft.trim() && !window.confirm('Replace current content with the starter template?')) return;
    setKbDraft(KB_TEMPLATE);
  };

  const filtered = notes.filter((n) => {
    const matchesSearch = n.display_name.toLowerCase().includes(search.toLowerCase());
    const matchesStatus =
      statusFilter === 'All Status' || n.status === statusFilter.toLowerCase();
    return matchesSearch && matchesStatus;
  });

  if (loading) {
    return (
      <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full">
        <p className="text-on-surface-variant font-body-md text-body-md">Loading…</p>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full">
        <p className="text-error font-body-md text-body-md">Project not found.</p>
        <Link to="/projects" className="text-primary hover:underline font-label-md text-label-md">← Back to Projects</Link>
      </div>
    );
  }

  const totalSize = notes.reduce((sum, n) => sum + (n.audio_file_size ?? 0), 0);
  const doneCount = notes.filter((n) => n.status === 'done').length;

  return (
    <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full">
      <Breadcrumb items={[
        { label: 'Home', to: '/' },
        { label: 'Projects', to: '/projects' },
        { label: project.name },
      ]} />

      {/* Tabs */}
      <div className="border-b border-outline-variant mb-space-6">
        <nav className="flex gap-space-6">
          {([
            ['overview', 'Overview'],
            ['system_prompt', 'System Prompt'],
            ['knowledge_base', 'Knowledge Base'],
            ['people', 'People'],
          ] as const).map(([t, label]) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`border-b-2 font-label-md text-label-md pb-3 px-1 transition-colors ${
                tab === t
                  ? 'border-primary-container text-primary'
                  : 'border-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* System Prompt tab — full width editor */}
      {tab === 'system_prompt' && (
        <div className="bg-surface-container-lowest rounded-lg p-space-6 flex flex-col gap-space-4">
          <div>
            <h2 className="font-headline-md text-headline-md text-on-surface mb-space-1">System Prompt</h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Overrides the default assistant persona for all LLM calls on this project.
            </p>
          </div>
          <textarea
            value={spDraft}
            onChange={(e) => setSpDraft(e.target.value)}
            rows={18}
            spellCheck={false}
            placeholder="e.g. You are helping the engineering team at Acme Corp. Focus on technical decisions, ticket references, and concrete action items with owners."
            className="w-full p-space-4 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-mono font-body-sm text-body-sm text-on-surface transition-all resize-none leading-relaxed"
          />
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              {spDraft.length} chars
            </span>
            <button
              onClick={handleSaveSp}
              disabled={spSaving}
              className="px-space-4 py-space-2 rounded font-label-md text-label-md text-on-primary bg-primary-container hover:bg-primary transition-colors shadow-sm flex items-center gap-space-2 disabled:opacity-50"
            >
              <span className="material-symbols-outlined text-[16px]">
                {spSaved ? 'check_circle' : 'save'}
              </span>
              {spSaving ? 'Saving…' : spSaved ? 'Saved' : 'Save'}
            </button>
          </div>
        </div>
      )}

      {/* Knowledge Base tab — full width editor */}
      {tab === 'knowledge_base' && (
        <div className="bg-surface-container-lowest rounded-lg p-space-6 flex flex-col gap-space-4">
          <div>
            <h2 className="font-headline-md text-headline-md text-on-surface mb-space-1">Knowledge Base</h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Markdown text injected as context when the LLM generates notes for this project.
              Use structured sections so the model can quickly find relevant facts.
            </p>
          </div>
          <textarea
            value={kbDraft}
            onChange={(e) => setKbDraft(e.target.value)}
            rows={18}
            spellCheck={false}
            placeholder={KB_TEMPLATE}
            className="w-full p-space-4 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-mono font-body-sm text-body-sm text-on-surface transition-all resize-none leading-relaxed"
          />
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              {kbDraft.length} chars
            </span>
            <div className="flex items-center gap-space-3">
              <button
                onClick={insertKbTemplate}
                className="px-space-4 py-space-2 rounded font-label-md text-label-md text-on-surface-variant border border-outline-variant bg-surface hover:bg-surface-container-high transition-colors flex items-center gap-space-2"
              >
                <span className="material-symbols-outlined text-[16px]">post_add</span>
                Insert template
              </button>
              <button
                onClick={handleSaveKb}
                disabled={kbSaving}
                className="px-space-4 py-space-2 rounded font-label-md text-label-md text-on-primary bg-primary-container hover:bg-primary transition-colors shadow-sm flex items-center gap-space-2 disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-[16px]">
                  {kbSaved ? 'check_circle' : 'save'}
                </span>
                {kbSaving ? 'Saving…' : kbSaved ? 'Saved' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* People tab — project speaker roster */}
      {tab === 'people' && (
        <div className="bg-surface-container-lowest rounded-lg p-space-6 flex flex-col gap-space-4 max-w-2xl">
          <div>
            <h2 className="font-headline-md text-headline-md text-on-surface mb-space-1">People</h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Known participants for this project. These names autocomplete when you label speakers in a
              recording, keeping labels consistent across meetings. New names you type while editing a transcript
              are added here automatically.
            </p>
          </div>
          <div className="flex flex-col gap-space-2">
            {speakers.length === 0 && (
              <p className="font-body-sm text-body-sm text-on-surface-variant/70 italic">
                No people yet. Add participants below, or they’ll appear here once you name speakers in a recording.
              </p>
            )}
            {speakers.map((s) => (
              <div key={s.id} className="flex items-center gap-space-2">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: speakerColor(s.name) }} />
                <input
                  value={s.name}
                  onChange={(e) => setSpeakerNameLocal(s.id, e.target.value)}
                  onBlur={(e) => handleRenameSpeaker(s.id, e.target.value)}
                  className="flex-1 p-space-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
                />
                <button
                  onClick={() => handleDeleteSpeaker(s.id)}
                  aria-label="Remove person"
                  className="p-space-1 rounded text-on-surface-variant hover:text-error hover:bg-surface-container-high transition-colors"
                >
                  <span className="material-symbols-outlined text-[18px]">delete</span>
                </button>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-space-2 pt-space-2 border-t border-outline-variant">
            <input
              value={newSpeaker}
              onChange={(e) => setNewSpeaker(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleAddSpeaker(); }}
              placeholder="Add a person…"
              className="flex-1 p-space-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
            />
            <button
              onClick={handleAddSpeaker}
              disabled={addingSpeaker || !newSpeaker.trim()}
              className="px-space-4 py-space-2 rounded font-label-md text-label-md text-on-primary bg-primary-container hover:bg-primary transition-colors flex items-center gap-space-2 disabled:opacity-50"
            >
              <span className="material-symbols-outlined text-[16px]">add</span>
              Add
            </button>
          </div>
        </div>
      )}

      {/* Overview tab — bento grid */}
      {tab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-4 items-start">
          {/* Left: Project info panel */}
          <aside className="lg:col-span-4 flex flex-col gap-space-4">
            <div className="bg-surface-container-lowest rounded-lg p-space-4 relative">
              {!inlineEditing ? (
                <>
                  <button
                    onClick={() => startInlineEdit(project)}
                    aria-label="Edit project"
                    className="absolute top-3 right-3 p-space-1 rounded hover:bg-surface-container-high transition-colors text-on-surface-variant hover:text-on-surface"
                  >
                    <span className="material-symbols-outlined text-[18px]">edit</span>
                  </button>
                  <div className={`inline-flex items-center justify-center w-10 h-10 rounded mb-3 ${projectIconClass[resolveColor(project.name, project.color)]}`}>
                    <span className="material-symbols-outlined text-[20px]">{project.icon || 'folder'}</span>
                  </div>
                  <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface mb-2 tracking-tight pr-8">
                    {project.name}
                  </h2>
                  {project.description ? (
                    <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
                      {project.description}
                    </p>
                  ) : (
                    <p className="font-body-md text-body-md text-on-surface-variant/50 leading-relaxed italic">
                      No description yet.
                    </p>
                  )}
                </>
              ) : (
                <div className="flex flex-col gap-space-3">
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    className="w-full p-space-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-headline-md text-headline-md text-on-surface transition-all"
                    autoFocus
                  />
                  <textarea
                    value={editDescription}
                    onChange={(e) => setEditDescription(e.target.value)}
                    placeholder="Central purpose of this project…"
                    rows={3}
                    className="w-full p-space-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all resize-none leading-relaxed"
                  />

                  {/* Color picker */}
                  <div>
                    <p className="font-label-sm text-label-sm text-on-surface-variant mb-1.5">Color</p>
                    <div className="flex flex-wrap gap-1.5">
                      {COLORS.map((c) => (
                        <button
                          key={c}
                          type="button"
                          onClick={() => setEditColor(c)}
                          className={`w-5 h-5 rounded-full transition-all ${colorSwatchClass[c]} ${
                            resolveColor(editName, editColor) === c
                              ? 'ring-2 ring-offset-1 ring-on-surface scale-110'
                              : 'opacity-70 hover:opacity-100 hover:scale-110'
                          }`}
                          title={c}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Icon picker */}
                  <div>
                    <p className="font-label-sm text-label-sm text-on-surface-variant mb-1.5">Icon</p>
                    <div className="flex flex-wrap gap-1">
                      {PROJECT_ICONS.map((ic) => (
                        <button
                          key={ic}
                          type="button"
                          onClick={() => setEditIcon(ic)}
                          className={`w-8 h-8 flex items-center justify-center rounded transition-colors ${
                            (editIcon || 'folder') === ic
                              ? `${projectIconClass[resolveColor(editName, editColor)]} ring-1 ring-primary`
                              : 'text-on-surface-variant hover:bg-surface-container-high'
                          }`}
                          title={ic}
                        >
                          <span className="material-symbols-outlined text-[18px]">{ic}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-space-2 justify-end">
                    <button
                      onClick={() => setInlineEditing(false)}
                      className="px-space-3 py-space-1 rounded font-label-md text-label-md text-on-surface-variant hover:bg-surface-container-high transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleInlineSave}
                      disabled={savingEdit}
                      className="px-space-3 py-space-1 rounded font-label-md text-label-md text-on-primary bg-primary-container hover:bg-primary transition-colors flex items-center gap-space-1 disabled:opacity-50"
                    >
                      <span className="material-symbols-outlined text-[14px]">save</span>
                      {savingEdit ? 'Saving…' : 'Save'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-space-4">
              <div className="bg-surface-container-lowest p-space-4 rounded-lg flex flex-col justify-between">
                <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Recordings</span>
                <div className="mt-space-2 flex items-baseline gap-space-2">
                  <span className="font-headline-lg text-headline-lg text-on-surface">{notes.length}</span>
                  <span className="font-label-md text-label-md text-on-surface-variant">files</span>
                </div>
              </div>
              <div className="bg-surface-container-lowest p-space-4 rounded-lg flex flex-col justify-between">
                <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Completed</span>
                <div className="mt-space-2 flex items-baseline gap-space-2">
                  <span className="font-headline-lg text-headline-lg text-primary-container">{doneCount}</span>
                  <span className="font-label-md text-label-md text-on-surface-variant">/ {notes.length}</span>
                </div>
              </div>
            </div>

            {totalSize > 0 && (
              <div className="bg-surface-container-lowest p-space-4 rounded-lg">
                <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Total Storage</span>
                <div className="mt-space-2">
                  <span className="font-body-lg text-on-surface">
                    {totalSize >= 1024 * 1024 * 1024
                      ? `${(totalSize / (1024 * 1024 * 1024)).toFixed(2)} GB`
                      : `${(totalSize / (1024 * 1024)).toFixed(1)} MB`}
                  </span>
                </div>
              </div>
            )}
          </aside>

          {/* Right: Note list */}
          <section className="lg:col-span-8 flex flex-col gap-space-4">
            <div className="flex flex-col sm:flex-row gap-3 items-center">
              <div className="flex flex-1 w-full items-center bg-surface-container-lowest border border-outline-variant rounded px-3 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all shadow-sm">
                <span className="material-symbols-outlined text-on-surface-variant text-[18px] mr-2">search</span>
                <input
                  className="bg-transparent border-none outline-none focus:ring-0 w-full font-body-sm text-body-sm text-on-surface placeholder:text-on-surface-variant p-0"
                  placeholder="Search recordings…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
              <Select
                value={statusFilter}
                onChange={setStatusFilter}
                options={[
                  { value: 'All Status', label: 'All Status' },
                  { value: 'Done', label: 'Done' },
                  { value: 'Pending', label: 'Pending' },
                  { value: 'Transcribing', label: 'Transcribing' },
                  { value: 'Error', label: 'Error' },
                ]}
                className="shrink-0"
              />
            </div>

            {notes.length === 0 ? (
              <div className="bg-surface-container-lowest rounded-lg p-space-8 text-center text-on-surface-variant font-body-md text-body-md">
                No recordings in this project yet. Upload recordings from the{' '}
                <Link to="/" className="text-primary hover:underline">main page</Link> and assign them here.
              </div>
            ) : (
              <div className="flex flex-col gap-space-4">
                {filtered.map((note) => (
                    <Link
                      key={note.id}
                      to={`/projects/${id}/notes/${note.id}`}
                      className={`bg-surface-container-lowest rounded-lg p-space-4 hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)] transition-all cursor-pointer relative group block ${
                        note.status === 'done' ? 'border-l-4 border-l-primary-container' : ''
                      }`}
                    >
                      <div className="absolute top-4 right-4">
                        <StatusBadge status={note.status} />
                      </div>
                      <div className="pr-28">
                        <h3 className="font-body-lg text-body-lg font-semibold text-on-surface mb-1 group-hover:text-primary-container transition-colors">
                          {note.display_name}
                        </h3>
                        <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-4 line-clamp-1">
                          {note.audio_file_name}
                        </p>
                        <div className="flex items-center gap-4 text-on-surface-variant font-label-sm text-label-sm">
                          <span className="flex items-center gap-1">
                            <span className="material-symbols-outlined text-[14px]">calendar_today</span>
                            {formatDate(note.created_at)}
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="material-symbols-outlined text-[14px]">folder</span>
                            {formatSize(note.audio_file_size)}
                          </span>
                          {note.domain_name && (
                            <span className="flex items-center gap-1">
                              <span className="material-symbols-outlined text-[14px]">label</span>
                              {note.domain_name}
                            </span>
                          )}
                        </div>
                      </div>
                    </Link>
                ))}
                {filtered.length === 0 && (
                  <div className="bg-surface-container-lowest rounded-lg p-space-8 text-center text-on-surface-variant font-body-md text-body-md">
                    No recordings match your filter.
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
