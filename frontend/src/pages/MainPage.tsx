import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import StatusBadge from '../components/StatusBadge';
import UploadModal from '../components/UploadModal';
import Select from '../components/Select';
import { getNotes, updateNote, deleteNote, searchNotes, bulkDeleteNotes, bulkUpdateNotes } from '../api/notes';
import { getProjects } from '../api/projects';
import { exportNote } from '../api/export';
import { projectTagBase, projectTagClass } from '../lib/domains';
import type { NoteBlock, Project } from '../api/types';

const STATUS_OPTIONS = [
  { value: '', label: 'All statuses' },
  { value: 'pending', label: 'Pending' },
  { value: 'transcribing', label: 'Transcribing' },
  { value: 'transcribed', label: 'Transcribed' },
  { value: 'summarizing', label: 'Summarizing' },
  { value: 'done', label: 'Done' },
  { value: 'error', label: 'Error' },
];

function formatSize(bytes: number | null): string {
  if (!bytes) return '—';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(ms: number | null): string {
  if (!ms) return '—';
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const date = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  const time = d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  return `${date} · ${time}`;
}

function hasFilters(search: string, projectFilter: string, statusFilter: string) {
  return !!(search || projectFilter || statusFilter);
}

export default function MainPage() {
  const location = useLocation();
  const [notes, setNotes] = useState<NoteBlock[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [projectFilter, setProjectFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [queuedFiles, setQueuedFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingName, setEditingName] = useState('');

  // Bulk select
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [bulkProjectId, setBulkProjectId] = useState('');
  const [editingProjectId, setEditingProjectId] = useState<number | null>(null);

  // Drag to reorder
  const [dragOverId, setDragOverId] = useState<number | null>(null);
  const dragSrcId = useRef<number | null>(null);

  const zoneRef = useRef<HTMLDivElement>(null);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadNotes = useCallback(async () => {
    const data = await getNotes();
    setNotes(data);
  }, []);

  useEffect(() => {
    Promise.all([getNotes(), getProjects()])
      .then(([n, p]) => { setNotes(n); setProjects(p); })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (location.state?.openUpload) {
      setShowUploadModal(true);
      window.history.replaceState({}, '');
    }
  }, [location.state]);

  // Debounced backend search when filters are active
  useEffect(() => {
    if (!hasFilters(search, projectFilter, statusFilter)) return;
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(async () => {
      const results = await searchNotes({
        q: search || undefined,
        project_id: projectFilter ? Number(projectFilter) : null,
        status: statusFilter || undefined,
      }).catch(() => null);
      if (results) setNotes(results);
    }, 300);
    return () => { if (searchTimeout.current) clearTimeout(searchTimeout.current); };
  }, [search, projectFilter, statusFilter]);

  // Reload full list when filters cleared
  useEffect(() => {
    if (!hasFilters(search, projectFilter, statusFilter)) {
      loadNotes();
    }
  }, [search, projectFilter, statusFilter, loadNotes]);

  const openModalWithFiles = (files: File[]) => {
    setQueuedFiles(files);
    setShowUploadModal(true);
  };

  const handleModalClose = () => {
    setShowUploadModal(false);
    setQueuedFiles([]);
  };

  const handleUploaded = (uploaded: NoteBlock[]) => {
    setNotes((prev) => [...uploaded, ...prev]);
    handleModalClose();
  };

  const startRename = (note: NoteBlock) => {
    setEditingId(note.id);
    setEditingName(note.display_name);
  };

  const commitRename = async (id: number) => {
    const trimmed = editingName.trim();
    if (trimmed) {
      const updated = await updateNote(id, { display_name: trimmed }).catch(() => null);
      if (updated) setNotes((prev) => prev.map((n) => (n.id === id ? updated : n)));
    }
    setEditingId(null);
  };

  const assignProject = async (noteId: number, projectId: number | null) => {
    const updated = await updateNote(noteId, { project_id: projectId }).catch(() => null);
    if (updated) setNotes((prev) => prev.map((n) => (n.id === noteId ? updated : n)));
  };

  const handleDelete = async (noteId: number) => {
    await deleteNote(noteId).catch(() => null);
    setNotes((prev) => prev.filter((n) => n.id !== noteId));
    setSelectedIds((prev) => { const next = new Set(prev); next.delete(noteId); return next; });
  };

  // Bulk actions
  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === notes.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(notes.map((n) => n.id)));
    }
  };

  const handleBulkDelete = async () => {
    const ids = [...selectedIds];
    await bulkDeleteNotes(ids).catch(() => null);
    setNotes((prev) => prev.filter((n) => !selectedIds.has(n.id)));
    setSelectedIds(new Set());
  };

  const handleBulkAssign = async () => {
    if (!bulkProjectId && bulkProjectId !== '0') return;
    const ids = [...selectedIds];
    const projectId = bulkProjectId ? Number(bulkProjectId) : null;
    await bulkUpdateNotes(ids, { project_id: projectId }).catch(() => null);
    setNotes((prev) =>
      prev.map((n) =>
        selectedIds.has(n.id)
          ? { ...n, project_id: projectId, project_name: projects.find((p) => p.id === projectId)?.name ?? null }
          : n
      )
    );
    setSelectedIds(new Set());
    setBulkProjectId('');
  };

  const handleBulkExport = async (format: 'markdown' | 'text') => {
    for (const id of selectedIds) {
      await exportNote(id, format).catch(() => null);
    }
  };

  // Drag to reorder
  const handleDragStart = (e: React.DragEvent, id: number) => {
    dragSrcId.current = id;
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent, id: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverId(id);
  };

  const handleDrop = async (e: React.DragEvent, targetId: number) => {
    e.preventDefault();
    setDragOverId(null);
    const srcId = dragSrcId.current;
    if (!srcId || srcId === targetId) return;

    const srcIdx = notes.findIndex((n) => n.id === srcId);
    const tgtIdx = notes.findIndex((n) => n.id === targetId);
    if (srcIdx === -1 || tgtIdx === -1) return;

    const reordered = [...notes];
    const [moved] = reordered.splice(srcIdx, 1);
    reordered.splice(tgtIdx, 0, moved);

    // Assign sort_order values
    const withOrder = reordered.map((n, i) => ({ ...n, sort_order: i }));
    setNotes(withOrder);

    // Persist only the ones that changed
    const updates = withOrder.filter((n, i) => notes[i]?.id !== n.id || notes[i]?.sort_order !== n.sort_order);
    await Promise.all(
      updates.map((n) => updateNote(n.id, { sort_order: n.sort_order }).catch(() => null))
    );
  };

  const clearFilters = () => {
    setSearch('');
    setProjectFilter('');
    setStatusFilter('');
  };

  const filtersActive = hasFilters(search, projectFilter, statusFilter);

  return (
    <>
      {showUploadModal && (
        <UploadModal
          initialFiles={queuedFiles}
          onClose={handleModalClose}
          onUploaded={handleUploaded}
        />
      )}

      <header className="bg-surface flex justify-between items-center w-full px-margin-desktop py-space-4 border-b border-outline-variant sticky top-0 z-40">
        <div className="flex items-center bg-surface-container-lowest border border-outline-variant rounded-DEFAULT px-3 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all w-96 max-w-full shadow-sm">
          <span className="material-symbols-outlined text-on-surface-variant text-[18px] mr-2">search</span>
          <input
            className="bg-transparent border-none outline-none font-body-sm text-body-sm text-on-surface placeholder:text-on-surface-variant w-full focus:ring-0 p-0"
            placeholder="Search transcripts &amp; summaries…"
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button onClick={() => setSearch('')} className="ml-1 text-on-surface-variant hover:text-on-surface">
              <span className="material-symbols-outlined text-[16px]">close</span>
            </button>
          )}
        </div>
        <div className="flex items-center gap-4 text-on-surface-variant">
          <span className="material-symbols-outlined cursor-pointer hover:text-primary transition-colors hover:bg-surface-container-low p-1 rounded-DEFAULT">sync</span>
          <span className="material-symbols-outlined cursor-pointer hover:text-primary transition-colors hover:bg-surface-container-low p-1 rounded-DEFAULT">notifications</span>
          <span className="material-symbols-outlined cursor-pointer hover:text-primary transition-colors hover:bg-surface-container-low p-1 rounded-DEFAULT">account_circle</span>
        </div>
      </header>

      <main className="w-full max-w-container-max mx-auto px-margin-desktop py-space-8 flex flex-col gap-space-8">

        {/* Upload trigger zone */}
        <section className="w-full">
          <div
            ref={zoneRef}
            className={`bg-surface-container-lowest border border-dashed rounded-xl p-space-6 flex flex-row items-center justify-between gap-4 transition-all duration-200 cursor-pointer group shadow-sm ${
              dragging ? 'border-primary bg-primary-fixed/20' : 'border-outline-variant'
            }`}
            onClick={() => setShowUploadModal(true)}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const files = Array.from(e.dataTransfer.files);
              if (files.length > 0) openModalWithFiles(files);
            }}
          >
            <div className="flex items-center gap-space-4">
              <div className="w-12 h-12 rounded-full bg-surface flex items-center justify-center group-hover:bg-surface-container-low transition-colors duration-300">
                <span className="material-symbols-outlined text-[24px] text-primary">{dragging ? 'download' : 'mic'}</span>
              </div>
              <div>
                <h2 className="font-headline-md text-headline-md text-on-surface mb-1 text-[16px] leading-[24px]">
                  {dragging ? 'Drop to open upload window' : 'New Recording'}
                </h2>
                <p className="font-body-sm text-body-sm text-on-surface-variant">Click to open upload window, or drag recordings here</p>
              </div>
            </div>
            <button
              className="border border-outline-variant bg-surface-container-lowest text-on-surface font-label-md text-label-md px-4 py-1.5 rounded-DEFAULT hover:border-primary hover:text-primary transition-colors shadow-sm"
              onClick={(e) => { e.stopPropagation(); setShowUploadModal(true); }}
            >
              Upload
            </button>
          </div>
        </section>

        {/* Notes Table */}
        <section className="w-full flex flex-col gap-space-4">
          <div className="flex items-center justify-between gap-space-3">
            <h3 className="font-headline-md text-headline-md text-on-surface shrink-0">
              All Recordings
              {notes.length > 0 && (
                <span className="ml-2 font-label-sm text-label-sm text-on-surface-variant bg-surface-container-high rounded-full px-2 py-0.5">
                  {notes.length}
                </span>
              )}
            </h3>
            <div className="flex flex-wrap items-center gap-2">
              <Select
                value={projectFilter}
                onChange={setProjectFilter}
                options={[
                  { value: '', label: 'All projects' },
                  ...projects.map((p) => ({ value: String(p.id), label: p.name })),
                ]}
              />

              <Select
                value={statusFilter}
                onChange={setStatusFilter}
                options={STATUS_OPTIONS}
              />

              {filtersActive && (
                <button
                  onClick={clearFilters}
                  className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant hover:text-error transition-colors px-2 py-1.5 rounded border border-transparent hover:border-outline-variant"
                >
                  <span className="material-symbols-outlined text-[14px]">filter_alt_off</span>
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Bulk action bar */}
          {selectedIds.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 bg-primary-container/20 border border-primary/30 rounded-lg px-3 py-2">
              <span className="font-label-sm text-label-sm text-on-surface">
                {selectedIds.size} selected
              </span>
              <div className="flex items-center gap-2 ml-auto flex-wrap">
                <Select
                  value={bulkProjectId}
                  onChange={setBulkProjectId}
                  options={[
                    { value: '', label: 'Assign project…' },
                    { value: '0', label: 'No Project' },
                    ...projects.map((p) => ({ value: String(p.id), label: p.name })),
                  ]}
                />
                {bulkProjectId !== '' && (
                  <button
                    onClick={handleBulkAssign}
                    className="font-label-sm text-label-sm bg-surface-container-lowest border border-outline-variant rounded px-3 py-1 hover:bg-surface-container-low transition-colors"
                  >
                    Apply
                  </button>
                )}
                <button
                  onClick={() => handleBulkExport('markdown')}
                  className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-3 py-1 hover:bg-surface-container-low transition-colors"
                >
                  <span className="material-symbols-outlined text-[14px]">download</span>
                  Export MD
                </button>
                <button
                  onClick={() => handleBulkExport('text')}
                  className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant border border-outline-variant rounded px-3 py-1 hover:bg-surface-container-low transition-colors"
                >
                  <span className="material-symbols-outlined text-[14px]">download</span>
                  Export TXT
                </button>
                <button
                  onClick={handleBulkDelete}
                  className="flex items-center gap-1 font-label-sm text-label-sm text-error border border-error/30 rounded px-3 py-1 hover:bg-error/10 transition-colors"
                >
                  <span className="material-symbols-outlined text-[14px]">delete</span>
                  Delete
                </button>
                <button
                  onClick={() => setSelectedIds(new Set())}
                  className="text-on-surface-variant hover:text-on-surface"
                >
                  <span className="material-symbols-outlined text-[18px]">close</span>
                </button>
              </div>
            </div>
          )}

          <div className="flex flex-col rounded-xl overflow-hidden bg-surface-container-lowest shadow-sm">
            {/* Table header */}
            <div className="grid grid-cols-[auto_minmax(0,1fr)_140px_130px_56px] xl:grid-cols-[auto_minmax(0,1fr)_140px_160px_130px_56px] gap-3 px-space-4 py-space-2 bg-surface border-b border-outline-variant text-on-surface-variant font-label-sm text-label-sm uppercase tracking-wider items-center">
              <input
                type="checkbox"
                className="rounded cursor-pointer"
                checked={notes.length > 0 && selectedIds.size === notes.length}
                onChange={toggleSelectAll}
                title="Select all"
              />
              <div>Name</div>
              <div className="text-center">Project</div>
              <div className="hidden xl:block text-center">Duration / Size</div>
              <div className="text-center">Status</div>
              <div />
            </div>

            {loading && (
              <div className="px-space-4 py-space-8 text-center text-on-surface-variant font-body-sm text-body-sm">
                Loading…
              </div>
            )}

            {!loading && notes.length === 0 && (
              <div className="px-space-4 py-space-8 text-center text-on-surface-variant font-body-sm text-body-sm">
                {filtersActive ? 'No results match your filters.' : 'No recordings yet — drop a recording above to get started.'}
              </div>
            )}

            {notes.map((note) => (
              <div
                key={note.id}
                draggable={!filtersActive}
                onDragStart={(e) => handleDragStart(e, note.id)}
                onDragOver={(e) => handleDragOver(e, note.id)}
                onDragLeave={() => setDragOverId(null)}
                onDrop={(e) => handleDrop(e, note.id)}
                className={`grid grid-cols-[auto_minmax(0,1fr)_140px_130px_56px] xl:grid-cols-[auto_minmax(0,1fr)_140px_160px_130px_56px] gap-3 items-center px-space-4 py-space-3 border-b border-outline-variant last:border-0 hover:bg-surface-container-low/50 transition-colors relative group ${
                  dragOverId === note.id ? 'border-t-2 border-t-primary' : ''
                } ${selectedIds.has(note.id) ? 'bg-primary-container/10' : ''}`}
              >
                {note.status === 'done' && !selectedIds.has(note.id) && (
                  <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-primary" />
                )}

                {/* Checkbox */}
                <input
                  type="checkbox"
                  className="rounded cursor-pointer"
                  checked={selectedIds.has(note.id)}
                  onChange={() => toggleSelect(note.id)}
                />

                {/* Name */}
                <div className="flex flex-col min-w-0 pr-4">
                  {editingId === note.id ? (
                    <input
                      autoFocus
                      className="font-headline-md text-[14px] leading-[20px] text-on-surface bg-surface-container-low border border-primary rounded px-1 focus:ring-0 w-full"
                      value={editingName}
                      onChange={(e) => setEditingName(e.target.value)}
                      onBlur={() => commitRename(note.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitRename(note.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                    />
                  ) : (
                    <Link
                      to={`/notes/${note.id}`}
                      className="font-headline-md text-[14px] leading-[20px] text-on-surface truncate hover:text-primary transition-colors"
                      onDoubleClick={(e) => { e.preventDefault(); startRename(note); }}
                      title="Click to open · Double-click to rename"
                    >
                      {note.display_name}
                    </Link>
                  )}
                  <p className="font-body-sm text-body-sm text-outline truncate mt-0.5 text-[11px]">
                    {formatDate(note.created_at)}
                  </p>
                </div>

                {/* Project */}
                <div className="min-w-0 flex justify-center">
                  {editingProjectId === note.id ? (
                    <Select
                      defaultOpen
                      value={String(note.project_id ?? '')}
                      onChange={(v) => {
                        assignProject(note.id, v ? Number(v) : null);
                        setEditingProjectId(null);
                      }}
                      onClose={() => setEditingProjectId(null)}
                      options={[
                        { value: '', label: 'No Project' },
                        ...projects.map((p) => ({ value: String(p.id), label: p.name })),
                      ]}
                      className="max-w-[140px]"
                    />
                  ) : (
                    <button
                      className="text-left"
                      onClick={() => setEditingProjectId(note.id)}
                      title="Click to change project"
                    >
                      {note.project_name ? (
                        <span className={`${projectTagBase} ${projectTagClass(note.project_name, note.project_color)} max-w-[130px] truncate`}>
                          {note.project_name}
                        </span>
                      ) : (
                        <span className="font-label-sm text-label-sm text-outline hover:text-on-surface-variant transition-colors">—</span>
                      )}
                    </button>
                  )}
                </div>

                {/* Duration / Size */}
                <div className="hidden xl:block text-on-surface-variant font-label-sm text-label-sm text-center">
                  {formatDuration(note.audio_duration_ms)} / {formatSize(note.audio_file_size)}
                </div>

                {/* Status */}
                <div className="flex justify-center">
                  <StatusBadge status={note.status} />
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    className="text-on-surface-variant hover:text-primary transition-colors p-1"
                    title="Export as Markdown"
                    onClick={() => exportNote(note.id, 'markdown')}
                  >
                    <span className="material-symbols-outlined text-[18px]">download</span>
                  </button>
                  <button
                    className="text-on-surface-variant hover:text-error transition-colors p-1"
                    onClick={() => handleDelete(note.id)}
                    title="Delete"
                  >
                    <span className="material-symbols-outlined text-[18px]">delete</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
