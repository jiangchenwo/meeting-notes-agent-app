import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { resolveColor, projectIconClass } from '../lib/domains';
import { getProjects, createProject, deleteProject } from '../api/projects';
import type { Project } from '../api/types';

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = (now.getTime() - d.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
  return d.toLocaleDateString();
}

function fmtSize(bytes: number): string {
  if (!bytes) return '';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}


export default function ProjectsOverview() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const newNameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getProjects()
      .then(setProjects)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (showCreate) newNameRef.current?.focus();
  }, [showCreate]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      const p = await createProject({ name });
      setProjects((prev) => [p, ...prev].sort((a, b) => a.name.localeCompare(b.name)));
      setNewName('');
      setShowCreate(false);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: number) => {
    await deleteProject(id).catch(() => null);
    setProjects((prev) => prev.filter((p) => p.id !== id));
  };

  const filtered = projects.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main className="flex-1 w-full flex flex-col min-h-screen">
      <header className="flex justify-between items-center px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full sticky top-0 bg-surface/90 backdrop-blur-md z-30 border-b border-outline-variant/30">
        <div>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface tracking-tight">
            Projects
          </h2>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Organise your recordings into projects for richer AI context.
          </p>
        </div>
      </header>

      <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full space-y-space-4">
        {/* Search + New Project */}
        <div className="flex items-center justify-between gap-space-3">
          <div className="flex flex-1 items-center bg-surface-container-lowest border border-outline-variant rounded px-3 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all shadow-sm max-w-xs">
            <span className="material-symbols-outlined text-on-surface-variant text-[18px] mr-2">search</span>
            <input
              className="flex-1 bg-transparent border-none outline-none focus:ring-0 font-body-sm text-body-sm text-on-surface placeholder:text-on-surface-variant"
              placeholder="Search projects…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <button
            onClick={() => setShowCreate((v) => !v)}
            className="bg-primary text-on-primary font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity shadow-sm"
          >
            <span className="material-symbols-outlined text-[18px]">add</span>
            New Project
          </button>
        </div>

        {/* New project form */}
        {showCreate && (
          <div className="bg-surface-container-lowest border border-primary/30 rounded-lg p-space-4 flex flex-col gap-3">
            <h4 className="font-label-md text-label-md text-on-surface">New Project</h4>
            <input
              ref={newNameRef}
              className="w-full bg-surface border border-outline-variant rounded px-3 py-2 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-0"
              placeholder="Project name…"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreate();
                if (e.key === 'Escape') { setShowCreate(false); setNewName(''); }
              }}
            />
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => { setShowCreate(false); setNewName(''); }}
                className="px-4 py-1.5 text-on-surface-variant font-label-md text-label-md hover:text-on-surface transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={!newName.trim() || creating}
                className="px-4 py-1.5 bg-primary text-on-primary rounded font-label-md text-label-md hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {creating ? 'Creating…' : 'Create Project'}
              </button>
            </div>
          </div>
        )}

        {/* Project list */}
        <div className="flex flex-col gap-space-3">
          {loading && (
            <div className="p-space-6 text-center text-on-surface-variant font-body-sm text-body-sm">
              Loading projects…
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="bg-surface-container-lowest rounded-lg p-space-8 text-center text-on-surface-variant font-body-sm text-body-sm">
              {projects.length === 0
                ? 'No projects yet — click "New Project" to create one.'
                : 'No projects match your search.'}
            </div>
          )}

          {filtered.map((project) => (
            <div
              key={project.id}
              className="flex gap-4 px-space-4 py-space-4 justify-between bg-surface-container-lowest rounded-lg hover:shadow-sm transition-all group"
            >
              <Link to={`/projects/${project.id}`} className="flex items-start gap-4 flex-1 min-w-0">
                <div className={`flex items-center justify-center rounded shrink-0 size-10 mt-0.5 transition-colors ${projectIconClass[resolveColor(project.name, project.color)]}`}>
                  <span className="material-symbols-outlined">{project.icon || 'folder'}</span>
                </div>
                <div className="flex flex-col gap-1 min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-on-surface font-body-md font-medium truncate">{project.name}</p>
                    <span className="font-label-sm text-label-sm text-outline shrink-0">
                      Updated {formatDate(project.updated_at)}
                    </span>
                  </div>
                  {project.description && (
                    <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-1">{project.description}</p>
                  )}
                  <div className="flex items-center justify-between gap-3 mt-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      {project.top_domains.map((d) => (
                        <span key={d} className="inline-flex items-center px-2 py-0.5 text-xs font-medium text-on-surface-variant bg-surface-container rounded">{d}</span>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="font-label-sm text-label-sm text-on-surface-variant">
                        {project.note_count} {project.note_count === 1 ? 'recording' : 'recordings'}
                      </span>
                      {project.total_size > 0 && (
                        <span className="flex items-center gap-0.5 font-label-sm text-label-sm text-outline">
                          <span className="material-symbols-outlined text-[13px]">storage</span>
                          {fmtSize(project.total_size)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
              <div className="shrink-0 flex items-center">
                <button
                  className="text-outline hover:text-error p-1 rounded hover:bg-surface-variant opacity-0 group-hover:opacity-100 transition-all"
                  onClick={() => handleDelete(project.id)}
                  title="Delete project"
                >
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
