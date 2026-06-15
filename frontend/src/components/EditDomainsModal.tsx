import { useRef, useState } from 'react';
import { updateDomain, deleteDomain, createDomain } from '../api/domains';
import { COLORS, colorSwatchClass, type DomainColor } from '../lib/domains';
import type { Domain } from '../api/types';

interface Props {
  domains: Domain[];
  onClose: () => void;
}

export default function EditDomainsModal({ domains: initial, onClose }: Props) {
  const [list, setList] = useState<Domain[]>(initial);
  const [names, setNames] = useState<Record<number, string>>(
    Object.fromEntries(initial.map((d) => [d.id, d.name])),
  );
  const [colorPickerOpen, setColorPickerOpen] = useState<number | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [newName, setNewName] = useState('');
  const [addingNew, setAddingNew] = useState(false);
  const [saving, setSaving] = useState(false);
  const newInputRef = useRef<HTMLInputElement>(null);

  // ── Drag-and-drop reorder ────────────────────────────────────────────────
  const handleDragStart = (i: number) => setDragIndex(i);

  const handleDragOver = (e: React.DragEvent, i: number) => {
    e.preventDefault();
    if (dragIndex === null || dragIndex === i) return;
    const next = [...list];
    const [moved] = next.splice(dragIndex, 1);
    next.splice(i, 0, moved);
    setList(next);
    setDragIndex(i);
  };

  const handleDragEnd = () => setDragIndex(null);

  // ── Inline name edit ─────────────────────────────────────────────────────
  const commitName = (id: number) => {
    const trimmed = (names[id] ?? '').trim();
    if (!trimmed) {
      setNames((prev) => ({ ...prev, [id]: list.find((d) => d.id === id)?.name ?? '' }));
      return;
    }
    setList((prev) => prev.map((d) => (d.id === id ? { ...d, name: trimmed } : d)));
    updateDomain(id, { name: trimmed }).catch(() => {});
  };

  // ── Color picker ─────────────────────────────────────────────────────────
  const setColor = (id: number, color: DomainColor | null) => {
    setList((prev) => prev.map((d) => (d.id === id ? { ...d, color } : d)));
    setColorPickerOpen(null);
    updateDomain(id, { color }).catch(() => {});
  };

  // ── Delete ───────────────────────────────────────────────────────────────
  const handleDelete = (id: number) => {
    deleteDomain(id).catch(() => {});
    setList((prev) => prev.filter((d) => d.id !== id));
  };

  // ── Add new domain ───────────────────────────────────────────────────────
  const handleAdd = async () => {
    const name = newName.trim();
    if (!name) return;
    setSaving(true);
    const created = await createDomain({ name }).catch(() => null);
    setSaving(false);
    if (created) {
      setList((prev) => [...prev, created]);
      setNames((prev) => ({ ...prev, [created.id]: created.name }));
      setNewName('');
      setAddingNew(false);
    }
  };

  const openAddRow = () => {
    setAddingNew(true);
    setTimeout(() => newInputRef.current?.focus(), 0);
  };

  // ── Done: persist sort order ─────────────────────────────────────────────
  const handleDone = () => {
    list.forEach((d, i) => updateDomain(d.id, { sort_order: i * 10 }).catch(() => {}));
    onClose();
  };

  const dotClass = (d: Domain) =>
    d.color ? `${colorSwatchClass[d.color as DomainColor]}` : 'bg-outline';

  return (
    <div className="fixed inset-0 bg-on-surface/40 backdrop-blur-[2px] z-[60] flex items-center justify-center p-4">
      <div className="bg-surface-container-lowest w-full max-w-md rounded-xl shadow-2xl flex flex-col overflow-hidden">

        {/* Header */}
        <div className="px-space-6 py-space-4 border-b border-outline-variant flex items-center justify-between">
          <h2 className="font-headline-md text-headline-md text-on-surface">Manage Domains</h2>
          <button
            onClick={handleDone}
            className="material-symbols-outlined text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer"
          >
            close
          </button>
        </div>

        {/* Domain rows */}
        <div className="p-space-6 space-y-space-3 max-h-[60vh] overflow-y-auto">
          {list.map((d, i) => (
            <div
              key={d.id}
              draggable
              onDragStart={() => handleDragStart(i)}
              onDragOver={(e) => handleDragOver(e, i)}
              onDragEnd={handleDragEnd}
              className={`flex items-center gap-space-3 p-space-2 bg-surface-container-low rounded border transition-all ${
                dragIndex === i ? 'border-primary opacity-60' : 'border-transparent hover:border-outline-variant'
              }`}
            >
              {/* Drag handle */}
              <span className="material-symbols-outlined text-outline cursor-grab text-[20px] shrink-0">
                drag_indicator
              </span>

              {/* Color dot + picker */}
              <div className="relative shrink-0">
                <button
                  onClick={() => setColorPickerOpen(colorPickerOpen === d.id ? null : d.id)}
                  className={`w-4 h-4 rounded-full ${dotClass(d)} transition-transform hover:scale-125 focus:outline-none`}
                  title="Change color"
                />
                {colorPickerOpen === d.id && (
                  <div className="absolute left-0 top-6 z-10 bg-surface-container-lowest border border-outline-variant rounded-lg shadow-lg p-2 flex flex-wrap gap-1.5 w-[120px]">
                    {COLORS.map((c) => (
                      <button
                        key={c}
                        onClick={() => setColor(d.id, c)}
                        className={`w-5 h-5 rounded-full ${colorSwatchClass[c]} hover:scale-110 transition-transform ${
                          d.color === c ? 'ring-2 ring-offset-1 ring-on-surface/40 scale-110' : ''
                        }`}
                        title={c}
                      />
                    ))}
                    <button
                      onClick={() => setColor(d.id, null)}
                      className={`w-5 h-5 rounded-full border-2 border-outline-variant bg-surface hover:scale-110 transition-transform ${
                        !d.color ? 'ring-2 ring-offset-1 ring-on-surface/40 scale-110' : ''
                      }`}
                      title="Default"
                    />
                  </div>
                )}
              </div>

              {/* Editable name */}
              <input
                className="flex-1 bg-transparent border-none focus:ring-0 font-body-md text-body-md font-medium p-0 text-on-surface placeholder:text-outline min-w-0"
                value={names[d.id] ?? d.name}
                onChange={(e) => setNames((prev) => ({ ...prev, [d.id]: e.target.value }))}
                onBlur={() => commitName(d.id)}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
              />

              {/* Delete */}
              <button
                onClick={() => handleDelete(d.id)}
                className="material-symbols-outlined text-outline hover:text-error transition-colors shrink-0 text-[20px]"
                title="Delete"
              >
                delete
              </button>
            </div>
          ))}

          {/* Add new domain row */}
          {addingNew ? (
            <div className="flex items-center gap-space-3 p-space-2 bg-surface-container-low rounded border border-primary/40">
              <span className="material-symbols-outlined text-outline text-[20px] shrink-0">drag_indicator</span>
              <div className="w-4 h-4 rounded-full bg-outline shrink-0" />
              <input
                ref={newInputRef}
                className="flex-1 bg-transparent border-none focus:ring-0 font-body-md text-body-md font-medium p-0 text-on-surface placeholder:text-outline min-w-0"
                placeholder="Domain name…"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleAdd();
                  if (e.key === 'Escape') { setAddingNew(false); setNewName(''); }
                }}
              />
              <button
                onClick={handleAdd}
                disabled={!newName.trim() || saving}
                className="material-symbols-outlined text-primary hover:text-primary/70 transition-colors shrink-0 text-[20px] disabled:opacity-40"
              >
                check
              </button>
              <button
                onClick={() => { setAddingNew(false); setNewName(''); }}
                className="material-symbols-outlined text-outline hover:text-error transition-colors shrink-0 text-[20px]"
              >
                close
              </button>
            </div>
          ) : (
            <button
              onClick={openAddRow}
              className="w-full flex items-center justify-center gap-2 p-space-2 border-2 border-dashed border-outline-variant rounded-lg text-on-surface-variant font-label-md hover:bg-surface-container hover:text-primary transition-all"
            >
              <span className="material-symbols-outlined text-[18px]">add</span>
              Add New Domain
            </button>
          )}
        </div>

        {/* Footer */}
        <div className="px-space-6 py-space-4 border-t border-outline-variant bg-surface-container-low flex justify-end gap-space-3">
          <button
            onClick={() => onClose()}
            className="px-space-4 py-2 text-on-surface-variant font-label-md hover:bg-surface-container-high rounded transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleDone}
            className="px-space-6 py-2 bg-primary-container text-on-primary font-label-md rounded shadow-sm hover:bg-primary transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
