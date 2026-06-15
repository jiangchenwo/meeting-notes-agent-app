import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  className?: string;
  size?: 'sm' | 'md';
  defaultOpen?: boolean;
  onClose?: () => void;
}

interface MenuCoords {
  top: number;
  left: number;
  width: number;
}

export default function Select({ value, onChange, options, className, size = 'sm', defaultOpen, onClose }: SelectProps) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const [coords, setCoords] = useState<MenuCoords | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const close = () => {
    setOpen(false);
    onClose?.();
  };

  // Position the portalled menu relative to the trigger so it escapes any
  // overflow-hidden ancestor (e.g. the rounded notes table).
  useLayoutEffect(() => {
    if (!open || !buttonRef.current) return;
    const update = () => {
      const r = buttonRef.current!.getBoundingClientRect();
      const menuH = menuRef.current?.offsetHeight ?? 0;
      const spaceBelow = window.innerHeight - r.bottom;
      const openUp = menuH > 0 && spaceBelow < menuH + 8 && r.top > spaceBelow;
      setCoords({
        top: openUp ? r.top - menuH - 4 : r.bottom + 4,
        left: r.left,
        width: r.width,
      });
    };
    update();
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const t = e.target as Node;
      if (containerRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      close();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const selected = options.find((o) => o.value === value);
  const py = size === 'md' ? 'py-2' : 'py-1.5';
  const font = size === 'md' ? 'font-body-md text-body-md' : 'font-body-sm text-body-sm';

  return (
    <div ref={containerRef} className={`relative ${className ?? ''}`}>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => open ? close() : setOpen(true)}
        className={`flex items-center w-full bg-surface-container-lowest border border-outline-variant rounded px-3 ${py} pr-8 ${font} focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all cursor-pointer shadow-sm`}
      >
        <span className={`flex-1 text-left truncate ${selected ? 'text-on-surface' : 'text-on-surface-variant'}`}>
          {selected?.label ?? ''}
        </span>
        <span className={`material-symbols-outlined absolute right-2.5 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none text-[18px] transition-transform duration-150 ${open ? 'rotate-180' : ''}`}>
          expand_more
        </span>
      </button>

      {open && coords && createPortal(
        <div
          ref={menuRef}
          style={{ position: 'fixed', top: coords.top, left: coords.left, minWidth: coords.width }}
          className="z-50 bg-white border border-outline-variant rounded-lg shadow-lg py-1 max-w-[280px] max-h-60 overflow-y-auto"
        >
          {options.map((opt) => (
            <button
              key={opt.value}
              type="button"
              className={`w-full flex items-center gap-2 text-left px-3 py-2 font-body-sm text-body-sm cursor-pointer transition-colors ${
                opt.value === value
                  ? 'text-primary bg-primary/5'
                  : 'text-on-surface hover:bg-surface-container-low'
              }`}
              onClick={() => {
                onChange(opt.value);
                close();
              }}
            >
              <span className="flex-1">{opt.label}</span>
              {opt.value === value && (
                <span className="material-symbols-outlined text-[16px] text-primary shrink-0">check</span>
              )}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}
