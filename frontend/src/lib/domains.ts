// Design-system helpers for coloring domains and projects.
//
// Colors are stored on the backend as a plain string (e.g. "blue") or null.
// When null, a stable color is derived from the entity's name so the same
// name always renders the same hue.
//
// Color language is recovered from the original Stitch design
// (stitch-designs/code.html): tinted bg-{c}-50, text-{c}-700, border-{c}-500.
// Shapes: domain labels render as a square box (rounded-sm); project labels
// render as a pill (rounded-full).
//
// NOTE: Tailwind's JIT scans this file, so every class name below must appear
// as a complete literal string. Do not build class names by concatenating a
// color fragment (e.g. `bg-${c}-50`) — those get purged from the build.

export type DomainColor =
  | 'red'
  | 'orange'
  | 'amber'
  | 'green'
  | 'teal'
  | 'cyan'
  | 'blue'
  | 'indigo'
  | 'purple'
  | 'rose';

interface PaletteEntry {
  /** Solid filled circle used in color pickers. */
  swatch: string;
  /** Tinted background + colored foreground for icon containers. */
  icon: string;
  /** Tint + colored text + border color for tag/chip labels (box or pill). */
  tag: string;
  /** Background + border + text for an active (selected) filter pill. */
  pillActive: string;
}

const PALETTE: Record<DomainColor, PaletteEntry> = {
  red: {
    swatch: 'bg-red-500',
    icon: 'bg-red-50 text-red-700',
    tag: 'bg-red-50 text-red-700 border-red-500',
    pillActive: 'bg-red-50 border-red-500 text-red-700',
  },
  orange: {
    swatch: 'bg-orange-500',
    icon: 'bg-orange-50 text-orange-700',
    tag: 'bg-orange-50 text-orange-700 border-orange-500',
    pillActive: 'bg-orange-50 border-orange-500 text-orange-700',
  },
  amber: {
    swatch: 'bg-amber-500',
    icon: 'bg-amber-50 text-amber-700',
    tag: 'bg-amber-50 text-amber-700 border-amber-500',
    pillActive: 'bg-amber-50 border-amber-500 text-amber-700',
  },
  green: {
    swatch: 'bg-green-500',
    icon: 'bg-green-50 text-green-700',
    tag: 'bg-green-50 text-green-700 border-green-500',
    pillActive: 'bg-green-50 border-green-500 text-green-700',
  },
  teal: {
    swatch: 'bg-teal-500',
    icon: 'bg-teal-50 text-teal-700',
    tag: 'bg-teal-50 text-teal-700 border-teal-500',
    pillActive: 'bg-teal-50 border-teal-500 text-teal-700',
  },
  cyan: {
    swatch: 'bg-cyan-500',
    icon: 'bg-cyan-50 text-cyan-700',
    tag: 'bg-cyan-50 text-cyan-700 border-cyan-500',
    pillActive: 'bg-cyan-50 border-cyan-500 text-cyan-700',
  },
  blue: {
    swatch: 'bg-blue-500',
    icon: 'bg-blue-50 text-blue-700',
    tag: 'bg-blue-50 text-blue-700 border-blue-500',
    pillActive: 'bg-blue-50 border-blue-500 text-blue-700',
  },
  indigo: {
    swatch: 'bg-indigo-500',
    icon: 'bg-indigo-50 text-indigo-700',
    tag: 'bg-indigo-50 text-indigo-700 border-indigo-500',
    pillActive: 'bg-indigo-50 border-indigo-500 text-indigo-700',
  },
  purple: {
    swatch: 'bg-purple-500',
    icon: 'bg-purple-50 text-purple-700',
    tag: 'bg-purple-50 text-purple-700 border-purple-500',
    pillActive: 'bg-purple-50 border-purple-500 text-purple-700',
  },
  rose: {
    swatch: 'bg-rose-500',
    icon: 'bg-rose-50 text-rose-700',
    tag: 'bg-rose-50 text-rose-700 border-rose-500',
    pillActive: 'bg-rose-50 border-rose-500 text-rose-700',
  },
};

/** All selectable colors, in picker order. */
export const COLORS = Object.keys(PALETTE) as DomainColor[];

function record(key: keyof PaletteEntry): Record<DomainColor, string> {
  return COLORS.reduce((acc, c) => {
    acc[c] = PALETTE[c][key];
    return acc;
  }, {} as Record<DomainColor, string>);
}

/** Keyed by color → solid swatch classes (for color pickers). */
export const colorSwatchClass = record('swatch');

/** Keyed by color → tinted icon-container classes. */
export const projectIconClass = record('icon');

function isDomainColor(value: string | null | undefined): value is DomainColor {
  return value != null && value in PALETTE;
}

/**
 * Resolve the color to use for an entity. Uses the explicit color when it is a
 * known palette color; otherwise derives a stable color from the name so the
 * same name always renders the same hue.
 */
export function resolveColor(
  name: string | null | undefined,
  color: string | null | undefined,
): DomainColor {
  if (isDomainColor(color)) return color;
  const s = name ?? '';
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = (hash * 31 + s.charCodeAt(i)) >>> 0;
  }
  return COLORS[hash % COLORS.length];
}

/** Base classes for a domain label: a colored square box. */
export const domainTagBase =
  'inline-flex items-center px-2 py-0.5 rounded-sm border text-xs font-medium tracking-wide';

/** Color classes for a domain tag label. */
export function domainTagClass(
  name: string | null | undefined,
  color: string | null | undefined,
): string {
  return PALETTE[resolveColor(name, color)].tag;
}

/** Base classes for a project label: a colored pill. */
export const projectTagBase =
  'inline-flex items-center px-2.5 py-0.5 rounded-full border text-xs font-medium tracking-wide';

/** Color classes for a project tag label. */
export function projectTagClass(
  name: string | null | undefined,
  color: string | null | undefined,
): string {
  return PALETTE[resolveColor(name, color)].tag;
}

/**
 * Color classes for a domain filter pill. Active pills use the domain color;
 * inactive pills are neutral (the caller supplies base padding + `border`).
 */
export function domainPillClass(
  name: string | null | undefined,
  color: string | null | undefined,
  active: boolean,
): string {
  if (!active) {
    return 'bg-transparent border-outline-variant text-on-surface-variant hover:bg-surface-container-high';
  }
  return PALETTE[resolveColor(name, color)].pillActive;
}
