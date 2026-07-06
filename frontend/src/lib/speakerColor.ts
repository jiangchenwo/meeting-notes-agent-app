// Deterministic color for a speaker label, so the same name reads the same
// everywhere (transcript segments, speaker chips, roster).
const SPEAKER_COLORS = ['#7c3aed', '#0891b2', '#059669', '#d97706', '#dc2626', '#2563eb', '#db2777', '#65a30d'];

export function speakerColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return SPEAKER_COLORS[h % SPEAKER_COLORS.length];
}
