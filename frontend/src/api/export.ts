export async function exportNote(id: number, format: 'markdown' | 'text'): Promise<void> {
  const response = await fetch(`/api/notes/${id}/export?format=${format}`);
  if (!response.ok) throw new Error('Export failed');
  const blob = await response.blob();
  const disposition = response.headers.get('Content-Disposition') ?? '';
  const match = disposition.match(/filename="(.+?)"/);
  const filename = match?.[1] ?? `note.${format === 'markdown' ? 'md' : 'txt'}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function copyText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}
