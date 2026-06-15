import type { NoteStatus } from '../api/types';

interface Props {
  status: NoteStatus;
}

const config: Record<NoteStatus, { dotClass: string; textClass: string; label: string; pulse?: boolean }> = {
  pending: {
    dotClass: 'bg-outline-variant',
    textClass: 'text-on-surface-variant',
    label: 'Pending',
  },
  transcribing: {
    dotClass: 'bg-blue-400',
    textClass: 'text-blue-600',
    label: 'Transcribing',
    pulse: true,
  },
  transcribed: {
    dotClass: 'bg-secondary',
    textClass: 'text-on-surface-variant',
    label: 'Transcribed',
  },
  summarizing: {
    dotClass: 'bg-amber-400',
    textClass: 'text-amber-600',
    label: 'Summarizing',
    pulse: true,
  },
  done: {
    dotClass: 'bg-primary',
    textClass: 'text-primary',
    label: 'Done',
  },
  error: {
    dotClass: 'bg-error',
    textClass: 'text-error',
    label: 'Error',
  },
};

export default function StatusBadge({ status }: Props) {
  const { dotClass, textClass, label, pulse } = config[status];
  return (
    <span className={`inline-flex items-center gap-1.5 font-label-sm text-label-sm ${textClass}`}>
      <span className={`w-2 h-2 rounded-full shrink-0 ${dotClass}${pulse ? ' animate-pulse' : ''}`} />
      {label}
    </span>
  );
}
