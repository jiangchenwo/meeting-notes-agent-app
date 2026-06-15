import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  getLabFixture,
  runLabPipeline,
  labTranscribeAudio,
  getAudioLibrary,
  runBatchItem,
  type LabRunResult,
  type WorkflowStep,
  type ActionItem,
  type WorkflowOverride,
  type AudioLibraryFile,
  type RougeScores,
  type BertScore,
} from '../../api/lab';
import { getLMStudioStatus } from '../../api/summarize';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DOMAINS = ['Education', 'Healthcare', 'Interview', 'Project'];

const ALL_AGENTS = [
  'Summarizer',
  'ActionItemExtractor',
  'DecisionLogger',
  'InterviewAgent',
  'LectureAgent',
] as const;

type AgentName = typeof ALL_AGENTS[number];
type Slot = 'A' | 'B';
type Mode = 'single' | 'ab' | 'batch';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(ms: number) { return (ms / 1000).toFixed(1) + 's'; }
function pct(v: number) { return (v * 100).toFixed(0) + '%'; }
function delta(a: number, b: number) {
  const d = b - a;
  return (d >= 0 ? '+' : '') + (d * 100).toFixed(0) + '%';
}
function betterColor(a: number, b: number, higherIsBetter = true) {
  if (higherIsBetter ? b > a : b < a) return 'text-primary font-semibold';
  if (higherIsBetter ? a > b : a < b) return 'text-on-surface-variant';
  return 'text-on-surface';
}
function formatBytes(b: number) {
  return b > 1_000_000 ? (b / 1_000_000).toFixed(1) + ' MB' : Math.round(b / 1024) + ' KB';
}

// ---------------------------------------------------------------------------
// ScoreBar
// ---------------------------------------------------------------------------
function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? 'bg-primary' : pct >= 60 ? 'bg-yellow-500' : 'bg-error';
  return (
    <div className="flex items-center gap-2 min-w-[110px]">
      <div className="flex-1 h-1.5 bg-surface-container rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-label-md text-label-md text-on-surface w-9 text-right tabular-nums">{pct}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// WorkflowStep row
// ---------------------------------------------------------------------------
function StepRow({ step, index }: { step: WorkflowStep; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isOk = step.status === 'done';
  const isCritique = step.phase === 'critique';
  const score = step.critique_score;
  return (
    <div className="border border-outline-variant rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-surface-container-low transition-colors"
      >
        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-surface-container flex items-center justify-center font-label-sm text-label-sm text-on-surface-variant">{index + 1}</span>
        <span className="flex-1 font-body-md text-body-md text-on-surface truncate">{step.name}</span>
        {step.attempt > 1 && <span className="px-1.5 py-0.5 rounded bg-surface-container font-label-sm text-label-sm text-on-surface-variant">retry #{step.attempt}</span>}
        <span className={`px-2 py-0.5 rounded-full font-label-sm text-label-sm ${isCritique ? 'bg-secondary-container text-secondary' : 'bg-primary-fixed text-on-primary-fixed'}`}>
          {isCritique ? 'critique' : 'extract'}
        </span>
        {score != null && (
          <span className={`px-2 py-0.5 rounded-full font-label-md text-label-md tabular-nums ${score >= 8 ? 'bg-green-100 text-green-800' : score >= 6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}`}>
            {score.toFixed(1)}/10
          </span>
        )}
        <span className={`material-symbols-outlined text-[16px] ${isOk ? 'text-primary' : 'text-error'}`}>{isOk ? 'check_circle' : 'error'}</span>
        <span className="font-label-md text-label-md text-on-surface-variant w-12 text-right tabular-nums">{fmt(step.duration_ms)}</span>
        <span className="material-symbols-outlined text-[16px] text-on-surface-variant">{expanded ? 'expand_less' : 'expand_more'}</span>
      </button>
      {expanded && (
        <div className="border-t border-outline-variant bg-surface-container-lowest">
          {step.error && <div className="px-4 pt-3 pb-1 text-error font-body-sm text-body-sm">{step.error}</div>}
          {isCritique && Array.isArray((step.output as Record<string, unknown>).issues) && (
            <div className="px-4 pt-3">
              <p className="font-label-md text-label-md text-on-surface-variant mb-1">Issues flagged</p>
              <ul className="list-disc pl-4 space-y-1">
                {((step.output as Record<string, unknown>).issues as string[]).map((issue, i) => (
                  <li key={i} className="font-body-sm text-body-sm text-on-surface">{issue}</li>
                ))}
              </ul>
            </div>
          )}
          <details className="px-4 py-3">
            <summary className="cursor-pointer font-label-md text-label-md text-on-surface-variant mb-2 select-none">Raw JSON</summary>
            <pre className="text-[11px] font-mono text-on-surface overflow-auto max-h-64 bg-surface-container p-3 rounded-lg mt-1">{JSON.stringify(step.output, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// WorkflowEditor
// ---------------------------------------------------------------------------
function WorkflowEditor({ value, onChange }: { value: WorkflowOverride | null; onChange: (v: WorkflowOverride | null) => void }) {
  const [useDefault, setUseDefault] = useState(value === null);
  const [steps, setSteps] = useState<AgentName[]>(value?.steps as AgentName[] ?? ['Summarizer', 'ActionItemExtractor']);
  const [critiqueSteps, setCritiqueSteps] = useState<Set<string>>(new Set(value?.critique_steps ?? ['Summarizer']));
  const [threshold, setThreshold] = useState(value?.critique_threshold ?? 7.0);
  const [retries, setRetries] = useState(value?.max_retries ?? 1);

  const emit = (s: AgentName[], cs: Set<string>, t: number, r: number) => {
    onChange({ steps: s, critique_steps: [...cs].filter(a => s.includes(a as AgentName)), critique_threshold: t, max_retries: r });
  };

  const toggle = (agent: AgentName) => {
    const next = steps.includes(agent) ? steps.filter(a => a !== agent) : [...steps, agent];
    setSteps(next as AgentName[]);
    emit(next as AgentName[], critiqueSteps, threshold, retries);
  };
  const move = (i: number, dir: -1 | 1) => {
    const next = [...steps];
    const j = i + dir;
    if (j < 0 || j >= next.length) return;
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
    emit(next, critiqueSteps, threshold, retries);
  };
  const toggleCritique = (agent: string) => {
    const next = new Set(critiqueSteps);
    next.has(agent) ? next.delete(agent) : next.add(agent);
    setCritiqueSteps(next);
    emit(steps, next, threshold, retries);
  };

  return (
    <details className="border border-outline-variant rounded-lg overflow-hidden">
      <summary className="cursor-pointer px-4 py-2.5 font-label-md text-label-md text-on-surface-variant hover:bg-surface-container-low select-none flex items-center gap-2">
        <span className="material-symbols-outlined text-[16px]">tune</span>
        Workflow editor
        {value && <span className="ml-auto px-2 py-0.5 rounded-full bg-primary text-on-primary font-label-sm text-label-sm">custom</span>}
      </summary>
      <div className="p-4 border-t border-outline-variant space-y-4">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={useDefault} onChange={e => { setUseDefault(e.target.checked); if (e.target.checked) onChange(null); else emit(steps, critiqueSteps, threshold, retries); }} className="rounded" />
          <span className="font-body-md text-body-md text-on-surface">Use domain default workflow</span>
        </label>
        {!useDefault && (
          <div className="space-y-3">
            <p className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide">Agent steps</p>
            <div className="space-y-1.5">
              {ALL_AGENTS.map((agent) => {
                const inSteps = steps.includes(agent);
                const stepIdx = steps.indexOf(agent);
                return (
                  <div key={agent} className={`flex items-center gap-2 px-3 py-2 rounded-lg ${inSteps ? 'bg-surface-container' : 'opacity-50'}`}>
                    <input type="checkbox" checked={inSteps} onChange={() => toggle(agent)} className="rounded flex-shrink-0" />
                    <span className="flex-1 font-body-md text-body-md text-on-surface">{agent}</span>
                    {inSteps && (
                      <>
                        <label className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant cursor-pointer">
                          <input type="checkbox" checked={critiqueSteps.has(agent)} onChange={() => toggleCritique(agent)} className="rounded" />
                          critique
                        </label>
                        <div className="flex gap-0.5">
                          <button onClick={() => move(stepIdx, -1)} disabled={stepIdx === 0} className="p-0.5 rounded text-on-surface-variant hover:text-on-surface disabled:opacity-30">
                            <span className="material-symbols-outlined text-[16px]">arrow_upward</span>
                          </button>
                          <button onClick={() => move(stepIdx, 1)} disabled={stepIdx === steps.length - 1} className="p-0.5 rounded text-on-surface-variant hover:text-on-surface disabled:opacity-30">
                            <span className="material-symbols-outlined text-[16px]">arrow_downward</span>
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="grid grid-cols-2 gap-4 pt-1">
              <div>
                <label className="font-label-md text-label-md text-on-surface-variant block mb-1">Critique threshold: {threshold.toFixed(1)}</label>
                <input type="range" min={5} max={10} step={0.5} value={threshold}
                  onChange={e => { const v = parseFloat(e.target.value); setThreshold(v); emit(steps, critiqueSteps, v, retries); }}
                  className="w-full" />
              </div>
              <div>
                <label className="font-label-md text-label-md text-on-surface-variant block mb-1">Max retries</label>
                <select value={retries} onChange={e => { const v = parseInt(e.target.value); setRetries(v); emit(steps, critiqueSteps, threshold, v); }}
                  className="w-full bg-surface-container rounded-lg px-2 py-1.5 font-body-sm text-body-sm text-on-surface border border-outline-variant">
                  {[0, 1, 2, 3].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>
            <div className="pt-1 text-primary font-body-sm text-body-sm">
              Steps: {steps.join(' → ')}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}

// ---------------------------------------------------------------------------
// AudioDropZone
// ---------------------------------------------------------------------------
function AudioDropZone({ onTranscript }: { onTranscript: (text: string, model: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };

  const handleFile = (f: File) => { setFile(f); setError(null); };

  const transcribe = async () => {
    if (!file) return;
    setIsTranscribing(true);
    setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    try {
      const res = await labTranscribeAudio(file);
      onTranscript(res.transcript, res.model_used);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      stopTimer();
      setIsTranscribing(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={e => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${isDragging ? 'border-primary bg-primary-fixed' : 'border-outline-variant hover:border-primary hover:bg-surface-container-low'}`}
      >
        <span className="material-symbols-outlined text-[36px] text-on-surface-variant block mb-2">upload_file</span>
        {file ? (
          <div>
            <p className="font-body-md text-body-md text-on-surface font-medium">{file.name}</p>
            <p className="font-body-sm text-body-sm text-on-surface-variant">{formatBytes(file.size)}</p>
          </div>
        ) : (
          <p className="font-body-md text-body-md text-on-surface-variant">Drop audio file or click to browse<br /><span className="font-body-sm text-body-sm">.mp3 .wav .m4a .ogg .webm</span></p>
        )}
        <input ref={inputRef} type="file" accept=".mp3,.wav,.m4a,.ogg,.webm" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
      </div>
      {error && <p className="font-body-sm text-body-sm text-error">{error}</p>}
      {file && (
        <button onClick={transcribe} disabled={isTranscribing}
          className="flex items-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-lg font-label-md text-label-md hover:opacity-90 disabled:opacity-40">
          {isTranscribing
            ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span>Transcribing… {elapsed}s</>
            : <><span className="material-symbols-outlined text-[16px]">graphic_eq</span>Transcribe with Whisper</>}
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// OutputTabs
// ---------------------------------------------------------------------------
function OutputTabs({ result }: { result: LabRunResult }) {
  const [active, setActive] = useState('summary');
  const tabs = [
    { id: 'summary', label: 'Summary' },
    { id: 'actions', label: `Actions (${result.action_items.length})` },
    { id: 'domain', label: 'Domain Output' },
    { id: 'naive', label: 'Naive Baseline' },
  ];
  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl overflow-hidden">
      <div className="flex border-b border-outline-variant overflow-x-auto">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActive(t.id)}
            className={`px-4 py-3 font-label-md text-label-md whitespace-nowrap transition-colors ${active === t.id ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant hover:text-on-surface'}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="p-5">
        {active === 'summary' && (result.summary_text
          ? <div className="prose prose-sm max-w-none prose-headings:text-on-surface prose-p:text-on-surface prose-li:text-on-surface prose-strong:text-on-surface"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary_text}</ReactMarkdown></div>
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No summary generated.</p>
        )}
        {active === 'actions' && (result.action_items.length > 0
          ? <div className="space-y-2">
            {result.action_items.map((item: ActionItem, i: number) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-surface-container rounded-lg">
                <span className={`mt-0.5 px-1.5 py-0.5 rounded font-label-sm text-label-sm flex-shrink-0 ${item.priority === 'high' ? 'bg-error-container text-on-error-container' : 'bg-surface-container-high text-on-surface-variant'}`}>{item.priority}</span>
                <div className="flex-1 min-w-0">
                  <p className="font-body-md text-body-md text-on-surface">{item.task}</p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="font-label-sm text-label-sm text-on-surface-variant">Owner: {item.owner || 'TBD'}</span>
                    {item.deadline && <span className="font-label-sm text-label-sm text-on-surface-variant">Due: {item.deadline}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No action items extracted.</p>
        )}
        {active === 'domain' && (result.suggestions_text
          ? <div className="prose prose-sm max-w-none prose-headings:text-on-surface prose-p:text-on-surface prose-li:text-on-surface prose-strong:text-on-surface"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.suggestions_text}</ReactMarkdown></div>
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No domain-specific output.</p>
        )}
        {active === 'naive' && (
          <div>
            <div className="mb-3 px-3 py-2 bg-surface-container rounded-lg">
              <p className="font-label-sm text-label-sm text-on-surface-variant">
                Single-call summarizer — no structured extraction, no agents, no critique.
                Coverage: {pct(result.eval.naive_coverage)} vs agentic {pct(result.eval.agent_coverage)}
              </p>
            </div>
            {result.naive_summary
              ? <div className="prose prose-sm max-w-none prose-headings:text-on-surface prose-p:text-on-surface prose-li:text-on-surface prose-strong:text-on-surface"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.naive_summary}</ReactMarkdown></div>
              : <p className="text-on-surface-variant font-body-sm text-body-sm">Naive baseline not available.</p>
            }
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvalPanel
// ---------------------------------------------------------------------------
function MetricRow({ label, agValue, naiveValue, higherBetter = true, fmt: fmtFn = pct }: {
  label: string; agValue: number | null; naiveValue: number | null; higherBetter?: boolean; fmt?: (v: number) => string;
}) {
  if (agValue == null) return null;
  const delta = naiveValue != null ? (agValue - naiveValue) * (higherBetter ? 1 : -1) : null;
  return (
    <tr>
      <td className="py-1.5 pr-4 font-body-sm text-body-sm text-on-surface-variant">{label}</td>
      <td className="py-1.5 pr-4 font-label-md text-label-md text-on-surface tabular-nums">{fmtFn(agValue)}</td>
      <td className="py-1.5 pr-4 font-label-md text-label-md text-on-surface-variant tabular-nums">{naiveValue != null ? fmtFn(naiveValue) : '—'}</td>
      <td className={`py-1.5 font-label-md text-label-md tabular-nums ${delta != null ? (delta > 0.005 ? 'text-primary' : delta < -0.005 ? 'text-error' : 'text-on-surface-variant') : ''}`}>
        {delta != null ? (delta >= 0 ? '+' : '') + fmtFn(Math.abs(delta)) : '—'}
      </td>
    </tr>
  );
}

function RougeBlock({ label, scores, naiveScores }: { label: string; scores: RougeScores | null | undefined; naiveScores?: RougeScores | null }) {
  if (!scores) return null;
  return (
    <div>
      <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-1">{label}</p>
      <table className="w-full text-left border-collapse">
        <thead><tr>
          <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant w-20">Metric</th>
          <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Agentic</th>
          {naiveScores && <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Naive</th>}
          {naiveScores && <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Δ</th>}
        </tr></thead>
        <tbody>
          {(['rouge1', 'rouge2', 'rougeL'] as const).map(k => (
            <tr key={k}>
              <td className="py-0.5 font-label-md text-label-md text-on-surface-variant">{k}</td>
              <td className="py-0.5 font-label-md text-label-md text-on-surface tabular-nums">{pct(scores[k])}</td>
              {naiveScores && <td className="py-0.5 font-label-md text-label-md text-on-surface-variant tabular-nums">{pct(naiveScores[k])}</td>}
              {naiveScores && <td className={`py-0.5 font-label-md text-label-md tabular-nums ${scores[k] - naiveScores[k] > 0.005 ? 'text-primary' : 'text-on-surface-variant'}`}>{(scores[k] - naiveScores[k] >= 0 ? '+' : '') + pct(scores[k] - naiveScores[k])}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BertBlock({ label, scores, naiveScores }: { label: string; scores: BertScore | null | undefined; naiveScores?: BertScore | null }) {
  if (!scores) return null;
  return (
    <div>
      <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-1">{label}</p>
      <table className="w-full text-left border-collapse">
        <thead><tr>
          <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant w-20">Metric</th>
          <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Agentic</th>
          {naiveScores && <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Naive</th>}
          {naiveScores && <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Δ</th>}
        </tr></thead>
        <tbody>
          {(['precision', 'recall', 'f1'] as const).map(k => (
            <tr key={k}>
              <td className="py-0.5 font-label-md text-label-md text-on-surface-variant">{k}</td>
              <td className="py-0.5 font-label-md text-label-md text-on-surface tabular-nums">{pct(scores[k])}</td>
              {naiveScores && <td className="py-0.5 font-label-md text-label-md text-on-surface-variant tabular-nums">{pct(naiveScores[k])}</td>}
              {naiveScores && <td className={`py-0.5 font-label-md text-label-md tabular-nums ${scores[k] - naiveScores[k] > 0.005 ? 'text-primary' : 'text-on-surface-variant'}`}>{(scores[k] - naiveScores[k] >= 0 ? '+' : '') + pct(scores[k] - naiveScores[k])}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvalPanel({ result }: { result: LabRunResult }) {
  const ev = result.eval;
  const schemaEntries = Object.entries(ev.schema_check ?? {});
  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="font-headline-md text-headline-md text-on-surface">Evaluation</h3>
        {result.chunked && (
          <span className="px-2 py-0.5 rounded-full bg-secondary-container text-secondary font-label-sm text-label-sm">chunked ({result.chunk_count} segments)</span>
        )}
      </div>

      {/* Coverage + action recall */}
      <div>
        <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-2">
          Ground-truth coverage ({ev.ground_truth_facts} facts)
        </p>
        <table className="w-full text-left border-collapse">
          <thead><tr>
            <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant"></th>
            <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Agentic</th>
            <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Naive</th>
            <th className="pb-1 font-label-sm text-label-sm text-on-surface-variant">Δ</th>
          </tr></thead>
          <tbody>
            <MetricRow label="Coverage" agValue={ev.agent_coverage} naiveValue={ev.naive_coverage} />
            <MetricRow label="Action recall" agValue={ev.action_recall} naiveValue={0} />
          </tbody>
        </table>
      </div>

      <div className="border-t border-outline-variant" />

      {/* ROUGE */}
      <RougeBlock label="ROUGE vs transcript" scores={ev.rouge_vs_transcript} naiveScores={ev.rouge_naive_vs_transcript} />

      {ev.rouge_vs_transcript && <div className="border-t border-outline-variant" />}

      {/* BERTScore */}
      <BertBlock label="BERTScore vs transcript" scores={ev.bertscore_vs_transcript} naiveScores={ev.bertscore_naive_vs_transcript} />

      {ev.bertscore_vs_transcript && <div className="border-t border-outline-variant" />}

      {/* Schema */}
      {schemaEntries.length > 0 && (
        <>
          <div className="space-y-2">
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Schema compliance</p>
            <div className="flex flex-wrap gap-1.5">
              {schemaEntries.map(([agent, ok]) => (
                <span key={agent} className={`flex items-center gap-1 px-2 py-0.5 rounded-full font-label-sm text-label-sm ${ok ? 'bg-primary-fixed text-on-primary-fixed' : 'bg-error-container text-on-error-container'}`}>
                  <span className="material-symbols-outlined text-[12px]">{ok ? 'check' : 'close'}</span>
                  {agent.replace('Agent', '')}
                </span>
              ))}
            </div>
          </div>
          <div className="border-t border-outline-variant" />
        </>
      )}

      {/* Hallucinations */}
      <div className="space-y-1">
        <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Suspected hallucinations</p>
        {ev.hallucination_count === 0
          ? <span className="font-body-sm text-body-sm text-primary flex items-center gap-1"><span className="material-symbols-outlined text-[14px]">check_circle</span>None detected</span>
          : <div className="flex flex-wrap gap-1">{ev.hallucinations.slice(0, 20).map(w => <span key={w} className="px-1.5 py-0.5 bg-error-container text-on-error-container rounded font-label-sm text-label-sm">{w}</span>)}</div>}
      </div>

      <div className="border-t border-outline-variant" />

      {/* Run stats */}
      <div className="grid grid-cols-2 gap-3">
        <div><p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Total time</p>
          <p className="font-headline-md text-headline-md text-on-surface tabular-nums">{fmt(result.total_ms)}</p></div>
        {result.confidence_score != null && (
          <div><p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Confidence</p>
            <p className="font-headline-md text-headline-md text-on-surface tabular-nums">{result.confidence_score.toFixed(1)}/10</p></div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ABComparisonView
// ---------------------------------------------------------------------------
function ABComparisonView({ a, b }: { a: LabRunResult; b: LabRunResult }) {
  const [showing, setShowing] = useState<Slot>('A');
  const ev_a = a.eval;
  const ev_b = b.eval;
  type Row = { label: string; va: number | null; vb: number | null; hi: boolean; fmtFn?: (v: number) => string };
  const rows: Row[] = [
    { label: 'Coverage', va: ev_a.agent_coverage, vb: ev_b.agent_coverage, hi: true },
    { label: 'Action recall', va: ev_a.action_recall, vb: ev_b.action_recall, hi: true },
    { label: 'ROUGE-L', va: ev_a.rouge_vs_transcript?.rougeL ?? null, vb: ev_b.rouge_vs_transcript?.rougeL ?? null, hi: true },
    { label: 'BERTScore F1', va: ev_a.bertscore_vs_transcript?.f1 ?? null, vb: ev_b.bertscore_vs_transcript?.f1 ?? null, hi: true },
    { label: 'Confidence', va: a.confidence_score, vb: b.confidence_score, hi: true, fmtFn: v => v.toFixed(1) + '/10' },
    { label: 'Total time', va: a.total_ms / 1000, vb: b.total_ms / 1000, hi: false, fmtFn: v => v.toFixed(1) + 's' },
  ];
  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-4">
      <h3 className="font-headline-md text-headline-md text-on-surface">A/B Comparison</h3>
      <div className="grid grid-cols-2 gap-2 text-center">
        <div className="bg-surface-container rounded-lg p-2">
          <p className="font-label-sm text-label-sm text-on-surface-variant">Run A workflow</p>
          <p className="font-body-sm text-body-sm text-on-surface">{a.workflow_plan.steps.join(' → ')}</p>
        </div>
        <div className="bg-surface-container rounded-lg p-2">
          <p className="font-label-sm text-label-sm text-on-surface-variant">Run B workflow</p>
          <p className="font-body-sm text-body-sm text-on-surface">{b.workflow_plan.steps.join(' → ')}</p>
        </div>
      </div>
      <table className="w-full text-left border-collapse">
        <thead><tr>
          <th className="pb-2 font-label-sm text-label-sm text-on-surface-variant">Metric</th>
          <th className="pb-2 font-label-sm text-label-sm text-on-surface-variant text-right">Run A</th>
          <th className="pb-2 font-label-sm text-label-sm text-on-surface-variant text-right">Run B</th>
          <th className="pb-2 font-label-sm text-label-sm text-on-surface-variant text-right">Δ (B−A)</th>
        </tr></thead>
        <tbody>
          {rows.filter(r => r.va != null || r.vb != null).map(({ label, va, vb, hi, fmtFn: f = pct }) => {
            const d = va != null && vb != null ? (vb - va) : null;
            const dGood = d != null ? (hi ? d > 0.005 : d < -0.005) : false;
            return (
              <tr key={label} className="border-t border-outline-variant">
                <td className="py-2 font-body-sm text-body-sm text-on-surface-variant">{label}</td>
                <td className={`py-2 font-label-md text-label-md text-right tabular-nums ${va != null && vb != null ? betterColor(va, vb, hi) : ''}`}>{va != null ? f(va) : '—'}</td>
                <td className={`py-2 font-label-md text-label-md text-right tabular-nums ${va != null && vb != null ? betterColor(vb, va, hi) : ''}`}>{vb != null ? f(vb) : '—'}</td>
                <td className={`py-2 font-label-md text-label-md text-right tabular-nums ${dGood ? 'text-primary' : d != null && (hi ? d < -0.005 : d > 0.005) ? 'text-error' : 'text-on-surface-variant'}`}>
                  {d != null ? (d >= 0 ? '+' : '') + f(Math.abs(d)) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="flex gap-2 pt-1">
        {(['A', 'B'] as Slot[]).map(s => (
          <button key={s} onClick={() => setShowing(s)} className={`px-3 py-1.5 rounded-full font-label-md text-label-md transition-colors ${showing === s ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
            Output {s}
          </button>
        ))}
      </div>
      <OutputTabs result={showing === 'A' ? a : b} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// BatchView
// ---------------------------------------------------------------------------
function BatchView({ workflowConfig }: { workflowConfig: WorkflowOverride | null }) {
  const [library, setLibrary] = useState<Record<string, AudioLibraryFile[]>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<LabRunResult[]>([]);
  const [running, setRunning] = useState(false);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  useEffect(() => {
    getAudioLibrary().then(setLibrary).catch(() => {});
  }, []);

  const allFiles = Object.entries(library).flatMap(([domain, files]) => files.map(f => ({ ...f, key: `${domain}/${f.filename}` })));

  const toggle = (key: string) => {
    const next = new Set(selected);
    next.has(key) ? next.delete(key) : next.add(key);
    setSelected(next);
  };

  const runSelected = async () => {
    const toRun = allFiles.filter(f => selected.has(f.key));
    setRunning(true);
    setResults([]);
    for (const f of toRun) {
      setCurrentFile(f.key);
      try {
        const res = await runBatchItem({ domain: f.domain, filename: f.filename, workflow_override: workflowConfig ?? undefined });
        setResults(prev => [...prev, res]);
      } catch {
        // still continue with next file
      }
    }
    setCurrentFile(null);
    setRunning(false);
  };

  const exportCsv = () => {
    const header = 'Domain,File,Chars,ROUGE-L,ROUGE-L naive,ΔROUGE-L,BERTScore F1,Coverage,Action Recall,Time(s)';
    const rows = results.map(r => [
      r.domain, r.filename, r.transcript_chars,
      r.eval.rouge_vs_transcript?.rougeL ?? '',
      r.eval.rouge_naive_vs_transcript?.rougeL ?? '',
      r.eval.rouge_vs_transcript && r.eval.rouge_naive_vs_transcript
        ? (r.eval.rouge_vs_transcript.rougeL - r.eval.rouge_naive_vs_transcript.rougeL).toFixed(3) : '',
      r.eval.bertscore_vs_transcript?.f1 ?? '',
      r.eval.agent_coverage,
      r.eval.action_recall ?? '',
      (r.total_ms / 1000).toFixed(1),
    ].join(','));
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'lab_batch_results.csv'; a.click();
  };

  return (
    <div className="space-y-4">
      {/* File selector */}
      <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-headline-md text-headline-md text-on-surface">Audio Library</h3>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            Add files to <code className="bg-surface-container px-1 rounded">backend/tests/audio/{'<Domain>'}/'</code>
          </span>
        </div>
        {allFiles.length === 0
          ? <p className="font-body-sm text-body-sm text-on-surface-variant">No audio files found. Add .mp3/.wav files to domain subfolders.</p>
          : (
            <div className="space-y-1">
              {allFiles.map(f => (
                <label key={f.key} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-container-low cursor-pointer">
                  <input type="checkbox" checked={selected.has(f.key)} onChange={() => toggle(f.key)} className="rounded flex-shrink-0" />
                  <span className="px-2 py-0.5 rounded-full bg-primary-fixed text-on-primary-fixed font-label-sm text-label-sm">{f.domain}</span>
                  <span className="flex-1 font-body-md text-body-md text-on-surface truncate">{f.filename}</span>
                  <span className="font-label-md text-label-md text-on-surface-variant">{formatBytes(f.size_bytes)}</span>
                </label>
              ))}
            </div>
          )}
        <div className="flex items-center gap-3 pt-1">
          <button onClick={runSelected} disabled={running || selected.size === 0}
            className="flex items-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-lg font-label-md text-label-md hover:opacity-90 disabled:opacity-40">
            {running
              ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span>Running {currentFile}…</>
              : <><span className="material-symbols-outlined text-[16px]">play_arrow</span>Run {selected.size} selected</>}
          </button>
          {results.length > 0 && (
            <button onClick={exportCsv} className="flex items-center gap-1 text-primary font-label-md text-label-md hover:underline">
              <span className="material-symbols-outlined text-[16px]">download</span>Export CSV
            </button>
          )}
        </div>
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <div className="bg-surface-container-lowest border border-outline-variant rounded-xl overflow-hidden">
          <table className="w-full text-left">
            <thead className="bg-surface-container">
              <tr>{['Domain', 'File', 'Chars', 'ROUGE-L', 'Δ ROUGE-L', 'BERTScore F1', 'Coverage', 'Action Recall', 'Time'].map(h => (
                <th key={h} className="px-3 py-2 font-label-sm text-label-sm text-on-surface-variant">{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const key = `${r.domain}/${r.filename}`;
                const rougeL = r.eval.rouge_vs_transcript?.rougeL;
                const rougeLNaive = r.eval.rouge_naive_vs_transcript?.rougeL;
                const bertF1 = r.eval.bertscore_vs_transcript?.f1;
                return (
                  <>
                    <tr key={key} onClick={() => setExpandedRow(expandedRow === key ? null : key)}
                      className="border-t border-outline-variant hover:bg-surface-container-low cursor-pointer transition-colors">
                      <td className="px-3 py-2"><span className="px-2 py-0.5 rounded-full bg-primary-fixed text-on-primary-fixed font-label-sm text-label-sm">{r.domain}</span></td>
                      <td className="px-3 py-2 font-body-sm text-body-sm text-on-surface max-w-[160px] truncate">{r.filename ?? '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface-variant tabular-nums">{r.transcript_chars?.toLocaleString() ?? '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{rougeL != null ? pct(rougeL) : '—'}</td>
                      <td className={`px-3 py-2 font-label-md text-label-md tabular-nums ${rougeL != null && rougeLNaive != null ? (rougeL - rougeLNaive > 0.005 ? 'text-primary' : 'text-on-surface-variant') : ''}`}>
                        {rougeL != null && rougeLNaive != null ? delta(rougeLNaive, rougeL) : '—'}
                      </td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{bertF1 != null ? pct(bertF1) : '—'}</td>
                      <td className="px-3 py-2"><ScoreBar value={r.eval.agent_coverage} /></td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{r.eval.action_recall != null ? pct(r.eval.action_recall) : '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface-variant tabular-nums">{fmt(r.total_ms)}</td>
                    </tr>
                    {expandedRow === key && (
                      <tr key={key + '-detail'} className="border-t border-outline-variant bg-surface-container-lowest">
                        <td colSpan={9} className="p-4">
                          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <h4 className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide">Workflow trace</h4>
                              {r.steps.map((s, si) => <StepRow key={si} step={s} index={si} />)}
                            </div>
                            <EvalPanel result={r} />
                          </div>
                          <div className="mt-4"><OutputTabs result={r} /></div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function TestingPlatform() {
  const [mode, setMode] = useState<Mode>('single');
  const [domain, setDomain] = useState('Project');
  const [transcript, setTranscript] = useState('');
  const [fixtureTranscript, setFixtureTranscript] = useState('');
  const [transcriptSource, setTranscriptSource] = useState<'text' | 'audio'>('text');
  const [knowledgeBase, setKnowledgeBase] = useState('');
  const [workflowConfig, setWorkflowConfig] = useState<WorkflowOverride | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [resultA, setResultA] = useState<LabRunResult | null>(null);
  const [resultB, setResultB] = useState<LabRunResult | null>(null);
  const [activeSlot, setActiveSlot] = useState<Slot>('A');
  const [error, setError] = useState<string | null>(null);
  const [lmOnline, setLmOnline] = useState<boolean | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = useCallback(() => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } }, []);

  // LM Studio status
  useEffect(() => {
    getLMStudioStatus().then(s => setLmOnline(s.connected)).catch(() => setLmOnline(false));
  }, []);

  // Load fixture when domain changes
  useEffect(() => {
    setResultA(null); setResultB(null); setError(null);
    getLabFixture(domain).then(fix => {
      setFixtureTranscript(fix.transcript);
      setTranscript(fix.transcript);
    }).catch(() => {});
  }, [domain]);

  const handleRun = async () => {
    setIsRunning(true); setError(null); setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    try {
      const res = await runLabPipeline({
        domain,
        transcript: transcript !== fixtureTranscript ? transcript : undefined,
        knowledge_base: knowledgeBase || undefined,
        workflow_override: workflowConfig ?? undefined,
        run_label: mode === 'ab' ? activeSlot : undefined,
      });
      if (mode === 'ab' && activeSlot === 'B') setResultB(res);
      else setResultA(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      stopTimer(); setIsRunning(false);
    }
  };

  const activeResult = mode === 'ab' && activeSlot === 'B' ? resultB : resultA;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface">Agent Testing Lab</h1>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Run the full multi-agent pipeline against domain fixtures or real audio. Evaluate with ROUGE, BERTScore, and structured metrics.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 mt-1">
          <span className={`w-2 h-2 rounded-full ${lmOnline == null ? 'bg-outline' : lmOnline ? 'bg-primary' : 'bg-error'}`} />
          <span className="font-label-md text-label-md text-on-surface-variant">
            {lmOnline == null ? 'Checking…' : lmOnline ? 'LM Studio connected' : 'LM Studio offline'}
          </span>
        </div>
      </div>

      {/* Mode switcher */}
      <div className="flex gap-2">
        {(['single', 'ab', 'batch'] as Mode[]).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-colors ${mode === m ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
            {m === 'single' ? 'Single run' : m === 'ab' ? 'A/B compare' : 'Batch test'}
          </button>
        ))}
      </div>

      {/* Batch mode */}
      {mode === 'batch' && <BatchView workflowConfig={workflowConfig} />}

      {/* Single / A/B mode */}
      {mode !== 'batch' && (
        <>
          {/* Domain selector */}
          <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-3">
            <p className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide">Domain</p>
            <div className="flex flex-wrap gap-2">
              {DOMAINS.map(d => (
                <button key={d} onClick={() => setDomain(d)}
                  className={`px-3 py-1.5 rounded-full font-label-md text-label-md transition-colors ${domain === d ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Transcript card */}
          <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-3">
            {/* Source toggle */}
            <div className="flex items-center gap-3">
              <p className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide flex-1">Transcript</p>
              <div className="flex gap-1 bg-surface-container rounded-lg p-0.5">
                {(['text', 'audio'] as const).map(s => (
                  <button key={s} onClick={() => setTranscriptSource(s)}
                    className={`px-3 py-1 rounded-md font-label-md text-label-md transition-colors ${transcriptSource === s ? 'bg-surface-container-lowest text-on-surface shadow-sm' : 'text-on-surface-variant hover:text-on-surface'}`}>
                    <span className="material-symbols-outlined text-[14px] mr-1 align-middle">{s === 'text' ? 'text_fields' : 'mic'}</span>
                    {s === 'text' ? 'Paste text' : 'Upload audio'}
                  </button>
                ))}
              </div>
              {transcriptSource === 'text' && transcript !== fixtureTranscript && (
                <button onClick={() => setTranscript(fixtureTranscript)} className="font-label-md text-label-md text-primary hover:underline">
                  Reset to fixture
                </button>
              )}
            </div>

            {transcriptSource === 'audio'
              ? <AudioDropZone onTranscript={(text, model) => { setTranscript(text); setTranscriptSource('text'); console.info('Transcribed with', model); }} />
              : <textarea value={transcript} onChange={e => setTranscript(e.target.value)} rows={10}
                  className="w-full bg-surface-container rounded-lg px-3 py-2 font-body-sm text-body-sm text-on-surface resize-y border border-outline-variant focus:outline-none focus:ring-1 focus:ring-primary"
                  placeholder="Paste a transcript or use the fixture above…" />
            }

            <details>
              <summary className="cursor-pointer font-label-md text-label-md text-on-surface-variant select-none">Project knowledge base (optional)</summary>
              <textarea value={knowledgeBase} onChange={e => setKnowledgeBase(e.target.value)} rows={3}
                className="mt-2 w-full bg-surface-container rounded-lg px-3 py-2 font-body-sm text-body-sm text-on-surface resize-y border border-outline-variant focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="Team member names, project context, abbreviations…" />
            </details>

            <WorkflowEditor value={workflowConfig} onChange={setWorkflowConfig} />

            {/* Run controls */}
            <div className="flex items-center gap-3 pt-1 flex-wrap">
              {mode === 'ab' && (
                <div className="flex gap-1 bg-surface-container rounded-lg p-0.5">
                  {(['A', 'B'] as Slot[]).map(s => (
                    <button key={s} onClick={() => setActiveSlot(s)}
                      className={`px-3 py-1 rounded-md font-label-md text-label-md transition-colors ${activeSlot === s ? 'bg-surface-container-lowest text-on-surface shadow-sm' : 'text-on-surface-variant hover:text-on-surface'}`}>
                      Run as {s}
                    </button>
                  ))}
                </div>
              )}
              <button onClick={handleRun} disabled={isRunning || !transcript.trim()}
                className="flex items-center gap-2 bg-primary text-on-primary px-5 py-2 rounded-lg font-label-md text-label-md hover:opacity-90 transition-opacity disabled:opacity-40">
                {isRunning
                  ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span>Running… {elapsed}s</>
                  : <><span className="material-symbols-outlined text-[16px]">play_arrow</span>{mode === 'ab' ? `Run ${activeSlot}` : 'Run Pipeline'}</>}
              </button>
              {activeResult && !isRunning && (
                <span className="font-label-md text-label-md text-on-surface-variant">
                  {mode === 'ab' ? `Run ${activeSlot} ` : ''}completed in {fmt(activeResult.total_ms)} · {activeResult.steps.length} steps
                </span>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-error-container border border-error rounded-xl p-4">
              <p className="font-label-md text-label-md text-on-error-container flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">error</span>{error}
              </p>
            </div>
          )}

          {/* A/B comparison */}
          {mode === 'ab' && resultA && resultB && <ABComparisonView a={resultA} b={resultB} />}

          {/* Single run results */}
          {mode !== 'ab' && resultA && (
            <>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-headline-md text-headline-md text-on-surface">Workflow Trace</h3>
                    <span className="font-label-sm text-label-sm text-on-surface-variant">{resultA.workflow_plan.steps.join(' → ')}</span>
                  </div>
                  <div className="space-y-2">
                    {resultA.steps.map((step, i) => <StepRow key={i} step={step} index={i} />)}
                  </div>
                </div>
                <EvalPanel result={resultA} />
              </div>
              <OutputTabs result={resultA} />
            </>
          )}

          {/* A/B partial state: show whichever slot has a result */}
          {mode === 'ab' && (resultA || resultB) && !(resultA && resultB) && (
            <>
              {(resultA || resultB) && (
                <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4">
                  <p className="font-body-sm text-body-sm text-on-surface-variant text-center">
                    Run {resultA ? 'B' : 'A'} to see the comparison.
                  </p>
                </div>
              )}
              {(resultA ?? resultB) && (
                <>
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-3">
                      <h3 className="font-headline-md text-headline-md text-on-surface">Workflow Trace (Run {resultA ? 'A' : 'B'})</h3>
                      <div className="space-y-2">{(resultA ?? resultB)!.steps.map((s, i) => <StepRow key={i} step={s} index={i} />)}</div>
                    </div>
                    <EvalPanel result={(resultA ?? resultB)!} />
                  </div>
                  <OutputTabs result={(resultA ?? resultB)!} />
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
