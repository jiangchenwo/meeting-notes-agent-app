import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Breadcrumb from '../components/Breadcrumb';
import {
  getLabDatasets,
  getLabCase,
  runLabPipeline,
  labTranscribeAudio,
  getAudioLibrary,
  runBatchItem,
  getLabHistory,
  getLabHistoryRun,
  deleteLabHistoryRun,
  type LabRunResult,
  type WorkflowStep,
  type ActionItem,
  type WorkflowOverride,
  type AudioLibraryFile,
  type RougeScores,
  type BertScore,
  type EvalCaseMeta,
  type SchemaCheckDetail,
  type RiskClassification,
  type LabHistoryEntry,
} from '../api/lab';
import { getLMStudioStatus } from '../api/summarize';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DOMAINS = ['General', 'Education', 'Healthcare', 'Interview', 'Project'] as const;
type DomainName = typeof DOMAINS[number];

const DOMAIN_COLORS: Record<DomainName, string> = {
  General:    'bg-surface-container text-on-surface',
  Education:  'bg-blue-100 text-blue-800',
  Healthcare: 'bg-red-100 text-red-800',
  Interview:  'bg-purple-100 text-purple-800',
  Project:    'bg-primary-fixed text-on-primary-fixed',
};

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
// Tiny helpers
// ---------------------------------------------------------------------------

function fmtMs(ms: number) { return (ms / 1000).toFixed(1) + 's'; }
function pct(v: number) { return (v * 100).toFixed(0) + '%'; }
function formatBytes(b: number) {
  return b > 1_000_000 ? (b / 1_000_000).toFixed(1) + ' MB' : Math.round(b / 1024) + ' KB';
}
function betterColor(a: number, b: number, higherIsBetter = true) {
  if (higherIsBetter ? b > a : b < a) return 'text-primary font-semibold';
  if (higherIsBetter ? a > b : a < b) return 'text-on-surface-variant';
  return 'text-on-surface';
}
function scoreColor(score: number) {
  return score >= 8 ? 'bg-green-100 text-green-800' : score >= 6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800';
}

// ---------------------------------------------------------------------------
// ScoreBar
// ---------------------------------------------------------------------------
function ScoreBar({ value, label }: { value: number; label?: string }) {
  const p = Math.round(value * 100);
  const color = p >= 80 ? 'bg-primary' : p >= 60 ? 'bg-yellow-500' : 'bg-error';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-container rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${p}%` }} />
      </div>
      <span className="font-label-md text-label-md text-on-surface w-9 text-right tabular-nums">{p}%</span>
      {label && <span className="font-label-sm text-label-sm text-on-surface-variant">{label}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TimingWaterfall — horizontal bars showing relative duration per step
// ---------------------------------------------------------------------------
function TimingWaterfall({ steps, totalMs }: { steps: WorkflowStep[]; totalMs: number }) {
  const effective = Math.max(totalMs, steps.reduce((s, st) => s + st.duration_ms, 1));
  return (
    <div className="space-y-1">
      {steps.map((step, i) => {
        const widthPct = Math.max(2, Math.round((step.duration_ms / effective) * 100));
        const isCritique = step.phase === 'critique';
        const barColor = step.status === 'error' ? 'bg-error' : isCritique ? 'bg-secondary' : 'bg-primary';
        return (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="w-36 truncate text-on-surface-variant font-label-sm text-label-sm flex-shrink-0">
              {step.attempt > 1 && <span className="text-warning mr-0.5">↺</span>}
              {step.name}
            </span>
            <div className="flex-1 h-4 bg-surface-container rounded overflow-hidden">
              <div
                className={`h-full ${barColor} rounded opacity-80 flex items-center pl-1`}
                style={{ width: `${widthPct}%`, minWidth: 4 }}
              >
                {widthPct > 8 && <span className="text-white font-label-sm text-label-sm leading-none">{fmtMs(step.duration_ms)}</span>}
              </div>
            </div>
            <span className="w-10 text-right font-label-md text-label-md text-on-surface-variant tabular-nums flex-shrink-0">
              {widthPct <= 8 ? fmtMs(step.duration_ms) : ''}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AgentOutputCard — structured rendering per agent type
// ---------------------------------------------------------------------------
function AgentOutputCard({ agentName, output }: { agentName: string; output: Record<string, unknown> }) {
  if (agentName === 'Summarizer') {
    const summary = output.summary as string | undefined;
    return summary
      ? <p className="font-body-sm text-body-sm text-on-surface line-clamp-4">{summary}</p>
      : null;
  }
  if (agentName === 'ActionItemExtractor') {
    const items = output.action_items as Array<Record<string, unknown>> | undefined ?? [];
    if (!items.length) return <p className="font-body-sm text-body-sm text-on-surface-variant italic">No action items extracted.</p>;
    return (
      <ul className="space-y-0.5">
        {items.slice(0, 4).map((item, i) => (
          <li key={i} className="flex items-start gap-1.5 font-body-sm text-body-sm text-on-surface">
            <span className={`flex-shrink-0 mt-0.5 px-1 rounded font-label-sm text-label-sm ${item.priority === 'high' ? 'bg-error-container text-on-error-container' : 'bg-surface-container text-on-surface-variant'}`}>
              {String(item.priority ?? 'med')}
            </span>
            <span className="truncate">{String(item.task ?? '')}</span>
            {item.owner ? <span className="text-on-surface-variant flex-shrink-0">→ {String(item.owner)}</span> : null}
          </li>
        ))}
        {items.length > 4 && <li className="font-label-sm text-label-sm text-on-surface-variant">+{items.length - 4} more</li>}
      </ul>
    );
  }
  if (agentName === 'DecisionLogger') {
    const decisions = output.decisions as Array<Record<string, unknown>> | undefined ?? [];
    if (!decisions.length) return <p className="font-body-sm text-body-sm text-on-surface-variant italic">No decisions logged.</p>;
    return (
      <ul className="space-y-0.5">
        {decisions.slice(0, 3).map((d, i) => (
          <li key={i} className="font-body-sm text-body-sm text-on-surface">
            <span className="font-medium">{String(d.decision ?? '')}</span>
            {d.rationale ? <span className="text-on-surface-variant ml-1 text-xs">— {String(d.rationale)}</span> : null}
          </li>
        ))}
      </ul>
    );
  }
  if (agentName === 'InterviewAgent') {
    const red = output.red_flags as string[] ?? [];
    const green = output.green_flags as string[] ?? [];
    return (
      <div className="space-y-1">
        {red.length > 0 && (
          <div>
            <span className="font-label-sm text-label-sm text-error mr-1">Red flags:</span>
            <span className="font-body-sm text-body-sm text-on-surface">{red.slice(0, 2).join(' · ')}</span>
          </div>
        )}
        {green.length > 0 && (
          <div>
            <span className="font-label-sm text-label-sm text-primary mr-1">Green flags:</span>
            <span className="font-body-sm text-body-sm text-on-surface">{green.slice(0, 2).join(' · ')}</span>
          </div>
        )}
      </div>
    );
  }
  if (agentName === 'LectureAgent') {
    const concepts = output.key_concepts as Array<Record<string, unknown>> ?? [];
    const objectives = output.learning_objectives as string[] ?? [];
    return (
      <div className="space-y-1">
        {concepts.length > 0 && (
          <p className="font-body-sm text-body-sm text-on-surface">
            <span className="font-label-sm text-label-sm text-on-surface-variant">Concepts: </span>
            {concepts.slice(0, 3).map(c => String(c.concept ?? '')).join(', ')}
            {concepts.length > 3 && ` +${concepts.length - 3} more`}
          </p>
        )}
        {objectives.length > 0 && (
          <p className="font-body-sm text-body-sm text-on-surface">
            <span className="font-label-sm text-label-sm text-on-surface-variant">Objectives: </span>
            {objectives[0]}{objectives.length > 1 && ` +${objectives.length - 1} more`}
          </p>
        )}
      </div>
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// JsonView — syntax-highlighted JSON renderer
// ---------------------------------------------------------------------------
function JsonValue({ value, depth = 0 }: { value: unknown; depth?: number }): React.ReactElement {
  if (value === null) return <span className="text-on-surface-variant italic">null</span>;
  if (typeof value === 'boolean') return <span className="text-blue-600">{value ? 'true' : 'false'}</span>;
  if (typeof value === 'number') return <span className="text-blue-700 tabular-nums">{value}</span>;
  if (typeof value === 'string') {
    const isLong = value.length > 120;
    return (
      <span className="text-green-700">
        &quot;{isLong ? value : value}&quot;
      </span>
    );
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-on-surface-variant">[]</span>;
    return (
      <span>
        <span className="text-on-surface-variant">[</span>
        <div style={{ marginLeft: (depth + 1) * 14 }}>
          {value.map((item, i) => (
            <div key={i}>
              <JsonValue value={item} depth={depth + 1} />
              {i < value.length - 1 && <span className="text-on-surface-variant">,</span>}
            </div>
          ))}
        </div>
        <span className="text-on-surface-variant">]</span>
      </span>
    );
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return <span className="text-on-surface-variant">{'{}'}</span>;
    return (
      <span>
        <span className="text-on-surface-variant">{'{'}</span>
        <div style={{ marginLeft: (depth + 1) * 14 }}>
          {entries.map(([k, v], i) => (
            <div key={k}>
              <span className="text-purple-700 font-medium">&quot;{k}&quot;</span>
              <span className="text-on-surface-variant">: </span>
              <JsonValue value={v} depth={depth + 1} />
              {i < entries.length - 1 && <span className="text-on-surface-variant">,</span>}
            </div>
          ))}
        </div>
        <span className="text-on-surface-variant">{'}'}</span>
      </span>
    );
  }
  return <span className="text-on-surface">{String(value)}</span>;
}

function JsonView({ data }: { data: unknown }) {
  return (
    <div className="text-[12px] font-mono leading-relaxed bg-surface-container rounded-lg p-3 overflow-auto max-h-96">
      <JsonValue value={data} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// WorkflowStep row — expandable with Output + Prompt tabs
// ---------------------------------------------------------------------------
function StepRow({ step, index, schemaCheck }: { step: WorkflowStep; index: number; schemaCheck?: SchemaCheckDetail }) {
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState<'output' | 'prompt'>('output');
  const isOk = step.status === 'done';
  const isCritique = step.phase === 'critique';
  const score = step.critique_score;
  const agentName = isCritique ? step.name.replace('Critic→', '').replace('Critic:', '') : step.name.split('[')[0];
  const tokenTotal = step.tokens ? step.tokens.input + step.tokens.output : null;

  return (
    <div className="border border-outline-variant rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-container-low transition-colors"
      >
        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-surface-container flex items-center justify-center font-label-sm text-label-sm text-on-surface-variant text-[10px]">{index + 1}</span>
        <span className="flex-1 font-body-md text-body-md text-on-surface truncate min-w-0">{step.name}</span>
        {step.attempt > 1 && <span className="px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-800 font-label-sm text-label-sm flex-shrink-0">retry #{step.attempt}</span>}
        <span className={`px-2 py-0.5 rounded-full font-label-sm text-label-sm flex-shrink-0 ${isCritique ? 'bg-secondary-container text-secondary' : 'bg-primary-fixed text-on-primary-fixed'}`}>
          {isCritique ? 'critique' : 'extract'}
        </span>
        {score != null && (
          <span className={`px-2 py-0.5 rounded-full font-label-md text-label-md tabular-nums flex-shrink-0 ${scoreColor(score)}`}>
            {score.toFixed(1)}/10
          </span>
        )}
        {schemaCheck && (
          <span className={`flex items-center gap-0.5 flex-shrink-0 ${schemaCheck.pass ? 'text-primary' : 'text-error'}`}>
            <span className="material-symbols-outlined text-[13px]">{schemaCheck.pass ? 'check_circle' : 'error'}</span>
          </span>
        )}
        <span className={`material-symbols-outlined text-[16px] flex-shrink-0 ${isOk ? 'text-primary' : 'text-error'}`}>{isOk ? 'check_circle' : 'error'}</span>
        <span className="font-label-md text-label-md text-on-surface-variant w-10 text-right tabular-nums flex-shrink-0">{fmtMs(step.duration_ms)}</span>
        <span className="material-symbols-outlined text-[16px] text-on-surface-variant flex-shrink-0">{expanded ? 'expand_less' : 'expand_more'}</span>
      </button>

      {expanded && (
        <div className="border-t border-outline-variant bg-surface-container-lowest">
          {/* Meta row: tokens + schema */}
          <div className="px-3 py-2 flex flex-wrap items-center gap-4 border-b border-outline-variant bg-surface-container">
            {step.tokens ? (
              <>
                <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Tokens</span>
                <span className="font-label-md text-label-md text-on-surface-variant">in: <span className="text-on-surface">{step.tokens.input.toLocaleString()}</span></span>
                <span className="font-label-md text-label-md text-on-surface-variant">out: <span className="text-on-surface">{step.tokens.output.toLocaleString()}</span></span>
                {tokenTotal && <span className="font-label-md text-label-md text-on-surface-variant">total: <span className="text-on-surface">{tokenTotal.toLocaleString()}</span></span>}
              </>
            ) : (
              <span className="font-label-sm text-label-sm text-on-surface-variant">No token data</span>
            )}
            {schemaCheck && !schemaCheck.pass && (
              <span className="ml-auto font-label-sm text-label-sm text-error flex items-center gap-1">
                <span className="material-symbols-outlined text-[13px]">error</span>
                Schema: {schemaCheck.missing.join(', ')}
              </span>
            )}
          </div>

          {/* Tab bar */}
          <div className="flex border-b border-outline-variant">
            <button
              onClick={() => setTab('output')}
              className={`px-4 py-2 font-label-md text-label-md transition-colors ${tab === 'output' ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant hover:text-on-surface'}`}
            >
              Output
            </button>
            {step.prompt && (
              <button
                onClick={() => setTab('prompt')}
                className={`px-4 py-2 font-label-md text-label-md transition-colors ${tab === 'prompt' ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant hover:text-on-surface'}`}
              >
                Prompt
              </button>
            )}
          </div>

          {/* Output tab */}
          {tab === 'output' && (
            <div className="divide-y divide-outline-variant">
              {step.error && (
                <div className="px-3 py-2 font-body-sm text-body-sm text-error flex items-center gap-2">
                  <span className="material-symbols-outlined text-[15px]">error</span>{step.error}
                </div>
              )}
              {/* Critique: dimensions + revision advice */}
              {isCritique && (
                <div className="px-3 py-3 space-y-3">
                  {(step.output as Record<string, unknown>).dimensions
                    ? (() => {
                        const d = (step.output as Record<string, unknown>).dimensions as Record<string, number>;
                        const DIMS: [string, string, number][] = [
                          ['Coverage', 'coverage', 4],
                          ['Accuracy', 'accuracy', 3],
                          ['Specificity', 'specificity', 2],
                          ['Structure', 'structure', 1],
                        ];
                        return (
                          <div>
                            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-2">Dimensions</p>
                            <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                              {DIMS.map(([label, key, max]) => {
                                const val = d[key] ?? 0;
                                const p = val / max;
                                return (
                                  <div key={key}>
                                    <div className="flex items-center justify-between mb-0.5">
                                      <span className="font-label-sm text-label-sm text-on-surface-variant">{label}</span>
                                      <span className="font-label-md text-label-md text-on-surface tabular-nums">{val}/{max}</span>
                                    </div>
                                    <div className="h-1.5 rounded-full bg-surface-container overflow-hidden">
                                      <div className={`h-full rounded-full transition-all ${p >= 1 ? 'bg-primary' : p >= 0.6 ? 'bg-yellow-500' : 'bg-error'}`} style={{ width: `${p * 100}%` }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })()
                    : null
                  }
                  {Array.isArray((step.output as Record<string, unknown>).issues) && (
                    <div>
                      <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-1.5">Revision advice</p>
                      {((step.output as Record<string, unknown>).issues as string[]).length === 0
                        ? <p className="font-body-sm text-body-sm text-primary flex items-center gap-1"><span className="material-symbols-outlined text-[14px]">check_circle</span>None — output meets quality bar</p>
                        : (
                          <ul className="list-disc pl-4 space-y-1">
                            {((step.output as Record<string, unknown>).issues as string[]).map((issue, i) => (
                              <li key={i} className="font-body-sm text-body-sm text-on-surface">{issue}</li>
                            ))}
                          </ul>
                        )
                      }
                    </div>
                  )}
                </div>
              )}
              {/* Extraction: structured preview */}
              {!isCritique && isOk && Object.keys(step.output).length > 0 && (
                <div className="px-3 py-2">
                  <AgentOutputCard agentName={agentName} output={step.output} />
                </div>
              )}
              {/* JSON output */}
              <div className="px-3 py-3">
                <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-2">JSON</p>
                <JsonView data={step.output} />
              </div>
            </div>
          )}

          {/* Prompt tab */}
          {tab === 'prompt' && step.prompt && (
            <div className="divide-y divide-outline-variant">
              <div className="px-3 py-3 space-y-1.5">
                <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">System</p>
                <pre className="text-[12px] font-mono text-on-surface-variant leading-relaxed bg-surface-container rounded-lg p-3 overflow-auto max-h-56 whitespace-pre-wrap break-words">{step.prompt.system}</pre>
              </div>
              <div className="px-3 py-3 space-y-1.5">
                <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">User</p>
                <pre className="text-[12px] font-mono text-on-surface leading-relaxed bg-surface-container rounded-lg p-3 overflow-auto max-h-80 whitespace-pre-wrap break-words">{step.prompt.user}</pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// RiskPanel
// ---------------------------------------------------------------------------
function RiskPanel({ risk }: { risk: RiskClassification }) {
  if (!risk.needs_review && risk.risk_flags.length === 0) return null;
  return (
    <div className={`rounded-xl p-4 space-y-2 ${risk.needs_review ? 'bg-error-container border border-error' : 'bg-surface-container border border-outline-variant'}`}>
      <div className="flex items-center gap-2">
        <span className={`material-symbols-outlined text-[18px] ${risk.needs_review ? 'text-error' : 'text-on-surface-variant'}`}>
          {risk.needs_review ? 'warning' : 'shield'}
        </span>
        <span className={`font-label-md text-label-md ${risk.needs_review ? 'text-on-error-container font-semibold' : 'text-on-surface-variant'}`}>
          {risk.needs_review ? 'Human review recommended' : 'No risk flags'}
        </span>
      </div>
      {risk.risk_flags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {risk.risk_flags.map(flag => (
            <span key={flag} className="px-2 py-0.5 bg-error text-on-error rounded-full font-label-sm text-label-sm">{flag}</span>
          ))}
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
  const [retries, setRetries] = useState(value?.max_retries ?? 2);

  const emit = (s: AgentName[], cs: Set<string>, t: number, r: number) =>
    onChange({ steps: s, critique_steps: [...cs].filter(a => s.includes(a as AgentName)), critique_threshold: t, max_retries: r });

  const toggle = (agent: AgentName) => {
    const next = steps.includes(agent) ? steps.filter(a => a !== agent) : [...steps, agent];
    setSteps(next as AgentName[]); emit(next as AgentName[], critiqueSteps, threshold, retries);
  };
  const move = (i: number, dir: -1 | 1) => {
    const next = [...steps]; const j = i + dir;
    if (j < 0 || j >= next.length) return;
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next); emit(next, critiqueSteps, threshold, retries);
  };
  const toggleCritique = (agent: string) => {
    const next = new Set(critiqueSteps);
    next.has(agent) ? next.delete(agent) : next.add(agent);
    setCritiqueSteps(next); emit(steps, next, threshold, retries);
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
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Agent steps</p>
            <div className="space-y-1.5">
              {ALL_AGENTS.map((agent) => {
                const inSteps = steps.includes(agent); const stepIdx = steps.indexOf(agent);
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
                  onChange={e => { const v = parseFloat(e.target.value); setThreshold(v); emit(steps, critiqueSteps, v, retries); }} className="w-full" />
              </div>
              <div>
                <label className="font-label-md text-label-md text-on-surface-variant block mb-1">Max retries</label>
                <select value={retries} onChange={e => { const v = parseInt(e.target.value); setRetries(v); emit(steps, critiqueSteps, threshold, v); }}
                  className="w-full bg-surface-container rounded-lg px-2 py-1.5 font-body-sm text-body-sm text-on-surface border border-outline-variant">
                  {[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>
            <div className="pt-1 text-primary font-body-sm text-body-sm">Steps: {steps.join(' → ')}</div>
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
    setIsTranscribing(true); setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    try {
      const res = await labTranscribeAudio(file);
      onTranscript(res.transcript, res.model_used);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { stopTimer(); setIsTranscribing(false); }
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
        {file
          ? <div><p className="font-body-md text-body-md text-on-surface font-medium">{file.name}</p><p className="font-body-sm text-body-sm text-on-surface-variant">{formatBytes(file.size)}</p></div>
          : <p className="font-body-md text-body-md text-on-surface-variant">Drop audio file or click to browse<br /><span className="font-body-sm text-body-sm">.mp3 .wav .m4a .ogg .webm</span></p>
        }
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
          ? <div className="prose prose-sm max-w-none"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary_text}</ReactMarkdown></div>
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No summary generated.</p>)}
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
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No action items extracted.</p>)}
        {active === 'domain' && (result.suggestions_text
          ? <div className="prose prose-sm max-w-none"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.suggestions_text}</ReactMarkdown></div>
          : <p className="text-on-surface-variant font-body-sm text-body-sm">No domain-specific output.</p>)}
        {active === 'naive' && (
          <div>
            <div className="mb-3 px-3 py-2 bg-surface-container rounded-lg">
              <p className="font-label-sm text-label-sm text-on-surface-variant">
                Single-call summarizer — no structured extraction, no agents, no critique.
                Coverage: {pct(result.eval.naive_coverage)} vs agentic {pct(result.eval.agent_coverage)}
              </p>
            </div>
            {result.naive_summary
              ? <div className="prose prose-sm max-w-none"><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.naive_summary}</ReactMarkdown></div>
              : <p className="text-on-surface-variant font-body-sm text-body-sm">Naive baseline not available.</p>}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvalPanel
// ---------------------------------------------------------------------------
function MetricRow({ label, agValue, naiveValue, higherBetter = true, fmtFn = pct }: {
  label: string; agValue: number | null; naiveValue: number | null; higherBetter?: boolean; fmtFn?: (v: number) => string;
}) {
  if (agValue == null) return null;
  const d = naiveValue != null ? (agValue - naiveValue) * (higherBetter ? 1 : -1) : null;
  return (
    <tr>
      <td className="py-1.5 pr-4 font-body-sm text-body-sm text-on-surface-variant">{label}</td>
      <td className="py-1.5 pr-4 font-label-md text-label-md text-on-surface tabular-nums">{fmtFn(agValue)}</td>
      <td className="py-1.5 pr-4 font-label-md text-label-md text-on-surface-variant tabular-nums">{naiveValue != null ? fmtFn(naiveValue) : '—'}</td>
      <td className={`py-1.5 font-label-md text-label-md tabular-nums ${d != null ? (d > 0.005 ? 'text-primary' : d < -0.005 ? 'text-error' : 'text-on-surface-variant') : ''}`}>
        {d != null ? (d >= 0 ? '+' : '') + fmtFn(Math.abs(d)) : '—'}
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
  const [goldExpanded, setGoldExpanded] = useState(false);
  const goldLabel = result.ground_truth?.gold_label as string | undefined;
  const schemaEntries = result.schema_checks
    ? Object.entries(result.schema_checks)
    : Object.entries(ev.schema_check ?? {}).map(([k, v]) => [k, { pass: v, missing: [], type_errors: [] }] as [string, SchemaCheckDetail]);

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-headline-md text-headline-md text-on-surface">Evaluation</h3>
        <div className="flex items-center gap-2">
          {result.chunked && <span className="px-2 py-0.5 rounded-full bg-secondary-container text-secondary font-label-sm text-label-sm">chunked ({result.chunk_count} seg)</span>}
          {result.case_id && <span className="px-2 py-0.5 rounded-full bg-surface-container text-on-surface-variant font-label-sm text-label-sm">{result.case_id}</span>}
        </div>
      </div>

      {/* Run stats row */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Total time</p>
          <p className="font-headline-md text-headline-md text-on-surface tabular-nums">{fmtMs(result.total_ms)}</p>
        </div>
        {result.confidence_score != null && (
          <div>
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Confidence</p>
            <p className={`font-headline-md text-headline-md tabular-nums ${result.confidence_score >= 8 ? 'text-primary' : result.confidence_score >= 6 ? 'text-yellow-600' : 'text-error'}`}>{result.confidence_score.toFixed(1)}/10</p>
          </div>
        )}
        <div>
          <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Steps</p>
          <p className="font-headline-md text-headline-md text-on-surface tabular-nums">{result.steps.length}</p>
        </div>
      </div>

      <div className="border-t border-outline-variant" />

      {/* Coverage + recall */}
      <div>
        <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide mb-2">
          Ground-truth coverage ({ev.ground_truth_facts} facts)
        </p>
        {ev.ground_truth_facts > 0 ? (
          <div className="space-y-1.5">
            <div>
              <div className="flex items-center justify-between mb-0.5">
                <span className="font-label-sm text-label-sm text-on-surface-variant">Agentic</span>
                <span className="font-label-sm text-label-sm text-on-surface-variant">vs Naive {pct(ev.naive_coverage)}</span>
              </div>
              <ScoreBar value={ev.agent_coverage} />
            </div>
            {ev.action_recall != null && (
              <div>
                <span className="font-label-sm text-label-sm text-on-surface-variant block mb-0.5">Action recall</span>
                <ScoreBar value={ev.action_recall} />
              </div>
            )}
          </div>
        ) : (
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
        )}
      </div>

      <div className="border-t border-outline-variant" />

      <RougeBlock label="ROUGE vs transcript" scores={ev.rouge_vs_transcript} naiveScores={ev.rouge_naive_vs_transcript} />
      {ev.rouge_vs_transcript && <div className="border-t border-outline-variant" />}
      <BertBlock label="BERTScore vs transcript" scores={ev.bertscore_vs_transcript} naiveScores={ev.bertscore_naive_vs_transcript} />
      {ev.bertscore_vs_transcript && <div className="border-t border-outline-variant" />}

      {ev.rouge_vs_gold && (
        <>
          <RougeBlock label="ROUGE vs gold reference" scores={ev.rouge_vs_gold} naiveScores={ev.rouge_naive_vs_gold} />
          <div className="border-t border-outline-variant" />
          <BertBlock label="BERTScore vs gold reference" scores={ev.bertscore_vs_gold} naiveScores={ev.bertscore_naive_vs_gold} />
          <div className="border-t border-outline-variant" />
          {goldLabel && (
            <div>
              <button
                onClick={() => setGoldExpanded(v => !v)}
                className="flex items-center gap-1 font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide hover:text-on-surface transition-colors"
              >
                <span className="material-symbols-outlined text-[14px]">{goldExpanded ? 'expand_less' : 'expand_more'}</span>
                Gold reference ({goldLabel.length} chars)
              </button>
              {goldExpanded && (
                <pre className="mt-2 p-3 bg-surface-container rounded-lg font-mono text-[11px] text-on-surface-variant whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
                  {goldLabel}
                </pre>
              )}
            </div>
          )}
          <div className="border-t border-outline-variant" />
        </>
      )}

      {/* Schema compliance */}
      {schemaEntries.length > 0 && (
        <>
          <div className="space-y-2">
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">Schema compliance</p>
            <div className="flex flex-wrap gap-1.5">
              {schemaEntries.map(([agent, check]) => (
                <span key={agent} className={`flex items-center gap-1 px-2 py-0.5 rounded-full font-label-sm text-label-sm ${check.pass ? 'bg-primary-fixed text-on-primary-fixed' : 'bg-error-container text-on-error-container'}`}>
                  <span className="material-symbols-outlined text-[12px]">{check.pass ? 'check' : 'close'}</span>
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// TracePanel — workflow trace with waterfall + step rows
// ---------------------------------------------------------------------------
function TracePanel({ result }: { result: LabRunResult }) {
  const [view, setView] = useState<'steps' | 'waterfall'>('steps');
  const schemaChecks = result.schema_checks ?? {};

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="font-headline-md text-headline-md text-on-surface">Workflow Trace</h3>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">{result.workflow_plan.steps.join(' → ')}</p>
        </div>
        <div className="flex gap-1 bg-surface-container rounded-lg p-0.5">
          {(['steps', 'waterfall'] as const).map(v => (
            <button key={v} onClick={() => setView(v)}
              className={`px-3 py-1 rounded-md font-label-md text-label-md capitalize transition-colors ${view === v ? 'bg-surface-container-lowest text-on-surface shadow-sm' : 'text-on-surface-variant hover:text-on-surface'}`}>
              {v}
            </button>
          ))}
        </div>
      </div>

      {view === 'waterfall' && <TimingWaterfall steps={result.steps} totalMs={result.total_ms} />}

      {view === 'steps' && (
        <div className="space-y-2">
          {result.steps.map((step, i) => {
            const agentName = step.phase === 'critique'
              ? step.name.replace('Critic→', '').replace('Critic:', '')
              : step.name.split('[')[0];
            return <StepRow key={i} step={step} index={i} schemaCheck={schemaChecks[agentName]} />;
          })}
        </div>
      )}

      {/* Risk panel */}
      {result.risk_classification && <RiskPanel risk={result.risk_classification} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// A/B Comparison
// ---------------------------------------------------------------------------
function ABComparisonView({ a, b }: { a: LabRunResult; b: LabRunResult }) {
  const [showing, setShowing] = useState<Slot>('A');
  type Row = { label: string; va: number | null; vb: number | null; hi: boolean; fmtFn?: (v: number) => string };
  const rows: Row[] = [
    { label: 'Coverage', va: a.eval.agent_coverage, vb: b.eval.agent_coverage, hi: true },
    { label: 'Action recall', va: a.eval.action_recall, vb: b.eval.action_recall, hi: true },
    { label: 'ROUGE-L', va: a.eval.rouge_vs_transcript?.rougeL ?? null, vb: b.eval.rouge_vs_transcript?.rougeL ?? null, hi: true },
    { label: 'BERTScore F1', va: a.eval.bertscore_vs_transcript?.f1 ?? null, vb: b.eval.bertscore_vs_transcript?.f1 ?? null, hi: true },
    { label: 'Confidence', va: a.confidence_score, vb: b.confidence_score, hi: true, fmtFn: v => v.toFixed(1) + '/10' },
    { label: 'Total time', va: a.total_ms / 1000, vb: b.total_ms / 1000, hi: false, fmtFn: v => v.toFixed(1) + 's' },
  ];
  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 space-y-4">
      <h3 className="font-headline-md text-headline-md text-on-surface">A/B Comparison</h3>
      <div className="grid grid-cols-2 gap-2 text-center">
        {([a, b] as [LabRunResult, LabRunResult]).map((r, i) => (
          <div key={i} className="bg-surface-container rounded-lg p-2">
            <p className="font-label-sm text-label-sm text-on-surface-variant">Run {i === 0 ? 'A' : 'B'} workflow</p>
            <p className="font-body-sm text-body-sm text-on-surface">{r.workflow_plan.steps.join(' → ')}</p>
          </div>
        ))}
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
          <button key={s} onClick={() => setShowing(s)}
            className={`px-3 py-1.5 rounded-full font-label-md text-label-md transition-colors ${showing === s ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
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

  useEffect(() => { getAudioLibrary().then(setLibrary).catch(() => {}); }, []);

  const allFiles = Object.entries(library).flatMap(([domain, files]) => files.map(f => ({ ...f, key: `${domain}/${f.filename}` })));
  const toggle = (key: string) => { const next = new Set(selected); next.has(key) ? next.delete(key) : next.add(key); setSelected(next); };

  const runSelected = async () => {
    const toRun = allFiles.filter(f => selected.has(f.key));
    setRunning(true); setResults([]);
    for (const f of toRun) {
      setCurrentFile(f.key);
      try {
        const res = await runBatchItem({ domain: f.domain, filename: f.filename, workflow_override: workflowConfig ?? undefined });
        setResults(prev => [...prev, res]);
      } catch { /* continue */ }
    }
    setCurrentFile(null); setRunning(false);
  };

  const exportCsv = () => {
    const header = 'Domain,File,Chars,ROUGE-L,ROUGE-L naive,ΔROUGE-L,BERTScore F1,Coverage,Action Recall,Time(s)';
    const rows = results.map(r => [
      r.domain, r.filename, r.transcript_chars,
      r.eval.rouge_vs_transcript?.rougeL ?? '',
      r.eval.rouge_naive_vs_transcript?.rougeL ?? '',
      r.eval.rouge_vs_transcript && r.eval.rouge_naive_vs_transcript
        ? (r.eval.rouge_vs_transcript.rougeL - r.eval.rouge_naive_vs_transcript.rougeL).toFixed(3) : '',
      r.eval.bertscore_vs_transcript?.f1 ?? '', r.eval.agent_coverage,
      r.eval.action_recall ?? '', (r.total_ms / 1000).toFixed(1),
    ].join(','));
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'lab_batch_results.csv'; a.click();
  };

  return (
    <div className="space-y-4">
      <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <h3 className="font-headline-md text-headline-md text-on-surface">Audio Library</h3>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            Add files to <code className="bg-surface-container px-1 rounded">backend/tests/audio/{'<Domain>'}/'</code>
          </span>
        </div>
        {allFiles.length === 0
          ? <p className="font-body-sm text-body-sm text-on-surface-variant">No audio files found. Add .mp3/.wav files to domain subfolders.</p>
          : <div className="space-y-1">
            {allFiles.map(f => (
              <label key={f.key} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-container-low cursor-pointer">
                <input type="checkbox" checked={selected.has(f.key)} onChange={() => toggle(f.key)} className="rounded flex-shrink-0" />
                <span className={`px-2 py-0.5 rounded font-label-sm text-label-sm ${DOMAIN_COLORS[f.domain as DomainName] ?? 'bg-surface-container text-on-surface'}`}>{f.domain}</span>
                <span className="flex-1 font-body-md text-body-md text-on-surface truncate">{f.filename}</span>
                <span className="font-label-md text-label-md text-on-surface-variant">{formatBytes(f.size_bytes)}</span>
              </label>
            ))}
          </div>}
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
                      <td className="px-3 py-2">
                        <span className={`px-2 py-0.5 rounded font-label-sm text-label-sm ${DOMAIN_COLORS[r.domain as DomainName] ?? 'bg-surface-container text-on-surface'}`}>{r.domain}</span>
                      </td>
                      <td className="px-3 py-2 font-body-sm text-body-sm text-on-surface max-w-[140px] truncate">{r.filename ?? '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface-variant tabular-nums">{r.transcript_chars?.toLocaleString() ?? '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{rougeL != null ? pct(rougeL) : '—'}</td>
                      <td className={`px-3 py-2 font-label-md text-label-md tabular-nums ${rougeL != null && rougeLNaive != null ? (rougeL - rougeLNaive > 0.005 ? 'text-primary' : 'text-on-surface-variant') : ''}`}>
                        {rougeL != null && rougeLNaive != null ? ((rougeL - rougeLNaive >= 0 ? '+' : '') + pct(rougeL - rougeLNaive)) : '—'}
                      </td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{bertF1 != null ? pct(bertF1) : '—'}</td>
                      <td className="px-3 py-2 min-w-[120px]"><ScoreBar value={r.eval.agent_coverage} /></td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface tabular-nums">{r.eval.action_recall != null ? pct(r.eval.action_recall) : '—'}</td>
                      <td className="px-3 py-2 font-label-md text-label-md text-on-surface-variant tabular-nums">{fmtMs(r.total_ms)}</td>
                    </tr>
                    {expandedRow === key && (
                      <tr key={key + '-detail'} className="border-t border-outline-variant bg-surface-container-lowest">
                        <td colSpan={9} className="p-4">
                          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                            <TracePanel result={r} />
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
// Dataset case selector
// ---------------------------------------------------------------------------

const SOURCE_LABELS: Record<string, string> = {
  synthetic:            'Built-in cases',
  qmsum:                'QMSum',
  'aci-bench':          'ACI-Bench',
  'mit-ocw':            'MIT OCW',
  'coding-interviews':  'Coding Interviews',
};

function DatasetSelector({ domain, onSelect }: { domain: string; onSelect: (transcript: string, caseId: string) => void }) {
  const [cases, setCases] = useState<EvalCaseMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingCase, setLoadingCase] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getLabDatasets(domain).then(setCases).catch(() => {}).finally(() => setLoading(false));
  }, [domain]);

  const loadCase = async (caseId: string) => {
    setLoadingCase(caseId);
    try {
      const c = await getLabCase(caseId);
      onSelect(c.transcript, c.id);
    } catch { /* ignore */ }
    finally { setLoadingCase(null); }
  };

  if (loading) return <p className="font-body-sm text-body-sm text-on-surface-variant">Loading cases…</p>;
  if (!cases.length) return <p className="font-body-sm text-body-sm text-on-surface-variant">No cases for this domain.</p>;

  // Group by source, synthetic first
  const grouped = cases.reduce<Record<string, EvalCaseMeta[]>>((acc, c) => {
    (acc[c.source] ??= []).push(c);
    return acc;
  }, {});
  const sourceOrder = ['synthetic', ...Object.keys(grouped).filter(s => s !== 'synthetic')];

  return (
    <div className="space-y-4">
      {sourceOrder.filter(s => grouped[s]).map(source => (
        <div key={source} className="space-y-1.5">
          <div className="flex items-center gap-2">
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wide">
              {SOURCE_LABELS[source] ?? source}
            </p>
            {source !== 'synthetic' && (
              <span className="px-1.5 py-0.5 bg-surface-container rounded font-label-sm text-label-sm text-on-surface-variant">
                public dataset
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {grouped[source].map(c => (
              <button key={c.id} onClick={() => loadCase(c.id)} disabled={loadingCase !== null}
                className={`flex flex-col items-start px-3 py-2 rounded-lg border text-left transition-colors disabled:opacity-50
                  ${loadingCase === c.id ? 'border-primary bg-primary-fixed' : 'border-outline-variant hover:bg-surface-container-low'}`}>
                <span className="font-label-md text-label-md text-on-surface">{c.title}</span>
                <span className="font-label-sm text-label-sm text-on-surface-variant">
                  {(c.transcript_length / 1000).toFixed(1)}k chars
                  {c.fact_count > 0 && ` · ${c.fact_count} facts`}
                </span>
                {loadingCase === c.id && (
                  <span className="font-label-sm text-label-sm text-primary flex items-center gap-1 mt-0.5">
                    <span className="material-symbols-outlined text-[12px] animate-spin">progress_activity</span>
                    Loading…
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HistoryPanel — list of persisted runs, click to reload
// ---------------------------------------------------------------------------

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function HistoryPanel({
  onLoad,
  refreshTrigger,
}: {
  onLoad: (result: LabRunResult) => void;
  refreshTrigger: number;
}) {
  const [list, setList] = useState<LabHistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getLabHistory().then(setList).catch(() => setList([])).finally(() => setLoading(false));
  }, [refreshTrigger]);

  const handleLoad = async (id: string) => {
    setLoadingId(id);
    try {
      const res = await getLabHistoryRun(id);
      onLoad(res);
    } catch { /* ignore */ } finally {
      setLoadingId(null);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteLabHistoryRun(id);
      setList(l => l.filter(e => e.id !== id));
    } catch { /* ignore */ }
  };

  if (loading) return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 text-center">
      <span className="font-body-sm text-body-sm text-on-surface-variant">Loading history…</span>
    </div>
  );

  if (!list.length) return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 text-center">
      <span className="font-body-sm text-body-sm text-on-surface-variant">No runs saved yet. Complete a pipeline run to see it here.</span>
    </div>
  );

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl divide-y divide-outline-variant overflow-hidden">
      {list.map(entry => (
        <button
          key={entry.id}
          onClick={() => handleLoad(entry.id)}
          disabled={loadingId === entry.id}
          className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-container transition-colors text-left group disabled:opacity-60"
        >
          {/* Domain badge */}
          <span className={`flex-shrink-0 px-2 py-0.5 rounded text-xs font-medium ${DOMAIN_COLORS[entry.domain as DomainName] ?? 'bg-surface-container text-on-surface'}`}>
            {entry.domain}
          </span>

          {/* Label / case id */}
          <span className="flex-1 min-w-0">
            <span className="font-body-sm text-body-sm text-on-surface truncate block">
              {entry.run_label || entry.case_id || <span className="italic text-on-surface-variant">Unlabeled</span>}
            </span>
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              {relativeTime(entry.saved_at)} · {entry.step_count} steps · {fmtMs(entry.total_ms)}
              {entry.chunked && <span className="ml-1 text-secondary">chunked</span>}
            </span>
          </span>

          {/* Scores */}
          <div className="flex-shrink-0 flex items-center gap-3 text-right">
            {entry.confidence_score != null && (
              <span className={`px-2 py-0.5 rounded-full font-label-sm text-label-sm ${scoreColor(entry.confidence_score)}`}>
                {entry.confidence_score.toFixed(1)}
              </span>
            )}
            {entry.eval_coverage != null && (
              <span className="font-label-sm text-label-sm text-on-surface-variant tabular-nums">
                {pct(entry.eval_coverage)} cov
              </span>
            )}
            {entry.eval_coverage_delta != null && entry.eval_coverage_delta > 0 && (
              <span className="font-label-sm text-label-sm text-primary tabular-nums">
                +{pct(entry.eval_coverage_delta)}
              </span>
            )}
          </div>

          {/* Load indicator / delete */}
          <div className="flex-shrink-0 flex items-center gap-1">
            {loadingId === entry.id
              ? <span className="material-symbols-outlined text-[16px] text-primary animate-spin">progress_activity</span>
              : <span className="material-symbols-outlined text-[16px] text-on-surface-variant opacity-0 group-hover:opacity-100 transition-opacity">open_in_new</span>
            }
            <button
              onClick={e => handleDelete(entry.id, e)}
              className="p-0.5 rounded hover:bg-error-container hover:text-on-error-container text-on-surface-variant opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <span className="material-symbols-outlined text-[16px]">delete</span>
            </button>
          </div>
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function TestingLab() {
  const [mode, setMode] = useState<Mode>('single');
  const [domain, setDomain] = useState<DomainName>('Project');
  const [transcript, setTranscript] = useState('');
  const [activeCase, setActiveCase] = useState<string | null>(null);
  const [transcriptSource, setTranscriptSource] = useState<'text' | 'audio' | 'dataset'>('dataset');
  const [knowledgeBase, setKnowledgeBase] = useState('');
  const [workflowConfig, setWorkflowConfig] = useState<WorkflowOverride | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [resultA, setResultA] = useState<LabRunResult | null>(null);
  const [resultB, setResultB] = useState<LabRunResult | null>(null);
  const [activeSlot, setActiveSlot] = useState<Slot>('A');
  const [error, setError] = useState<string | null>(null);
  const [lmOnline, setLmOnline] = useState<boolean | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = useCallback(() => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } }, []);

  useEffect(() => {
    getLMStudioStatus().then(s => setLmOnline(s.connected)).catch(() => setLmOnline(false));
  }, []);

  // Reset results when domain changes
  useEffect(() => {
    setResultA(null); setResultB(null); setError(null);
    setActiveCase(null); setTranscript('');
  }, [domain]);

  const handleRun = async () => {
    setIsRunning(true); setError(null); setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    try {
      const body: Parameters<typeof runLabPipeline>[0] = {
        domain,
        knowledge_base: knowledgeBase || undefined,
        workflow_override: workflowConfig ?? undefined,
        run_label: mode === 'ab' ? activeSlot : undefined,
      };
      if (activeCase && transcriptSource === 'dataset') {
        body.case_id = activeCase;
      } else if (transcript.trim()) {
        body.transcript = transcript;
      } else {
        setError('Provide a transcript or select a dataset case.'); stopTimer(); setIsRunning(false); return;
      }
      const res = await runLabPipeline(body);
      if (mode === 'ab' && activeSlot === 'B') setResultB(res);
      else setResultA(res);
      setHistoryRefresh(n => n + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { stopTimer(); setIsRunning(false); }
  };

  const activeResult = mode === 'ab' && activeSlot === 'B' ? resultB : resultA;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <Breadcrumb items={[{ label: 'Testing Lab' }]} />

      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface">Agent Testing Lab</h1>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Run the full multi-agent pipeline against built-in text cases or custom transcripts. Compare agentic vs naive baseline.
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
      <div className="flex items-center gap-2">
        {(['single', 'ab', 'batch'] as Mode[]).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded-full font-label-md text-label-md transition-colors ${mode === m ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
            {m === 'single' ? 'Single run' : m === 'ab' ? 'A/B compare' : 'Batch test'}
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={() => setHistoryOpen(o => !o)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full font-label-md text-label-md transition-colors ${historyOpen ? 'bg-secondary-container text-on-secondary-container' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}
        >
          <span className="material-symbols-outlined text-[16px]">history</span>
          History
        </button>
      </div>

      {/* History panel */}
      {historyOpen && (
        <HistoryPanel
          refreshTrigger={historyRefresh}
          onLoad={res => { setResultA(res); setHistoryOpen(false); }}
        />
      )}

      {/* Batch mode */}
      {mode === 'batch' && <BatchView workflowConfig={workflowConfig} />}

      {/* Single / A/B mode */}
      {mode !== 'batch' && (
        <>
          {/* Domain selector — square tags per UI guidelines */}
          <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-3">
            <p className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide">Domain</p>
            <div className="flex flex-wrap gap-2">
              {DOMAINS.map(d => (
                <button key={d} onClick={() => setDomain(d)}
                  className={`px-3 py-1.5 rounded font-label-md text-label-md transition-colors ${domain === d ? DOMAIN_COLORS[d] + ' ring-2 ring-primary' : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high'}`}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          {/* Input card */}
          <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 space-y-4">
            {/* Source toggle */}
            <div className="flex items-center gap-3 flex-wrap">
              <p className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wide flex-1 min-w-[80px]">Input</p>
              <div className="flex gap-1 bg-surface-container rounded-lg p-0.5">
                {(['dataset', 'text', 'audio'] as const).map(s => (
                  <button key={s} onClick={() => setTranscriptSource(s)}
                    className={`px-3 py-1 rounded-md font-label-md text-label-md capitalize transition-colors ${transcriptSource === s ? 'bg-surface-container-lowest text-on-surface shadow-sm' : 'text-on-surface-variant hover:text-on-surface'}`}>
                    <span className="material-symbols-outlined text-[14px] mr-1 align-middle">{s === 'dataset' ? 'dataset' : s === 'text' ? 'text_fields' : 'mic'}</span>
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {transcriptSource === 'dataset' && (
              <DatasetSelector domain={domain} onSelect={(t, id) => { setTranscript(t); setActiveCase(id); }} />
            )}
            {transcriptSource === 'audio' && (
              <AudioDropZone onTranscript={(text) => { setTranscript(text); setActiveCase(null); setTranscriptSource('text'); }} />
            )}
            {(transcriptSource === 'text' || (transcriptSource === 'dataset' && activeCase)) && (
              <div className="space-y-2">
                {activeCase && transcriptSource === 'dataset' && (
                  <div className="flex items-center gap-2 px-3 py-2 bg-surface-container rounded-lg">
                    <span className="material-symbols-outlined text-[16px] text-primary">description</span>
                    <span className="font-label-md text-label-md text-on-surface flex-1">Using case: <strong>{activeCase}</strong></span>
                    <button onClick={() => { setActiveCase(null); setTranscript(''); }} className="text-on-surface-variant hover:text-error">
                      <span className="material-symbols-outlined text-[16px]">close</span>
                    </button>
                  </div>
                )}
                <textarea
                  value={transcript}
                  onChange={e => { setTranscript(e.target.value); if (activeCase) setActiveCase(null); }}
                  rows={transcriptSource === 'dataset' ? 6 : 10}
                  className="w-full bg-surface-container rounded-lg px-3 py-2 font-body-sm text-body-sm text-on-surface resize-y border border-outline-variant focus:outline-none focus:ring-1 focus:ring-primary"
                  placeholder="Paste a transcript or select a dataset case above…"
                />
              </div>
            )}

            <details>
              <summary className="cursor-pointer font-label-md text-label-md text-on-surface-variant select-none">Project knowledge base (optional)</summary>
              <textarea value={knowledgeBase} onChange={e => setKnowledgeBase(e.target.value)} rows={3}
                className="mt-2 w-full bg-surface-container rounded-lg px-3 py-2 font-body-sm text-body-sm text-on-surface resize-y border border-outline-variant focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="Team names, project context, abbreviations…" />
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
              <button onClick={handleRun} disabled={isRunning || (!transcript.trim() && !activeCase)}
                className="flex items-center gap-2 bg-primary text-on-primary px-5 py-2 rounded-lg font-label-md text-label-md hover:opacity-90 transition-opacity disabled:opacity-40">
                {isRunning
                  ? <><span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span>Running… {elapsed}s</>
                  : <><span className="material-symbols-outlined text-[16px]">play_arrow</span>{mode === 'ab' ? `Run ${activeSlot}` : 'Run Pipeline'}</>}
              </button>
              {activeResult && !isRunning && (
                <span className="font-label-md text-label-md text-on-surface-variant">
                  Completed in {fmtMs(activeResult.total_ms)} · {activeResult.steps.length} steps
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

          {/* A/B comparison — both results present */}
          {mode === 'ab' && resultA && resultB && <ABComparisonView a={resultA} b={resultB} />}

          {/* Single run results */}
          {mode !== 'ab' && resultA && (
            <>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <TracePanel result={resultA} />
                <EvalPanel result={resultA} />
              </div>
              <OutputTabs result={resultA} />
            </>
          )}

          {/* A/B partial: show whichever slot is ready */}
          {mode === 'ab' && (resultA || resultB) && !(resultA && resultB) && (
            <>
              <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-4 text-center">
                <p className="font-body-sm text-body-sm text-on-surface-variant">Run {resultA ? 'B' : 'A'} to see the comparison.</p>
              </div>
              {(resultA ?? resultB) && (
                <>
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    <TracePanel result={(resultA ?? resultB)!} />
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
