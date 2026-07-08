import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import Select from '../components/Select';
import { getDomains, getTemplate, createTemplate, updateTemplate } from '../api/domains';
import type { Domain } from '../api/types';

const PLACEHOLDERS = [
  { key: '{transcript}', label: 'Transcript', desc: 'Full meeting transcript text' },
  { key: '{domain}', label: 'Domain', desc: 'Domain name (e.g. Project)' },
  { key: '{project_context}', label: 'Project Context', desc: 'Project system prompt / context' },
  { key: '{knowledge_base}', label: 'Knowledge Base', desc: 'Project knowledge base text' },
];

const OUTPUT_SECTIONS = [
  { key: 'summary', label: 'Summary' },
  { key: 'action_items', label: 'Action Items' },
  { key: 'suggestions', label: 'Suggestions' },
];

export default function EditTemplate() {
  const { templateId } = useParams<{ templateId: string }>();
  const isNew = templateId === 'new';
  const navigate = useNavigate();

  const [domains, setDomains] = useState<Domain[]>([]);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [domainId, setDomainId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState('');
  const [outputSections, setOutputSections] = useState<string[]>(['summary', 'action_items']);
  const [workflowConfig, setWorkflowConfig] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [isBuiltin, setIsBuiltin] = useState(false);
  const [loading, setLoading] = useState(!isNew);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const loadDomains = getDomains().then(setDomains);
    if (!isNew && templateId) {
      Promise.all([loadDomains, getTemplate(Number(templateId)).then((t) => {
        setName(t.name);
        setDescription(t.description);
        setDomainId(t.domain_id);
        setPrompt(t.prompt_template);
        setOutputSections(t.output_sections);
        setWorkflowConfig(t.workflow_config ?? '');
        setAdvancedOpen(Boolean(t.workflow_config));
        setIsBuiltin(t.is_builtin);
      })]).finally(() => setLoading(false));
    } else {
      loadDomains.finally(() => setLoading(false));
    }
  }, [templateId, isNew]);

  const insertPlaceholder = (text: string) => {
    const el = textareaRef.current;
    if (!el) return;
    const start = el.selectionStart ?? prompt.length;
    const end = el.selectionEnd ?? prompt.length;
    const next = prompt.slice(0, start) + text + prompt.slice(end);
    setPrompt(next);
    setTimeout(() => {
      el.focus();
      el.selectionStart = el.selectionEnd = start + text.length;
    }, 0);
  };

  const toggleSection = (key: string) => {
    setOutputSections((prev) =>
      prev.includes(key) ? prev.filter((s) => s !== key) : [...prev, key]
    );
  };

  const handleSave = async () => {
    if (!name.trim()) { setError('Template name is required.'); return; }
    const wf = workflowConfig.trim();
    if (wf) {
      try {
        JSON.parse(wf);
      } catch {
        setError('Workflow config is not valid JSON.');
        return;
      }
    }
    setSaving(true);
    setError('');
    try {
      const body = {
        name: name.trim(),
        description: description.trim(),
        domain_id: domainId,
        prompt_template: prompt,
        output_sections: outputSections,
        workflow_config: wf || null,
      };
      if (isNew) {
        await createTemplate(body);
      } else {
        await updateTemplate(Number(templateId), body);
      }
      navigate('/management');
    } catch (e) {
      // Backend 422 carries the WorkflowSpec validation detail.
      const msg = e instanceof Error && e.message ? e.message.slice(0, 400) : '';
      setError(msg ? `Failed to save: ${msg}` : 'Failed to save. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-on-surface-variant font-body-md text-body-md">
        Loading…
      </div>
    );
  }

  return (
    <main className="flex-1 bg-surface">
      <div className="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-space-8">

        {/* Header */}
        <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-space-8">
          <div>
            <div className="flex items-center gap-2 text-on-surface-variant mb-2">
              <Link to="/management" className="font-label-md text-label-md hover:text-primary transition-colors flex items-center gap-1">
                <span className="material-symbols-outlined text-[16px]">arrow_back</span>
                Templates
              </Link>
              <span className="font-label-md text-label-md">/</span>
              <span className="font-label-md text-label-md text-primary">
                {isNew ? 'New Template' : 'Edit Template'}
              </span>
            </div>
            <h1 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface">
              {isNew ? 'New Template' : name || 'Edit Template'}
            </h1>
          </div>
          <div className="flex items-center gap-space-3 w-full sm:w-auto">
            <Link
              to="/management"
              className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2 border border-outline-variant rounded bg-surface-container-lowest text-on-surface-variant font-label-md text-label-md hover:bg-surface-container-low transition-colors"
            >
              Cancel
            </Link>
            <button
              onClick={handleSave}
              disabled={saving || !name.trim()}
              className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2 bg-primary text-on-primary rounded font-label-md text-label-md hover:bg-primary/90 transition-colors shadow-sm disabled:opacity-50"
            >
              <span className="material-symbols-outlined text-[18px]">save</span>
              {saving ? 'Saving…' : isNew ? 'Create Template' : 'Save Changes'}
            </button>
          </div>
        </header>

        {error && (
          <div className="mb-space-6 bg-error-container text-on-error-container rounded-lg px-4 py-3 font-body-sm text-body-sm">
            {error}
          </div>
        )}

        {isBuiltin && (
          <div className="mb-space-6 bg-surface-container-high text-on-surface-variant rounded-lg px-4 py-3 font-body-sm text-body-sm flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px]">info</span>
            This is a built-in template. Changes will be saved.
          </div>
        )}

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-6">

          {/* Left: Config */}
          <div className="lg:col-span-4 flex flex-col gap-space-5">

            {/* Template name */}
            <div className="bg-surface-container-lowest rounded-lg p-space-4 flex flex-col gap-space-4">
              <h2 className="font-headline-md text-headline-md text-on-surface border-b border-outline-variant pb-2">
                Configuration
              </h2>

              <div className="flex flex-col gap-1.5">
                <label className="font-label-md text-label-md text-on-surface-variant" htmlFor="template-name">
                  Template Name <span className="text-error">*</span>
                </label>
                <input
                  id="template-name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Technical Meeting"
                  className="w-full bg-surface border border-outline-variant rounded px-3 py-2 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-0 transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="font-label-md text-label-md text-on-surface-variant" htmlFor="template-description">
                  Description
                </label>
                <textarea
                  id="template-description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Short summary of what this template produces."
                  rows={2}
                  className="w-full bg-surface border border-outline-variant rounded px-3 py-2 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-0 resize-none transition-colors"
                />
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Shown on the template card. Not sent to the LLM.
                </p>
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="font-label-md text-label-md text-on-surface-variant" htmlFor="domain-select">
                  Domain
                </label>
                <Select
                  value={String(domainId ?? '')}
                  onChange={(v) => setDomainId(v ? Number(v) : null)}
                  options={[
                    { value: '', label: 'No specific domain' },
                    ...domains.map((d) => ({ value: String(d.id), label: d.name })),
                  ]}
                  size="md"
                />
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Limits which templates appear when a note has this domain selected.
                </p>
              </div>
            </div>

            {/* Output Sections */}
            <div className="bg-surface-container-lowest rounded-lg p-space-4">
              <h3 className="font-headline-md text-headline-md text-on-surface border-b border-outline-variant pb-2 mb-space-4">
                Output Sections
              </h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-3">
                Choose which sections the LLM should generate.
              </p>
              <div className="flex flex-col gap-3">
                {OUTPUT_SECTIONS.map((s) => (
                  <label key={s.key} className="flex items-center gap-3 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={outputSections.includes(s.key)}
                      onChange={() => toggleSection(s.key)}
                      className="w-4 h-4 rounded border-outline-variant text-primary focus:ring-primary focus:ring-1 cursor-pointer"
                    />
                    <span className="font-body-md text-body-md text-on-surface group-hover:text-primary transition-colors">
                      {s.label}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Advanced: agent workflow override */}
            <div className="bg-surface-container-lowest rounded-lg p-space-4">
              <button
                onClick={() => setAdvancedOpen((o) => !o)}
                className="w-full flex items-center justify-between font-headline-md text-headline-md text-on-surface"
              >
                <span>Advanced: Agent Workflow</span>
                <span className="material-symbols-outlined text-[20px] text-on-surface-variant">
                  {advancedOpen ? 'expand_less' : 'expand_more'}
                </span>
              </button>
              {advancedOpen && (
                <div className="mt-space-3 flex flex-col gap-space-2">
                  <p className="font-body-sm text-body-sm text-on-surface-variant">
                    JSON override for which agents run and which get a quality pass. Leave empty to use the domain default.
                    Agents: <code className="bg-surface-container rounded px-1">Summarizer</code>,{' '}
                    <code className="bg-surface-container rounded px-1">ActionItemExtractor</code>,{' '}
                    <code className="bg-surface-container rounded px-1">DecisionLogger</code>,{' '}
                    <code className="bg-surface-container rounded px-1">InterviewAgent</code>,{' '}
                    <code className="bg-surface-container rounded px-1">LectureAgent</code>.
                  </p>
                  <textarea
                    value={workflowConfig}
                    onChange={(e) => setWorkflowConfig(e.target.value)}
                    rows={8}
                    spellCheck={false}
                    placeholder={`{\n  "steps": ["Summarizer", "DecisionLogger"],\n  "critique_steps": ["Summarizer"],\n  "critique_threshold": 8,\n  "max_retries": 2\n}`}
                    className="w-full bg-surface border border-outline-variant rounded px-3 py-2 font-body-sm text-[12px] text-on-surface focus:outline-none focus:border-primary focus:ring-0 resize-y transition-colors font-mono leading-relaxed"
                  />
                  <p className="font-body-sm text-[11px] text-on-surface-variant">
                    Steps also accept <code className="bg-surface-container rounded px-1">{'{"agent": "Summarizer", "prompt_override": "…"}'}</code> to
                    replace the template prompt for that step only. Validated on save.
                  </p>
                </div>
              )}
            </div>

            {/* Placeholder reference */}
            <div className="bg-surface-container-lowest rounded-lg p-space-4">
              <h3 className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider mb-space-3">
                Available Placeholders
              </h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-3">
                Click to insert at cursor position in the prompt.
              </p>
              <div className="flex flex-col gap-2">
                {PLACEHOLDERS.map((p) => (
                  <button
                    key={p.key}
                    onClick={() => insertPlaceholder(p.key)}
                    className="flex items-start gap-2 text-left p-2 rounded border border-outline-variant hover:border-primary hover:bg-primary-fixed/10 transition-all group"
                  >
                    <code className="font-label-sm text-[11px] text-primary bg-primary-fixed/20 rounded px-1.5 py-0.5 shrink-0 mt-0.5">
                      {p.key}
                    </code>
                    <div>
                      <div className="font-label-md text-label-md text-on-surface group-hover:text-primary transition-colors">
                        {p.label}
                      </div>
                      <div className="font-body-sm text-[11px] text-on-surface-variant">{p.desc}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Prompt Editor */}
          <div className="lg:col-span-8 flex flex-col gap-space-4">
            <div className="bg-surface-container-lowest rounded-lg overflow-hidden flex flex-col" style={{ minHeight: 500 }}>
              <div className="px-space-4 py-space-3 border-b border-outline-variant bg-surface flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-on-surface-variant text-[18px]">edit_note</span>
                  <span className="font-label-md text-label-md text-on-surface">Prompt Template</span>
                </div>
                <span className="font-body-sm text-body-sm text-on-surface-variant">
                  {prompt.length} chars
                </span>
              </div>

              <div className="p-space-4 flex-1 flex flex-col">
                <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-3">
                  Write your prompt below. Use the placeholder buttons on the left to insert dynamic values.
                  The transcript and project context are automatically injected by the system.
                </p>
                <textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={`Describe what the LLM should do with the recording.\n\nExample:\nSummarize the key decisions and action items from this meeting.\nFocus on: technical decisions, owners, deadlines.\n\nTranscript:\n{transcript}`}
                  className="flex-1 w-full bg-surface border border-outline-variant rounded px-3 py-3 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-0 resize-none leading-relaxed transition-colors"
                  style={{ minHeight: 360, fontFamily: 'inherit' }}
                />
              </div>

              {/* Quick-insert inline chips */}
              <div className="px-space-4 pb-space-3 flex flex-wrap gap-2 border-t border-outline-variant/50 pt-space-3">
                <span className="font-label-sm text-label-sm text-on-surface-variant mr-1">Quick insert:</span>
                {PLACEHOLDERS.map((p) => (
                  <button
                    key={p.key}
                    onClick={() => insertPlaceholder(p.key)}
                    className="font-label-sm text-[11px] text-primary bg-primary-fixed/20 border border-primary/20 rounded px-2 py-0.5 hover:bg-primary/10 transition-colors"
                  >
                    {p.key}
                  </button>
                ))}
              </div>
            </div>

            {/* Preview: what the assembled prompt looks like */}
            <div className="bg-surface-container-lowest rounded-lg p-space-4">
              <h3 className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider mb-space-3 flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">preview</span>
                How it's assembled
              </h3>
              <div className="font-body-sm text-body-sm text-on-surface-variant space-y-2 bg-surface rounded border border-outline-variant/50 p-3">
                <div className="flex items-start gap-2">
                  <span className="text-[10px] bg-secondary-container text-on-secondary-container rounded px-1.5 py-0.5 font-label-sm shrink-0 mt-0.5">System</span>
                  <span>You are a meeting notes assistant. <span className="italic opacity-60">[+ project system prompt]</span></span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-[10px] bg-tertiary-container text-on-tertiary-container rounded px-1.5 py-0.5 font-label-sm shrink-0 mt-0.5">Context</span>
                  <span className="italic opacity-60">[project knowledge base]</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-[10px] bg-primary-container text-on-primary-container rounded px-1.5 py-0.5 font-label-sm shrink-0 mt-0.5">User</span>
                  <span className="line-clamp-3 whitespace-pre-wrap">
                    {prompt || <span className="italic opacity-50">← your prompt will appear here</span>}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
