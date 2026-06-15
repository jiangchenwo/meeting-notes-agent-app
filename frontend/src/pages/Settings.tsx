import { useEffect, useState } from 'react';
import { apiFetch } from '../api/client';
import Select from '../components/Select';
import { getLMConfig, updateLMConfig, getLMStudioStatus, getWhisperConfig, updateWhisperConfig } from '../api/summarize';
import type { LMConfig, LMStudioStatus, WhisperConfig } from '../api/types';

type RestoreResult = { ok: boolean; restored: { domains: number; templates: number } };

export default function Settings() {
  const [restoring, setRestoring] = useState(false);
  const [restoreMsg, setRestoreMsg] = useState<string | null>(null);

  const [lmConfig, setLmConfig] = useState<LMConfig>({ base_url: '', model: '', max_tokens: 4096, max_response_tokens: 2048, global_system_prompt: '' });
  const [lmStatus, setLmStatus] = useState<LMStudioStatus | null>(null);
  const [lmSaving, setLmSaving] = useState(false);
  const [lmSaveMsg, setLmSaveMsg] = useState<string | null>(null);
  const [lmTesting, setLmTesting] = useState(false);

  const [whisperConfig, setWhisperConfig] = useState<WhisperConfig>({ binary_path: '', model: 'base', model_path: '', available_models: [] });
  const [whisperSaving, setWhisperSaving] = useState(false);
  const [whisperSaveMsg, setWhisperSaveMsg] = useState<string | null>(null);

  useEffect(() => {
    getLMConfig().then(setLmConfig).catch(() => {});
    getLMStudioStatus().then(setLmStatus).catch(() => setLmStatus({ connected: false, models: [] }));
    getWhisperConfig().then(setWhisperConfig).catch(() => {});
  }, []);

  const handleBackup = () => {
    const a = document.createElement('a');
    a.href = '/api/settings/backup';
    a.download = '';
    a.click();
  };

  const handleRestoreDefaults = async () => {
    setRestoring(true);
    setRestoreMsg(null);
    try {
      const result = await apiFetch<RestoreResult>('/settings/restore-defaults', { method: 'POST' });
      const { domains, templates } = result.restored;
      if (domains === 0 && templates === 0) {
        setRestoreMsg('All defaults are already present — nothing to restore.');
      } else {
        setRestoreMsg(`Restored ${domains} domain${domains !== 1 ? 's' : ''} and ${templates} template${templates !== 1 ? 's' : ''}.`);
      }
    } catch {
      setRestoreMsg('Restore failed — check that the backend is running.');
    } finally {
      setRestoring(false);
    }
  };

  const handleSaveLMConfig = async () => {
    setLmSaving(true);
    setLmSaveMsg(null);
    try {
      const saved = await updateLMConfig(lmConfig);
      setLmConfig(saved);
      setLmSaveMsg('Saved.');
    } catch {
      setLmSaveMsg('Save failed.');
    } finally {
      setLmSaving(false);
    }
  };

  const handleTestConnection = async () => {
    setLmTesting(true);
    try {
      const status = await getLMStudioStatus();
      setLmStatus(status);
    } catch {
      setLmStatus({ connected: false, models: [] });
    } finally {
      setLmTesting(false);
    }
  };

  const handleSaveWhisper = async () => {
    setWhisperSaving(true);
    setWhisperSaveMsg(null);
    try {
      const saved = await updateWhisperConfig({
        binary_path: whisperConfig.binary_path,
        model: whisperConfig.model,
        model_path: whisperConfig.model_path,
      });
      setWhisperConfig(saved);
      setWhisperSaveMsg('Saved.');
    } catch {
      setWhisperSaveMsg('Save failed.');
    } finally {
      setWhisperSaving(false);
    }
  };

  return (
    <main className="flex-1 w-full flex flex-col min-h-screen">
      <header className="flex justify-between items-center px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full sticky top-0 bg-surface/90 backdrop-blur-md z-30 border-b border-outline-variant/30">
        <div>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface tracking-tight">
            Settings
          </h2>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Configure transcription, LM Studio, and backup your data.
          </p>
        </div>
      </header>

      <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full space-y-space-6 max-w-xl">

        {/* Whisper (Transcription) */}
        <section className="bg-surface-container-lowest rounded-xl overflow-hidden">
          <div className="px-space-6 py-space-4 border-b border-outline-variant">
            <h3 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px] text-primary">mic</span>
              Whisper Transcription
            </h3>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
              Configure the whisper.cpp binary and model for transcription.
            </p>
          </div>

          <div className="p-space-6 space-y-space-4">
            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Whisper Binary Path</label>
              <input
                type="text"
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
                placeholder="/path/to/whisper.cpp/build/bin"
                value={whisperConfig.binary_path}
                onChange={(e) => setWhisperConfig((c) => ({ ...c, binary_path: e.target.value }))}
              />
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Directory containing the <code className="bg-surface-container rounded px-1">whisper-cli</code> binary.
              </p>
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Model</label>
              <Select
                value={whisperConfig.model}
                onChange={(v) => setWhisperConfig((c) => ({ ...c, model: v }))}
                options={whisperConfig.available_models.map((m) => ({ value: m.name, label: m.name }))}
                size="md"
              />
              {/* Model info table */}
              {whisperConfig.available_models.length > 0 && (
                <div className="mt-space-2 rounded border border-outline-variant overflow-hidden">
                  <div className="grid grid-cols-4 gap-0 text-[11px] font-label-sm text-on-surface-variant bg-surface border-b border-outline-variant px-3 py-1.5 uppercase tracking-wider">
                    <div>Model</div><div>Size</div><div>Speed</div><div>Quality</div>
                  </div>
                  {whisperConfig.available_models.map((m) => (
                    <div
                      key={m.name}
                      className={`grid grid-cols-4 gap-0 text-[11px] font-body-sm px-3 py-1.5 border-b border-outline-variant/50 last:border-0 ${
                        whisperConfig.model === m.name ? 'bg-primary-container/20 text-on-surface' : 'text-on-surface-variant'
                      }`}
                    >
                      <div className={whisperConfig.model === m.name ? 'font-bold text-primary' : ''}>{m.name}</div>
                      <div>{m.size}</div>
                      <div>{m.speed}</div>
                      <div>{m.quality}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
                Model File Path
                <span className="ml-2 font-body-sm text-body-sm text-on-surface-variant normal-case tracking-normal">
                  (optional — auto-detected from binary path)
                </span>
              </label>
              <input
                type="text"
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
                placeholder="/path/to/models/ggml-base.bin"
                value={whisperConfig.model_path}
                onChange={(e) => setWhisperConfig((c) => ({ ...c, model_path: e.target.value }))}
              />
            </div>

            {whisperSaveMsg && (
              <p className="font-body-sm text-body-sm text-primary">{whisperSaveMsg}</p>
            )}

            <button
              onClick={handleSaveWhisper}
              disabled={whisperSaving}
              className="bg-primary text-on-primary font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity shadow-sm disabled:opacity-50"
            >
              <span className={`material-symbols-outlined text-[18px] ${whisperSaving ? 'animate-spin' : ''}`}>
                {whisperSaving ? 'sync' : 'save'}
              </span>
              Save
            </button>
          </div>
        </section>

        {/* LM Studio */}
        <section className="bg-surface-container-lowest rounded-xl overflow-hidden">
          <div className="px-space-6 py-space-4 border-b border-outline-variant flex items-center justify-between">
            <div>
              <h3 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px] text-primary">psychology</span>
                LM Studio
              </h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
                Local LLM connection for generating summaries.
              </p>
            </div>
            {lmStatus && (
              <span className={`flex items-center gap-1.5 font-label-sm text-label-sm px-2.5 py-1 rounded-full border ${
                lmStatus.connected
                  ? 'bg-primary/10 text-primary border-primary/20'
                  : 'bg-error/10 text-error border-error/20'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${lmStatus.connected ? 'bg-primary' : 'bg-error'}`} />
                {lmStatus.connected ? `Connected · ${lmStatus.models.length} model${lmStatus.models.length !== 1 ? 's' : ''}` : 'Not connected'}
              </span>
            )}
          </div>

          <div className="p-space-6 space-y-space-4">
            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Base URL</label>
              <input
                type="text"
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
                placeholder="http://localhost:1234/v1"
                value={lmConfig.base_url}
                onChange={(e) => setLmConfig((c) => ({ ...c, base_url: e.target.value }))}
              />
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
                Model
                <span className="ml-2 font-body-sm text-body-sm text-on-surface-variant normal-case tracking-normal">
                  (leave blank to use whatever is loaded)
                </span>
              </label>
              {lmStatus?.connected && lmStatus.models.length > 0 ? (
                <Select
                  value={lmConfig.model}
                  onChange={(v) => setLmConfig((c) => ({ ...c, model: v }))}
                  options={[
                    { value: '', label: '— use currently loaded model —' },
                    ...lmStatus.models.map((m) => ({ value: m, label: m })),
                  ]}
                  size="md"
                />
              ) : (
                <input
                  type="text"
                  className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all"
                  placeholder="e.g. lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
                  value={lmConfig.model}
                  onChange={(e) => setLmConfig((c) => ({ ...c, model: e.target.value }))}
                />
              )}
              {!lmStatus?.connected && (
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Click <strong>Test</strong> to connect and load available models.
                </p>
              )}
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
                Max Context Tokens
              </label>
              <input
                type="number"
                min={512}
                max={128000}
                step={512}
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all w-40"
                value={lmConfig.max_tokens}
                onChange={(e) => setLmConfig((c) => ({ ...c, max_tokens: Number(e.target.value) }))}
              />
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Transcripts longer than this are truncated before being sent to the LLM.
              </p>
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
                Max Response Tokens
              </label>
              <input
                type="number"
                min={256}
                max={32000}
                step={256}
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all w-40"
                value={lmConfig.max_response_tokens}
                onChange={(e) => setLmConfig((c) => ({ ...c, max_response_tokens: Number(e.target.value) }))}
              />
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Maximum tokens the LLM can generate. Increase if summaries are cut short (default: 2048).
              </p>
            </div>

            <div className="flex flex-col gap-space-1">
              <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">
                Global System Prompt
              </label>
              <textarea
                rows={8}
                className="px-space-3 py-2 rounded border border-outline-variant bg-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none font-body-md text-body-md text-on-surface transition-all resize-y leading-relaxed"
                value={lmConfig.global_system_prompt}
                onChange={(e) => setLmConfig((c) => ({ ...c, global_system_prompt: e.target.value }))}
              />
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Applied to every summarization task. Project-specific prompts are appended after this.
              </p>
            </div>

            {lmSaveMsg && (
              <p className="font-body-sm text-body-sm text-primary">{lmSaveMsg}</p>
            )}

            <div className="flex gap-space-3">
              <button
                onClick={handleTestConnection}
                disabled={lmTesting}
                className="border border-outline-variant bg-surface-container-lowest text-on-surface font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:bg-surface-container-low transition-colors shadow-sm disabled:opacity-50"
              >
                <span className={`material-symbols-outlined text-[18px] ${lmTesting ? 'animate-spin' : ''}`}>
                  {lmTesting ? 'sync' : 'wifi'}
                </span>
                Test
              </button>
              <button
                onClick={handleSaveLMConfig}
                disabled={lmSaving}
                className="bg-primary text-on-primary font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity shadow-sm disabled:opacity-50"
              >
                <span className={`material-symbols-outlined text-[18px] ${lmSaving ? 'animate-spin' : ''}`}>
                  {lmSaving ? 'sync' : 'save'}
                </span>
                Save
              </button>
            </div>
          </div>
        </section>

        {/* Templates & Domains */}
        <section className="bg-surface-container-lowest rounded-xl overflow-hidden">
          <div className="px-space-6 py-space-4 border-b border-outline-variant">
            <h3 className="font-headline-md text-headline-md text-on-surface">Templates &amp; Domains</h3>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
              Export all templates and domains as JSON, or restore the built-in defaults.
            </p>
          </div>

          <div className="divide-y divide-outline-variant/40">
            <div className="px-space-6 py-space-4 flex items-center justify-between gap-4">
              <div>
                <p className="font-label-md text-label-md text-on-surface">Export backup</p>
                <p className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
                  Downloads a JSON file with all domains, templates, and projects.
                </p>
              </div>
              <button
                onClick={handleBackup}
                className="shrink-0 border border-outline-variant bg-surface-container-lowest text-on-surface font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:bg-surface-container-low transition-colors shadow-sm"
              >
                <span className="material-symbols-outlined text-[18px]">download</span>
                Download
              </button>
            </div>

            <div className="px-space-6 py-space-4 flex items-center justify-between gap-4">
              <div>
                <p className="font-label-md text-label-md text-on-surface">Restore defaults</p>
                <p className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
                  Re-adds any missing built-in domains and templates. Custom ones are not affected.
                </p>
                {restoreMsg && (
                  <p className="font-body-sm text-[12px] text-primary mt-1">{restoreMsg}</p>
                )}
              </div>
              <button
                onClick={handleRestoreDefaults}
                disabled={restoring}
                className="shrink-0 border border-outline-variant bg-surface-container-lowest text-on-surface font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:bg-surface-container-low transition-colors shadow-sm disabled:opacity-50"
              >
                {restoring ? (
                  <>
                    <span className="material-symbols-outlined text-[18px] animate-spin">sync</span>
                    Restoring…
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-[18px]">restore</span>
                    Restore
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
