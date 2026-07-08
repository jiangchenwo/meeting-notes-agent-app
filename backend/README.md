# Backend

FastAPI service that owns all application state and orchestrates the other three processes over HTTP: the host-native [ASR service](../asr-service/) (`:9000`) for transcription, LM Studio (`:1234`, any OpenAI-compatible endpoint) for generation, and the [frontend](../frontend/) (`:5173` dev) as its only client. Runs on `:8000`.

Everything is local-first: SQLite on disk, audio blobs on disk, settings as JSON files on disk. No cloud services, no `.env` for app settings.

## Quick start

Python tooling is **uv exclusively** (never pip/venv directly).

```bash
uv sync
uv run uvicorn main:app --reload          # http://localhost:8000  (docs at /docs)
uv run pytest                             # all tests (no LLM required)
uv run pytest -k "speaker"                # by keyword
uv run --group eval python -m eval        # offline eval vs live LM Studio (see Eval below)
```

Containerized, the backend is built from [Dockerfile](Dockerfile) and started with `make up` at the repo root (see the [root README](../README.md#getting-started)); `tests/` and `eval/` are excluded from the image.

## Module map

```
backend/
├── main.py                # app wiring + startup: storage migrate → create_all → column ALTERs →
│                          #   duration backfill → seed → reset stuck notes → telemetry
├── database.py            # engine + SessionLocal; SQLite WAL mode via connect listener
├── models.py              # all SQLAlchemy models
├── schemas.py             # all Pydantic request/response schemas
├── seed.py                # built-in domains/templates (idempotent, runs at startup)
├── storage.py             # /data volume layout (Docker) + legacy-layout migration
├── config_store.py        # JSON-file settings store under CONFIG_DIR
├── lm_config.py           # LM Studio settings (base URL, model, tokens, output_mode…)
├── asr_config.py          # ASR service settings (base URL, diarize default)
├── telemetry_config.py    # Phoenix tracing settings (enabled, endpoint, capture_content)
├── telemetry.py           # OTel TracerProvider lifecycle + trace_span()/workflow_span()
├── asr_client.py          # HTTP client for the ASR service
├── transcript_format.py   # build_speaker_transcript(): segments_json → "Speaker: text"
├── audio_utils.py         # duration probing
├── routers/               # one file per API concern (see Request surface)
├── agents/                # the agentic pipeline (see below)
├── eval/                  # offline eval harness (see below)
└── tests/                 # pytest suite; never hits a real LLM
```

## Request surface

All routes live under `/api`. One router file per concern:

| Router | Endpoints |
| --- | --- |
| `uploads.py` | `POST /api/upload` — audio file → new note (`pending`) |
| `notes.py` | notes CRUD, `/search`, `/bulk-update`, `/bulk-delete` |
| `transcribe.py` | `POST /api/notes/{id}/transcribe` (BackgroundTask → ASR service), `GET/PATCH …/transcription` |
| `workflow.py` | `POST /api/notes/{id}/run-workflow` (BackgroundTask → agentic pipeline), `GET …/workflow-run`, `GET …/workflow-run/steps` |
| `summarize.py` | `POST …/summarize` (thin alias for run-workflow), `GET/PATCH …/summary`, `GET …/prompt-preview` |
| `projects.py` | projects CRUD + per-project speaker roster (`…/speakers`) |
| `domains.py` | domains + templates CRUD; validates `workflow_config` JSON on write (422) |
| `settings.py` | `GET/PUT /api/settings/{llm,asr,telemetry}`, connection status probes, backup, restore-defaults |
| `export.py` | `GET /api/notes/{id}/export` — Markdown / plain text |

## Data model

A recording is a `NoteBlock` moving through a status machine:

```
pending → transcribing → transcribed → summarizing → done   (or error)
```

- `Transcription` / `Summary` — 1:1 tables hanging off a note. `Transcription.segments_json` holds the timestamped segments (each with a per-segment `speaker`); `full_text` is the flattened text **without** speaker labels — speaker-labeled transcripts are re-derived on demand by `transcript_format.build_speaker_transcript(full_text, segments_json)`.
- `Project` — groups notes; its `custom_system_prompt` + `knowledge_base` are injected into every LLM prompt for its notes. `ProjectSpeaker` is the per-project speaker-name roster.
- `Domain` + `Template` — select which agent workflow and prompt run. A template's `workflow_config` JSON overrides the domain's workflow plan.
- `WorkflowRun` + `WorkflowStepResult` — one row per pipeline run / per step, written live while the run executes (status, per-step tokens, model, critique scores, `trace_id`). The frontend polls these for the per-agent progress UI.

## Transcription flow

`POST /api/notes/{id}/transcribe` schedules a FastAPI BackgroundTask (`routers/transcribe.py::_run_transcription`) → `asr_client.transcribe_via_asr` POSTs the audio to the ASR service `/transcribe` (multipart, `diarize` flag) → segments land in `Transcription.segments_json`, flattened text in `full_text`, and the note becomes `transcribed`.

## Agentic pipeline (`agents/`)

There is **one** summarization path: the Pydantic AI pipeline. `POST /api/notes/{id}/run-workflow` schedules `agents/orchestrator.py::run_workflow` as a BackgroundTask. Layered so the core engine is DB-free and reusable by the eval harness:

| Module | Role |
| --- | --- |
| `workflow_spec.py` | `WorkflowSpec` — validated data, no code. Per-domain plans in `DOMAIN_WORKFLOWS`; a template's `workflow_config` overrides them (422 on the write path, lenient fallback to the domain default on read). |
| `pipeline.py` | DB-free `run_pipeline()`: serial step loop → critique/retry → chunked map-reduce for over-length transcripts → assembly → non-LLM verifiers. Persistence only via a `PipelineObserver` callback. |
| `definitions.py` | Model-less Pydantic AI `Agent`s + **all** prompt composition (the prompt-preview endpoint reuses these builders). |
| `outputs.py` | Typed output models. **`model_dump()` keys are the persisted `result_json` / frontend contract — do not rename fields.** |
| `llm.py` | `build_model(cfg)` — per-run `OpenAIChatModel` from `lm_config.load()`, never at import time. `LLM_TIMEOUT_SECONDS = 480` (QMSum-scale prompts legitimately take 3+ min locally). |
| `orchestrator.py` | Thin DB wrapper: builds `NoteDeps`, runs the pipeline with an observer writing `WorkflowRun`/`WorkflowStepResult` rows live, then writes the `Summary`. |
| `context.py` | `NoteDeps` dataclass — everything prompt composition needs (domain, template prompt, project prompt/knowledge base, global system prompt). |
| `verifiers.py` | Non-LLM checks: output-schema shape + domain risk flags, recorded in `raw_sections_json`. |
| `tools.py` | `search_knowledge_base` agent tool. |

Execution shape per run:

```
Orchestrator (rule-based, no extra LLM call)
  ├─ Chunking (only if transcript exceeds the context window): map-reduce Summarizer → condensed transcript
  ├─ Extraction (serial): Summarizer → ActionItemExtractor → DecisionLogger / domain agent
  ├─ Critique: Critic scores selected steps 0–10; below-threshold steps retry with the
  │   critique feedback in the prompt (up to max_retries)
  ├─ Verification (non-LLM): schema shape + risk flags
  └─ Assembly (pure Python): Summary text, action items, suggestions, raw_sections_json
```

Invariants worth knowing before touching it:

- **Serial only.** LM Studio loads one model at a time — nothing may introduce parallel agent calls.
- **The Summarizer always generates plain text.** Its output is a single string, and LM Studio's grammar-constrained (`response_format: json_schema`) generation degenerates into whitespace loops on long inputs. Only the structured agents (extractors, critic, domain agents) use `lm_config`'s `output_mode`: `native` (json_schema, grammar-enforced, default) or `prompted` (schema in prompt — fallback for engines that reject json_schema). Never tool-call mode.
- **Retries keep the best-scoring attempt**, never a worse retry; a degenerately short summary of a long transcript gets an explicit length note appended to the retry feedback.
- **Critique scores are recomputed in Python** from the rubric dimensions — the LLM's own total is never trusted. Unparseable critiques fall back to an advisory 5.0.
- Composed prompts are persisted under `_prompt` in each step's `result_json`; token counts flow to the DB columns.

## Configuration (`config_store`, not env vars)

App settings are *runtime config* edited from the in-app Settings page — **never `.env`**. `config_store.py` reads/writes JSON files under `CONFIG_DIR` (the backend dir locally, `/data/config` in Docker). Each settings domain is a thin module with a fixed filename + `DEFAULTS` dict, surfaced through `routers/settings.py`:

| Module | File | Holds |
| --- | --- | --- |
| `lm_config.py` | `lm_config.json` | LM Studio base URL, model, max tokens, global system prompt, `output_mode` |
| `asr_config.py` | `asr_config.json` | ASR service base URL, diarization default |
| `telemetry_config.py` | `telemetry_config.json` | tracing enabled, OTLP endpoint, capture_content |

To add a setting: copy that pattern (new module + `DEFAULTS`, wire GET/PUT in `routers/settings.py`).

## Storage & database

**Local dev (default):** flat layout at the repo root — `notes.db`, `uploads/`, `*.json` config next to the code. `storage.py` is a no-op.

**Docker:** one persistent `/data` volume split by `storage.py` into `db/notes.db`, `uploads/`, `config/*.json`, driven by env vars set in the [Dockerfile](Dockerfile) (`DATABASE_URL`, `UPLOAD_DIR`, `CONFIG_DIR`). `storage.ensure_and_migrate()` runs at startup **before the DB is opened** and relocates any legacy flat files, so old volumes upgrade in place.

SQLite is opened in **WAL mode** (`synchronous=NORMAL`, `busy_timeout=5000`) via a `connect` listener in `database.py`, so background transcription/summarization writes don't block HTTP reads.

**Migrations:** `Base.metadata.create_all` creates new *tables* automatically but never adds a *column* to an existing table. When adding a column to an existing model, also append an idempotent `ALTER TABLE … ADD COLUMN` to the loop in `main.py` (~lines 22–47).

**Startup sequence** (`main.py`, top to bottom): `storage.ensure_and_migrate()` → `create_all` → column ALTERs → audio-duration backfill/probe → `seed()` (built-in domains/templates) → reset notes stuck in `transcribing`/`summarizing` from a crashed run → `configure_telemetry()` (flushed on shutdown).

## Tracing (optional, off by default)

Settings → Tracing sends OTel spans for every agent run — prompts, outputs, per-step latency, tokens — to a local [Arize Phoenix](https://docs.arize.com/phoenix):

```bash
uvx arize-phoenix serve    # UI at http://localhost:6006
```

Implemented in `telemetry.py` + `telemetry_config.py` with a **private** TracerProvider — disabled means truly no-op, no network chatter. `PUT /api/settings/telemetry` re-applies without a restart. `WorkflowRun.trace_id` links a run to its trace; a dead endpoint only warns.

## Eval harness (`eval/`)

Offline CLI that runs the DB-free `run_pipeline` against live LM Studio and scores the output (serial, `max_concurrency=1` — LM Studio constraint; exits nonzero on failures):

```bash
uv run --group eval python -m eval --list                    # see the cases
uv run --group eval python -m eval --domain General --judge  # hand-authored cases + LLM judge
uv run python -m eval.download_datasets                      # one-time public-dataset fetch (gitignored eval/data/)
uv run --group eval python -m eval --public all --limit 3    # MeetingBank + QMSum samples
uv run --group eval python -m eval --public all --baseline   # vanilla single-call comparison floor
uv run --group eval python -m eval --public all --trace --json-out report.json
```

- Hand-authored cases (`cases.py`): token-based fact coverage, action recall, hallucination flags (proper nouns absent from the transcript), pipeline confidence, optional `--judge` LLM grading.
- Public cases (`public_cases.py`): scored by `ReferenceAlignment` against each dataset's human reference summary — summary-only and whole-document content-token recall.
- `--baseline` swaps the pipeline for one vanilla LLM call (same model/settings/prompt) so score deltas measure what the workflow adds; `--trace` force-sends every eval LLM call to the local Phoenix, one trace per case; `--json-out` includes per-case scores, durations, and the full generated output for offline diffing.

Benchmark history, per-case diagnosis, and the workflow changes the eval drove: [docs/eval-agentic-vs-baseline.md](../docs/eval-agentic-vs-baseline.md).

## Testing

```bash
uv run pytest                             # run from backend/ — repo root collects asr-service tests and fails
uv run pytest tests/test_pipeline.py::test_name
```

- `tests/conftest.py` sets pydantic-ai's `ALLOW_MODEL_REQUESTS = False` — **no test can hit a real LLM**. Agent behavior is scripted with `TestModel`/`FunctionModel` via `Agent.override`.
- DB tests use in-memory SQLite; `test_orchestrator_e2e.py` covers the full run-workflow → rows → Summary path.
- Keep runs light and serial (no parallel pytest) — LM Studio typically holds a multi-GB model in RAM on the dev machine.
