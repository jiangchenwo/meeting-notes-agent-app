# Meeting Notes Agent V2 backend

This directory contains the standalone v2 backend. It has its own Python
package, lockfile, database migrations, tests, model runtime, workflow
components, and evaluation tools. It does not import code from the legacy
`backend/` package.

The backend is still under construction. The HTTP service currently exposes a
health check and instruction-preset CRUD routes. Evidence processing,
instruction planning, bounded dispatch, persistence, and model calls are
available as Python components and evaluation entry points, but there is no
end-to-end note-generation route yet.

## Quick start

Run commands from `backend-v2/`.

```bash
uv sync --all-groups
uv run alembic upgrade head
uv run uvicorn notes_agent_v2.app:create_app --factory --reload
```

The default migration URL is `sqlite:///notes-agent-v2.db`. The default app
factory does not create repositories, so the health route works while preset
routes return `503 persistence is not configured`. An embedding application can
construct `Database`, `Repositories`, and then call
`create_app(repositories=repositories)`.

Useful service URLs:

| URL | Purpose |
| --- | --- |
| `GET /api/v2/health` | Service health |
| `/docs` | Interactive OpenAPI documentation |
| `/openapi.json` | OpenAPI schema |

## HTTP API

The preset API stores reusable user instructions. It accepts instruction
content and descriptive metadata, but it does not accept model, tool, budget,
or routing controls.

| Method and path | Behavior |
| --- | --- |
| `POST /api/v2/presets` | Create a preset |
| `GET /api/v2/presets` | List active presets; use `?tag=value` to filter |
| `GET /api/v2/presets/{preset_id}` | Read one preset |
| `PATCH /api/v2/presets/{preset_id}` | Update supplied fields |
| `DELETE /api/v2/presets/{preset_id}` | Tombstone a preset |

Preset deletion is logical. Existing generation runs retain their instruction
snapshot even if the source preset is edited or tombstoned later.

## Code organization

| Path | Responsibility |
| --- | --- |
| `src/notes_agent_v2/domain/` | Strict, framework-independent contracts for transcripts, evidence, runs, documents, planning, and quality |
| `src/notes_agent_v2/persistence/` | SQLAlchemy models, migrations, immutable run records, snapshots, and repositories |
| `src/notes_agent_v2/runtime/` | Configuration, model identity checks, context accounting, request budgets, tool authorization, and provider adapters |
| `src/notes_agent_v2/workflow/` | Transcript preflight, evidence extraction and verification, consolidation, salience, planning, and bounded dispatch |
| `src/notes_agent_v2/evaluation/` | Feature specifications, metrics, judge support, safe traces, sealed bundles, qualification, and reporting |
| `scripts/` | Offline evaluations, live runtime checks, judge qualification, reports, and trace inspection |
| `config/` | Runtime profiles, model settings, judge settings, and the feature registry |

## Runtime configuration

`config/runtime.json` defines the provider, LM Studio endpoints, expected model
identity, timeouts, profile catalog, and context envelope. `config/profiles.json`
defines stage-specific generation and tool limits. The checked-in values are
development defaults, not production activation.

Configuration precedence is:

1. Process environment
2. The selected `.env` file
3. The selected JSON file

Copy `.env.example` to `.env` for local overrides. Select alternate files with
`NOTES_RUNTIME_CONFIG_FILE` and `NOTES_RUNTIME_ENV_FILE`. Relative profile paths
are resolved from the runtime JSON file.

Common overrides include:

| Variable | Setting |
| --- | --- |
| `NOTES_RUNTIME_INFERENCE_BASE_URL` | OpenAI-compatible inference URL ending in `/v1` |
| `NOTES_RUNTIME_CONTROL_BASE_URL` | LM Studio server URL used for native model inspection |
| `NOTES_RUNTIME_MODEL_KEY` | Expected loaded model key |
| `NOTES_RUNTIME_MODEL_INSTANCE_ID` | Optional exact loaded instance |
| `NOTES_RUNTIME_MODEL_LOADED_CONTEXT` | Expected loaded context size |
| `NOTES_RUNTIME_PROFILES_PATH` | Alternate profile catalog |
| `NOTES_RUNTIME_INFERENCE_TIMEOUT_SECONDS` | Model request timeout |
| `NOTES_RUNTIME_API_TOKEN` | Optional API token; environment only |

API tokens in JSON are rejected. The loader also rejects unsupported providers,
placeholder model identities, malformed profiles, and context envelopes whose
parts do not match the hard context limit.

The application runtime currently supports `lm_studio_openai`. Its inference
adapter uses the OpenAI-compatible chat-completions API, while model identity is
read from LM Studio's native model endpoint. The code never loads, unloads, or
reconfigures a model. Load the intended model in LM Studio before running a live
probe or evaluation.

The provider and gateway are separated by protocols. A future provider can
implement the same boundary without changing request budgets, context
accounting, tool policy, or safe call records.

## Runtime safeguards

Every model request goes through a bounded gateway. The gateway checks the
loaded model identity, resolves an application-owned profile, counts the prompt
with the loaded model tokenizer, reserves request and token budgets, and limits
tool use to an explicit session policy. It records fingerprints and accounting
metadata rather than prompt or response bodies.

Profiles remain marked `candidate`. A successful probe or evaluation produces
authorization for the exact model, profile, prompt, schema, and fixture
fingerprints that were tested. Changing one of those inputs invalidates the
authorization instead of silently carrying it forward.

## Evidence and instruction workflow

The evidence path normalizes a transcript, computes a token-aware chunk plan,
extracts atomic candidate facts with citations, verifies each candidate against
the source transcript, and consolidates supported facts without dropping
provenance. Evidence tools expose only allowlisted facts and project context for
the active run.

The instruction path turns a user instruction into a generation brief, ranks
supported facts for salience, creates a plan from a closed capability schema,
and dispatches a bounded set of blocks. Application code validates capability
names and all fact or project-context references. The model cannot select
providers, profiles, tools, retry counts, budgets, or executable operations.

Failures use typed statuses and error codes. Invalid model output does not widen
scope or fall through to unrestricted generation.

## Persistence

Apply migrations with:

```bash
uv run alembic upgrade head
```

Alembic reads `sqlalchemy.url` from `alembic.ini`. Applications that need a
different URL can call `upgrade_database(url)` or provide an adjusted Alembic
configuration. The `Database` helper enables SQLite foreign keys, WAL mode, and
a five-second busy timeout.

Repositories preserve source snapshots and append-only run history. Stage
artifacts and their completion events are committed in one transaction. Facts,
documents, critic issues, quality reports, and safe model-call records are
validated against their owning run before storage.

## Evaluation

Evaluation is part of each feature contract. `config/evaluation/features.json`
records the hypothesis, baseline and treatment, metric gates, trace
requirements, request and cost ceilings, seeds, and invalidation conditions.

Offline checks do not contact a model:

```bash
uv run python scripts/evaluate_contract_persistence.py \
  --feature domain.strict_contracts \
  --private-out /private/tmp/notes-agent-eval/strict-contracts

uv run python scripts/evaluate_evidence_memory.py \
  --feature evidence.cited_atomic_extraction \
  --out /private/tmp/notes-agent-eval/cited-extraction

uv run python scripts/evaluate_instruction_runtime.py \
  --feature all \
  --output /private/tmp/notes-agent-eval/instruction-runtime

uv run python scripts/evaluate_generation_quality.py \
  --feature generation.fact_covered_outline \
  --private-out /private/tmp/notes-agent-eval/fact-covered-outline
```

Each output directory must be new. Use a private location outside the repository
for live traces, model exchanges, authorizations, and evaluation bundles.

Live model and remote-judge commands require explicit opt-in flags. They also
require qualification artifacts whose fingerprints match the current runtime
and development set.

| Command | Purpose | Authorization or opt-in |
| --- | --- | --- |
| `scripts/probe_lm_studio.py` | Check the loaded LM Studio model and required capabilities | Uses a prior runtime authorization |
| `scripts/qualify_development_runtime.py` | Run the fixed runtime qualification schedule | `--allow-live` |
| `scripts/evaluate_evidence_effectiveness.py` | Compare evidence behavior on the qualified development set | `--allow-live` |
| `scripts/evaluate_instruction_effectiveness.py` | Measure brief, salience, and capability planning behavior | `--allow-live` |
| `scripts/qualify_remote_judge.py` | Calibrate an OpenAI-compatible judge against a fixed suite | `--allow-remote-judge` |
| `scripts/evaluate_source_verification_judge.py` | Judge source-verification quality | `--allow-remote-judge` |

Run remote-judge preflight before spending requests:

```bash
uv run python scripts/qualify_remote_judge.py \
  --suite tests/fixtures/evaluation/judge-calibration.json \
  --env-file .env \
  --preflight-only
```

The judge is disabled in `config/evaluation/judge.json` by default. Configure an
OpenAI-compatible endpoint through `NOTES_EVAL_JUDGE_*` variables. Tokens stay
in the environment. Cost rates, a total cost ceiling, request pacing, timeout,
and temperature are configuration values.

A completed live evaluation can pass or fail its quality gates. Connectivity,
authorization, or missing prerequisites are reported separately from completed
quality failures, so a failed score does not masquerade as a blocked run.

## Traces, bundles, and reports

Evaluation traces are JSON Lines files with bounded, redacted events. Validators
check event order, required terminal events, request counts, and prohibited
fields. Bundles are written atomically, sealed with file digests, and verified
before they are accepted.

Inspect a saved run with:

```bash
uv run python scripts/inspect_evaluation_trace.py \
  --run <private-bundle-directory> \
  --verify
```

Build a deterministic report from case-level results with:

```bash
uv run python scripts/report_evaluation.py \
  --feature <feature-id> \
  --results <results.json> \
  --output <report.json>
```

Keep the full bundle, not only the summary. The trace, case observations,
configuration fingerprints, accounting records, manifest, and report are needed
to reproduce or audit an important result.

## Tests

Run the offline suite:

```bash
uv run pytest -q
```

The `lm_studio` marker identifies live tests. They are skipped unless the local
runtime and its authorization are supplied explicitly. Run one only when LM
Studio is ready, using the runtime report generated by a successful probe:

```bash
NOTES_RUNTIME_REPORT=<runtime-authorization.json> \
  uv run pytest -m lm_studio -q
```

For a narrower check, pass a test module or node ID to `pytest`. Database tests
use temporary SQLite files and apply the real Alembic migration.

## Current boundaries

- Only the LM Studio OpenAI-compatible application runtime is implemented.
- The remote judge is a development evaluation dependency, not an application
  runtime provider.
- The default ASGI factory does not wire persistence.
- Presets are the only HTTP CRUD resource.
- Workflow components do not yet form a public end-to-end generation endpoint.
- Candidate profiles and evaluation results do not imply production activation.
