# Meeting Notes Agent

A local-first web app that transcribes audio recordings and turns them into structured meeting notes with self-hosted AI. Everything — audio, transcripts, notes, and the models — stays on your machine.

## What it does

```
Upload Audio → Select Project + Domain + Template → Transcribe → Summarize → Export
```

1. Drop in an audio file (MP3, WAV, M4A up to 2 GB)
2. Assign it to a **project** and pick a **domain** (General, Education, Healthcare, Interview, Project)
3. The local ASR service transcribes it, with optional speaker diarization
4. A local LLM runs a multi-agent workflow that writes a summary, action items, and domain-specific suggestions
5. Review, edit, and export the notes as Markdown or plain text

## Features

**AI agents write your notes** — each transcript runs a pipeline of focused agents on your local LLM:

- **Multi-agent pipeline** — a Summarizer, extractors for action items and decisions, and domain agents for interviews and lectures
- **Self-checking** — a critic agent grades each draft and retries weak steps, keeping the best result
- **Live progress** — watch every agent's status, score, and duration
- **Customizable** — set the agent steps, critique targets, and prompts per template
- **Project context** — a shared system prompt and knowledge base feed every agent

**Also:** local transcription with speaker diarization · note search, labels, and bulk actions · Markdown / text export · all config in-app (no `.env`) · optional [Phoenix](https://docs.arize.com/phoenix) tracing

**Roadmap:** RAG / semantic search across notes · real-time microphone transcription · calendar, webhook (Notion / Obsidian), and folder-watch integrations · multi-user & auth

## Getting started

The app runs in **Docker** (frontend + backend via `docker compose`). Two things stay on the host: **LM Studio** and the **ASR service** (it needs direct GPU access).

### Prerequisites

- Docker (Desktop on Mac/Windows, or Engine + the compose plugin on Linux)
- [LM Studio](https://lmstudio.ai) running with a model loaded (default port `1234`)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) — for the host-native ASR service

### 1. Start the ASR service (host-native)

```bash
cd asr-service
uv sync
./start.sh    # serves on http://localhost:9000
```

See [`asr-service/README.md`](asr-service/README.md) for model caching and details.

### 2. Start the app

```bash
make up      # builds both images and starts the app at http://localhost:8080
make logs    # tail logs
make down    # stop
```

(These are thin wrappers around `docker compose` — `docker compose up -d --build` works too.)

### 3. Verify connections

Open http://localhost:8080 → **Settings**. The defaults already point at your host machine for both LM Studio (`:1234`) and the ASR service (`:9000`) — click **Test connection** on each. All connection settings (base URLs, model, token limits, system prompt) live on this page.

All app data — the database, uploaded audio, and settings — persists on a single Docker volume and survives container recreation; data from an earlier version is migrated automatically on startup.

### Deploying to another machine

Publish multi-arch images to a registry, then pull them on the target host (the ASR service must still be set up host-native there):

```bash
make buildx REGISTRY=ghcr.io/you TAG=1.0    # build + push (linux/amd64 + arm64)
make pull up REGISTRY=ghcr.io/you TAG=1.0   # on the target host
```

### Development (without Docker)

For hot reload, run the backend and frontend directly (Node.js 20+ required):

```bash
cd backend && uv sync
uv run uvicorn main:app --reload   # http://localhost:8000

cd frontend && npm install
npm run dev                        # http://localhost:5173, proxies /api → :8000
```

In this mode, point Settings → ASR / LM Studio at `http://localhost:9000` and `http://localhost:1234/v1`. The module map, data model, API surface, pipeline internals, and eval harness are documented in [backend/README.md](backend/README.md).

## How to use

### 1. Upload a recording

On the **Home** page, drag-and-drop an audio file (MP3, WAV, M4A) onto the upload zone, or click to browse. The file appears as a note block with status **Pending**.

### 2. Assign context

Click the note block's name to open it. In the right panel, assign:

- **Project** — groups related recordings and injects shared context into the LLM
- **Domain** — selects which agents run (General, Education, Healthcare, Interview, Project)
- **Template** — selects the prompt that drives the summary output

### 3. Transcribe

Click **Transcribe**. When it finishes, the timestamped transcript appears in the Segments and Full Text tabs — speakers can be renamed, and the Full Text tab is editable before summarizing.

### 4. Generate notes

Click **Summarize** (it becomes **Re-summarize** once a summary exists). The Summary tab shows live per-agent progress — each step with its status, retries, quality score, and duration — then switches to the finished notes. Use **Preview Prompt** to inspect the exact prompt before running.

Results appear across tabs, each editable inline:

- **Summary** — structured narrative in Markdown
- **Action Items** — checklist with owner, deadline, and priority
- **Suggestions** — domain-specific output (see table below)

### 5. Export

Use the **Export** button on any note to download it as Markdown or plain text, or the copy button on each tab to paste into Notion, Obsidian, or any other tool.

### Managing projects

Open **Projects** to create and organize projects. Inside a project:

- **Overview** — see all recordings; rename or update the description
- **System Prompt** — a persona instruction applied to every note in this project (e.g. "Focus on engineering decisions and ticket references")
- **Knowledge Base** — Markdown context (team members, glossary, recurring topics) injected automatically when generating notes

### Settings

Everything is configured in-app and persists across restarts: the ASR service URL, the LM Studio endpoint (plus model, token limits, system prompt, structured-output mode), and the optional tracing toggle. **Test connection** buttons verify each service is reachable.

## How it works

Instead of one "summarize everything" prompt, each note runs a short **pipeline of focused agents** built on [Pydantic AI](https://ai.pydantic.dev). Every agent has a single job and a narrow prompt, a critic scores the important outputs against a rubric, and any step that falls short is retried with the critique as feedback. Everything runs serially against your local LM Studio endpoint — one model, one call at a time.

### The agents

| Agent | Job | Output |
| --- | --- | --- |
| **Summarizer** | The narrative summary — one section per topic, keeping the names, numbers, decisions, and outcomes | Markdown text |
| **ActionItemExtractor** | Concrete tasks, each with an owner, deadline, and priority | Structured list |
| **DecisionLogger** | Explicit decisions and the rationale behind them | Structured list |
| **InterviewAgent** | Candidate highlights, red/green flags, suggested follow-up questions | Structured |
| **LectureAgent** | Key concepts, learning objectives, assignments, quiz questions | Structured |
| **Critic** | Scores another agent's output 0–10 against a rubric and returns specific revision advice — it grades, it never rewrites | Score + issues |

The Summarizer always emits plain Markdown; the extractor and analysis agents return schema-constrained JSON, so their fields map straight to the UI tabs.

### The pipeline

```
Transcript
  │
  ├─ Chunk        only if it exceeds the model's context window:
  │               summarize each chunk, then condense into one transcript
  │
  ├─ Extract      serial run of the domain's agents:
  │               Summarizer → ActionItemExtractor → DecisionLogger / domain agent
  │
  ├─ Critique     the Critic scores selected steps; a step below the threshold
  │  + retry      is re-run with the critique in its prompt (up to max_retries).
  │               The best-scoring attempt is kept — never a worse retry.
  │
  ├─ Verify       no LLM: schema-shape checks + domain risk flags
  │               (e.g. a "needs review" flag on medical/clinical content)
  │
  └─ Assemble     pure Python: Summary · Action Items · Suggestions
```

The critic scores four dimensions — coverage (0–4), accuracy (0–3), specificity (0–2), and structure (0–1) — and the total is recomputed in Python from those parts, so the model can't inflate its own grade. Below the domain's threshold the step retries with the specific misses named as feedback; a long meeting that comes back with a too-short summary also gets an explicit "expand this" note.

### What each domain extracts

The note's domain decides which agents run alongside the Summarizer, and which outputs get a critique pass:

| Domain         | Beyond the summary                                                           | Quality-checked                |
| -------------- | ---------------------------------------------------------------------------- | ------------------------------ |
| **General**    | Action items, decision log                                                   | Summary                        |
| **Education**  | Key concepts, learning objectives, assignments, quiz questions, action items | Summary + lecture extraction   |
| **Healthcare** | Action items                                                                 | Summary + action items         |
| **Interview**  | Candidate highlights, red/green flags, suggested follow-up questions         | Summary + candidate assessment |
| **Project**    | Action items, decision log with rationale                                    | Summary                        |

### Customizing the workflow

A template can override the agent workflow for any note it's assigned to — edit **Advanced: Agent Workflow** in the template editor:

```json
{
  "steps": ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
  "critique_steps": ["Summarizer"],
  "critique_threshold": 8,
  "max_retries": 2
}
```

The four fields:

- **`steps`** — the agents to run, in order (1–8). A step can be a bare name or `{"agent": "Summarizer", "prompt_override": "…"}` to swap the prompt for that step only.
- **`critique_steps`** — which of those steps the Critic reviews. Must be a subset of `steps`.
- **`critique_threshold`** — the 0–10 score a step must reach; below it, the step retries with feedback.
- **`max_retries`** — how many extra attempts a below-threshold step gets (0–3).

Invalid configs are rejected on save; anything omitted falls back to the domain default.

### Tracing (optional)

Every AI run can emit traces — prompts, outputs, per-step latency and token usage — to a local Arize Phoenix instance. Everything stays on your machine.

```bash
uvx arize-phoenix serve   # Phoenix UI at http://localhost:6006
```

Then enable **Settings → Tracing (Phoenix)** in the app (no restart needed). A "capture content" toggle omits prompt/response text from traces for sensitive recordings. When the app runs in Docker, set the endpoint to `http://host.docker.internal:6006` so the container reaches Phoenix on your host.

## Repository layout

| Path                         | What it is                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [backend/](backend/)         | FastAPI API, agentic pipeline, eval harness —**[backend/README.md](backend/README.md)** is the full reference |
| [frontend/](frontend/)       | React + TypeScript + Vite web app                                                                              |
| [asr-service/](asr-service/) | Host-native transcription + diarization service (own README)                                                   |
| [docs/](docs/)               | Benchmark & methodology notes ([eval-agentic-vs-baseline.md](docs/eval-agentic-vs-baseline.md))                |

## Stack

| Layer         | Choice                                                                |
| ------------- | ---------------------------------------------------------------------- |
| Frontend      | React + TypeScript + Vite, Tailwind CSS                               |
| Backend       | Python + FastAPI, SQLite                                              |
| Transcription | Host-native ASR service — MLX-Whisper (Metal) + pyannote diarization |
| LLM           | LM Studio (or any OpenAI-compatible local endpoint)                   |

## License

[Apache 2.0](LICENSE)
