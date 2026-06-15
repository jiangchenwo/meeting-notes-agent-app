# Meeting Notes Agent

A local-first, offline-capable web app that transcribes audio recordings and generates structured meeting notes using open-source, self-hosted AI. No cloud dependencies — all audio and data stays on your machine.

## What it does

```
Upload Audio → Select Project + Domain + Template → Transcribe → Summarize → Export
```

1. Drop in an audio file (MP3, WAV, M4A up to 2 GB)
2. Assign it to a **project** and pick a **domain** (General, Education, Healthcare, Interview, Project)
3. [whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcribes it locally
4. A local LLM generates a structured summary, action items, and domain-specific suggestions
5. Review, edit, and export the notes as Markdown or plain text

## Features

**Now**

- Local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) transcription — timestamped and editable
- AI note generation — summary, action items, and domain-specific suggestions
- Agentic multi-agent pipeline (orchestrator → agents → critic → verifiers) via API + Testing Lab
- Projects with shared system prompt & knowledge base; custom domains & templates
- Note management — full-text search, filters, color labels, drag-to-reorder, bulk ops
- Testing Lab — benchmark the pipeline against eval datasets with quality metrics
- Markdown / plain-text export; LM Studio + whisper configuration

**Upcoming**

- Wire the agentic workflow into the note page (today it runs via API + the Testing Lab)
- RAG / semantic search across all notes
- Real-time microphone transcription
- Calendar, webhook (Notion / Obsidian), and folder-watch integrations
- Multi-user / auth

## Agentic workflow

Instead of a single "summarize everything" LLM call, the app runs a sequence of focused agents — each with one job and a tight prompt. A deterministic orchestrator ([`agents/orchestrator.py`](backend/agents/orchestrator.py)) selects the agents based on domain, runs them serially (LM Studio loads one model at a time), then a critic agent reviews selected steps and retries any that fall below a quality threshold.

### Architecture

```
Orchestrator (rule-based, no extra LLM call)
    │
    ├── Chunking phase  (only when the transcript exceeds the context window)
    │   └── Map-reduce Summarizer over overlapping chunks → condensed transcript
    │
    ├── Extraction phase  (serial — one model at a time)
    │   ├── Summarizer          — narrative summary in Markdown
    │   ├── ActionItemExtractor — [{task, owner, deadline, priority}]
    │   ├── DecisionLogger      — [{decision, rationale, made_by}]
    │   └── Domain agent        — domain-specific structured output
    │
    ├── Critique phase
    │   └── Critic              — scores output 1–10, flags gaps, rewrites if needed
    │       └── Retry loop      — re-runs the step with critique notes until threshold met
    │
    ├── Verification phase  (non-LLM)
    │   ├── SchemaVerifier      — checks each agent's output shape
    │   └── RiskClassifier      — flags risky/sensitive content by domain
    │
    └── Assembly  (pure Python, no LLM call)
        └── Writes Summary record + raw_sections_json (incl. schema/risk results)
```

### Domain routing

Each domain activates a different set of agents and critique settings ([`agents/workflows.py`](backend/agents/workflows.py)). General is the baseline fallback for any unmatched domain:


| Domain         | Agents                                          | Critiqued step | Threshold |
| ---------------- | ------------------------------------------------- | ---------------- | ----------- |
| **General**    | Summarizer, ActionItemExtractor                 | Summarizer     | 8/10      |
| **Education**  | Summarizer, LectureAgent, ActionItemExtractor   | Summarizer     | 8/10      |
| **Healthcare** | Summarizer, ActionItemExtractor                 | Summarizer     | 8/10      |
| **Interview**  | Summarizer, InterviewAgent                      | InterviewAgent | 8/10      |
| **Project**    | Summarizer, ActionItemExtractor, DecisionLogger | Summarizer     | 8/10      |

### Domain-specific outputs

Domain agents extract structured data surfaced in the **Suggestions** tab:

- **Project** — explicit decisions with rationale, alongside the action items
- **Education** — key concepts with definitions, learning objectives, quiz questions
- **Interview** — red flags, green flags, candidate highlights, suggested follow-up questions

### Template overrides

A template can override the workflow for any note it's assigned to via the `workflow_config` JSON field:

```json
{
  "steps": ["Summarizer", "ActionItemExtractor", "DecisionLogger"],
  "critique_steps": ["Summarizer"],
  "critique_threshold": 8,
  "max_retries": 2
}
```

## Getting started

### Prerequisites

- Node.js 20+
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for Python package management
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) compiled binary
- [LM Studio](https://lmstudio.ai) with a model loaded

### Backend

```bash
cd backend
uv sync
uv run uvicorn main:app --reload   # http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

`notes.db` and `uploads/` are created automatically at the repo root on first run.

Optional overrides in `backend/.env`:

```env
WHISPER_BINARY_PATH=/usr/local/bin/whisper-cpp
WHISPER_MODEL=base
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=your-model-name
```

## How to use

### 1. Upload a recording

Go to the **Home** page and drag-and-drop an audio file (MP3, WAV, M4A) onto the upload zone, or click to browse. The file appears as a note block with status **Pending**.

### 2. Assign context

Click the note block's name to open it. In the right panel, assign:

- **Project** — groups related recordings and injects shared context into the LLM
- **Domain** — selects the domain workflow (General, Education, Healthcare, Interview, Project)
- **Template** — selects which prompt drives the summary output

### 3. Transcribe

Click **Transcribe**. The status changes to **Transcribing** while whisper.cpp processes the audio locally. When done it switches to **Transcribed** and the timestamped transcript appears in the Segments and Full Text tabs.

You can edit the transcript directly on the Full Text tab before summarizing.

### 4. Generate notes

Click **Summarize** (it becomes **Re-summarize** once a summary exists). The status changes to **Summarizing** while the LLM runs; the page polls and switches to the **Summary** tab when it finishes. Use **Preview Prompt** to inspect the exact system and user messages first.

Results appear across tabs, each editable inline:

- **Summary** — structured narrative in Markdown
- **Action Items** — checklist with owner, deadline, and priority
- **Suggestions** — domain-specific output (decisions, concepts, interview flags, etc.)

To run the full multi-agent pipeline — orchestrator, critic retries, and schema/risk verification — and compare it against the single-call path with quality metrics, open the **Testing Lab** (`/lab`).

### 5. Export

Use the **Export** button on any note to download as Markdown or plain text. Individual copy buttons on each tab let you paste directly into Notion, Obsidian, or any other tool.

### Managing projects

Open **Projects** to create and organize projects. Inside a project:

- **Overview** — see all recordings; click the pencil icon to rename or update the description
- **System Prompt** — write a persona instruction that overrides the default assistant for this project (e.g. "Focus on engineering decisions and ticket references")
- **Knowledge Base** — add structured Markdown context (team members, glossary, recurring topics) that the LLM injects automatically when generating notes

### Configuring whisper and the LLM

Open **Settings** to set the whisper.cpp binary path, model size, and LM Studio endpoint. Use **Test Connection** to verify LM Studio is reachable before running a workflow.

---

## Stack


| Layer         | Choice                                       |
| --------------- | ---------------------------------------------- |
| Frontend      | React + TypeScript + Vite                    |
| Styling       | Tailwind CSS v3 (Material You green palette) |
| Routing       | React Router v6                              |
| Backend       | Python + FastAPI                             |
| Database      | SQLite                                       |
| Transcription | whisper.cpp (local binary)                   |
| LLM           | LM Studio (OpenAI-compatible local endpoint) |

## Design principles

- **Offline-first** — everything runs locally; no audio or data leaves the machine
- **Block-based UI** — each audio file is an independent unit with its own status
- **One agent, one job** — each LLM call has a single focused task and a tight prompt
- **Prompt transparency** — users can inspect the exact prompt sent to the LLM
- **LLM-agnostic** — uses the OpenAI-compatible endpoint; swap any local model
- **Backwards-compatible** — the legacy single-call summarizer still works alongside the workflow

## License

[Apache 2.0](LICENSE)
