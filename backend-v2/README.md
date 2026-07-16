# Meeting Notes Agent V2 backend

This directory contains the standalone v2 backend. It has its own Python
package, dependency lockfile, tests, and `/api/v2` routes. It does not import
code from the legacy `backend/` package.

Install the locked environment and run the service:

```bash
uv sync --all-groups
uv run uvicorn notes_agent_v2.app:create_app --factory --reload
```

Run the offline test suite with:

```bash
uv run pytest -q
```

Tests marked `lm_studio` are live checks and are not part of the default
offline suite.

## Runtime configuration

Model identity, context allocation, endpoints, timeouts, and the profile file
are defined in `config/runtime.json`. The checked-in defaults target a local LM
Studio server, but these values are configuration rather than gateway
constants.

Environment variables override JSON values. The loader can also read a `.env`
file; copy `.env.example` to `.env` for local overrides. A process environment
variable takes precedence over the same value in `.env`.

Use `NOTES_RUNTIME_CONFIG_FILE` to select another JSON file or
`NOTES_RUNTIME_ENV_FILE` to select an alternate `.env` file. API tokens are
accepted only through `NOTES_RUNTIME_API_TOKEN` in the environment or `.env`;
tokens in JSON are rejected.

Only the `lm_studio_openai` provider is implemented today. The runtime gateway
uses provider and control protocols so another API can be added later without
changing its budgeting, normalization, tool authorization, or recording flow.
