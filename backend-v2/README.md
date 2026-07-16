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
