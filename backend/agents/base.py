import json
import re
import httpx
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Structured output from a single agent step."""
    agent_name: str
    status: str          # "done" | "error" | "skipped"
    output: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_check: dict | None = None
    attempt: int = 1
    duration_ms: int = 0


@dataclass
class WorkflowContext:
    note_id: int
    transcript: str
    domain_name: str
    template_name: str
    template_prompt: str
    project_system_prompt: str
    project_knowledge_base: str
    cfg: dict
    results: dict[str, Any] = field(default_factory=dict)
    critique_results: dict[str, dict] = field(default_factory=dict)
    previous_attempt: str = ""


class LLMAgent:
    name: str = "LLMAgent"

    def _call_llm(self, system: str, user: str, cfg: dict) -> tuple[str, int, int]:
        base_url = cfg["base_url"].rstrip("/")
        model = cfg.get("model") or ""
        max_response_tokens = cfg.get("max_response_tokens", 1024)

        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_response_tokens,
        }
        if model:
            payload["model"] = model

        with httpx.Client(timeout=180) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload)
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", len(system + user) // 4)
        output_tokens = usage.get("completion_tokens", len(content) // 4)
        return content, input_tokens, output_tokens

    @staticmethod
    def _heal_json_escapes(s: str) -> str:
        """Fix invalid JSON escape sequences (e.g. LaTeX \psi, \sigma) by doubling the backslash."""
        return re.sub(r'\\([^"\\/bfnrtu\n\r\t])', r'\\\\\1', s)

    def _parse_json(self, content: str, fallback: dict) -> dict:
        def _try(s: str) -> dict | None:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(self._heal_json_escapes(s))
            except json.JSONDecodeError:
                return None

        result = _try(content)
        if result is not None:
            return result

        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
        if match:
            result = _try(match.group(1))
            if result is not None:
                return result

        match = re.search(r"\{[\s\S]+\}", content)
        if match:
            result = _try(match.group(0))
            if result is not None:
                return result

        return fallback

    def _truncate_transcript(self, transcript: str, cfg: dict) -> str:
        max_chars = max(500, (cfg.get("max_tokens", 4096) - 800) * 4)
        if len(transcript) > max_chars:
            return transcript[:max_chars] + "\n\n[Transcript truncated due to context limit]"
        return transcript

    def run(self, ctx: WorkflowContext) -> dict:
        raise NotImplementedError
