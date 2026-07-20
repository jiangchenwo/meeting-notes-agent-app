from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import httpx

from notes_agent_v2.runtime.adapter import normalize_response
from notes_agent_v2.runtime.contracts import (
    CapabilityProbe,
    ProbeStatus,
    RuntimeCapabilities,
)
from notes_agent_v2.runtime.lm_studio import ExpectedModel, LMStudioControlClient
from notes_agent_v2.runtime.probes import build_public_report, run_capability_probes
from notes_agent_v2.runtime.settings import load_runtime_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = Path("/private/tmp/notes-agent-v2-runtime-probe.trace.json")


def validate_runtime_authorization(path: Path, expected_model: ExpectedModel) -> None:
    payload = json.loads(path.read_text())
    model = payload.get("model") or payload.get("identity")
    if not isinstance(model, dict):
        raise ValueError("runtime authorization has no model identity")
    expected = {
        "model_key": expected_model.model_key,
        "architecture": expected_model.architecture,
        "format": expected_model.format,
        "bits_per_weight": expected_model.bits_per_weight,
        "loaded_context": expected_model.loaded_context,
    }
    mismatches = [key for key, value in expected.items() if model.get(key) != value]
    if mismatches:
        raise ValueError("runtime authorization drift: " + ", ".join(sorted(mismatches)))


def require_private_trace_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("private trace must be written outside the repository")
    return resolved


class LiveProbeRunner:
    def __init__(
        self,
        *,
        base_url: str,
        model_key: str,
        api_token: str | None,
        timeout_seconds: float,
        loaded_context: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_key = model_key
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self.timeout_seconds = timeout_seconds
        self.loaded_context = loaded_context
        self.trace: list[dict[str, Any]] = []

    def run_probe(self, name: str) -> CapabilityProbe:
        started = time.monotonic()
        trace_id = f"probe-{uuid4().hex}"
        try:
            observed = getattr(self, f"_{name}")()
            status = ProbeStatus.passed
            error_code = None
        except Exception as exc:
            observed = {}
            status = ProbeStatus.failed
            error_code = type(exc).__name__
            self.trace.append(
                {"name": name, "trace_id": trace_id, "error": str(exc)}
            )
        return CapabilityProbe(
            name=name,
            status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            observed=observed,
            error_code=error_code,
            trace_id=trace_id,
        )

    def _chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {"model": self.model_key, **payload}
        with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds)) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=request_payload,
            )
            response.raise_for_status()
            data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage") or {}
        self.trace.append(
            {
                "finish_reason": choice.get("finish_reason"),
                "model": data.get("model", self.model_key),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "reasoning_observed": bool(message.get("reasoning_content")),
                "tool_call_count": len(message.get("tool_calls") or []),
            }
        )
        return message

    def _system(self) -> dict[str, Any]:
        sentinel = f"SYSTEM-{uuid4().hex[:8]}"
        message = self._chat(
            {
                "messages": [
                    {"role": "system", "content": f"Reply with exactly {sentinel}."},
                    {"role": "user", "content": "Confirm."},
                ],
                "reasoning_effort": "none",
                "temperature": 0,
                "max_tokens": 64,
            }
        )
        if (message.get("content") or "").strip() != sentinel:
            raise ValueError("system instruction mismatch")
        return {"matched": True}

    def _reasoning(self) -> dict[str, Any]:
        sentinel = f"FINAL-{uuid4().hex[:8]}"
        message = self._chat(
            {
                "messages": [{"role": "user", "content": f"Reply with exactly {sentinel}."}],
                "temperature": 0,
                "max_tokens": 512,
            }
        )
        normalized = normalize_response({"message": message})
        if sentinel not in normalized.final_content:
            raise ValueError("reasoning probe has no final answer")
        return {"reasoning_observed": normalized.reasoning_observed}

    def _schema(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        }
        message = self._chat(
            {
                "messages": [{"role": "user", "content": "Return the required object."}],
                "reasoning_effort": "none",
                "temperature": 0,
                "max_tokens": 128,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "probe", "strict": True, "schema": schema},
                },
            }
        )
        if json.loads(message.get("content") or "{}").get("status") != "ok":
            raise ValueError("schema response is invalid")
        return {"valid": True}

    def _tool_request(self) -> dict[str, Any]:
        message = self._chat(
            {
                "messages": [{"role": "user", "content": "Call lookup_probe with key alpha."}],
                "tools": [_tool_schema()],
                "max_tokens": 128,
            }
        )
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "lookup_probe":
            raise ValueError("tool request was not native")
        return {"tool_call_count": 1}

    def _tool_round_trip(self) -> dict[str, Any]:
        first = self._chat(
            {
                "messages": [{"role": "user", "content": "Call lookup_probe with key alpha."}],
                "tools": [_tool_schema()],
                "max_tokens": 128,
            }
        )
        calls = first.get("tool_calls") or []
        if not calls:
            raise ValueError("tool call missing")
        nonce = f"VALUE-{uuid4().hex[:8]}"
        final = self._chat(
            {
                "messages": [
                    {"role": "user", "content": "Call lookup_probe with key alpha."},
                    first,
                    {"role": "tool", "tool_call_id": calls[0]["id"], "content": nonce},
                ],
                "tools": [_tool_schema()],
                "max_tokens": 128,
            }
        )
        if nonce not in (final.get("content") or ""):
            raise ValueError("tool result was not used")
        return {"round_trip": True}

    def _tool_rejection(self) -> dict[str, Any]:
        message = self._chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Do not call a tool. Reply with exactly REJECTED.",
                    }
                ],
                "tools": [_tool_schema()],
                "tool_choice": "none",
                "max_tokens": 64,
            }
        )
        if message.get("tool_calls"):
            raise ValueError("tool choice none was ignored")
        return {"rejected": True}

    def _reasoning_replay(self) -> dict[str, Any]:
        message = self._chat(
            {
                "messages": [{"role": "user", "content": "Reply with exactly REPLAY-OK."}],
                "max_tokens": 512,
            }
        )
        normalized = normalize_response({"message": message})
        if "REPLAY-OK" not in normalized.final_content:
            raise ValueError("safe replay failed")
        return {"safe_final": True}

    def _context(self) -> dict[str, Any]:
        message = self._chat(
            {
                "messages": [{"role": "user", "content": "Reply with exactly CONTEXT-OK."}],
                "reasoning_effort": "none",
                "max_tokens": 64,
            }
        )
        if "CONTEXT-OK" not in (message.get("content") or ""):
            raise ValueError("context probe failed")
        return {"loaded_context": self.loaded_context}


def _tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_probe",
            "description": "Return a probe value.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the approved local LM Studio runtime.")
    parser.add_argument("--runtime-report", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--authorization-out", type=Path)
    parser.add_argument("--trace-out", type=Path, default=DEFAULT_TRACE)
    args = parser.parse_args()

    settings = load_runtime_settings(args.config)
    validate_runtime_authorization(args.runtime_report, settings.model)
    trace_out = require_private_trace_path(args.trace_out)
    control = LMStudioControlClient(
        str(settings.control_base_url).rstrip("/"),
        api_token=settings.api_token,
        timeout_seconds=settings.control_timeout_seconds,
    )
    identity = control.resolve_instance(settings.model)
    capabilities = RuntimeCapabilities(
        system_prompt=True,
        reasoning=True,
        tool_request=True,
        tool_round_trip=True,
        native_schema=True,
        exact_tokenizer=True,
    )
    runner = LiveProbeRunner(
        base_url=str(settings.inference_base_url),
        model_key=identity.model_key,
        api_token=settings.api_token,
        timeout_seconds=settings.inference_timeout_seconds,
        loaded_context=settings.model.loaded_context,
    )
    report = run_capability_probes(identity, capabilities, runner)
    public = build_public_report(report)

    trace_out.parent.mkdir(parents=True, exist_ok=True)
    trace_out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runtime_fingerprint": report.fingerprint,
                "exchanges": runner.trace,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    rendered = json.dumps(public, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    else:
        print(rendered, end="")
    if args.authorization_out:
        authorization_out = require_private_trace_path(args.authorization_out)
        authorization_out.parent.mkdir(parents=True, exist_ok=True)
        authorization_out.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
    return 0 if report.readiness.value == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
