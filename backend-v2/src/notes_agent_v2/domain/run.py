from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    preflight = "preflight"
    extract = "extract"
    verify = "verify"
    consolidate = "consolidate"
    audience = "audience"
    salience = "salience"
    plan = "plan"
    outline = "outline"
    write = "write"
    assemble = "assemble"
    critic = "critic"
    accept = "accept"
    revise_1 = "revise_1"
    revise_2 = "revise_2"
    finalize = "finalize"


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    budget_exhausted = "budget_exhausted"
    review_required = "review_required"
    rejected = "rejected"


_TRANSITIONS = {
    RunStatus.queued: {RunStatus.running, RunStatus.cancelled},
    RunStatus.running: {
        RunStatus.completed,
        RunStatus.failed,
        RunStatus.cancelled,
        RunStatus.budget_exhausted,
        RunStatus.review_required,
        RunStatus.rejected,
    },
}


def validate_run_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid run transition: {current.value} -> {target.value}")
