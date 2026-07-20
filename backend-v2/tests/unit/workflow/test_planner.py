import json
from types import SimpleNamespace

from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.planner import create_capability_plan
from notes_agent_v2.workflow.salience import SalienceRecord


class ScriptedGateway:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.requests = []

    def call(self, request, *, budget, validate):
        self.requests.append(request)
        content = json.dumps(self.payloads.pop(0))
        if not validate(content):
            raise ValueError("invalid scripted result")
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _brief(*, eligible=("overview", "narrative", "decisions", "actions")):
    return GenerationBrief(
        audience="general",
        desired_depth="standard",
        constraints=(),
        requested_emphasis=("overview",),
        forbidden_content=(),
        uncertainty=(),
        eligible_blocks=eligible,
    )


def _salience(
    fact_id: str,
    *,
    kind: str = "fact",
    mandatory: bool = False,
    verification: str = "supported",
):
    return SalienceRecord(
        fact_id=fact_id,
        kind=kind,
        status="asserted",
        verification=verification,
        instruction_relevance=0.5,
        meeting_importance=0.5,
        decision_action_weight=0,
        recency_correction_weight=0.4,
        confidence=0.8,
        score=0.5,
        mandatory=mandatory,
    )


def _block(identifier, capability, fact_ids=(), context_ids=(), constraints=()):
    return {
        "id": identifier,
        "capability": capability,
        "purpose": f"Create {capability}",
        "fact_ids": list(fact_ids),
        "project_context_ids": list(context_ids),
        "constraints": list(constraints),
    }


def _run(payloads, *, salience=None, context=(), brief=None):
    gateway = ScriptedGateway(payloads)
    result = create_capability_plan(
        run_id="run-1",
        instruction="Summarize the decisions and actions.",
        brief=brief or _brief(),
        salience=salience or (_salience("f000001"),),
        approved_project_context_ids=context,
        gateway=gateway,
        budget=RunBudget(max_model_requests=2),
    )
    return result, gateway


def test_uses_reasoned_analysis_then_structured_finalization() -> None:
    payload = {"blocks": [_block("b001", "overview", ("f000001",))]}
    result, gateway = _run([payload, payload])
    assert result.status == "ready"
    assert result.plan is not None
    assert result.plan.blocks[0].capability == "overview"
    assert [item.profile_name for item in gateway.requests] == [
        "planning_reasoned",
        "planning_structured_off",
    ]


def test_prompts_publish_exact_reference_allowlists() -> None:
    supported = _salience("f000001")
    uncertain = _salience("f000002", verification="uncertain")
    payload = {"blocks": [_block("b001", "overview", ("f000001",))]}
    result, gateway = _run(
        [payload, payload],
        salience=(supported, uncertain),
        context=("pc000001",),
    )
    assert result.status == "ready"
    for request in gateway.requests:
        body = json.loads(request.messages[1]["content"])
        authoritative = body.get("authoritative_input", body)
        assert authoritative["allowed_fact_ids"] == ["f000001"]
        assert authoritative["allowed_project_context_ids"] == ["pc000001"]
        assert "f000002" not in request.messages[0]["content"]
        assert "fact_ids must contain only allowed_fact_ids" in request.messages[0][
            "content"
        ]


def test_repairs_only_missing_mandatory_assignments_deterministically() -> None:
    salience = (
        _salience("f000001", kind="decision", mandatory=True),
        _salience("f000002", kind="action", mandatory=True),
    )
    payload = {
        "blocks": [
            _block("b001", "overview"),
            _block("b002", "decisions"),
            _block("b003", "actions", ("f000002",)),
        ]
    }
    result, _ = _run([payload, payload], salience=salience)
    assert result.status == "ready"
    assert result.plan is not None
    assert result.plan.blocks[1].fact_ids == ("f000001",)
    assert result.plan.blocks[2].fact_ids == ("f000002",)


def test_rejects_unknown_fact_and_unapproved_context_references() -> None:
    payload = {
        "blocks": [
            _block("b001", "overview", ("f999999",), ("pc999999",))
        ]
    }
    result, _ = _run([payload, payload], context=("pc000001",))
    assert result.status == "planning_failed"
    assert result.error_code == "unknown_fact_reference"


def test_rejects_uncertain_facts_and_empty_optional_blocks() -> None:
    uncertain = _salience("f000001", verification="uncertain")
    payload = {
        "blocks": [
            _block("b001", "overview"),
            _block("b002", "decisions", ("f000001",)),
        ]
    }
    result, _ = _run([payload, payload], salience=(uncertain,))
    assert result.status == "planning_failed"
    assert result.error_code == "uncertain_fact_reference"


def test_rejects_missing_narrative_capability_and_ineligible_capability() -> None:
    payload = {
        "blocks": [
            _block("b001", "overview"),
            _block("b002", "risks", ("f000001",)),
        ]
    }
    result, _ = _run([payload, payload], brief=_brief(eligible=("overview",)))
    assert result.status == "planning_failed"
    assert result.error_code == "ineligible_capability"


def test_custom_blocks_require_explicit_constraints() -> None:
    invalid = {"blocks": [_block("b001", "custom", ("f000001",))]}
    result, _ = _run(
        [invalid, invalid], brief=_brief(eligible=("overview", "custom"))
    )
    assert result.status == "planning_failed"


def test_invalid_analysis_fails_closed_without_finalization() -> None:
    result, gateway = _run([{"blocks": []}])
    assert result.status == "planning_failed"
    assert result.error_code == "invalid_capability_analysis"
    assert len(gateway.requests) == 1
