import json
from types import SimpleNamespace

from notes_agent_v2.domain.evidence import EvidenceSpan, Fact
from notes_agent_v2.runtime.budget import RunBudget
from notes_agent_v2.workflow.audience import GenerationBrief
from notes_agent_v2.workflow.salience import rank_salience


class RelevanceGateway:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.requests = []

    def call(self, request, *, budget, validate):
        self.requests.append(request)
        content = json.dumps(
            {
                "items": [
                    {"fact_id": fact_id, "instruction_relevance": score}
                    for fact_id, score in self.scores.items()
                ]
            }
        )
        assert validate(content)
        return SimpleNamespace(response=SimpleNamespace(final_content=content))


def _fact(
    identifier: str,
    text: str,
    *,
    kind: str = "fact",
    status: str = "asserted",
    confidence: float = 0.8,
    verification: str = "supported",
    source_count: int = 1,
    utterance: int = 1,
) -> Fact:
    return Fact(
        id=identifier,
        text=text,
        kind=kind,
        status=status,
        speaker_ids=(),
        owner=None,
        due_text=None,
        confidence=confidence,
        verification=verification,
        evidence=(
            EvidenceSpan(
                utterance_ids=(f"u{utterance:06d}",), quote=text
            ),
        ),
        source_candidate_ids=tuple(
            f"fc{index:06d}" for index in range(1, source_count + 1)
        ),
        supersedes_fact_ids=(),
        conflicts_with_fact_ids=(),
    )


def _brief(*, forbidden: tuple[str, ...] = ()) -> GenerationBrief:
    return GenerationBrief(
        audience="general",
        desired_depth="standard",
        constraints=(),
        requested_emphasis=("overview", "narrative"),
        forbidden_content=forbidden,
        uncertainty=(),
        eligible_blocks=("overview", "narrative"),
    )


def test_verified_decisions_actions_and_corrections_are_mandatory() -> None:
    facts = (
        _fact("f000001", "A decision.", kind="decision", status="approved"),
        _fact("f000002", "An action.", kind="action", status="proposed"),
        _fact("f000003", "A correction.", kind="correction", status="rejected"),
    )

    ranked = rank_salience(
        run_id="run-1",
        instruction="Summarize.",
        brief=_brief(),
        facts=facts,
        gateway=RelevanceGateway({item.id: 0.1 for item in facts}),
        budget=RunBudget(max_model_requests=1),
    )

    assert {item.fact_id for item in ranked if item.mandatory} == {
        "f000001",
        "f000002",
        "f000003",
    }
    assert [item.status for item in ranked] == ["rejected", "approved", "proposed"]


def test_instruction_relevance_improves_ranking_and_repetition_cannot_dominate() -> None:
    repeated = _fact(
        "f000001", "Repeated background.", source_count=12, confidence=1, utterance=1
    )
    relevant = _fact(
        "f000002", "Security risk blocks launch.", kind="risk", utterance=2
    )

    ranked = rank_salience(
        run_id="run-1",
        instruction="Focus on launch risks.",
        brief=_brief(),
        facts=(repeated, relevant),
        gateway=RelevanceGateway({"f000001": 0.0, "f000002": 1.0}),
        budget=RunBudget(max_model_requests=1),
    )

    assert [item.fact_id for item in ranked] == ["f000002", "f000001"]
    assert ranked[0].instruction_relevance == 1
    assert ranked[1].meeting_importance <= 1


def test_uncertain_facts_are_never_mandatory() -> None:
    uncertain = _fact(
        "f000001",
        "Maybe approved.",
        kind="decision",
        status="uncertain",
        verification="uncertain",
        confidence=1,
    )
    ranked = rank_salience(
        run_id="run-1",
        instruction="Include this decision.",
        brief=_brief(),
        facts=(uncertain,),
        gateway=RelevanceGateway({"f000001": 1.0}),
        budget=RunBudget(max_model_requests=1),
    )
    assert ranked[0].mandatory is False


def test_explicit_category_exclusion_changes_selection_not_truth() -> None:
    decision = _fact(
        "f000001", "The launch was rejected.", kind="decision", status="rejected"
    )
    ranked = rank_salience(
        run_id="run-1",
        instruction="Exclude decisions.",
        brief=_brief(forbidden=("decisions",)),
        facts=(decision,),
        gateway=RelevanceGateway({"f000001": 1.0}),
        budget=RunBudget(max_model_requests=1),
    )
    assert ranked[0].mandatory is False
    assert ranked[0].kind == "decision"
    assert ranked[0].status == "rejected"


def test_score_formula_components_are_normalized_and_exact() -> None:
    fact = _fact(
        "f000001", "Approved launch.", kind="decision", status="approved", confidence=0.8
    )
    item = rank_salience(
        run_id="run-1",
        instruction="Summarize launch.",
        brief=_brief(),
        facts=(fact,),
        gateway=RelevanceGateway({"f000001": 0.9}),
        budget=RunBudget(max_model_requests=1),
    )[0]
    assert all(
        0 <= value <= 1
        for value in (
            item.instruction_relevance,
            item.meeting_importance,
            item.decision_action_weight,
            item.recency_correction_weight,
            item.confidence,
            item.score,
        )
    )
    assert item.score == round(
        0.35 * item.instruction_relevance
        + 0.25 * item.meeting_importance
        + 0.20 * item.decision_action_weight
        + 0.10 * item.recency_correction_weight
        + 0.10 * item.confidence,
        6,
    )


def test_identical_inputs_produce_identical_ordering() -> None:
    facts = (
        _fact("f000002", "Second.", utterance=2),
        _fact("f000001", "First.", utterance=1),
    )
    args = {
        "run_id": "run-1",
        "instruction": "Summarize.",
        "brief": _brief(),
        "facts": facts,
        "budget": RunBudget(max_model_requests=1),
    }
    left = rank_salience(
        **args, gateway=RelevanceGateway({"f000001": 0.5, "f000002": 0.5})
    )
    args["budget"] = RunBudget(max_model_requests=1)
    right = rank_salience(
        **args, gateway=RelevanceGateway({"f000001": 0.5, "f000002": 0.5})
    )
    assert left == right
    assert [item.fact_id for item in left] == ["f000001", "f000002"]
