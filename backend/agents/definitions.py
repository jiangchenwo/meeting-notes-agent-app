"""Pydantic AI agent definitions and prompt composition.

Agents are defined once at module level with no model attached — the model is
supplied per-run from lm_config (see llm.build_model), so Settings changes
apply without a restart.

Instruction text is built by plain functions (INSTRUCTION_BUILDERS) that the
@agent.instructions decorators delegate to; the pipeline and the
prompt-preview endpoint reuse the same functions so the persisted `_prompt`
matches what was actually sent.
"""
from pydantic_ai import Agent, RunContext

from .context import NoteDeps
from .tools import search_knowledge_base

# How much of the previous draft a critique-retry sees. Too small and the
# reviser loses content on long meetings (revisions came back shorter and more
# generic); the full draft would double the context on small local models.
REVISION_DRAFT_CHARS = 4000

# ---------------------------------------------------------------------------
# Instruction (system prompt) builders
# ---------------------------------------------------------------------------

def summarizer_instructions(deps: NoteDeps) -> str:
    parts = []
    if deps.global_system_prompt:
        parts.append(deps.global_system_prompt.strip())
    core = (
        "You are a focused meeting summarizer. Your only job: write a clear, thorough summary.\n\n"
        "Rules:\n"
        "- Structure: one ## section per agenda item or major topic — cover every distinct topic, even brief ones; never merge or drop topics\n"
        "- Within each section give the specifics: names, numbers, dates, amounts, identifiers, who argued what, and the outcome\n"
        "- When a formal title, motion, bill, or document name is read aloud, quote its exact wording — do not paraphrase it\n"
        "- State every decision with its outcome (e.g. adopted, rejected, vote result) and who made it\n"
        "- Summarize the content directly — never narrate the meeting from the outside ('the meeting covered…', 'a discussion was held…')\n"
        "- Do NOT replace specifics with generic language\n"
        "- Match depth to length: a long meeting warrants a proportionally detailed summary — never compress it into a few lines\n"
        "- Describe commitments and next steps in prose; do NOT format them as a to-do checklist (a separate agent extracts action items)"
    )
    if deps.project_system_prompt:
        core += f"\n\nAdditional instructions: {deps.project_system_prompt}"
    parts.append(core)
    return "\n\n".join(parts)


def action_item_instructions(deps: NoteDeps) -> str:
    return (
        "You extract concrete action items from meeting transcripts.\n\n"
        "Rules:\n"
        "- One action item = one specific task (starts with a verb)\n"
        '- owner: person explicitly assigned, or "TBD"\n'
        "- deadline: date or phrase if mentioned, null otherwise\n"
        '- priority: "high" if urgent/blocking, "medium" otherwise\n'
        '- Skip vague intentions like "we should think about X" — only concrete commitments\n'
        "- Return an empty list if there are no action items"
    )


def decision_logger_instructions(deps: NoteDeps) -> str:
    return (
        "You extract explicit decisions from meeting transcripts.\n\n"
        "Rules:\n"
        "- A decision = the group concluded, agreed, or chose something concrete\n"
        "- Include rationale if stated; use empty string if not mentioned\n"
        '- made_by: name(s) or "group" if a collective decision\n'
        "- Do NOT include action items or vague intentions\n"
        "- Return an empty list if no decisions were made"
    )


def interview_instructions(deps: NoteDeps) -> str:
    return (
        "You analyze job interview transcripts.\n\n"
        "- red_flags: concrete concerns (vague answers, gaps, contradictions)\n"
        "- green_flags: concrete positives (specific examples, depth, clarity)\n"
        "- suggested_followups: questions that would help further assess the candidate\n"
        "Return empty lists if nothing found."
    )


def lecture_instructions(deps: NoteDeps) -> str:
    return (
        "You analyze educational content transcripts (lectures, lessons, tutorials).\n"
        "Extract key concepts (with definitions and importance high|medium|low), "
        "learning objectives, assignments, and quiz questions with answers.\n"
        "Return empty lists if nothing found."
    )


def critic_instructions() -> str:
    return (
        "You are a strict quality reviewer for AI-generated meeting notes.\n"
        "Your only job is to evaluate and give specific revision advice — do NOT rewrite the content.\n\n"
        "Evaluate the provided content against the source transcript on four dimensions:\n\n"
        "Coverage (0–4): How completely are the transcript's key topics, decisions, and outcomes captured?\n"
        "  4 — Every distinct topic and agenda item present, with participants, decisions, and outcomes\n"
        "  3 — One minor topic or detail missed\n"
        "  2 — One notable topic missing OR significant detail lost\n"
        "  1 — Multiple topics missing\n"
        "  0 — Bulk of content absent\n\n"
        "Accuracy (0–3): Are all claims directly supported by the transcript?\n"
        "  3 — Every claim traceable to the transcript; no hallucinations\n"
        "  2 — Minor imprecision (slight paraphrase, not fabrication)\n"
        "  1 — One claim not present in the transcript\n"
        "  0 — Multiple invented details\n\n"
        "Specificity (0–2): Are concrete names, numbers, and decisions preserved?\n"
        "  2 — Specific names, amounts, dates, and decisions used where present in the transcript;\n"
        "      formal titles and motions quoted exactly as read aloud\n"
        "  1 — Some specifics replaced with vague language ('the team', 'some amount')\n"
        "  0 — Entirely generic; no concrete details retained\n\n"
        "Structure (0–1): Is the output well-organized and readable?\n"
        "  1 — Logical flow, clear sections\n"
        "  0 — Disorganized or hard to follow\n\n"
        "Rules:\n"
        "- Do NOT round up — if content is missing, deduct the full dimension points\n"
        "- issues: for each deduction, cite the specific missing topic or wrong claim from the transcript\n"
        "  and state exactly what needs to be added or corrected (actionable advice for the reviser)"
    )


INSTRUCTION_BUILDERS = {
    "Summarizer": summarizer_instructions,
    "ActionItemExtractor": action_item_instructions,
    "DecisionLogger": decision_logger_instructions,
    "InterviewAgent": interview_instructions,
    "LectureAgent": lecture_instructions,
}

# ---------------------------------------------------------------------------
# User prompt builders
# ---------------------------------------------------------------------------

def build_user_prompt(
    agent_name: str,
    deps: NoteDeps,
    transcript: str,
    previous_attempt: str = "",
    quality_notes: list[str] | None = None,
) -> str:
    if agent_name == "Summarizer":
        kb_snippet = ""
        if deps.project_knowledge_base:
            relevant = search_knowledge_base(deps.project_knowledge_base, deps.template_prompt)
            if relevant:
                kb_snippet = f"\nRelevant project context:\n{relevant}\n"
        retry_hint = ""
        if previous_attempt:
            retry_hint = (
                "\nThis is a revision. Previous draft (do NOT copy — write an improved version):\n"
                f"{previous_attempt[:REVISION_DRAFT_CHARS]}\n"
            )
        body = (
            f"Domain: {deps.domain_name}\n"
            f"Template instructions: {deps.template_prompt}\n"
            + kb_snippet
            + retry_hint
            + f"\nTranscript:\n{transcript}"
        )
    elif agent_name == "ActionItemExtractor":
        summary_hint = ""
        summary = deps.prior_results.get("Summarizer", {}).get("summary", "")
        if summary:
            summary_hint = (
                "\nSummary (for context only):\n"
                f"{summary[:500]}\n"
                "\nExtract every action item from the transcript, including any "
                "already mentioned in the summary above.\n"
            )
        body = f"Transcript:\n{transcript}{summary_hint}"
        if previous_attempt:
            body += (
                "\n\nThis is a revision. Previous attempt (do NOT copy — write an improved version):\n"
                f"{previous_attempt[:REVISION_DRAFT_CHARS]}"
            )
    else:
        body = f"Transcript:\n{transcript}"
        if previous_attempt:
            body += (
                "\n\nThis is a revision. Previous attempt (do NOT copy — write an improved version):\n"
                f"{previous_attempt[:REVISION_DRAFT_CHARS]}"
            )

    if quality_notes:
        body += "\n\nQuality review notes to address:\n" + "\n".join(f"- {n}" for n in quality_notes)
    return body


def build_critic_user_prompt(step_name: str, content: str, transcript: str) -> str:
    return (
        f"Section: {step_name}\n\n"
        f"Content to review:\n{content}\n\n"
        f"Transcript (ground truth):\n{transcript}"
    )


# ---------------------------------------------------------------------------
# Agents — no model bound here; supplied per-run
# ---------------------------------------------------------------------------

def _make_agent(name: str) -> Agent:
    agent = Agent(deps_type=NoteDeps, retries=2, name=name)
    builder = INSTRUCTION_BUILDERS[name]

    @agent.instructions
    def _instructions(ctx: RunContext[NoteDeps]) -> str:
        return builder(ctx.deps)

    return agent


AGENT_REGISTRY: dict[str, Agent] = {name: _make_agent(name) for name in INSTRUCTION_BUILDERS}

critic = Agent(deps_type=NoteDeps, retries=1, name="Critic")


@critic.instructions
def _critic_instructions(ctx: RunContext[NoteDeps]) -> str:
    return critic_instructions()
