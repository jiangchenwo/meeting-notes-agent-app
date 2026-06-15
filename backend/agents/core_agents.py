import json
from .base import LLMAgent, WorkflowContext
from .tools import search_knowledge_base


class Summarizer(LLMAgent):
    name = "Summarizer"

    def run(self, ctx: WorkflowContext) -> dict:
        transcript = self._truncate_transcript(ctx.transcript, ctx.cfg)

        kb_snippet = ""
        if ctx.project_knowledge_base:
            relevant = search_knowledge_base(ctx.project_knowledge_base, ctx.template_prompt)
            if relevant:
                kb_snippet = f"\nRelevant project context:\n{relevant}\n"

        extra = ""
        if ctx.project_system_prompt:
            extra = f"\nAdditional instructions: {ctx.project_system_prompt}\n"

        system = (
            "You are a focused meeting summarizer. Your only job: write a clear, thorough summary.\n\n"
            "Rules:\n"
            "- Capture all key topics, decisions, and outcomes discussed\n"
            "- Use Markdown: ## headings for sections, **bold** for key terms, bullet lists\n"
            "- Be thorough — cover every significant topic\n"
            "- Include specific names, numbers, and decisions — do NOT replace them with generic language\n"
            "- Do NOT list action items inline; those are handled by a separate agent\n"
            + extra +
            "\nOutput ONLY valid JSON: {\"summary\": \"<markdown string>\"}\n"
            "No text outside the JSON object."
        )

        retry_hint = ""
        if ctx.previous_attempt:
            retry_hint = (
                f"\nThis is a revision. Previous draft (do NOT copy — write an improved version):\n"
                f"{ctx.previous_attempt[:1500]}\n"
            )

        user = (
            f"Domain: {ctx.domain_name}\n"
            f"Template instructions: {ctx.template_prompt}\n"
            + kb_snippet
            + retry_hint +
            f"\nTranscript:\n{transcript}"
        )

        content, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        parsed = self._parse_json(content, {"summary": content})
        return {
            "summary": parsed.get("summary", content),
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }


class ActionItemExtractor(LLMAgent):
    name = "ActionItemExtractor"

    def run(self, ctx: WorkflowContext) -> dict:
        transcript = self._truncate_transcript(ctx.transcript, ctx.cfg)

        summary_hint = ""
        if "Summarizer" in ctx.results:
            s = ctx.results["Summarizer"].get("summary", "")[:500]
            summary_hint = f"\nSummary (for context — find action items not already explicit here):\n{s}\n"

        system = (
            "You extract concrete action items from meeting transcripts.\n\n"
            "Rules:\n"
            "- One action item = one specific task (starts with a verb)\n"
            "- owner: person explicitly assigned, or \"TBD\"\n"
            "- deadline: date or phrase if mentioned, null otherwise\n"
            "- priority: \"high\" if urgent/blocking, \"medium\" otherwise\n"
            "- Skip vague intentions like \"we should think about X\" — only concrete commitments\n"
            "- Return empty array if there are no action items\n\n"
            "Output ONLY valid JSON:\n"
            "{\"action_items\": [{\"task\": \"...\", \"owner\": \"TBD\", \"deadline\": null, \"priority\": \"medium\"}]}\n"
            "No text outside the JSON object."
        )

        user = f"Transcript:\n{transcript}{summary_hint}"

        content, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        parsed = self._parse_json(content, {"action_items": []})
        raw = parsed.get("action_items", [])
        if not isinstance(raw, list):
            raw = []
        items = [
            i if isinstance(i, dict) else {"task": str(i), "owner": "TBD", "deadline": None, "priority": "medium"}
            for i in raw
        ]
        return {
            "action_items": items,
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }


class DecisionLogger(LLMAgent):
    name = "DecisionLogger"

    def run(self, ctx: WorkflowContext) -> dict:
        transcript = self._truncate_transcript(ctx.transcript, ctx.cfg)

        system = (
            "You extract explicit decisions from meeting transcripts.\n\n"
            "Rules:\n"
            "- A decision = the group concluded, agreed, or chose something concrete\n"
            "- Include rationale if stated; use empty string if not mentioned\n"
            "- made_by: name(s) or \"group\" if a collective decision\n"
            "- Do NOT include action items or vague intentions\n"
            "- Return empty array if no decisions were made\n\n"
            "Output ONLY valid JSON:\n"
            "{\"decisions\": [{\"decision\": \"...\", \"rationale\": \"...\", \"made_by\": \"...\"}]}\n"
            "No text outside the JSON object."
        )

        user = f"Transcript:\n{transcript}"

        content, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        parsed = self._parse_json(content, {"decisions": []})
        return {
            "decisions": parsed.get("decisions", []),
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }


class Critic(LLMAgent):
    name = "Critic"

    def run_critique(self, ctx: WorkflowContext, step_name: str, content: str) -> dict:
        system = (
            "You are a strict quality reviewer for AI-generated meeting notes.\n"
            "Your only job is to evaluate and give specific revision advice — do NOT rewrite the content.\n\n"
            "Evaluate the provided content against the source transcript on four dimensions:\n\n"
            "Coverage (0–4): How completely are the transcript's key topics, decisions, and outcomes captured?\n"
            "  4 — All major topics, participants, decisions, and outcomes present\n"
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
            "  2 — Specific names, amounts, dates, and decisions used where present in the transcript\n"
            "  1 — Some specifics replaced with vague language ('the team', 'some amount')\n"
            "  0 — Entirely generic; no concrete details retained\n\n"
            "Structure (0–1): Is the output well-organized and readable?\n"
            "  1 — Logical flow, clear sections\n"
            "  0 — Disorganized or hard to follow\n\n"
            "Total score = sum of all four dimensions (0–10).\n\n"
            "Calibration:\n"
            "  ≥ 9 — excellent, ready to use as-is\n"
            "  8   — good but one thing should be tightened\n"
            "  7   — one concrete gap or imprecision still present\n"
            "  ≤ 6 — notable quality problem; must be revised\n\n"
            "Rules:\n"
            "- Do NOT round up — if content is missing, deduct the full dimension points\n"
            "- score MUST equal the sum of the four dimension values\n"
            "- issues: for each deduction, cite the specific missing topic or wrong claim from the transcript\n"
            "  and state exactly what needs to be added or corrected (actionable advice for the reviser)\n\n"
            "Output ONLY valid JSON:\n"
            "{\"dimensions\": {\"coverage\": 3, \"accuracy\": 3, \"specificity\": 1, \"structure\": 1}, "
            "\"score\": 8, \"issues\": [\"Add the budget approval decision made by Sarah — currently missing\"]}\n"
            "No text outside the JSON object."
        )

        user = (
            f"Section: {step_name}\n\n"
            f"Content to review:\n{content}\n\n"
            f"Transcript (ground truth):\n{ctx.transcript}"
        )

        raw, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        fallback = {"dimensions": {}, "score": 5, "issues": ["Could not parse critique response"]}
        parsed = self._parse_json(raw, fallback)

        # Enforce consistency: recompute score from dimensions if both are present
        dims = parsed.get("dimensions", {})
        if dims and all(k in dims for k in ("coverage", "accuracy", "specificity", "structure")):
            computed = dims["coverage"] + dims["accuracy"] + dims["specificity"] + dims["structure"]
            score = float(computed)
        else:
            score = float(parsed.get("score", 5))

        return {
            "dimensions": dims,
            "score": score,
            "issues": parsed.get("issues", []),
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }

    def run(self, ctx: WorkflowContext) -> dict:
        raise NotImplementedError("Use run_critique(ctx, step_name, content) instead")
