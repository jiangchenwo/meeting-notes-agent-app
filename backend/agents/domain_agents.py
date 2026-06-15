from .base import LLMAgent, WorkflowContext


class InterviewAgent(LLMAgent):
    name = "InterviewAgent"

    def run(self, ctx: WorkflowContext) -> dict:
        transcript = self._truncate_transcript(ctx.transcript, ctx.cfg)
        system = (
            "You analyze job interview transcripts.\n\n"
            "Output ONLY valid JSON:\n"
            "{\n"
            "  \"questions_asked\": [{\"question\": \"...\", \"type\": \"behavioral|technical|situational|other\"}],\n"
            "  \"candidate_highlights\": [\"...\"],\n"
            "  \"red_flags\": [\"...\"],\n"
            "  \"green_flags\": [\"...\"],\n"
            "  \"suggested_followups\": [\"...\"]\n"
            "}\n"
            "- red_flags: concrete concerns (vague answers, gaps, contradictions)\n"
            "- green_flags: concrete positives (specific examples, depth, clarity)\n"
            "- suggested_followups: questions that would help further assess the candidate\n"
            "Return empty arrays if nothing found. No text outside the JSON object."
        )
        user = f"Transcript:\n{transcript}"
        content, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        parsed = self._parse_json(content, {
            "questions_asked": [], "candidate_highlights": [],
            "red_flags": [], "green_flags": [], "suggested_followups": [],
        })
        return {
            "questions_asked": parsed.get("questions_asked", []),
            "candidate_highlights": parsed.get("candidate_highlights", []),
            "red_flags": parsed.get("red_flags", []),
            "green_flags": parsed.get("green_flags", []),
            "suggested_followups": parsed.get("suggested_followups", []),
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }


class LectureAgent(LLMAgent):
    name = "LectureAgent"

    def run(self, ctx: WorkflowContext) -> dict:
        transcript = self._truncate_transcript(ctx.transcript, ctx.cfg)
        system = (
            "You analyze educational content transcripts (lectures, lessons, tutorials).\n\n"
            "Output ONLY valid JSON:\n"
            "{\n"
            "  \"key_concepts\": [{\"concept\": \"...\", \"definition\": \"...\", \"importance\": \"high|medium|low\"}],\n"
            "  \"learning_objectives\": [\"...\"],\n"
            "  \"assignments\": [{\"task\": \"...\", \"due\": \"...\", \"notes\": \"...\"}],\n"
            "  \"quiz_questions\": [{\"question\": \"...\", \"answer\": \"...\"}]\n"
            "}\n"
            "Return empty arrays if nothing found. No text outside the JSON object."
        )
        user = f"Transcript:\n{transcript}"
        content, in_tok, out_tok = self._call_llm(system, user, ctx.cfg)
        parsed = self._parse_json(content, {
            "key_concepts": [], "learning_objectives": [], "assignments": [], "quiz_questions": [],
        })
        return {
            "key_concepts": parsed.get("key_concepts", []),
            "learning_objectives": parsed.get("learning_objectives", []),
            "assignments": parsed.get("assignments", []),
            "quiz_questions": parsed.get("quiz_questions", []),
            "_tokens": {"input": in_tok, "output": out_tok},
            "_prompt": {"system": system, "user": user},
        }
