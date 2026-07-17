from __future__ import annotations

import argparse
import json
from pathlib import Path

from notes_agent_v2.evaluation.reporting import build_report, render_report_json, render_report_markdown
from notes_agent_v2.evaluation.runner import EvaluationResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic report from case-level evaluation results.")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = tuple(EvaluationResult.model_validate(item) for item in json.loads(args.results.read_text())["results"])
    report = build_report(args.feature, results)
    args.output.write_text(render_report_json(report))
    args.output.with_suffix(".md").write_text(render_report_markdown(report))
    print(f"verdict={report.verdict} pairs={report.pair_count} digest={report.result_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
