from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from notes_agent_v2.evaluation.artifacts import EvaluationBundleWriter
from notes_agent_v2.evaluation.reporting import build_report, render_report_json, render_report_markdown
from notes_agent_v2.evaluation.runner import EvaluationCell, EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a scripted paired feature evaluation offline.")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.cells.read_text())
    cells = [EvaluationCell.model_validate(item["cell"]) for item in payload["cells"]]
    scripted = {EvaluationCell.model_validate(item["cell"]).key: item["result"] for item in payload["cells"]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=args.private_out.parent) as temporary:
        work = Path(temporary)
        runner = EvaluationRunner(work, lambda cell: scripted[cell.key])
        results = runner.run(cells)
        with runner.trace.span("report", feature_id=args.feature, report_fingerprint=fingerprint) as report_span:
            report = build_report(args.feature, results)
            report_span.terminal(artifact_digests={"report": hashlib.sha256(render_report_json(report).encode()).hexdigest()})
        for result in results:
            result_bytes = json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
            with runner.trace.span("artifact", feature_id=args.feature, case_id=result.cell.case_id, variant=result.cell.variant, seed=result.cell.seed) as artifact_span:
                artifact_span.terminal(artifact_digests={"case_result": hashlib.sha256(result_bytes).hexdigest()})
        writer = EvaluationBundleWriter(args.private_out, run_id=f"{args.feature}-scripted", fingerprint=fingerprint)
        with runner.trace.span("artifact", feature_id=args.feature, artifact_type="evaluation_bundle") as bundle_span:
            for result in results:
                writer.write_json(f"cases/{result.cell.key}.json", result.model_dump(mode="json"))
            writer.write_json("result-index.json", {"feature_id": args.feature, "result_keys": [item.cell.key for item in results]})
            writer.write_text("report.json", render_report_json(report))
            writer.write_text("report.md", render_report_markdown(report))
            bundle_span.terminal()
        writer.write_text("events.jsonl", (work / "events.jsonl").read_text())
        manifest = writer.seal()
    print(f"verdict={report.verdict} pairs={report.pair_count} bundle_digest={manifest.bundle_digest}")
    return 0 if report.verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
