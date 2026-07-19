from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from notes_agent_v2.evaluation.contract_persistence import (
    PHASE3_CASE_COUNTS,
    evaluate_phase3_feature,
    write_phase3_evaluation_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one offline Phase 3 conformance evaluation.")
    parser.add_argument("--feature", choices=tuple(PHASE3_CASE_COUNTS), required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()
    if args.private_out.exists():
        parser.error("private output already exists")
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase3-evaluation-", dir=args.private_out.parent) as temporary:
        report, results, trace_path = evaluate_phase3_feature(args.feature, Path(temporary))
        manifest = write_phase3_evaluation_bundle(
            args.private_out,
            report=report,
            results=results,
            trace_path=trace_path,
        )
    print(
        f"verdict={report.verdict} cases={report.case_count} "
        f"requests={report.request_count} bundle_digest={manifest.bundle_digest}"
    )
    return 0 if report.verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
