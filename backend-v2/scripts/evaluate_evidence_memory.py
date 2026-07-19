from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from notes_agent_v2.evaluation.evidence_memory import (
    PHASE4_CASE_COUNTS,
    evaluate_phase4_feature,
    write_phase4_evaluation_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Phase 4 offline feature evaluation")
    parser.add_argument("--feature", required=True, choices=tuple(PHASE4_CASE_COUNTS))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="phase4-eval-") as work:
        report, results, trace_path = evaluate_phase4_feature(
            args.feature, Path(work)
        )
        write_phase4_evaluation_bundle(
            args.out, report=report, results=results, trace_path=trace_path
        )
    print(report.verdict)
    return 0 if report.verdict != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
