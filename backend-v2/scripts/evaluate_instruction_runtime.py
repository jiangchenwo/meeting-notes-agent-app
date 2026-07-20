from __future__ import annotations

import argparse
from pathlib import Path

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.instruction_runtime import (
    OFFLINE_CASE_COUNTS,
    evaluate_planning_feature,
    write_planning_evaluation_bundle,
)
from notes_agent_v2.evaluation.tracing import validate_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature",
        choices=tuple(sorted(OFFLINE_CASE_COUNTS)) + ("all",),
        default="all",
    )
    args = parser.parse_args()
    features = (
        tuple(sorted(OFFLINE_CASE_COUNTS))
        if args.feature == "all"
        else (args.feature,)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for feature_id in features:
        name = feature_id.replace(".", "-")
        work = args.output / f".{name}-work"
        bundle = args.output / name
        report, results, trace_path = evaluate_planning_feature(feature_id, work)
        validate_trace(trace_path)
        manifest = write_planning_evaluation_bundle(
            bundle, report=report, results=results, trace_path=trace_path
        )
        if verify_bundle(bundle) != manifest:
            raise RuntimeError("sealed bundle verification mismatch")
        print(
            f"{feature_id}: verdict={report.verdict} "
            f"cases={report.treatment_correct}/{report.case_count} "
            f"bundle={manifest.bundle_digest}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
