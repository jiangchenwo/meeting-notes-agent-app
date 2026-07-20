from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from notes_agent_v2.evaluation.generation_quality import (
    OFFLINE_CASE_COUNTS,
    evaluate_generation_feature,
    write_generation_evaluation_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an offline generation-quality evaluation."
    )
    parser.add_argument("--feature", choices=tuple(OFFLINE_CASE_COUNTS), required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()
    if args.private_out.exists():
        parser.error("private output already exists")
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="generation-quality-", dir=args.private_out.parent
    ) as temporary:
        report, results, trace_path = evaluate_generation_feature(
            args.feature, Path(temporary)
        )
        manifest = write_generation_evaluation_bundle(
            args.private_out,
            report=report,
            results=results,
            trace_path=trace_path,
        )
    print(
        f"verdict={report.verdict} cases={report.case_count} "
        f"requests={report.provider_requests} "
        f"bundle_digest={manifest.bundle_digest}"
    )
    return 1 if report.verdict == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
