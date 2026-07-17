from __future__ import annotations

import argparse
from pathlib import Path

from notes_agent_v2.evaluation.artifacts import verify_bundle
from notes_agent_v2.evaluation.tracing import validate_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a safe evaluation trace and its sealed bundle.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    trace = validate_trace(args.run / "events.jsonl")
    bundle = verify_bundle(args.run) if args.verify else None
    print(f"spans={trace.span_count} requests={trace.request_count} failures={trace.failure_count}")
    if bundle:
        print(f"bundle_digest={bundle.bundle_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
