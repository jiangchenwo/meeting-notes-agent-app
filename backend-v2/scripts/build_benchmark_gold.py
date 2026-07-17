from __future__ import annotations

import argparse
import json
from pathlib import Path

from notes_agent_v2.evaluation.gold import GoldCandidate, build_gold


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic private 32-case development set.")
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.candidate_inventory.read_text())
    candidates = [GoldCandidate.model_validate(item) for item in inventory["candidates"]]
    reserved: set[str] = set()
    if args.reserved_manifest:
        reserved = set(json.loads(args.reserved_manifest.read_text()).get("case_ids", []))
    manifest = build_gold(candidates, args.private_out, reserved_case_ids=reserved)
    print(f"cases={len(manifest.cases)} synthetic=0 status={manifest.status} selection_digest={manifest.selection_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
