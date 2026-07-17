from __future__ import annotations

import argparse
from pathlib import Path

from notes_agent_v2.evaluation.fetch import extract_archive, fetch_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicitly fetch and safely extract one benchmark archive.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--url")
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-to", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    archive = args.archive
    if args.url:
        if args.download_to is None:
            parser.error("--url requires --download-to")
        actual = fetch_archive(args.url, args.download_to, allow_network=args.allow_network, max_bytes=args.max_bytes)
        if actual != args.checksum.lower():
            args.download_to.unlink(missing_ok=True)
            parser.error("download checksum mismatch")
        archive = args.download_to
    files = extract_archive(archive, args.output, args.checksum, max_bytes=args.max_bytes)
    print(f"extracted_files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
