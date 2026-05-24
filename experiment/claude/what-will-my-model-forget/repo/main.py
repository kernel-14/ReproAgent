"""Minimal CLI for the WWMF PaperBench reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wwmf.paper_protocol import run_all_paper_protocols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    payload = run_all_paper_protocols(data_root=args.data_root, output_dir=args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wwmf_protocol_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

