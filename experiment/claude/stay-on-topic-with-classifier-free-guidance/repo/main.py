#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stay_on_topic_cfg.runner import run_full_plan, run_runtime_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stay-on-topic CFG reproduction runner")
    parser.add_argument("--mode", choices=["runtime_smoke", "dry_run", "full"], default="runtime_smoke")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config", default="configs/experiment_matrix.yaml")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.mode in {"runtime_smoke", "dry_run"}:
        result = run_runtime_smoke(output_dir=output_dir, seed=args.seed, config_path=args.config)
    else:
        result = run_full_plan(output_dir=output_dir, seed=args.seed, config_path=args.config)
    print(f"wrote {len(result.artifacts)} artifacts to {output_dir}")
    for artifact in result.artifacts:
        print(f"- {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

