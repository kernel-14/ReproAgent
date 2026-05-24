"""Top-level entrypoint for the Refined Coreset Selection reproduction."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from src.lbcs_reproduction import run_all_experiments


def _default_config(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "dataset": "cifar10",
        "model": "resnet18",
        "method": "LBCS",
        "epsilon": 0.3,
        "initial_k": 600,
        "batch_size": 64,
        "seed": 42,
    }


def run_reproduction(mode: str = "runtime_smoke") -> dict[str, Any]:
    """Run the paper-visible LBCS datasets, methods, sweeps, and artifacts."""
    config = _default_config(mode)
    output_dir = "results"
    manifest = run_all_experiments(output_dir=output_dir, mode=mode)

    summary = {
        "paper": "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints",
        "mode": mode,
        "config": config,
        "method": "LBCS",
        "algorithm_1": "lexicographic bilevel coreset selection",
        "algorithm_2": "refinement binary search called at step 4 of Algorithm 1",
        "baselines": ["Uniform", "EL2N", "GraNd", "Influential", "Moderate", "CCS", "Probabilistic"],
        "datasets": ["F-MNIST", "SVHN", "CIFAR-10", "CIFAR-100", "MNIST-S"],
        "artifacts": {
            "dataset_access": f"{output_dir}/dataset_access_report.json",
            "figure1": f"{output_dir}/figure1_objectives.json",
            "table2": f"{output_dir}/table2.csv",
            "figure3": f"{output_dir}/table2_figure3.json",
            "table3": f"{output_dir}/table3.csv",
            "figure2_figure4": f"{output_dir}/section5_3_figures_2_4.json",
            "table5": f"{output_dir}/table5.csv",
            "table6": f"{output_dir}/table6.csv",
            "table9": f"{output_dir}/table9.csv",
        },
        "manifest": manifest,
    }
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="runtime_smoke", choices=["runtime_smoke", "docker_validate", "full"])
    parser.add_argument("--output", default="results/reproduction_summary.json")
    args = parser.parse_args()

    summary = run_reproduction(args.mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "mode": args.mode, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
