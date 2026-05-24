#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source experiment_runners/papers.sh

PYTHON="${PYTHON:-python}"
STAGE="${REPROAGENT_STAGE:-repair}"

if [ "$#" -gt 0 ]; then
  PAPERS=("$@")
fi

unset PAPERBENCH_REPRO_DISABLE_SEMANTIC_ANCHOR

for paper in "${PAPERS[@]}"; do
  run_id="no_reference_repo_${paper//-/_}_$(date +%Y%m%d_%H%M%S)"
  echo "[ReproAgent/no_reference_repo] $paper -> $run_id"
  "$PYTHON" -u ablation/no_reference_repo/run_ablation.py "$paper" --run-id "$run_id" --stage "$STAGE"
done
