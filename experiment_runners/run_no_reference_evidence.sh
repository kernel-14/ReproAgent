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

unset PAPERBENCH_REPRO_DISABLE_IMPLEMENTATION_REQUIREMENTS

for paper in "${PAPERS[@]}"; do
  run_id="no_reference_evidence_${paper//-/_}_$(date +%Y%m%d_%H%M%S)"
  echo "[ReproAgent/no_reference_evidence] $paper -> $run_id"
  "$PYTHON" -u ablation/no_reference_evidence/run_ablation.py "$paper" --run-id "$run_id" --stage "$STAGE"
done
