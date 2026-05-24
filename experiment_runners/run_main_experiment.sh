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

for paper in "${PAPERS[@]}"; do
  run_id="main_${paper//-/_}_$(date +%Y%m%d_%H%M%S)"
  echo "[ReproAgent/main] $paper -> $run_id"
  "$PYTHON" -u run_paperbench.py "$paper" --run-id "$run_id" --stage "$STAGE"
done
