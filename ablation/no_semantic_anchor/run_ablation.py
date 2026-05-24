#!/usr/bin/env python
"""Run ReproAgent with semantic anchors disabled."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproagent.pipeline.paperbench_cli import main


def _configure_ablation() -> None:
    os.environ["PAPERBENCH_REPRO_DISABLE_SEMANTIC_ANCHOR"] = "1"


if __name__ == "__main__":
    _configure_ablation()
    raise SystemExit(main())
