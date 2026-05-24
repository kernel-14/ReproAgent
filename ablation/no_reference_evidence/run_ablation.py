#!/usr/bin/env python
"""Run ReproAgent with reference evidence disabled."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproagent.pipeline.paperbench_cli import main


def _with_no_clone_references(argv: list[str]) -> list[str]:
    if "--no-clone-references" in argv or "--clone-references" in argv:
        return argv
    return [*argv, "--no-clone-references"]


if __name__ == "__main__":
    raise SystemExit(main(_with_no_clone_references(sys.argv[1:])))
