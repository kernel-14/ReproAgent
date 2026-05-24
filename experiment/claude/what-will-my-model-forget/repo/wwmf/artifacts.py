"""Artifact helpers for compact WWMF outputs."""

import json
from pathlib import Path


def write_json_artifact(path: str | Path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out

