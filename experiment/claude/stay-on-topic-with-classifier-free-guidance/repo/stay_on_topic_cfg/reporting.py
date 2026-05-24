from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Mapping[str, Any]) -> str:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path.name


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path.name


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> str:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return path.name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path.name


def write_manifest(output_dir: Path, artifacts: list[str]) -> str:
    manifest = {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "smoke_outputs_are_not_paper_scores": True,
    }
    return write_json(output_dir / "artifact_manifest.json", manifest)

