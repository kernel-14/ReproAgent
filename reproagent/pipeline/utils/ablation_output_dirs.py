from __future__ import annotations

import shutil
from pathlib import Path


def clear_ablation_output_dirs(dest: Path) -> None:
    """Remove stale per-paper run artifacts before starting a fresh ablation run."""
    for name in ("logs", "meta", "node", "repo", "score"):
        path = dest / name
        if not path.exists():
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()
            continue
        for child in list(path.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
