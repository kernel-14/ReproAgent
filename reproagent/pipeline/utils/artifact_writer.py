"""Artifact manifest writer for reproagent run outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "1.0"
_MANIFEST_NAME = "artifact_manifest.json"
_RUN_MARKERS = {"nodes", "state_snapshots", "contracts", "attempts", "workspace", "reports", "debug"}
_CONTRACT_FILENAMES = {
    "boundary_requirements.json",
    "canonical_ir.json",
    "canonical_ir_validation.json",
    "repo_handoff.json",
    "generation_manifest.json",
    "global_contract.json",
    "handoff.json",
    "package_file_planning.json",
    "project_plan.json",
    "repo_plan.json",
    "semantic_assertions.json",
    "upstream_intent.json",
    "validated_repo_handoff.json",
}
_SOURCE_OF_TRUTH_FILENAMES = {
    "artifact_manifest.json",
    "canonical_ir.json",
    "handoff.json",
    "quality_status.json",
    "repo_plan.json",
    "stage_status.json",
    "stage_attempts.json",
    "upstream_intent.json",
    "workflow_events.jsonl",
}
_REPORT_FILENAMES = {
    "run_summary.json",
    "terminal_outcome.json",
    "validation_report.json",
    "benchmark_report.json",
    "file_provenance.json",
    "repair_review.json",
    "recovery_tickets.json",
}
_DEDUP_KIND_ALLOWLIST = {"output", "report", "state"}
_DEDUP_MIN_BYTES = 512


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_jsonable(payload: object) -> object:
    return payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_hash(data: bytes, path: Path) -> str:
    """Hash JSON artifacts by normalized payload so legacy formatting still deduplicates."""
    if path.suffix == ".json":
        try:
            normalized = json.dumps(
                json.loads(data.decode("utf-8")),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            ).encode("utf-8")
            return _content_hash(normalized)
        except Exception:
            pass
    return _content_hash(data)


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / _MANIFEST_NAME


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = _manifest_path(run_dir)
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "artifacts": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    artifacts = payload.get("artifacts", []) if isinstance(payload, dict) else []
    return {
        "schema_version": str(payload.get("schema_version", _SCHEMA_VERSION)) if isinstance(payload, dict) else _SCHEMA_VERSION,
        "artifacts": artifacts if isinstance(artifacts, list) else [],
    }


def _save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = _manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _find_existing_by_hash(
    manifest: dict[str, Any],
    content_hash: str,
    *,
    exclude_path: str = "",
) -> dict[str, Any] | None:
    if not content_hash:
        return None
    for item in list(manifest.get("artifacts", []) or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("canonical_path", "") or "") == exclude_path:
            continue
        if str(item.get("authority", "") or "") == "compatibility_alias":
            continue
        if str(item.get("content_hash", "") or "") == content_hash:
            return item
    return None


def _relative_path(run_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def infer_run_dir(path: Path) -> Path | None:
    """Infer a run root from a known artifact path, returning None when unsafe."""
    resolved = path.resolve()
    for ancestor in [resolved.parent, *resolved.parents]:
        if (ancestor / _MANIFEST_NAME).exists():
            return ancestor
        marker = ancestor.name
        if marker in _RUN_MARKERS and ancestor.parent != ancestor:
            return ancestor.parent
    return None


def classify_artifact_path(
    path: Path,
    *,
    logical_name: str = "",
    kind: str = "output",
    stage: str = "",
    node: str = "",
    authority: str = "derived",
    retention: str = "keep",
) -> dict[str, str]:
    """Normalize manifest metadata for well-known reproagent artifact names."""
    filename = path.name
    normalized_kind = kind
    normalized_stage = stage
    normalized_node = node
    normalized_authority = authority
    normalized_retention = retention
    normalized_logical_name = logical_name or path.stem

    if filename in _CONTRACT_FILENAMES:
        normalized_kind = "contract"
    if filename in _REPORT_FILENAMES:
        normalized_kind = "report"
    if filename in {"stage_status.json", "stage_attempts.json", "quality_status.json", "latest_state.json"}:
        normalized_kind = "state"
    if path.suffix in {".md", ".txt", ".jsonl"}:
        normalized_kind = "log" if normalized_kind == "output" else normalized_kind
    if "state_snapshots" in path.parts:
        normalized_kind = "state"
        normalized_authority = "debug_snapshot"
        normalized_retention = "debug"

    if filename in _SOURCE_OF_TRUTH_FILENAMES:
        normalized_authority = "source_of_truth"
    elif filename in _CONTRACT_FILENAMES and normalized_authority == "derived":
        normalized_authority = "source_of_truth" if filename in {"canonical_ir.json", "repo_plan.json", "handoff.json", "upstream_intent.json"} else "derived"

    if filename in {"current_prompt.md", "last_attempt.json"} or "tasks" in path.parts:
        normalized_retention = "debug"
        if normalized_authority == "derived":
            normalized_authority = "debug_snapshot"

    if not normalized_stage:
        parts = list(path.parts)
        if "nodes" in parts:
            index = parts.index("nodes")
            if index + 1 < len(parts):
                normalized_node = normalized_node or parts[index + 1]
                normalized_stage = normalized_stage or parts[index + 1]
        elif filename in _CONTRACT_FILENAMES:
            normalized_stage = "contract"
        elif filename in _REPORT_FILENAMES:
            normalized_stage = "report"

    return {
        "logical_name": normalized_logical_name,
        "kind": normalized_kind,
        "stage": normalized_stage,
        "node": normalized_node,
        "authority": normalized_authority,
        "retention": normalized_retention,
    }


def register_artifact(
    *,
    run_dir: Path,
    path: Path,
    logical_name: str = "",
    kind: str = "output",
    stage: str = "",
    node: str = "",
    attempt_id: str = "",
    authority: str = "derived",
    retention: str = "keep",
    schema_name: str = "",
    schema_version: str = "",
    content_hash: str = "",
    size_bytes: int = 0,
    aliases: list[str] | None = None,
    depends_on: list[str] | None = None,
    producer: str = "",
    consumer_hints: list[str] | None = None,
) -> None:
    """Upsert one artifact record into artifact_manifest.json."""
    run_dir = Path(run_dir)
    canonical_path = _relative_path(run_dir, Path(path))
    if canonical_path == _MANIFEST_NAME:
        return
    manifest = _load_manifest(run_dir)
    alias_target = ""
    effective_authority = authority
    effective_hash = content_hash
    effective_size = int(size_bytes or 0)
    if (
        effective_authority != "source_of_truth"
        and effective_authority != "compatibility_alias"
        and kind in _DEDUP_KIND_ALLOWLIST
        and effective_size >= _DEDUP_MIN_BYTES
    ):
        existing = _find_existing_by_hash(manifest, effective_hash, exclude_path=canonical_path)
        if existing is not None:
            alias_target = str(existing.get("canonical_path", "") or "")
            effective_authority = "compatibility_alias"
    record = {
        "artifact_id": canonical_path,
        "logical_name": logical_name or Path(canonical_path).stem,
        "kind": kind,
        "stage": stage,
        "node": node,
        "attempt_id": attempt_id,
        "canonical_path": canonical_path,
        "aliases": list(dict.fromkeys([*list(aliases or []), *([alias_target] if alias_target else [])])),
        "producer": producer,
        "consumer_hints": list(consumer_hints or []),
        "content_hash": effective_hash,
        "size_bytes": effective_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retention": retention,
        "authority": effective_authority,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "depends_on": list(depends_on or []),
    }
    artifacts = [item for item in manifest["artifacts"] if item.get("canonical_path") != canonical_path]
    artifacts.append(record)
    artifacts.sort(key=lambda item: str(item.get("canonical_path", "")))
    manifest["artifacts"] = artifacts
    _save_manifest(run_dir, manifest)


def write_artifact(
    *,
    run_dir: Path,
    path: Path,
    payload: object,
    logical_name: str = "",
    kind: str = "output",
    stage: str = "",
    node: str = "",
    attempt_id: str = "",
    authority: str = "derived",
    retention: str = "keep",
    schema_name: str = "",
    schema_version: str = "",
    alias_of: str = "",
    depends_on: list[str] | None = None,
) -> None:
    """Write an artifact and register it in the run manifest.

    Compatibility aliases are pointer files so legacy paths do not duplicate large payloads.
    """
    run_dir = Path(run_dir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if authority == "compatibility_alias" and alias_of:
        json_payload: object = {"artifact_ref": alias_of, "canonical_path": alias_of}
    else:
        json_payload = _to_jsonable(payload)
    data = json.dumps(json_payload, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")
    content_hash = _artifact_hash(data, path)
    alias_target = ""
    effective_authority = authority
    if (
        authority not in {"source_of_truth", "compatibility_alias"}
        and kind in _DEDUP_KIND_ALLOWLIST
        and len(data) >= _DEDUP_MIN_BYTES
    ):
        existing = _find_existing_by_hash(
            _load_manifest(run_dir),
            content_hash,
            exclude_path=_relative_path(run_dir, path),
        )
        if existing is not None:
            alias_target = str(existing.get("canonical_path", "") or "")
            effective_authority = "compatibility_alias"
    if effective_authority == "compatibility_alias" and (alias_target or alias_of):
        target = alias_target or alias_of
        pointer_data = json.dumps(
            {"artifact_ref": target, "canonical_path": target},
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        path.write_bytes(pointer_data)
    else:
        path.write_bytes(data)
    register_artifact(
        run_dir=run_dir,
        path=path,
        logical_name=logical_name,
        kind=kind,
        stage=stage,
        node=node,
        attempt_id=attempt_id,
        authority=effective_authority,
        retention=retention,
        schema_name=schema_name,
        schema_version=schema_version,
        content_hash=content_hash,
        size_bytes=len(data),
        aliases=[item for item in [alias_of, alias_target] if item],
        depends_on=depends_on,
    )


def write_text_artifact(
    *,
    run_dir: Path,
    path: Path,
    content: str,
    logical_name: str = "",
    kind: str = "log",
    stage: str = "",
    node: str = "",
    authority: str = "debug_snapshot",
    retention: str = "debug",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = str(content).encode("utf-8")
    path.write_bytes(data)
    register_artifact(
        run_dir=Path(run_dir),
        path=path,
        logical_name=logical_name,
        kind=kind,
        stage=stage,
        node=node,
        authority=authority,
        retention=retention,
        content_hash=_content_hash(data),
        size_bytes=len(data),
    )


def register_existing_file(
    path: Path,
    *,
    run_dir: Path | None = None,
    logical_name: str = "",
    kind: str = "output",
    stage: str = "",
    node: str = "",
    authority: str = "derived",
    retention: str = "keep",
) -> None:
    """Register a file written by legacy code when it is inside a run directory."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return
    resolved_run_dir = Path(run_dir) if run_dir is not None else infer_run_dir(path)
    if resolved_run_dir is None:
        return
    metadata = classify_artifact_path(
        path,
        logical_name=logical_name,
        kind=kind,
        stage=stage,
        node=node,
        authority=authority,
        retention=retention,
    )
    data = path.read_bytes()
    content_hash = _artifact_hash(data, path)
    manifest = _load_manifest(resolved_run_dir)
    alias_target = ""
    effective_authority = metadata["authority"]
    if (
        effective_authority not in {"source_of_truth", "compatibility_alias"}
        and metadata["kind"] in _DEDUP_KIND_ALLOWLIST
        and len(data) >= _DEDUP_MIN_BYTES
    ):
        existing = _find_existing_by_hash(
            manifest,
            content_hash,
            exclude_path=_relative_path(resolved_run_dir, path),
        )
        if existing is not None:
            alias_target = str(existing.get("canonical_path", "") or "")
            effective_authority = "compatibility_alias"
            pointer_payload = {"artifact_ref": alias_target, "canonical_path": alias_target}
            path.write_text(json.dumps(pointer_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    register_artifact(
        run_dir=resolved_run_dir,
        path=path,
        logical_name=metadata["logical_name"],
        kind=metadata["kind"],
        stage=metadata["stage"],
        node=metadata["node"],
        authority=effective_authority,
        retention=metadata["retention"],
        content_hash=content_hash,
        size_bytes=len(data),
        aliases=[alias_target] if alias_target else [],
    )
