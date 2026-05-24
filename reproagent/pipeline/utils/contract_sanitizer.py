"""Sanitize generation contracts so old or LLM-written plans cannot request scaffold outputs."""

from __future__ import annotations

import re
from typing import Any

_SCAFFOLD_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("schema-only", "measured-code-backed"),
    ("dry-run-only", "bounded-route-backed"),
    ("registry values", "executable constants/default accessors"),
    ("writer/declaration hooks", "writer functions that call evaluation/metric code"),
    ("declaration hooks", "writer functions that call evaluation/metric code"),
    ("scaffold", "complete executable implementation"),
    ("skeleton", "complete executable implementation"),
    ("stub", "complete executable implementation"),
    ("placeholder", "bounded executable implementation"),
    ("toy demo", "bounded faithful implementation"),
    ("dummy value", "measured bounded value"),
    ("mock metric", "implemented metric function with bounded inputs"),
    ("initial draft", "complete faithful implementation"),
    ("initial version", "complete faithful implementation"),
)


def _normalize_scaffold_request(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    changed = cleaned
    for old, new in _SCAFFOLD_REPLACEMENTS:
        if old.lower() in lowered:
            changed = re.sub(re.escape(old), new, changed, flags=re.IGNORECASE)
            lowered = changed.lower()
    return changed


def sanitize_scope_boundary(scope: Any) -> dict[str, list[str]]:
    """Keep generation-facing scope as preserve/focus obligations only."""
    if not isinstance(scope, dict):
        return {}
    sanitized: dict[str, list[str]] = {}
    for key in ("preserve", "implementation_focus"):
        values: list[str] = []
        for raw in list(scope.get(key, []) or []):
            text = _normalize_scaffold_request(str(raw or ""))
            if not text:
                continue
            if text not in values:
                values.append(text)
        if values:
            sanitized[key] = values[:24]
    return sanitized


def sanitize_contract_list(items: Any, *, field: str = "") -> list[str]:
    """Return contract strings rewritten away from scaffold/declaration obligations."""
    del field
    sanitized: list[str] = []
    seen: set[str] = set()
    for raw in list(items or []):
        text = _normalize_scaffold_request(str(raw or ""))
        if not text:
            continue
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        sanitized.append(key)
    return sanitized


def sanitize_contract_text(text: Any) -> str:
    """Sanitize a free-form generation prompt while preserving implementation guardrails."""
    rendered = str(text or "").strip()
    if not rendered:
        return ""
    lines: list[str] = []
    for line in rendered.splitlines():
        normalized = _normalize_scaffold_request(line)
        if not normalized:
            continue
        lines.append(normalized)
    return "\n".join(line for line in lines if line).strip()


def synthesize_missing_call_symbols(task: dict[str, Any]) -> list[str]:
    """Infer minimal active-route call symbols when older plans left calls_symbols empty."""
    existing = [str(item).strip() for item in list(task.get("calls_symbols", []) or []) if str(item).strip()]
    symbols = list(existing)
    obligations = " ".join(str(item or "").lower() for item in list(task.get("method_obligations", []) or []))
    writes = [str(item or "").strip() for item in list(task.get("writes_artifacts", []) or []) if str(item).strip()]
    text = " ".join([obligations, " ".join(writes)]).lower()

    def add(*values: str) -> None:
        for value in values:
            if value and value not in symbols:
                symbols.append(value)

    if writes or any(token in text for token in ("artifact", "figure", "table", "report", "plot")):
        for path in writes[:8]:
            base = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
            if slug:
                add(f"write_{slug}_artifact")
    for kind, number in re.findall(r"\b(figure|fig|table)\.?\s*([0-9]+[a-z]?)\b", text):
        normalized_kind = "figure" if kind in {"fig", "figure"} else "table"
        add(f"run_{normalized_kind}_{number}_route", f"write_{normalized_kind}_{number}_artifact")
    return symbols[:24]


def sanitize_task_contract(task: dict[str, Any]) -> dict[str, Any]:
    """Sanitize one task dict used by generation and review."""
    payload = dict(task or {})
    payload.pop("stop_rule_or_pruning_rationale", None)
    payload["interface_contract"] = sanitize_contract_list(payload.get("interface_contract", []), field="interface_contract")
    payload["method_obligations"] = sanitize_contract_list(payload.get("method_obligations", []), field="method_obligations")
    payload["review_points"] = sanitize_contract_list(payload.get("review_points", []), field="review_points")
    payload["generation_prompt"] = sanitize_contract_text(payload.get("generation_prompt", ""))
    payload["scope_boundary"] = sanitize_scope_boundary(payload.get("scope_boundary", {}))
    payload["calls_symbols"] = synthesize_missing_call_symbols(payload)
    return payload
