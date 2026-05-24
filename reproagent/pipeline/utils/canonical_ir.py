"""Shadow canonical IR construction and validation helpers for reproagent."""

from __future__ import annotations

from collections import Counter
import re

from reproagent.pipeline.schemas import (
    CanonicalContractStage,
    CanonicalIREdge,
    CanonicalIRMismatch,
    CanonicalIROutput,
    CanonicalIRValidationOutput,
    CanonicalFileNode,
    CanonicalRequirement,
    CanonicalSurfaceNode,
    CanonicalWorkPackage,
    EvidenceContract,
    PaperBenchReproState,
    SemanticAssertion,
    ValidatorExpectation,
)

_ARTIFACT_ROOT_PREFIXES = (
    "result/",
    "results/",
    "output/",
    "outputs/",
    "artifact/",
    "artifacts/",
    "report/",
    "reports/",
    "figure/",
    "figures/",
    "plot/",
    "plots/",
    "checkpoint/",
    "checkpoints/",
    "submission/",
    "submissions/",
    "prediction/",
    "predictions/",
    "logs/",
)

_ARTIFACT_FILE_SUFFIXES = (
    ".csv",
    ".tsv",
    ".jsonl",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".html",
)

_ARTIFACT_ENCODED_SUFFIXES = (
    "json",
    "jsonl",
    "csv",
    "tsv",
    "parquet",
    "npy",
    "npz",
    "pkl",
    "pickle",
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "html",
)

_GENERIC_SEMANTIC_TOKENS = {
    "implement",
    "implementation",
    "method",
    "model",
    "training",
    "evaluation",
    "config",
    "artifact",
    "result",
    "results",
    "metric",
    "metrics",
    "baseline",
    "paper",
    "chunk",
    "requirement",
    "实现",
    "支持",
    "代码",
    "训练",
    "评估",
    "模型",
    "方法",
    "配置",
    "参数",
    "路径",
    "指标",
    "结果",
    "实验",
    "接口",
}


def _dedupe(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _sanitize_id(value: str) -> str:
    return str(value or "").strip().replace("/", "_").replace(".", "_").replace("-", "_")


def _normalize_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _looks_like_repo_relative_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if not normalized or normalized.endswith("/") or " " in normalized:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-")
    return all(char in allowed for char in normalized)


def _looks_like_implementation_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if not normalized or normalized.endswith("/"):
        return False
    name = normalized.rsplit("/", 1)[-1].lower()
    if name in {"readme.md", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}:
        return True
    if "." not in name:
        return normalized in {"main", "src", "configs"}
    suffix = name.rsplit(".", 1)[-1]
    return suffix in {"py", "md", "txt", "toml", "yaml", "yml", "json", "ini", "cfg", "sh"}


def _looks_like_encoded_artifact_name(path: str) -> bool:
    normalized = _normalize_path(path)
    name = normalized.rsplit("/", 1)[-1].lower()
    if not name:
        return False
    parts = [part for part in name.replace("-", "_").split("_") if part]
    if len(parts) < 2:
        return name in {"results", "outputs", "metrics", "predictions", "figures", "plots", "checkpoints"}
    if parts[0] in {"result", "results", "output", "outputs", "prediction", "predictions", "figure", "figures", "plot", "plots", "metric", "metrics", "checkpoint", "checkpoints"}:
        return True
    if parts[-1] in _ARTIFACT_ENCODED_SUFFIXES and any(
        part in {"result", "results", "output", "outputs", "prediction", "predictions", "figure", "figures", "plot", "plots", "metric", "metrics", "table", "tables", "summary", "checkpoint", "checkpoints"}
        for part in parts[:-1]
    ):
        return True
    return False


def _looks_like_artifact_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(token in lowered for token in ("(", ")", "->", "=", ",")):
        return False
    if any(lowered == prefix.rstrip("/") or lowered.startswith(prefix) for prefix in _ARTIFACT_ROOT_PREFIXES):
        return True
    if _looks_like_encoded_artifact_name(normalized):
        return True
    if any(char in lowered for char in "*[]{}") and "/" in lowered:
        return True
    name = lowered.rsplit("/", 1)[-1]
    if name.endswith(_ARTIFACT_FILE_SUFFIXES):
        return True
    if name.endswith(".json"):
        stem = name.rsplit(".", 1)[0]
        if stem in {"metrics", "results", "summary", "trend_summary", "experiment_plan", "ablation_matrix", "fidelity_scores", "critical_segments", "refinement_curves"}:
            return True
        if any(token in stem for token in ("prediction", "checkpoint")):
            return True
    if name in {"metrics", "results", "summary", "figures", "plots", "checkpoints", "predictions"}:
        return True
    if _looks_like_implementation_path(normalized):
        return False
    return bool(
        "/" in normalized
        and any(token in normalized.lower() for token in ("result", "artifact", "metric", "report", "prediction", "output"))
    )


def _looks_like_contract_output_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(token in lowered for token in ("(", ")", "->", "=", ",")):
        return False
    return (
        _looks_like_artifact_path(normalized)
        or lowered.startswith(_ARTIFACT_ROOT_PREFIXES)
        or _looks_like_encoded_artifact_name(normalized)
        or lowered.endswith(_ARTIFACT_FILE_SUFFIXES)
        or any(token in lowered for token in ("result/", "results_", "prediction", "output/"))
    )


def _normalized_artifact_paths(paths: list[str] | None) -> list[str]:
    return _dedupe(
        [
            normalized
            for path in list(paths or [])
            for normalized in [_normalize_path(path)]
            if _looks_like_repo_relative_path(normalized) and _looks_like_contract_output_path(normalized)
        ]
    )


def _unresolved_result_targets_for_index(
    state: PaperBenchReproState,
    artifact_producer_by_path: dict[str, str],
    artifact_owner_by_path: dict[str, str],
    fallback_owner_file_path,
) -> list[dict[str, object]]:
    unresolved: list[dict[str, object]] = []
    for target in list(state.global_contract.result_targets if state.global_contract else []):
        target_artifacts = _normalized_artifact_paths(list(target.artifact_paths or []))
        if not target_artifacts:
            continue
        required_inputs = [
            _normalize_path(path)
            for path in list(target.required_inputs or [])
            if _normalize_path(path)
        ]
        if not required_inputs:
            required_inputs = _dedupe([artifact_producer_by_path.get(path, "") for path in target_artifacts])
        owner_work_packages = [
            str(item).strip()
            for item in list(target.owner_work_packages or [])
            if str(item).strip()
        ]
        if not owner_work_packages:
            owner_work_packages = _dedupe([artifact_owner_by_path.get(path, "") for path in target_artifacts])
        has_artifact_writer = any(artifact_producer_by_path.get(path, "") for path in target_artifacts)
        has_owner_producer = any(
            fallback_owner_file_path(owner_work_package_id, required_inputs)
            for owner_work_package_id in owner_work_packages
        )
        if has_artifact_writer or has_owner_producer:
            continue
        unresolved.append(
            {
                "target_id": str(target.target_id or "").strip(),
                "owner_work_package": next((owner for owner in owner_work_packages if owner), ""),
                "artifact_paths": target_artifacts,
            }
        )
    return unresolved


def _work_package_by_id(state: PaperBenchReproState) -> dict[str, object]:
    return {
        item.work_package_id: item
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(item.work_package_id or "").strip()
    }


def _known_work_package_ids(state: PaperBenchReproState) -> set[str]:
    return {
        str(item.work_package_id or "").strip()
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(item.work_package_id or "").strip()
    }


def _known_work_package_ids_ordered(state: PaperBenchReproState) -> list[str]:
    return _dedupe(
        [
            str(item.work_package_id or "").strip()
            for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
            if str(item.work_package_id or "").strip()
        ]
    )


def _tokenize_owner_text(value: str) -> set[str]:
    return {
        item
        for item in re.split(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", str(value or "").lower())
        if len(item) >= 2
    }


def _tokenize_semantic_text(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").lower().replace("-", "_")
        tokens.update(
            span
            for span in re.findall(r"[a-z0-9]+", text)
            if len(span) >= 2
        )
        for cjk_span in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(cjk_span) >= 2:
                tokens.add(cjk_span)
                for ngram_size in (2, 3, 4):
                    if len(cjk_span) >= ngram_size:
                        tokens.update(
                            cjk_span[index : index + ngram_size]
                            for index in range(0, len(cjk_span) - ngram_size + 1)
                        )
        for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", text):
            for part in token.replace("/", "_").split("_"):
                part = part.strip(".,;:()[]{}'\"`")
                if len(part) >= 2 and not re.fullmatch(r"[\u4e00-\u9fff]+", part):
                    tokens.add(part)
    return tokens


def _semantic_key_terms(*values: object) -> set[str]:
    return {
        token
        for token in _tokenize_semantic_text(*values)
        if token not in _GENERIC_SEMANTIC_TOKENS
        and (len(token) >= 3 or re.fullmatch(r"[\u4e00-\u9fff]{2}", token))
        and not token.isdigit()
        and not token.startswith("chunk_")
    }


def _unit_semantic_values(unit: object) -> list[str]:
    values: list[str] = []
    for attr in (
        "unit_id",
        "statement",
        "hypothesis",
        "decision_value",
        "stop_rule_or_pruning_rationale",
    ):
        value = str(getattr(unit, attr, "") or "").strip()
        if value:
            values.append(value)
    for attr in (
        "paper_evidence",
        "source_paragraph_ids",
        "citation_refs",
        "implementation_surfaces",
        "code_obligations",
        "runtime_interfaces",
        "expected_artifacts",
        "suggested_module_kinds",
        "implementation_notes",
    ):
        values.extend(str(item).strip() for item in list(getattr(unit, attr, []) or []) if str(item).strip())
    return values


def _unit_lookup(state: PaperBenchReproState) -> tuple[dict[str, object], dict[str, list[object]], list[object]]:
    units = [
        unit
        for unit in list(state.unit_extraction.units if state.unit_extraction else [])
        if str(getattr(unit, "unit_id", "") or "").strip()
    ]
    by_unit_id = {str(unit.unit_id or "").strip(): unit for unit in units}
    by_source_id: dict[str, list[object]] = {}
    for unit in units:
        source_ids = _dedupe(
            [
                str(unit.unit_id or "").strip(),
                *[str(item).strip() for item in list(unit.source_paragraph_ids or []) if str(item).strip()],
                *[str(item).strip() for item in list(unit.citation_refs or []) if str(item).strip()],
            ]
        )
        for source_id in source_ids:
            by_source_id.setdefault(source_id, []).append(unit)
    return by_unit_id, by_source_id, units


def _resolve_requirement_units(
    state: PaperBenchReproState,
    requirement: object,
    *,
    source_unit_ids: list[str],
) -> list[object]:
    by_unit_id, by_source_id, units = _unit_lookup(state)
    source_id_set = set(source_unit_ids)
    key_terms = _semantic_key_terms(
        getattr(requirement, "title", ""),
        getattr(requirement, "scope", ""),
        getattr(requirement, "description", ""),
        *list(getattr(requirement, "acceptance_criteria", []) or []),
    )
    requirement_terms = _tokenize_semantic_text(
        getattr(requirement, "requirement_id", ""),
        getattr(requirement, "title", ""),
        getattr(requirement, "scope", ""),
        getattr(requirement, "description", ""),
        *list(getattr(requirement, "acceptance_criteria", []) or []),
        *source_unit_ids,
    )
    unit_terms_by_id: dict[str, set[str]] = {}
    for unit in units:
        unit_id = str(getattr(unit, "unit_id", "") or "").strip()
        if unit_id:
            unit_terms_by_id[unit_id] = _tokenize_semantic_text(*_unit_semantic_values(unit))
    key_doc_frequency = Counter(
        term
        for terms in unit_terms_by_id.values()
        for term in set(terms) & key_terms
    )

    def key_term_weight(term: str) -> float:
        frequency = max(int(key_doc_frequency.get(term, 0) or 0), 1)
        rarity = 1.0 / float(frequency)
        lexical_bonus = 1.5 if re.search(r"[a-z0-9]", term) else min(max(len(term), 2), 8) / 4.0
        return rarity * lexical_bonus

    ranked: list[tuple[float, int, int, str, object]] = []
    for unit in units:
        unit_id = str(getattr(unit, "unit_id", "") or "").strip()
        if not unit_id:
            continue
        unit_source_ids = {
            unit_id,
            *[str(item).strip() for item in list(getattr(unit, "source_paragraph_ids", []) or []) if str(item).strip()],
            *[str(item).strip() for item in list(getattr(unit, "citation_refs", []) or []) if str(item).strip()],
        }
        unit_terms = unit_terms_by_id.get(unit_id, set())
        overlap = len(requirement_terms & unit_terms)
        key_overlap_terms = key_terms & unit_terms
        key_overlap = len(key_overlap_terms)
        weighted_key_overlap = sum(key_term_weight(term) for term in key_overlap_terms)
        source_score = 80 if source_id_set.intersection(unit_source_ids) else 0
        if unit_id in by_unit_id and unit_id in source_id_set:
            source_score = 120
        specificity = 0 if unit_id.startswith("unit_") else 1 if unit_id.startswith("paper_semantic_chunk_") else 2
        score = source_score + min(overlap, 8) + (weighted_key_overlap * 24.0) - (specificity * 8)
        if score > 0:
            ranked.append((score, key_overlap, specificity, unit_id, unit))
    ranked.sort(key=lambda item: (-item[0], item[2], -item[1], item[3]))
    return [unit for _score, _overlap, _specificity, _unit_id, unit in ranked[:8]]


def _requirement_semantic_projection(
    state: PaperBenchReproState,
    requirement: object,
    *,
    mapped_unit_ids: list[str],
) -> dict[str, object]:
    title = str(getattr(requirement, "title", "") or "").strip()
    description = str(getattr(requirement, "description", "") or "").strip()
    scope = str(getattr(requirement, "scope", "") or "").strip()
    criteria = _dedupe([str(item) for item in list(getattr(requirement, "acceptance_criteria", []) or [])])
    source_unit_ids = _dedupe(
        [
            str(item).strip()
            for item in list(getattr(requirement, "source_unit_ids", []) or []) + list(mapped_unit_ids or [])
            if str(item).strip()
        ]
    )
    units = _resolve_requirement_units(state, requirement, source_unit_ids=source_unit_ids)
    resolved_unit_ids = _dedupe(
        [str(getattr(unit, "unit_id", "") or "").strip() for unit in units if str(getattr(unit, "unit_id", "") or "").strip()]
    )
    unit_obligations = _dedupe(
        [
            str(item).strip()
            for unit in units
            for item in [
                getattr(unit, "statement", ""),
                *list(getattr(unit, "code_obligations", []) or [])[:4],
                *list(getattr(unit, "runtime_interfaces", []) or [])[:3],
                *list(getattr(unit, "expected_artifacts", []) or [])[:3],
            ]
            if str(item).strip()
        ]
    )
    requirement_id = str(getattr(requirement, "requirement_id", "") or "").strip()
    fallback_title = title or description or (unit_obligations[0] if unit_obligations else requirement_id)
    combined_description = _dedupe(
        [
            description,
            f"Scope: {scope}" if scope else "",
            *unit_obligations[:8],
        ]
    )
    acceptance_criteria = _dedupe([*criteria, *unit_obligations[:8]])
    if not acceptance_criteria and fallback_title:
        acceptance_criteria = [fallback_title]
    statement_parts = _dedupe(
        [
            fallback_title,
            description,
            f"Scope: {scope}" if scope else "",
            *criteria[:6],
            *unit_obligations[:8],
        ]
    )
    return {
        "title": fallback_title[:240],
        "description": "\n".join(combined_description)[:3000],
        "acceptance_criteria": acceptance_criteria[:16],
        "semantic_statement": "\n".join(statement_parts)[:3000],
        "source_unit_ids": resolved_unit_ids or source_unit_ids,
    }


def _requirement_ids_for_work_package(state: PaperBenchReproState, work_package_id: str) -> list[str]:
    valid_requirement_ids = {
        item.requirement_id
        for item in list(state.boundary_requirements.boundary_requirements if state.boundary_requirements else [])
        if str(item.requirement_id or "").strip()
    }
    pipeline_requirements = {
        requirement_id
        for item in list(state.pipeline_plan.plan_nodes if state.pipeline_plan else [])
        for requirement_id in list(item.requirement_ids or [])
        if str(requirement_id or "").strip()
    }
    contract_requirements = set()
    for item in list(state.global_contract.work_package_contracts if state.global_contract else []):
        if item.work_package_id != work_package_id:
            continue
        contract_requirements.update(
            requirement_id
            for requirement_id in list(item.requirement_ids or [])
            if str(requirement_id or "").strip()
        )
    requirement_ids = contract_requirements or pipeline_requirements or valid_requirement_ids
    return [item for item in _dedupe(list(requirement_ids)) if item in valid_requirement_ids]


def _file_owner_resolution(
    state: PaperBenchReproState,
    canonical_path: str,
    explicit_owner: str = "",
    related_node_ids: list[str] | None = None,
    purpose: str = "",
) -> tuple[str, str]:
    known_work_packages = set(_known_work_package_ids_ordered(state))
    ordered_work_packages = _known_work_package_ids_ordered(state)
    explicit = str(explicit_owner or "").strip()
    if explicit and explicit in known_work_packages:
        return explicit, "explicit_file_plan_owner"

    candidate_sources: list[list[str]] = []

    produced_candidates = [
        str(item.work_package_id or "").strip()
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(item.work_package_id or "").strip()
        and canonical_path in {
            _normalize_path(path)
            for path in list(item.produces or [])
            if _normalize_path(path)
        }
    ]
    candidate_sources.append(_dedupe([item for item in produced_candidates if item in known_work_packages]))

    inventory_candidates: list[str] = []
    for inventory in dict(state.global_contract.inventory_owners or {}).values() if state.global_contract else []:
        if not isinstance(inventory, dict):
            continue
        for work_package_id, owned_paths in inventory.items():
            normalized_owned = {
                _normalize_path(path)
                for path in list(owned_paths or [])
                if _normalize_path(path)
            }
            if canonical_path in normalized_owned and str(work_package_id or "").strip() in known_work_packages:
                inventory_candidates.append(str(work_package_id or "").strip())
    candidate_sources.append(_dedupe(inventory_candidates))

    result_target_candidates: list[str] = []
    for target in list(state.global_contract.result_targets if state.global_contract else []):
        required_inputs = {
            _normalize_path(path)
            for path in list(target.required_inputs or [])
            if _normalize_path(path)
        }
        artifact_paths = set(_normalized_artifact_paths(list(target.artifact_paths or [])))
        if canonical_path not in required_inputs and canonical_path not in artifact_paths:
            continue
        result_target_candidates.extend(
            str(item).strip()
            for item in list(target.owner_work_packages or [])
            if str(item).strip() in known_work_packages
        )
    candidate_sources.append(_dedupe(result_target_candidates))

    package_layout = dict(state.architecture.package_layout if state.architecture else {})
    layout_candidates = [
        str(work_package_id or "").strip()
        for work_package_id, paths in package_layout.items()
        if str(work_package_id or "").strip() in known_work_packages
        and canonical_path in {
            _normalize_path(path)
            for path in list(paths or [])
            if _normalize_path(path)
        }
    ]
    candidate_sources.append(_dedupe(layout_candidates))

    related_requirement_ids: list[str] = []
    plan_node_by_id = {
        str(item.node_id or "").strip(): item
        for item in list(state.pipeline_plan.plan_nodes if state.pipeline_plan else [])
        if str(item.node_id or "").strip()
    }
    for node_id in list(related_node_ids or []):
        node = plan_node_by_id.get(str(node_id or "").strip())
        if node is not None:
            related_requirement_ids.extend(list(node.requirement_ids or []))
    requirement_candidates = [
        str(item.work_package_id or "").strip()
        for item in list(state.work_package_planning.work_packages if state.work_package_planning else [])
        if str(item.work_package_id or "").strip() in known_work_packages
        and set(_requirement_ids_for_work_package(state, str(item.work_package_id or "").strip()))
        & set(_dedupe(related_requirement_ids))
    ]
    candidate_sources.append(_dedupe(requirement_candidates))

    for source_index, candidates in enumerate(candidate_sources):
        if len(candidates) == 1:
            return candidates[0], (
                "work_package_produces"
                if source_index == 0
                else "global_inventory_owner"
                if source_index == 1
                else "global_result_target"
                if source_index == 2
                else "architecture_package_layout"
                if source_index == 3
                else "related_requirement_ids"
            )

    if ordered_work_packages:
        path_tokens = _tokenize_owner_text(canonical_path)
        purpose_tokens = _tokenize_owner_text(purpose)
        scored: list[tuple[int, str]] = []
        for work_package in list(state.work_package_planning.work_packages if state.work_package_planning else []):
            work_package_id = str(work_package.work_package_id or "").strip()
            if work_package_id not in known_work_packages:
                continue
            owner_text = " ".join(
                [
                    work_package_id,
                    str(work_package.goal or ""),
                    " ".join(list(work_package.produces or [])),
                    " ".join(list(work_package.interface_contract or [])),
                    " ".join(list(work_package.method_obligations or [])),
                    " ".join(list(work_package.tags or [])),
                ]
            )
            owner_tokens = _tokenize_owner_text(owner_text)
            score = len(path_tokens & owner_tokens) * 3 + len(purpose_tokens & owner_tokens)
            if score > 0:
                scored.append((score, work_package_id))
        if scored:
            best_score = max(score for score, _ in scored)
            best_ids = [work_package_id for score, work_package_id in scored if score == best_score]
            for work_package_id in ordered_work_packages:
                if work_package_id in best_ids:
                    return work_package_id, "semantic_owner_score"

    if len(ordered_work_packages) == 1:
        return ordered_work_packages[0], "single_canonical_work_package"
    if ordered_work_packages:
        return ordered_work_packages[0], "deterministic_first_canonical_work_package"
    return "", "unresolved"


def _file_owner_from_provenance(state: PaperBenchReproState, canonical_path: str, explicit_owner: str = "") -> str:
    owner, _reason = _file_owner_resolution(state, canonical_path, explicit_owner=explicit_owner)
    return owner


def build_canonical_ir(state: PaperBenchReproState) -> CanonicalIROutput:
    """Build the stage-A shadow canonical IR from existing planning artifacts."""

    requirements: list[CanonicalRequirement] = []
    work_packages: list[CanonicalWorkPackage] = []
    contract_stages: list[CanonicalContractStage] = []
    file_nodes: list[CanonicalFileNode] = []
    surface_nodes: list[CanonicalSurfaceNode] = []
    edges: list[CanonicalIREdge] = []
    semantic_assertions: list[SemanticAssertion] = []
    evidence_contracts: list[EvidenceContract] = []
    validator_expectations: list[ValidatorExpectation] = []

    work_package_map = _work_package_by_id(state)
    source_unit_ids = _dedupe(
        [
            item.unit_id
            for item in list(state.unit_extraction.units if state.unit_extraction else [])
            if str(item.unit_id or "").strip()
        ]
    )
    synthesized_requirement_ids_by_work_package: dict[str, list[str]] = {}

    for requirement in list(state.boundary_requirements.boundary_requirements if state.boundary_requirements else []):
        requirement_id = str(requirement.requirement_id or "").strip()
        if not requirement_id:
            continue
        exact_work_package_units = _dedupe(
            [
                unit_id
                for work_package in work_package_map.values()
                for unit_id in list(getattr(work_package, "owned_unit_ids", []) or [])
                if _requirement_ids_for_work_package(state, str(getattr(work_package, "work_package_id", "") or "")) == [requirement_id]
            ]
        )
        semantic_projection = _requirement_semantic_projection(
            state,
            requirement,
            mapped_unit_ids=exact_work_package_units,
        )
        mapped_unit_ids = list(semantic_projection["source_unit_ids"]) or exact_work_package_units or list(source_unit_ids)
        requirements.append(
            CanonicalRequirement(
                requirement_id=requirement_id,
                title=str(semantic_projection["title"]),
                category=str(requirement.category or "experiment").strip(),
                description=str(semantic_projection["description"]),
                source_unit_ids=mapped_unit_ids,
                acceptance_criteria=list(semantic_projection["acceptance_criteria"]),
            )
        )

        assertion_id = f"assertion_{_sanitize_id(requirement_id)}_semantic_requirement"
        evidence_id = f"evidence_{_sanitize_id(requirement_id)}"
        expectation_id = f"expectation_{_sanitize_id(requirement_id)}"
        semantic_assertions.append(
            SemanticAssertion(
                assertion_id=assertion_id,
                requirement_id=requirement_id,
                assertion_type="semantic_requirement",
                statement=str(semantic_projection["semantic_statement"]),
                status="unknown",
            )
        )
        evidence_contracts.append(
            EvidenceContract(
                evidence_contract_id=evidence_id,
                assertion_id=assertion_id,
                evidence_kind="file_presence",
                canonical_paths=[],
                owner_work_package_ids=[],
            )
        )
        validator_expectations.append(
            ValidatorExpectation(
                expectation_id=expectation_id,
                assertion_id=assertion_id,
                expectation_kind="semantic_evidence_check",
                evidence_contract_id=evidence_id,
                pass_condition="active implementation surfaces preserve the requirement statement and acceptance criteria",
                status="unknown",
            )
        )

    if not requirements:
        for index, work_package in enumerate(
            list(state.work_package_planning.work_packages if state.work_package_planning else []),
            start=1,
        ):
            work_package_id = str(work_package.work_package_id or "").strip()
            if not work_package_id:
                continue
            requirement_id = f"req_{_sanitize_id(work_package_id)}"
            title = str(work_package.goal or work_package_id).strip()
            unit_ids = _dedupe(list(getattr(work_package, "owned_unit_ids", []) or [])) or list(source_unit_ids)
            criteria = _dedupe(
                [
                    *[str(item) for item in list(getattr(work_package, "method_obligations", []) or [])[:4]],
                    *[str(item) for item in list(getattr(work_package, "evidence_needs", []) or [])[:4]],
                ]
            )
            if not criteria and title:
                criteria = [title]
            requirements.append(
                CanonicalRequirement(
                    requirement_id=requirement_id,
                    title=title[:160] or f"Recovered paper requirement {index}",
                    category="experiment",
                    description=title,
                    source_unit_ids=unit_ids,
                    acceptance_criteria=criteria,
                )
            )
            synthesized_requirement_ids_by_work_package[work_package_id] = [requirement_id]
            assertion_id = f"assertion_{_sanitize_id(requirement_id)}_semantic_requirement"
            evidence_id = f"evidence_{_sanitize_id(requirement_id)}"
            expectation_id = f"expectation_{_sanitize_id(requirement_id)}"
            semantic_assertions.append(
                SemanticAssertion(
                    assertion_id=assertion_id,
                    requirement_id=requirement_id,
                    assertion_type="semantic_requirement",
                    statement=title or requirement_id,
                    status="unknown",
                )
            )
            evidence_contracts.append(
                EvidenceContract(
                    evidence_contract_id=evidence_id,
                    assertion_id=assertion_id,
                    evidence_kind="file_presence",
                    canonical_paths=[],
                    owner_work_package_ids=[],
                )
            )
            validator_expectations.append(
                ValidatorExpectation(
                    expectation_id=expectation_id,
                    assertion_id=assertion_id,
                    expectation_kind="semantic_evidence_check",
                    evidence_contract_id=evidence_id,
                    pass_condition="active implementation surfaces preserve the recovered work-package requirement",
                    status="unknown",
                )
            )

    for work_package in list(state.work_package_planning.work_packages if state.work_package_planning else []):
        work_package_id = str(work_package.work_package_id or "").strip()
        if not work_package_id:
            continue
        requirement_ids = (
            _requirement_ids_for_work_package(state, work_package_id)
            or synthesized_requirement_ids_by_work_package.get(work_package_id, [])
        )
        work_packages.append(
            CanonicalWorkPackage(
                work_package_id=work_package_id,
                goal=str(work_package.goal or "").strip(),
                requirement_ids=requirement_ids,
                source_unit_ids=_dedupe(list(work_package.owned_unit_ids or [])),
                produces=_dedupe([_normalize_path(path) for path in list(work_package.produces or []) if _normalize_path(path)]),
                depends_on=_dedupe(list(work_package.depends_on or [])),
            )
        )
        for requirement_id in requirement_ids:
            edges.append(
                CanonicalIREdge(
                    edge_id=f"edge_req_{_sanitize_id(requirement_id)}_wp_{_sanitize_id(work_package_id)}",
                    edge_type="req_to_wp",
                    source_id=requirement_id,
                    target_id=work_package_id,
                )
            )

    for stage_name in list(state.global_contract.canonical_stage_sequence if state.global_contract else []):
        stage_id = f"stage_{_sanitize_id(str(stage_name))}"
        contract_stages.append(
            CanonicalContractStage(
                stage_id=stage_id,
                label=str(stage_name or "").strip(),
                owner_work_package_ids=_dedupe(
                    [
                        item.work_package_id
                        for item in list(state.global_contract.work_package_contracts if state.global_contract else [])
                        if str(item.work_package_id or "").strip()
                    ]
                ),
                result_target_ids=_dedupe(
                    [
                        item.target_id
                        for item in list(state.global_contract.result_targets if state.global_contract else [])
                        if str(item.target_id or "").strip()
                    ]
                ),
            )
        )

    owner_resolution_fallbacks: list[dict[str, str]] = []
    for file_plan in list(state.package_file_planning_output.file_plans if state.package_file_planning_output else []):
        canonical_path = _normalize_path(file_plan.target_file)
        if not canonical_path:
            continue
        file_id = f"file_{_sanitize_id(canonical_path)}"
        blueprint = next(
            (
                item
                for item in list(state.architecture.file_blueprints if state.architecture else [])
                if _normalize_path(item.path) == canonical_path
            ),
            None,
        )
        related_node_ids = _dedupe(
            list(file_plan.related_node_ids or [])
            + list(blueprint.related_node_ids if blueprint is not None else [])
        )
        owner_work_package_id, owner_resolution_reason = _file_owner_resolution(
            state,
            canonical_path,
            explicit_owner=str(file_plan.work_package_id or "").strip(),
            related_node_ids=related_node_ids,
            purpose=" ".join(
                [
                    str(file_plan.purpose or ""),
                    str(blueprint.purpose if blueprint is not None else ""),
                ]
            ),
        )
        if owner_resolution_reason in {
            "semantic_owner_score",
            "single_canonical_work_package",
            "deterministic_first_canonical_work_package",
            "unresolved",
        }:
            owner_resolution_fallbacks.append(
                {
                    "canonical_path": canonical_path,
                    "owner_work_package_id": owner_work_package_id,
                    "reason": owner_resolution_reason,
                }
            )
        related_requirement_ids = _requirement_ids_for_work_package(state, owner_work_package_id) if owner_work_package_id else []
        file_surface_ids: list[str] = []
        file_nodes.append(
            CanonicalFileNode(
                file_id=file_id,
                canonical_path=canonical_path,
                owner_work_package_id=owner_work_package_id,
                contract_stage_ids=[item.stage_id for item in contract_stages],
                related_requirement_ids=related_requirement_ids,
                related_plan_node_ids=related_node_ids,
                surfaces=file_surface_ids,
            )
        )
        if owner_work_package_id:
            edges.append(
                CanonicalIREdge(
                    edge_id=f"edge_wp_{_sanitize_id(owner_work_package_id)}_file_{_sanitize_id(canonical_path)}",
                    edge_type="wp_to_file",
                    source_id=owner_work_package_id,
                    target_id=file_id,
                )
            )

    file_node_by_path = {item.canonical_path: item for item in file_nodes}
    file_node_by_id = {item.file_id: item for item in file_nodes}
    registered_surface_keys: set[tuple[str, str, str, str]] = set()

    def _register_surface(surface_kind: str, path: str, *, owner_file_path: str = "", owner_work_package_id: str = "") -> None:
        canonical_path = _normalize_path(path)
        if not canonical_path:
            return
        owner_path = _normalize_path(owner_file_path) or canonical_path
        owner_file = file_node_by_path.get(owner_path)
        if surface_kind in {"entrypoint", "config", "stable_interface"} and canonical_path not in file_node_by_path:
            return
        if surface_kind in {"artifact", "producer"} and owner_file is None:
            return
        resolved_owner = owner_work_package_id or (owner_file.owner_work_package_id if owner_file is not None else "")
        surface_key = (
            surface_kind,
            canonical_path,
            owner_file.file_id if owner_file is not None and surface_kind in {"artifact", "producer"} else "",
            resolved_owner if surface_kind in {"artifact", "producer"} else "",
        )
        if surface_key in registered_surface_keys:
            return
        registered_surface_keys.add(surface_key)
        owner_suffix = f"_{_sanitize_id(resolved_owner)}" if surface_kind in {"artifact", "producer"} and resolved_owner else ""
        surface_id = f"surface_{surface_kind}_{_sanitize_id(canonical_path)}{owner_suffix}"
        expectation_ids = [
            item.expectation_id
            for item in validator_expectations
            if item.assertion_id.startswith("assertion_")
        ]
        surface_nodes.append(
            CanonicalSurfaceNode(
                surface_id=surface_id,
                surface_kind=surface_kind,
                canonical_path=canonical_path,
                owner_file_id=owner_file.file_id if owner_file is not None else "",
                owner_work_package_id=resolved_owner,
                validator_expectation_ids=expectation_ids[:1],
            )
        )
        if owner_file is not None:
            owner_file.surfaces = _dedupe(list(owner_file.surfaces) + [surface_id])
            edges.append(
                CanonicalIREdge(
                    edge_id=f"edge_file_{_sanitize_id(owner_file.file_id)}_surface_{_sanitize_id(surface_id)}",
                    edge_type="file_to_surface",
                    source_id=owner_file.file_id,
                    target_id=surface_id,
                )
            )

    for path in list(state.architecture.execution_entrypoints if state.architecture else []):
        _register_surface("entrypoint", path)
    if not any(item.surface_kind == "entrypoint" for item in surface_nodes):
        fallback_entrypoint = next(
            (
                item.canonical_path
                for item in file_nodes
                if item.canonical_path.endswith("main.py")
            ),
            next(
                (
                    item.canonical_path
                    for item in file_nodes
                    if item.canonical_path.endswith(".py")
                ),
                "",
            ),
        )
        if fallback_entrypoint:
            _register_surface("entrypoint", fallback_entrypoint)
    for path in list(state.architecture.config_surfaces if state.architecture else []):
        _register_surface("config", path)
    if not any(item.surface_kind == "config" for item in surface_nodes):
        for item in file_nodes:
            lowered = item.canonical_path.lower()
            if lowered.endswith((".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")) or "config" in lowered:
                _register_surface("config", item.canonical_path)
    for path in list(state.architecture.stable_interfaces if state.architecture else []):
        _register_surface("stable_interface", path)
    if not any(item.surface_kind == "stable_interface" for item in surface_nodes):
        for item in file_nodes:
            if item.canonical_path.endswith(".py"):
                _register_surface("stable_interface", item.canonical_path)
                break

    def _fallback_owner_file_path(owner_work_package_id: str, required_inputs: list[str] | None = None) -> str:
        preferred_inputs = [
            _normalize_path(path)
            for path in list(required_inputs or [])
            if _normalize_path(path) in file_node_by_path
        ]
        if preferred_inputs:
            return preferred_inputs[0]
        owned_paths = [
            item.canonical_path
            for item in file_nodes
            if str(item.owner_work_package_id or "").strip() == str(owner_work_package_id or "").strip()
        ]
        python_owned = [path for path in owned_paths if path.endswith(".py")]
        if python_owned:
            return python_owned[0]
        if owned_paths:
            return owned_paths[0]
        surface_owner_paths = [
            surface.canonical_path
            for surface in surface_nodes
            if surface.surface_kind == "producer"
            and str(surface.owner_work_package_id or "").strip() == str(owner_work_package_id or "").strip()
            and _normalize_path(surface.canonical_path) in file_node_by_path
        ]
        if surface_owner_paths:
            return surface_owner_paths[0]
        artifact_owner_paths = [
            file_node_by_id[surface.owner_file_id].canonical_path
            for surface in surface_nodes
            if surface.surface_kind == "artifact"
            and str(surface.owner_work_package_id or "").strip() == str(owner_work_package_id or "").strip()
            and surface.owner_file_id in file_node_by_id
        ]
        if artifact_owner_paths:
            return artifact_owner_paths[0]
        return ""

    artifact_producer_by_path: dict[str, str] = {}
    artifact_owner_by_path: dict[str, str] = {}
    for file_plan in list(state.package_file_planning_output.file_plans if state.package_file_planning_output else []):
        for artifact_path in _normalized_artifact_paths(list(file_plan.writes_artifacts or [])):
            _register_surface("artifact", artifact_path, owner_file_path=file_plan.target_file, owner_work_package_id=file_plan.work_package_id)
            _register_surface("producer", file_plan.target_file, owner_file_path=file_plan.target_file, owner_work_package_id=file_plan.work_package_id)
            artifact_producer_by_path.setdefault(artifact_path, _normalize_path(file_plan.target_file))
            artifact_owner_by_path.setdefault(artifact_path, str(file_plan.work_package_id or "").strip())
    for target in list(state.global_contract.result_targets if state.global_contract else []):
        owner_work_packages = [
            str(item).strip()
            for item in list(target.owner_work_packages or [])
            if str(item).strip()
        ]
        required_inputs = [
            _normalize_path(path)
            for path in list(target.required_inputs or [])
            if _normalize_path(path)
        ]
        target_artifacts = _normalized_artifact_paths(list(target.artifact_paths or []))
        if not owner_work_packages:
            owner_work_packages = _dedupe([artifact_owner_by_path.get(path, "") for path in target_artifacts])
        if not required_inputs:
            required_inputs = _dedupe([artifact_producer_by_path.get(path, "") for path in target_artifacts])
        for owner_work_package_id in owner_work_packages:
            owner_file_path = _fallback_owner_file_path(owner_work_package_id, required_inputs)
            if owner_file_path:
                _register_surface(
                    "producer",
                    owner_file_path,
                    owner_file_path=owner_file_path,
                    owner_work_package_id=owner_work_package_id,
                )
            for artifact_path in target_artifacts:
                _register_surface(
                    "artifact",
                    artifact_path,
                    owner_file_path=owner_file_path,
                    owner_work_package_id=owner_work_package_id,
                )

    evidence_owner_map: dict[str, list[str]] = {}
    evidence_path_map: dict[str, list[str]] = {}
    for work_package in work_packages:
        for requirement_id in work_package.requirement_ids:
            evidence_owner_map.setdefault(requirement_id, [])
            evidence_owner_map[requirement_id].append(work_package.work_package_id)
    file_plan_by_work_package: dict[str, list[object]] = {}
    for file_plan in list(state.package_file_planning_output.file_plans if state.package_file_planning_output else []):
        work_package_id = str(file_plan.work_package_id or "").strip()
        if work_package_id:
            file_plan_by_work_package.setdefault(work_package_id, []).append(file_plan)
    for work_package in work_packages:
        artifact_paths: list[str] = []
        for file_plan in file_plan_by_work_package.get(work_package.work_package_id, []):
            artifact_paths.extend(_normalized_artifact_paths(list(file_plan.writes_artifacts or [])))
        if not artifact_paths:
            artifact_paths.extend(_normalized_artifact_paths(list(work_package.produces or [])))
        for requirement_id in work_package.requirement_ids:
            evidence_path_map.setdefault(requirement_id, [])
            evidence_path_map[requirement_id].extend(artifact_paths)

    for evidence_contract in evidence_contracts:
        assertion_key = (
            evidence_contract.assertion_id
            .replace("assertion_", "", 1)
            .replace("_static_presence", "", 1)
            .replace("_semantic_requirement", "", 1)
        )
        requirement_id = assertion_key
        requirement_id = requirement_id.replace("_", "-") if requirement_id not in evidence_owner_map else requirement_id
        for requirement in requirements:
            if _sanitize_id(requirement.requirement_id) != assertion_key:
                continue
            evidence_contract.owner_work_package_ids = _dedupe(evidence_owner_map.get(requirement.requirement_id, []))
            evidence_contract.canonical_paths = _dedupe(evidence_path_map.get(requirement.requirement_id, []))
            break

    validation_index = {
        "requirement_ids": [item.requirement_id for item in requirements],
        "semantic_requirement_statements": {
            item.requirement_id: item.statement
            for item in semantic_assertions
            if str(item.requirement_id or "").strip() and str(item.statement or "").strip()
        },
        "work_package_ids": [item.work_package_id for item in work_packages],
        "registered_paths": sorted(item.canonical_path for item in file_nodes),
        "registered_surfaces": sorted(item.canonical_path for item in surface_nodes),
        "unresolved_result_targets": _unresolved_result_targets_for_index(
            state,
            artifact_producer_by_path,
            artifact_owner_by_path,
            _fallback_owner_file_path,
        ),
        "owner_resolution_fallbacks": owner_resolution_fallbacks,
    }

    return CanonicalIROutput(
        requirements=requirements,
        work_packages=work_packages,
        contract_stages=contract_stages,
        file_nodes=file_nodes,
        surface_nodes=surface_nodes,
        edges=edges,
        semantic_assertions=semantic_assertions,
        evidence_contracts=evidence_contracts,
        validator_expectations=validator_expectations,
        validation_index=validation_index,
    )


def validate_canonical_ir(state: PaperBenchReproState, canonical_ir: CanonicalIROutput) -> CanonicalIRValidationOutput:
    """Validate shadow canonical IR closure without blocking the legacy path."""

    mismatches: list[CanonicalIRMismatch] = []
    requirement_ids = {item.requirement_id for item in canonical_ir.requirements}
    registered_paths = set(canonical_ir.validation_index.get("registered_paths", []))
    registered_surface_paths = {item.canonical_path for item in canonical_ir.surface_nodes}

    unit_ids = {
        item.unit_id
        for item in list(state.unit_extraction.units if state.unit_extraction else [])
        if str(item.unit_id or "").strip()
    }
    mapped_unit_ids = {
        unit_id
        for item in canonical_ir.requirements
        for unit_id in list(item.source_unit_ids or [])
        if str(unit_id or "").strip()
    }
    for unit_id in sorted(unit_ids - mapped_unit_ids):
        mismatches.append(
            CanonicalIRMismatch(
                category="unmapped_unit",
                message=f"unit `{unit_id}` is not mapped into canonical requirements",
                severity="warning",
                related_ids=[unit_id],
            )
        )

    for work_package in canonical_ir.work_packages:
        invalid_requirement_ids = [
            requirement_id
            for requirement_id in list(work_package.requirement_ids or [])
            if requirement_id not in requirement_ids
        ]
        if invalid_requirement_ids:
            mismatches.append(
                CanonicalIRMismatch(
                    category="invalid_requirement_ref",
                    message=(
                        f"work package `{work_package.work_package_id}` references unknown requirement ids: "
                        + ", ".join(invalid_requirement_ids[:6])
                    ),
                    severity="retry_generate",
                    related_ids=[work_package.work_package_id, *invalid_requirement_ids],
                )
            )

    work_package_ids = {item.work_package_id for item in canonical_ir.work_packages}
    file_plan_owner_by_path = {
        _normalize_path(item.target_file): str(item.work_package_id or "").strip()
        for item in list(state.package_file_planning_output.file_plans if state.package_file_planning_output else [])
        if _normalize_path(item.target_file)
    }
    for file_node in canonical_ir.file_nodes:
        path = _normalize_path(file_node.canonical_path)
        owner = str(file_node.owner_work_package_id or "").strip()
        file_plan_owner = file_plan_owner_by_path.get(path, "")
        if not owner:
            mismatches.append(
                CanonicalIRMismatch(
                    category="unowned_registered_file",
                    message=f"registered file `{path}` has no canonical owner work package",
                    severity="retry_generate",
                    related_ids=[path, *list(file_node.related_plan_node_ids or [])],
                )
            )
        elif owner not in work_package_ids:
            mismatches.append(
                CanonicalIRMismatch(
                    category="invalid_file_owner",
                    message=f"registered file `{path}` points to non-canonical owner `{owner}`",
                    severity="retry_generate",
                    related_ids=[path, owner],
                )
            )
        if file_plan_owner and file_plan_owner != owner:
            severity = "retry_generate" if file_plan_owner not in work_package_ids else "warning"
            mismatches.append(
                CanonicalIRMismatch(
                    category="file_plan_owner_drift",
                    message=(
                        f"file plan owner `{file_plan_owner}` for `{path}` does not match "
                        f"canonical owner `{owner or 'unassigned'}`"
                    ),
                    severity=severity,
                    related_ids=[path, file_plan_owner, owner],
                )
            )

    for item in list(canonical_ir.validation_index.get("owner_resolution_fallbacks", []) or []):
        if not isinstance(item, dict):
            continue
        path = _normalize_path(str(item.get("canonical_path") or ""))
        owner = str(item.get("owner_work_package_id") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not path or reason in {"", "unresolved"}:
            continue
        mismatches.append(
            CanonicalIRMismatch(
                category="inferred_file_owner",
                message=f"registered file `{path}` owner was inferred as `{owner}` via `{reason}`",
                severity="warning",
                related_ids=[path, owner, reason],
            )
        )

    for path in list(state.architecture.execution_entrypoints if state.architecture else []):
        normalized = _normalize_path(path)
        if normalized and normalized not in registered_surface_paths:
            mismatches.append(
                CanonicalIRMismatch(
                    category="unmapped_surface",
                    message=f"execution entrypoint `{normalized}` is not registered in canonical IR",
                    severity="retry_generate",
                    related_ids=[normalized],
                )
            )

    for path in list(state.architecture.config_surfaces if state.architecture else []):
        normalized = _normalize_path(path)
        if normalized and normalized not in registered_surface_paths:
            severity = "retry_generate" if normalized in registered_paths else "warning"
            category = "unmapped_surface" if severity == "retry_generate" else "path_drift"
            mismatches.append(
                CanonicalIRMismatch(
                    category=category,
                    message=(
                        f"config surface `{normalized}` is not registered in canonical IR"
                        if category == "unmapped_surface"
                        else f"config surface `{normalized}` is path drift outside canonical file graph"
                    ),
                    severity=severity,
                    related_ids=[normalized],
                )
            )

    for path in list(state.architecture.stable_interfaces if state.architecture else []):
        normalized = _normalize_path(path)
        if normalized and normalized not in registered_surface_paths:
            mismatches.append(
                CanonicalIRMismatch(
                    category="unmapped_surface",
                    message=f"stable interface `{normalized}` is not registered in canonical IR",
                    severity="retry_generate",
                    related_ids=[normalized],
                )
            )

    for item in list(canonical_ir.validation_index.get("unresolved_result_targets", []) or []):
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or "").strip()
        owner_work_package = str(item.get("owner_work_package") or "").strip()
        artifact_paths = [
            _normalize_path(path)
            for path in list(item.get("artifact_paths", []) or [])
            if _normalize_path(path)
        ]
        mismatches.append(
            CanonicalIRMismatch(
                category="unresolved_artifact_producer",
                message=(
                    f"result target `{target_id or 'unknown_target'}` owned by "
                    f"`{owner_work_package or 'unknown_owner'}` has no canonical producer file"
                ),
                severity="retry_generate",
                related_ids=[value for value in [target_id, owner_work_package, *artifact_paths] if value],
            )
        )

    registered_assertions = {item.assertion_id for item in canonical_ir.semantic_assertions}
    registered_evidence_contracts = {item.evidence_contract_id for item in canonical_ir.evidence_contracts}
    for requirement_id in sorted(requirement_ids):
        if not any(item.requirement_id == requirement_id for item in canonical_ir.semantic_assertions):
            mismatches.append(
                CanonicalIRMismatch(
                    category="missing_assertion",
                    message=f"requirement `{requirement_id}` has no semantic assertion",
                    severity="retry_generate",
                    related_ids=[requirement_id],
                )
            )
    for expectation in canonical_ir.validator_expectations:
        if expectation.assertion_id not in registered_assertions:
            mismatches.append(
                CanonicalIRMismatch(
                    category="missing_assertion",
                    message=f"validator expectation `{expectation.expectation_id}` points to missing assertion `{expectation.assertion_id}`",
                    severity="retry_generate",
                    related_ids=[expectation.expectation_id, expectation.assertion_id],
                )
            )
        if expectation.evidence_contract_id and expectation.evidence_contract_id not in registered_evidence_contracts:
            mismatches.append(
                CanonicalIRMismatch(
                    category="missing_evidence_contract",
                    message=f"validator expectation `{expectation.expectation_id}` points to missing evidence contract `{expectation.evidence_contract_id}`",
                    severity="retry_generate",
                    related_ids=[expectation.expectation_id, expectation.evidence_contract_id],
                )
            )

    mismatch_summary = dict(Counter(item.category for item in mismatches))
    gate_actions = _dedupe([item.severity for item in mismatches]) or ["warning"]
    semantic_validation_report = {
        "semantic_assertions": [
            {
                "assertion_id": item.assertion_id,
                "requirement_id": item.requirement_id,
                "status": item.status,
            }
            for item in canonical_ir.semantic_assertions
        ],
        "validator_expectations": [
            {
                "expectation_id": item.expectation_id,
                "assertion_id": item.assertion_id,
                "status": item.status,
            }
            for item in canonical_ir.validator_expectations
        ],
    }
    planning_failure_layer = ""
    if mismatches:
        if any(item.severity == "retry_generate" for item in mismatches):
            planning_failure_layer = "planning_contract"
        else:
            planning_failure_layer = "planning_semantic"
    return CanonicalIRValidationOutput(
        passed=not mismatches,
        planning_failure_layer=planning_failure_layer,
        mismatches=mismatches,
        mismatch_summary=mismatch_summary,
        gate_actions=gate_actions,
        semantic_validation_report=semantic_validation_report,
    )
