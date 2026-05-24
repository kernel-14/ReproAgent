"""Deterministic evidence-grounding helpers for reproagent middle stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reproagent.pipeline.schemas import (
    EvidenceBundleOutput,
    EvidenceLinkOutput,
    PaperBenchReproState,
    WorkPackagePlanningOutput,
)


_WEAK_TERMS = {
    "agent",
    "and",
    "artifact",
    "baseline",
    "class",
    "config",
    "configuration",
    "data",
    "dataset",
    "entrypoint",
    "env",
    "file",
    "files",
    "for",
    "from",
    "function",
    "in",
    "interface",
    "main",
    "method",
    "model",
    "module",
    "or",
    "result",
    "results",
    "self",
    "test",
    "tests",
    "the",
    "unit",
    "via",
    "with",
    "writer",
}

_METHOD_SIGNATURE_TERMS = {
    "critical",
    "explanation",
    "fidelity",
    "horizon",
    "importance",
    "mask",
    "objective",
    "policy",
    "refinement",
    "retrain",
    "rollin",
    "trajectory",
}

_METHOD_CONTEXT_TERMS = {
    "algorithm",
    "exploration",
    "guide",
    "jump",
    "loss",
    "network",
    "optimization",
    "rl",
    "roll",
    "state",
    "train",
    "training",
}

_ENVIRONMENT_TERMS = {
    "drive",
    "environment",
    "gym",
    "simulator",
    "simulation",
    "task",
    "world",
}

_EVALUATION_TERMS = {
    "ablation",
    "benchmark",
    "compare",
    "comparison",
    "efficiency",
    "eval",
    "evaluate",
    "evaluation",
    "metric",
    "metrics",
    "protocol",
    "seed",
}


@dataclass(frozen=True)
class _CandidateScore:
    link: EvidenceLinkOutput
    score: float
    repo_role: str
    package_type: str
    package_overlap: list[str]
    signature_hits: list[str]
    method_context_hits: list[str]
    surface_hits: list[str]
    weak_hits: list[str]


def _tokenize_text(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        for chunk in text.replace("/", " ").replace("_", " ").replace("-", " ").split():
            normalized = chunk.strip(" .,:;()[]{}")
            if len(normalized) >= 3 and not normalized.isdigit():
                tokens.add(normalized)
    return tokens


def _nonweak_terms(terms: set[str]) -> set[str]:
    return {term for term in terms if term not in _WEAK_TERMS}


def _survey_terms(survey: Any) -> set[str]:
    return _tokenize_text(
        getattr(survey, "title", ""),
        getattr(survey, "repository_url", ""),
        getattr(survey, "readme_summary", ""),
        getattr(survey, "file_tree_summary", ""),
        *(list(getattr(survey, "top_level_files", []) or [])),
        *(list(getattr(survey, "top_python_files", []) or [])),
        *(list(getattr(survey, "likely_reusable_files", []) or [])),
        *(list(getattr(survey, "protocol_clues", []) or [])),
    )


def _classify_repo_role(survey: Any) -> str:
    terms = _survey_terms(survey)
    method_score = len(terms.intersection(_METHOD_SIGNATURE_TERMS)) + len(terms.intersection(_METHOD_CONTEXT_TERMS)) * 0.35
    env_score = len(terms.intersection(_ENVIRONMENT_TERMS))
    eval_score = len(terms.intersection(_EVALUATION_TERMS))
    if method_score >= 1.0 and method_score >= env_score:
        return "method_reference"
    if env_score >= 1:
        return "environment_adapter"
    if eval_score >= 1:
        return "evaluation_protocol"
    return "general_reference"


def _package_type(work_package: Any) -> str:
    tags = {str(item or "").strip().lower() for item in list(getattr(work_package, "tags", []) or [])}
    terms = _tokenize_text(
        getattr(work_package, "goal", ""),
        *tags,
        *list(getattr(work_package, "evidence_needs", []) or []),
        *list(getattr(work_package, "method_obligations", []) or []),
    )
    if "claim" in tags or "tests" in tags:
        return "claim"
    if "task" in tags or "entrypoint" in tags:
        return "task"
    if "protocol" in tags or "baseline_or_ablation" in tags:
        return "protocol"
    if {"method", "model_or_method", "training_loop"}.intersection(tags) or terms.intersection(_METHOD_SIGNATURE_TERMS):
        return "method"
    if "evaluation" in tags or terms.intersection(_EVALUATION_TERMS):
        return "protocol"
    return "general"


def _reference_rank_for_work_package(survey: Any, package_terms: set[str], package_type: str) -> float:
    terms = _survey_terms(survey)
    role = _classify_repo_role(survey)
    overlap = len(_nonweak_terms(package_terms).intersection(_nonweak_terms(terms)))
    signature_hits = len(terms.intersection(_METHOD_SIGNATURE_TERMS))
    env_hits = len(terms.intersection(_ENVIRONMENT_TERMS))
    eval_hits = len(terms.intersection(_EVALUATION_TERMS))
    score = overlap * 0.6 + signature_hits * 0.7 + eval_hits * 0.25
    if package_type == "method" and role == "method_reference":
        score += 2.0
    elif package_type == "method" and role == "environment_adapter":
        score -= 0.8 + env_hits * 0.1
    elif package_type in {"protocol", "claim"} and role in {"evaluation_protocol", "method_reference"}:
        score += 0.8
    elif package_type in {"protocol", "task"} and role == "environment_adapter":
        score += 0.25
    return score


def _candidate_reference_ids_for_work_package(state: PaperBenchReproState, work_package) -> list[str]:
    surveys = list(getattr(state, "reference_repo_surveys", []) or [])
    survey_ids = {str(getattr(item, "ref_id", "") or "").strip() for item in surveys if str(getattr(item, "ref_id", "") or "").strip()}
    explicit = [
        str(ref_id or "").strip()
        for ref_id in list(getattr(work_package, "reference_ids", []) or [])
        if str(ref_id or "").strip() in survey_ids
    ]

    package_terms = _tokenize_text(
        work_package.goal,
        *work_package.tags,
        *work_package.evidence_needs,
        *work_package.method_obligations,
        *[value for values in work_package.inventories.values() for value in values],
    )
    package_type = _package_type(work_package)
    derived_ranked: list[tuple[float, str]] = []
    for survey in surveys:
        ref_id = str(getattr(survey, "ref_id", "") or "").strip()
        if not ref_id:
            continue
        score = _reference_rank_for_work_package(survey, package_terms, package_type)
        if score > 0:
            derived_ranked.append((score, ref_id))
    derived_ranked.sort(key=lambda item: (-item[0], item[1]))
    derived = [ref_id for score, ref_id in derived_ranked if score > 0]

    actionable = list(getattr(getattr(state, "reference_selection", None), "actionable_references", []) or [])
    if actionable:
        ranked: list[tuple[int, str]] = []
        for reference in actionable:
            ref_terms = _tokenize_text(
                getattr(reference, "title", ""),
                getattr(reference, "readme_summary", ""),
                *(list(getattr(reference, "reusable_modules", []) or [])),
                *(list(getattr(reference, "likely_reusable_files", []) or [])),
                *(list(getattr(reference, "top_python_files", []) or [])),
                *(list(getattr(reference, "protocol_clues", []) or [])),
            )
            overlap = len(package_terms.intersection(ref_terms))
            if overlap > 0:
                ranked.append((overlap, str(getattr(reference, "ref_id", "") or "").strip()))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        derived.extend([ref_id for score, ref_id in ranked if score > 0 and ref_id])

    if explicit or derived:
        return list(dict.fromkeys([*explicit, *derived]))[:6]

    # Do not silently bind a work package to arbitrary repositories. A package
    # with no lexical/survey overlap should remain paper-only so later gates can
    # decide whether that is acceptable instead of receiving misleading evidence.
    return []


def _link_sort_key(link: EvidenceLinkOutput) -> tuple[float, str, str]:
    return (-float(link.confidence or 0.0), str(link.ref_id or ""), str(link.file_path or ""))


def _score_candidate(
    *,
    work_package: Any,
    package_terms: set[str],
    package_type: str,
    ref_id: str,
    survey: Any,
    evidence: Any,
    owned_unit_ids: set[str],
    evidence_unit_ids: set[str],
) -> tuple[_CandidateScore | None, dict[str, Any]]:
    repo_role = _classify_repo_role(survey)
    tags = {str(item or "").strip().lower() for item in list(getattr(work_package, "tags", []) or [])}
    evidence_terms = _tokenize_text(
        getattr(evidence, "symbol_name", ""),
        getattr(evidence, "symbol_kind", ""),
        getattr(evidence, "file_path", ""),
        getattr(evidence, "relevance_reason", ""),
        *list(getattr(evidence, "matched_keywords", []) or []),
        *list(getattr(evidence, "matched_surfaces", []) or []),
        *list(getattr(evidence, "matched_artifacts", []) or []),
    )
    keyword_terms = _tokenize_text(*list(getattr(evidence, "matched_keywords", []) or []))
    surface_terms = _tokenize_text(*list(getattr(evidence, "matched_surfaces", []) or []))
    package_overlap = sorted(_nonweak_terms(package_terms).intersection(_nonweak_terms(evidence_terms)))[:8]
    weak_hits = sorted(package_terms.intersection(evidence_terms).intersection(_WEAK_TERMS))[:8]
    signature_hits = sorted(evidence_terms.intersection(_METHOD_SIGNATURE_TERMS))[:8]
    method_context_hits = sorted(evidence_terms.intersection(_METHOD_CONTEXT_TERMS))[:8]
    surface_hits = sorted(_nonweak_terms(tags).intersection(_nonweak_terms(surface_terms)))[:8]
    role_bonus = 0.0
    role_compatible = False
    if package_type == "method" and repo_role == "method_reference":
        role_bonus = 0.28
        role_compatible = True
    elif package_type in {"protocol", "claim"} and repo_role in {"method_reference", "evaluation_protocol"}:
        role_bonus = 0.18
        role_compatible = True
    elif package_type in {"protocol", "task"} and repo_role == "environment_adapter":
        role_bonus = 0.08
        role_compatible = True
    elif repo_role == "general_reference":
        role_bonus = 0.04

    raw_score = min(0.22, max(0.0, float(getattr(evidence, "score", 0.0) or 0.0)) * 0.018)
    score = (
        0.12
        + raw_score
        + len(package_overlap) * 0.07
        + len(signature_hits) * 0.12
        + len(method_context_hits) * 0.035
        + len(surface_hits) * 0.035
        + role_bonus
        + max(0, len(keyword_terms.difference(_WEAK_TERMS)) - 1) * 0.015
    )

    reason = ""
    if package_type == "method" and repo_role == "environment_adapter" and not signature_hits:
        score -= 0.35
        reason = "rejected_method_package_environment_repo_without_signature_anchor"
    elif not (signature_hits or package_overlap or surface_hits or role_compatible or score >= 0.48):
        reason = "rejected_weak_generic_match"
    elif score < 0.33:
        reason = "rejected_below_threshold"

    diagnostics = {
        "ref_id": ref_id,
        "repo_role": repo_role,
        "file_path": str(getattr(evidence, "file_path", "") or ""),
        "symbol_name": str(getattr(evidence, "symbol_name", "") or ""),
        "score": round(score, 4),
        "raw_symbol_score": float(getattr(evidence, "score", 0.0) or 0.0),
        "accepted": not bool(reason),
        "reason": reason or "accepted",
        "package_overlap": package_overlap,
        "signature_hits": signature_hits,
        "method_context_hits": method_context_hits,
        "surface_hits": surface_hits,
        "weak_hits": weak_hits,
    }
    if reason:
        return None, diagnostics

    matched_keywords = [
        term
        for term in list(getattr(evidence, "matched_keywords", []) or [])
        if str(term or "").strip().lower() not in _WEAK_TERMS and not str(term or "").strip().isdigit()
    ]
    if not matched_keywords:
        matched_keywords = [*signature_hits, *package_overlap, *surface_hits, *method_context_hits]

    link = EvidenceLinkOutput(
        unit_id=(
            next(iter(evidence_unit_ids.intersection(owned_unit_ids)), "")
            or (work_package.owned_unit_ids[0] if work_package.owned_unit_ids else "")
        ),
        ref_id=ref_id,
        file_path=str(getattr(evidence, "file_path", "") or ""),
        snippet_preview=str(getattr(evidence, "snippet", "") or "")[:1200],
        why_relevant=str(getattr(evidence, "relevance_reason", "") or "").strip()
        or f"{ref_id}:{getattr(evidence, 'file_path', '')} grounds `{work_package.work_package_id}`.",
        confidence=min(1.0, max(0.05, score)),
        matched_keywords=list(dict.fromkeys(matched_keywords))[:8],
    )
    return (
        _CandidateScore(
            link=link,
            score=score,
            repo_role=repo_role,
            package_type=package_type,
            package_overlap=package_overlap,
            signature_hits=signature_hits,
            method_context_hits=method_context_hits,
            surface_hits=surface_hits,
            weak_hits=weak_hits,
        ),
        diagnostics,
    )


def _select_diverse_candidates(candidates: list[_CandidateScore], limit: int = 6) -> list[_CandidateScore]:
    selected: list[_CandidateScore] = []
    per_ref: dict[str, int] = {}
    per_file: dict[tuple[str, str], int] = {}

    for candidate in candidates:
        ref_id = str(candidate.link.ref_id or "")
        file_path = str(candidate.link.file_path or "")
        if per_ref.get(ref_id, 0) >= 4:
            continue
        if per_file.get((ref_id, file_path), 0) >= 2:
            continue
        selected.append(candidate)
        per_ref[ref_id] = per_ref.get(ref_id, 0) + 1
        per_file[(ref_id, file_path)] = per_file.get((ref_id, file_path), 0) + 1
        if len(selected) >= limit:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _paper_context_for_owned_units(
    state: PaperBenchReproState,
    owned_unit_ids: list[str],
) -> list[str]:
    """Build self-contained paper evidence when repo-symbol grounding is unavailable."""
    if state.unit_extraction is None:
        return []
    units_by_id = {
        str(unit.unit_id or "").strip(): unit
        for unit in list(state.unit_extraction.units or [])
        if str(unit.unit_id or "").strip()
    }
    context: list[str] = []
    for unit_id in owned_unit_ids:
        unit = units_by_id.get(str(unit_id or "").strip())
        if unit is None:
            continue
        source_ids = [
            str(item or "").strip()
            for item in list(getattr(unit, "source_paragraph_ids", []) or [])
            if str(item or "").strip()
        ][:3]
        evidence_items = [
            str(item or "").strip()
            for item in list(getattr(unit, "paper_evidence", []) or [])
            if str(item or "").strip()
        ][:2]
        if not evidence_items and source_ids:
            statement = str(getattr(unit, "statement", "") or "").strip()
            if statement:
                evidence_items = [statement]
        if not evidence_items:
            continue
        source_suffix = f" ({', '.join(source_ids)})" if source_ids else ""
        for evidence in evidence_items:
            context.append(f"paper:{unit_id}{source_suffix}: {evidence[:800]}")
        if len(context) >= 12:
            break
    return context[:12]


def _build_evidence_bundles(
    state: PaperBenchReproState,
    work_package_output: WorkPackagePlanningOutput,
) -> tuple[list[EvidenceBundleOutput], list[EvidenceLinkOutput]]:
    """Ground work packages against prepared reference repo surveys using unit-driven symbol evidence."""
    survey_by_ref_id = {item.ref_id: item for item in state.reference_repo_surveys}
    bundles: list[EvidenceBundleOutput] = []
    evidence_graph: list[EvidenceLinkOutput] = []
    diagnostics_payload: list[dict[str, Any]] = []

    for work_package in work_package_output.work_packages:
        package_terms = _tokenize_text(
            work_package.goal,
            *work_package.tags,
            *work_package.evidence_needs,
            *work_package.method_obligations,
            *[value for values in work_package.inventories.values() for value in values],
        )
        package_type = _package_type(work_package)
        candidates: list[_CandidateScore] = []
        considered: list[dict[str, Any]] = []
        owned_unit_ids = {
            str(unit_id or "").strip()
            for unit_id in list(work_package.owned_unit_ids or [])
            if str(unit_id or "").strip()
        }

        candidate_reference_ids = _candidate_reference_ids_for_work_package(state, work_package)
        for ref_id in candidate_reference_ids:
            survey = survey_by_ref_id.get(ref_id)
            if survey is None:
                continue
            for evidence in list(getattr(survey, "symbol_evidence", []) or []):
                evidence_unit_ids = {
                    str(unit_id or "").strip()
                    for unit_id in list(getattr(evidence, "matched_unit_ids", []) or [])
                    if str(unit_id or "").strip()
                }
                if owned_unit_ids and not evidence_unit_ids.intersection(owned_unit_ids):
                    continue
                candidate, item_diagnostics = _score_candidate(
                    work_package=work_package,
                    package_terms=package_terms,
                    package_type=package_type,
                    ref_id=ref_id,
                    survey=survey,
                    evidence=evidence,
                    owned_unit_ids=owned_unit_ids,
                    evidence_unit_ids=evidence_unit_ids,
                )
                considered.append(item_diagnostics)
                if candidate is not None:
                    candidates.append(candidate)

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                -item.score,
                item.link.ref_id,
                item.link.file_path,
                item.link.unit_id,
            ),
        )
        candidates = _select_diverse_candidates(ranked_candidates, limit=6)
        links = [candidate.link for candidate in candidates]
        evidence_graph.extend(links)
        repo_context_summary = [
            f"{work_package.work_package_id} <- {link.ref_id}:{link.file_path}"
            for link in links
        ]
        paper_context_summary = _paper_context_for_owned_units(
            state,
            [str(unit_id or "").strip() for unit_id in list(work_package.owned_unit_ids or [])],
        )
        context_summary = [*repo_context_summary, *paper_context_summary]
        diagnostics_payload.append(
            {
                "work_package_id": work_package.work_package_id,
                "package_type": package_type,
                "candidate_reference_ids": candidate_reference_ids,
                "accepted_count": len(links),
                "paper_context_count": len(paper_context_summary),
                "rejected_count": max(0, len(considered) - len(links)),
                "top_candidates": sorted(considered, key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("ref_id", ""))))[:12],
            }
        )
        has_repo_reference_scope = any(
            str(ref_id or "").strip() in survey_by_ref_id
            for ref_id in list(getattr(work_package, "reference_ids", []) or [])
        )

        bundles.append(
            EvidenceBundleOutput(
                work_package_id=work_package.work_package_id,
                focus=work_package.goal,
                owned_unit_ids=list(work_package.owned_unit_ids),
                evidence_links=links,
                context_summary=context_summary[:12],
                grounding_status=(
                    "grounded"
                    if links
                    else (
                        "self_contained"
                        if paper_context_summary or (not candidate_reference_ids and not has_repo_reference_scope)
                        else "ungrounded"
                    )
                ),
            )
        )

    state.temp_data["evidence_diagnostics"] = diagnostics_payload
    return bundles, evidence_graph
