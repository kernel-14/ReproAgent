"""Prepare-stage implementations for reproagent workflow."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from reproagent.pipeline.config import build_github_repo_config
from reproagent.pipeline.prompts import (
    build_input_normalization_prompt,
    build_unit_extraction_prompt,
)
from reproagent.pipeline.schemas import (
    ExtractedUnit,
    PaperBenchReproState,
    InputNormalizationOutput,
    PaperChunk,
    PreparedReferenceRepositorySurvey,
    UnitExtractionOutput,
    VerificationTarget,
)
from reproagent.pipeline.utils.ref_repo_clone import clone_reference_repository
from reproagent.pipeline.utils.ref_repo_search_tool import (
    search_reference_repository,
    validate_official_candidate,
)
from reproagent.pipeline.utils.intent_contract import ensure_upstream_intent_contract
from reproagent.pipeline.utils.artifact_writer import register_existing_file
from reproagent.pipeline.utils.early_quality import prepare_quality_gate_report, unit_extraction_quality_report
from reproagent.pipeline.utils.evidence_contracts import (
    evidence_contract_gaps,
    flatten_evidence_contract,
    infer_evidence_contract,
)
from reproagent.pipeline.utils import workflow_runtime

logger = logging.getLogger(__name__)

_GENERIC_PAPER_DERIVED_UNIT_IDS = {
    "paper_evidence_matrix",
    "paper_named_experiment_protocols",
    "paper_addendum_constraints",
    "paper_environment_inventory",
    "paper_dataset_inventory",
    "paper_task_environment_setup",
    "paper_method_core",
    "paper_training_or_optimization_loop",
    "paper_evaluation_protocol",
    "paper_contract_dataset_metric_protocol",
    "paper_contract_method_baseline_protocol",
    "paper_contract_experiment_artifact_protocol",
    "paper_contract_sweep_hyperparameter_protocol",
    "paper_contract_environment_protocol",
}


class PrepareQualityGateError(RuntimeError):
    """Legacy compatibility error for callers that still import the old gate exception."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        reasons = list(report.get("blocking_reasons", []) or [])
        preview = "; ".join(str(item) for item in reasons[:8])
        suffix = f"; +{len(reasons) - 8} more" if len(reasons) > 8 else ""
        super().__init__(preview + suffix if preview else "prepare quality gate failed")

_GITHUB_REPO_URL_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
    re.IGNORECASE,
)
_NAMED_EXPERIMENT_RE = re.compile(
    r"\b(?:Experiment|Exp\.?|Study|Evaluation|Ablation)\s+"
    r"(?:[IVX]{1,6}(?![A-Za-z])(?:\s*-\s*[IVX]{1,6}(?![A-Za-z]))?|\d+[A-Za-z]?)"
    r"(?:(?:\s*:\s*|\s+[–—-]\s+)[^\n.;]{0,100})?",
    re.IGNORECASE,
)
_RESULT_TABLE_FIGURE_RE = re.compile(r"\b(?:Table|Figure|Fig\.?)\s+\d+[A-Za-z]?\b", re.IGNORECASE)
_VERSIONED_ENV_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*-v\d+\b"
)
_COMMON_ENV_NAME_RE = re.compile(
    r"\b(?:Hopper|Walker2d|Reacher|HalfCheetah|MountainCar(?:Continuous)?|Ant|Humanoid|Swimmer|Selfish Mining|"
    r"CAGE(?: Challenge)?(?: 2)?|CybORG|Autonomous Driving|MetaDrive|Malware Mutation|MalConv)\b",
    re.IGNORECASE,
)
_FORK_RESUME_STAGE_NAMES = [
    "input_normalization",
    "reference_acquisition",
]

_FORK_SKIP_TOPLEVEL = {
    "repo_snapshots",
    "repo_validation_runtime",
    "repo",
    "validated_repo_init",
}

_FORK_SKIP_RELATIVE_PATHS = {
    "artifact_manifest.json",
    "handoff.json",
    "latest_state.json",
    "project_manifest.json",
    "quality_status.json",
    "run_manifest.json",
    "run_summary.json",
    "stage_attempts.json",
    "stage_status.json",
    "usage_summary.json",
    "workflow_events.jsonl",
}

_FORK_SKIP_PREFIXES = (
    "logs",
    "score_",
    "state_snapshots",
    "nodes/plan",
    "nodes/generate",
    "nodes/repair",
)

_IN_PLACE_STAGE_ORDER = [
    "input_normalization",
    "unit_extraction",
    "reference_acquisition",
    "prepare_quality_gate",
    "topic_profile_synthesis",
    "work_package_planning",
    "package_evidence_grounding",
    "global_contract_synthesis",
    "architecture_planning",
    "package_file_planning",
    "canonical_ir_synthesis",
    "local_file_generation",
    "repair_validation",
    "repair_plan",
    "repair_regeneration",
]

_IN_PLACE_RESUME_TEMP_DATA_PREFIXES = (
    "repair_",
    "validation_",
)

_IN_PLACE_RESUME_TEMP_DATA_KEYS = {
    "repo_handoff",
    "validated_repo_handoff",
    "handoff",
    "repair_ticket",
    "runtime_probe",
    "pending_repair_regeneration_attempt",
    "stage_execution_mode",
    "node_errors",
    "degraded_backlog",
    "terminal_outcome",
    "terminal_outcome_reason",
}


def _flatten_text_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        if value.strip():
            values.append(value.strip())
    elif isinstance(value, dict):
        for nested in value.values():
            values.extend(_flatten_text_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_flatten_text_values(nested))
    return values


def _direct_github_repository_url(payload: Any) -> str:
    for value in _flatten_text_values(payload):
        match = _GITHUB_REPO_URL_RE.search(value)
        if match:
            return match.group(0).removesuffix(".git")
    return ""


def _has_downloaded_contents(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _json_default_for_prepare_quality(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _build_prepare_quality_gate_report(state: PaperBenchReproState) -> dict[str, Any]:
    units = list(state.unit_extraction.units if state.unit_extraction else [])
    preparation = dict(state.temp_data.get("reference_repo_preparation", {}) or {})
    surveys = list(state.reference_repo_surveys or state.temp_data.get("reference_repo_surveys", []) or [])
    return prepare_quality_gate_report(
        paper_text=_paperbench_input_text(state),
        units=units,
        reference_repo_preparation=preparation,
        reference_repo_surveys=surveys,
    )


def _write_prepare_quality_gate(
    state: PaperBenchReproState,
    *,
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    get_output_dir: Callable[[PaperBenchReproState], Any],
) -> dict[str, Any]:
    report = _build_prepare_quality_gate_report(state)
    if report.get("status") != "passed":
        report = {
            **report,
            "degraded": True,
            "continue_with_best_effort": True,
            "next_action": "enter_plan_degraded_best_effort",
        }
    state.temp_data["prepare_quality_gate"] = report
    write_stage_output(state, "prepare_quality_gate.json", report)
    workflow_runtime.write_review_artifact(
        state,
        "prepare_quality_gate",
        {
            "stage_name": "prepare_quality_gate",
            "review_status": report.get("status", "unknown"),
            "blocking_reasons": list(report.get("blocking_reasons", []) or []),
            "warnings": list(report.get("warnings", []) or []),
            "active_unit_count": report.get("active_unit_count", 0),
            "prepared_reference_count": report.get("prepared_reference_count", 0),
        },
        get_output_dir=get_output_dir,
        json_default=_json_default_for_prepare_quality,
    )
    if report.get("status") != "passed":
        backlog = state.temp_data.setdefault("degraded_backlog", [])
        payload = {
            "stage": "prepare_quality_gate",
            "code": "prepare_quality_gate_degraded",
            "message": "prepare quality gate reported issues; continuing with best available units and reference evidence",
            "reasons": list(report.get("blocking_reasons", []) or [])[:16],
        }
        if isinstance(backlog, list) and payload not in backlog:
            backlog.append(payload)
    return report


def _resolve_reference_repository_request(
    item: dict[str, Any],
    *,
    github_config: dict[str, Any],
) -> dict[str, Any]:
    ref_id = str(item.get("ref_id", "") or item.get("paper_id", "") or item.get("id", "")).strip()
    paper_path = str(item.get("paper_path", "")).strip()
    paper_title = str(item.get("title", "")).strip()
    paper_url = str(item.get("paper_url", "") or item.get("url", "")).strip()
    search_only = bool(item.get("search_only", False))
    direct_repository_url = "" if search_only else _direct_github_repository_url(item)
    if not (direct_repository_url or paper_path or paper_title or paper_url):
        return {}
    if not ref_id:
        if direct_repository_url:
            ref_id = (
                direct_repository_url.rstrip("/").rsplit("/", 2)[-2]
                + "__"
                + direct_repository_url.rstrip("/").rsplit("/", 1)[-1]
            )
        elif paper_path:
            ref_id = Path(paper_path).stem
        elif paper_title:
            ref_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper_title).strip("_").lower()[:80]
    if direct_repository_url:
        validate_explicit = bool(github_config.get("validate_explicit_references", False))
        validation = validate_official_candidate(direct_repository_url, github_config=github_config) if validate_explicit else {}
        validation_valid = bool(validation.get("valid"))
        repository_url = str(validation.get("repository_url", "") or direct_repository_url).strip()
        repository_type = (
            "official"
            if validation_valid
            else (str(item.get("repository_type", "")).strip() or "explicit")
        )
        repository_origin = (
            str(item.get("repository_origin", "")).strip()
            or ("official" if validation_valid else "community")
        )
        matched_signals = ["idea_reference contains github url"]
        if validate_explicit:
            matched_signals.extend(list(validation.get("matched_signals", [])))
        else:
            matched_signals.append("explicit github url accepted without remote validation")
        return {
            "ref_id": ref_id,
            "title": paper_title,
            "paper_path": paper_path,
            "paper_url": paper_url,
            "repository_url": repository_url,
            "repository_origin": repository_origin,
            "repository_type": repository_type,
            "reference_role": str(item.get("reference_role", "") or ""),
            "matched_signals": matched_signals,
            "resolve_reason": str(validation.get("reason", "")).strip(),
            "source": str(item.get("source", "") or "paperbench_input"),
            "resolve_status": "found",
            "search_only": search_only,
        }

    resolve_result = search_reference_repository(
        ref_id=ref_id,
        paper_path="" if search_only else paper_path,
        paper_title=paper_title,
        paper_url=paper_url,
        github_config=github_config,
    )
    repository_type = str(resolve_result.get("repository_type", "")).strip() or "not_found"
    return {
        "ref_id": ref_id,
        "title": paper_title or str(resolve_result.get("title", "")).strip(),
        "paper_path": paper_path,
        "paper_url": paper_url,
        "repository_url": str(resolve_result.get("repository_url", "")).strip(),
        "repository_origin": (
            "official" if repository_type == "official"
            else "community" if repository_type == "reproduction"
            else str(item.get("repository_origin", "")).strip()
        ),
        "repository_type": repository_type,
        "reference_role": str(item.get("reference_role", "") or ""),
        "matched_signals": list(resolve_result.get("matched_signals", [])),
        "resolve_reason": str(resolve_result.get("reason", "")).strip(),
        "source": str(item.get("source", "") or "paperbench_input"),
        "resolve_status": str(resolve_result.get("status", "")).strip() or "not_found",
        "search_only": search_only,
    }


_SECTION_HEADING_RE = re.compile(
    r"^(?P<md>#{1,6})\s+(?P<md_title>.+?)\s*$"
    r"|^\\(?P<latex>section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{(?P<latex_title>.+?)\}\s*$",
    re.MULTILINE,
)


def _section_title(match: re.Match[str], index: int) -> str:
    title = (match.group("md_title") or match.group("latex_title") or "").strip()
    if title:
        return " ".join(title.split())
    return f"section_{index + 1}"


def _load_paper_text(state: PaperBenchReproState) -> tuple[str, str]:
    """Return paper text and source path from input fields."""
    explicit_text = str(getattr(state.input, "paper_text", "") or "").strip()
    if not explicit_text:
        explicit_text = str(getattr(state.input, "proposal_text", "") or "").strip()
    if explicit_text:
        return explicit_text, ""
    paper_path = str(getattr(state.input, "paper_path", "") or "").strip()
    if not paper_path:
        paper_path = str(getattr(state.input, "proposal_path", "") or "").strip()
    if paper_path:
        path = Path(paper_path).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path.resolve())
    return str(state.input.target or ""), ""


def _paper_sections(text: str) -> list[tuple[str, int, int, str]]:
    """Split markdown/LaTeX-ish paper text by section headings with offsets."""
    if not text.strip():
        return []
    matches = list(_SECTION_HEADING_RE.finditer(text, re.MULTILINE))
    if not matches:
        return [("document", 0, len(text), text.strip())]
    sections: list[tuple[str, int, int, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(("front_matter", 0, matches[0].start(), prefix))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = _section_title(match, index)
        body = text[start:end].strip()
        if body:
            sections.append((title, start, end, body))
    return sections or [("document", 0, len(text), text.strip())]


def _paper_chunk_index(chunks: list[PaperChunk]) -> list[dict[str, Any]]:
    """Compact chunk index for prompts; full chunk text stays in paper_chunks.json."""
    return [
        {
            "chunk_id": item.chunk_id,
            "section_title": item.section_title,
            "ordinal": item.ordinal,
            "source_path": item.source_path,
            "char_start": item.char_start,
            "char_end": item.char_end,
            "token_estimate": item.token_estimate,
            "split_reason": item.split_reason,
            "preview": item.text[:500],
        }
        for item in chunks
    ]


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _slug_result_ref(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or "result"


def _result_ref_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"\d+", str(value or ""))
    number = int(match.group(0)) if match else 999
    return number, str(value or "")


def _chunk_preview(chunk: PaperChunk, max_chars: int = 3000) -> str:
    text = " ".join(str(chunk.text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _state_paper_chunks(state: PaperBenchReproState) -> list[PaperChunk]:
    """Return paper chunks from state, temp_data, or the current run artifact."""
    chunks = list(state.paper_chunks or [])
    if chunks:
        return chunks
    raw_chunks = state.temp_data.get("paper_chunks") if isinstance(state.temp_data, dict) else None
    if isinstance(raw_chunks, list) and raw_chunks:
        recovered = [item if isinstance(item, PaperChunk) else PaperChunk.model_validate(item) for item in raw_chunks]
        state.paper_chunks = recovered
        return recovered
    run_id = str(getattr(state, "run_id", "") or "").strip()
    candidate_paths: list[Path] = []
    if run_id:
        candidate_paths.append(Path("output") / "reproagent" / run_id / "nodes" / "prepare" / "paper_chunks.json")
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                recovered = [PaperChunk.model_validate(item) for item in payload if isinstance(item, dict)]
                if recovered:
                    state.paper_chunks = recovered
                    state.temp_data["paper_chunks"] = [item.model_dump(mode="json") for item in recovered]
                    return recovered
        except Exception:
            continue
    return []


def _find_paper_chunks(
    chunks: list[PaperChunk],
    *,
    title_terms: tuple[str, ...],
    text_terms: tuple[str, ...],
    limit: int = 2,
) -> list[PaperChunk]:
    scored: list[tuple[int, int, PaperChunk]] = []
    for chunk in chunks:
        title = str(chunk.section_title or "").lower()
        text = str(chunk.text or "").lower()
        title_match = any(term in title for term in title_terms)
        text_match = any(term in text for term in text_terms)
        if not (title_match or text_match):
            continue
        score = (100 if title_match else 0) + (20 if text_match else 0)
        if any(skip in title for skip in ("front_matter", "introduction", "related work", "acknowledgements", "conclusion")):
            score -= 40
        if any(preferred in title for preferred in ("proposed technique", "technique detail", "experiment setup", "experiment design", "evaluation")):
            score += 25
        scored.append((score, -int(chunk.ordinal or 0), chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _score, _ordinal, chunk in scored[:limit]]


def _paper_unit(
    *,
    unit_id: str,
    unit_type: str,
    statement: str,
    chunks: list[PaperChunk],
    surfaces: list[str],
    obligations: list[str],
    interfaces: list[str],
    artifacts: list[str],
    module_kinds: list[str],
    hypothesis: str = "",
    decision_value: str = "",
    stop_rule_or_pruning_rationale: str = "",
) -> ExtractedUnit:
    section_names = _ordered_unique([chunk.section_title for chunk in chunks])
    source_ids = _ordered_unique([chunk.chunk_id for chunk in chunks])
    evidence = [_chunk_preview(chunk) for chunk in chunks if _chunk_preview(chunk)]
    return ExtractedUnit(
        unit_id=unit_id,
        type=unit_type,
        statement=statement,
        hypothesis=hypothesis,
        decision_value=decision_value,
        stop_rule_or_pruning_rationale=stop_rule_or_pruning_rationale,
        paper_evidence=evidence[:3],
        source_paragraph_ids=source_ids,
        citation_refs=[],
        verification_targets=[
            VerificationTarget(
                kind="artifact",
                description=f"Generated code covers paper-derived unit `{unit_id}` from {', '.join(section_names[:3])}.",
            )
        ],
        implementation_surfaces=_ordered_unique(surfaces),
        code_obligations=_ordered_unique(obligations),
        runtime_interfaces=_ordered_unique(interfaces),
        expected_artifacts=_ordered_unique(artifacts),
        suggested_module_kinds=_ordered_unique(module_kinds),
        implementation_notes=[
            "Paper-derived code implementation unit; extracted from paper chunks/addenda."
        ],
        status="active",
    )


def _slugify_unit_fragment(text: str, *, max_parts: int = 5) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", str(text or "").lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "section",
        "paper",
        "results",
        "experiment",
        "experiments",
    }
    parts = [word for word in words if word not in stop][:max_parts]
    return "_".join(parts) or "paper_unit"


def _chunk_semantic_profile(chunk: PaperChunk) -> tuple[str, str, list[str], list[str], list[str], list[str]]:
    """Infer a chunk-grounded implementation profile without using evaluator metadata."""
    title = str(chunk.section_title or "").lower()
    text = str(chunk.text or "").lower()
    combined = title + "\n" + text
    if any(token in combined for token in ("classifier", "pre-trained", "pretrained", "fine-tune", "fine tune")):
        return (
            "classifier_loader_finetuning",
            "method",
            ["model_or_method", "training_loop", "config", "tests"],
            [
                "Implement classifier/model loading and fine-tuning hooks described in this paper chunk.",
                "Expose optimizer, learning rate, batch size, checkpoint, and class-count settings in config.",
            ],
            ["load_classifier(config)", "finetune_classifier(config)"],
            ["results/config_resolved.json", "results/training_trace.json"],
        )
    if any(token in combined for token in ("adaptor", "adapter", "shift module", "attention", "upsampling", "down-pooling", "down pooling")):
        return (
            "adapter_shift_module",
            "method",
            ["model_or_method", "policy_adapter", "config", "tests"],
            [
                "Implement the paper-stated adaptor/shift-module architecture with visible layer components.",
                "Expose architecture dimensions and activation/attention settings through config.",
            ],
            ["make_adapter(config)", "apply_shift_module(features, config)"],
            ["results/model_registry.json"],
        )
    if any(token in combined for token in ("adversarial", "worst-case", "worst case", "inner loop", "min-max", "min max", "noise selection")):
        return (
            "adversarial_noise_selection",
            "method",
            ["model_or_method", "training_loop", "metric_formula", "config", "tests"],
            [
                "Implement the adversarial or worst-case inner-loop selection described in this paper chunk.",
                "Expose iteration count, noise distribution, objective sign, and stopping controls in config.",
            ],
            ["select_adversarial_noise(config)", "inner_loop_objective(batch, config)"],
            ["results/adversarial_trace.json"],
        )
    if any(token in combined for token in ("loss", "objective", "similarity", "gradient", "guidance", "classifier-guided", "guided")):
        return (
            "training_loss_objective",
            "method",
            ["model_or_method", "training_loop", "metric_formula", "tests"],
            [
                "Implement the paper-specific loss/objective terms from this chunk.",
                "Keep each loss term separately named so validation can see how the paper objective is assembled.",
            ],
            ["compute_paper_loss(batch, config)", "loss term registry"],
            ["results/loss_trace.json"],
        )
    if any(token in combined for token in ("ddpm", "ldm", "diffusion", "autoencoder", "pretrained model", "denoising")):
        return (
            "diffusion_model_wrapper",
            "method",
            ["model_or_method", "data_pipeline", "config", "tests"],
            [
                "Implement wrappers for the paper-stated pretrained diffusion/autoencoder model family.",
                "Expose checkpoint/model identifiers and smoke-safe fallback behavior.",
            ],
            ["load_diffusion_model(config)", "sample_or_denoise(config)"],
            ["results/model_registry.json"],
        )
    if any(token in combined for token in ("dataset", "ffhq", "lsun", "imagenet", "cifar", "celeba", "benchmark")):
        return (
            "dataset_registry",
            "task",
            ["data_pipeline", "config", "tests"],
            [
                "Create code-visible dataset/benchmark registry entries named in this paper chunk.",
                "Keep downloads lazy and expose smoke fixtures for any external dataset paths.",
            ],
            ["dataset registry", "data loader factory"],
            ["results/dataset_registry.json", "results/data_manifest.json"],
        )
    if any(token in combined for token in ("fid", "lpips", "accuracy", "metric", "score", "precision", "recall")):
        return (
            "evaluation_metrics",
            "protocol",
            ["evaluation", "metric_formula", "artifact_writer", "tests"],
            [
                "Implement paper-stated metric formulas and aggregation rows.",
                "Bind each metric to the methods, datasets, and artifacts named in the paper chunk.",
            ],
            ["evaluate_metrics(config)", "metric registry"],
            ["results/metrics.json", "results/tables/summary.csv"],
        )
    if any(token in combined for token in ("figure", "fig.", "table", "ablation", "sensitivity", "gamma", "omega")):
        return (
            "artifact_experiment_protocol",
            "protocol",
            ["evaluation", "baseline_or_ablation", "artifact_writer", "config", "tests"],
            [
                "Implement code-visible experiment/artifact rows for the figures, tables, or sensitivity settings in this chunk.",
                "Expose paper-stated sweeps and fixed hyperparameters without running expensive full experiments during prepare.",
            ],
            ["experiment registry", "artifact writer"],
            ["results/experiment_registry.json", "results/artifact_manifest.json"],
        )
    return (
        "method_chunk",
        "method",
        ["model_or_method", "config", "tests"],
        [
            "Implement the paper-specific method details described in this chunk.",
            "Expose named configuration and runtime entrypoints rather than leaving this chunk as prose.",
        ],
        ["paper method component"],
        ["results/config_resolved.json"],
    )


def _paper_chunk_semantic_units(state: PaperBenchReproState, *, limit: int = 12) -> list[ExtractedUnit]:
    """Backfill specialized units directly from high-signal paper chunks."""
    chunks = _state_paper_chunks(state)
    scored: list[tuple[int, int, PaperChunk]] = []
    for chunk in chunks:
        title = str(chunk.section_title or "").lower()
        text = str(chunk.text or "")
        lowered = text.lower()
        if not text.strip():
            continue
        score = 0
        for token in (
            "algorithm",
            "method",
            "loss",
            "training",
            "classifier",
            "adaptor",
            "adapter",
            "adversarial",
            "dataset",
            "ddpm",
            "ldm",
            "diffusion",
            "fid",
            "lpips",
            "figure",
            "table",
            "sensitivity",
            "hyperparameter",
        ):
            if token in lowered:
                score += 15
        if any(term in title for term in ("method", "technique", "algorithm", "experiment", "evaluation", "result", "implementation", "appendix")):
            score += 25
        if any(skip in title for skip in ("front_matter", "abstract", "introduction", "related work", "conclusion", "acknowledgement")):
            score -= 30
        if score <= 0:
            continue
        scored.append((score, -int(chunk.ordinal or 0), chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    units: list[ExtractedUnit] = []
    used_ids: set[str] = set()
    for _score, _ordinal, chunk in scored[:limit]:
        label, unit_type, surfaces, obligations, interfaces, artifacts = _chunk_semantic_profile(chunk)
        title_slug = _slugify_unit_fragment(chunk.section_title or label, max_parts=4)
        text_slug = _slugify_unit_fragment(chunk.text or label, max_parts=3)
        unit_id = f"paper_semantic_{chunk.chunk_id}_{label}_{title_slug}_{text_slug}"
        unit_id = re.sub(r"[^A-Za-z0-9_]+", "_", unit_id)[:120]
        if unit_id in used_ids:
            continue
        used_ids.add(unit_id)
        units.append(
            _paper_unit(
                unit_id=unit_id,
                unit_type=unit_type,
                statement=f"Implement chunk-grounded `{label}` obligations from `{chunk.section_title}`.",
                chunks=[chunk],
                surfaces=surfaces,
                obligations=[
                    *obligations,
                    "Preserve concrete method, dataset, metric, figure/table, and hyperparameter names visible in this paper chunk.",
                    "Represent this chunk through concrete implementation units in addition to any paper_evidence_matrix or generic protocol units.",
                ],
                interfaces=interfaces,
                artifacts=artifacts,
                module_kinds=list(surfaces),
                hypothesis="This paper chunk contains specialized implementation semantics needed before planning can be faithful.",
                decision_value="Forces plan/generation to allocate concrete implementation surfaces to this chunk rather than a generic repository scaffold.",
                stop_rule_or_pruning_rationale="Implementation scope: keep one bounded unit per high-signal chunk and preserve distinct methods, datasets, metrics, and artifacts.",
            )
        )
    return units


def _named_experiment_anchor_unit(state: PaperBenchReproState) -> ExtractedUnit | None:
    chunks = list(state.paper_chunks or [])
    if not chunks:
        return None
    scored: list[tuple[int, int, PaperChunk]] = []
    for chunk in chunks:
        text = str(chunk.text or "")
        title = str(chunk.section_title or "").lower()
        named = _NAMED_EXPERIMENT_RE.findall(text)
        result_refs = _RESULT_TABLE_FIGURE_RE.findall(text)
        if not named and not result_refs:
            continue
        score = len(named) * 30 + min(len(result_refs), 4) * 8
        if any(term in title for term in ("experiment design", "experiment results", "evaluation", "result", "ablation")):
            score += 25
        if any(re.fullmatch(r"(?:table|figure|fig\.?)\s+[12][A-Za-z]?", ref, re.IGNORECASE) for ref in result_refs):
            score += 20
        if "appendix" in title or "additional experiment" in title:
            score -= 25
        if any(term in title for term in ("introduction", "related work", "conclusion", "acknowledgement")):
            score -= 30
        scored.append((score, -int(chunk.ordinal or 0), chunk))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [chunk for _score, _ordinal, chunk in scored[:3]]
    joined = "\n\n".join(chunk.text for chunk in selected)
    named_experiments = _ordered_unique([match.group(0).strip(" .;:") for match in _NAMED_EXPERIMENT_RE.finditer(joined)])
    result_refs = sorted(
        _ordered_unique([match.group(0).strip(" .;:") for match in _RESULT_TABLE_FIGURE_RE.finditer(joined)]),
        key=_result_ref_sort_key,
    )
    statement_parts = []
    if named_experiments:
        statement_parts.append("named experiment protocols: " + ", ".join(named_experiments[:10]))
    if result_refs:
        statement_parts.append("paper result artifacts: " + ", ".join(result_refs[:10]))
    statement = (
        "Implement explicit named experiment/result-protocol anchors from the paper"
        + (f" ({'; '.join(statement_parts)})." if statement_parts else ".")
    )
    obligations = [
        "Represent named paper experiments as explicit code/config registries or protocol entries with runnable ownership.",
        "For each named experiment protocol, preserve the environment/task set, compared methods or baselines, and required measurements when stated by the paper.",
        "Expose result-table/figure aggregation or artifact-writing surfaces for the named protocols without requiring full training during code generation.",
    ]
    artifacts = [
        "results/experiment_registry.json",
        "results/tables/experiment_results.csv",
        "results/metrics.json",
    ]
    for result_ref in result_refs[:8]:
        slug = _slug_result_ref(result_ref)
        if result_ref.lower().startswith(("figure", "fig")):
            artifacts.append(f"results/figures/{slug}.png")
        else:
            artifacts.append(f"results/tables/{slug}.csv")
    return _paper_unit(
        unit_id="paper_named_experiment_protocols",
        unit_type="protocol",
        statement=statement,
        chunks=selected,
        surfaces=["evaluation", "baseline_or_ablation", "artifact_writer", "config"],
        obligations=obligations,
        interfaces=["experiment registry", "result aggregation command or callable"],
        artifacts=artifacts,
        module_kinds=["evaluation", "reporting", "configuration"],
        hypothesis=(
            "Named experiment and result-table anchors preserve the paper's decisive comparisons, "
            "metrics, and artifact expectations for semantic review."
        ),
        decision_value=(
            "Determines which experiment protocols must be materialized as code/config registries "
            "before spending budget on generation or execution."
        ),
        stop_rule_or_pruning_rationale=(
            "Keep every paper-visible protocol discoverable, but represent repeated seeds, K sweeps, "
            "and environment repetitions through registries/config rather than separate low-value runners."
        ),
    )


def _addendum_text_unit(state: PaperBenchReproState) -> ExtractedUnit | None:
    design = state.input.experiment_design if isinstance(state.input.experiment_design, dict) else {}
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    text = "\n\n".join(
        part
        for part in [
            str(paperbench.get("addendum_text", "") or "").strip(),
        ]
        if part
    )
    if not text:
        return None
    preview = " ".join(text.split())
    if len(preview) > 900:
        preview = preview[:900].rstrip() + "..."
    return ExtractedUnit(
        unit_id="paper_addendum_constraints",
        type="protocol",
        statement="Implement binding code-level clarifications from PaperBench addenda as active reproduction coverage.",
        hypothesis="Addendum clarifications identify exact configuration, metric, artifact, and runnable-route details that change reproduction correctness.",
        decision_value="Connects addendum-derived implementation details to concrete config, evaluation, artifact, and documentation surfaces.",
        stop_rule_or_pruning_rationale=(
            "Implementation scope: encode addendum clarifications as executable selectors, bounded routes, metric choices, and artifact ownership."
        ),
        paper_evidence=[preview],
        source_paragraph_ids=["addendum.md"],
        citation_refs=[],
        verification_targets=[
            VerificationTarget(
                kind="artifact",
                description="Generated code and documentation expose addendum-level implementation coverage.",
            )
        ],
        implementation_surfaces=["config", "evaluation", "artifact_writer"],
        code_obligations=[
            "Translate addendum clarifications into repository configuration, evaluation, and documentation behavior."
        ],
        runtime_interfaces=["configuration flags or documented setup commands"],
        expected_artifacts=["README documentation", "configuration artifact"],
        suggested_module_kinds=["configuration", "evaluation", "documentation"],
        implementation_notes=[
            "implementation_surface_inventory: config, evaluation, artifact_writer, documentation",
            "artifact_inventory: README documentation; configuration artifact",
        ],
        status="active",
    )


def _paperbench_input_text(state: PaperBenchReproState) -> str:
    design = state.input.experiment_design if isinstance(state.input.experiment_design, dict) else {}
    paperbench = design.get("paperbench") if isinstance(design.get("paperbench"), dict) else {}
    return "\n\n".join(
        part
        for part in [
            "\n\n".join(str(chunk.text or "") for chunk in list(state.paper_chunks or [])),
            str(paperbench.get("addendum_text", "") or "").strip(),
        ]
        if part
    )


def _normalize_env_anchor(value: str) -> str:
    raw = " ".join(str(value or "").replace("_", "-").split()).strip(" ,.;:")
    if not raw:
        return ""
    lower = raw.lower()
    aliases = {
        "hopper-v3": "Hopper-v3",
        "hopper": "Hopper",
        "walker2d-v3": "Walker2d-v3",
        "walker2d": "Walker2d",
        "reacher-v2": "Reacher-v2",
        "reacher": "Reacher",
        "halfcheetah-v3": "HalfCheetah-v3",
        "halfcheetah": "HalfCheetah",
        "mountaincarcontinuous-v0": "MountainCarContinuous-v0",
        "mountaincarcontinuous": "MountainCarContinuous",
        "selfish mining": "selfish mining",
        "cage": "CAGE Challenge 2",
        "cage challenge": "CAGE Challenge 2",
        "cage challenge 2": "CAGE Challenge 2",
        "cyborg": "CybORG",
        "autonomous driving": "autonomous driving",
        "metadrive": "MetaDrive",
        "malware mutation": "Malware Mutation",
        "malconv": "MalConv",
    }
    return aliases.get(lower, raw)


def _paper_environment_inventory(state: PaperBenchReproState) -> list[str]:
    text = _paperbench_input_text(state)
    if not text.strip():
        return []
    candidates: list[str] = []
    candidates.extend(match.group(0).strip() for match in _VERSIONED_ENV_RE.finditer(text))
    candidates.extend(_normalize_env_anchor(match.group(0)) for match in _COMMON_ENV_NAME_RE.finditer(text))
    return _ordered_unique([_normalize_env_anchor(item) for item in candidates if _normalize_env_anchor(item)])


def _paper_environment_inventory_unit(state: PaperBenchReproState) -> ExtractedUnit | None:
    environments = _paper_environment_inventory(state)
    if not environments:
        return None
    chunks = _find_paper_chunks(
        list(state.paper_chunks or []),
        title_terms=("experiment setup", "environment", "application", "appendix c.2", "evaluation"),
        text_terms=("environment", "environments", "mujoco", "application", "hopper", "walker2d", "reacher"),
        limit=4,
    )
    paperbench_text = _paperbench_input_text(state)
    evidence = [f"Paper-derived environment/task anchors: {', '.join(environments[:18])}."]
    if chunks:
        evidence.extend(_chunk_preview(chunk, max_chars=1400) for chunk in chunks[:2])
    preview = " ".join(paperbench_text.split())
    if preview and len(evidence) < 3:
        evidence.append(preview[:1400])
    obligations = [
        "Expose explicit environment/task registry entries with ids, aliases, setup metadata, factory/config hooks, and paper task metadata for: "
        + ", ".join(environments[:18]),
        "Keep external simulator dependencies lazy and provide readiness checks with bounded generation-time execution.",
    ]
    return ExtractedUnit(
        unit_id="paper_environment_inventory",
        type="task",
        statement="Implement explicit paper-derived environment and task inventory.",
        hypothesis="The core method claim must be testable on the paper's named environment families through one registry contract.",
        decision_value="Decides which environment adapters and lazy dependency checks are required before generation can be judged.",
        stop_rule_or_pruning_rationale=(
            "Implementation scope: keep simulator variants registry-visible with bounded fixtures and explicit full-simulator routes."
        ),
        paper_evidence=evidence[:4],
        source_paragraph_ids=_ordered_unique([chunk.chunk_id for chunk in chunks] + ["addendum.md"]),
        citation_refs=[],
        verification_targets=[
            VerificationTarget(
                kind="artifact",
                description="Generated code exposes concrete environment/task registry entries and scope flags.",
            )
        ],
        implementation_surfaces=["environment", "config", "tests", "artifact_writer"],
        code_obligations=obligations,
        runtime_interfaces=["environment registry", "make_environment(config)", "environment readiness check"],
        expected_artifacts=["results/environment_registry.json", "results/environment_readiness.json"],
        suggested_module_kinds=["environment", "configuration", "tests"],
        implementation_notes=["Deterministically extracted from paper/addendum text."],
        status="active",
    )


def _paper_dataset_inventory_unit(state: PaperBenchReproState) -> ExtractedUnit | None:
    contract = infer_evidence_contract(_paperbench_input_text(state))
    datasets = _contract_item_names(contract, "datasets")
    if not datasets:
        return None
    chunks = _find_paper_chunks(
        list(state.paper_chunks or []),
        title_terms=("datasets", "dataset", "experiment setup", "evaluation", "benchmarks"),
        text_terms=("dataset", "datasets", "benchmark", "imagenet", "coco", "vqa", "corruption"),
        limit=4,
    )
    evidence = ["Paper-derived dataset/benchmark anchors: " + ", ".join(datasets[:24]) + "."]
    if chunks:
        evidence.extend(_chunk_preview(chunk, max_chars=1400) for chunk in chunks[:2])
    obligations = [
        "Expose explicit dataset/benchmark registry entries with aliases, split/sample policy, preprocessing hints, and lazy availability checks for: "
        + ", ".join(datasets[:24]),
        "Keep dataset downloads optional/lazy during generation; provide manifest entries and smoke fixtures instead of requiring full benchmark assets.",
        "Bind each dataset entry to the metric/artifact protocols that consume it with benchmark-specific loader coverage.",
    ]
    return ExtractedUnit(
        unit_id="paper_dataset_inventory",
        type="task",
        statement="Implement explicit paper-derived dataset and benchmark inventory.",
        hypothesis="The paper's comparison claims depend on named datasets/benchmarks being discoverable in code rather than implied by prose.",
        decision_value="Decides which data loaders, preprocessing stubs, and benchmark manifests must exist before generation.",
        stop_rule_or_pruning_rationale=(
            "Implementation scope: keep full dataset runs behind explicit commands and use bounded smoke fixtures during generation."
        ),
        paper_evidence=evidence[:4],
        source_paragraph_ids=_ordered_unique([chunk.chunk_id for chunk in chunks] + ["paper.md", "addendum.md"]),
        citation_refs=[],
        verification_targets=[
            VerificationTarget(kind="artifact", description="Generated code exposes concrete dataset/benchmark registry entries.")
        ],
        implementation_surfaces=["data_pipeline", "config", "tests", "artifact_writer"],
        code_obligations=obligations,
        runtime_interfaces=["dataset registry", "make_dataset(config)", "dataset readiness check"],
        expected_artifacts=["results/dataset_registry.json", "results/data_manifest.json"],
        suggested_module_kinds=["data_pipeline", "configuration", "tests"],
        implementation_notes=["dataset_inventory: " + "; ".join(datasets[:24])],
        status="active",
    )


def _paper_specialized_method_units(state: PaperBenchReproState) -> list[ExtractedUnit]:
    chunks = list(state.paper_chunks or [])
    text = _paperbench_input_text(state)
    lowered = text.lower()
    units: list[ExtractedUnit] = []

    def add(
        *,
        unit_id: str,
        unit_type: str,
        statement: str,
        title_terms: tuple[str, ...],
        text_terms: tuple[str, ...],
        surfaces: list[str],
        obligations: list[str],
        interfaces: list[str],
        artifacts: list[str],
        module_kinds: list[str],
        hypothesis: str,
        decision_value: str,
        stop_rule_or_pruning_rationale: str,
    ) -> None:
        matched = _find_paper_chunks(chunks, title_terms=title_terms, text_terms=text_terms, limit=4)
        if not matched:
            return
        units.append(
            _paper_unit(
                unit_id=unit_id,
                unit_type=unit_type,
                statement=statement,
                chunks=matched,
                surfaces=surfaces,
                obligations=obligations,
                interfaces=interfaces,
                artifacts=artifacts,
                module_kinds=module_kinds,
                hypothesis=hypothesis,
                decision_value=decision_value,
                stop_rule_or_pruning_rationale=stop_rule_or_pruning_rationale,
            )
        )

    if ("forward-optimization" in lowered or "foa" in lowered) and ("cma" in lowered or "prompt" in lowered):
        add(
            unit_id="paper_forward_optimization_adaptation",
            unit_type="method",
            statement="Implement the paper's forward-only adaptation method and prompt optimizer.",
            title_terms=("forward-only prompt adaptation", "method", "proposed", "algorithm"),
            text_terms=("forward-optimization", "foa", "cma", "prompt", "fitness function", "backpropagation-free"),
            surfaces=["model_or_method", "adaptation", "optimizer", "config", "tests"],
            obligations=[
                "Implement a forward-pass-only adaptation path that does not require backpropagation or model weight updates.",
                "Implement learnable input prompts with a code-visible token arrangement when the paper specifies one.",
                "Implement or wrap the derivative-free CMA/CMA-ES optimizer with population sampling, fitness evaluation, and distribution updates.",
                "Implement the unsupervised fitness terms stated by the paper, including prediction entropy and source/test activation-statistic discrepancy when present.",
                "Expose default and sweepable adaptation hyperparameters through config/registry entries rather than hard-coded constants.",
            ],
            interfaces=["adapt(model, batch, config)", "prompt optimizer config", "fitness function"],
            artifacts=["results/adaptation_trace.json", "results/config_resolved.json"],
            module_kinds=["model_or_method", "optimization", "configuration", "tests"],
            hypothesis="The core contribution is that adaptation can be done through forward-only prompt optimization rather than gradient-based model updates.",
            decision_value="Determines whether generation must implement a real method core or only evaluation scaffolding.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve the optimizer and config surface with bounded smoke execution for online adaptation streams.",
        )

    if "activation shifting" in lowered or "back-to-source" in lowered:
        add(
            unit_id="paper_activation_shifting",
            unit_type="method",
            statement="Implement the paper's activation shifting or source-statistic alignment mechanism.",
            title_terms=("activation shifting", "method", "proposed", "algorithm"),
            text_terms=("activation shifting", "back-to-source", "source statistics", "cls", "moving average"),
            surfaces=["model_or_method", "adaptation", "metric_formula", "config"],
            obligations=[
                "Compute and store source-domain activation statistics required by the paper.",
                "Implement test-time activation shifting/alignment using the paper's dynamic direction or moving-average update rule when stated.",
                "Expose activation-layer, step-size, moving-average, and source-statistic sample-count configuration.",
                "Keep activation editing separate from prompt optimization so ablations can enable or disable each component.",
            ],
            interfaces=["source statistics store", "activation_shift(features, config)", "ablation switches"],
            artifacts=["results/source_statistics.json", "results/adaptation_trace.json"],
            module_kinds=["model_or_method", "data_pipeline", "configuration", "tests"],
            hypothesis="The activation alignment component is a separate mechanism claimed to improve adaptation beyond prompt optimization alone.",
            decision_value="Determines whether ablation and component toggles can test the paper's mechanism rather than a monolithic method.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve paper-stated activation statistics, update rules, and ablation toggles.",
        )

    if "quantized" in lowered or "8-bit" in lowered or "6-bit" in lowered:
        add(
            unit_id="paper_quantized_model_protocol",
            unit_type="protocol",
            statement="Implement quantized-model preparation and evaluation hooks when the paper evaluates low-precision models.",
            title_terms=("experiments", "implementation details", "quantized", "evaluation"),
            text_terms=("quantized", "8-bit", "6-bit", "ptq", "low precision", "memory"),
            surfaces=["model_or_method", "evaluation", "config", "artifact_writer"],
            obligations=[
                "Expose model precision/quantization variants as registry or config entries.",
                "Implement preparation hooks for quantized models without requiring full quantization during smoke generation.",
                "Bind quantized variants to the same evaluation metrics and result artifacts as full-precision models.",
                "Record memory or runtime-efficiency metrics when the paper uses them as a contribution claim.",
            ],
            interfaces=["model precision registry", "quantization preparation hook", "evaluation command"],
            artifacts=["results/model_registry.json", "results/metrics.json"],
            module_kinds=["model_or_method", "evaluation", "configuration", "reporting"],
            hypothesis="The paper's practicality claim depends on preserving low-precision model variants alongside full-precision evaluation.",
            decision_value="Determines which model registry variants and efficiency artifacts must be generated.",
            stop_rule_or_pruning_rationale="Implementation scope: expose full quantization hooks with smoke-safe preparation defaults and explicit full-conversion mode.",
        )

    if (
        "adversarial" in lowered
        and ("pgd" in lowered or "apgd" in lowered or "autoattack" in lowered)
        and any(token in lowered for token in ("robust accuracy", "attack setup", "perturbation", "epsilon"))
    ):
        add(
            unit_id="paper_adversarial_robustness_protocol",
            unit_type="protocol",
            statement="Implement adversarial attack and robustness-evaluation protocol described by the paper.",
            title_terms=("attack setup", "evaluation", "experiments", "adversarial"),
            text_terms=("adversarial", "attack", "pgd", "apgd", "epsilon", "robust"),
            surfaces=["evaluation", "attack_method", "metric_formula", "artifact_writer", "config"],
            obligations=[
                "Expose attack algorithms, precision/mode choices, perturbation budgets, and iteration counts as config-visible protocol entries.",
                "Implement clean and adversarial metric aggregation for every model/dataset combination claimed by the paper.",
                "Represent transfer, targeted, or untargeted attack variants separately when the paper distinguishes them.",
                "Bind robustness trends and table/figure artifacts to attack settings through attack-specific evaluator routes.",
            ],
            interfaces=["attack registry", "run_attack(config)", "robustness evaluation command"],
            artifacts=["results/attack_registry.json", "results/metrics.json"],
            module_kinds=["evaluation", "configuration", "reporting", "tests"],
            hypothesis="The robustness claim is decided by attack protocol fidelity and attack-parameter coverage, not just model definitions.",
            decision_value="Determines which attack methods and perturbation budgets must be implemented before robustness results are credible.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve paper-stated attack variants and budgets with bounded smoke fixtures.",
        )

    if "simformer" in lowered and ("condition mask" in lowered or "m_c" in lowered):
        add(
            unit_id="paper_simformer_tokenizer_condition_masks",
            unit_type="method",
            statement="Implement Simformer tokenization and condition-mask sampling for arbitrary conditionals.",
            title_terms=("simformer architecture", "tokenization", "training", "method"),
            text_terms=("condition mask", "m_c", "tokenizer", "identifier", "condition state", "posterior mask", "likelihood mask"),
            surfaces=["model_or_method", "data_pipeline", "config", "tests"],
            obligations=[
                "Represent each parameter/data variable as tokens containing identifier, value, metadata when available, and condition-state embedding.",
                "Implement condition-mask sampling for joint, posterior, likelihood, and Bernoulli random masks using paper/addendum probabilities.",
                "Expose posterior/likelihood/arbitrary conditional query masks through config-visible interfaces.",
                "Support function-valued parameters or time/space metadata through identifier/Fourier metadata embeddings when present.",
            ],
            interfaces=["tokenize(theta, x, metadata, condition_mask)", "sample_condition_mask(batch_size, config)"],
            artifacts=["results/mask_policy.json", "results/tokenizer_registry.json"],
            module_kinds=["model_or_method", "data_pipeline", "configuration", "tests"],
            hypothesis="Simformer flexibility depends on token-level conditioning, not a fixed posterior-only density estimator.",
            decision_value="Determines whether downstream generation preserves all-in-one posterior, likelihood, and arbitrary conditional sampling.",
            stop_rule_or_pruning_rationale="Implementation scope: keep mask policies explicit and smoke-testable, with conditional variants represented through config/selectors.",
        )

    if "simformer" in lowered and ("attention mask" in lowered or "graph inversion" in lowered or "m_e" in lowered):
        add(
            unit_id="paper_simformer_attention_graph_masks",
            unit_type="method",
            statement="Implement Simformer dependency-aware attention masks and graph inversion updates.",
            title_terms=("graph inversion", "attention mask", "dependency", "method", "appendix"),
            text_terms=("attention mask", "m_e", "graph inversion", "directed", "undirected", "dependency", "mask"),
            surfaces=["model_or_method", "config", "tests"],
            obligations=[
                "Represent directed and undirected simulator dependency masks for task variables.",
                "Implement graph inversion/update logic that adds dependencies induced by observed/conditioned variables to the base attention mask.",
                "Expose task-specific mask constructors for Gaussian linear, two moons, Gaussian mixture, SLCP, HMM, and Lotka-Volterra style dependencies when stated.",
            ],
            interfaces=["build_attention_mask(task, condition_mask, metadata)", "invert_graph_dependencies(base_mask, condition_mask)"],
            artifacts=["results/attention_mask_registry.json"],
            module_kinds=["model_or_method", "configuration", "tests"],
            hypothesis="Using known simulator dependencies in attention masks is a core Simformer efficiency claim.",
            decision_value="Determines whether plan/generation can reproduce structured-mask variants instead of a generic transformer.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve mask constructors and graph inversion logic for the paper-stated dependency structures.",
        )

    if "simformer" in lowered and ("score-based diffusion" in lowered or "vesde" in lowered or "vpsde" in lowered):
        add(
            unit_id="paper_simformer_diffusion_training_sampling",
            unit_type="method",
            statement="Implement Simformer score-based diffusion training and reverse conditional sampling.",
            title_terms=("score-based diffusion", "training and sampling", "extended benchmark", "method"),
            text_terms=("score-based diffusion", "denoising score", "reverse diffusion", "vesde", "vpsde", "probability flow ode"),
            surfaces=["model_or_method", "training_loop", "evaluation", "config"],
            obligations=[
                "Implement denoising score-matching loss with clean conditioned variables and noisy unconditioned variables.",
                "Support VESDE/VPSDE-style SDE configuration and reverse diffusion sampling for arbitrary conditionals.",
                "Expose evaluation-step count for reverse SDE/ODE sampling and preserve the paper's efficiency threshold checks.",
            ],
            interfaces=["train_score_model(batch, mask, sde_config)", "sample_conditional(observed, condition_mask, sde_config)"],
            artifacts=["results/diffusion_config.json", "results/sampling_trace.json"],
            module_kinds=["model_or_method", "training", "evaluation", "configuration"],
            hypothesis="The all-in-one estimator is a transformer score model trained through conditional diffusion, not a standard SBI density estimator.",
            decision_value="Determines whether reproduction captures the main model family and sampling algorithm.",
            stop_rule_or_pruning_rationale="Implementation scope: expose full SDE/sampling contracts with tiny-dimension bounded smoke defaults.",
        )

    if "simformer" in lowered and ("lotka" in lowered or "hodgkin" in lowered or "sird" in lowered or "slcp" in lowered):
        add(
            unit_id="paper_simformer_sbi_task_registry",
            unit_type="task",
            statement="Implement Simformer SBI task registry and benchmark-specific simulators/summary statistics.",
            title_terms=("tasks", "task dependencies", "benchmark", "applications", "appendix"),
            text_terms=("two moons", "gaussian mixture", "slcp", "lotka", "sird", "hodgkin", "summary statistics"),
            surfaces=["data_pipeline", "environment", "evaluation", "config", "tests"],
            obligations=[
                "Expose task registry entries for Gaussian Linear, Two Moons, Gaussian Mixture, SLCP, HMM, Lotka-Volterra, SIRD, Hodgkin-Huxley, and gravitational-wave tasks when present.",
                "Implement or stub simulator/summary-statistic contracts with task-specific parameter/data dimensions, priors, observation grids, and dependency masks.",
                "Use the sbi library for NPE, NRE, and NLE baselines as required by the addendum, with lazy bounded smoke fixtures.",
                "Implement C2ST with a random forest classifier using 100 trees for benchmark comparison.",
            ],
            interfaces=["task registry", "simulate(task, theta, config)", "compute_summary_statistics(task, x)"],
            artifacts=["results/task_registry.json", "results/c2st_metrics.json"],
            module_kinds=["data_pipeline", "evaluation", "configuration", "tests"],
            hypothesis="The method's claims are benchmark-dependent; missing task contracts collapse the paper into a generic diffusion demo.",
            decision_value="Determines which simulators, baselines, and metrics must exist before plan can be useful.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve full task metadata and metric contracts with small smoke fixtures for expensive simulators.",
        )


    has_diffusion_transfer_context = bool(
        ("diffusion" in lowered or "ddpm" in lowered or "ldm" in lowered or "dpm" in lowered)
        and ("few-shot" in lowered or "10-shot" in lowered or "transfer" in lowered or "source domain" in lowered or "target domain" in lowered)
    )

    if has_diffusion_transfer_context and ("similarity-guided" in lowered or "similarity guided" in lowered or "adversarial noise" in lowered):
        add(
            unit_id="paper_diffusion_ant_training_objective",
            unit_type="method",
            statement="Implement diffusion transfer learning with similarity-guided training and adversarial noise selection.",
            title_terms=("method", "similarity-guided", "adversarial noise", "algorithm"),
            text_terms=("similarity-guided", "adversarial noise", "binary classifier", "adapter", "ddpm loss", "min-max"),
            surfaces=["model_or_method", "training_loop", "config", "tests"],
            obligations=[
                "Implement similarity-guided diffusion loss using a fixed source/target classifier signal to guide target-domain adaptation.",
                "Implement adversarial noise selection as an inner maximization over Gaussian noise before the denoising loss update.",
                "Fine-tune only the paper-stated adaptor or shift-module parameters while keeping the pretrained diffusion backbone frozen when configured.",
                "Expose DDPM and LDM variants through config-visible method entries rather than collapsing both into one generic diffusion demo.",
            ],
            interfaces=["similarity_guided_loss(batch, classifier, config)", "select_adversarial_noise(batch, model, config)", "train_ant_step(batch, config)"],
            artifacts=["results/ant_training_trace.json", "results/method_registry.json", "results/config_resolved.json"],
            module_kinds=["model_or_method", "training", "configuration", "tests"],
            hypothesis="The core contribution is the combination of classifier-guided transfer direction and adaptive worst-case noise selection, not generic fine-tuning.",
            decision_value="Determines whether plan/generation captures the paper's causal method components before implementation.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve the loss, inner-loop, frozen-backbone, and adaptor contracts with tiny smoke training defaults.",
        )
        add(
            unit_id="paper_diffusion_fewshot_domain_registry",
            unit_type="task",
            statement="Implement few-shot source-to-target image generation domain registry for diffusion transfer experiments.",
            title_terms=("datasets", "experiment setup", "few-shot", "evaluation"),
            text_terms=("ffhq", "lsun church", "babies", "sunglasses", "raphael", "sketches", "modigliani", "haunted houses", "landscape drawings"),
            surfaces=["data_pipeline", "evaluation", "config", "artifact_writer", "tests"],
            obligations=[
                "Expose source domains FFHQ and LSUN Church and their paper-stated 10-shot target domains as registry rows.",
                "Represent source-to-target pairs such as FFHQ to Babies/Sunglasses/Raphael/Sketches/Modigliani and LSUN Church to Haunted Houses/Landscape drawings.",
                "Keep dataset downloads lazy and provide smoke fixtures/manifests for target exemplars and generated samples.",
            ],
            interfaces=["domain registry", "make_fewshot_dataset(pair, config)", "domain_pair_manifest(config)"],
            artifacts=["results/domain_registry.json", "results/data_manifest.json"],
            module_kinds=["data_pipeline", "evaluation", "configuration", "tests"],
            hypothesis="The transfer claim is defined by explicit source/target domain pairs and 10-shot data constraints.",
            decision_value="Determines which datasets and pairings generation must expose for faithful reproduction.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve image pair metadata, full dataset hooks, and fixture paths for bounded generation.",
        )
        add(
            unit_id="paper_diffusion_metrics_baselines_artifacts",
            unit_type="protocol",
            statement="Implement diffusion few-shot metrics, baselines, ablations, and result artifact protocol.",
            title_terms=("evaluation metrics", "baselines", "ablation", "results", "appendix"),
            text_terms=("fid", "intra-lpips", "ddpm-pa", "tgan", "ada", "ewc", "cdc", "dcl", "gamma", "omega", "training iteration"),
            surfaces=["evaluation", "baseline_or_ablation", "metric_formula", "artifact_writer", "config", "tests"],
            obligations=[
                "Implement FID and Intra-LPIPS metric aggregation contracts with generated-image manifests and lazy feature extraction.",
                "Expose baseline rows for GAN-based and DPM-based comparisons including TGAN, ADA, EWC, CDC, DCL, and DDPM-PA when present.",
                "Expose ablation switches for adaptor-only, similarity guidance without adversarial noise, full ANT, gamma/omega sweeps, and training-iteration sweeps.",
                "Persist table/figure artifacts without fabricating full benchmark scores; smoke outputs must be clearly separated from full-run artifacts.",
            ],
            interfaces=["metric registry", "baseline registry", "ablation registry", "result table writer"],
            artifacts=["results/metrics.json", "results/baseline_registry.json", "results/ablation_registry.json", "results/tables/summary.csv"],
            module_kinds=["evaluation", "reporting", "configuration", "tests"],
            hypothesis="The paper's claims are decided by diversity/fidelity metrics plus method/baseline/ablation contrasts.",
            decision_value="Determines whether plan/generation covers the decisive experimental comparisons with bounded execution routes.",
            stop_rule_or_pruning_rationale="Implementation scope: represent all high-information comparisons and sweeps with bounded fixtures plus explicit full-experiment routes.",
        )

    has_rl_context = bool(
        re.search(
            r"\b(rl|reinforcement learning|policy gradient|ppo|on-policy|off-policy|actor-critic|q-learning|episode reward|gym environment)\b",
            lowered,
        )
    )

    if (
        has_rl_context
        and ("policy gradient" in lowered or "ppo" in lowered or "on-policy" in lowered)
        and ("off-policy" in lowered or "importance sampling" in lowered)
        and ("multiple policies" in lowered or "leader" in lowered or "followers" in lowered or "parallel environments" in lowered)
    ):
        add(
            unit_id="paper_rl_multi_policy_offpolicy_aggregation",
            unit_type="method",
            statement="Implement the paper's multi-policy on-policy RL update with off-policy data aggregation.",
            title_terms=("split and aggregate", "aggregating data", "method", "algorithm", "policy gradients"),
            text_terms=("off-policy updates", "importance sampling", "multiple policies", "leader", "followers", "lambda", "clipped surrogate"),
            surfaces=["model_or_method", "training_loop", "baseline_or_ablation", "config", "tests"],
            obligations=[
                "Represent M policy blocks sharing a trainer interface instead of collapsing the method to a single PPO policy.",
                "Implement on-policy PPO-style clipped surrogate updates and the paper-stated off-policy importance-weighted aggregation term.",
                "Expose leader/follower or policy-index update sets, off-policy source sets, lambda/off-policy ratio, and subsampling controls through config.",
                "Implement critic target handling for on-policy n-step returns and off-policy one-step returns when the paper distinguishes them.",
                "Persist update diagnostics separating on-policy loss, off-policy loss, critic loss, and aggregation/source-policy metadata.",
            ],
            interfaces=["multi-policy trainer", "compute_on_policy_loss(batch)", "compute_off_policy_loss(target_policy, source_batches)", "aggregation config"],
            artifacts=["results/update_traces.json", "results/method_registry.json", "results/config_resolved.json"],
            module_kinds=["model_or_method", "training", "configuration", "tests"],
            hypothesis="The paper's contribution is the decision to reuse data across concurrently sampled policies while preserving an on-policy update structure.",
            decision_value="Determines whether generation implements the core contribution rather than a generic PPO training loop.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve RL update equations, config surfaces, and smoke-testable batch flow with explicit full-training routes.",
        )

    if has_rl_context and (("leader" in lowered and "follower" in lowered) or "symmetric aggregation" in lowered or "symmetric off-policy" in lowered):
        add(
            unit_id="paper_rl_leader_follower_aggregation_variants",
            unit_type="method",
            statement="Implement leader-follower and symmetric aggregation variants as explicit ablation switches.",
            title_terms=("leader-follower", "symmetric aggregation", "ablation", "analysis", "method"),
            text_terms=("leader", "followers", "symmetric", "all other policies", "off-policy data", "ablation"),
            surfaces=["model_or_method", "baseline_or_ablation", "training_loop", "config", "tests"],
            obligations=[
                "Encode the leader-only aggregation variant where policy i=1 consumes follower data and followers remain on-policy.",
                "Encode the symmetric aggregation variant where each policy can consume data from all other policies.",
                "Keep variant toggles code-visible so result artifacts can compare canonical, symmetric, and no/off-policy variants.",
                "Record source-policy sets and target-policy sets per update for semantic review.",
            ],
            interfaces=["aggregation variant registry", "select_source_policies(policy_index, variant, M)", "ablation config"],
            artifacts=["results/ablation_registry.json", "results/update_source_sets.json"],
            module_kinds=["model_or_method", "training", "configuration", "tests"],
            hypothesis="The leader/follower design is a causal choice tested by ablations, so it must remain distinguishable from symmetric aggregation.",
            decision_value="Supports decisions about which aggregation structure explains the reported performance rather than only final scores.",
            stop_rule_or_pruning_rationale="Implementation scope: represent all paper-stated aggregation variants with bounded smoke updates and explicit full-experiment routes.",
        )

    if has_rl_context and "entropy" in lowered and ("sigma" in lowered or "exploration" in lowered) and ("followers" in lowered or "policies" in lowered):
        add(
            unit_id="paper_rl_entropy_diversity_schedule",
            unit_type="method",
            statement="Implement follower entropy-diversity regularization and its sweepable coefficient schedule.",
            title_terms=("entropy", "diversity", "ablation", "training", "hyperparameters"),
            text_terms=("entropy", "sigma", "exploration", "followers", "coefficient", "diversity"),
            surfaces=["training_loop", "model_or_method", "config", "baseline_or_ablation", "tests"],
            obligations=[
                "Apply entropy regularization to follower policies separately from the leader when the paper states that distinction.",
                "Expose sigma/entropy coefficient values and per-environment overrides as config-visible sweep entries.",
                "Keep entropy-on/off and coefficient variants available for ablation artifacts.",
                "Record chosen coefficients in resolved config and training traces.",
            ],
            interfaces=["entropy schedule config", "policy_loss_with_entropy(policy_index, config)", "sweep registry"],
            artifacts=["results/sensitivity_report.json", "results/config_resolved.json"],
            module_kinds=["training", "configuration", "model_or_method", "tests"],
            hypothesis="The paper attributes part of performance to data diversity from follower entropy regularization.",
            decision_value="Determines whether generation can explain entropy/sigma ablation trends instead of hard-coding a single method path.",
            stop_rule_or_pruning_rationale="Implementation scope: expose the sweep and defaults with bounded generation coverage and explicit full-grid routes.",
        )

    if (
        has_rl_context
        and ("environment" in lowered or "task" in lowered or "simulator" in lowered)
        and ("successes" in lowered or "episode reward" in lowered or "reward" in lowered)
        and ("allegro" in lowered or "shadow hand" in lowered or "regrasping" in lowered or "reorientation" in lowered or "throw" in lowered or "isaacgym" in lowered or "parallel environments" in lowered)
    ):
        add(
            unit_id="paper_rl_task_metric_registry",
            unit_type="task",
            statement="Implement paper-specific RL task/environment and metric registry.",
            title_terms=("task", "environment", "experiment setup", "results", "appendix"),
            text_terms=("regrasping", "throw", "reorientation", "shadow hand", "allegro", "successes", "episode reward", "parallel environments"),
            surfaces=["environment", "metric_formula", "evaluation", "config", "tests"],
            obligations=[
                "Create task registry rows for each named RL task/environment family and preserve aliases from the paper.",
                "Bind each task to its metric type, such as success count/success rate or episode reward, without collapsing metrics into a generic reward field.",
                "Expose lazy simulator availability checks and smoke fixtures for high-cost GPU environments.",
                "Record task metadata such as observation/action hints, goal semantics, horizon/curriculum hints, and in-scope metric artifacts when available.",
            ],
            interfaces=["task registry", "make_environment(task_id, config)", "compute_task_metric(task_id, trajectories)"],
            artifacts=["results/environment_registry.json", "results/metrics.json", "results/environment_readiness.json"],
            module_kinds=["environment", "evaluation", "configuration", "tests"],
            hypothesis="RL reproduction fidelity depends on preserving task-specific metric semantics alongside the reusable trainer.",
            decision_value="Determines which environment adapters, fixtures, and metric functions generation must implement first.",
            stop_rule_or_pruning_rationale="Implementation scope: keep task contracts and smoke fixtures executable while exposing full GPU simulator hooks.",
        )

    if has_rl_context and ("baseline" in lowered or "baselines" in lowered) and ("ppo" in lowered or "pql" in lowered or "pbt" in lowered or "dexpbt" in lowered):
        add(
            unit_id="paper_rl_baseline_comparison_protocol",
            unit_type="protocol",
            statement="Implement paper-stated RL baselines and comparison registry.",
            title_terms=("baselines", "results", "related work", "experiment setup", "evaluation"),
            text_terms=("baseline", "ppo", "pql", "pbt", "dexpbt", "comparison", "table"),
            surfaces=["baseline_or_ablation", "evaluation", "training_loop", "config", "tests"],
            obligations=[
                "Expose separate baseline entries for PPO-style on-policy training, population/PBT-style variants, and off-policy/PQL-style variants when stated.",
                "Keep baseline configs distinct from the proposed method so comparison artifacts cannot reuse one generic trainer silently.",
                "Bind each baseline to task metrics and result table/figure artifacts.",
                "Allow unavailable external baselines to degrade to documented adapters or fixtures while preserving their interface and config contract.",
            ],
            interfaces=["baseline registry", "make_baseline(name, config)", "run_comparison(config)"],
            artifacts=["results/baseline_registry.json", "results/tables/baseline_comparison.csv", "results/metrics.json"],
            module_kinds=["evaluation", "training", "configuration", "tests"],
            hypothesis="The paper's main claim is comparative; missing baselines make method implementation insufficient for semantic scoring.",
            decision_value="Determines whether plan/generation can reproduce the decision that the proposed method outperforms named alternatives.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve baseline interfaces and configs with bounded default generation and explicit full-training routes.",
        )

    if has_rl_context and ("figure" in lowered or "table" in lowered) and ("ablation" in lowered or "variant" in lowered or "sensitivity" in lowered or "performance curves" in lowered):
        add(
            unit_id="paper_rl_result_ablation_artifacts",
            unit_type="protocol",
            statement="Implement RL result, ablation, and sensitivity artifact protocol.",
            title_terms=("results", "analysis", "ablation", "figure", "table"),
            text_terms=("figure", "table", "ablation", "variant", "performance curves", "sensitivity", "outperforms"),
            surfaces=["evaluation", "baseline_or_ablation", "artifact_writer", "config", "tests"],
            obligations=[
                "Represent main comparison tables/curves and ablation figures as explicit artifact contracts with provenance fields.",
                "Expose variants such as no off-policy, high off-policy ratio, symmetric aggregation, entropy coefficients, and canonical method when stated.",
                "Record expected trend obligations such as baseline outperformance or worse ablation performance without fabricating benchmark scores.",
                "Generate smoke/demo artifacts from fixtures separately from full-run result artifacts.",
            ],
            interfaces=["experiment registry", "result aggregator", "artifact writer"],
            artifacts=["results/experiment_registry.json", "results/artifact_manifest.json", "results/tables/summary.csv", "results/figures/ablation_curves.png"],
            module_kinds=["evaluation", "reporting", "configuration", "tests"],
            hypothesis="A faithful reproduction must preserve which experimental contrasts decide the method's contribution.",
            decision_value="Lets plan/generation prioritize high-information comparisons instead of filling generic tables.",
            stop_rule_or_pruning_rationale="Implementation scope: encode all decisive variants and trends through registries, fixtures, and explicit full-run selectors.",
        )

    if (
        has_rl_context
        and ("hyperparameter" in lowered or "learning rate" in lowered or "batch" in lowered or "horizon" in lowered or "mini-batch" in lowered)
        and ("ppo" in lowered or "policy" in lowered or "reinforcement learning" in lowered or "environment" in lowered)
    ):
        add(
            unit_id="paper_rl_training_hyperparameter_protocol",
            unit_type="protocol",
            statement="Implement RL training architecture and hyperparameter protocol from the paper.",
            title_terms=("training hyperparameters", "implementation details", "experiment setup", "appendix"),
            text_terms=("learning rate", "discount", "tau", "horizon", "mini-batch", "clip", "lstm", "mlp", "num_envs"),
            surfaces=["training_loop", "model_or_method", "config", "artifact_writer", "tests"],
            obligations=[
                "Expose optimizer, discount, GAE/tau, clipping, horizon, minibatch, epoch, and gradient norm settings through resolved configs.",
                "Preserve policy architecture differences such as recurrent versus MLP policies when paper tasks require them.",
                "Separate smoke defaults from paper/full-run settings so cost-saving tests preserve the stated protocol.",
                "Persist resolved per-task training config artifacts for review.",
            ],
            interfaces=["training config schema", "policy factory", "resolved config writer"],
            artifacts=["results/config_resolved.json", "results/training_trace.json"],
            module_kinds=["training", "model_or_method", "configuration", "tests"],
            hypothesis="Hyperparameter and architecture fidelity controls whether the RL method is semantically tied to the paper's training regime.",
            decision_value="Determines which settings are mandatory code/config obligations and which full sweeps can be deferred.",
            stop_rule_or_pruning_rationale="Implementation scope: keep full-scale settings explicit with small smoke fixtures by default.",
        )

    if "bbox-adapter" in lowered or "bbox adapter" in lowered or "bbox-adapter" in lowered:
        add(
            unit_id="paper_bbox_energy_adapter_nce",
            unit_type="method",
            statement="Implement BBox-Adapter energy model and ranking-based NCE adapter update.",
            title_terms=("adapter update", "black-box llm adaptation as ebm", "method"),
            text_terms=("energy-based", "ebm", "ranking-based nce", "noise contrastive", "spectral normalization", "adapter update"),
            surfaces=["model_or_method", "training_loop", "config", "tests"],
            obligations=[
                "Implement a smaller white-box adapter scoring function g_theta(x, y) over black-box LLM outputs.",
                "Implement ranking-based NCE loss that ranks target/positive responses above source or adapted negative generations.",
                "Include spectral normalization or equivalent bounded-score stabilization when configured.",
                "Preserve the black-box contract: no model weights, hidden states, or output probabilities from the base LLM are required.",
            ],
            interfaces=["adapter.score(prompt, response)", "ranking_nce_loss(positive, negatives)", "train_adapter(batch)"],
            artifacts=["results/adapter_training_trace.json", "results/loss_curves.json"],
            module_kinds=["model_or_method", "training", "configuration", "tests"],
            hypothesis="BBox-Adapter adapts black-box LLM generations by learning an energy adapter through ranking NCE.",
            decision_value="Determines whether generation implements the paper's actual adapter rather than generic prompt tuning.",
            stop_rule_or_pruning_rationale="Implementation scope: keep base LLM calls mockable and bounded while preserving explicit adapter scoring/training logic.",
        )
        add(
            unit_id="paper_bbox_sentence_beam_inference",
            unit_type="method",
            statement="Implement BBox-Adapter sentence-level beam search with adapter scoring.",
            title_terms=("adaptive inference", "model inference", "method"),
            text_terms=("sentence-level beam search", "beam size", "candidate set", "adapter assigns scores", "top-k"),
            surfaces=["model_or_method", "evaluation", "artifact_writer", "config"],
            obligations=[
                "Generate n candidates per beam from the black-box proposal generator for each sentence step.",
                "Score partial chains with the adapter and keep top-k beams until stop signal or maximum steps.",
                "Persist candidate/score traces so inference decisions are inspectable without black-box probabilities.",
            ],
            interfaces=["generate_candidates(prompt, prefix, n)", "beam_search_with_adapter(prompt, config)"],
            artifacts=["results/beam_search_traces.json", "results/predictions.jsonl"],
            module_kinds=["model_or_method", "evaluation", "reporting", "configuration"],
            hypothesis="Adapted inference is performed by adapter-guided candidate selection, not by modifying the black-box LLM.",
            decision_value="Determines whether downstream evaluation can reproduce the paper's inference-time mechanism.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve the scoring and top-k selection contract with mock/local generators for smoke tests.",
        )
        add(
            unit_id="paper_bbox_online_feedback_loop",
            unit_type="protocol",
            statement="Implement BBox-Adapter online adaptation with positive/negative feedback pools.",
            title_terms=("online adaptation", "algorithm", "method", "ai feedback"),
            text_terms=("online adaptation", "positive", "negative", "ai feedback", "human feedback", "previous adaptations"),
            surfaces=["training_loop", "data_pipeline", "evaluation", "artifact_writer", "config"],
            obligations=[
                "Maintain positive and negative sample pools across online adaptation iterations.",
                "Update positives from ground truth, human feedback, or AI feedback and negatives from previous adapted generations.",
                "Run iterative sample-update-train cycles and record positive/negative score curves.",
            ],
            interfaces=["feedback selector", "online_adapt(dataset, generator, adapter, config)"],
            artifacts=["results/online_adaptation_log.json", "results/positive_negative_curves.json"],
            module_kinds=["training", "data_pipeline", "evaluation", "reporting"],
            hypothesis="The paper's no-ground-truth and self-improvement claims rely on online feedback-driven pool updates.",
            decision_value="Determines whether plan/generation covers static supervised and online AI-feedback settings.",
            stop_rule_or_pruning_rationale="Implementation scope: keep the feedback pool state machine explicit with bounded online rounds for smoke tests.",
        )
        add(
            unit_id="paper_bbox_qa_benchmark_registry",
            unit_type="task",
            statement="Implement BBox-Adapter QA/toxicity benchmark registry and metric protocol.",
            title_terms=("datasets", "experimental setup", "appendix", "main results"),
            text_terms=("gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen", "mixtral", "gpt-3.5", "vram"),
            surfaces=["data_pipeline", "evaluation", "metric_formula", "artifact_writer", "config", "tests"],
            obligations=[
                "Expose dataset registry entries for GSM8K, StrategyQA, TruthfulQA, ScienceQA, and ToxiGen when present, with train/test sample counts and prompt templates.",
                "Implement accuracy/toxicity/cost/VRAM metric aggregation and table artifacts for BBox-Adapter, SFT-LoRA, Azure-SFT, and base-model comparisons.",
                "Apply the addendum scope: VRAM measurements are required only for the 0.1B adapter version.",
                "Preserve black-box model adapters for GPT-3.5-turbo and Mixtral-8x7B style proposal generators through mockable interfaces.",
            ],
            interfaces=["dataset registry", "evaluate_predictions(dataset, predictions)", "cost_vram_report(config)"],
            artifacts=["results/dataset_registry.json", "results/metrics.json", "results/cost_vram_report.json"],
            module_kinds=["data_pipeline", "evaluation", "configuration", "reporting", "tests"],
            hypothesis="The paper's performance and cost claims depend on concrete QA/toxicity datasets and metric tables.",
            decision_value="Determines which loaders, prompts, metrics, and artifacts must be present before generation.",
            stop_rule_or_pruning_rationale="Implementation scope: preserve prompt, dataset, and metric contracts exactly with fixtures for expensive/API-backed benchmarks.",
        )

    return units


def _contract_item_names(contract: dict[str, Any], key: str) -> list[str]:
    return _ordered_unique(
        [
            str(item.get("name", "") or "").strip()
            for item in list(contract.get(key, []) or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        ]
    )


def _contract_parameter_rows(contract: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for item in list(contract.get("parameter_sweeps", []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        values = _ordered_unique([str(value) for value in list(item.get("values", []) or []) if str(value).strip()])
        rows.append(f"{name} values={','.join(values)}" if values else name)
    return rows


def _contract_trend_rows(contract: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for item in list(contract.get("trend_obligations", []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        if name == "endpoint_low":
            rows.append("endpoint_low: p=0 and p=1 must be represented as lowest/minimum boundary cases")
        elif name == "sweep_insensitive":
            rows.append("sweep_insensitive: parameter sweep should preserve stable/insensitive trend claim")
        elif name == "baseline_outperformance":
            rows.append("baseline_outperformance: proposed method should be compared against explicit baselines")
        elif name == "positive_parameter_improves":
            rows.append("positive_parameter_improves: nonzero/positive parameter values should preserve the reported improvement trend")
        else:
            rows.append(name)
    return rows


def _contract_metric_rows(contract: dict[str, Any]) -> list[str]:
    return _contract_item_names(contract, "metrics")


def _contract_artifact_rows(contract: dict[str, Any]) -> list[str]:
    artifact_names = _contract_item_names(contract, "artifacts")
    return _ordered_unique(artifact_names)



def _paper_contract_specialized_units(state: PaperBenchReproState) -> list[ExtractedUnit]:
    """Split inferred paper evidence contract into actionable implementation units."""
    paperbench_text = _paperbench_input_text(state)
    if not paperbench_text.strip():
        return []
    contract = infer_evidence_contract(paperbench_text)
    chunks = list(state.paper_chunks or [])
    named_experiments = _contract_item_names(contract, "named_experiments")
    environments = _contract_item_names(contract, "environments")
    datasets = _contract_item_names(contract, "datasets")
    methods = _contract_item_names(contract, "methods")
    metrics = _contract_metric_rows(contract)
    artifacts = _contract_artifact_rows(contract)
    parameters = _contract_parameter_rows(contract)
    trends = _contract_trend_rows(contract)
    protocols = _contract_item_names(contract, "protocol_obligations")
    fixed = _contract_item_names(contract, "fixed_hyperparameters")
    units: list[ExtractedUnit] = []

    def _chunks(title_terms: tuple[str, ...], text_terms: tuple[str, ...]) -> list[PaperChunk]:
        matched = _find_paper_chunks(chunks, title_terms=title_terms, text_terms=text_terms, limit=4)
        if matched:
            return matched
        return _find_paper_chunks(
            chunks,
            title_terms=("method", "experiment", "evaluation", "appendix", "benchmark"),
            text_terms=text_terms,
            limit=3,
        )

    def _add_contract_unit(
        *,
        unit_id: str,
        unit_type: str,
        statement: str,
        inventory_label: str,
        inventory: list[str],
        title_terms: tuple[str, ...],
        text_terms: tuple[str, ...],
        surfaces: list[str],
        obligations: list[str],
        interfaces: list[str],
        artifacts_out: list[str],
        module_kinds: list[str],
        hypothesis: str,
        decision_value: str,
        stop_rule_or_pruning_rationale: str,
    ) -> None:
        items = _ordered_unique([str(item) for item in inventory if str(item).strip()])
        if not items:
            return
        matched = _chunks(title_terms, text_terms)
        evidence = [f"paper evidence contract {inventory_label}: " + "; ".join(items[:24])]
        if matched:
            evidence.extend(_chunk_preview(chunk, max_chars=1200) for chunk in matched[:3])
            source_ids = _ordered_unique([chunk.chunk_id for chunk in matched] + ["paper.md", "addendum.md"])
        else:
            preview = " ".join(paperbench_text.split())
            evidence.append(preview[:1600])
            source_ids = ["paper.md", "addendum.md"]
        units.append(
            ExtractedUnit(
                unit_id=unit_id,
                type=unit_type,
                statement=statement,
                hypothesis=hypothesis,
                decision_value=decision_value,
                stop_rule_or_pruning_rationale=stop_rule_or_pruning_rationale,
                paper_evidence=evidence[:4],
                source_paragraph_ids=source_ids,
                citation_refs=[],
                verification_targets=[
                    VerificationTarget(
                        kind="artifact",
                        description=f"Generated code exposes {inventory_label} from the paper evidence contract.",
                    )
                ],
                implementation_surfaces=_ordered_unique(surfaces),
                code_obligations=_ordered_unique(obligations),
                runtime_interfaces=_ordered_unique(interfaces),
                expected_artifacts=_ordered_unique(artifacts_out),
                suggested_module_kinds=_ordered_unique(module_kinds),
                implementation_notes=[
                    f"contract_inventory:{inventory_label}: " + "; ".join(items[:24]),
                    "Deterministically inferred from paper/addendum text.",
                ],
                status="active",
            )
        )

    _add_contract_unit(
        unit_id="paper_contract_dataset_metric_protocol",
        unit_type="task",
        statement="Implement paper-specific dataset, benchmark, and metric contracts.",
        inventory_label="datasets_metrics",
        inventory=[*datasets, *metrics],
        title_terms=("datasets", "benchmarks", "evaluation", "experiment setup", "metrics"),
        text_terms=("dataset", "benchmark", "metric", "accuracy", "score", "evaluation", "c2st", "nll"),
        surfaces=["data_pipeline", "evaluation", "metric_formula", "config", "tests"],
        obligations=[
            "Create explicit dataset/benchmark registry entries and aliases for: " + ", ".join(datasets[:24]) if datasets else "Create explicit benchmark registry entries from the paper.",
            "Implement metric formula or aggregation contracts for: " + ", ".join(metrics[:24]) if metrics else "Bind paper metrics to evaluation artifacts.",
            "Keep full data downloads lazy and expose smoke fixtures/readiness checks for every named benchmark.",
        ],
        interfaces=["dataset registry", "metric registry", "evaluate_predictions(config)"],
        artifacts_out=["results/dataset_registry.json", "results/metrics.json", "results/data_manifest.json"],
        module_kinds=["data_pipeline", "evaluation", "configuration", "tests"],
        hypothesis="Paper scores are decided by named datasets and metrics, not by a generic loader or accuracy placeholder.",
        decision_value="Determines whether plan/generation knows the concrete benchmark and metric surface before code writing.",
        stop_rule_or_pruning_rationale="Implementation scope: represent full benchmark runs through registries and fixtures with lazy dataset execution.",
    )
    _add_contract_unit(
        unit_id="paper_contract_method_baseline_protocol",
        unit_type="method",
        statement="Implement paper-specific method and baseline registry contracts.",
        inventory_label="methods_baselines",
        inventory=methods,
        title_terms=("method", "approach", "algorithm", "baseline", "ablation"),
        text_terms=("method", "algorithm", "baseline", "ablation", "adapter", "diffusion", "training"),
        surfaces=["model_or_method", "baseline_or_ablation", "training_loop", "config", "tests"],
        obligations=[
            "Expose code-visible method/baseline entries for: " + ", ".join(methods[:24]),
            "Separate proposed method, baselines, and ablation switches through distinct comparison routes.",
            "Bind each method entry to the datasets, metrics, and artifacts that decide its paper claim.",
        ],
        interfaces=["method registry", "baseline registry", "make_method(config)"],
        artifacts_out=["results/method_registry.json", "results/ablation_registry.json"],
        module_kinds=["model_or_method", "training", "configuration", "tests"],
        hypothesis="The reproduction is only faithful if the proposed method and comparison baselines are separately executable or inspectable.",
        decision_value="Determines which method components and baseline toggles generation must implement.",
        stop_rule_or_pruning_rationale="Implementation scope: preserve paper/addendum-stated methods and baselines.",
    )
    _add_contract_unit(
        unit_id="paper_contract_experiment_artifact_protocol",
        unit_type="protocol",
        statement="Implement paper-specific experiment, trend, and result-artifact contracts.",
        inventory_label="experiments_trends_artifacts",
        inventory=[*named_experiments, *trends, *artifacts],
        title_terms=("experiment", "results", "analysis", "ablation", "figure", "table"),
        text_terms=("experiment", "result", "figure", "table", "trend", "outperform", "ablation"),
        surfaces=["evaluation", "baseline_or_ablation", "artifact_writer", "config", "tests"],
        obligations=[
            "Represent named experiments/trends/artifacts as config-visible rows: " + "; ".join([*named_experiments, *trends, *artifacts][:24]),
            "Write result tables/figures/manifests with explicit provenance and no fabricated benchmark scores.",
            "Bind each artifact row to the method, dataset/environment, metric, and sweep dimensions needed by the paper.",
        ],
        interfaces=["experiment registry", "artifact writer", "result aggregation command"],
        artifacts_out=["results/experiment_registry.json", "results/artifact_manifest.json", "results/tables/summary.csv"],
        module_kinds=["evaluation", "reporting", "configuration", "tests"],
        hypothesis="PaperBench evaluation checks decisive paper artifact and trend representation plus importable code.",
        decision_value="Determines which result outputs and comparisons must exist before generation exits.",
        stop_rule_or_pruning_rationale="Implementation scope: preserve all artifact contracts with bounded smoke rows for repeated seeds or large sweeps.",
    )
    _add_contract_unit(
        unit_id="paper_contract_sweep_hyperparameter_protocol",
        unit_type="protocol",
        statement="Implement paper-specific sweep, protocol, and fixed-hyperparameter contracts.",
        inventory_label="sweeps_protocols_hyperparameters",
        inventory=[*parameters, *protocols, *fixed],
        title_terms=("implementation details", "hyperparameters", "training", "appendix", "protocol"),
        text_terms=("hyperparameter", "sweep", "learning rate", "batch", "epoch", "protocol", "seed"),
        surfaces=["config", "training_loop", "evaluation", "artifact_writer", "tests"],
        obligations=[
            "Expose sweep/fixed-hyperparameter config entries for: " + "; ".join([*parameters, *protocols, *fixed][:24]),
            "Keep default, smoke, and full-run modes separate so cost-saving execution preserves paper-stated settings.",
            "Persist resolved config artifacts for every experiment or evaluation command.",
        ],
        interfaces=["config schema", "sweep registry", "resolved config artifact"],
        artifacts_out=["results/config_resolved.json", "results/sensitivity_report.json"],
        module_kinds=["configuration", "training", "evaluation", "tests"],
        hypothesis="Sweep and hyperparameter fidelity controls whether a lightweight reproduction remains semantically tied to the paper.",
        decision_value="Determines which knobs generation must expose and which expensive variants can be deferred safely.",
        stop_rule_or_pruning_rationale="Implementation scope: keep full sweeps explicit and executable through bounded configs.",
    )
    _add_contract_unit(
        unit_id="paper_contract_environment_protocol",
        unit_type="task",
        statement="Implement paper-specific environment/task contracts.",
        inventory_label="environments_tasks",
        inventory=environments,
        title_terms=("environment", "task", "benchmark", "application", "experiment setup"),
        text_terms=("environment", "task", "benchmark", "simulator", "dataset", "application"),
        surfaces=["environment", "data_pipeline", "evaluation", "config", "tests"],
        obligations=[
            "Expose environment/task registry entries for: " + ", ".join(environments[:24]),
            "Keep external environment dependencies lazy and provide readiness checks/smoke fixtures.",
            "Bind every environment/task entry to the methods and metrics that use it.",
        ],
        interfaces=["environment registry", "make_environment(config)", "environment readiness check"],
        artifacts_out=["results/environment_registry.json", "results/environment_readiness.json"],
        module_kinds=["environment", "data_pipeline", "configuration", "tests"],
        hypothesis="Environment/task coverage is a first-order reproduction contract when paper claims depend on named tasks.",
        decision_value="Determines which adapters and environment/task selectors generation must implement.",
        stop_rule_or_pruning_rationale="Implementation scope: keep task contracts explicit with fixture-backed simulator coverage and full simulator hooks.",
    )
    return units


def _paper_evidence_matrix_unit(state: PaperBenchReproState) -> ExtractedUnit | None:
    paperbench_text = _paperbench_input_text(state)
    if not paperbench_text.strip():
        return None
    contract = infer_evidence_contract(paperbench_text)
    if not contract.get("requires_evidence_matrix"):
        return None
    flat_contract = flatten_evidence_contract(contract)
    named_experiments = _contract_item_names(contract, "named_experiments")
    environments = _contract_item_names(contract, "environments")
    datasets = _contract_item_names(contract, "datasets")
    methods = _contract_item_names(contract, "methods")
    metrics = _contract_metric_rows(contract)
    artifacts = _contract_artifact_rows(contract)
    parameters = _contract_parameter_rows(contract)
    trends = _contract_trend_rows(contract)

    matrix_rows: list[str] = []
    if named_experiments:
        for experiment in named_experiments[:12]:
            row_parts = [experiment]
            if environments:
                row_parts.append("environments=" + ",".join(environments[:8]))
            if datasets:
                row_parts.append("datasets=" + ",".join(datasets[:8]))
            if methods:
                row_parts.append("methods=" + ",".join(methods[:8]))
            if metrics:
                row_parts.append("metrics=" + ",".join(metrics[:6]))
            if parameters:
                row_parts.append("parameters=" + ",".join(parameters[:6]))
            if trends:
                row_parts.append("trends=" + ",".join(trends[:4]))
            matrix_rows.append(" -> ".join(row_parts))
    else:
        row_parts = ["paper_evidence_matrix"]
        if environments:
            row_parts.append("environments=" + ",".join(environments[:8]))
        if datasets:
            row_parts.append("datasets=" + ",".join(datasets[:8]))
        if methods:
            row_parts.append("methods=" + ",".join(methods[:8]))
        if metrics:
            row_parts.append("metrics=" + ",".join(metrics[:6]))
        if parameters:
            row_parts.append("parameters=" + ",".join(parameters[:6]))
        if trends:
            row_parts.append("trends=" + ",".join(trends[:4]))
        matrix_rows.append(" -> ".join(row_parts))

    chunks = _find_paper_chunks(
        list(state.paper_chunks or []),
        title_terms=("experiment design", "experiment results", "evaluation", "ablation", "sensitivity", "analysis"),
        text_terms=("experiment", "table", "figure", "fig.", "sensitivity", "sweep", "baseline", "environment", "alpha", "lambda"),
        limit=4,
    )
    evidence = [
        "paper evidence contract: " + json.dumps(flat_contract, ensure_ascii=False, sort_keys=True),
        "obligation_matrix: " + "; ".join(matrix_rows[:12]),
    ]
    if chunks:
        evidence.extend(_chunk_preview(chunk, max_chars=1200) for chunk in chunks[:2])
    else:
        preview = " ".join(paperbench_text.split())
        evidence.append(preview[:1600])

    expected_artifacts = [
        "results/evidence_contract_matrix.json",
        "results/experiment_registry.json",
        "results/metrics.json",
    ]
    if environments:
        expected_artifacts.append("results/environment_registry.json")
    if datasets:
        expected_artifacts.append("results/dataset_registry.json")
    if artifacts:
        expected_artifacts.append("results/artifact_manifest.json")
    if parameters or trends:
        expected_artifacts.append("results/sensitivity_report.json")

    inventory_notes = [
        "obligation_matrix: " + "; ".join(matrix_rows[:12]),
    ]
    if named_experiments:
        inventory_notes.append("experiment_inventory: " + "; ".join(named_experiments[:12]))
    if environments:
        inventory_notes.append("environment_inventory: " + "; ".join(environments[:12]))
    if datasets:
        inventory_notes.append("dataset_inventory: " + "; ".join(datasets[:12]))
    if methods:
        inventory_notes.append("method_inventory: " + "; ".join(methods[:12]))
        inventory_notes.append("baseline_inventory: " + "; ".join(methods[:12]))
    if metrics:
        inventory_notes.append("measurement_inventory: " + "; ".join(metrics[:12]))
    if parameters:
        inventory_notes.append("parameter_inventory: " + "; ".join(parameters[:12]))
    if trends:
        inventory_notes.append("result_trend_inventory: " + "; ".join(trends[:12]))
    if artifacts:
        inventory_notes.append("result_artifact_inventory: " + "; ".join(artifacts[:12]))
    inventory_notes.append("artifact_inventory: " + "; ".join(expected_artifacts[:12]))

    obligations = [
        "Implement a code/config-visible paper evidence obligation matrix that binds named experiments, datasets/environments, baselines, parameter sweeps, and result trends to active owners.",
        "Each matrix row must bind the paper/addendum-stated experiment or ablation to its datasets/environments/tasks, methods/baselines, parameter sweep values when stated, expected trend or decision claim, and result artifacts.",
        "Represent required sweeps and endpoint trends through bounded registries, config, and artifact writers with full-mode hooks for researcher runs.",
        "Bind these paper-derived obligations into the primary experiment/config/evaluation code paths and artifact writers.",
        *inventory_notes,
    ]
    return ExtractedUnit(
        unit_id="paper_evidence_matrix",
        type="protocol",
        statement="Implement a paper-derived evidence obligation matrix extracted from the paper/addendum.",
        hypothesis="Preserving the paper's named experiment, dataset, environment, method, parameter, trend, and artifact obligations is necessary for the generated repository to reproduce the decisive contribution claims.",
        decision_value="Lets prepare and plan stages reject missing paper-stated Experiment/ablation/sensitivity coverage before generation, instead of waiting for post-generation semantic review.",
        stop_rule_or_pruning_rationale=(
            "Keep all paper/addendum-stated obligations discoverable, but encode repeated seeds and low-marginal variants as registry/config rows unless they change a stated decision or trend."
        ),
        paper_evidence=evidence[:5],
        source_paragraph_ids=_ordered_unique([chunk.chunk_id for chunk in chunks] + ["paper.md", "addendum.md"]),
        citation_refs=[],
        verification_targets=[
            VerificationTarget(
                kind="artifact",
                description="Generated code exposes an evidence obligation matrix binding experiments, tasks, methods, parameters, trends, and artifact paths.",
            )
        ],
        implementation_surfaces=["evaluation", "baseline_or_ablation", "artifact_writer", "config", "tests"],
        code_obligations=_ordered_unique(obligations),
        runtime_interfaces=["evidence obligation matrix registry", "experiment registry", "parameter sweep config"],
        expected_artifacts=_ordered_unique(expected_artifacts),
        suggested_module_kinds=["evaluation", "configuration", "reporting", "tests"],
        implementation_notes=[
            "Deterministically inferred from paper/addendum text before planning; downstream stages must preserve these inventory lines.",
            *inventory_notes,
        ],
        status="active",
    )


def _paper_derived_unit_candidates(state: PaperBenchReproState) -> list[ExtractedUnit]:
    chunks = _state_paper_chunks(state)
    if chunks and not state.paper_chunks:
        state.paper_chunks = chunks
    specs = [
        (
            "paper_task_environment_setup",
            "task",
            "Implement the task/environment setup described by the paper.",
            ("experiment setup", "environment selection", "environment", "application detail", "details of applications", "extra introduction to applications", "applications", "appendix c.2"),
            ("environment selection", "environment", "environments", "simulator", "application", "benchmark", "task", "we use", "we select"),
            ["config", "data_pipeline", "environment"],
            ["Create configuration/data-loading code for the paper's task environments and inputs.", "Expose explicit environment/task registry entries, initialization metadata, and any normalization/sparse-reward setup stated by the paper."],
            ["environment/config factory", "environment registry"],
            ["configuration artifact"],
            ["environment", "data_loading", "configuration"],
            "Environment setup is required only where it enables the paper's decisive method and metric comparisons.",
            "Determines which task/config surfaces must be implemented before training or evaluation can be meaningful.",
            "Keep all paper-visible environments discoverable through bounded fixtures and full-simulator configuration routes.",
        ),
        (
            "paper_method_core",
            "method",
            "Implement the core method/algorithm proposed in the paper.",
            ("proposed technique", "technical overview", "technique detail", "method"),
            ("algorithm", "method", "objective", "explanation", "policy", "loss"),
            ["model_or_method"],
            ["Create reusable method code for the paper's proposed algorithm and core transformations."],
            ["callable method component"],
            [],
            ["method_core"],
            "The proposed method mechanism is the main causal claim the reproduction must expose in code.",
            "If this unit is weak, later experiment tables cannot rescue the reproduction because the core contribution is absent.",
            "Prioritize the canonical paper method path before peripheral method variants.",
        ),
        (
            "paper_training_or_optimization_loop",
            "method",
            "Implement the training/optimization loop and optimization behavior described by the paper.",
            ("technique detail", "experiment setup", "experiment design"),
            ("train", "training", "optimization", "optimizer", "epoch", "hyperparameter"),
            ["training_loop", "config"],
            ["Create a runnable training or optimization routine with the paper's optimization/configuration controls."],
            ["training command or callable training routine"],
            ["training log artifact"],
            ["training", "configuration"],
            "The paper's claimed improvement depends on the specified optimization sequence plus executable model definitions.",
            "Determines whether the generated repo can run the decisive intervention path under bounded smoke or full modes.",
            "Expose sweepable hyperparameters in config with bounded execution selectors for repeated variants.",
        ),
        (
            "paper_evaluation_protocol",
            "protocol",
            "Implement the evaluation protocol, metrics, and comparison surfaces described by the paper.",
            ("evaluation", "experiment design", "experiment results"),
            ("evaluate", "evaluation", "metric", "score", "compare", "baseline", "ablation"),
            ["evaluation", "baseline_or_ablation", "artifact_writer"],
            ["Create evaluation code that computes paper-relevant metrics and comparison artifacts without fabricating results."],
            ["evaluation command or callable evaluation routine"],
            ["metrics artifact", "result table artifact"],
            ["evaluation", "reporting"],
            "The reproduction must compute the metrics that decide the paper's main comparison and claimed effects.",
            "Determines which outputs can be semantically reviewed before full execution and which comparisons matter for scoring.",
            "Avoid extra derived metrics or exhaustive ablations unless they change the core decision or are stated in the paper/addendum.",
        ),
    ]
    units: list[ExtractedUnit] = []
    if chunks:
        for (
            unit_id,
            unit_type,
            statement,
            title_terms,
            text_terms,
            surfaces,
            obligations,
            interfaces,
            artifacts,
            module_kinds,
            hypothesis,
            decision_value,
            stop_rule_or_pruning_rationale,
        ) in specs:
            matched = _find_paper_chunks(
                chunks,
                title_terms=title_terms,
                text_terms=text_terms,
                limit=3 if unit_id in {"paper_task_environment_setup", "paper_evaluation_protocol"} else 2,
            )
            if not matched:
                continue
            units.append(
                _paper_unit(
                    unit_id=unit_id,
                    unit_type=unit_type,
                    statement=statement,
                    chunks=matched,
                    surfaces=surfaces,
                    obligations=obligations,
                    interfaces=interfaces,
                    artifacts=artifacts,
                    module_kinds=module_kinds,
                    hypothesis=hypothesis,
                    decision_value=decision_value,
                    stop_rule_or_pruning_rationale=stop_rule_or_pruning_rationale,
                )
            )
    evidence_matrix_unit = _paper_evidence_matrix_unit(state)
    if evidence_matrix_unit is not None:
        units.append(evidence_matrix_unit)
    addendum_unit = _addendum_text_unit(state)
    if addendum_unit is not None:
        units.append(addendum_unit)
    environment_inventory_unit = _paper_environment_inventory_unit(state)
    if environment_inventory_unit is not None:
        units.append(environment_inventory_unit)
    dataset_inventory_unit = _paper_dataset_inventory_unit(state)
    if dataset_inventory_unit is not None:
        units.append(dataset_inventory_unit)
    units.extend(_paper_chunk_semantic_units(state))
    units.extend(_paper_contract_specialized_units(state))
    units.extend(_paper_specialized_method_units(state))
    return units


def _ensure_paper_derived_unit_coverage(state: PaperBenchReproState, result: UnitExtractionOutput) -> UnitExtractionOutput:
    named_anchor = _named_experiment_anchor_unit(state)
    paper_candidates = _paper_derived_unit_candidates(state)
    candidate_by_id = {unit.unit_id: unit for unit in paper_candidates}
    evidence_matrix_anchor = candidate_by_id.get("paper_evidence_matrix")
    env_anchor = candidate_by_id.get("paper_task_environment_setup")
    env_inventory_anchor = candidate_by_id.get("paper_environment_inventory")
    evaluation_anchor = candidate_by_id.get("paper_evaluation_protocol")
    existing_ids = {str(item.unit_id or "").strip() for item in result.units}
    existing_text = "\n".join(
        [
            str(unit.statement or "")
            + "\n"
            + "\n".join(str(item) for item in list(unit.paper_evidence or []))
            + "\n"
            + "\n".join(str(item) for item in list(unit.code_obligations or []))
            for unit in result.units
        ]
    )
    paper_text = "\n".join(str(chunk.text or "") for chunk in _state_paper_chunks(state))
    paper_environment_items = _paper_environment_inventory(state)
    paper_evidence_contract = infer_evidence_contract(_paperbench_input_text(state))
    missing_evidence_matrix = bool(
        evidence_matrix_anchor is not None
        and evidence_matrix_anchor.unit_id not in existing_ids
        and paper_evidence_contract.get("requires_evidence_matrix")
        and evidence_contract_gaps(paper_evidence_contract, existing_text)
    )
    paper_mentions_versioned_env = bool(_VERSIONED_ENV_RE.search(paper_text))
    existing_mentions_versioned_env = bool(_VERSIONED_ENV_RE.search(existing_text))
    missing_specific_environment_inventory = bool(
        env_inventory_anchor is not None
        and env_inventory_anchor.unit_id not in existing_ids
        and paper_environment_items
        and not any(item and item in existing_text for item in paper_environment_items[:8])
    )
    missing_named_anchor = bool(
        named_anchor is not None
        and named_anchor.unit_id not in existing_ids
        and (
            not any(marker in existing_text for marker in ("Experiment II", "Experiment III", "Experiment 2", "Experiment 3"))
            or "experiment registry" not in existing_text.lower()
        )
    )
    missing_environment_anchor = bool(
        env_anchor is not None
        and env_anchor.unit_id not in existing_ids
        and paper_mentions_versioned_env
        and (
            not existing_mentions_versioned_env
            or "environment registry" not in existing_text.lower()
        )
    )
    missing_evaluation_anchor = bool(
        evaluation_anchor is not None
        and evaluation_anchor.unit_id not in existing_ids
        and (
            "fidelity score" not in existing_text.lower()
            or "result table" not in existing_text.lower()
        )
    )
    paper_sourced_count = 0
    input_source_ids = {"addendum.md"}
    for unit in result.units:
        source_ids = [str(item or "") for item in list(unit.source_paragraph_ids or [])]
        if any(item.startswith("chunk_") or item in input_source_ids for item in source_ids):
            paper_sourced_count += 1
    candidates = list(paper_candidates)
    priority_candidates: list[ExtractedUnit] = []
    if missing_evidence_matrix and evidence_matrix_anchor is not None:
        priority_candidates.append(evidence_matrix_anchor)
    if missing_named_anchor and named_anchor is not None:
        priority_candidates.append(named_anchor)
    if missing_environment_anchor and env_anchor is not None:
        priority_candidates.append(env_anchor)
    if missing_specific_environment_inventory and env_inventory_anchor is not None:
        priority_candidates.append(env_inventory_anchor)
    if missing_evaluation_anchor and evaluation_anchor is not None:
        priority_candidates.append(evaluation_anchor)
    specialized_missing_candidates = [
        unit
        for unit in candidates
        if unit.unit_id not in existing_ids
        and unit.unit_id.startswith("paper_")
        and unit.unit_id not in _GENERIC_PAPER_DERIVED_UNIT_IDS
    ]
    forced_candidates = priority_candidates + [
        unit for unit in specialized_missing_candidates if unit.unit_id not in {item.unit_id for item in priority_candidates}
    ]
    if forced_candidates:
        forced_ids = {unit.unit_id for unit in forced_candidates}
        candidates = forced_candidates + [unit for unit in candidates if unit.unit_id not in forced_ids]
    if named_anchor is not None and named_anchor.unit_id not in {unit.unit_id for unit in candidates}:
        candidates.append(named_anchor)
    candidate_by_id = {unit.unit_id: unit for unit in candidates}
    backfilled_units: list[ExtractedUnit] = []
    backfilled = 0
    refreshed = 0
    for unit in result.units:
        unit_id = str(unit.unit_id or "").strip()
        candidate = candidate_by_id.get(unit_id)
        if candidate is None:
            backfilled_units.append(unit)
            continue
        if unit_id in _GENERIC_PAPER_DERIVED_UNIT_IDS:
            backfilled_units.append(candidate)
            refreshed += 1
            continue
        updates: dict[str, Any] = {}
        for field_name in ("hypothesis", "decision_value", "stop_rule_or_pruning_rationale"):
            if not str(getattr(unit, field_name, "") or "").strip():
                candidate_value = str(getattr(candidate, field_name, "") or "").strip()
                if candidate_value:
                    updates[field_name] = candidate_value
        if updates:
            backfilled_units.append(unit.model_copy(update=updates))
            backfilled += 1
        else:
            backfilled_units.append(unit)
    if refreshed or backfilled:
        result = result.model_copy(update={"units": backfilled_units})
    if (
        paper_sourced_count >= 3
        and not missing_named_anchor
        and not missing_evidence_matrix
        and not missing_environment_anchor
        and not missing_specific_environment_inventory
        and not missing_evaluation_anchor
        and not specialized_missing_candidates
    ):
        if refreshed or backfilled:
            notes = list(result.extraction_notes or [])
            if refreshed:
                notes.append(f"Refreshed {refreshed} deterministic paper-derived implementation units from the current paper evidence contract.")
            if backfilled:
                notes.append(f"Backfilled decision-value fields for {backfilled} paper-derived implementation units.")
            return result.model_copy(update={"extraction_notes": notes})
        return result
    if not candidates:
        return result
    merged_units = list(result.units)
    added = 0
    for unit in candidates:
        if unit.unit_id in existing_ids:
            continue
        merged_units.append(unit)
        existing_ids.add(unit.unit_id)
        added += 1
    if added <= 0:
        if refreshed or backfilled:
            notes = list(result.extraction_notes or [])
            if refreshed:
                notes.append(f"Refreshed {refreshed} deterministic paper-derived implementation units from the current paper evidence contract.")
            if backfilled:
                notes.append(f"Backfilled decision-value fields for {backfilled} paper-derived implementation units.")
            return result.model_copy(update={"extraction_notes": notes})
        return result
    notes = list(result.extraction_notes or [])
    notes.append(f"Added {added} paper-derived implementation units because model extraction lacked paper-chunk/addendum grounding.")
    if refreshed:
        notes.append(f"Refreshed {refreshed} deterministic paper-derived implementation units from the current paper evidence contract.")
    if backfilled:
        notes.append(f"Backfilled decision-value fields for {backfilled} paper-derived implementation units.")
    coverage = list(dict.fromkeys([*list(result.section_coverage or []), "paper_code_implementation"]))
    return result.model_copy(update={"units": merged_units, "extraction_notes": notes, "section_coverage": coverage})


def _normalize_unit_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize loose model-emitted unit labels into schema-supported types."""
    normalized = dict(payload or {})

    def list_value(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        if value is None:
            return []
        rendered = str(value).strip()
        return [rendered] if rendered else []

    list_fields = (
        "paper_evidence",
        "source_paragraph_ids",
        "citation_refs",
        "implementation_surfaces",
        "code_obligations",
        "runtime_interfaces",
        "expected_artifacts",
        "suggested_module_kinds",
        "implementation_notes",
    )
    units: list[dict[str, Any]] = []
    for item in list(normalized.get("units", []) or []):
        if not isinstance(item, dict):
            continue
        unit_payload = dict(item)
        if not str(unit_payload.get("unit_id", "") or "").strip():
            for key, value in list(unit_payload.items()):
                key_text = str(key or "").strip()
                value_text = str(value or "").strip()
                if re.fullmatch(r"(?:unit|paper)_[A-Za-z0-9_]+", key_text):
                    unit_payload.pop(key, None)
                    if isinstance(value, dict):
                        nested_payload = dict(value)
                        unit_payload = {**nested_payload, **unit_payload}
                        if not str(unit_payload.get("unit_id", "") or "").strip():
                            unit_payload["unit_id"] = key_text
                    else:
                        if re.fullmatch(r"unit_\d+", key_text):
                            unit_payload["unit_id"] = value_text if re.fullmatch(r"unit_\d+", value_text) else key_text
                        elif value_text:
                            unit_payload["unit_id"] = key_text
                            unit_payload.setdefault("statement", value_text)
                    break
        for field_name in list_fields:
            if field_name in unit_payload:
                unit_payload[field_name] = list_value(unit_payload.get(field_name))
        raw_type = str(unit_payload.get("type", "") or "").strip().lower()
        if raw_type in {"evaluation", "eval", "benchmark", "metric"}:
            unit_payload["type"] = "protocol"
        elif raw_type in {"result", "results", "output", "report"}:
            unit_payload["type"] = "artifact"
        elif raw_type not in {"task", "method", "protocol", "claim", "artifact"}:
            unit_payload["type"] = "protocol" if any(
                token in " ".join(
                    [
                        str(unit_payload.get("statement", "") or ""),
                        *[str(value) for value in list(unit_payload.get("implementation_surfaces", []) or [])],
                    ]
                ).lower()
                for token in ("evaluate", "evaluation", "metric", "benchmark", "protocol")
            ) else "method"
        units.append(unit_payload)
    normalized["units"] = units
    for field_name in ("extraction_notes", "section_coverage"):
        if field_name in normalized:
            normalized[field_name] = [str(item) for item in list_value(normalized.get(field_name)) if str(item).strip()]
    return normalized


def _split_section_paragraphs(section_text: str, max_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", section_text) if item.strip()]
    if not paragraphs:
        return [section_text.strip()] if section_text.strip() else []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = projected
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _build_paper_chunks(state: PaperBenchReproState) -> list[PaperChunk]:
    """Build section-first chunks, falling back to paragraph splits for long sections."""
    text, source_path = _load_paper_text(state)
    max_chars = max(1000, int(getattr(state.input, "chunk_max_chars", 6000) or 6000))
    chunks: list[PaperChunk] = []
    ordinal = 1
    cursor = 0
    for section_title, section_start, _section_end, section_text in _paper_sections(text):
        parts = [section_text] if len(section_text) <= max_chars else _split_section_paragraphs(section_text, max_chars)
        split_reason = "section" if len(parts) == 1 else "paragraph_overflow"
        for part_index, part in enumerate(parts, start=1):
            relative_start = text.find(part, cursor)
            if relative_start < 0:
                relative_start = section_start
            relative_end = relative_start + len(part)
            cursor = relative_end
            suffix = f"_{part_index:02d}" if len(parts) > 1 else ""
            chunks.append(
                PaperChunk(
                    chunk_id=f"chunk_{ordinal:03d}{suffix}",
                    section_title=section_title,
                    ordinal=ordinal,
                    source_path=source_path,
                    text=part,
                    char_start=relative_start,
                    char_end=relative_end,
                    token_estimate=max(1, len(part) // 4),
                    split_reason=split_reason,
                )
            )
            ordinal += 1
    return chunks


def _write_prepare_json(
    output_dir: Path,
    filename: str,
    payload: object,
    *,
    json_default: Callable[[Any], Any],
    logical_name: str = "",
    kind: str = "output",
    authority: str = "derived",
) -> Path:
    """Write a prepare artifact under nodes/prepare."""
    prepare_dir = output_dir / "nodes" / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    path = prepare_dir / filename
    json_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    register_existing_file(
        path,
        run_dir=output_dir,
        logical_name=logical_name or filename.rsplit(".", 1)[0],
        kind=kind,
        stage="prepare",
        node="prepare",
        authority=authority,
    )
    return path


def _read_prepare_json(output_dir: Path, filename: str) -> Any:
    """Read a prepare artifact, accepting legacy root-level files for old runs."""
    for path in (output_dir / "nodes" / "prepare" / filename, output_dir / filename):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(filename)


def _load_existing_reference_prepare_artifacts(output_dir: Path) -> tuple[dict[str, Any] | None, list[Any] | None, dict[str, Any] | None]:
    try:
        preparation = _read_prepare_json(output_dir, "reference_repo_preparation.json")
    except Exception:
        preparation = None
    try:
        surveys = _read_prepare_json(output_dir, "reference_repo_surveys.json")
    except Exception:
        surveys = None
    try:
        manifest = _read_prepare_json(output_dir, "resource_manifest.json")
    except Exception:
        manifest = None
    if isinstance(preparation, dict) and list(preparation.get("prepared_repositories", []) or []):
        return preparation, surveys if isinstance(surveys, list) else [], manifest if isinstance(manifest, dict) else None
    return None, None, None


def _build_code_only_resource_manifest(
    *,
    reference_repo_preparation: dict[str, Any],
    reference_repo_surveys: list[Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Prepare-stage manifest for the PaperBench reproduction pipeline."""
    return {
        "schema_version": "paperbench_code_only_v1",
        "prepare_status": "completed",
        "code_only": True,
        "materialization_policy": "paperbench_code_generation_only",
        "reference_repositories": dict(reference_repo_preparation or {}),
        "reference_repo_surveys": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in list(reference_repo_surveys or [])
        ],
        "warnings": list(warnings or []),
    }


def _clear_in_place_resume_runtime_state(state: PaperBenchReproState, requested_resume_stage: str) -> None:
    """Drop downstream runtime state so in-place resume re-enters from a clean stage boundary."""
    if requested_resume_stage != "local_file_generation":
        return

    state.project_root = ""
    state.project_manifest = {}
    state.code = ""
    state.execution_result = None
    state.preflight_result = None
    state.experiment_results = {}
    state.evaluation = None
    state.generate_stage_output = None
    state.runtime_probe = None
    state.validation_report = None
    state.benchmark_report = None
    state.repair_ticket = None
    state.requirement_anchor = None
    state.repair_eval_report = None
    state.repair_plan = None
    state.repair_log = None
    state.current_node = ""
    state.failed_node = ""
    state.error_message = ""
    state.status = "pending"
    state.terminal_outcome = "completed"
    state.terminal_outcome_reason = ""

    temp_data = dict(state.temp_data or {})
    for key in list(temp_data.keys()):
        if key in _IN_PLACE_RESUME_TEMP_DATA_KEYS:
            temp_data.pop(key, None)
            continue
        if any(key.startswith(prefix) for prefix in _IN_PLACE_RESUME_TEMP_DATA_PREFIXES):
            temp_data.pop(key, None)
    state.temp_data = temp_data


def _should_skip_fork_copy(relative_path: Path) -> bool:
    normalized = relative_path.as_posix()
    if not normalized:
        return False
    top_level = normalized.split("/", 1)[0]
    if top_level in _FORK_SKIP_TOPLEVEL:
        return True
    if normalized in _FORK_SKIP_RELATIVE_PATHS:
        return True
    if any(top_level == prefix.rstrip("_") or top_level.startswith(prefix) for prefix in ("score_",)):
        return True
    return any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in _FORK_SKIP_PREFIXES)


def _rewrite_forked_text_paths(output_dir: Path, source_dir: Path) -> None:
    source_text = str(source_dir.resolve())
    target_text = str(output_dir.resolve())
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_path = path.relative_to(output_dir)
        except Exception:
            relative_path = path
        if "ref_repos" in relative_path.parts or "ref_repo" in relative_path.parts:
            continue
        if _should_skip_fork_copy(relative_path):
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if source_text not in original:
            continue
        path.write_text(original.replace(source_text, target_text), encoding="utf-8")


def _fork_previous_run_artifacts(
    state: PaperBenchReproState,
    *,
    output_dir: Path,
) -> None:
    source_run_id = str(state.input.fork_from_run_id or "").strip()
    if not source_run_id or str(state.input.resume_from_run_id or "").strip():
        return

    source_dir = output_dir.parent / source_run_id
    if not source_dir.exists():
        raise FileNotFoundError(f"fork source run not found: {source_run_id}")

    for path in sorted(source_dir.rglob("*")):
        relative_path = path.relative_to(source_dir)
        if _should_skip_fork_copy(relative_path):
            continue
        target_path = output_dir / relative_path
        if path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)

    for shared_dir_name in ("datasets", "benchmarks", "baselines", "ref_repos", "ref_repo"):
        source_shared_dir = source_dir / shared_dir_name
        target_shared_dir = output_dir / shared_dir_name
        if target_shared_dir.exists() or not source_shared_dir.exists():
            continue
        try:
            os.symlink(source_shared_dir, target_shared_dir, target_is_directory=True)
        except OSError:
            if source_shared_dir.is_dir():
                shutil.copytree(source_shared_dir, target_shared_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(source_shared_dir, target_shared_dir)

    _rewrite_forked_text_paths(output_dir, source_dir)

    source_stage_status_path = source_dir / "stage_status.json"
    copied_stage_status: dict[str, Any] = {}

    def _can_reuse_forked_stage(stage_name: str) -> bool:
        if stage_name != "reference_acquisition":
            return True
        try:
            payload = json.loads(
                (source_dir / "nodes" / "prepare" / "reference_repo_preparation.json").read_text(encoding="utf-8")
            )
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        requested = list(payload.get("requested_repositories", []) or [])
        prepared = list(payload.get("prepared_repositories", []) or [])
        failed = list(payload.get("failed_repositories", []) or [])
        current_reference_count = len([item for item in list(state.input.idea_references or []) if isinstance(item, dict)])
        if len(requested) != current_reference_count:
            return False
        if failed:
            return False
        if len(requested) == 0:
            return True
        if len(prepared) != len(requested):
            return False
        try:
            survey_payload = json.loads(
                (source_dir / "nodes" / "prepare" / "reference_repo_surveys.json").read_text(encoding="utf-8")
            )
        except Exception:
            return False
        if not isinstance(survey_payload, list) or len(survey_payload) < len(prepared):
            return False
        survey_ref_ids = {
            str(item.get("ref_id", "") or "").strip()
            for item in survey_payload
            if isinstance(item, dict) and str(item.get("ref_id", "") or "").strip()
        }
        prepared_ref_ids = {
            str(item.get("ref_id", "") or "").strip()
            for item in prepared
            if isinstance(item, dict) and str(item.get("ref_id", "") or "").strip()
        }
        return bool(prepared_ref_ids) and prepared_ref_ids.issubset(survey_ref_ids)

    if source_stage_status_path.exists():
        try:
            payload = json.loads(source_stage_status_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            for stage_name in _FORK_RESUME_STAGE_NAMES:
                entry = dict(payload.get(stage_name, {}) or {})
                if entry.get("status") != "completed":
                    continue
                if not _can_reuse_forked_stage(stage_name):
                    continue
                entry["resume_source"] = "forked_from_run"
                entry["forked_from_run_id"] = source_run_id
                copied_stage_status[stage_name] = entry

    state.temp_data["stage_status"] = copied_stage_status
    state.temp_data["fork_from_run_id"] = source_run_id


def start_impl(
    state: PaperBenchReproState,
    *,
    new_run_id: Callable[[], str],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    json_default: Callable[[Any], Any],
    normalize_dataset_requests: Callable[[dict[str, Any]], list[dict[str, Any]]],
    build_dataset_preparation_payload: Callable[[list[dict[str, Any]], Any, list[dict[str, Any]]], dict[str, Any]],
    update_input_dataset_status: Callable[[dict[str, Any], list[dict[str, Any]]], None],
    update_input_benchmark_status: Callable[[dict[str, Any], list[dict[str, Any]]], None],
    prepare_benchmarks: Callable[[dict[str, Any], Path], tuple[dict[str, Any], list[dict[str, Any]]]],
    prepare_baselines: Callable[..., dict[str, Any]],
    build_resource_manifest: Callable[..., dict[str, Any]],
    build_runtime_probe: Callable[[], Any],
    load_stage_status: Callable[[PaperBenchReproState], dict[str, Any]],
    write_stage_status: Callable[[PaperBenchReproState, dict[str, Any]], None],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("start - Initializing reproagent workflow...")
    state.iteration_count = 0
    resume_run_id = state.input.resume_from_run_id.strip()
    fork_run_id = state.input.fork_from_run_id.strip()
    in_place_resume = bool(getattr(state.input, "resume_in_place", False))
    requested_resume_stage = str(getattr(state.input, "resume_start_stage", "") or "").strip()
    state.run_id = resume_run_id or state.run_id or new_run_id()
    upstream_intent = ensure_upstream_intent_contract(state)
    output_dir = get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    if in_place_resume:
        stage_status = load_stage_status(state)
        if requested_resume_stage:
            _clear_in_place_resume_runtime_state(state, requested_resume_stage)
            reset_started = False
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            for stage_name in _IN_PLACE_STAGE_ORDER:
                if stage_name == requested_resume_stage:
                    reset_started = True
                if not reset_started:
                    continue
                entry = dict(stage_status.get(stage_name, {}) or {})
                output_paths = list(entry.get("output_paths", [])) or []
                stage_status[stage_name] = {
                    "status": "pending",
                    "attempt_id": entry.get("attempt_id", ""),
                    "invalidated_at": timestamp,
                    "reason": f"in_place_resume_from:{requested_resume_stage}",
                    "output_paths": output_paths,
                }
            write_stage_status(state, stage_status)
        save_tracking_artifacts(state)
        return state
    if fork_run_id and not resume_run_id:
        _fork_previous_run_artifacts(state, output_dir=output_dir)
    forked_reference_preparation, forked_reference_surveys, forked_resource_manifest = (
        _load_existing_reference_prepare_artifacts(output_dir) if fork_run_id and not resume_run_id else (None, None, None)
    )
    state.temp_data.setdefault("workflow_started_at", time.perf_counter())
    paper_chunks = _build_paper_chunks(state)
    state.paper_chunks = paper_chunks
    state.temp_data["paper_chunks"] = [item.model_dump(mode="json") for item in paper_chunks]
    state.input.experiment_design.setdefault("paper_chunks_artifact", "nodes/prepare/paper_chunks.json")
    state.input.experiment_design.setdefault("paper_chunk_count", len(paper_chunks))
    state.input.experiment_design.setdefault("paper_chunk_index", _paper_chunk_index(paper_chunks))
    if bool(state.input.experiment_design.get("include_full_paper_chunks_in_design", False)):
        state.input.experiment_design.setdefault("paper_chunks", state.temp_data["paper_chunks"])
    ref_repo_root = output_dir / "nodes" / "prepare" / "ref_repos"
    del normalize_dataset_requests
    del build_dataset_preparation_payload
    del prepare_benchmarks
    del prepare_baselines
    dataset_preparation = {
        "mode": "code_only",
        "download_root": "",
        "requested_datasets": [],
        "downloaded_datasets": [],
        "failed_datasets": [],
        "notes": ["Dataset materialization is intentionally skipped by the code-generation pipeline."],
    }
    benchmark_preparation = {
        "mode": "code_only",
        "prepared_benchmarks": [],
        "unresolved_benchmarks": [],
        "notes": ["Benchmark materialization is intentionally skipped by the code-generation pipeline."],
    }
    baseline_preparation = {
        "mode": "code_only",
        "prepared_baselines": [],
        "failed_baselines": [],
        "notes": ["Baseline materialization is intentionally skipped by the code-generation pipeline."],
    }
    download_results: list[dict[str, Any]] = []
    benchmark_results: list[dict[str, Any]] = []
    github_config_overrides = state.input.github_config if isinstance(state.input.github_config, dict) else {}
    github_config = build_github_repo_config(github_config_overrides)
    update_input_dataset_status(state.input.experiment_design, download_results)
    update_input_benchmark_status(state.input.experiment_design, benchmark_results)
    requested_repositories: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    for item in state.input.idea_references:
        if not isinstance(item, dict):
            continue
        resolved_request = _resolve_reference_repository_request(item, github_config=github_config)
        if not resolved_request:
            continue
        ref_id = str(resolved_request.get("ref_id", "")).strip()
        request_key = ref_id or str(item.get("title", "")).strip().lower()
        if not request_key:
            continue
        if request_key in seen_requests:
            continue
        seen_requests.add(request_key)
        requested_repositories.append(resolved_request)
    reference_repo_preparation = forked_reference_preparation or {
        "clone_root": str(ref_repo_root.resolve()),
        "requested_repositories": requested_repositories,
        "prepared_repositories": [],
        "failed_repositories": [],
        "notes": ["Repository cloning is performed by the reference_acquisition stage."],
    }
    reference_repo_surveys = list(forked_reference_surveys or [])
    del build_resource_manifest
    resource_manifest = forked_resource_manifest or _build_code_only_resource_manifest(
        reference_repo_preparation=reference_repo_preparation,
        reference_repo_surveys=reference_repo_surveys,
        warnings=[],
    )
    _write_prepare_json(output_dir, "input.json", state.input.model_dump(mode="json"), json_default=json_default)
    _write_prepare_json(
        output_dir,
        "upstream_intent.json",
        upstream_intent,
        json_default=json_default,
        kind="contract",
        authority="source_of_truth",
    )
    _write_prepare_json(
        output_dir,
        "paper_chunks.json",
        state.temp_data["paper_chunks"],
        json_default=json_default,
        logical_name="paper_chunks",
        kind="context",
        authority="source_of_truth",
    )
    state.temp_data["dataset_preparation"] = dataset_preparation
    state.temp_data["benchmark_preparation"] = benchmark_preparation
    state.temp_data["baseline_preparation"] = baseline_preparation
    state.temp_data["reference_repo_preparation"] = reference_repo_preparation
    state.temp_data["reference_repo_surveys"] = reference_repo_surveys
    state.temp_data["resource_manifest"] = resource_manifest
    _write_prepare_json(output_dir, "reference_repo_preparation.json", reference_repo_preparation, json_default=json_default)
    if reference_repo_surveys:
        _write_prepare_json(output_dir, "reference_repo_surveys.json", reference_repo_surveys, json_default=json_default)
    _write_prepare_json(output_dir, "resource_manifest.json", resource_manifest, json_default=json_default)
    state.runtime_probe = build_runtime_probe()
    _write_prepare_json(output_dir, "runtime_probe.json", state.runtime_probe, json_default=json_default)
    write_stage_status(state, load_stage_status(state))
    save_tracking_artifacts(state)
    return state


def input_normalization_impl(
    state: PaperBenchReproState,
    *,
    build_input_normalization_context: Callable[[PaperBenchReproState], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("normalize - Normalizing upstream experiment input...")
    input_payload = build_input_normalization_context(state)

    def _compute() -> InputNormalizationOutput:
        system, user = build_input_normalization_prompt(
            limit_json_for_prompt(input_payload),
            language=state.input.language,
        )
        payload = invoke_json_stage("input_normalization", "input_normalization", system, user)
        return InputNormalizationOutput.model_validate(payload)

    def _load() -> InputNormalizationOutput:
        output_dir = get_output_dir(state)
        return InputNormalizationOutput.model_validate(_read_prepare_json(output_dir, "input_normalization.json"))

    def _write(result: InputNormalizationOutput) -> None:
        write_stage_output(state, "input_normalization.json", result)

    state.normalized_input = run_or_resume_stage(
        state,
        "input_normalization",
        input_payload,
        _compute,
        _load,
        _write,
    )
    save_tracking_artifacts(state)
    return state


def unit_extraction_impl(
    state: PaperBenchReproState,
    *,
    build_unit_extraction_context: Callable[[PaperBenchReproState], dict[str, Any]],
    limit_json_for_prompt: Callable[[Any], str],
    invoke_json_stage: Callable[[str, str, str, str], dict[str, Any]],
    get_output_dir: Callable[[PaperBenchReproState], Any],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("units - Extracting implementation units...")
    input_payload = build_unit_extraction_context(state)
    stage_debug: dict[str, Any] = {"quality_gate": {}}

    def _compute() -> UnitExtractionOutput:
        system, user = build_unit_extraction_prompt(
            limit_json_for_prompt(input_payload),
            language=state.input.language,
        )
        payload = invoke_json_stage("unit_extraction", "unit_extraction", system, user)
        return _ensure_paper_derived_unit_coverage(
            state,
            UnitExtractionOutput.model_validate(_normalize_unit_extraction_payload(payload)),
        )

    def _load() -> UnitExtractionOutput:
        output_dir = get_output_dir(state)
        return _ensure_paper_derived_unit_coverage(
            state,
            UnitExtractionOutput.model_validate(
                _normalize_unit_extraction_payload(_read_prepare_json(output_dir, "unit_extraction.json"))
            ),
        )

    def _write(result: UnitExtractionOutput) -> None:
        write_stage_output(state, "units.json", result.units)
        write_stage_output(state, "unit_extraction.json", result)
        quality_gate = unit_extraction_quality_report(
            paper_text=_paperbench_input_text(state),
            units=list(result.units or []),
        )
        stage_debug["quality_gate"] = quality_gate
        workflow_runtime.write_review_artifact(
            state,
            "unit_extraction",
            {
                "stage_name": "unit_extraction",
                "quality_gate": quality_gate,
                "review_status": quality_gate.get("status", "unknown"),
            },
            get_output_dir=get_output_dir,
            json_default=lambda value: value.model_dump(mode="json") if hasattr(value, "model_dump") else str(value),
        )

    state.unit_extraction = run_or_resume_stage(
        state,
        "unit_extraction",
        input_payload,
        _compute,
        _load,
        _write,
    )
    save_tracking_artifacts(state)
    return state


def reference_acquisition_impl(
    state: PaperBenchReproState,
    *,
    get_output_dir: Callable[[PaperBenchReproState], Any],
    get_reference_repo_surveys: Callable[[PaperBenchReproState, Any], list[Any]],
    build_resource_manifest: Callable[..., dict[str, Any]],
    write_stage_output: Callable[[PaperBenchReproState, str, object], None],
    run_or_resume_stage: Callable[..., Any],
    save_tracking_artifacts: Callable[[PaperBenchReproState], None],
) -> PaperBenchReproState:
    logger.info("references - Resolving, cloning, and surveying reference repositories...")
    input_payload = {
        "idea_references": state.input.idea_references,
        "boundary_requirements": state.boundary_requirements.model_dump(mode="json") if state.boundary_requirements else {},
        "github_config": state.input.github_config,
    }

    def _compute() -> tuple[dict[str, Any], list[Any]]:
        output_dir = get_output_dir(state)
        ref_repo_root = output_dir / "nodes" / "prepare" / "ref_repos"
        ref_repo_root.mkdir(parents=True, exist_ok=True)
        github_config_overrides = state.input.github_config if isinstance(state.input.github_config, dict) else {}
        github_config = build_github_repo_config(github_config_overrides)

        resolved_reference_repositories = []
        seen_ref_ids: set[str] = set()
        for item in state.input.idea_references:
            if not isinstance(item, dict):
                continue
            resolved_request = _resolve_reference_repository_request(item, github_config=github_config)
            if not resolved_request:
                continue
            ref_id = str(resolved_request.get("ref_id", "")).strip()
            key = (
                ref_id
                or str(resolved_request.get("paper_path", "")).strip()
                or str(resolved_request.get("repository_url", "")).strip()
                or str(resolved_request.get("title", "")).strip().lower()
            )
            if key in seen_ref_ids:
                continue
            seen_ref_ids.add(key)
            paper_path = str(resolved_request.get("paper_path", "")).strip()
            direct_repository_url = str(resolved_request.get("repository_url", "")).strip()
            resolved_reference_repositories.append({**resolved_request, "source": "idea_gen"})

        clone_requests = [
            item for item in resolved_reference_repositories
            if item["repository_url"] and item["resolve_status"] == "found"
        ]
        clone_results = [
            clone_reference_repository(
                item["repository_url"],
                ref_repo_root,
                github_config=github_config,
            )
            for item in clone_requests
        ]
        result_by_url = {
            str(item.get("repository_url", "")).strip().lower(): item
            for item in clone_results
            if isinstance(item, dict) and item.get("repository_url")
        }
        prepared_repositories = []
        failed_repositories = []
        for request in resolved_reference_repositories:
            result = result_by_url.get(request["repository_url"].lower(), {}) if request["repository_url"] else {}
            payload = {
                "ref_id": request["ref_id"],
                "title": request["title"],
                "paper_path": request["paper_path"],
                "paper_url": request["paper_url"],
                "repository_url": request["repository_url"],
                "repository_origin": request["repository_origin"],
                "repository_type": request["repository_type"],
                "reference_role": str(request.get("reference_role", "") or ""),
                "matched_signals": request["matched_signals"],
                "resolve_reason": request["resolve_reason"],
                "source": request["source"],
                "search_only": bool(request.get("search_only", False)),
                "local_repo_path": str(result.get("local_repo_path", "")),
                "default_branch": str(result.get("default_branch", "")),
                "status": (
                    str(result.get("status", ""))
                    if request["repository_url"] and request["resolve_status"] == "found"
                    else "resolve_failed"
                ) or "failed",
                "success": (
                    request["resolve_status"] == "found"
                    and str(result.get("status", "")) in {"cloned", "reused"}
                ),
                "error_message": (
                    str(result.get("reason", "")).strip()
                    if request["resolve_status"] == "found"
                    else request["resolve_reason"] or "No trusted repository found"
                ),
            }
            if payload["success"]:
                prepared_repositories.append(payload)
            else:
                failed_repositories.append(payload)

        reference_repo_preparation = {
            "clone_root": str(ref_repo_root.resolve()),
            "requested_repositories": resolved_reference_repositories,
            "prepared_repositories": prepared_repositories,
            "failed_repositories": failed_repositories,
        }
        state.temp_data["reference_repo_preparation"] = reference_repo_preparation
        surveys = get_reference_repo_surveys(state, state.boundary_requirements)
        resource_manifest = _build_code_only_resource_manifest(
            reference_repo_preparation=reference_repo_preparation,
            reference_repo_surveys=surveys,
            warnings=[],
        )
        state.temp_data["resource_manifest"] = resource_manifest
        return reference_repo_preparation, surveys

    def _load() -> tuple[dict[str, Any], list[Any]]:
        output_dir = get_output_dir(state)
        reference_repo_preparation = _read_prepare_json(output_dir, "reference_repo_preparation.json")
        surveys = _read_prepare_json(output_dir, "reference_repo_surveys.json")
        try:
            state.temp_data["resource_manifest"] = _read_prepare_json(output_dir, "resource_manifest.json")
        except FileNotFoundError:
            pass
        return reference_repo_preparation, surveys

    def _write(result: tuple[dict[str, Any], list[Any]]) -> None:
        reference_repo_preparation, surveys = result
        write_stage_output(state, "reference_repo_preparation.json", reference_repo_preparation)
        write_stage_output(state, "reference_repos.json", reference_repo_preparation.get("prepared_repositories", []))
        write_stage_output(state, "reference_repo_surveys.json", surveys)
        write_stage_output(state, "resource_manifest.json", state.temp_data.get("resource_manifest", {}))

    reference_repo_preparation, surveys = run_or_resume_stage(
        state,
        "reference_acquisition",
        input_payload,
        _compute,
        _load,
        _write,
    )
    state.temp_data["reference_repo_preparation"] = reference_repo_preparation
    state.reference_repo_surveys = [
        item if hasattr(item, "model_dump") else PreparedReferenceRepositorySurvey.model_validate(item)
        for item in surveys
    ] if surveys else []
    state.temp_data["resource_manifest"] = _build_code_only_resource_manifest(
        reference_repo_preparation=reference_repo_preparation,
        reference_repo_surveys=state.reference_repo_surveys,
        warnings=[],
    )
    write_stage_output(state, "resource_manifest.json", state.temp_data["resource_manifest"])
    gate_report = _write_prepare_quality_gate(
        state,
        write_stage_output=write_stage_output,
        get_output_dir=get_output_dir,
    )
    stage_status = workflow_runtime._load_stage_status(state, get_output_dir=get_output_dir)
    stage_status["prepare_quality_gate"] = {
        "status": "completed",
        "output_paths": [
            "nodes/prepare/prepare_quality_gate.json",
            "nodes/prepare/prepare_quality_gate.review.json",
        ],
        "quality_status": "passed" if gate_report.get("status") == "passed" else "degraded_best_effort",
        "degraded": gate_report.get("status") != "passed",
        "continue_with_best_effort": True,
        "blocking_reason_count": len(list(gate_report.get("blocking_reasons", []) or [])),
    }
    if gate_report.get("status") != "passed":
        stage_status["prepare_quality_gate"]["warning_type"] = "PrepareQualityGateDegraded"
        stage_status["prepare_quality_gate"]["warning_message"] = "; ".join(
            str(item) for item in list(gate_report.get("blocking_reasons", []) or [])[:8]
        )
    workflow_runtime._write_stage_status(
        state,
        stage_status,
        get_output_dir=get_output_dir,
        json_default=_json_default_for_prepare_quality,
    )
    save_tracking_artifacts(state)
    return state
