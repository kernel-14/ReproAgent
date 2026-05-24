"""Upstream intent lock helpers for reproagent."""

from __future__ import annotations

from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproInput, PaperBenchReproState, UpstreamIntentContract


_PAPERBENCH_PROMPT_EXCLUDED_TOP_LEVEL_KEYS = {
    "case_dir",
    "case_id",
    "forbidden_shortcuts",
    "judge_addendum_path",
    "judge_addendum_text",
    "paperbench_blacklist",
    "paperbench_case",
    "paperbench_case_dir",
    "paperbench_case_id",
    "paperbench_constraints",
    "paperbench_id",
    "paperbench_has_judge_addendum",
    "paperbench_rubric_items",
    "paperbench_rubric_item_count",
    "rubric",
    "rubric_items",
    "rubric_summary",
    "rubric_path",
}

_PAPERBENCH_PROMPT_EXCLUDED_NESTED_KEYS = {
    "blacklist",
    "blacklist_path",
    "case_id",
    "case_dir",
    "id",
    "judge_addendum_path",
    "judge_addendum_text",
    "rubric",
    "rubric_items",
    "rubric_summary",
    "rubric_path",
}

_PAPERBENCH_PROMPT_FILTERED_LIST_KEYS = {
    "expected_artifacts",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        rendered = str(item or "").strip()
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        result.append(rendered)
    return result


def _dict_list(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            result.append(dict(item))
        elif item:
            result.append({"name": str(item)})
    return result


def _collect_experiment_values(experiment_design: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_string_list(experiment_design.get(key)))
    return _string_list(values)


def _filter_rubric_list_entries(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [item for item in value if "rubric" not in str(item or "").lower()]


def paperbench_prompt_safe_experiment_design(experiment_design: dict[str, Any] | None) -> dict[str, Any]:
    """Return prompt-facing PaperBench metadata without evaluator rubric content."""
    if not isinstance(experiment_design, dict):
        return {}
    safe = dict(experiment_design)
    for key in _PAPERBENCH_PROMPT_EXCLUDED_TOP_LEVEL_KEYS:
        safe.pop(key, None)
    for key in _PAPERBENCH_PROMPT_FILTERED_LIST_KEYS:
        if key in safe:
            safe[key] = _filter_rubric_list_entries(safe.get(key))
    paperbench = safe.get("paperbench")
    if isinstance(paperbench, dict):
        nested = dict(paperbench)
        for key in _PAPERBENCH_PROMPT_EXCLUDED_NESTED_KEYS:
            nested.pop(key, None)
        safe["paperbench"] = nested
    return safe


def build_upstream_intent_contract(input_payload: PaperBenchReproInput) -> UpstreamIntentContract:
    """Freeze upstream-provided experiment intent before downstream planning mutates state."""
    experiment_design = paperbench_prompt_safe_experiment_design(input_payload.experiment_design)
    paperbench = experiment_design.get("paperbench") if isinstance(experiment_design.get("paperbench"), dict) else {}
    paper_context = {
        "paper_path": str(getattr(input_payload, "paper_path", "") or ""),
        "paper_text_present": bool(str(getattr(input_payload, "paper_text", "") or "").strip()),
        "paperbench_title": str(paperbench.get("title", "") or experiment_design.get("paperbench_title", "") or ""),
        "paperbench_assets": list(paperbench.get("assets", []) or []),
        "paperbench_has_addendum": bool(str(paperbench.get("addendum_text", "") or "").strip()),
        "paperbench_input_sources": ["paper.md", "addendum.md", "assets"],
        "reference_source_policy": "Use prepared local references with provenance markers when they support an implementation route.",
    }
    dataset_items = _dict_list(experiment_design.get("datasets"))
    benchmark_items = _dict_list(experiment_design.get("benchmarks"))
    metric_names = _collect_experiment_values(
        experiment_design,
        "metrics",
        "metric",
        "primary_metric",
        "evaluation_metrics",
    )
    benchmark_names = [
        str(item.get("name") or item.get("benchmark") or item.get("id") or "").strip()
        for item in benchmark_items
        if str(item.get("name") or item.get("benchmark") or item.get("id") or "").strip()
    ]
    explicit_constraints = _collect_experiment_values(
        experiment_design,
        "constraints",
        "explicit_constraints",
        "requirements",
        "notes",
        "paperbench_constraints",
    ) + [
        "Input source is the paper, not the proposal.",
        "Generate faithful repository code with bounded default execution and full runnable research routes.",
        "Use addendum.md as binding implementation clarification when present.",
        "Use prepared, provenance-tracked local references when constructing reference-backed implementation patterns.",
    ]
    contract = UpstreamIntentContract(
        target=str(input_payload.target or "").strip(),
        language=str(input_payload.language or "zh").strip() or "zh",
        experiment_design=experiment_design,
        idea_references=[dict(item) for item in list(input_payload.idea_references or []) if isinstance(item, dict)],
        idea_reference_summaries=[
            dict(item) for item in list(input_payload.idea_reference_summaries or []) if isinstance(item, dict)
        ],
        dataset_contract={
            "datasets": dataset_items,
            "download_policy": experiment_design.get("download_policy", ""),
        },
        benchmark_contract={
            "benchmarks": benchmark_items,
            "metrics": metric_names,
            "paper_context": paper_context,
        },
        required_datasets=dataset_items,
        required_metrics=metric_names,
        required_baselines=_collect_experiment_values(experiment_design, "baselines", "baseline", "required_baselines"),
        required_ablations=_collect_experiment_values(experiment_design, "ablations", "ablation", "required_ablations"),
        expected_artifacts=_collect_experiment_values(
            experiment_design,
            "expected_artifacts",
            "artifacts",
            "outputs",
        ),
        explicit_constraints=explicit_constraints,
        forbidden_shortcuts=_collect_experiment_values(
            experiment_design,
            "forbidden_shortcuts",
            "disallowed_shortcuts",
            "must_not",
        ),
        allowed_approximations=_collect_experiment_values(
            experiment_design,
            "allowed_approximations",
            "engineering_approximations",
            "approximations",
        ),
        source_fields=[
            "target",
            "language",
            "experiment_design",
            "idea_references",
            "idea_reference_summaries",
            "paper_path",
            "paper_text",
        ],
    )
    if not contract.forbidden_shortcuts:
        contract.forbidden_shortcuts = [
            "Metrics and benchmark reports must be produced by repository code paths.",
            "The upstream task definition remains the semantic baseline for planning.",
            "Required datasets, baselines, ablations, and metric obligations remain visible in implementation contracts.",
        ]
    return contract


def ensure_upstream_intent_contract(state: PaperBenchReproState) -> UpstreamIntentContract:
    """Return the locked upstream intent, creating it once if missing."""
    if state.upstream_intent is None:
        state.upstream_intent = build_upstream_intent_contract(state.input)
    state.upstream_intent.experiment_design = paperbench_prompt_safe_experiment_design(
        state.upstream_intent.experiment_design
    )
    if isinstance(state.upstream_intent.benchmark_contract, dict):
        paper_context = state.upstream_intent.benchmark_contract.get("paper_context")
        if isinstance(paper_context, dict):
            paper_context.pop("paperbench_rubric_item_count", None)
            paper_context.pop("paperbench_case_id", None)
            paper_context.pop("paperbench_case_dir", None)
            paper_context.pop("paperbench_has_judge_addendum", None)
    state.temp_data["upstream_intent"] = state.upstream_intent.model_dump(mode="json")
    return state.upstream_intent


def upstream_intent_payload(state: PaperBenchReproState) -> dict[str, Any]:
    """JSON payload used by prompts, handoff, and reports."""
    payload = ensure_upstream_intent_contract(state).model_dump(mode="json")
    payload["experiment_design"] = paperbench_prompt_safe_experiment_design(payload.get("experiment_design"))
    payload.pop("forbidden_shortcuts", None)
    payload.pop("explicit_constraints", None)
    payload.pop("idea_references", None)
    payload.pop("idea_reference_summaries", None)
    payload.pop("target", None)
    experiment_design = payload.get("experiment_design")
    if isinstance(experiment_design, dict):
        for key in ("paperbench_blacklist", "paperbench_constraints", "forbidden_shortcuts"):
            experiment_design.pop(key, None)
        paperbench = experiment_design.get("paperbench")
        if isinstance(paperbench, dict):
            paperbench.pop("github_references", None)
            paperbench.pop("reference_repo_candidates", None)
            paperbench.pop("dataset_hints", None)
            paperbench.pop("blacklist", None)
            paperbench.pop("blacklist_path", None)
    payload["implementation_source_policy"] = {
        "input_documents": ["paper.md", "addendum.md when present", "assets when present"],
        "reference_usage": "Prepared, provenance-tracked local references can supply implementation-route patterns.",
        "implementation_scope": "Plan paper-derived methods, datasets, metrics, protocols, artifacts, and runnable routes.",
    }
    benchmark_contract = payload.get("benchmark_contract")
    if isinstance(benchmark_contract, dict):
        paper_context = benchmark_contract.get("paper_context")
        if isinstance(paper_context, dict):
            paper_context.pop("paperbench_rubric_item_count", None)
            paper_context.pop("paperbench_case_id", None)
            paper_context.pop("paperbench_case_dir", None)
            paper_context.pop("paperbench_has_judge_addendum", None)
            paper_context.pop("paperbench_blacklist", None)
            paper_context["paperbench_input_sources"] = ["paper.md", "addendum.md", "assets"]
    return payload
