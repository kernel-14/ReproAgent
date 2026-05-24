"""Prompt templates for reproagent workflow."""

import json


def _build_stage_system_prompt(stage_label: str, responsibility: str, language: str = "en") -> str:
    """Build a short stage-specific system prompt."""
    return f"""You are the `reproagent` planner for stage `{stage_label}`.

This stage is responsible for: {responsibility}

Stay within the current stage boundary, use only the provided context, and return strict JSON.
Output language: {language}
"""


def _build_structured_user_prompt(
    instruction: str,
    schema_block: str,
    rules: list[str],
    context_json: str,
) -> str:
    """Build a user prompt carrying schema, rules, and serialized context."""
    rule_lines = "\n".join(f"- {item}" for item in rules)
    return (
        f"{instruction}\n\n"
        "Return ONLY valid JSON with exactly this shape:\n"
        f"{schema_block}\n\n"
        "Rules:\n"
        f"{rule_lines}\n\n"
        "Context:\n\n"
        f"{context_json}"
    )


# Prompt design note:
# - system prompt: short stage identity and boundary only
# - user prompt: task instruction + compact output schema + rules + serialized context
# - runtime validation: enforced by downstream schema parsing and model validation
def _build_stage_fix_prompt(stage_label: str, focus: str, context_json: str, language: str = "zh") -> tuple[str, str]:
    """Shared repair prompt for Ralph-loop execution stages."""
    system = _build_stage_system_prompt(
        stage_label=stage_label,
        responsibility=f"repair the repository to pass the `{stage_label}` execution gate",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=f"Repair the project based on this `{stage_label}` stage context.",
        schema_block="""{
  "summary": "brief diagnosis",
  "suggestions": ["short actionable suggestion"],
  "install_commands": ["python -m pip install ..."],
  "updated_files": {
    "relative/path.py": "full file content"
  }
}""",
        rules=[
            f"Focus on `{stage_label}` only: {focus}",
            "`updated_files` must be a JSON object keyed by relative project path.",
            "Return changed files as full file contents.",
            "Use JSON string values for file contents.",
            "Include install commands that are necessary for this stage.",
            "Use an empty `install_commands` array when the stage uses existing dependencies.",
            "Return strict JSON with no surrounding prose or markdown fences.",
        ],
        context_json=context_json,
    )
    return system, user


def build_input_normalization_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-0 prompt: normalize the upstream request into a stable planning input."""
    system = _build_stage_system_prompt(
        stage_label="input_normalization",
        responsibility="normalize the upstream experiment request into a stable, implementation-oriented planning input",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Normalize this upstream experiment request.",
        schema_block="""{
  "normalized_target": "normalized task statement",
  "target_summary": "short summary",
  "task_type": "generic_experiment",
  "key_entities": ["entity"],
  "explicit_constraints": ["constraint"],
  "expected_outputs": ["output"],
  "assumption_notes": ["assumption"]
}""",
        rules=[
            "Preserve the user's actual intent and constraints as the reproduction target.",
            "For PaperBench inputs, treat `paper.md`, `addendum.md`, assets, and prepared provenance metadata as the source-of-truth implementation inputs.",
            "For PaperBench inputs, normalize the request as a code reproduction task from the paper.",
            "Summarize the request into a cleaner implementation brief for downstream stages.",
            "Extract concrete entities such as datasets, methods, models, metrics, artifacts, environments, and baselines when they are explicitly present.",
            "Include assumptions that help planning while staying conservative.",
            "Leave repository structure and file layout for later planning stages.",
        ],
        context_json=context_json,
    )
    return system, user


def build_unit_extraction_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-1 prompt: extract implementation units from the normalized request."""
    system = _build_stage_system_prompt(
        stage_label="unit_extraction",
        responsibility="extract implementation units covering task, method, protocol, claim, and artifact requirements from the normalized request",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Extract implementation units from this normalized experiment request.",
        schema_block="""{
  "units": [
    {
      "unit_id": "unit_001",
      "type": "task",
      "statement": "implement a runnable experiment entrypoint",
      "hypothesis": "The minimal runnable experiment path is sufficient to exercise the paper's core contribution claim.",
      "decision_value": "Confirms which method/evaluation implementation surfaces downstream planning should prioritize.",
      "paper_evidence": ["normalized target excerpt"],
      "source_paragraph_ids": ["target:1"],
      "citation_refs": [],
      "verification_targets": [{"kind": "artifact", "description": "main entrypoint exists"}],
      "implementation_surfaces": ["entrypoint", "artifact_writer"],
      "code_obligations": ["Create a runnable entrypoint that produces the declared artifacts."],
      "runtime_interfaces": ["CLI command or callable main()"],
      "expected_artifacts": ["results/metrics.json"],
      "suggested_module_kinds": ["entrypoint", "reporting"],
      "implementation_notes": ["Keep this unit code-generation focused; implement full training/evaluation code paths and use bounded default execution during generation."],
      "status": "active"
    }
  ],
  "extraction_notes": ["short note"],
  "section_coverage": ["task", "artifact"]
}""",
        rules=[
            "Extract a compact but sufficient set of units that can drive downstream planning.",
            "For PaperBench inputs, extract implementation units only from selected paper chunks and addendum clarifications.",
            "Use selected paper chunks, addendum clarifications, assets, and prepared provenance policy as the source of unit content.",
            "Use `paper_evidence_contract.required_claim_inventory` as the minimum paper-derived inventory to preserve.",
            "Every item in `paper_evidence_contract.closure_items` should appear in a unit statement, code obligation, runtime interface, expected artifact, or inventory note.",
            "If the paper/addendum names many decisive tables, figures, baselines, datasets, metrics, or hyperparameter protocols, create enough atomic units to preserve those obligations separately with explicit names.",
            "Every named table or figure that reports an implementation-relevant experiment must be represented in either a dedicated unit or an explicit artifact/protocol unit that names the exact table/figure and its required methods, datasets, metrics, and artifact writer.",
            "Every named baseline/model/method family that participates in a reported comparison must appear by exact name in a method, baseline, training, or evaluation unit, especially when the paper names Azure-SFT, LoRA, DDPM-PA, TGAN, EWC, or similar concrete methods.",
            "When one paper table/figure decomposes into several implementation actions, split it into atomic units for the active code paths: data/model setup, training or adaptation loop, metric computation, and artifact writer.",
            "Prefer units for task, method, protocol, claim, and artifact coverage when they are present.",
            "When the paper mentions environments, datasets, policies/models, training/pretraining, metrics, baselines, ablations, refinement, or named experiments, extract units for those implementation code paths instead of collapsing them into artifact or protocol units.",
            "For PaperBench papers, make unit obligations inventory-oriented: explicitly name datasets/environments, models/methods/baselines, metrics, named experiments, tables/figures, parameter values, trends, and active implementation scope when the paper/addendum states them.",
            "When selected paper chunks or `paper_evidence_contract.formula_algorithm_contract` contain equations, mathematical symbols, named masks, adapter matrices, salience/importance scores, search procedures, schedules, losses, or metric formulas, create dedicated method/training/evaluation units whose `code_obligations` quote the required symbols, numeric constants, and algorithm steps as executable implementation requirements.",
            "Do not collapse formula-heavy method sections into broad phrases such as `implement the method`; preserve the formula variables, update equations, search procedure, mask/rank semantics, and metric definitions as concrete obligations.",
            "When a paper/addendum states a parameter sweep or sensitivity study, preserve exact values such as `{0, 0.25, 0.5, 0.75, 1}` in `code_obligations` and `implementation_notes` as executable parameter coverage.",
            "When a paper/addendum states a result trend such as an endpoint being lowest, a parameter being insensitive, or a positive value improving results, encode that trend in `code_obligations` as bounded semantic-review metadata.",
            "When addendum text narrows the requested reproduction target, express only the resulting active implementation coverage in `code_obligations`, `runtime_interfaces`, `expected_artifacts`, and inventory notes.",
            "Prefer explicit inventory lines in `implementation_notes` when useful: `environment_inventory: ...`, `dataset_inventory: ...`, `method_inventory: ...`, `baseline_inventory: ...`, `measurement_inventory: ...`, `parameter_inventory: ...`, `result_trend_inventory: ...`, `result_artifact_inventory: ...`, `implementation_surface_inventory: ...`.",
            "Phrase addendum/provenance policy as active coverage, source provenance, executable selectors, artifact ownership, and bounded full-run routes. Output fields should describe what to implement.",
            "For experiment, ablation, baseline, metric, or result-claim units, explicitly fill `hypothesis` and `decision_value`; put positive implementation scope in `code_obligations`, `runtime_interfaces`, `expected_artifacts`, and `implementation_notes`.",
            "`hypothesis` should state the paper contribution or mechanism the unit tests, not just restate the table row.",
            "`decision_value` should explain what implementation, reproduction, or comparison decision would change if this unit succeeds or fails.",
            "Implementation scope should state the required route, artifact coverage, and bounded execution focus for deciding the paper claim or satisfying benchmark-visible coverage.",
            "Use smoke, dry-run, readiness, and artifact-manifest behavior as support units alongside method, training, refinement, environment, policy/model, metric, baseline, and evaluation implementation units.",
            "When the paper requires large external models, datasets, or per-example training, extract obligations for the real loader/factory, optimizer/refinement loop, checkpoint/adapter output, pairwise evaluation, and metric computation. Default execution may be bounded, but the generated repo must contain the full implementation route.",
            "Every unit should be concrete, implementation-oriented, and non-duplicative.",
            "`implementation_surfaces` should name code surfaces such as entrypoint, data_pipeline, model_or_method, environment_adapter, environment_factory, policy_adapter, policy_factory, training_loop, pretraining, evaluation, metric_formula, baseline_or_ablation, refinement_algorithm, artifact_writer, config, environment, or tests.",
            "`code_obligations` should state what generated code must implement, not what experiment results should be achieved.",
            "`runtime_interfaces` should capture expected CLI/function/config surfaces when the paper or addenda imply them.",
            "`expected_artifacts` should list concrete output artifacts only when they are required by the paper/addenda; otherwise leave it empty.",
            "`suggested_module_kinds` should be module categories rather than final file paths; keep repository file-tree decisions for later stages.",
            "Use `paper_evidence` as direct excerpts or compact paraphrases from the normalized request context.",
            "Use exact paragraph boundaries when available; otherwise use stable synthetic ids such as `target:1` in `source_paragraph_ids`.",
            "Include at least one verification target per unit when possible.",
        ],
        context_json=context_json,
    )
    return system, user


def build_boundary_requirements_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-1 prompt: extract lightweight experiment boundaries."""
    system = _build_stage_system_prompt(
        stage_label="boundary_requirements",
        responsibility="extract lightweight experiment boundaries and requirement scope from the upstream request",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build boundary requirements from this context.",
        schema_block="""{
  "boundary_requirements": [
    {"requirement_id": "req_001", "title": "short title", "category": "experiment", "scope": "dataset_support", "description": "brief boundary description", "source_unit_ids": ["unit_001"], "acceptance_criteria": ["criterion 1"]}
  ],
  "requirement_scope_items": ["scope item"]
}""",
        rules=[
            "The target is to implement the upstream experiment request as code.",
            "When PaperBench context is present, the paper, addenda, assets, and prepared provenance policy define the active reproduction boundary.",
            "Keep requirements lightweight and boundary-oriented.",
            "Focus on experiment intent and methodological boundaries only.",
            "Extract requirement types such as task, method, key output expectations, dataset, benchmark/protocol, baseline/model, metric, ablation, and hyperparameter when they are present in the upstream context.",
            "Every boundary requirement should cite one or more `source_unit_ids` from the provided units whenever possible.",
            "Describe the experiment behavior and active implementation scope.",
            "Leave file tree, module split, and code structure for later planning stages.",
        ],
        context_json=context_json,
    )
    return system, user


def build_topic_profile_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-1.5 prompt: derive topic guidance without forcing a template."""
    system = _build_stage_system_prompt(
        stage_label="topic_profile",
        responsibility="derive topic guidance, coverage policy, and prompt hints for downstream planning without overfitting the repository structure",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build a compact topic profile from this context.",
        schema_block="""{
  "primary_topic": "generic_experiment",
  "active_topics": ["classification"],
  "reference_topics": ["tabular_ml"],
  "experiment_traits": ["cpu_only", "baseline_comparison"],
  "asset_types": ["dataset", "metrics", "report"],
  "coverage_policy": "balanced",
  "coverage_hints": {"baseline_inventory": ["LogisticRegression", "RandomForestClassifier"], "measurement_inventory": ["accuracy", "macro_f1"]},
  "prompt_guidance": ["Prefer lightweight sklearn-style experiment structure and CPU-friendly training routes."]
}""",
        rules=[
            "This stage provides conditional guidance for downstream planning.",
            "Use the upstream target and boundary requirements as the main signal; local references are only weak evidence.",
            "`coverage_policy` should be `force` only when the task clearly requires broad explicit coverage; otherwise prefer `balanced` or `advisory`.",
            "`coverage_hints` should emphasize baseline inventory, benchmark/measurement inventory, prerequisite assets, and obvious experiment traits when present.",
            "Keep `prompt_guidance` short, concrete, and useful for planning/generation stages.",
        ],
        context_json=context_json,
    )
    return system, user


def build_reference_selection_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-2 prompt: select actionable local reference repositories."""
    system = _build_stage_system_prompt(
        stage_label="reference_selection",
        responsibility="Extract actionable insights from the current list of locally cloned reference repositories and establish their alignment with the experiment requirements.",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Extract key information from the provided local repository context and filter for actionable reference items.",
        schema_block="""{
  "actionable_references": [
    {"ref_id": "paperbench_ref_001", "title": "reference title", "paper_url": "https://...", "repository_url": "https://...", "repository_origin": "official", "local_repo_path": "/abs/path/to/local/repo", "default_branch": "main", "supported_requirement_ids": ["req_001"], "reusable_modules": ["train_mask_net"], "insights": ["how this repo can be reused"], "file_tree": "optional tree summary", "readme_summary": "optional README summary", "top_python_files": ["main.py"], "likely_reusable_files": ["trainer.py"], "protocol_clues": ["train.py: parser.add_argument('--seed', ...)"], "requirement_coverage": [{"requirement_id": "req_001", "title": "method core", "scope": "core_method", "source_unit_ids": ["unit_001"], "keyword_hits": 2, "matched_keywords": ["policy"], "matched_files": ["trainer.py"], "match_locations": ["trainer.py:18: class Trainer"], "code_snippets": ["trainer.py:18: class Trainer ..."]}]}
  ],
  "reference_relations": [
    {"ref_id": "paperbench_ref_001", "supported_scope_items": ["core_method"], "reference_role": "core_method"}
  ]
}""",
        rules=[
            "The repositories have already been cloned locally by the start node.",
            "Use the upstream prepared-reference gate as the repository availability signal.",
            "Use the provided local repository paths, README summaries, likely reusable files, protocol clues, requirement coverage, matched files, and candidate snippets to decide how each prepared repo supports the current requirements.",
            "Use repositories and `ref_id` values from the provided `prepared_reference_repositories`.",
            "Preserve every prepared local repo as an actionable reference unless the repo survey is clearly empty or invalid.",
            "`supported_requirement_ids` should reflect requirement coverage evidenced by the local repo survey.",
            "`reusable_modules` should name concrete reusable implementation areas inferred from the local repo layout or matched files.",
            "Add concise reuse insights.",
            "When the survey already includes `top_python_files`, `likely_reusable_files`, `protocol_clues`, `requirement_coverage`, or `symbol_evidence`, carry useful items into the reference summary.",
            "Treat `reference_relations` as a compact summary of scope coverage.",
            "Return reference support metadata only.",
        ],
        context_json=context_json,
    )
    return system, user


def build_pipeline_plan_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-3 prompt: build flat, traceable implementation plan nodes."""
    system = _build_stage_system_prompt(
        stage_label="pipeline_plan",
        responsibility="decompose the experiment into flat, traceable implementation plan nodes",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=(
            "Build a faithful, complete, executable reproduction contract from this context. "
            "The plan must preserve the paper's required claims as active code-route obligations, "
            "as concrete implementation ownership, routes, artifacts, and validation hooks."
        ),
        schema_block="""{
  "plan_nodes": [
    {"node_id": "exp_001", "parent_node_id": "", "name": "node name", "level": "experiment", "description": "what this node owns and implements", "hypothesis": "contribution hypothesis this node tests", "decision_value": "decision enabled by this node", "requirement_ids": ["req_001"], "ref_id": "paperbench_ref_001", "reusable_module": "train_mask_net", "depends_on": ["mod_001"], "traceable": true, "code_snippet": "short reusable source snippet", "insight": "how to adapt or reuse it"}
  ],
  "coverage_summary": {"total_requirements": 1, "covered_requirements": 1, "uncovered_requirement_ids": []}
}""",
        rules=[
            "Decompose top-down with levels experiment -> module -> function.",
            "Treat this plan as the binding contract for a judgeable reproduction repository.",
            "Use `unit_route_matrix` as the prepare-unit routing checklist. Every implementation-bearing row needs an owner package, active implementation path, route path, and validation hook by the time architecture/file planning closes.",
            "Final output must be a flat node list, not nested children.",
            "Use `paper_evidence_contract.required_claim_inventory` and work-package `obligation_matrix` as mandatory anchors. Plan nodes must keep exact named claims visible and bound to requirement_ids/work packages.",
            "Name the datasets, methods/baselines, metrics, parameters, and artifacts that each node owns for claim inventory coverage.",
            "Keep sibling nodes semantically distinct.",
            "Bind nodes to stable ref_id + reusable_module whenever possible.",
            "Use the provided local cloned repo survey to ground `ref_id`, `reusable_module`, and any traceable `code_snippet`.",
            "Use only `ref_id`, `reusable_module`, and snippets present in the provided surveys.",
            "Prefer `symbol_evidence` first, then `requirement_coverage.code_snippets`, when selecting grounded reference snippets.",
            "Focus on experiment implementation logic only: method flow, data flow, model flow, evaluation flow, and experiment-specific helper logic.",
            "Experiment-level nodes must be hypothesis- and decision-value-driven, with each node tied to a distinct implementation path or benchmark-visible requirement.",
            "For each experiment-level node, include `hypothesis` and `decision_value`; put implementation scope in `description`, required routes, artifacts, and `scope_boundary`/inventories.",
            "Scope language should describe what this node implements and which bounded default route exercises it, especially for paper-named mechanisms, solvers, datasets, baselines, metrics, and train/eval routes.",
            "Close method, dataset, model, baseline, metric, training, evaluation, and artifact-producing units through active Python routes that load or call the relevant README/config/registry/support surfaces.",
            "When the context names paired mechanisms or alternatives such as ODE/SDE, train/eval, online/offline, actor/critic, exact/approximate solvers, or sampler families, preserve each named side as a first-class node or interface obligation using positive implementation wording.",
            "Preserve benchmark-visible required coverage, but prefer the smallest experiment set that can distinguish the paper's core method claim, main baseline comparison, key metric behavior, and any required failure/robustness condition.",
            "For traceable nodes, include a concrete `code_snippet` and an `insight` explaining how the snippet should be reused or adapted.",
            "`code_snippet` should be concise and correspond to the declared `ref_id` + `reusable_module`.",
            "For ungrounded nodes, keep `code_snippet` and `insight` empty.",
        ],
        context_json=context_json,
    )
    return system, user


def build_work_package_planning_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Middle-stage prompt: synthesize work packages before contract freezing."""
    system = _build_stage_system_prompt(
        stage_label="work_package_planning",
        responsibility="group extracted units into implementation-oriented work packages with explicit coverage and ownership",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=(
            "Build faithful, complete work packages from this context. Each package must own executable "
            "reproduction obligations for the paper through code routes, artifacts, interfaces, and validation hooks."
        ),
        schema_block="""{
  "work_packages": [
    {
      "work_package_id": "<paper_specific_work_package_id>",
      "goal": "build runnable repository surfaces",
      "hypothesis": "the repository can expose the paper's core experiment path without claiming full training results",
      "decision_value": "unblocks semantic review of method/evaluation code before expensive execution",
      "owned_unit_ids": ["unit_001"],
      "tags": ["entrypoint", "artifact"],
      "reference_ids": ["ref_001"],
      "depends_on": [],
      "produces": ["main.py", "results/metrics.json"],
      "interface_contract": ["main entrypoint exists"],
      "evidence_needs": ["runnable repository surface"],
      "inventories": {"obligation_matrix": ["Experiment I: main comparison -> results/metrics.json"], "environment_inventory": ["task family"], "method_inventory": ["baseline", "proposed"], "parameter_inventory": ["alpha values if required"], "result_trend_inventory": ["expected trend if required"], "artifact_inventory": ["results/metrics.json"]},
      "scope_boundary": {"preserve": ["unit_001: implement the exact paper/addendum method, protocol, metric, and artifact obligation"], "implementation_focus": ["active code routes and artifacts this package must own"]},
      "method_obligations": ["preserve stable artifact contract"]
    }
  ],
  "coverage_summary": {"total_units": 1, "covered_units": 1, "uncovered_unit_ids": []},
  "planning_notes": ["group units by implementation ownership rather than by document section"]
}""",
        rules=[
            "Group units into a small number of implementation-oriented work packages.",
            "Plan consumes Prepare-stage implementation units and maps those units into executable work-package ownership.",
            "Treat work packages as mandatory owners for judgeable reproduction code paths.",
            "Every active unit should be owned by at least one work package with concrete implementation surfaces.",
            "For every package, fill `scope_boundary.preserve` with concrete owned-unit obligations to faithfully and completely implement; this is the positive implementation scope.",
            "Use `scope_boundary.implementation_focus` to state the active code surfaces, artifact writers, and routes that should implement the preserved units.",
            "In `goal`, `evidence_needs`, `method_obligations`, `scope_boundary`, and inventories, write active implementation coverage and provenance-backed source usage.",
            "Use `paper_evidence_contract.required_claim_inventory` as a checklist. Every named experiment, dataset/environment, method/baseline, metric, table/figure, parameter value, trend, protocol, and implementation obligation must be owned by an implementation package.",
            "`obligation_matrix` must include exact names from `paper_evidence_contract.closure_items` so claim items stay visible beyond generic `baseline`, `metric`, `table`, or `artifact` wording.",
            "When units name concrete tables, figures, baselines, datasets, metrics, costs, prompts, or hyperparameters, preserve those names in package inventories and method_obligations with concrete owners.",
            "A package that owns several table/figure units should expose which active code path implements each row family: setup, training/adaptation, evaluation metric, and artifact writer.",
            "Promote named experiments, environment/task inventories, baseline or method variants, benchmark measurements, prerequisite assets, parameter sweeps, expected result trends, and artifact coverage into inventories when they are present.",
            "When a unit or boundary requirement names paired mechanisms or alternatives such as ODE/SDE, train/eval, online/offline, actor/critic, or exact sampler/solver families, preserve each named mechanism explicitly in the relevant package goal, interface_contract, method_inventory, parameter_inventory, and method_obligations.",
            "Use stable inventory keys when applicable: `obligation_matrix`, `experiment_inventory`, `environment_inventory`, `method_inventory`, `policy_inventory`, `model_inventory`, `baseline_inventory`, `measurement_inventory`, `parameter_inventory`, `result_trend_inventory`, `refinement_inventory`, `artifact_inventory`, `result_artifact_inventory`, and `implementation_surface_inventory`.",
            "Named experiment sections such as Experiment I/II/III, ablations, case studies, or result-table protocols should remain visible as semantic anchors in `experiment_inventory`.",
            "`obligation_matrix` should bind each paper/addendum-visible named experiment/ablation/sensitivity check to the relevant environments/tasks, methods/baselines, parameters, expected trend or decision claim, and result artifacts.",
            "Each method/evaluation/experiment package should state `hypothesis` and `decision_value`; put required implementation scope in `goal`, `interface_contract`, `method_obligations`, inventories, and `scope_boundary.preserve`.",
            "Group exhaustive paper tables into decision-bearing packages: core method mechanism, main baseline comparison, metric/claim validation, and required robustness or ablation checks. Represent repeated variants through inventories/config when they share one implementation path.",
            "For bounded execution, document required config/registry entries, active code routes, artifact writers, and reporting logic for all paper/addendum-visible sweeps or trends in positive implementation fields.",
            "`method_obligations` should explain what code must expose for those inventories, such as environment factories, policy/model adapters, training or pretraining entrypoints, refinement algorithms, experiment registries, method selectors, metric formulas/aggregation, and result artifact writers.",
            "If `paper_evidence_contract.formula_algorithm_contract` contains anchors, propagate those anchors into method/training/evaluation work packages with exact symbols, formulas, numeric constants, search/schedule steps, and the files that will own the executable implementation.",
            "Method, training, evaluation, baseline, and refinement units should have executable code owners; artifact and protocol packages provide supplementary reporting and validation ownership.",
            "Keep work package ids short and stable.",
            "Use reference ids only when the prepared reference repositories actually support the package.",
            "Keep final file-layout decisions for the architecture and file-planning stages.",
        ],
        context_json=context_json,
    )
    return system, user


def build_work_package_planning_repair_prompt(
    *,
    context_json: str,
    previous_output_json: str,
    validation_errors: list[str],
    language: str = "zh",
) -> tuple[str, str]:
    """Repair prompt for work-package planning after deterministic review failures."""
    system = _build_stage_system_prompt(
        stage_label="work_package_planning_repair",
        responsibility="repair the work-package plan so it closes coverage and ownership issues reported by deterministic review",
        language=language,
    )
    repair_feedback = json.dumps(
        {
            "previous_output": json.loads(previous_output_json),
            "validation_errors": list(validation_errors),
        },
        ensure_ascii=False,
        indent=2,
    )
    user = _build_structured_user_prompt(
        instruction="Repair the work-package planning result using the deterministic review feedback.",
        schema_block="""{
  "work_packages": [
    {
      "work_package_id": "<paper_specific_work_package_id>",
      "goal": "build runnable repository surfaces",
      "hypothesis": "the package hypothesis",
      "decision_value": "decision enabled by this package",
      "owned_unit_ids": ["unit_001"],
      "tags": ["entrypoint", "artifact"],
      "reference_ids": ["ref_001"],
      "depends_on": [],
      "produces": ["main.py", "results/metrics.json"],
      "interface_contract": ["main entrypoint exists"],
      "evidence_needs": ["runnable repository surface"],
      "inventories": {"obligation_matrix": ["Experiment I: main comparison -> results/metrics.json"], "environment_inventory": ["task family"], "method_inventory": ["baseline", "proposed"], "parameter_inventory": ["alpha values if required"], "result_trend_inventory": ["expected trend if required"], "artifact_inventory": ["results/metrics.json"]},
      "scope_boundary": {"preserve": ["unit_001: implement the exact paper/addendum method, protocol, metric, and artifact obligation"], "implementation_focus": ["active code routes and artifacts this package must own"]},
      "method_obligations": ["preserve stable artifact contract"]
    }
  ],
  "coverage_summary": {"total_units": 1, "covered_units": 1, "uncovered_unit_ids": []},
  "planning_notes": ["repaired after deterministic work-package review feedback"]
}""",
        rules=[
            "Use the deterministic validation_errors as mandatory repair targets, not as optional advice.",
            "Preserve valid package ids, ownership bindings, and supported references whenever possible.",
            "Preserve or add `hypothesis`, `decision_value`, and positive implementation scope in package goals, contracts, inventories, and `scope_boundary.preserve`.",
            "Preserve or add `scope_boundary` for every repaired package using the structure `preserve` and `implementation_focus` only.",
            "`scope_boundary.preserve` must restate what owned Prepare units require the package to implement; `implementation_focus` must name active code routes and artifacts.",
            "Repair missing `obligation_matrix`, `experiment_inventory`, `environment_inventory`, `method_inventory`, `parameter_inventory`, `result_trend_inventory`, or artifact coverage when deterministic quality feedback reports missing paper/addendum-derived evidence obligations.",
            "Keep packages decision-bearing and use inventories/config to represent repeated variants when they share an implementation route.",
            "Repair only what is needed to close uncovered units, duplicate ids, empty outputs, or ownership drift.",
            "Keep downstream file-layout decisions in their existing stage.",
            "Return only strict JSON.",
        ],
        context_json=context_json + "\n\nRepair Feedback:\n\n" + repair_feedback,
    )
    return system, user


def build_global_contract_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-3.5 prompt: freeze a cross-stage contract snapshot before architecture."""
    system = _build_stage_system_prompt(
        stage_label="global_contract",
        responsibility="freeze a cross-stage contract snapshot covering work packages, inventories, result targets, and validation gates",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Synthesize the global contract from this context.",
        schema_block="""{
  "contract_version": "1.0",
  "canonical_stage_sequence": ["<setup_or_config_stage>", "<core_method_stage>", "<evaluation_stage>"],
  "work_package_contracts": [
    {"work_package_id": "<paper_specific_work_package_id>", "goal": "define runnable surfaces", "depends_on": [], "requirement_ids": ["req_001"], "reference_ids": ["ref_001"], "interface_contract": ["reference-derived or task-derived entrypoint exists"], "method_obligations": ["implement the runnable entrypoint and artifact writer"], "produces": ["cli", "results directory"], "inventories": {"experiment_inventory": ["classification experiment"], "implementation_surface_inventory": ["entrypoint", "artifact_writer"]}, "scope_boundary": {"preserve": ["unit_001: implement the exact required method/protocol/artifact route"], "implementation_focus": ["entrypoint and artifact writer"]}, "grounding_status": "grounded", "evidence_summary": ["ref_001 supports runnable experiment entrypoint structure"]}
  ],
  "inventories": {"obligation_matrix": ["main comparison: dataset x baseline x metric -> results/metrics.json"], "baseline_inventory": ["LogisticRegression", "RandomForestClassifier"], "parameter_inventory": [], "result_trend_inventory": []},
  "inventory_owners": {"baseline_inventory": {"LogisticRegression": ["evaluation_protocol"]}},
  "result_targets": [
    {"target_id": "measurement:accuracy", "kind": "measurement", "name": "accuracy", "owner_work_packages": ["evaluation_protocol"], "required_inputs": ["test predictions"], "artifact_paths": ["results/metrics.json"], "coverage_notes": ["must be written to metrics artifact"]}
  ],
  "benchmark_expectations": {},
  "validation_gates": ["static_contract_gate", "smoke_gate", "coverage_gate"],
  "contract_notes": ["Freeze entrypoint, artifact paths, baseline inventory, and measurement targets before architecture planning."]
}""",
        rules=[
            "This stage is the single contract snapshot used by architecture, generation, and validation.",
            "Copy the paper-derived inventory from `paper_evidence_contract.required_claim_inventory` into contract inventories and inventory owners. Missing exact names here will usually cause low final scores.",
            "Every contract inventory item must have an owner work package and, when it represents a table/figure/metric/result, at least one result target or artifact path.",
            "Carry each work package `scope_boundary` into `work_package_contracts`; preserve entries are mandatory implementation scope and implementation_focus names active routes/artifacts.",
            "Promote obligation matrix, named experiment inventory, environment/task inventory, baseline/method inventory, benchmark/measurement inventory, parameter sweeps, expected result trends, prerequisite assets, and result targets into first-class contract objects when they are present in the task.",
            "Preserve hypothesis, decision-value, and positive implementation scope from units and work packages in work-package contracts, inventories, result targets, and validation gates where applicable.",
            "Validation gates should prioritize benchmark-visible evidence for core contribution mechanisms, main comparisons, decisive metrics, and any paper-required sweeps.",
            "A required sweep may be bounded while keeping its parameter values, expected trend/failure condition, owner package, config or registry surface, and artifact/reporting target visible.",
            "Preserve unit-derived `method_obligations` and `implementation_surface_inventory`; these are mandatory downstream code-generation obligations.",
            "Match paper-benchmark complexity to the task while making the runnable path, artifact path, and result coverage explicit.",
            "Use concise work-package ids and keep the stage sequence implementation-oriented.",
            "`validation_gates` should usually distinguish static contract, smoke execution, and coverage-oriented checks.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 prompt: synthesize final repository architecture and file coverage."""
    system = _build_stage_system_prompt(
        stage_label="architecture",
        responsibility="synthesize the final repository architecture and file coverage for downstream generation from the task model, ref-repo model, and contract targets",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Synthesize the architecture from this context.",
        schema_block="""{
  "target_stack": ["<runtime_or_framework>"],
  "target_file_tree": ["<relative/path_a>", "<relative/path_b>"],
  "file_blueprints": [
    {"path": "<relative/path_a>", "purpose": "<what_this_file_owns>", "kind": "source | test | config | doc | script", "related_node_ids": ["<stage3_node_id>"], "based_on_references": ["<ref_id>"], "implementation_strategy": "new | adapted | reused"}
  ],
  "dependency_graph": [
    {"source_path": "<consumer_file>", "target_path": "<dependency_file>", "dependency_type": "<imports_or_other_dependency>"}
  ],
  "stable_interfaces": ["<public_or_stable_module_surface>"],
  "execution_entrypoints": ["<researcher_facing_entry_file>"],
  "config_surfaces": ["<config_or_env_surface>"],
  "package_layout": {"<work_package_id>": ["<owned_file_a>", "<owned_file_b>"]},
  "dependency_rules": ["<package_or_file_level_dependency_rule>"],
  "protocol_stages": ["<ordered_runtime_stage_name>"],
  "result_targets": ["<artifact_or_result_target_path>"],
  "architecture_reference_ids": ["<ref_id>"],
  "rationale": "<brief_rationale>"
}""",
        rules=[
            "This stage is the final authority for implementation file coverage.",
            "Use the provided `task_model` as the paper/task obligation model and `ref_repo_model` as the repository-structure prior.",
            "Use `paper_evidence_contract.required_claim_inventory` and the global-contract inventory owners as file-coverage requirements.",
            "Use `unit_route_matrix` as a closure checklist: every implementation-bearing prepare `unit_id` must map to a work package, an active Python owner file, and an entry/training/evaluation/reporting route.",
            "For every method/baseline/dataset/metric/hyperparameter/artifact inventory item, choose an active code owner: entrypoint, config/registry loaded by entrypoint, training/adaptation loop, evaluation/metric function, or artifact writer.",
            "Downstream task projection is deterministic one-file-one-task from this architecture.",
            "The example values above show schema shape only; choose file names from this task's obligations and prepared reference evidence.",
            "Use the provided local cloned reference repos and their surveys as implementation topology evidence, adapting the relevant structures to this task.",
            "The final structure must follow the provided plan, contract targets, and experiment obligations.",
            "`target_file_tree` is the exact set of implementation files that downstream generation will create.",
            "Derive the repository architecture from the current task obligations plus the prepared reference repository file trees and surveys. If a reference repo exposes relevant source paths, adapt that topology and name files after those concrete surfaces.",
            "Use a small conventional Python package layout when the reference repos provide no usable implementation structure for the obligation, with task-specific boundaries.",
            "Use paper-specific or reference-specific file names when available; place implementation modules directly under `src/` only when that is the clearest faithful structure.",
            "If compatibility with legacy imports is useful, plan `src/__init__.py` or package `__init__.py` alias surfaces explicitly rather than distorting the reference-derived layout.",
            "Every file in `target_file_tree` must appear exactly once in `file_blueprints`.",
            "Include exactly one file blueprint for every path in `target_file_tree`.",
            "`package_layout` values must only contain paths from `target_file_tree`; include work packages that need dedicated generated files.",
            "`package_layout` must give implementation-bearing `unit_route_matrix` rows at least one active Python owner file for that package plus a reachable route file when a unit owns method/data/model/baseline/metric/train/eval/artifact work.",
            "A single generated file may cover multiple paper obligations, but every generated file should have one most relevant owning work package.",
            "Choose file names from the task and reference evidence using paper-specific or reference-specific implementation nouns.",
            "If `main.py` or any other entrypoint file appears in `target_file_tree`, it must also appear in `file_blueprints` with a concrete purpose.",
            "Each blueprint must describe a file that is actually intended to be generated downstream.",
            "`dependency_graph` may only reference paths that exist in `target_file_tree`.",
            "Keep the file tree concise and implementation-oriented, but complete enough that no hidden support files are needed later.",
            "Include helper files only when they are required by the plan and experiment obligations.",
            "`related_node_ids` must come from the provided Stage-3 plan nodes and should preserve the ownership binding between plan nodes and target files.",
            "When using reference evidence, ground it in the provided local repo paths, likely reusable files, protocol clues, and matched implementation areas.",
            "Respect the provided global contract: entry surfaces, result targets, baseline inventory, benchmark/measurement coverage, and artifact paths should remain coherent.",
            "For papers with named experiments, multiple environments/tasks, baseline/method variants, or result tables, materialize explicit registry/config/runner/reporting surfaces so static review can find the experiment matrix in code.",
            "Paper-critical registry/config paths should be imported or loaded by a planned Python entrypoint, training/evaluation loop, or reporter.",
            "Represent the paper's experiment matrix as decision-bearing registries: core contribution hypothesis, main comparison, decisive metric, and required robustness/ablation checks. Use config and registry coverage for repeated variants that share one implementation path while preserving every paper-required variant by name.",
            "Architecture should make positive bounded-execution scope visible in config/docs/runner surfaces when paper-visible sweeps are represented by executable defaults.",
            "When the task includes pretraining, refinement, imitation learning, or algorithm comparisons, include concrete training/refinement/baseline surfaces with direct owner files.",
            "The architecture must expose one clear runnable entry surface and enough file coverage to satisfy the mandatory main experiment path before optional expansions.",
            "Return stable_interfaces, execution_entrypoints, config_surfaces, package_layout, dependency_rules, protocol_stages, and result_targets as explicit architecture contract fields.",
            "Prefer file boundaries that are stable and directly projectable into downstream package-scoped file planning and generation.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_task_model_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 helper prompt: derive architecture task model before synthesis."""
    system = _build_stage_system_prompt(
        stage_label="architecture_task_model",
        responsibility="extract a runnable task model that captures the main execution path, package responsibilities, and reproducibility obligations before repository synthesis",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build the architecture task model from this context.",
        schema_block="""{
  "execution_entry": "brief description of the canonical researcher-facing entry path",
  "runnable_flow": ["ordered runtime stage"],
  "method_spine": ["method-critical module or responsibility"],
  "package_responsibilities": [
    {
      "work_package_id": "wp_core",
      "responsibilities": ["what this package must own"],
      "method_obligations": ["paper-specific obligation"],
      "interface_surfaces": ["surface or boundary it must expose"],
      "owned_unit_ids": ["unit_001"]
    }
  ],
  "interface_closure": ["how config/method/eval/reporting connect"],
  "evidence_to_module_mapping": [
    {
      "work_package_id": "wp_core",
      "influenced_paths": ["src/core.py"],
      "supporting_references": ["ref_001"],
      "notes": ["why this package should shape those modules"]
    }
  ],
  "reproducibility_readiness": ["mandatory runnable or reporting surface"]
}""",
        rules=[
            "Return the runnable task model, package responsibilities, interface closure, and readiness surfaces for later architecture synthesis.",
            "Focus on the canonical runnable experiment path, method backbone, and reproducibility closure.",
            "Every input work_package_id must appear exactly once in `package_responsibilities`.",
            "Use package responsibilities and evidence bundles to keep the output paper-specific rather than generic.",
            "Surface mandatory artifacts, reporting paths, baseline or ablation obligations only when they are actually implied by the context.",
            "Capture the decision-bearing experiment spine: what hypothesis is tested, which comparison/metric changes the conclusion, and which paper-required sweeps share one bounded execution route.",
            "`runnable_flow` should be an ordered list of concrete experiment stages, not abstract principles.",
            "`reproducibility_readiness` should emphasize mandatory closure surfaces such as entrypoint, config, dependency manifest, environment guidance, and result artifacts.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_task_view_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 helper prompt: synthesize a task-only architecture proposal."""
    system = _build_stage_system_prompt(
        stage_label="architecture_task_view",
        responsibility="propose a repository architecture candidate purely from the current task obligations without relying on reference repo structure priors",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build a task-view architecture candidate from this context.",
        schema_block="""{
  "target_stack": ["python"],
  "target_file_tree": ["<reference_or_task_entry.py>", "<paper_specific_module.py>"],
  "file_blueprints": [
    {"path": "<reference_or_task_entry.py>", "purpose": "own the canonical experiment entry flow", "kind": "source", "related_node_ids": ["exp_001"], "based_on_references": [], "implementation_strategy": "new"}
  ],
  "dependency_graph": [
    {"source_path": "<reference_or_task_entry.py>", "target_path": "<paper_specific_module.py>", "dependency_type": "imports"}
  ],
  "stable_interfaces": ["<reference_or_task_entry.py>"],
  "execution_entrypoints": ["<reference_or_task_entry.py>"],
  "config_surfaces": ["<paper_specific_config_path>"],
  "package_layout": {"<work_package_id>": ["<reference_or_task_entry.py>"]},
  "dependency_rules": ["entrypoint imports package modules only through stable interfaces"],
  "protocol_stages": ["query planning", "retrieval", "filtering", "evaluation"],
  "result_targets": ["results/metrics.json"],
  "architecture_reference_ids": [],
  "rationale": "task-view architecture candidate grounded in task obligations"
}""",
        rules=[
            "Use the task-side context: normalized input, units, work packages, pipeline plan, global contract, and benchmark expectations.",
            "Derive this candidate from task semantics and implementation obligations.",
            "Materialize a complete multi-file repository architecture when the task implies multiple modules.",
            "The architecture must explicitly cover the method spine, package responsibilities, canonical entrypoint, config surface, and artifact-producing surfaces.",
            "The task-view candidate must preserve the decision-bearing experiment spine from units/work packages and use compact file boundaries for repeated ablations or table cells.",
            "Use a multi-file architecture when the task has distinct packages, routes, or artifacts.",
            "Every work package should have at least one materialized contract surface when the task implies runnable code ownership.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_ref_view_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 helper prompt: synthesize a reference-informed architecture proposal."""
    system = _build_stage_system_prompt(
        stage_label="architecture_ref_view",
        responsibility="propose a repository architecture candidate from grounded reference-repo structure, reusable modules, and evidence patterns",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build a reference-view architecture candidate from this context.",
        schema_block="""{
  "target_stack": ["python"],
  "target_file_tree": ["<reference_repo_entry_or_module.py>", "<reference_repo_method_module.py>"],
  "file_blueprints": [
    {"path": "<reference_repo_method_module.py>", "purpose": "own a reference-grounded reusable method surface", "kind": "source", "related_node_ids": ["exp_002"], "based_on_references": ["ref_001"], "implementation_strategy": "adapted"}
  ],
  "dependency_graph": [
    {"source_path": "<reference_repo_entry_or_module.py>", "target_path": "<reference_repo_method_module.py>", "dependency_type": "imports"}
  ],
  "stable_interfaces": ["<reference_repo_method_module.py>"],
  "execution_entrypoints": ["<reference_repo_entry_or_module.py>"],
  "config_surfaces": ["<reference_or_task_config_path>"],
  "package_layout": {"<work_package_id>": ["<reference_repo_method_module.py>"]},
  "dependency_rules": ["reference-inspired modules should preserve stable package seams"],
  "protocol_stages": ["query planning", "retrieval", "filtering", "evaluation"],
  "result_targets": ["results/metrics.json"],
  "architecture_reference_ids": ["ref_001"],
  "rationale": "reference-view architecture candidate grounded in surveyed repos"
}""",
        rules=[
            "Use the reference-side context: prepared reference repositories, repo surveys, evidence bundles, protocol clues, likely reusable files, and evidence file patterns.",
            "When reference structure is unavailable, return a compact faithful architecture candidate with empty `architecture_reference_ids`.",
            "Carry forward grounded structural priors such as package seams, config surfaces, entrypoints, evaluation modules, and artifact writers when the evidence supports them.",
            "Use reference structure to implement decisive method/evaluation paths and adapt only the relevant experiment surfaces.",
            "Keep enough files to cover the current task contract even when one reference repo is simple.",
            "Prefer architecture paths and package layout that can absorb grounded reusable modules while still fitting the current task contract.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_synthesis_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 prompt: synthesize the final architecture from task/ref candidates."""
    system = _build_stage_system_prompt(
        stage_label="architecture_synthesis",
        responsibility="synthesize the final repository architecture from the task-view candidate, the reference-view candidate, and the current contract targets",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Synthesize the final architecture by reconciling the task-view and reference-view candidates.",
        schema_block="""{
  "target_stack": ["<runtime_or_framework>"],
  "target_file_tree": ["<relative/path_a>", "<relative/path_b>"],
  "file_blueprints": [
    {"path": "<relative/path_a>", "purpose": "<what_this_file_owns>", "kind": "source | test | config | doc | script", "related_node_ids": ["<stage3_node_id>"], "based_on_references": ["<ref_id>"], "implementation_strategy": "new | adapted | reused"}
  ],
  "dependency_graph": [
    {"source_path": "<consumer_file>", "target_path": "<dependency_file>", "dependency_type": "<imports_or_other_dependency>"}
  ],
  "stable_interfaces": ["<public_or_stable_module_surface>"],
  "execution_entrypoints": ["<researcher_facing_entry_file>"],
  "config_surfaces": ["<config_or_env_surface>"],
  "package_layout": {"<work_package_id>": ["<owned_file_a>", "<owned_file_b>"]},
  "dependency_rules": ["<package_or_file_level_dependency_rule>"],
  "protocol_stages": ["<ordered_runtime_stage_name>"],
  "result_targets": ["<artifact_or_result_target_path>"],
  "architecture_reference_ids": ["<ref_id>"],
  "rationale": "<brief_rationale>"
}""",
        rules=[
            "Treat the task-view candidate as the primary semantic authority and the reference-view candidate as the structural prior.",
            "When candidates disagree, preserve task semantics first and then absorb grounded reference structure where compatible.",
            "The final architecture must be a complete multi-file repository design when the task contract implies multiple modules or package responsibilities.",
            "Use `unit_route_matrix` to verify the final architecture: implementation-bearing prepare units need active owner files and reachable entry/training/evaluation/reporting routes.",
            "Every file in `target_file_tree` must appear exactly once in `file_blueprints`.",
            "Resolve conflicts in favor of a closed architecture with explicit runnable routes.",
            "Package layout, entrypoints, config surfaces, artifact producers, and method spine materialization must all be explicit.",
            "Close method, dataset, model, baseline, metric, training, evaluation, and artifact-producing units with active code owners and reachable route files.",
            "Preserve the smallest decision-bearing experiment set: core mechanism, main baseline comparison, decisive metric, and required robustness/ablation coverage. Represent repeated paper-visible variants through executable config matrices and positive scope notes.",
        ],
        context_json=context_json,
    )
    return system, user


def build_architecture_repair_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4 repair prompt: fix architecture contract deviations without redesigning the repo freely."""
    system = _build_stage_system_prompt(
        stage_label="architecture_repair",
        responsibility="repair a repository architecture so it satisfies deterministic architecture contract checks",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Repair this architecture based on the reported deviations.",
        schema_block="""{
  "target_stack": ["<runtime_or_framework>"],
  "target_file_tree": ["<relative/path_a>", "<relative/path_b>"],
  "file_blueprints": [
    {"path": "<relative/path_a>", "purpose": "<what_this_file_owns>", "kind": "source | test | config | doc | script", "related_node_ids": ["<stage3_node_id>"], "based_on_references": ["<ref_id>"], "implementation_strategy": "new | adapted | reused"}
  ],
  "dependency_graph": [
    {"source_path": "<consumer_file>", "target_path": "<dependency_file>", "dependency_type": "<imports_or_other_dependency>"}
  ],
  "stable_interfaces": ["<public_or_stable_module_surface>"],
  "execution_entrypoints": ["<researcher_facing_entry_file>"],
  "config_surfaces": ["<config_or_env_surface>"],
  "package_layout": {"<work_package_id>": ["<owned_file_a>", "<owned_file_b>"]},
  "dependency_rules": ["<package_or_file_level_dependency_rule>"],
  "protocol_stages": ["<ordered_runtime_stage_name>"],
  "result_targets": ["<artifact_or_result_target_path>"],
  "architecture_reference_ids": ["<ref_id>"],
  "rationale": "<brief_rationale>"
}""",
        rules=[
            "Return a repaired architecture JSON object.",
            "Preserve the existing design and change the fields needed to close the listed deviations.",
            "Preserve valid file boundaries, related_node_ids, and architecture_reference_ids whenever possible.",
            "When deviations mention unit route coverage, repair package_layout, active owner files, entrypoints, and dependency_graph so the prepare `unit_id` has a reachable Python route.",
            "Keep the architecture concise, runnable, and projectable into package-scoped file planning.",
            "Keep the architecture complete enough to cover package responsibilities, routes, and artifacts.",
            "Every file in `target_file_tree` must still appear exactly once in `file_blueprints`.",
            "`dependency_graph` may only reference files that are present in `target_file_tree`.",
        ],
        context_json=context_json,
    )
    return system, user


def build_package_file_planning_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Stage-4.5 prompt: freeze package-scoped file plans from the architecture."""
    system = _build_stage_system_prompt(
        stage_label="package_file_planning",
        responsibility="freeze package-scoped file plans, dependencies, interfaces, and review focus from the architecture and global contract",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=(
            "Build the package file planning result as a complete code-generation contract. Each file plan "
            "must describe faithful implementation work needed for the judgeable reproduction repo."
        ),
        schema_block="""{
  "file_plans": [
    {
      "target_file": "main.py",
      "task_id": "task_main",
      "work_package_id": "<work_package_id>",
      "purpose": "implement the main experiment entrypoint",
      "related_node_ids": ["node_main"],
      "owned_units": ["unit_001"],
      "reference_ids": ["ref_001"],
      "depends_on": [],
      "blocking_dependencies": [],
      "requires_stable_dependencies": true,
      "interface_contract": ["expose the canonical experiment entry surface"],
      "implementation_surfaces": ["entrypoint", "artifact_writer"],
      "method_obligations": ["implement CLI argument parsing and write results/metrics.json"],
      "hypothesis": "canonical entrypoint exercises the decisive reproduction path",
      "decision_value": "shows whether the repo can run the experiment path needed for semantic review",
      "context_sources": ["node:node_main", "ref:ref_001"],
      "consumes": [],
      "produces": ["main.py"],
      "defines_symbols": ["main"],
      "calls_symbols": [],
      "writes_artifacts": [],
      "reads_artifacts": [],
      "allowed_scope": {"read": ["README.md"], "write": ["main.py"]},
      "scope_boundary": {"preserve": ["unit_001: implement the file-owned paper/addendum route completely"], "implementation_focus": ["CLI and artifact writer code in main.py"]},
      "generation_prompt": "implement main.py as the canonical experiment entrypoint and preserve the contract-owned execution closure",
      "validation_hooks": ["python_syntax"],
      "review_points": ["ensure artifact output path is preserved"],
    }
  ],
  "planning_notes": ["keep one closed file plan per target file and preserve package contract closure"]
}""",
        rules=[
            "This stage is the final file-level planning stage before local generation.",
            "Every planned file must be intended as complete reproduction code for its owned obligations.",
            "Return one package-scoped file plan for each architecture `target_file_tree` path, and only those paths.",
            "`owned_units` must contain prepare `unit_id` values from `units` / `unit_route_matrix`; keep Stage-3 ids in `related_node_ids`.",
            "Use `unit_route_matrix.rows` as the mandatory closure checklist. For every implementation-bearing row, ensure at least one file plan owns the prepare unit, at least one owned file is an active Python implementation file, and at least one entry/training/evaluation/reporting route file calls or loads that implementation.",
            "For each relevant `paper_evidence_contract.closure_items` entry, make sure at least one file plan contains the exact item name or clear alias in `method_obligations`, `writes_artifacts`, `generation_prompt`, or `review_points` and owns an active code route.",
            "For each relevant `paper_evidence_contract.formula_algorithm_contract` anchor, assign the exact required symbols, numeric values, formulas, search/schedule steps, and algorithm steps to concrete method/model/training/evaluation/metric/config files through `method_obligations`, `defines_symbols`, `calls_symbols`, `generation_prompt`, and `review_points`; generic interface or artifact labels are insufficient.",
            "File plans must state which planned Python source loads each registry/config and which source writes each named table/figure artifact; data files should be indexed by active source owners.",
            "Preserve grouped source-package paths from the architecture during file planning.",
            "For Python package `__init__.py` files, make the compatibility/export contract explicit when legacy import surfaces are required.",
            "Every file plan must correspond to an existing architecture blueprint path.",
            "Keep dependencies, package ownership, and related node bindings consistent with the provided architecture and pipeline plan.",
            "Respect the global contract, package interfaces, canonical entry surface, and declared artifact outputs.",
            "Carry package `method_obligations` and `implementation_surface_inventory` into each relevant file plan's `method_obligations`, `implementation_surfaces`, `generation_prompt`, and review points.",
            "Carry `hypothesis`, `decision_value`, required implementation scope, and `scope_boundary.preserve` into relevant file plan fields and review points.",
            "Carry package `scope_boundary` into each relevant file plan. File-level `scope_boundary.preserve` must say what this file must faithfully and completely implement; `implementation_focus` must name active code routes/artifacts.",
            "Use file-level `scope_boundary.preserve` and `implementation_focus` to keep named methods, baselines, datasets, metrics, tables/figures, hyperparameters, training/evaluation routes, and artifact writers owned by the file's Prepare units visible.",
            "File plans should implement decision-bearing experiment code paths. Use configuration/registries for repeated variants whose marginal information gain is low, while keeping benchmark-visible required variants discoverable.",
            "Carry package inventories for named experiments, environments/tasks, baselines/variants, measurements, and result artifacts into the relevant file plan's `method_obligations`, `writes_artifacts`, `generation_prompt`, and `review_points`.",
            "For files owning orchestration, evaluation, reporting, config, or entrypoints, require explicit registries or artifact writers for the named experiment/result matrix with concrete runner/reporting surfaces.",
            "Assign each paper-critical method/baseline/metric/table/figure obligation to at least one active Python entrypoint, training/evaluation, or reporting file that calls or writes the corresponding artifact; README, config, JSON registry, or manifest files can index those routes.",
            "For files owning method/model/data/baseline/metric/training/evaluation units, include concrete `defines_symbols`; for route files owning entrypoint/orchestration/reporting, include concrete `calls_symbols` that reach those owners.",
            "For low-cost smoke paths, require schema/provenance artifacts that exercise the same implementation routes used for named datasets, model loaders, baselines, fine-tuning/refinement jobs, metric formulas, or figure/table writers.",
            "Plan paper-visible table/figure/metric/prediction/report outputs as measured outputs produced through the implementation route; use explicit full-mode commands for expensive complete results.",
            "Prefer narrow `allowed_scope`, concrete `review_points`, and generation prompts that are specific to the file's package responsibility.",
            "Plan each architecture file's contents within the architecture file set.",
        ],
        context_json=context_json,
    )
    return system, user


def build_package_file_planning_repair_prompt(
    *,
    context_json: str,
    previous_output_json: str,
    validation_errors: list[str],
    language: str = "zh",
) -> tuple[str, str]:
    """Repair prompt for package-file planning after deterministic contract review failures."""
    system = _build_stage_system_prompt(
        stage_label="package_file_planning_repair",
        responsibility="repair the package-scoped file plans so they satisfy deterministic contract-closure review",
        language=language,
    )
    repair_feedback = json.dumps(
        {
            "previous_output": json.loads(previous_output_json),
            "validation_errors": list(validation_errors),
        },
        ensure_ascii=False,
        indent=2,
    )
    user = _build_structured_user_prompt(
        instruction="Repair the package file planning result using the deterministic contract review feedback.",
        schema_block="""{
  "file_plans": [
    {
      "target_file": "main.py",
      "task_id": "task_main",
      "work_package_id": "<work_package_id>",
      "purpose": "implement the main experiment entrypoint",
      "related_node_ids": ["node_main"],
      "owned_units": ["unit_001"],
      "reference_ids": ["ref_001"],
      "depends_on": [],
      "blocking_dependencies": [],
      "requires_stable_dependencies": true,
      "interface_contract": ["expose the canonical experiment entry surface"],
      "implementation_surfaces": ["entrypoint", "artifact_writer"],
      "method_obligations": ["implement CLI argument parsing and write results/metrics.json"],
      "context_sources": ["node:node_main", "ref:ref_001"],
      "consumes": [],
      "produces": ["main.py"],
      "defines_symbols": ["main"],
      "calls_symbols": [],
      "writes_artifacts": [],
      "reads_artifacts": [],
      "allowed_scope": {"read": ["README.md"], "write": ["main.py"]},
      "scope_boundary": {"preserve": ["unit_001: implement the file-owned paper/addendum route completely"], "implementation_focus": ["CLI and artifact writer code in main.py"]},
      "generation_prompt": "implement main.py as the canonical experiment entrypoint and preserve the contract-owned execution closure",
      "validation_hooks": ["python_syntax"],
      "review_points": ["ensure artifact output path is preserved"]
    }
  ],
  "planning_notes": ["repaired after deterministic package-file review feedback"]
}""",
        rules=[
            "Use the deterministic validation_errors as mandatory repair targets, not as optional advice.",
            "Preserve valid file ownership, dependency ordering, and reference bindings whenever possible.",
            "Preserve file-level `method_obligations`, `implementation_surfaces`, and `scope_boundary.preserve` when repairing.",
            "Use `scope_boundary.preserve` to restate active owned-unit coverage and `implementation_focus` to name the corresponding routes/artifacts.",
            "When validation_errors mention implementation-unit route coverage, repair `owned_units` with prepare `unit_id` values, add active Python owner files where the architecture already provides them, and add `calls_symbols`/review points on route files.",
            "Repair only what is needed to close contract coverage, artifact wiring, entrypoint coverage, or execution-closure issues.",
            "Return only strict JSON.",
        ],
        context_json=context_json + "\n\nRepair Feedback:\n\n" + repair_feedback,
    )
    return system, user


def build_package_file_planning_schema_fix_prompt(
    *,
    context_json: str,
    invalid_payload_json: str,
    validation_error: str,
    language: str = "zh",
) -> tuple[str, str]:
    """Repair a package-file-planning JSON payload that failed schema validation."""
    system = _build_stage_system_prompt(
        stage_label="package_file_planning_schema_fix",
        responsibility="repair invalid package_file_planning JSON so it exactly matches the required schema",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=(
            "The previous package_file_planning output failed schema validation. "
            "Return a corrected full package_file_planning JSON object."
        ),
        schema_block="""{
  "file_plans": [
    {
      "target_file": "main.py",
      "task_id": "task_main",
      "work_package_id": "<work_package_id>",
      "purpose": "implement the main experiment entrypoint",
      "related_node_ids": ["node_main"],
      "owned_units": ["unit_001"],
      "reference_ids": ["ref_001"],
      "depends_on": [],
      "blocking_dependencies": [],
      "requires_stable_dependencies": true,
      "interface_contract": ["expose the canonical experiment entry surface"],
      "implementation_surfaces": ["entrypoint", "artifact_writer"],
      "method_obligations": ["implement CLI argument parsing and write results/metrics.json"],
      "context_sources": ["node:node_main", "ref:ref_001"],
      "consumes": [],
      "produces": ["main.py"],
      "defines_symbols": ["main"],
      "calls_symbols": [],
      "writes_artifacts": [],
      "reads_artifacts": [],
      "allowed_scope": {"read": ["README.md"], "write": ["main.py"]},
      "scope_boundary": {"preserve": ["unit_001: implement the file-owned paper/addendum route completely"], "implementation_focus": ["CLI and artifact writer code in main.py"]},
      "generation_prompt": "implement main.py as the canonical experiment entrypoint and preserve the contract-owned execution closure",
      "validation_hooks": ["python_syntax"],
      "review_points": ["ensure artifact output path is preserved"],
    }
  ],
  "planning_notes": ["keep one closed file plan per target file and preserve package contract closure"]
}""",
        rules=[
            "Keep the planned file set aligned with the architecture coverage.",
            "Preserve any valid `method_obligations` and `implementation_surfaces`; add them when the context clearly assigns code obligations to the file.",
            "Preserve or add `scope_boundary`; keep `preserve` as mandatory implementation scope and `implementation_focus` as active code route/artifact focus.",
            "`owned_units` must be prepare `unit_id` values, not Stage-3 `node_id` values.",
            "Return the corrected full JSON object with no surrounding prose or markdown fences.",
        ],
        context_json=(
            f"Validation error:\n{validation_error}\n\n"
            f"Invalid payload:\n{invalid_payload_json}\n\n"
            f"Original planning context:\n{context_json}"
        ),
    )
    return system, user

def build_result_report_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Result-report prompt: summarize final workflow outcome."""
    system = _build_stage_system_prompt(
        stage_label="result_report",
        responsibility="summarize the final experiment outcome into a concise decision-oriented report",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Generate the final experiment result report from this context.",
        schema_block="""{
  "title": "short report title",
  "objective": "one short paragraph describing the experiment objective",
  "executive_summary": "one short paragraph summarizing the final result",
  "outcome": "brief statement of final status and what was achieved",
  "key_results": ["result 1", "result 2"],
  "evidence": [{"label": "evidence label", "detail": "supporting detail"}],
  "limitations": ["limitation 1"],
  "next_steps": ["next step 1"],
  "artifacts": ["important artifact path or identifier"]
}""",
        rules=[
            "Work for the experiment type expressed in the provided context.",
            "Use only information explicitly present in the provided context.",
            "If a detail is unavailable, say so plainly instead of inventing it.",
            "Focus on final outcome, most important evidence, and practical interpretation.",
            "Keep the report concise and decision-oriented.",
            "`key_results`, `evidence`, `limitations`, `next_steps`, and `artifacts` must always be arrays.",
            "Keep each array concise.",
            "Return exactly the schema keys.",
        ],
        context_json=context_json,
    )
    return system, user


def build_file_generation_prompt(generation_bundle_json: str, language: str = "zh") -> tuple[str, str]:
    """Generation prompt: produce executable project files from one manifest bundle."""
    system = _build_stage_system_prompt(
        stage_label="file_generation",
        responsibility="generate executable project files from the prepared generation manifest",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction=(
            "Generate faithful, complete, judgeable reproduction project files from this generation bundle. "
            "The output must implement the paper obligations as executable code/config/reporting routes."
        ),
        schema_block="""{
  "summary": "short generation summary",
  "project_files": {
    "relative/path.py": "full file content"
  }
}""",
        rules=[
            "`project_files` must be a JSON object keyed by relative file path.",
            "Each value must be the full file content, not a diff.",
            "Follow `ordered_tasks` and `task_inputs` to decide generation order and dependency usage.",
            "Every generated file should be complete for its declared obligations with faithful code/config/reporting content.",
            "For each target file, focus on that file's own `task_inputs` only; use dependency context only to call stable interfaces.",
            "Treat paper-derived claim inventory carried in the generation bundle as mandatory implementation content. Preserve exact named datasets, methods/baselines, metrics, hyperparameters, tables, and figures in runnable code/config/reporting paths.",
            "When implementing a registry/config for paper claims, ensure the entrypoint, training/evaluation loop, or artifact writer imports or loads it. A detached registry that is never called is not a reproduction.",
            "Use `snippet_candidates` and their `insight` fields when they are relevant to the target files; if a target file has snippet candidates, adapt at least one relevant reference-backed code/config/protocol pattern or leave a short `reference_grounding: <ref_id> <source_path>` incompatibility note in that file.",
            "Reference snippets are evidence for implementation shape and protocol details. Preserve their source path/ref id as provenance markers so repair can verify grounding later.",
            "Honor the provided topic profile and global contract guidance, especially entrypoints, artifact paths, result targets, baseline inventory, and measurement outputs.",
            "Derive evidence obligations from the supplied paper/addendum/assets, prepared provenance policy, and grounded reference repositories.",
            "Materialize named experiments, environments/tasks, baselines/variants, and measurements as code/config registries, selectable options, artifact writers, or aggregation functions in the files that own them.",
            "For environment/task inventories, write explicit registry entries with environment ids, aliases, setup metadata, and initialization/factory surfaces when the target file owns environments or config.",
            "For named experiment inventories, write an explicit protocol matrix that connects experiments to environments/tasks, compared methods, required measurements, and result artifact paths.",
            "For baseline/method inventories, expose selectable method names and adapter functions/classes rather than only mentioning them in README text.",
            "When a task has `writes_artifacts`, implement the writer or declaration surface in code so the artifact path is discoverable by static review.",
            "When a task mentions result tables, figures, curves, or named experiments, expose a concrete result-schema or table-writer path with paper-specific artifact names.",
            "Result schemas, protocol manifests, dry-run readiness manifests, and artifact writers are supplementary surfaces alongside method, environment, policy/model, training, refinement, baseline, metric, and evaluation implementation.",
            "Canonical train, evaluate, refine, compare, and run-all modes must route to concrete implementation functions/classes. Explicit smoke or dry-run modes may emit readiness summaries only when they call the real bounded code paths.",
            "Benchmark-visible metrics/tables/figures/predictions/reports must be computed bounded measured outputs from the implementation route or exposed through a full-mode path that requires real assets.",
            "Paper-derived method, metric, training, refinement, and environment obligations must appear in primary code paths or registries that those paths call, with support manifests as supplementary evidence.",
            "If the paper names external datasets or large models, implement lazy full-mode loaders and training/evaluation functions for those exact assets; full mode can require optional dependencies or user-provided paths.",
            "If the paper requires per-example updates, pairwise labels, Cartesian-product evaluation, train/test splits, or checkpoint/adaptor outputs, implement those loops explicitly and expose their artifact outputs.",
            "Implement concrete dataset obtain/prepare/validate functions, model loader/factory functions, metric formulas/aggregation, attack/adaptation algorithms, training/evaluation loops, per-sample bookkeeping, and dedicated table/figure artifact writers whenever the paper/addendum/ref evidence implies those surfaces.",
            "Satisfy dataset/model/metric/protocol obligations with executable code/config paths and supporting documentation or manifests.",
            "Prefer generating all files declared by the generation bundle.",
            "The declared main entrypoint file must be present.",
            "Keep the output focused on implementation only; generate only interfaces, JSON examples, and support files requested by the plan.",
            "Return strict JSON with no surrounding prose or markdown fences.",
        ],
        context_json=generation_bundle_json,
    )
    return system, user


def build_preflight_fix_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Repair prompt wrapper for the Ralph preflight stage."""
    return _build_stage_fix_prompt(
        "preflight",
        "Fix project structure, missing files, entrypoints, syntax errors, and basic import issues before execution.",
        context_json,
        language,
    )


def build_runtime_smoke_fix_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Repair prompt wrapper for the legacy runtime-smoke stage."""
    return _build_stage_fix_prompt(
        "runtime-smoke",
        "Fix lightweight execution failures so the project can start successfully in the runtime-smoke stage.",
        context_json,
        language,
    )


def build_docker_validate_fix_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Repair prompt wrapper for the legacy docker-validation stage."""
    return _build_stage_fix_prompt(
        "docker-validate",
        "Fix docker validation failures, metrics artifact generation, and metrics contract issues.",
        context_json,
        language,
    )


def build_repair_plan_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Build the repo/problem-driven repair-plan generation prompt."""
    system = _build_stage_system_prompt(
        stage_label="repair_plan_gen",
        responsibility="build a repo-wide repair plan that preserves task semantics while fixing the current repository",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Build a repo/problem-driven repair plan from the requirement anchor, current repo evaluation findings, and current validation failures.",
        schema_block="""{
  "summary": "brief repair strategy",
  "problem_list": ["one blocking repository problem"],
  "semantic_guardrails": ["one semantic invariant that must not drift"],
  "runtime_guardrails": ["one runtime closure invariant"],
  "recommended_surfaces": ["relative/path.py or repo surface"],
  "forbidden_shortcuts": ["one semantic or runtime shortcut risk"],
  "repair_guidance": ["one concrete repo-wide repair instruction"],
  "review_focus": ["one repo-level review focus"],
  "acceptance_criteria": ["one acceptance criterion"],
  "round_budget": 30
}""",
        rules=[
            "Use the frozen requirement anchor as the stable semantic baseline for repair.",
            "If `runtime_first_blockers` is non-empty, prioritize the exact SyntaxError/ImportError/ModuleNotFoundError/entrypoint traceback first; semantic and artifact expansion waits until the repo starts.",
            "When there is no startup blocker, prioritize repo-level semantic alignment first, runnable closure second, and optional cleanup last.",
            "Use `recommended_surfaces` as a conservative recommendation, not as a mandatory allowlist.",
            "Use `semantic_guardrails` and `runtime_guardrails` for non-negotiable repair boundaries.",
            "Use `forbidden_shortcuts` as the schema field for semantic and runtime closure risks.",
            "Use `repair_guidance` for repo-wide repair constraints.",
            "Keep repair aligned to the existing repo phases, routes, and task.",
            "`round_budget` should be a positive integer and can stay aligned with the outer repair-loop budget.",
        ],
        context_json=context_json,
    )
    return system, user


def build_repair_plan_review_prompt(context_json: str, language: str = "zh") -> tuple[str, str]:
    """Build the repo/problem-driven repair-plan evaluation prompt."""
    system = _build_stage_system_prompt(
        stage_label="repair_plan_eval",
        responsibility="review and tighten the generated repo-wide repair plan so repair remains semantically aligned, runnable, and minimal",
        language=language,
    )
    user = _build_structured_user_prompt(
        instruction="Review this repair plan and return the tightened repair plan assessment.",
        schema_block="""{
  "approved": true,
  "summary": "brief evaluation summary",
  "semantic_risks": ["semantic drift risk"],
  "runtime_risks": ["runtime closure risk"],
  "accepted_surfaces": ["relative/path.py or repo surface"],
  "required_review_points": ["review point"],
  "acceptance_criteria": ["acceptance criterion"],
  "round_budget": 30
}""",
        rules=[
            "Review the plan against the requirement anchor, current repo evaluation findings, and validation failure context.",
            "If `runtime_first_blockers` is non-empty, reject or tighten plans that spread effort across broad semantic/artifact work before fixing the exact startup traceback.",
            "Tighten the repair surface instead of expanding it unless expansion is clearly necessary to fix the repo.",
            "Use `semantic_risks` and `runtime_risks` for the semantic and runtime invariants repair must preserve.",
            "Use `required_review_points` and `acceptance_criteria` to sharpen downstream repo-level review and validation.",
            "Keep `accepted_surfaces` concrete and minimal.",
            "Keep the review aligned to existing repo phases, routes, and task.",
            "`round_budget` should remain practical and may match the outer repair-loop budget when needed.",
        ],
        context_json=context_json,
    )
    return system, user
