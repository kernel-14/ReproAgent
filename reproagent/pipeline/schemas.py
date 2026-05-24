"""Experiment generation schemas."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PaperBenchReproInput(BaseModel):
    """Experiment generation input."""
    target: str
    paper_path: str = ""
    paper_text: str = ""
    proposal_path: str = ""
    proposal_text: str = ""
    chunk_max_chars: int = 6000
    language: str = "zh"
    thread_id: str = ""
    max_iterations: int = 30
    resume_from_run_id: str = ""
    fork_from_run_id: str = ""
    resume_in_place: bool = False
    resume_start_stage: str = ""
    stage_review_repair_budget: int = 3
    experiment_design: dict[str, Any] = Field(default_factory=dict)
    github_config: dict[str, Any] = Field(default_factory=dict)
    github_upload_config: dict[str, Any] = Field(default_factory=dict)
    idea_references: list[dict[str, Any]] = Field(default_factory=list)
    idea_reference_summaries: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, value: int) -> int:
        if value < 1 or value > 30:
            raise ValueError("max_iterations must be between 1 and 30")
        return value


class PaperChunk(BaseModel):
    """One section/paragraph chunk from the target paper."""
    chunk_id: str
    section_title: str = ""
    ordinal: int = 0
    source_path: str = ""
    text: str = ""
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    split_reason: Literal["section", "paragraph_overflow", "document"] = "section"


class UpstreamIntentContract(BaseModel):
    """Frozen source-of-truth contract for the upstream experiment request."""
    schema_version: str = "1.0"
    target: str = ""
    language: str = "zh"
    experiment_design: dict[str, Any] = Field(default_factory=dict)
    idea_references: list[dict[str, Any]] = Field(default_factory=list)
    idea_reference_summaries: list[dict[str, Any]] = Field(default_factory=list)
    dataset_contract: dict[str, Any] = Field(default_factory=dict)
    benchmark_contract: dict[str, Any] = Field(default_factory=dict)
    required_datasets: list[dict[str, Any]] = Field(default_factory=list)
    required_metrics: list[str] = Field(default_factory=list)
    required_baselines: list[str] = Field(default_factory=list)
    required_ablations: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    explicit_constraints: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    allowed_approximations: list[str] = Field(default_factory=list)
    source_fields: list[str] = Field(default_factory=list)


class FileProvenanceRecord(BaseModel):
    """Trace one generated file back to upstream intent and planning contracts."""
    path: str
    generated: bool = True
    source_requirement_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    source_reference_ids: list[str] = Field(default_factory=list)
    owned_unit_ids: list[str] = Field(default_factory=list)
    owner_work_package_ids: list[str] = Field(default_factory=list)
    related_plan_node_ids: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)


class InputNormalizationOutput(BaseModel):
    """Normalized upstream request payload aligned with reproduction input normalization."""
    normalized_target: str = ""
    target_summary: str = ""
    task_type: str = "generic_experiment"
    key_entities: list[str] = Field(default_factory=list)
    explicit_constraints: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    assumption_notes: list[str] = Field(default_factory=list)


class VerificationTarget(BaseModel):
    """Lightweight verification target aligned with reproduction unit extraction."""
    kind: str
    description: str


class ExtractedUnit(BaseModel):
    """Compact implementation unit aligned with reproduction's unit concept."""
    unit_id: str
    type: Literal["task", "method", "protocol", "claim", "artifact"]
    statement: str
    hypothesis: str = ""
    decision_value: str = ""
    stop_rule_or_pruning_rationale: str = ""
    paper_evidence: list[str] = Field(default_factory=list)
    source_paragraph_ids: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    verification_targets: list[VerificationTarget] = Field(default_factory=list)
    implementation_surfaces: list[str] = Field(default_factory=list)
    code_obligations: list[str] = Field(default_factory=list)
    runtime_interfaces: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    suggested_module_kinds: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    status: str = "active"


class UnitExtractionOutput(BaseModel):
    """Structured unit-extraction output."""
    units: list[ExtractedUnit] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
    section_coverage: list[str] = Field(default_factory=list)


class BoundaryRequirement(BaseModel):
    """One lightweight boundary requirement."""
    requirement_id: str
    title: str
    category: str = "experiment"
    scope: str = ""
    description: str = ""
    source_unit_ids: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        requirement_id = str(payload.get("requirement_id", "") or "").strip()
        if not requirement_id:
            for key in ("require_id", "id"):
                alias_value = str(payload.get(key, "") or "").strip()
                if alias_value:
                    payload["requirement_id"] = alias_value
                    break
        if "acceptance_criteria" not in payload:
            criteria = payload.get("criteria")
            if isinstance(criteria, list):
                payload["acceptance_criteria"] = criteria
            elif isinstance(criteria, str) and criteria.strip():
                payload["acceptance_criteria"] = [criteria.strip()]
            elif criteria is not None:
                try:
                    payload["acceptance_criteria"] = [
                        str(item).strip()
                        for item in list(criteria)
                        if str(item).strip()
                    ]
                except TypeError:
                    pass
        return payload


class BoundaryRequirementsOutput(BaseModel):
    """Stage-1 boundary requirements output."""
    boundary_requirements: list[BoundaryRequirement] = Field(default_factory=list)
    requirement_scope_items: list[str] = Field(default_factory=list)


class TopicProfileOutput(BaseModel):
    """Stage-1.5 topic guidance derived from the task and boundary requirements."""
    primary_topic: str = "generic_experiment"
    active_topics: list[str] = Field(default_factory=list)
    reference_topics: list[str] = Field(default_factory=list)
    experiment_traits: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)
    coverage_policy: Literal["force", "balanced", "advisory"] = "balanced"
    coverage_hints: dict[str, list[str]] = Field(default_factory=dict)
    prompt_guidance: list[str] = Field(default_factory=list)


class ReferenceRequirementCoverage(BaseModel):
    """One requirement-to-repository coverage row extracted from local survey."""
    requirement_id: str
    title: str = ""
    scope: str = ""
    source_unit_ids: list[str] = Field(default_factory=list)
    keyword_hits: int = 0
    matched_keywords: list[str] = Field(default_factory=list)
    matched_files: list[str] = Field(default_factory=list)
    match_locations: list[str] = Field(default_factory=list)
    code_snippets: list[str] = Field(default_factory=list)


class ReferenceSymbolEvidence(BaseModel):
    """One symbol/snippet-level evidence row extracted from a prepared reference repo."""
    evidence_id: str
    ref_id: str = ""
    file_path: str = ""
    symbol_name: str = ""
    symbol_kind: str = ""
    start_line: int = 0
    end_line: int = 0
    snippet: str = ""
    matched_unit_ids: list[str] = Field(default_factory=list)
    matched_requirement_ids: list[str] = Field(default_factory=list)
    matched_surfaces: list[str] = Field(default_factory=list)
    matched_artifacts: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    relevance_reason: str = ""
    score: float = 0.0


class PreparedReferenceRepositorySurvey(BaseModel):
    """Structured local survey for one prepared reference repository."""
    ref_id: str
    title: str
    paper_path: str = ""
    paper_url: str = ""
    repository_url: str = ""
    repository_origin: str = ""
    repository_type: str = ""
    reference_role: str = ""
    local_repo_path: str = ""
    default_branch: str = ""
    status: str = ""
    readme_summary: str = ""
    file_tree_summary: str = ""
    top_level_files: list[str] = Field(default_factory=list)
    top_python_files: list[str] = Field(default_factory=list)
    likely_reusable_files: list[str] = Field(default_factory=list)
    protocol_clues: list[str] = Field(default_factory=list)
    source_file_count: int = 0
    requirement_coverage: list[ReferenceRequirementCoverage] = Field(default_factory=list)
    symbol_evidence: list[ReferenceSymbolEvidence] = Field(default_factory=list)


class ActionableReference(BaseModel):
    """One retained actionable reference from idea_gen."""
    ref_id: str
    title: str
    paper_url: str = ""
    repository_url: str = ""
    repository_origin: Literal["official", "community", "unknown"] = "unknown"
    raw_repository_origin: str = ""
    source_kind: str = ""
    local_repo_path: str = ""
    default_branch: str = ""
    supported_requirement_ids: list[str] = Field(default_factory=list)
    reusable_modules: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    file_tree: str = ""
    readme_summary: str = ""
    top_python_files: list[str] = Field(default_factory=list)
    likely_reusable_files: list[str] = Field(default_factory=list)
    protocol_clues: list[str] = Field(default_factory=list)
    requirement_coverage: list[ReferenceRequirementCoverage] = Field(default_factory=list)
    symbol_evidence: list[ReferenceSymbolEvidence] = Field(default_factory=list)

    @field_validator("repository_origin", mode="before")
    @classmethod
    def normalize_repository_origin(cls, value: object) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"official", "community", "unknown"}:
            return normalized
        if normalized in {"third_party", "thirdparty", "library", "external", "unofficial"}:
            return "community"
        return "unknown"


class ReferenceRelation(BaseModel):
    """Relation between one reference and requirement scopes."""
    ref_id: str
    supported_scope_items: list[str] = Field(default_factory=list)
    reference_role: str = ""


class ReferenceSelectionOutput(BaseModel):
    """Stage-2 reference selection output."""
    actionable_references: list[ActionableReference] = Field(default_factory=list)
    reference_relations: list[ReferenceRelation] = Field(default_factory=list)


class GlobalContractResultTarget(BaseModel):
    """One result target or artifact target in the global contract."""
    target_id: str
    kind: str
    name: str
    owner_work_packages: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class GlobalContractWorkPackage(BaseModel):
    """Condensed work-package contract used by downstream planning and generation."""
    work_package_id: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    interface_contract: list[str] = Field(default_factory=list)
    method_obligations: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    inventories: dict[str, list[str]] = Field(default_factory=dict)
    scope_boundary: dict[str, list[str]] = Field(default_factory=dict)
    grounding_status: str = ""
    evidence_summary: list[str] = Field(default_factory=list)


class GlobalContractOutput(BaseModel):
    """Cross-stage contract snapshot for architecture, generation, and validation."""
    contract_version: str = "1.0"
    canonical_stage_sequence: list[str] = Field(default_factory=list)
    work_package_contracts: list[GlobalContractWorkPackage] = Field(default_factory=list)
    inventories: dict[str, list[str]] = Field(default_factory=dict)
    inventory_owners: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    result_targets: list[GlobalContractResultTarget] = Field(default_factory=list)
    benchmark_expectations: dict[str, Any] = Field(default_factory=dict)
    validation_gates: list[str] = Field(default_factory=list)
    contract_notes: list[str] = Field(default_factory=list)


class WorkPackageItem(BaseModel):
    """Condensed work package aligned with reproduction mid-stage planning."""
    work_package_id: str
    goal: str
    hypothesis: str = ""
    decision_value: str = ""
    stop_rule_or_pruning_rationale: str = ""
    owned_unit_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    interface_contract: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)
    inventories: dict[str, list[str]] = Field(default_factory=dict)
    scope_boundary: dict[str, list[str]] = Field(default_factory=dict)
    method_obligations: list[str] = Field(default_factory=list)


class WorkPackageCoverageSummary(BaseModel):
    """Coverage summary for work-package planning."""
    total_units: int = 0
    covered_units: int = 0
    uncovered_unit_ids: list[str] = Field(default_factory=list)


class WorkPackagePlanningOutput(BaseModel):
    """Structured work-package planning output."""
    work_packages: list[WorkPackageItem] = Field(default_factory=list)
    coverage_summary: WorkPackageCoverageSummary = Field(default_factory=WorkPackageCoverageSummary)
    planning_notes: list[str] = Field(default_factory=list)


class EvidenceLinkOutput(BaseModel):
    """One grounded evidence link for a work package."""
    unit_id: str
    ref_id: str
    file_path: str = ""
    snippet_preview: str = ""
    why_relevant: str = ""
    confidence: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)


class EvidenceBundleOutput(BaseModel):
    """Compact evidence bundle aligned with reproduction evidence grounding."""
    work_package_id: str
    focus: str
    owned_unit_ids: list[str] = Field(default_factory=list)
    evidence_links: list[EvidenceLinkOutput] = Field(default_factory=list)
    context_summary: list[str] = Field(default_factory=list)
    grounding_status: str = "ungrounded"


class PipelinePlanNode(BaseModel):
    """Flat plan node used by the new planning pipeline."""
    node_id: str
    parent_node_id: str = ""
    name: str
    level: Literal["experiment", "module", "function"]
    description: str = ""
    hypothesis: str = ""
    decision_value: str = ""
    stop_rule_or_pruning_rationale: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    ref_id: str = ""
    reusable_module: str = ""
    depends_on: list[str] = Field(default_factory=list)
    traceable: bool = False
    code_snippet: str = ""
    insight: str = ""


class PipelineCoverageSummary(BaseModel):
    """Coverage summary for the new flat plan."""
    total_requirements: int = 0
    covered_requirements: int = 0
    uncovered_requirement_ids: list[str] = Field(default_factory=list)


class PipelinePlanOutput(BaseModel):
    """Stage-3 planning output."""
    plan_nodes: list[PipelinePlanNode] = Field(default_factory=list)
    coverage_summary: PipelineCoverageSummary = Field(default_factory=PipelineCoverageSummary)


class ArchitectureFileBlueprint(BaseModel):
    """One target file blueprint."""
    path: str
    purpose: str = ""
    kind: Literal["source", "test", "config", "doc", "script"] = "source"
    related_node_ids: list[str] = Field(default_factory=list)
    based_on_references: list[str] = Field(default_factory=list)
    implementation_strategy: Literal["new", "adapted", "reused"] = "adapted"


class ArchitectureDependency(BaseModel):
    """One file-level dependency edge."""
    source_path: str
    target_path: str
    dependency_type: str = "imports"


class ArchitectureTaskPackageResponsibility(BaseModel):
    """Task-model view for one work package during architecture planning."""
    work_package_id: str
    responsibilities: list[str] = Field(default_factory=list)
    method_obligations: list[str] = Field(default_factory=list)
    interface_surfaces: list[str] = Field(default_factory=list)
    owned_unit_ids: list[str] = Field(default_factory=list)


class ArchitectureEvidenceMapping(BaseModel):
    """Task-model evidence-to-module hint used during architecture synthesis."""
    work_package_id: str
    influenced_paths: list[str] = Field(default_factory=list)
    supporting_references: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArchitectureTaskModelOutput(BaseModel):
    """Intermediate task-model summary used by architecture planning."""
    execution_entry: str = ""
    runnable_flow: list[str] = Field(default_factory=list)
    method_spine: list[str] = Field(default_factory=list)
    package_responsibilities: list[ArchitectureTaskPackageResponsibility] = Field(default_factory=list)
    interface_closure: list[str] = Field(default_factory=list)
    evidence_to_module_mapping: list[ArchitectureEvidenceMapping] = Field(default_factory=list)
    reproducibility_readiness: list[str] = Field(default_factory=list)


class ArchitectureOutput(BaseModel):
    """Stage-4 architecture output."""
    target_stack: list[str] = Field(default_factory=list)
    target_file_tree: list[str] = Field(default_factory=list)
    file_blueprints: list[ArchitectureFileBlueprint] = Field(default_factory=list)
    dependency_graph: list[ArchitectureDependency] = Field(default_factory=list)
    stable_interfaces: list[str] = Field(default_factory=list)
    execution_entrypoints: list[str] = Field(default_factory=list)
    config_surfaces: list[str] = Field(default_factory=list)
    package_layout: dict[str, list[str]] = Field(default_factory=dict)
    dependency_rules: list[str] = Field(default_factory=list)
    protocol_stages: list[str] = Field(default_factory=list)
    result_targets: list[str] = Field(default_factory=list)
    architecture_reference_ids: list[str] = Field(default_factory=list)
    unresolved_review_failures: list[str] = Field(default_factory=list)
    rationale: str = ""


class TaskItem(BaseModel):
    """One executable task item derived from architecture."""
    task_id: str
    file_path: str
    work_package_id: str = ""
    purpose: str = ""
    hypothesis: str = ""
    decision_value: str = ""
    stop_rule_or_pruning_rationale: str = ""
    related_node_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    blocking_dependencies: list[str] = Field(default_factory=list)
    requires_stable_dependencies: bool = True
    interface_contract: list[str] = Field(default_factory=list)
    implementation_surfaces: list[str] = Field(default_factory=list)
    method_obligations: list[str] = Field(default_factory=list)
    defines_symbols: list[str] = Field(default_factory=list)
    calls_symbols: list[str] = Field(default_factory=list)
    writes_artifacts: list[str] = Field(default_factory=list)
    reads_artifacts: list[str] = Field(default_factory=list)
    allowed_scope: dict[str, list[str]] = Field(default_factory=dict)
    scope_boundary: dict[str, list[str]] = Field(default_factory=dict)
    review_points: list[str] = Field(default_factory=list)
    status: Literal["pending", "generated", "reviewed"] = "pending"


class ReferenceSnippetCandidate(BaseModel):
    """Reusable reference snippet bound to a stable ref id and reusable module."""
    ref_id: str
    repository_url: str = ""
    reusable_module: str = ""
    code_snippet: str = ""
    insight: str = ""
    supported_task_ids: list[str] = Field(default_factory=list)
    supported_file_paths: list[str] = Field(default_factory=list)


class GenerationTaskInput(BaseModel):
    """Prepared generation context for one ordered task."""
    task_id: str
    file_path: str
    work_package_id: str = ""
    dependency_files: list[str] = Field(default_factory=list)
    related_node_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    snippet_candidates: list[ReferenceSnippetCandidate] = Field(default_factory=list)
    interface_contract: list[str] = Field(default_factory=list)
    implementation_surfaces: list[str] = Field(default_factory=list)
    method_obligations: list[str] = Field(default_factory=list)
    defines_symbols: list[str] = Field(default_factory=list)
    calls_symbols: list[str] = Field(default_factory=list)
    writes_artifacts: list[str] = Field(default_factory=list)
    reads_artifacts: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    decision_value: str = ""
    generation_prompt: str = ""
    context_sources: list[str] = Field(default_factory=list)
    allowed_scope: dict[str, list[str]] = Field(default_factory=dict)
    scope_boundary: dict[str, list[str]] = Field(default_factory=dict)
    review_points: list[str] = Field(default_factory=list)
    paper_claim_inventory: dict[str, list[str]] = Field(default_factory=dict)
    paper_claim_closure_items: list[dict[str, str]] = Field(default_factory=list)
    paper_claim_closure_rules: list[str] = Field(default_factory=list)
    paper_evidence_contract: dict[str, Any] = Field(default_factory=dict)
    prepare_quality_gate_summary: dict[str, Any] = Field(default_factory=dict)
    generation_context: dict[str, Any] = Field(default_factory=dict)


class GenerationManifest(BaseModel):
    """Prepared manifest consumed by the generation node."""
    ordered_tasks: list[str] = Field(default_factory=list)
    task_inputs: list[GenerationTaskInput] = Field(default_factory=list)
    review_points: list[str] = Field(default_factory=list)
    tasks: list[TaskItem] = Field(default_factory=list)
    edges: list[ArchitectureDependency] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)


class PlanDecision(BaseModel):
    """Experiment plan decision."""
    plan: str
    steps: list[str] = Field(default_factory=list)
    project_plan: dict[str, Any] = Field(default_factory=dict)


class CodeGenResult(BaseModel):
    """Code generation result."""
    code: str
    language: str = "python"
    explanation: str = ""


class FileSpec(BaseModel):
    """One generated file in the planned project layout."""
    path: str
    purpose: str
    dependencies: list[str] = Field(default_factory=list)
    required: bool = True


class ArtifactContract(BaseModel):
    """Required artifact contract for generated experiment projects."""
    metrics_path: str = "results/metrics.json"
    required_files: list[str] = Field(default_factory=lambda: ["results/metrics.json"])
    optional_files: list[str] = Field(default_factory=list)


class ProjectPlan(BaseModel):
    """Structured project-level implementation plan."""
    project_type: str = "single_experiment_project"
    summary: str = ""
    entrypoints: dict[str, str] = Field(default_factory=dict)
    runtime_contract: dict[str, Any] = Field(default_factory=dict)
    file_specs: list[FileSpec] = Field(default_factory=list)
    artifact_contract: ArtifactContract = Field(default_factory=ArtifactContract)


class RepoFilePlan(BaseModel):
    """Unified file-level plan used by generation, validation, and repair."""
    target_file: str
    task_id: str = ""
    work_package_id: str = ""
    purpose: str = ""
    hypothesis: str = ""
    decision_value: str = ""
    stop_rule_or_pruning_rationale: str = ""
    related_node_ids: list[str] = Field(default_factory=list)
    owned_units: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    blocking_dependencies: list[str] = Field(default_factory=list)
    requires_stable_dependencies: bool = True
    interface_contract: list[str] = Field(default_factory=list)
    implementation_surfaces: list[str] = Field(default_factory=list)
    method_obligations: list[str] = Field(default_factory=list)
    context_sources: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    defines_symbols: list[str] = Field(default_factory=list)
    calls_symbols: list[str] = Field(default_factory=list)
    writes_artifacts: list[str] = Field(default_factory=list)
    reads_artifacts: list[str] = Field(default_factory=list)
    allowed_scope: dict[str, list[str]] = Field(default_factory=dict)
    scope_boundary: dict[str, list[str]] = Field(default_factory=dict)
    generation_prompt: str = ""
    validation_hooks: list[str] = Field(default_factory=list)
    review_points: list[str] = Field(default_factory=list)


class PackageFilePlanningOutput(BaseModel):
    """Stage-4.5 package-scoped file planning output."""
    file_plans: list[RepoFilePlan] = Field(default_factory=list)
    planning_notes: list[str] = Field(default_factory=list)
    unresolved_review_failures: list[str] = Field(default_factory=list)


class RepoCanonicalRoute(BaseModel):
    """Repo-level canonical execution closure derived for generation and validation."""
    summary: str = ""
    entry_surface: str = ""
    required_inputs: list[str] = Field(default_factory=list)
    stage_sequence: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    example_invocation: str = ""


class RepoStagePublicSurface(BaseModel):
    """One public stage surface exposed by the generated repository plan."""
    stage_name: str
    path: str
    surface_kind: str = ""
    purpose: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class RepoArtifactContract(BaseModel):
    """One artifact contract entry connecting outputs to producer surfaces."""
    artifact_key: str
    relative_path: str
    owner_work_package: str = ""
    producer_surface: str = ""
    stage_name: str = ""
    description: str = ""
    required: bool = True


class RepoPlan(BaseModel):
    """Unified repo-level execution plan aligned with reproduction repo_plan semantics."""
    package_name: str = "generated_experiment"
    summary: str = ""
    architecture: ArchitectureOutput = Field(default_factory=ArchitectureOutput)
    work_packages: list[WorkPackageItem] = Field(default_factory=list)
    evidence_bundles: list[EvidenceBundleOutput] = Field(default_factory=list)
    files: list[RepoFilePlan] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    canonical_route: RepoCanonicalRoute = Field(default_factory=RepoCanonicalRoute)
    stage_public_surfaces: list[RepoStagePublicSurface] = Field(default_factory=list)
    artifact_contract: list[RepoArtifactContract] = Field(default_factory=list)
    structure_decisions: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    global_contract: dict[str, Any] = Field(default_factory=dict)
    topic_profile: dict[str, Any] = Field(default_factory=dict)
    canonical_ir: dict[str, Any] = Field(default_factory=dict)
    canonical_ir_validation: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Code execution result."""
    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_summary: dict[str, Any] = Field(default_factory=dict)


class EvaluationDecision(BaseModel):
    """Evaluation decision."""
    action: Literal["ITERATE", "COMPLETE"]
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class PreflightResult(BaseModel):
    """Static preflight result for a generated project."""
    status: Literal["passed", "warnings", "failed"] = "warnings"
    checks: list[dict[str, Any]] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    warning_messages: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class RepairTicket(BaseModel):
    """Structured repair ticket distilled from deterministic validation."""
    failure_type: str = ""
    reason: str = ""
    trigger_signals: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    allowed_changes: list[str] = Field(default_factory=list)
    required_fix_targets: list[str] = Field(default_factory=list)
    next_fix_scope: list[str] = Field(default_factory=list)
    forbidden_changes: list[str] = Field(default_factory=list)


class IterationCheckpoint(BaseModel):
    """Persisted iteration checkpoint."""
    current_iteration: int = 0
    best_round: int | None = None
    best_metrics: dict[str, float] = Field(default_factory=dict)
    termination_reason: str = ""
    latest_stage: str = ""
    latest_status: str = ""
    generated_files: list[str] = Field(default_factory=list)


class GenerationCheckpoint(BaseModel):
    """One generation or repair checkpoint entry."""
    checkpoint_id: str
    stage: str
    focus_id: str
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RuntimeProbe(BaseModel):
    """Runtime/environment probe summary for validation and manifesting."""
    execution_mode: str = "local"
    python_executable: str = ""
    gpu_available: bool = False
    gpu_probe_command: str = ""
    available_commands: list[str] = Field(default_factory=list)
    missing_commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    """One validation gate check."""
    name: str
    category: str
    passed: bool
    details: str
    affected_units: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    affected_work_packages: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Structured validation report aligned with the reproduction pipeline."""
    passed: bool = False
    static_status: str = "unknown"
    static_contract_status: str = "unknown"
    smoke_status: str = "skipped"
    dynamic_status: str = "skipped"
    overall_status: str = "unknown"
    quality_level: str = "scaffold_only"
    runtime_probe: RuntimeProbe = Field(default_factory=RuntimeProbe)
    artifact_checks: list[ValidationCheck] = Field(default_factory=list)
    implementation_checks: list[ValidationCheck] = Field(default_factory=list)
    semantic_checks: list[ValidationCheck] = Field(default_factory=list)
    trace_checks: list[ValidationCheck] = Field(default_factory=list)
    integration_checks: list[ValidationCheck] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    repair_recommendations: list[str] = Field(default_factory=list)
    planning_failure_layer: str = ""
    semantic_validation_report: dict[str, Any] = Field(default_factory=dict)


class CanonicalRequirement(BaseModel):
    """Canonical requirement node projected from boundary requirements and units."""

    requirement_id: str
    title: str = ""
    category: str = "experiment"
    description: str = ""
    source_unit_ids: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class CanonicalWorkPackage(BaseModel):
    """Canonical work-package node with closed requirement ownership."""

    work_package_id: str
    goal: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    source_unit_ids: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class CanonicalContractStage(BaseModel):
    """Canonical contract stage or result-owner layer."""

    stage_id: str
    label: str = ""
    owner_work_package_ids: list[str] = Field(default_factory=list)
    result_target_ids: list[str] = Field(default_factory=list)


class CanonicalFileNode(BaseModel):
    """Canonical file node registered during planning."""

    file_id: str
    canonical_path: str
    owner_work_package_id: str = ""
    contract_stage_ids: list[str] = Field(default_factory=list)
    related_requirement_ids: list[str] = Field(default_factory=list)
    related_plan_node_ids: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)


class CanonicalSurfaceNode(BaseModel):
    """Canonical public or validation-relevant surface."""

    surface_id: str
    surface_kind: Literal["entrypoint", "config", "stable_interface", "artifact", "producer"] = "artifact"
    canonical_path: str
    owner_file_id: str = ""
    owner_work_package_id: str = ""
    validator_expectation_ids: list[str] = Field(default_factory=list)


class CanonicalIREdge(BaseModel):
    """Typed relationship between canonical IR nodes."""

    edge_id: str
    edge_type: str
    source_id: str
    target_id: str


class SemanticAssertion(BaseModel):
    """Requirement-level semantic assertion compiled for downstream validation."""

    assertion_id: str
    requirement_id: str = ""
    assertion_type: str = "static_presence"
    statement: str = ""
    status: Literal["passed", "failed", "unknown", "not_checkable"] = "unknown"


class EvidenceContract(BaseModel):
    """Where validator or reviewer should find evidence for an assertion."""

    evidence_contract_id: str
    assertion_id: str
    evidence_kind: str = "file_presence"
    canonical_paths: list[str] = Field(default_factory=list)
    owner_work_package_ids: list[str] = Field(default_factory=list)


class ValidatorExpectation(BaseModel):
    """Validator-facing expectation compiled from semantic assertions."""

    expectation_id: str
    assertion_id: str
    expectation_kind: str = "deterministic_check"
    evidence_contract_id: str = ""
    pass_condition: str = ""
    status: Literal["passed", "failed", "unknown", "not_checkable"] = "unknown"


class CanonicalIROutput(BaseModel):
    """Shadow canonical IR emitted from reproagent plan stage."""

    mode: Literal["shadow"] = "shadow"
    requirements: list[CanonicalRequirement] = Field(default_factory=list)
    work_packages: list[CanonicalWorkPackage] = Field(default_factory=list)
    contract_stages: list[CanonicalContractStage] = Field(default_factory=list)
    file_nodes: list[CanonicalFileNode] = Field(default_factory=list)
    surface_nodes: list[CanonicalSurfaceNode] = Field(default_factory=list)
    edges: list[CanonicalIREdge] = Field(default_factory=list)
    semantic_assertions: list[SemanticAssertion] = Field(default_factory=list)
    evidence_contracts: list[EvidenceContract] = Field(default_factory=list)
    validator_expectations: list[ValidatorExpectation] = Field(default_factory=list)
    validation_index: dict[str, Any] = Field(default_factory=dict)


class CanonicalIRMismatch(BaseModel):
    """One planning mismatch surfaced by shadow canonical IR validation."""

    category: str
    message: str
    severity: Literal["warning", "retry_generate", "degraded_handoff"] = "warning"
    related_ids: list[str] = Field(default_factory=list)


class CanonicalIRValidationOutput(BaseModel):
    """Validation report for shadow canonical IR closure."""

    mode: Literal["shadow"] = "shadow"
    passed: bool = True
    planning_failure_layer: str = ""
    mismatches: list[CanonicalIRMismatch] = Field(default_factory=list)
    mismatch_summary: dict[str, int] = Field(default_factory=dict)
    gate_actions: list[str] = Field(default_factory=list)
    semantic_validation_report: dict[str, Any] = Field(default_factory=dict)


class RepairAction(BaseModel):
    """One repair-stage action summary."""
    round_id: int
    action_type: str
    reason: str
    touched_work_packages: list[str] = Field(default_factory=list)
    touched_files: list[str] = Field(default_factory=list)
    outcome: str = ""


class RepairLog(BaseModel):
    """Structured repair-stage log."""
    converged: bool = False
    rounds_attempted: int = 0
    actions: list[RepairAction] = Field(default_factory=list)


class RepairPlanDraft(BaseModel):
    """Repo/problem-driven draft plan for the repair loop."""
    summary: str = ""
    problem_list: list[str] = Field(default_factory=list)
    failure_focus: list[str] = Field(default_factory=list)
    semantic_guardrails: list[str] = Field(default_factory=list)
    semantic_must_keep: list[str] = Field(default_factory=list)
    runtime_guardrails: list[str] = Field(default_factory=list)
    recommended_surfaces: list[str] = Field(default_factory=list)
    preferred_files: list[str] = Field(default_factory=list)
    preferred_work_packages: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    repair_guidance: list[str] = Field(default_factory=list)
    generation_guidance: list[str] = Field(default_factory=list)
    evaluation_guidance: list[str] = Field(default_factory=list)
    review_focus: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    round_budget: int = 30


class RepairPlanReview(BaseModel):
    """Evaluation-side review of the repo-wide repair plan draft."""
    approved: bool = True
    summary: str = ""
    semantic_risks: list[str] = Field(default_factory=list)
    runtime_risks: list[str] = Field(default_factory=list)
    accepted_surfaces: list[str] = Field(default_factory=list)
    validated_files: list[str] = Field(default_factory=list)
    validated_work_packages: list[str] = Field(default_factory=list)
    required_review_points: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    round_budget: int = 30


class RepairPlan(BaseModel):
    """Final repo/problem-driven repair plan merged from plan and eval agents."""
    summary: str = ""
    problem_list: list[str] = Field(default_factory=list)
    semantic_guardrails: list[str] = Field(default_factory=list)
    runtime_guardrails: list[str] = Field(default_factory=list)
    recommended_surfaces: list[str] = Field(default_factory=list)
    selected_files: list[str] = Field(default_factory=list)
    selected_work_packages: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    repair_guidance: list[str] = Field(default_factory=list)
    review_points: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    round_budget: int = 30


class RepairEvalFinding(BaseModel):
    """Repo-level repair finding used to drive plan and regeneration."""
    finding_id: str
    category: Literal["semantic", "runtime", "integration", "artifact", "anti_shortcut"]
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    summary: str
    evidence: list[str] = Field(default_factory=list)
    affected_surfaces: list[str] = Field(default_factory=list)
    suggested_focus: list[str] = Field(default_factory=list)
    assertion_ids: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    related_symbols: list[str] = Field(default_factory=list)
    failure_layer: str = ""
    fix_hint: str = ""


class RepairEvalReport(BaseModel):
    """Repo-level repair evaluation report."""
    summary: str = ""
    semantic_status: Literal["aligned", "drift_risk", "misaligned"] = "drift_risk"
    runtime_status: Literal["closed", "partial", "broken"] = "partial"
    anti_shortcut_status: Literal["clean", "suspect", "violated"] = "clean"
    findings: list[RepairEvalFinding] = Field(default_factory=list)
    must_keep: list[str] = Field(default_factory=list)
    repair_focus: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)


class RequirementAnchor(BaseModel):
    """Frozen semantic anchor reused across repair rounds."""
    source: str = "plan_boundary_requirements"
    summary: str = ""
    goal: str = ""
    semantic_invariants: list[str] = Field(default_factory=list)
    runtime_invariants: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    acceptance_signals: list[str] = Field(default_factory=list)


class StageMetric(BaseModel):
    """Per-stage runtime accounting."""
    stage_name: str
    elapsed_seconds: float
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    actual_total_tokens: int = 0
    agent_calls: int = 0
    agent_calls_with_usage: int = 0
    usage_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StageAttemptRecord(BaseModel):
    """Append-only attempt record for one workflow stage execution."""
    attempt_id: str
    stage_name: str
    status: str
    started_at: str = ""
    completed_at: str = ""
    failed_at: str = ""
    recovered_at: str = ""
    resume_source: str = ""
    recovery_action: str = ""
    error_type: str = ""
    error_message: str = ""
    output_paths: list[str] = Field(default_factory=list)
    input_hash: str = ""
    pipeline_signature: str = ""


class QualityStatus(BaseModel):
    """Separated run/quality status for reproagent consumers."""
    run_status: Literal["pending", "running", "finalizing", "completed"] = "pending"
    quality_status: Literal[
        "validated",
        "repaired",
        "degraded_contract",
        "runtime_risk",
        "unverified",
        "failed_with_evidence",
    ] = "unverified"
    handoff_ready: bool = False
    terminal_outcome: str = ""
    terminal_outcome_reason: str = ""
    validation_passed: bool = False
    validation_quality_level: str = "scaffold_only"
    failure_categories: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    next_recommended_action: str = ""


class ExperimentImplementationHandoff(BaseModel):
    """Canonical PaperBench Repro repository handoff contract."""
    schema_version: str = "1.0"
    producer: str = "reproagent"
    run_id: str = ""
    target: str = ""
    intent_contract: dict[str, Any] = Field(default_factory=dict)
    repo_root: str = ""
    project_root: str = ""
    entrypoint: str = ""
    install_command: str = ""
    smoke_command: str = ""
    metric_contract: dict[str, Any] = Field(default_factory=dict)
    artifact_contract: Any = Field(default_factory=dict)
    editable_scope: list[str] = Field(default_factory=list)
    editable_paths: list[str] = Field(default_factory=list)
    protected_scope: list[str] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)
    validation_evidence: dict[str, Any] = Field(default_factory=dict)
    known_risks: list[Any] = Field(default_factory=list)
    repair_history: dict[str, Any] = Field(default_factory=dict)
    handoff_ready: bool = False
    quality_status: dict[str, Any] = Field(default_factory=dict)
    file_provenance: list[dict[str, Any]] = Field(default_factory=list)
    repo_handoff: dict[str, Any] = Field(default_factory=dict)
    validated_repo_handoff: dict[str, Any] = Field(default_factory=dict)


class UsageSummary(BaseModel):
    """Workflow-level usage summary."""
    estimated_total_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    actual_total_tokens: int = 0
    agent_calls: int = 0
    agent_calls_with_usage: int = 0
    wall_clock_seconds: float = 0.0
    stage_breakdown: list[StageMetric] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    """Lightweight benchmark/coverage report."""
    benchmark_name: str = ""
    case_id: str = ""
    task_id: str = ""
    expected_artifacts: list[str] = Field(default_factory=list)
    artifact_match_ratio: float = 0.0
    rubric_requirements: list[str] = Field(default_factory=list)
    rubric_match_ratio: float = 0.0
    notes: list[str] = Field(default_factory=list)


class StageRunSummary(BaseModel):
    """Manifest view of one stage run."""
    stage_name: str
    status: str = "pending"
    resume_source: str = ""
    skipped: bool = False
    error_type: str = ""
    output_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PhaseRunSummary(BaseModel):
    """Manifest view of one top-level workflow phase."""
    phase_name: str
    status: str = "pending"
    stage_names: list[str] = Field(default_factory=list)
    output_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """High-level run summary derived from stage outputs."""
    completed_stage_count: int = 0
    failed_stage_count: int = 0
    invalidated_stage_count: int = 0
    resumed_stage_count: int = 0
    completed_phase_count: int = 0
    failed_phase_count: int = 0
    primary_topic: str = ""
    coverage_policy: str = ""
    experiment_traits: list[str] = Field(default_factory=list)
    work_package_count: int = 0
    grounded_work_package_count: int = 0
    result_target_count: int = 0
    validation_passed: bool = False
    validation_static_contract_status: str = "unknown"
    validation_smoke_status: str = "skipped"
    validation_overall_status: str = "unknown"
    validation_quality_level: str = "scaffold_only"
    validation_failure_categories: list[str] = Field(default_factory=list)
    validation_blocked_reasons: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Top-level run manifest aligned with the reproduction system."""
    run_id: str
    target: str
    run_dir: str
    stages: list[str] = Field(default_factory=list)
    stage_order: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    phase_order: list[str] = Field(default_factory=list)
    runtime_overrides: dict[str, Any] = Field(default_factory=dict)
    artifact_index: dict[str, str] = Field(default_factory=dict)
    stage_review_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    stage_status_summary: list[StageRunSummary] = Field(default_factory=list)
    phase_status_summary: list[PhaseRunSummary] = Field(default_factory=list)
    run_summary: RunSummary = Field(default_factory=RunSummary)


class GenerateStageOutput(BaseModel):
    """Structured output contract from the generate stage."""
    project_plan: dict[str, Any] = Field(default_factory=dict)
    generation_manifest: dict[str, Any] = Field(default_factory=dict)
    topic_profile: dict[str, Any] = Field(default_factory=dict)
    global_contract: dict[str, Any] = Field(default_factory=dict)
    generated_files: list[str] = Field(default_factory=list)
    file_count: int = 0
    iteration_checkpoint: dict[str, Any] = Field(default_factory=dict)
    experiment_status: str = "pending"
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    quality_level: str = "scaffold_only"
    iteration_state: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: str = ""


class PaperBenchReproState(BaseModel):
    """Experiment generation state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: PaperBenchReproInput
    upstream_intent: Optional[UpstreamIntentContract] = None
    plan: str = ""
    project_plan: ProjectPlan = Field(default_factory=ProjectPlan)
    repo_plan: Optional[RepoPlan] = None
    normalized_input: Optional[InputNormalizationOutput] = None
    paper_chunks: list[PaperChunk] = Field(default_factory=list)
    unit_extraction: Optional[UnitExtractionOutput] = None
    boundary_requirements: Optional[BoundaryRequirementsOutput] = None
    topic_profile: Optional[TopicProfileOutput] = None
    reference_repo_surveys: list[PreparedReferenceRepositorySurvey] = Field(default_factory=list)
    work_package_planning: Optional[WorkPackagePlanningOutput] = None
    evidence_bundles: list[EvidenceBundleOutput] = Field(default_factory=list)
    evidence_graph: list[EvidenceLinkOutput] = Field(default_factory=list)
    reference_selection: Optional[ReferenceSelectionOutput] = None
    pipeline_plan: Optional[PipelinePlanOutput] = None
    global_contract: Optional[GlobalContractOutput] = None
    architecture: Optional[ArchitectureOutput] = None
    package_file_planning_output: Optional[PackageFilePlanningOutput] = None
    canonical_ir: Optional[CanonicalIROutput] = None
    canonical_ir_validation: Optional[CanonicalIRValidationOutput] = None
    planning_failure_layer: str = ""
    generation_manifest: Optional[GenerationManifest] = None
    generated_files: list[str] = Field(default_factory=list)
    project_root: str = ""
    project_manifest: dict[str, Any] = Field(default_factory=dict)
    code: str = ""
    execution_result: Optional[ExecutionResult] = None
    preflight_result: Optional[PreflightResult] = None
    experiment_results: dict[str, Any] = Field(default_factory=dict)
    evaluation: Optional[EvaluationDecision] = None
    generate_stage_output: Optional[GenerateStageOutput] = None
    generation_checkpoints: list[GenerationCheckpoint] = Field(default_factory=list)
    usage_summary: Optional[UsageSummary] = None
    quality_status: Optional[QualityStatus] = None
    run_manifest: Optional[RunManifest] = None
    runtime_probe: Optional[RuntimeProbe] = None
    validation_report: Optional[ValidationReport] = None
    benchmark_report: Optional[BenchmarkReport] = None
    repair_ticket: Optional[RepairTicket] = None
    requirement_anchor: Optional[RequirementAnchor] = None
    repair_eval_report: Optional[RepairEvalReport] = None
    repair_plan: Optional[RepairPlan] = None
    repair_log: Optional[RepairLog] = None
    checkpoint_path: str = ""
    iteration_count: int = 0
    temp_data: dict[str, Any] = Field(default_factory=dict)

    run_id: str = ""
    execution_history: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)

    status: Literal[
        "pending",
        "running",
        "failed",
        "completed",
        "completed_with_degraded_contract",
        "completed_with_runtime_risk",
        "completed_unverified",
    ] = "pending"
    terminal_outcome: Literal[
        "failed",
        "completed",
        "completed_with_degraded_contract",
        "completed_with_runtime_risk",
        "completed_unverified",
    ] = "completed"
    terminal_outcome_reason: str = ""
    current_node: str = ""
    failed_node: str = ""
    error_message: str = ""
