"""Configuration for reproagent workflow."""
import os
from pathlib import Path
from dataclasses import dataclass, field


def _load_env():
    """Load environment variables from .env file."""
    module_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / ".env",
        module_root / ".env",
        *[parent / ".env" for parent in module_root.parents[:3]],
    ]
    seen: set[Path] = set()
    for env_file in candidates:
        env_file = env_file.resolve()
        if env_file in seen or not env_file.exists():
            continue
        seen.add(env_file)
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip("'\"")
                os.environ.setdefault(key.strip(), value)

_load_env()


def _provider_slot() -> str:
    slot = os.getenv("PAPERBENCH_REPRO_PROVIDER_SLOT", "").strip()
    if not slot:
        return ""
    return slot if slot.startswith("_") else f"_{slot}"


def _slot_env(name: str) -> str:
    slot = _provider_slot()
    if not slot:
        return ""
    return os.getenv(f"{name}{slot}", "").strip()


def _legacy_env_names(name: str) -> list[str]:
    if name.startswith("PAPERBENCH_REPRO_"):
        return ["EXP_GEN_" + name.removeprefix("PAPERBENCH_REPRO_")]
    if name.startswith("PAPERAGENT_PAPERBENCH_REPRO_"):
        return ["PAPERAGENT_EXP_GEN_" + name.removeprefix("PAPERAGENT_PAPERBENCH_REPRO_")]
    return []


def _reproagent_env(name: str, *, default: str = "") -> str:
    for candidate in [name, *_legacy_env_names(name)]:
        value = os.getenv(candidate, "").strip()
        if value:
            return value
    return default


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_default = "1" if default else "0"
    return _reproagent_env(name, default=raw_default).lower() in {"1", "true", "yes"}


def semantic_anchor_disabled() -> bool:
    """Return whether paper-derived semantic anchors should be withheld from generation/repair."""
    return _env_flag("PAPERBENCH_REPRO_DISABLE_SEMANTIC_ANCHOR", default=False)


def _reproagent_codex_model_provider() -> str | None:
    provider = _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_MODEL_PROVIDER")
    if provider:
        return provider
    if _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_BASE_URL"):
        return "paperagent_reproagent"
    return _reproagent_env("PAPERBENCH_REPRO_CODEX_MODEL_PROVIDER") or None


@dataclass
class NodeLLMConfig:
    """LLM configuration for a single node."""
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = None


@dataclass
class WorkflowRuntimeConfig:
    """Runtime configuration for reproagent workflow."""
    default_max_iterations: int = 30
    max_allowed_iterations: int = 30
    execution_timeout: int = 86400  # 24 hours for full training
    soft_timeout: int = 43200  # 12 hours soft warning
    preflight_timeout: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PREFLIGHT_TIMEOUT", default="60")))
    runtime_smoke_timeout: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_DRY_RUN_TIMEOUT", default="600")))
    docker_validate_timeout: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_QUICK_EVAL_TIMEOUT", default="1800")))
    repair_context_window_tokens: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_CONTEXT_WINDOW_TOKENS", default="32000")))
    repair_context_threshold: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_REPAIR_CONTEXT_THRESHOLD", default="0.65")))
    max_in_iteration_retries: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_MAX_IN_ITERATION_RETRIES", default="8")))
    max_stage_fix_rounds: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_MAX_STAGE_FIX_ROUNDS", default="3")))
    task_review_max_attempts: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_TASK_REVIEW_MAX_ATTEMPTS", default="3")))
    disable_task_review: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DISABLE_TASK_REVIEW", default="0").lower() in {"1", "true", "yes"})
    metrics_contract_version: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_METRICS_CONTRACT_VERSION", default="v1"))
    required_metrics_keys: list[str] = field(default_factory=lambda: ["main_results", "ablation_results", "training_log"])
    primary_metric: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_PRIMARY_METRIC", default="score"))
    higher_is_better: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_HIGHER_IS_BETTER", default="1") not in {"0", "false", "False"})
    improvement_threshold: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_IMPROVEMENT_THRESHOLD", default="0.0")))
    plateau_patience: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PLATEAU_PATIENCE", default="2")))
    memory_max_active_lessons: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_MEMORY_MAX_ACTIVE_LESSONS", default="30")))
    memory_max_recent_mistakes: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_MEMORY_MAX_RECENT_MISTAKES", default="12")))
    memory_compact_threshold_chars: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_MEMORY_COMPACT_THRESHOLD_CHARS", default="12000")))
    planning_fanout_min_work_packages: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PLANNING_FANOUT_MIN_WORK_PACKAGES", default="4")))
    planning_fanout_prompt_chars: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PLANNING_FANOUT_PROMPT_CHARS", default="16000")))
    planning_fanout_reference_threshold: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PLANNING_FANOUT_REFERENCE_THRESHOLD", default="6")))
    planning_fanout_retry_threshold: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_PLANNING_FANOUT_RETRY_THRESHOLD", default="1")))
    work_package_planning_fanout_enabled: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_WORK_PACKAGE_FANOUT", default="0").lower() in {"1", "true", "yes"})
    deterministic_work_package_planning: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DETERMINISTIC_WORK_PACKAGES", default="0").lower() not in {"0", "false", "False"})
    deterministic_pipeline_plan: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DETERMINISTIC_PIPELINE_PLAN", default="0").lower() not in {"0", "false", "False"})
    allow_plan_fallback_continue: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_ALLOW_PLAN_FALLBACK_CONTINUE", default="0").lower() in {"1", "true", "yes"})
    deterministic_global_contract: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DETERMINISTIC_GLOBAL_CONTRACT", default="0").lower() not in {"0", "false", "False"})
    deterministic_architecture_synthesis: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DETERMINISTIC_ARCH_SYNTHESIS", default="0").lower() not in {"0", "false", "False"})
    deterministic_package_file_planning: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_DETERMINISTIC_PACKAGE_FILES", default="0").lower() not in {"0", "false", "False"})
    strict_generate_complete: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_STRICT_GENERATE_COMPLETE", default="1").lower() not in {"0", "false", "False"})
    allow_stage_artifact_recovery_after_failure: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_ALLOW_STAGE_ARTIFACT_RECOVERY", default="0").lower() in {"1", "true", "yes"})
    architecture_fanout_enabled: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_ARCHITECTURE_FANOUT", default="0").lower() in {"1", "true", "yes"})
    package_file_planning_fanout_enabled: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_PACKAGE_FILE_FANOUT", default="0").lower() in {"1", "true", "yes"})
    disable_semantic_anchor: bool = field(default_factory=semantic_anchor_disabled)
    repair_local_scope_max_files: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_LOCAL_SCOPE_MAX_FILES", default="8")))
    repair_local_scope_max_work_packages: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_LOCAL_SCOPE_MAX_WORK_PACKAGES", default="4")))
    repair_repo_wide_retry_threshold: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_REPO_WIDE_RETRY_THRESHOLD", default="1")))
    repair_round_budget_max: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_ROUND_BUDGET_MAX", default="5")))
    repair_score_feedback_enabled: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_REPAIR_SCORE_FEEDBACK", default="1").lower() not in {"0", "false", "False"})
    repair_score_feedback_max_items: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_SCORE_FEEDBACK_MAX_ITEMS", default="32")))
    repair_score_feedback_max_files: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_SCORE_FEEDBACK_MAX_FILES", default="8")))
    repair_regression_guard_enabled: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_REPAIR_REGRESSION_GUARD", default="1").lower() not in {"0", "false", "False"})
    repair_guard_allow_large_rewrites: bool = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_ALLOW_LARGE_REWRITES", default="0").lower() in {"1", "true", "yes"})
    repair_guard_large_file_min_lines: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_LARGE_FILE_MIN_LINES", default="120")))
    repair_guard_large_file_min_bytes: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_LARGE_FILE_MIN_BYTES", default="6000")))
    repair_guard_min_line_retention: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_MIN_LINE_RETENTION", default="0.45")))
    repair_guard_min_byte_retention: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_MIN_BYTE_RETENTION", default="0.45")))
    repair_guard_min_symbol_retention: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_MIN_SYMBOL_RETENTION", default="0.70")))
    repair_guard_min_anchor_retention: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_REPAIR_GUARD_MIN_ANCHOR_RETENTION", default="0.70")))
    acceptance_checks: list[str] = field(default_factory=lambda: [
        "python -m py_compile {code_file}",
    ])
    check_timeout: int = 60


@dataclass
class GitHubRepoConfig:
    """GitHub access configuration for reference repo search and clone."""
    api_key: str | None = field(
        default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_GITHUB_TOKEN")
        or os.getenv("GITHUB_TOKEN", "").strip()
        or None
    )
    api_base_url: str = field(
        default_factory=lambda: (
            _reproagent_env("PAPERBENCH_REPRO_GITHUB_API_BASE_URL", default="https://api.github.com")
            or "https://api.github.com"
        ).rstrip("/")
    )
    ssh_enabled: bool = field(
        default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_GITHUB_SSH_ENABLED").lower() in {"1", "true", "yes"}
    )
    ssh_host: str = field(
        default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_GITHUB_SSH_HOST", default="github.com") or "github.com"
    )
    ssh_key_path: str | None = field(
        default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_GITHUB_SSH_KEY_PATH") or None
    )
    ssh_command: str | None = field(
        default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_GITHUB_SSH_COMMAND") or None
    )
    clone_timeout_seconds: int = field(
        default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_GITHUB_CLONE_TIMEOUT", default="120"))
    )
    validate_explicit_references: bool = field(
        default_factory=lambda: _reproagent_env(
            "PAPERBENCH_REPRO_VALIDATE_EXPLICIT_GITHUB_REFS",
            default="0",
        ).lower() in {"1", "true", "yes"}
    )


@dataclass
class CodeGenerationConfig:
    """Code generation configuration for repo editing/generation."""
    mode: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_CODEGEN_MODE", default="llm_agent"))  # "llm_agent" | "claude_only" | "collaborative" | "codex_cli_collaborative"
    repair_mode: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_REPAIR_CODEGEN_MODE"))
    claude_cli_path: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CC_CLI_PATH") or "claude")
    codex_cli_path: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_CLI_PATH") or "/root/.claude/bin/codeagent-wrapper")
    claude_model: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CC_MODEL") or _reproagent_env("PAPERBENCH_REPRO_CLAUDE_MODEL", default="sonnet"))
    claude_effort: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CC_EFFORT") or _reproagent_env("PAPERBENCH_REPRO_CLAUDE_EFFORT", default="high"))
    codex_model: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_MODEL") or _reproagent_env("PAPERBENCH_REPRO_CODEX_MODEL", default="gpt-5"))
    codex_model_provider: str | None = field(default_factory=_reproagent_codex_model_provider)
    codex_base_url: str | None = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_BASE_URL") or _reproagent_env("PAPERBENCH_REPRO_CODEX_BASE_URL") or None)
    codex_reasoning_effort: str = field(default_factory=lambda: _reproagent_env("PAPERAGENT_PAPERBENCH_REPRO_CODEX_REASONING_EFFORT") or _reproagent_env("PAPERBENCH_REPRO_CODEX_REASONING_EFFORT", default="high"))
    max_collaboration_rounds: int = 3
    enable_fallback: bool = True


@dataclass
class StructuredStageConfig:
    """Backend policy for structured JSON stages across reproagent."""

    default_backend: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_BACKEND", default="llm"))
    repair_backend: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_REPAIR_STRUCTURED_STAGE_BACKEND", default="llm"))
    llm_model: str = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_MODEL", default="gpt-5.4"))
    llm_api_key: str | None = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_API_KEY") or None)
    llm_base_url: str | None = field(default_factory=lambda: _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_BASE_URL") or None)
    llm_temperature: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_TEMPERATURE", default="0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_MAX_TOKENS", default="12000")))
    llm_timeout_seconds: float = field(default_factory=lambda: float(_reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_TIMEOUT", default="300")))
    llm_max_retries: int = field(default_factory=lambda: int(_reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_MAX_RETRIES", default="1")))


NODE_CONFIGS = {
    "plan": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "input_normalization": NodeLLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=6000),
    "unit_extraction": NodeLLMConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=10000),
    "boundary_requirements": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "topic_profile": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "work_package_planning": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "select_references": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "pipeline_plan": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "global_contract": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "architecture": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "package_file_planning": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "repair_validation": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "repair_regeneration": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "file_generation": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "preflight_repair": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "runtime_smoke_repair": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "docker_validate_repair": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=12000),
    "reviewer_agent": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "evaluate": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=8000),
    "report": NodeLLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=4000),
}

WORKFLOW_CONFIG = WorkflowRuntimeConfig()
GITHUB_REPO_CONFIG = GitHubRepoConfig()
CODEGEN_CONFIG = CodeGenerationConfig()
STRUCTURED_STAGE_CONFIG = StructuredStageConfig()


def get_node_config(node_name: str) -> NodeLLMConfig:
    """Return the configured LLM settings for a node."""
    return NODE_CONFIGS.get(node_name, NodeLLMConfig())


def get_workflow_config() -> WorkflowRuntimeConfig:
    """Return the configured runtime limits for the workflow."""
    return WORKFLOW_CONFIG


def get_codegen_config() -> CodeGenerationConfig:
    """Return the code generation configuration."""
    return CODEGEN_CONFIG


def get_structured_stage_config() -> StructuredStageConfig:
    """Return the structured-stage backend policy."""
    return STRUCTURED_STAGE_CONFIG


def get_github_repo_config() -> GitHubRepoConfig:
    """Return the GitHub repository search/clone configuration."""
    return GITHUB_REPO_CONFIG


def build_github_repo_config(overrides: dict | None = None) -> dict[str, str | bool]:
    """Merge explicit runtime overrides onto the default GitHub repo config."""
    config = get_github_repo_config()
    merged: dict[str, str | bool] = {
        "api_key": config.api_key or "",
        "api_base_url": config.api_base_url,
        "ssh_enabled": config.ssh_enabled,
        "ssh_host": config.ssh_host,
        "ssh_key_path": config.ssh_key_path or "",
        "ssh_command": config.ssh_command or "",
        "clone_timeout_seconds": config.clone_timeout_seconds,
        "validate_explicit_references": config.validate_explicit_references,
    }
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def create_node_model(node_name: str):
    """Create LLM instance for a specific node."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain_openai is required. Install with: pip install langchain-openai")

    config = get_node_config(node_name)
    global_model = _reproagent_env("PAPERBENCH_REPRO_NODE_MODEL")
    global_api_key = _reproagent_env("PAPERBENCH_REPRO_NODE_API_KEY")
    global_base_url = _reproagent_env("PAPERBENCH_REPRO_NODE_BASE_URL")
    kwargs = {
        "model": global_model or config.model,
        "temperature": config.temperature,
        "api_key": (
            global_api_key
            or config.api_key
            or _slot_env("OPENAI_API_KEY")
            or _slot_env("DF_API_KEY")
            or os.getenv("DF_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
        "base_url": (
            global_base_url
            or config.base_url
            or _slot_env("OPENAI_BASE_URL")
            or _slot_env("DF_API_URL")
            or os.getenv("DF_API_URL")
            or os.getenv("OPENAI_BASE_URL")
        ),
    }
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    node_timeout = (
        _reproagent_env("PAPERBENCH_REPRO_NODE_TIMEOUT")
        or _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_TIMEOUT")
    )
    if node_timeout:
        kwargs["timeout"] = float(node_timeout)
    node_max_retries = (
        _reproagent_env("PAPERBENCH_REPRO_NODE_MAX_RETRIES")
        or _reproagent_env("PAPERBENCH_REPRO_STRUCTURED_STAGE_MAX_RETRIES")
    )
    if node_max_retries:
        kwargs["max_retries"] = int(node_max_retries)
    node_max_tokens = _reproagent_env("PAPERBENCH_REPRO_NODE_MAX_TOKENS")
    if node_max_tokens:
        kwargs["max_tokens"] = int(node_max_tokens)
    return ChatOpenAI(**kwargs)


def create_structured_stage_model(stage_name: str):
    """Create the LLM used by structured JSON planning/review stages."""
    del stage_name
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain_openai is required. Install with: pip install langchain-openai")

    config = get_structured_stage_config()
    kwargs = {
        "model": config.llm_model,
        "temperature": config.llm_temperature,
        "api_key": (
            config.llm_api_key
            or _slot_env("OPENAI_API_KEY")
            or _slot_env("DF_API_KEY")
            or os.getenv("DF_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
        "base_url": (
            config.llm_base_url
            or _slot_env("OPENAI_BASE_URL")
            or _slot_env("DF_API_URL")
            or os.getenv("DF_API_URL")
            or os.getenv("OPENAI_BASE_URL")
        ),
        "max_tokens": config.llm_max_tokens,
        "max_retries": config.llm_max_retries,
    }
    if config.llm_timeout_seconds > 0:
        kwargs["timeout"] = config.llm_timeout_seconds
    return ChatOpenAI(**kwargs)
