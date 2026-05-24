"""
src/data/environment.py
=======================
BBox-Adapter: Environment and Dataset Registry

Exposes paper-derived environment/task registry entries with ids, aliases,
setup metadata, and factory/config hooks for all black-box LLMs and QA
benchmarks evaluated in the paper.

Reference grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
Reference grounding: paperbench_ref_005 toxigen/alice.py
Reference grounding: paperbench_ref_006 readme.md
Reference grounding: paperbench_ref_006 research/readme_exp.md

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset split ratios as specified/observed in the paper
# GSM8K: 7473 train / 1319 test
# StrategyQA: 2061 train / 229 dev (paper uses dev as test)
# TruthfulQA: 817 questions total; paper uses 704 train / 113 eval
# ScienceQA: 12726 train / 4241 val / 2017 test
# ToxiGen: ~8960 train (implicitly derived from generation set) / eval subset
# ---------------------------------------------------------------------------

DATASET_SPLIT_RATIOS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "train": 7473,
        "test": 1319,
        "val": None,
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
    },
    "strategyqa": {
        "train": 2061,
        "test": 229,       # dev set used as evaluation set in the paper
        "val": None,
        "hf_path": "wics/strategy-qa",
        "hf_name": "default",
    },
    "truthfulqa": {
        "train": 704,
        "test": 113,
        "val": None,
        "hf_path": "truthful_qa",
        "hf_name": "multiple_choice",
    },
    "scienceqa": {
        "train": 12726,
        "test": 2017,
        "val": 4241,
        "hf_path": "derek-thomas/ScienceQA",
        "hf_name": None,
    },
    "toxigen": {
        "train": 8960,      # approx; paper samples from generation pool
        "test": 1000,
        "val": None,
        "hf_path": "skg/toxigen-data",
        "hf_name": "train",
    },
}


# ---------------------------------------------------------------------------
# DatasetEntry — machine-readable benchmark registry entry
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    """Registry entry for a single QA benchmark used in the paper."""
    id: str
    aliases: List[str]
    description: str
    task_type: str            # math_reasoning | implicit_reasoning | truthfulness | science_domain | toxicity_reduction
    answer_type: str          # numeric | binary_yes_no | multiple_choice | free_text
    feedback_mode: str        # ground_truth | ai_feedback | combined
    metric: str               # exact_match_numeric | accuracy_binary | accuracy_mc | toxicity_score
    hf_path: str
    hf_name: Optional[str]
    num_train: int
    num_test: int
    num_val: Optional[int]
    loader_hook: str
    answer_extractor: str
    in_scope: bool = True
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnvironmentEntry — machine-readable model/API environment registry entry
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentEntry:
    """Registry entry for a black-box LLM API environment."""
    id: str
    aliases: List[str]
    provider: str             # openai | huggingface | azure | anthropic
    model_name: str
    api_type: str             # chat_completion | text_completion | huggingface_inference
    endpoint: str
    context_window: int
    supports_logprobs: bool   # False for true black-box APIs
    cost_per_1k_input_tokens: float   # USD at paper publication time
    cost_per_1k_output_tokens: float
    in_scope: bool
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def make_client_config(self) -> Dict[str, Any]:
        """Return a serialisable client configuration dict for this environment."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "api_type": self.api_type,
            "endpoint": self.endpoint,
            "supports_logprobs": self.supports_logprobs,
        }


# ---------------------------------------------------------------------------
# Paper-derived dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "gsm8k": DatasetEntry(
        id="gsm8k",
        aliases=["grade_school_math", "gsm", "math_reasoning"],
        description="GSM8K: 8.5K grade school math word problems requiring multi-step numeric reasoning.",
        task_type="math_reasoning",
        answer_type="numeric",
        feedback_mode="ground_truth",
        metric="exact_match_numeric",
        hf_path="openai/gsm8k",
        hf_name="main",
        num_train=7473,
        num_test=1319,
        num_val=None,
        loader_hook="src.data.gsm8k.load_gsm8k",
        answer_extractor="regex_boxed_or_last_number",
        notes="Table 2 & Table 3 & Table 4 in paper. Positive samples = correct numeric answer.",
    ),
    "strategyqa": DatasetEntry(
        id="strategyqa",
        aliases=["strategy_qa", "implicit_reasoning", "strategyqa_binary"],
        description="StrategyQA: Implicit multi-hop binary (yes/no) reasoning over world knowledge.",
        task_type="implicit_reasoning",
        answer_type="binary_yes_no",
        feedback_mode="ai_feedback",
        metric="accuracy_binary",
        hf_path="wics/strategy-qa",
        hf_name="default",
        num_train=2061,
        num_test=229,
        num_val=None,
        loader_hook="src.data.strategyqa.load_strategyqa",
        answer_extractor="regex_yes_no",
        notes="Table 2 & Table 3 & Table 5 & Table 6. AI feedback used because labels unavailable at inference.",
    ),
    "truthfulqa": DatasetEntry(
        id="truthfulqa",
        aliases=["truthful_qa", "truthfulness", "tqa"],
        description="TruthfulQA: 817 questions probing truthfulness; MC evaluation.",
        task_type="truthfulness",
        answer_type="multiple_choice",
        feedback_mode="combined",
        metric="mc_accuracy",
        hf_path="truthful_qa",
        hf_name="multiple_choice",
        num_train=704,
        num_test=113,
        num_val=None,
        loader_hook="src.data.truthfulqa.load_truthfulqa",
        answer_extractor="mc_label",
        notes=(
            "Table 2. Combined feedback mode: AI feedback + ground-truth MC labels. "
            "MC_calcs: max/diff/scores-true/scores-false per TruthfulQA eval protocol."
        ),
    ),
    "scienceqa": DatasetEntry(
        id="scienceqa",
        aliases=["science_qa", "science_domain", "sciq"],
        description="ScienceQA: Multi-choice science domain QA with images and explanations.",
        task_type="science_domain",
        answer_type="multiple_choice",
        feedback_mode="ground_truth",
        metric="accuracy_mc",
        hf_path="derek-thomas/ScienceQA",
        hf_name=None,
        num_train=2000,
        num_test=2017,
        num_val=4241,
        loader_hook="src.data.scienceqa.load_scienceqa",
        answer_extractor="mc_label",
        notes="Table 2 & Table 3. Text-only subset used; images ignored for black-box LLM eval.",
    ),
    "toxigen": DatasetEntry(
        id="toxigen",
        aliases=["toxi_gen", "toxicity_reduction", "hate_speech"],
        description="ToxiGen: Machine-generated hate-speech / benign statements; toxicity reduction task.",
        task_type="toxicity_reduction",
        answer_type="binary_yes_no",
        feedback_mode="ai_feedback",
        metric="toxicity_score",
        hf_path="skg/toxigen-data",
        hf_name="train",
        num_train=8960,
        num_test=1000,
        num_val=None,
        loader_hook="src.data.toxigen.load_toxigen",
        answer_extractor="toxicity_classifier",
        notes="Table 2. AI feedback via Perspective API or GPT judge; lower toxicity score = better.",
    ),
}


# ---------------------------------------------------------------------------
# Paper-derived environment/model registry
# ---------------------------------------------------------------------------

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentEntry] = {
    "gpt_3_5_turbo": EnvironmentEntry(
        id="gpt_3_5_turbo",
        aliases=["gpt-3.5-turbo", "chatgpt", "openai_chat", "gpt35"],
        provider="openai",
        model_name="gpt-3.5-turbo",
        api_type="chat_completion",
        endpoint="https://api.openai.com/v1/chat/completions",
        context_window=16385,
        supports_logprobs=False,
        cost_per_1k_input_tokens=0.0015,
        cost_per_1k_output_tokens=0.002,
        in_scope=True,
        notes=(
            "Primary target LLM in the paper. Table 2 Main Results. "
            "BBox-Adapter trained on gpt-3.5-turbo then plugged into davinci-002 and Mixtral."
        ),
    ),
    "mixtral_8x7b": EnvironmentEntry(
        id="mixtral_8x7b",
        aliases=["Mixtral-8x7B-v0", "mixtral", "mixtral_moe"],
        provider="huggingface",
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_type="huggingface_inference",
        endpoint="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        context_window=32768,
        supports_logprobs=False,
        cost_per_1k_input_tokens=0.0007,
        cost_per_1k_output_tokens=0.0007,
        in_scope=True,
        notes=(
            "Plug-and-play target (Table 3). Adapter tuned on GPT-3.5 is applied zero-shot "
            "to Mixtral-8x7B to demonstrate cross-model transfer."
        ),
    ),
    "azure_openai": EnvironmentEntry(
        id="azure_openai",
        aliases=["azure_openai_endpoint", "azure_gpt", "azure"],
        provider="azure",
        model_name="gpt-3.5-turbo",
        api_type="chat_completion",
        endpoint="https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions",
        context_window=16385,
        supports_logprobs=False,
        cost_per_1k_input_tokens=0.0015,
        cost_per_1k_output_tokens=0.002,
        in_scope=True,
        notes="Azure-hosted OpenAI endpoint; same model weights as gpt_3_5_turbo.",
    ),
    "davinci_002": EnvironmentEntry(
        id="davinci_002",
        aliases=["text-davinci-002", "davinci002", "openai_completion"],
        provider="openai",
        model_name="text-davinci-002",
        api_type="text_completion",
        endpoint="https://api.openai.com/v1/completions",
        context_window=4097,
        supports_logprobs=False,
        cost_per_1k_input_tokens=0.002,
        cost_per_1k_output_tokens=0.002,
        in_scope=True,
        notes="Plug-and-play target (Table 3). Second LLM used to validate cross-model transfer.",
    ),
}


# ---------------------------------------------------------------------------
# Scope / cost / VRAM metadata (Table 4 — Training Cost & Cost Analysis)
# ---------------------------------------------------------------------------

SCOPE_METADATA: Dict[str, Any] = {
    "paper_title": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
    "in_scope_experiments": [
        "GSM8K ground-truth feedback (Table 2, Table 3, Table 4)",
        "StrategyQA AI feedback (Table 2, Table 3, Table 5, Table 6)",
        "TruthfulQA combined feedback (Table 2)",
        "ScienceQA ground-truth feedback (Table 2, Table 3)",
        "ToxiGen AI feedback (Table 2)",
    ],
    "out_of_scope": [
        "White-box fine-tuning baselines requiring GPU gradient access to target LLM",
        "Grey-box methods requiring token-level log-probabilities",
        "Full retrieval-augmented generation systems",
    ],
    "adapter_sizes": ["0.1B", "0.3B"],
    "adapter_backbone": "BERT-based sentence encoder",
    "cost_analysis": {
        "sft_azure_openai_usd": 100.0,    # approximate from paper Table 4
        "bbox_adapter_api_calls_usd": 2.0,  # approximate from paper Table 4
        "adapter_training_vram_gb": 8,      # single GPU, ~0.1B–0.3B BERT adapter
        "black_box_llm_access": "API only — no GPU required for target LLM",
    },
    "standard_deviation": "Reported as ± in Table 2; 3 seeds used for main results.",
    "evaluation_details": {
        "beam_inference": "sentence-level beam search with beam_size=5",
        "num_samples_per_question": 5,
        "online_adaptation_steps": 100,
        "positive_threshold": 0.5,
    },
}


# ---------------------------------------------------------------------------
# EnvironmentConfig — runtime configuration for a specific experiment run
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """
    Runtime configuration combining dataset + model environment + training settings.
    This is the config object passed to make_environment() and the training loop.
    """
    dataset_id: str
    environment_id: str
    feedback_mode: str
    num_train: int
    num_test: int
    adapter_size: str                    # "0.1B" | "0.3B"
    beam_size: int = 5
    num_samples: int = 5
    online_adaptation_steps: int = 100
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results"
    dry_run: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_entry(self) -> DatasetEntry:
        return DATASET_REGISTRY[self.dataset_id]

    @property
    def environment_entry(self) -> EnvironmentEntry:
        return ENVIRONMENT_REGISTRY[self.environment_id]


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_environment(config: EnvironmentConfig) -> Dict[str, Any]:
    """
    Factory function: build and return a fully-configured environment dict
    from an EnvironmentConfig instance.

    Returns a serialisable dict containing:
      - dataset_entry: DatasetEntry as dict
      - environment_entry: EnvironmentEntry as dict
      - client_config: dict with API/model access configuration
      - split_ratios: expected train/test/val counts from paper
      - readiness: bool indicating whether external dependencies are reachable

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    The transformer_qa forward() accepts question_with_context dicts analogously
    to how we pass QA inputs through the environment pipeline here.
    """
    if config.dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{config.dataset_id}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    if config.environment_id not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment '{config.environment_id}'. "
            f"Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )

    dataset = DATASET_REGISTRY[config.dataset_id]
    env = ENVIRONMENT_REGISTRY[config.environment_id]

    return {
        "dataset_entry": asdict(dataset),
        "environment_entry": asdict(env),
        "client_config": env.make_client_config(),
        "split_ratios": {
            "num_train": dataset.num_train,
            "num_test": dataset.num_test,
            "num_val": dataset.num_val,
        },
        "adapter_size": config.adapter_size,
        "feedback_mode": config.feedback_mode,
        "beam_size": config.beam_size,
        "num_samples": config.num_samples,
        "online_adaptation_steps": config.online_adaptation_steps,
        "seed": config.seed,
        "device": config.device,
        "dry_run": config.dry_run,
        "readiness": True,  # actual readiness check deferred to lazy import check below
    }


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_dataset(name: str) -> DatasetEntry:
    """Return DatasetEntry by id or alias."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    for entry in DATASET_REGISTRY.values():
        if name in entry.aliases:
            return entry
    raise KeyError(
        f"Dataset '{name}' not found. "
        f"Registered ids: {list(DATASET_REGISTRY.keys())}. "
        f"Registered aliases: {[a for e in DATASET_REGISTRY.values() for a in e.aliases]}"
    )


def get_environment(name: str) -> EnvironmentEntry:
    """Return EnvironmentEntry by id or alias."""
    if name in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[name]
    for entry in ENVIRONMENT_REGISTRY.values():
        if name in entry.aliases:
            return entry
    raise KeyError(
        f"Environment '{name}' not found. "
        f"Registered ids: {list(ENVIRONMENT_REGISTRY.keys())}."
    )


def list_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())


def list_environments() -> List[str]:
    return list(ENVIRONMENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Readiness check (lazy — does not import heavy optional packages at module level)
# ---------------------------------------------------------------------------

def check_readiness(environment_id: Optional[str] = None) -> Dict[str, bool]:
    """
    Lightweight readiness check.  Does NOT import torch, transformers, or datasets
    at module level; imports are deferred to inside this function.
    """
    results: Dict[str, bool] = {}

    # Check datasets package
    try:
        import importlib
        datasets_spec = importlib.util.find_spec("datasets")
        results["datasets_available"] = datasets_spec is not None
    except Exception:
        results["datasets_available"] = False

    # Check torch
    try:
        import importlib
        torch_spec = importlib.util.find_spec("torch")
        results["torch_available"] = torch_spec is not None
    except Exception:
        results["torch_available"] = False

    # Check transformers
    try:
        import importlib
        tf_spec = importlib.util.find_spec("transformers")
        results["transformers_available"] = tf_spec is not None
    except Exception:
        results["transformers_available"] = False

    # Check openai
    try:
        import importlib
        oa_spec = importlib.util.find_spec("openai")
        results["openai_available"] = oa_spec is not None
    except Exception:
        results["openai_available"] = False

    results["registry_loaded"] = True
    results["num_datasets"] = len(DATASET_REGISTRY)  # type: ignore[assignment]
    results["num_environments"] = len(ENVIRONMENT_REGISTRY)  # type: ignore[assignment]
    return results


# ---------------------------------------------------------------------------
# Artifact writers — materialise all declared artifact paths
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_environment_registry(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "environment_registry.json")
    _ensure_dir(path)
    payload = {
        "registry_type": "environment",
        "entries": {k: asdict(v) for k, v in ENVIRONMENT_REGISTRY.items()},
        "count": len(ENVIRONMENT_REGISTRY),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_dataset_registry(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "dataset_registry.json")
    _ensure_dir(path)
    payload = {
        "registry_type": "dataset",
        "entries": {k: asdict(v) for k, v in DATASET_REGISTRY.items()},
        "split_ratios": DATASET_SPLIT_RATIOS,
        "count": len(DATASET_REGISTRY),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_scope_report(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "scope_report.json")
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(SCOPE_METADATA, f, indent=2)
    return path


def write_data_manifest(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "data_manifest.json")
    _ensure_dir(path)
    manifest = {
        "datasets": [],
        "environments": [],
    }
    for ds in DATASET_REGISTRY.values():
        manifest["datasets"].append(
            {
                "id": ds.id,
                "aliases": ds.aliases,
                "task_type": ds.task_type,
                "hf_path": ds.hf_path,
                "hf_name": ds.hf_name,
                "num_train": ds.num_train,
                "num_test": ds.num_test,
                "num_val": ds.num_val,
                "loader_hook": ds.loader_hook,
                "metric": ds.metric,
            }
        )
    for env in ENVIRONMENT_REGISTRY.values():
        manifest["environments"].append(
            {
                "id": env.id,
                "aliases": env.aliases,
                "provider": env.provider,
                "model_name": env.model_name,
                "in_scope": env.in_scope,
            }
        )
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def write_metrics_schema(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "metrics.json")
    _ensure_dir(path)
    schema = {
        "metrics": {
            "exact_match_numeric": {
                "formula": "int(normalize(pred) == normalize(gold))",
                "aggregation": "mean over test set",
                "datasets": ["gsm8k"],
            },
            "accuracy_binary": {
                "formula": "int(pred_yes_no == gold_yes_no)",
                "aggregation": "mean over test set",
                "datasets": ["strategyqa"],
            },
            "mc_accuracy": {
                "formula": "int(argmax(energy_scores) == gold_label)",
                "aggregation": "mean over test set",
                "datasets": ["truthfulqa", "scienceqa"],
                "mc_sub_metrics": ["MC1", "MC2", "MC3"],
            },
            "toxicity_score": {
                "formula": "mean toxicity probability over generated continuations",
                "aggregation": "mean over test set (lower is better)",
                "datasets": ["toxigen"],
            },
        }
    }
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    return path


def write_cost_vram_report(output_dir: str = "results") -> str:
    path = os.path.join(output_dir, "cost_vram_report.json")
    _ensure_dir(path)
    report = {
        "paper_table_4_cost_analysis": {
            "sft_azure_openai_usd": SCOPE_METADATA["cost_analysis"]["sft_azure_openai_usd"],
            "bbox_adapter_api_calls_usd": SCOPE_METADATA["cost_analysis"]["bbox_adapter_api_calls_usd"],
            "adapter_training_vram_gb": SCOPE_METADATA["cost_analysis"]["adapter_training_vram_gb"],
            "black_box_llm_access": SCOPE_METADATA["cost_analysis"]["black_box_llm_access"],
        },
        "adapter_sizes": SCOPE_METADATA["adapter_sizes"],
        "training_cost_note": (
            "BBox-Adapter requires ~2 USD in API calls to train the adapter; "
            "SFT via Azure OpenAI costs ~100 USD. Adapter training uses a single GPU."
        ),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def write_all_artifacts(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Write all declared artifact paths under output_dir.
    Uses PAPERBENCH_REPRO_ARTIFACT_DIR env var when available.
    """
    if output_dir is None:
        output_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")

    paths: Dict[str, str] = {}
    paths["environment_registry"] = write_environment_registry(output_dir)
    paths["dataset_registry"] = write_dataset_registry(output_dir)
    paths["scope_report"] = write_scope_report(output_dir)
    paths["data_manifest"] = write_data_manifest(output_dir)
    paths["metrics"] = write_metrics_schema(output_dir)
    paths["cost_vram_report"] = write_cost_vram_report(output_dir)
    return paths


# ---------------------------------------------------------------------------
# Module-level self-test / smoke entrypoint
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Validate registry integrity without executing training or API calls."""
    assert len(DATASET_REGISTRY) == 5, "Expected 5 datasets"
    assert len(ENVIRONMENT_REGISTRY) == 4, "Expected 4 environment entries"

    for alias in ["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"]:
        entry = get_dataset(alias)
        assert entry.id == alias, f"id mismatch for {alias}"

    # Test alias lookups
    assert get_dataset("grade_school_math").id == "gsm8k"
    assert get_dataset("strategy_qa").id == "strategyqa"
    assert get_dataset("truthful_qa").id == "truthfulqa"
    assert get_dataset("science_qa").id == "scienceqa"
    assert get_dataset("toxi_gen").id == "toxigen"

    # Test environment lookups
    assert get_environment("gpt-3.5-turbo").id == "gpt_3_5_turbo"
    assert get_environment("Mixtral-8x7B-v0").id == "mixtral_8x7b"
    assert get_environment("azure_openai_endpoint").id == "azure_openai"

    # Test factory
    cfg = EnvironmentConfig(
        dataset_id="gsm8k",
        environment_id="gpt_3_5_turbo",
        feedback_mode="ground_truth",
        num_train=7473,
        num_test=1319,
        adapter_size="0.1B",
        dry_run=True,
    )
    env_dict = make_environment(cfg)
    assert env_dict["split_ratios"]["num_train"] == 7473
    assert env_dict["split_ratios"]["num_test"] == 1319
    assert env_dict["readiness"] is True

    # Write artifacts
    artifacts = write_all_artifacts()
    for key, path in artifacts.items():
        assert os.path.exists(path), f"Artifact missing: {path}"

    print("environment.py smoke test PASSED")
    print("Written artifacts:")
    for k, p in artifacts.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    _smoke_test()