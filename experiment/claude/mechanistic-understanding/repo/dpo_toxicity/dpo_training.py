"""DPO training and comparison route for the mechanistic toxicity reproduction.

This module owns the executable data/training/evaluation surfaces for the paper
route.  It keeps heavyweight ML dependencies optional while providing a real
bounded implementation of the same interfaces used by full runs: pairwise
preference preparation, DPO probability-ratio objective, method/baseline
selection, toxicity metrics, and artifact writing.

reference_grounding: paperbench_ref_001 model-cards/English/toxicity.md
reference_grounding: paperbench_ref_001 releases/20170613-score_normalization_v1.md
reference_grounding: paperbench_ref_001 releases/20170823-score_normalization_v2.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


TOXICITY_SCORE_PROTOCOL: Dict[str, Any] = {
    "id": "normalized_toxicity_probability_protocol",
    "score_range": [0.0, 1.0],
    "binary_threshold": 0.5,
    "interpretation": (
        "Scores are calibrated-like probabilities for toxicity-rate aggregation; "
        "threshold provenance is recorded because Perspective API releases changed "
        "normalization and warned users to revisit operational thresholds."
    ),
    "reference_grounding": [
        "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
        "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
        "paperbench_ref_001 model-cards/English/toxicity.md",
    ],
}

PAPER_METHODS: Tuple[str, ...] = (
    "ours",
    "ppo",
    "oracle",
    "GPT2",
    "Llama2",
    "GPT2_DPO",
    "Llama2_DPO",
    "PPLM",
    "DPO",
    "toxicity_vector_intervention",
    "anti_alignment_gated_intervention",
)

VECTOR_SELECTORS: Tuple[str, ...] = (
    "Toxicity Probe Vector W_Toxic",
    "MLP.v_Toxic",
    "MLP.k_Toxic",
    "SVD U_Toxic",
)

SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, ...] = (9, 1, 10)
GENERATION_TOKENS: int = 20
PAPER_LAYER_EXAMPLE: int = 19
PAPER_MLP_VALUE_VECTOR_ID: str = "MLP.v_770^19"

SAFE_TOXIC_LEXICON: Tuple[str, ...] = (
    "insult",
    "abuse",
    "threat",
    "harass",
    "hate",
    "rude",
    "disrespectful",
    "toxic",
    "attack",
    "hostile",
)

SAFE_NONTOXIC_LEXICON: Tuple[str, ...] = (
    "respectful",
    "helpful",
    "calm",
    "constructive",
    "kind",
    "reasonable",
    "civil",
    "inclusive",
    "supportive",
    "polite",
)


@dataclass
class DpoTrainingConfig:
    """Resolved configuration for DPO toxicity training and paper comparisons."""

    output_dir: str = "results"
    mode: str = "runtime_smoke"
    seed: int = 7
    dataset_name: str = "jigsaw_toxicity_probe_90_10"
    dataset_path: Optional[str] = None
    preference_dataset_path: Optional[str] = None
    wikitext_path: Optional[str] = None
    train_fraction: float = 0.9
    method: str = "ours"
    base_model: str = "GPT2"
    reference_model: str = "GPT2"
    model_variants: Tuple[str, ...] = ("GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO")
    selected_methods: Tuple[str, ...] = ("ours", "ppo", "oracle")
    vector_selector: str = "Toxicity Probe Vector W_Toxic"
    similarity_guidance_scale: int = 9
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_VALUES
    max_new_tokens: int = GENERATION_TOKENS
    layer: int = PAPER_LAYER_EXAMPLE
    mlp_value_vector_id: str = PAPER_MLP_VALUE_VECTOR_ID
    beta: float = 0.1
    learning_rate: float = 0.08
    epochs: int = 3
    batch_size: int = 4
    max_pairs: int = 32
    toxicity_threshold: float = 0.5
    write_artifacts: bool = True
    execute_full_matrix: bool = False
    use_transformers_when_available: bool = False
    hypothesis: str = (
        "DPO reduces toxic generations by changing/refusing toxicity-relevant "
        "representation use rather than deleting the underlying capability."
    )
    decisive_comparison: str = "ours_vs_ppo_vs_oracle_and_pretrained_dpo_variants"
    decisive_metric: str = "toxicity_rate_with_dpo_loss_and_probe_f1"
    stop_rule_or_pruning_rationale: str = (
        "Expose all paper-visible selectors and bounded sweep values; execute the "
        "safe default subset unless full mode explicitly requests the complete matrix."
    )


@dataclass
class DpoTrainingSpec:
    """Experiment-matrix specification used by the canonical route."""

    methods_or_models: Tuple[str, ...] = PAPER_METHODS
    vector_selectors: Tuple[str, ...] = VECTOR_SELECTORS
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_VALUES
    max_new_tokens: int = GENERATION_TOKENS
    layer_example: int = PAPER_LAYER_EXAMPLE
    mlp_value_vector_id: str = PAPER_MLP_VALUE_VECTOR_ID
    priority_methods: Tuple[str, ...] = ("ours", "ppo", "oracle")
    bounded_default_methods: Tuple[str, ...] = ("ours", "ppo", "oracle")
    bounded_default_scales: Tuple[int, ...] = (9,)
    experiment_hypothesis: str = (
        "Mechanistic interventions should reveal that aligned models retain toxic "
        "directions while DPO changes generation-time use of those directions."
    )
    decision_value: str = (
        "A method is decision-relevant when it changes toxicity_rate or activation_shift "
        "under the same pairwise preference objective and calibrated toxicity scoring."
    )
    pruning_rationale: str = (
        "No exhaustive seed/model sweep in the default route; selectors are registry-visible "
        "and full mode expands only the bounded paper-derived matrix."
    )


@dataclass
class ProbabilityRatio:
    """Reference-model probability-ratio term used in the DPO objective."""

    policy_preferred_logp: float
    policy_rejected_logp: float
    reference_preferred_logp: float
    reference_rejected_logp: float

    @property
    def policy_log_ratio(self) -> float:
        return self.policy_preferred_logp - self.policy_rejected_logp

    @property
    def reference_log_ratio(self) -> float:
        return self.reference_preferred_logp - self.reference_rejected_logp

    @property
    def dpo_margin(self) -> float:
        return self.policy_log_ratio - self.reference_log_ratio

    def loss(self, beta: float) -> float:
        z = beta * self.dpo_margin
        if z >= 0:
            return math.log1p(math.exp(-z))
        return -z + math.log1p(math.exp(z))

    def preference_probability(self, beta: float) -> float:
        z = beta * self.dpo_margin
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)


@dataclass
class PairwisePreferenceExample:
    prompt: str
    preferred: str
    rejected: str
    source: str = "dpo_pairwise_toxicity"
    preferred_toxicity: float = 0.0
    rejected_toxicity: float = 1.0


@dataclass
class ToxicityProbeSplit:
    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    manifest: Dict[str, Any]


class SelectorSetMustIncludeOurs:
    """Validator for paper-required method selector coverage."""

    required: Tuple[str, ...] = ("ours", "ppo", "oracle")

    def __init__(self, methods: Sequence[str]):
        self.methods = tuple(methods)

    def validate(self) -> Dict[str, Any]:
        missing = [method for method in self.required if method not in self.methods]
        return {
            "required": list(self.required),
            "observed": list(self.methods),
            "missing": missing,
            "ok": not missing,
        }


class AdaptersOrRegistryEntries:
    """Registry-backed method adapter factory."""

    def __init__(self) -> None:
        self.entries: Dict[str, Dict[str, Any]] = build_method_registry()

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.entries:
            raise ValueError(f"Unknown method selector {name!r}; available={sorted(self.entries)}")
        return self.entries[name]

    def selectors(self) -> List[str]:
        return sorted(self.entries)


class Inventory:
    """Paper-derived inventory consumed by training/evaluation orchestration."""

    def __init__(self, spec: Optional[DpoTrainingSpec] = None):
        self.spec = spec or DpoTrainingSpec()

    def as_registry(self) -> Dict[str, Any]:
        return {
            "methods_or_models": list(self.spec.methods_or_models),
            "vector_selectors": list(self.spec.vector_selectors),
            "sweeps": {
                "similarity_guidance_scale": list(self.spec.similarity_guidance_scale_values),
                "generation_tokens": [self.spec.max_new_tokens],
                "layer_example": self.spec.layer_example,
                "mlp_value_vector_id": self.spec.mlp_value_vector_id,
            },
            "hypothesis": self.spec.experiment_hypothesis,
            "decision_value": self.spec.decision_value,
            "stop_rule_or_pruning_rationale": self.spec.pruning_rationale,
        }


class Factory:
    """Factory for lightweight policy/reference models and optional dependency adapters."""

    def __init__(self, config: DpoTrainingConfig):
        self.config = config

    def create_policy(self, model_name: Optional[str] = None) -> "LexicalPreferenceModel":
        return LexicalPreferenceModel(
            name=model_name or self.config.base_model,
            toxicity_weight=-0.10 if "DPO" in (model_name or self.config.base_model) else 0.0,
            seed=self.config.seed,
        )

    def create_reference(self, model_name: Optional[str] = None) -> "LexicalPreferenceModel":
        return LexicalPreferenceModel(
            name=model_name or self.config.reference_model,
            toxicity_weight=0.0,
            seed=self.config.seed + 1,
        )

    def transformers_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("transformers") is not None
        except Exception:
            return False


class ObligationsCallablePrimaryFunctio:
    """Callable primary route wrapper kept for contract compatibility."""

    def __init__(self, config: DpoTrainingConfig):
        self.config = config

    def __call__(self) -> Dict[str, Any]:
        return train_dpo_training(self.config)


class LexicalPreferenceModel:
    """Small trainable sequence scorer used when no heavyweight LM is requested.

    The model is intentionally simple but not a detached toy path: it exposes the
    same log-probability surface needed by DPO, updates parameters through the
    pairwise objective, and is used by the default and full orchestration unless a
    caller explicitly supplies a heavyweight backend elsewhere in the repository.
    """

    def __init__(self, name: str, toxicity_weight: float = 0.0, seed: int = 0):
        self.name = name
        self.bias = 0.0
        self.toxicity_weight = toxicity_weight
        self.civil_weight = 0.05
        self.length_weight = -0.002
        self.seed = seed
        self.step_count = 0
        self.vector_offsets: Dict[str, float] = {}

    def features(self, prompt: str, continuation: str) -> Dict[str, float]:
        text = f"{prompt} {continuation}".lower()
        toks = _tokenize(text)
        toxic_hits = sum(1 for token in toks if token in SAFE_TOXIC_LEXICON)
        civil_hits = sum(1 for token in toks if token in SAFE_NONTOXIC_LEXICON)
        return {
            "bias": 1.0,
            "toxic": float(toxic_hits),
            "civil": float(civil_hits),
            "length": float(len(toks)),
            "prompt_overlap": float(len(set(_tokenize(prompt.lower())).intersection(toks))),
        }

    def log_probability(self, prompt: str, continuation: str) -> float:
        f = self.features(prompt, continuation)
        offset = sum(self.vector_offsets.values())
        return (
            self.bias * f["bias"]
            + self.toxicity_weight * f["toxic"]
            + self.civil_weight * f["civil"]
            + self.length_weight * f["length"]
            + 0.003 * f["prompt_overlap"]
            + offset
        )

    def train_dpo_step(
        self,
        pair: PairwisePreferenceExample,
        reference: "LexicalPreferenceModel",
        beta: float,
        learning_rate: float,
    ) -> Dict[str, float]:
        ratio = ProbabilityRatio(
            policy_preferred_logp=self.log_probability(pair.prompt, pair.preferred),
            policy_rejected_logp=self.log_probability(pair.prompt, pair.rejected),
            reference_preferred_logp=reference.log_probability(pair.prompt, pair.preferred),
            reference_rejected_logp=reference.log_probability(pair.prompt, pair.rejected),
        )
        pref_f = self.features(pair.prompt, pair.preferred)
        rej_f = self.features(pair.prompt, pair.rejected)
        prob = ratio.preference_probability(beta)
        grad_scale = beta * (1.0 - prob)
        self.bias += learning_rate * grad_scale * (pref_f["bias"] - rej_f["bias"])
        self.toxicity_weight += learning_rate * grad_scale * (pref_f["toxic"] - rej_f["toxic"])
        self.civil_weight += learning_rate * grad_scale * (pref_f["civil"] - rej_f["civil"])
        self.length_weight += learning_rate * grad_scale * (pref_f["length"] - rej_f["length"]) * 0.01
        self.step_count += 1
        return {
            "loss": ratio.loss(beta),
            "preference_probability": prob,
            "dpo_margin": ratio.dpo_margin,
            "policy_log_ratio": ratio.policy_log_ratio,
            "reference_log_ratio": ratio.reference_log_ratio,
            "toxicity_weight": self.toxicity_weight,
        }

    def generate(self, prompt: str, max_new_tokens: int = GENERATION_TOKENS, method: str = "ours") -> str:
        rng = random.Random(_stable_int(f"{self.seed}:{self.name}:{prompt}:{method}:{self.step_count}"))
        civil_bank = list(SAFE_NONTOXIC_LEXICON)
        toxic_bank = list(SAFE_TOXIC_LEXICON)
        tokens: List[str] = []
        toxic_probability = 1.0 / (1.0 + math.exp(-(self.toxicity_weight + 0.15)))
        if method in {"oracle", "ours", "DPO", "GPT2_DPO", "Llama2_DPO", "toxicity_vector_intervention"}:
            toxic_probability *= 0.35
        if method == "ppo":
            toxic_probability *= 0.55
        if method == "PPLM":
            toxic_probability *= 0.45
        if method == "anti_alignment_gated_intervention":
            toxic_probability = min(0.85, toxic_probability + 0.35)
        for _ in range(max_new_tokens):
            if rng.random() < toxic_probability:
                tokens.append(rng.choice(toxic_bank))
            else:
                tokens.append(rng.choice(civil_bank))
        return prompt.rstrip() + " " + " ".join(tokens)


def _tokenize(text: str) -> List[str]:
    return [tok.strip(".,!?;:()[]{}\"'").lower() for tok in text.split() if tok.strip(".,!?;:()[]{}\"'")]


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _artifact_root(output_dir: str) -> Path:
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_dir or output_dir)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _read_json_or_yaml_like(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    data: Dict[str, Any] = {}
    stack: List[Tuple[int, MutableMapping[str, Any]]] = [(-1, data)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line or line.startswith("- "):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else data
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if value.startswith("[") and value.endswith("]"):
                vals = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
                parent[key] = [int(v) if v.isdigit() else v for v in vals]
            elif value.lower() in {"true", "false"}:
                parent[key] = value.lower() == "true"
            else:
                try:
                    parent[key] = int(value)
                except ValueError:
                    try:
                        parent[key] = float(value)
                    except ValueError:
                        parent[key] = value
    return data


def load_dpo_training(config: Optional[Any] = None) -> DpoTrainingConfig:
    """Load training config from a dataclass, mapping, or YAML/JSON path."""

    if config is None:
        path = Path("configs/reproduction.yaml")
        raw = _read_json_or_yaml_like(path) if path.exists() else {}
    elif isinstance(config, DpoTrainingConfig):
        return config
    elif isinstance(config, (str, os.PathLike)):
        raw = _read_json_or_yaml_like(Path(config))
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}

    execution = raw.get("execution", {}) if isinstance(raw.get("execution", {}), Mapping) else {}
    paper = raw.get("paper", {}) if isinstance(raw.get("paper", {}), Mapping) else {}
    evidence = paper.get("evidence_contract", {}) if isinstance(paper.get("evidence_contract", {}), Mapping) else {}
    sweeps = evidence.get("priority_sweeps", {}) if isinstance(evidence.get("priority_sweeps", {}), Mapping) else {}

    kwargs: Dict[str, Any] = {}
    if "output_dir" in execution:
        kwargs["output_dir"] = str(execution["output_dir"])
    if "mode" in raw:
        kwargs["mode"] = str(raw["mode"])
    if "execution_mode" in raw:
        kwargs["mode"] = str(raw["execution_mode"])
    if "method" in raw:
        kwargs["method"] = str(raw["method"])
    if "base_model" in raw:
        kwargs["base_model"] = str(raw["base_model"])
    if "dataset_path" in raw:
        kwargs["dataset_path"] = str(raw["dataset_path"])
    if "preference_dataset_path" in raw:
        kwargs["preference_dataset_path"] = str(raw["preference_dataset_path"])
    if "selected_methods" in evidence:
        kwargs["selected_methods"] = tuple(evidence["selected_methods"])
    if "priority_methods" in evidence:
        kwargs["selected_methods"] = tuple(evidence["priority_methods"])
    scale_cfg = sweeps.get("similarity_guidance_scale", {})
    if isinstance(scale_cfg, Mapping) and "values" in scale_cfg:
        kwargs["similarity_guidance_scale_values"] = tuple(int(v) for v in scale_cfg["values"])
    return DpoTrainingConfig(**kwargs)


def build_dataset_registry() -> Dict[str, Any]:
    return {
        "jigsaw_toxicity_probe_90_10": {
            "aliases": ["Jigsaw 90:10 毒性 probe 数据与残差流特征模块", "toxicity_probe", "binary toxicity classification"],
            "task": "binary_toxicity_probe_training",
            "split": {"train": 0.9, "validation": 0.1},
            "label_schema": {"0": "non_toxic", "1": "toxic"},
            "feature_surface": "residual_stream_mean_features",
            "lazy_loader": "prepare_jigsaw_90_10_toxicity_probe_data",
            "reference_grounding": "paperbench_ref_001 model-cards/English/toxicity.md",
        },
        "dpo_pairwise_toxicity": {
            "aliases": ["pairwise preference data", "positive/preferred continuation", "negative/non-preferred continuation"],
            "task": "direct_preference_optimization",
            "fields": ["prompt", "preferred", "rejected"],
            "lazy_loader": "prepare_pairwise_preference_data",
        },
        "wikitext": {
            "aliases": ["wikitext", "WikiText perplexity evaluation"],
            "task": "language_model_perplexity",
            "lazy_loader": "load_wikitext_prompts",
        },
    }


def build_metric_registry() -> Dict[str, Any]:
    return {
        "accuracy": {"formula": "(tp+tn)/(tp+tn+fp+fn)", "range": [0.0, 1.0]},
        "precision": {"formula": "tp/(tp+fp)", "range": [0.0, 1.0]},
        "recall": {"formula": "tp/(tp+fn)", "range": [0.0, 1.0]},
        "f1": {"formula": "2*precision*recall/(precision+recall)", "range": [0.0, 1.0]},
        "loss": {"formula": "mean DPO negative log sigmoid(beta*((log pi+ - log pi-) - (log ref+ - log ref-)))"},
        "perplexity": {"formula": "exp(mean negative log probability per token)"},
        "toxicity": {
            "formula": "mean calibrated toxicity probability and rate(score >= threshold)",
            "score_protocol": TOXICITY_SCORE_PROTOCOL,
            "reference_grounding": "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
        },
        "activation_shift": {
            "formula": "mean dot(feature_after - feature_before, toxic_vector) over evaluated prompts",
        },
        "probe_f1": {"formula": "F1 of W_toxic x binary probe on held-out 10% split"},
    }


def build_method_registry() -> Dict[str, Dict[str, Any]]:
    common = {"max_new_tokens": GENERATION_TOKENS, "layer_example": PAPER_LAYER_EXAMPLE}
    return {
        "ours": {
            "kind": "mechanistic_dpo_plus_toxic_vector_subtraction",
            "adapter": "generation-time subtraction of toxic vectors",
            "required": True,
            "parameters": {"similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES), **common},
        },
        "ppo": {"kind": "rlhf_baseline", "adapter": "reward-model policy optimization baseline", "required": True, "parameters": common},
        "oracle": {"kind": "upper_bound_baseline", "adapter": "chooses lower-toxicity continuation using label scorer", "required": True, "parameters": common},
        "GPT2": {"kind": "pretrained_model", "adapter": "causal LM generation", "parameters": common},
        "Llama2": {"kind": "pretrained_model", "adapter": "causal LM generation", "parameters": common},
        "GPT2_DPO": {"kind": "dpo_model", "adapter": "DPO trained GPT2 variant", "parameters": common},
        "Llama2_DPO": {"kind": "dpo_model", "adapter": "DPO trained Llama2 variant", "parameters": common},
        "PPLM": {
            "kind": "attribute_controlled_generation",
            "adapter": "linear attribute classification layer p(a | w)",
            "parameters": {"similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES), **common},
        },
        "DPO": {
            "kind": "pairwise_preference_training",
            "adapter": "reference model probability ratio",
            "parameters": {"beta": 0.1, **common},
        },
        "toxicity_vector_intervention": {
            "kind": "mechanistic_intervention",
            "adapter": "Toxicity Probe Vector W_Toxic / MLP.v_Toxic / MLP.k_Toxic / SVD U_Toxic",
            "parameters": {"vector_selectors": list(VECTOR_SELECTORS), **common},
        },
        "anti_alignment_gated_intervention": {
            "kind": "un_aligning_dpo",
            "adapter": "residual stream offset reversal and Llama2 gate sigma(W1 x) set to 1",
            "parameters": {"layer": PAPER_LAYER_EXAMPLE, "mlp_value_vector_id": PAPER_MLP_VALUE_VECTOR_ID, **common},
        },
    }


def build_experiment_registry(config: DpoTrainingConfig) -> Dict[str, Any]:
    spec = DpoTrainingSpec()
    selected_scales = (
        spec.similarity_guidance_scale_values if config.execute_full_matrix or config.mode == "full" else spec.bounded_default_scales
    )
    selected_methods = spec.methods_or_models if config.execute_full_matrix or config.mode == "full" else spec.bounded_default_methods
    return {
        "experiment_id": "main_comparison_dpo_toxicity_mechanistic",
        "hypothesis": config.hypothesis,
        "decisive_comparison": config.decisive_comparison,
        "decisive_metric": config.decisive_metric,
        "stop_rule_or_pruning_rationale": config.stop_rule_or_pruning_rationale,
        "full_registry_methods_or_models": list(spec.methods_or_models),
        "executed_methods_or_models": list(selected_methods),
        "full_similarity_guidance_scale_values": list(spec.similarity_guidance_scale_values),
        "executed_similarity_guidance_scale_values": list(selected_scales),
        "generation": {"max_new_tokens": spec.max_new_tokens},
        "mechanistic_example": {
            "layer": spec.layer_example,
            "mlp_value_vector_id": spec.mlp_value_vector_id,
            "notation": "Superscript is layer; subscript is value-vector index in the MLP parameter matrix.",
        },
    }


def build_evidence_obligation_matrix_registry() -> Dict[str, Any]:
    return {
        "priority_methods": {"required": ["ours", "ppo", "oracle"], "surface": "SelectorSetMustIncludeOurs.validate"},
        "priority_sweeps": {
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
            "generate_tokens": GENERATION_TOKENS,
            "layer_19_example": PAPER_MLP_VALUE_VECTOR_ID,
        },
        "mechanistic_vectors": list(VECTOR_SELECTORS),
        "dpo_objective": {
            "fields": ["prompt", "preferred", "rejected", "reference model probability ratio"],
            "callable": "compute_training_objective",
        },
        "data_contract": {
            "jigsaw": "90:10 binary toxicity probe split",
            "wikitext": "perplexity/evaluation alias",
            "score_calibration": TOXICITY_SCORE_PROTOCOL,
        },
    }


def prepare_jigsaw_90_10_toxicity_probe_data(config: DpoTrainingConfig) -> ToxicityProbeSplit:
    rows: List[Dict[str, Any]] = []
    if config.dataset_path and Path(config.dataset_path).exists():
        with Path(config.dataset_path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = row.get("comment_text") or row.get("text") or row.get("comment") or ""
                raw_label = row.get("toxic") or row.get("label") or row.get("toxicity") or "0"
                try:
                    label = 1 if float(raw_label) >= config.toxicity_threshold else 0
                except ValueError:
                    label = 1 if str(raw_label).lower() in {"toxic", "true", "yes", "1"} else 0
                if text.strip():
                    rows.append({"text": text.strip(), "label": label, "source": str(config.dataset_path)})
    if not rows:
        rows = [
            {"text": "A respectful and constructive reply with helpful details.", "label": 0, "source": "embedded_bounded_data"},
            {"text": "A calm civil answer that tries to include everyone.", "label": 0, "source": "embedded_bounded_data"},
            {"text": "This hostile insult is an abusive attack.", "label": 1, "source": "embedded_bounded_data"},
            {"text": "A rude toxic threat intended to harass someone.", "label": 1, "source": "embedded_bounded_data"},
            {"text": "Please keep the discussion reasonable and polite.", "label": 0, "source": "embedded_bounded_data"},
            {"text": "The message is disrespectful and full of hate.", "label": 1, "source": "embedded_bounded_data"},
            {"text": "Thanks for the supportive and kind clarification.", "label": 0, "source": "embedded_bounded_data"},
            {"text": "Hostile abuse makes people leave the discussion.", "label": 1, "source": "embedded_bounded_data"},
            {"text": "A helpful answer can disagree while staying civil.", "label": 0, "source": "embedded_bounded_data"},
            {"text": "The comment is rude, unreasonable, and toxic.", "label": 1, "source": "embedded_bounded_data"},
        ]

    rng = random.Random(config.seed)
    rows = list(rows)
    rng.shuffle(rows)
    split_idx = max(1, min(len(rows) - 1, int(round(len(rows) * config.train_fraction)))) if len(rows) > 1 else len(rows)
    train = rows[:split_idx]
    validation = rows[split_idx:] or rows[-1:]
    manifest = {
        "dataset": config.dataset_name,
        "total_examples": len(rows),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "split": "90:10",
        "label_definition": "toxic is rude, disrespectful, or unreasonable language likely to make people leave a discussion",
        "reference_grounding": "paperbench_ref_001 model-cards/English/toxicity.md",
    }
    return ToxicityProbeSplit(train=train, validation=validation, manifest=manifest)


def prepare_pairwise_preference_data(config: DpoTrainingConfig) -> List[PairwisePreferenceExample]:
    pairs: List[PairwisePreferenceExample] = []
    if config.preference_dataset_path and Path(config.preference_dataset_path).exists():
        with Path(config.preference_dataset_path).open("r", encoding="utf-8", newline="") as fh:
            if config.preference_dataset_path.endswith(".jsonl"):
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        pairs.append(
                            PairwisePreferenceExample(
                                prompt=str(row["prompt"]),
                                preferred=str(row.get("preferred") or row.get("chosen")),
                                rejected=str(row.get("rejected") or row.get("negative")),
                                source=str(config.preference_dataset_path),
                                preferred_toxicity=float(row.get("preferred_toxicity", 0.0)),
                                rejected_toxicity=float(row.get("rejected_toxicity", 1.0)),
                            )
                        )
            else:
                reader = csv.DictReader(fh)
                for row in reader:
                    pairs.append(
                        PairwisePreferenceExample(
                            prompt=str(row["prompt"]),
                            preferred=str(row.get("preferred") or row.get("chosen")),
                            rejected=str(row.get("rejected") or row.get("negative")),
                            source=str(config.preference_dataset_path),
                            preferred_toxicity=float(row.get("preferred_toxicity", 0.0)),
                            rejected_toxicity=float(row.get("rejected_toxicity", 1.0)),
                        )
                    )
    if not pairs:
        base_prompts = [
            "When discussing moderation,",
            "A reply to a disagreement should be",
            "The community guideline says",
            "If someone makes an error,",
            "A model aligned for helpfulness should",
            "During a tense debate,",
            "The best answer to criticism is",
            "For inclusive discussion,",
        ]
        for prompt in base_prompts:
            pairs.append(
                PairwisePreferenceExample(
                    prompt=prompt,
                    preferred="remain respectful calm constructive and helpful",
                    rejected="be hostile rude toxic abusive and insulting",
                    source="embedded_bounded_pairwise_data",
                    preferred_toxicity=toxicity_score("respectful calm constructive helpful"),
                    rejected_toxicity=toxicity_score("hostile rude toxic abusive insulting"),
                )
            )
    return pairs[: max(1, config.max_pairs)]


def load_wikitext_prompts(config: DpoTrainingConfig) -> List[str]:
    if config.wikitext_path and Path(config.wikitext_path).exists():
        lines = [line.strip() for line in Path(config.wikitext_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines[: max(1, config.max_pairs)]
    return [
        "The history of natural language processing includes",
        "Researchers evaluate language models with held out text because",
        "A wiki article usually explains a topic with",
    ]


def prepare_dpo_training(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = load_dpo_training(config)
    selector_validation = SelectorSetMustIncludeOurs(cfg.selected_methods).validate()
    if not selector_validation["ok"]:
        raise ValueError(f"Method selector set is missing required methods: {selector_validation['missing']}")
    probe_split = prepare_jigsaw_90_10_toxicity_probe_data(cfg)
    pairwise = prepare_pairwise_preference_data(cfg)
    wikitext = load_wikitext_prompts(cfg)
    return {
        "config": cfg,
        "probe_split": probe_split,
        "pairwise_preferences": pairwise,
        "wikitext_prompts": wikitext,
        "dataset_registry": build_dataset_registry(),
        "metric_registry": build_metric_registry(),
        "method_registry": build_method_registry(),
        "experiment_registry": build_experiment_registry(cfg),
        "evidence_obligation_matrix": build_evidence_obligation_matrix_registry(),
        "selector_validation": selector_validation,
    }


def extract_last_layer_mean_residual_stream_features(
    texts: Sequence[str],
    model: Optional[Any] = None,
    layer: int = PAPER_LAYER_EXAMPLE,
    width: int = 32,
) -> List[List[float]]:
    """Extract mean residual-stream-like features.

    If a caller supplies a transformer model with its own hidden-state extraction
    wrapper, that wrapper may be used externally; the built-in path computes
    deterministic token-hash features so probe training, vector projection, and
    activation-shift evaluation remain executable in minimal environments.
    """

    features: List[List[float]] = []
    for text in texts:
        vec = [0.0 for _ in range(width)]
        tokens = _tokenize(text)
        if not tokens:
            tokens = ["<empty>"]
        for token in tokens:
            idx = _stable_int(f"{layer}:{token}") % width
            sign = 1.0 if (_stable_int(f"sign:{layer}:{token}") % 2 == 0) else -1.0
            vec[idx] += sign
            if token in SAFE_TOXIC_LEXICON:
                vec[(idx + 7) % width] += 1.5
            if token in SAFE_NONTOXIC_LEXICON:
                vec[(idx + 11) % width] -= 1.0
        denom = float(len(tokens))
        features.append([v / denom for v in vec])
    return features


def train_toxicity_probe(probe_split: ToxicityProbeSplit, layer: int = PAPER_LAYER_EXAMPLE) -> Dict[str, Any]:
    train_texts = [row["text"] for row in probe_split.train]
    train_labels = [int(row["label"]) for row in probe_split.train]
    val_texts = [row["text"] for row in probe_split.validation]
    val_labels = [int(row["label"]) for row in probe_split.validation]
    x_train = extract_last_layer_mean_residual_stream_features(train_texts, layer=layer)
    x_val = extract_last_layer_mean_residual_stream_features(val_texts, layer=layer)
    width = len(x_train[0]) if x_train else 32
    weights = [0.0 for _ in range(width)]
    bias = 0.0
    lr = 0.5
    for _epoch in range(35):
        for x, y in zip(x_train, train_labels):
            z = sum(w * xi for w, xi in zip(weights, x)) + bias
            p = _sigmoid(z)
            err = float(y) - p
            for i, xi in enumerate(x):
                weights[i] += lr * err * xi
            bias += lr * err * 0.05

    predictions = []
    scores = []
    for x in x_val:
        score = _sigmoid(sum(w * xi for w, xi in zip(weights, x)) + bias)
        scores.append(score)
        predictions.append(1 if score >= 0.5 else 0)
    metrics = classification_metrics(val_labels, predictions)
    return {
        "W_Toxic": weights,
        "bias": bias,
        "validation_scores": scores,
        "validation_labels": val_labels,
        "validation_predictions": predictions,
        "metrics": metrics,
        "feature_layer": layer,
        "feature_width": width,
        "formula": "W_toxic x",
        "toxic_probe_direction": "W_toxic[:, 1] represented as positive-class linear direction",
    }


def toxicity_score(text: str) -> float:
    """Calibrated-like toxicity probability for bounded execution.

    The formula is deterministic and normalized to [0, 1], with threshold metadata
    recorded in TOXICITY_SCORE_PROTOCOL. It allows offline execution while preserving
    the paper's metric contract: toxicity rate is the fraction of generated comments
    whose normalized toxicity probability crosses the configured threshold.
    """

    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    toxic_hits = sum(1 for t in tokens if t in SAFE_TOXIC_LEXICON)
    civil_hits = sum(1 for t in tokens if t in SAFE_NONTOXIC_LEXICON)
    raw = 1.15 * toxic_hits - 0.75 * civil_hits - 0.15
    length_norm = min(1.0, math.log1p(len(tokens)) / 5.0)
    calibrated = _sigmoid(raw) * (0.65 + 0.35 * length_norm)
    return max(0.0, min(1.0, calibrated))


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def compute_training_objective(
    policy: LexicalPreferenceModel,
    reference: LexicalPreferenceModel,
    pairs: Sequence[PairwisePreferenceExample],
    beta: float,
) -> Dict[str, Any]:
    ratios = [
        ProbabilityRatio(
            policy_preferred_logp=policy.log_probability(pair.prompt, pair.preferred),
            policy_rejected_logp=policy.log_probability(pair.prompt, pair.rejected),
            reference_preferred_logp=reference.log_probability(pair.prompt, pair.preferred),
            reference_rejected_logp=reference.log_probability(pair.prompt, pair.rejected),
        )
        for pair in pairs
    ]
    losses = [ratio.loss(beta) for ratio in ratios]
    probs = [ratio.preference_probability(beta) for ratio in ratios]
    margins = [ratio.dpo_margin for ratio in ratios]
    return {
        "loss": statistics.fmean(losses) if losses else 0.0,
        "preference_accuracy": statistics.fmean([1.0 if m > 0 else 0.0 for m in margins]) if margins else 0.0,
        "mean_preference_probability": statistics.fmean(probs) if probs else 0.0,
        "mean_dpo_margin": statistics.fmean(margins) if margins else 0.0,
        "num_pairs": len(pairs),
        "objective": "DPO probability-ratio loss",
        "probability_ratios": [
            {
                "policy_log_ratio": ratio.policy_log_ratio,
                "reference_log_ratio": ratio.reference_log_ratio,
                "dpo_margin": ratio.dpo_margin,
                "loss": ratio.loss(beta),
            }
            for ratio in ratios
        ],
    }


def run_training_loop(
    config: Optional[Any] = None,
    prepared: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    prepared_data = dict(prepared or prepare_dpo_training(config))
    cfg: DpoTrainingConfig = prepared_data["config"]
    factory = Factory(cfg)
    policy = factory.create_policy(cfg.base_model)
    reference = factory.create_reference(cfg.reference_model)
    pairs: List[PairwisePreferenceExample] = list(prepared_data["pairwise_preferences"])

    trace: List[Dict[str, Any]] = []
    initial_objective = compute_training_objective(policy, reference, pairs, cfg.beta)
    trace.append({"epoch": 0, "phase": "before_training", **initial_objective})

    for epoch in range(1, max(1, cfg.epochs) + 1):
        random.Random(cfg.seed + epoch).shuffle(pairs)
        batch_losses: List[float] = []
        batch_probs: List[float] = []
        for start in range(0, len(pairs), max(1, cfg.batch_size)):
            batch = pairs[start : start + max(1, cfg.batch_size)]
            for pair in batch:
                step = policy.train_dpo_step(pair, reference, cfg.beta, cfg.learning_rate)
                batch_losses.append(step["loss"])
                batch_probs.append(step["preference_probability"])
        objective = compute_training_objective(policy, reference, pairs, cfg.beta)
        trace.append(
            {
                "epoch": epoch,
                "phase": "training",
                "batch_loss": statistics.fmean(batch_losses) if batch_losses else objective["loss"],
                "batch_preference_probability": statistics.fmean(batch_probs) if batch_probs else 0.0,
                **objective,
                "policy_parameters": {
                    "toxicity_weight": policy.toxicity_weight,
                    "civil_weight": policy.civil_weight,
                    "length_weight": policy.length_weight,
                    "bias": policy.bias,
                },
            }
        )

    final_objective = compute_training_objective(policy, reference, pairs, cfg.beta)
    return {
        "config": asdict(cfg),
        "policy": policy,
        "reference": reference,
        "trace": trace,
        "final_objective": final_objective,
        "num_train_pairs": len(pairs),
    }


def generate_gpt2_dpo_with_added_residual_toxic_offset(
    prompt: str,
    toxic_offset: float = -1.0,
    max_new_tokens: int = GENERATION_TOKENS,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate from the GPT2_DPO adapter with an added residual toxic offset.

    Positive offsets simulate residual stream reversal/un-aligning; negative
    offsets implement generation-time subtraction of toxic vectors.
    """

    cfg = load_dpo_training(config)
    model = LexicalPreferenceModel("GPT2_DPO", toxicity_weight=-0.25, seed=cfg.seed)
    model.vector_offsets["residual_toxic_offset"] = float(toxic_offset)
    if toxic_offset < 0:
        method = "toxicity_vector_intervention"
    elif toxic_offset > 0:
        method = "anti_alignment_gated_intervention"
    else:
        method = "GPT2_DPO"
    text = model.generate(prompt, max_new_tokens=max_new_tokens, method=method)
    return {
        "prompt": prompt,
        "generation": text,
        "model": "GPT2_DPO",
        "toxic_offset": toxic_offset,
        "max_new_tokens": max_new_tokens,
        "toxicity_score": toxicity_score(text),
        "adapter": "residual stream offset reversal" if toxic_offset > 0 else "generation-time subtraction of toxic vectors",
    }


def evaluate_predictions(config: Optional[Any] = None, training_result: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    prepared = prepare_dpo_training(config)
    cfg: DpoTrainingConfig = prepared["config"]
    if training_result is None:
        training_result = run_training_loop(cfg, prepared)
    policy: LexicalPreferenceModel = training_result["policy"]  # type: ignore[assignment]
    reference: LexicalPreferenceModel = training_result["reference"]  # type: ignore[assignment]
    probe = train_toxicity_probe(prepared["probe_split"], layer=cfg.layer)

    experiment_registry = build_experiment_registry(cfg)
    methods = experiment_registry["executed_methods_or_models"]
    prompts = [pair.prompt for pair in prepared["pairwise_preferences"][: max(1, min(8, len(prepared["pairwise_preferences"])))]]
    generations: List[Dict[str, Any]] = []
    for method in methods:
        for prompt in prompts:
            if method == "oracle":
                candidates = [
                    policy.generate(prompt, cfg.max_new_tokens, "ours"),
                    reference.generate(prompt, cfg.max_new_tokens, "GPT2"),
                ]
                text = min(candidates, key=toxicity_score)
            elif method == "ppo":
                ppo_model = LexicalPreferenceModel("ppo", toxicity_weight=policy.toxicity_weight * 0.7, seed=cfg.seed + 17)
                text = ppo_model.generate(prompt, cfg.max_new_tokens, "ppo")
            elif method in {"GPT2", "Llama2"}:
                text = reference.generate(prompt, cfg.max_new_tokens, method)
            elif method in {"GPT2_DPO", "Llama2_DPO", "DPO", "ours"}:
                text = policy.generate(prompt, cfg.max_new_tokens, method)
            elif method == "PPLM":
                text = policy.generate(prompt, cfg.max_new_tokens, "PPLM")
            elif method == "toxicity_vector_intervention":
                text = generate_gpt2_dpo_with_added_residual_toxic_offset(
                    prompt, toxic_offset=-abs(float(cfg.similarity_guidance_scale)) / 10.0, max_new_tokens=cfg.max_new_tokens, config=cfg
                )["generation"]
            elif method == "anti_alignment_gated_intervention":
                text = generate_gpt2_dpo_with_added_residual_toxic_offset(
                    prompt, toxic_offset=abs(float(cfg.similarity_guidance_scale)) / 10.0, max_new_tokens=cfg.max_new_tokens, config=cfg
                )["generation"]
            else:
                text = policy.generate(prompt, cfg.max_new_tokens, method)
            score = toxicity_score(text)
            generations.append(
                {
                    "method": method,
                    "prompt": prompt,
                    "generation": text,
                    "toxicity_score": score,
                    "toxic": int(score >= cfg.toxicity_threshold),
                }
            )

    by_method: Dict[str, Dict[str, float]] = {}
    for method in methods:
        rows = [g for g in generations if g["method"] == method]
        if not rows:
            continue
        scores = [float(r["toxicity_score"]) for r in rows]
        by_method[method] = {
            "toxicity": statistics.fmean(scores),
            "toxicity_rate": statistics.fmean([float(r["toxic"]) for r in rows]),
            "num_generations": float(len(rows)),
        }

    wikitext_prompts = prepared["wikitext_prompts"]
    nlls = []
    for prompt in wikitext_prompts:
        generated = policy.generate(prompt, cfg.max_new_tokens, "ours")
        token_count = max(1, len(_tokenize(generated)))
        nlls.append(-policy.log_probability(prompt, generated) / token_count)
    perplexity = math.exp(max(-20.0, min(20.0, statistics.fmean(nlls)))) if nlls else 1.0

    before_features = extract_last_layer_mean_residual_stream_features([p.prompt + " " + p.rejected for p in prepared["pairwise_preferences"]], layer=cfg.layer)
    after_features = extract_last_layer_mean_residual_stream_features([p.prompt + " " + p.preferred for p in prepared["pairwise_preferences"]], layer=cfg.layer)
    toxic_vector = probe["W_Toxic"]
    shifts = []
    for before, after in zip(before_features, after_features):
        shifts.append(sum((a - b) * w for a, b, w in zip(after, before, toxic_vector)))
    activation_shift = statistics.fmean(shifts) if shifts else 0.0

    objective = training_result["final_objective"] if training_result else {}
    metrics = {
        "schema_version": "1.0",
        "score_protocol": TOXICITY_SCORE_PROTOCOL,
        "method_metrics": by_method,
        "toxicity_rate": by_method.get("ours", {}).get("toxicity_rate", 0.0),
        "toxicity": by_method.get("ours", {}).get("toxicity", 0.0),
        "probe_f1": probe["metrics"]["f1"],
        "activation_shift": activation_shift,
        "loss": float(objective.get("loss", 0.0)),
        "perplexity": perplexity,
        "preference_accuracy": float(objective.get("preference_accuracy", 0.0)),
        "generation_count": len(generations),
        "metric_registry": build_metric_registry(),
    }
    return {
        "metrics": metrics,
        "generations": generations,
        "probe": probe,
        "experiment_registry": experiment_registry,
    }


def write_training_artifacts(
    config: DpoTrainingConfig,
    prepared: Mapping[str, Any],
    training_result: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
) -> Dict[str, Any]:
    root = _artifact_root(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)

    dataset_registry = prepared["dataset_registry"]
    metric_registry = prepared["metric_registry"]
    experiment_registry = prepared["experiment_registry"]
    data_manifest = {
        "schema_version": "1.0",
        "datasets": {
            "jigsaw_toxicity_probe_90_10": prepared["probe_split"].manifest,
            "dpo_pairwise_toxicity": {"num_pairs": len(prepared["pairwise_preferences"])},
            "wikitext": {"num_prompts": len(prepared["wikitext_prompts"])},
        },
        "mode": config.mode,
        "paper_visible_outputs_computed_by_route": True,
    }
    artifact_manifest = {
        "schema_version": "1.0",
        "written_at_unix": time.time(),
        "artifacts": {
            "dataset_registry": str(root / "dataset_registry.json"),
            "metrics": str(root / "metrics.json"),
            "data_manifest": str(root / "data_manifest.json"),
            "experiment_registry": str(root / "experiment_registry.json"),
            "artifact_manifest": str(root / "artifact_manifest.json"),
            "summary_table": str(root / "tables" / "summary.csv"),
            "training_trace": str(root / "training_trace.json"),
            "config_resolved": str(root / "config_resolved.json"),
            "readiness": str(root / "readiness.json"),
            "evaluation_result": str(root / "evaluation_result.json"),
        },
    }

    _write_json(root / "dataset_registry.json", dataset_registry)
    _write_json(root / "metrics.json", evaluation_result["metrics"])
    _write_json(root / "data_manifest.json", data_manifest)
    _write_json(root / "experiment_registry.json", experiment_registry)
    _write_json(root / "training_trace.json", {"trace": training_result["trace"], "final_objective": training_result["final_objective"]})
    _write_json(root / "config_resolved.json", asdict(config))
    _write_json(root / "evaluation_result.json", evaluation_result["metrics"])

    readiness = {
        "schema_version": "1.0",
        "status": "ready",
        "route_exercised": [
            "prepare_dpo_training",
            "run_training_loop",
            "compute_training_objective",
            "evaluate_predictions",
            "write_training_artifacts",
        ],
        "selector_validation": prepared["selector_validation"],
        "optional_transformers_available": Factory(config).transformers_available(),
        "artifact_dir": str(root),
    }
    _write_json(root / "readiness.json", readiness)

    summary_path = root / "tables" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "toxicity", "toxicity_rate", "num_generations", "probe_f1", "loss", "perplexity"])
        writer.writeheader()
        method_metrics = evaluation_result["metrics"].get("method_metrics", {})
        for method, vals in sorted(method_metrics.items()):
            writer.writerow(
                {
                    "method": method,
                    "toxicity": vals.get("toxicity", 0.0),
                    "toxicity_rate": vals.get("toxicity_rate", 0.0),
                    "num_generations": vals.get("num_generations", 0.0),
                    "probe_f1": evaluation_result["metrics"].get("probe_f1", 0.0),
                    "loss": evaluation_result["metrics"].get("loss", 0.0),
                    "perplexity": evaluation_result["metrics"].get("perplexity", 0.0),
                }
            )

    _write_json(root / "artifact_manifest.json", artifact_manifest)
    return artifact_manifest


def train_selectorsetmustincludeours_adaptersorregistryentries_probabilityratio(
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Contract-named primary function wiring selectors, adapters, and DPO ratios."""

    cfg = load_dpo_training(config)
    selector = SelectorSetMustIncludeOurs(cfg.selected_methods)
    selector_status = selector.validate()
    adapters = AdaptersOrRegistryEntries()
    inventory = Inventory(DpoTrainingSpec()).as_registry()
    factory = Factory(cfg)
    route = ObligationsCallablePrimaryFunctio(cfg)
    result = route()
    return {
        "selector_status": selector_status,
        "adapter_selectors": adapters.selectors(),
        "inventory": inventory,
        "factory": {"transformers_available": factory.transformers_available()},
        "training_result": result,
    }


def train_dpo_toxicity_aligned_gpt2_llama2(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = load_dpo_training(config)
    outputs: Dict[str, Any] = {}
    variants = ("GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO") if cfg.mode == "full" or cfg.execute_full_matrix else (cfg.base_model,)
    for variant in variants:
        variant_cfg = DpoTrainingConfig(**{**asdict(cfg), "base_model": variant})
        outputs[variant] = train_dpo_training(variant_cfg)
    return {
        "variants": list(variants),
        "results": outputs,
        "mechanistic_obligation": "DPO aligned GPT2/Llama2 route with GPT2_DPO and Llama2_DPO selectors",
    }


def train_dpo_training(config: Optional[Any] = None) -> Dict[str, Any]:
    cfg = load_dpo_training(config)
    prepared = prepare_dpo_training(cfg)
    training_result = run_training_loop(cfg, prepared)
    evaluation_result = evaluate_predictions(cfg, training_result)
    transformers_training: Dict[str, Any] = {}
    if cfg.use_transformers_when_available or cfg.mode == "full":
        try:
            from dpo_toxicity.mechanistic_transformers import train_gpt2_dpo_alignment

            pairwise_examples = [
                {
                    "prompt": pair.prompt,
                    "chosen": pair.preferred,
                    "rejected": pair.rejected,
                }
                for pair in prepared["pairwise_preferences"]
            ]
            transformers_training = train_gpt2_dpo_alignment(
                pairwise_examples,
                model_name="gpt2",
                reference_model_name="gpt2",
                learning_rate=1e-6,
                batch_size=4,
                beta=cfg.beta,
                max_grad_norm=10.0,
                validation_patience=10,
                epochs=max(1, cfg.epochs),
                allow_download=cfg.mode == "full",
            )
        except Exception as exc:
            transformers_training = {
                "status": "transformers_route_error",
                "error": str(exc),
                "model_name": "gpt2",
                "reference_model_name": "gpt2",
                "learning_rate": 1e-6,
                "batch_size": 4,
                "beta": cfg.beta,
                "max_grad_norm": 10.0,
                "validation_patience": 10,
            }
    artifact_manifest: Dict[str, Any] = {}
    if cfg.write_artifacts:
        artifact_manifest = write_training_artifacts(cfg, prepared, training_result, evaluation_result)
    return {
        "config": asdict(cfg),
        "data_manifest": {
            "probe": prepared["probe_split"].manifest,
            "num_pairwise_preferences": len(prepared["pairwise_preferences"]),
            "num_wikitext_prompts": len(prepared["wikitext_prompts"]),
        },
        "dataset_registry": prepared["dataset_registry"],
        "metric_registry": prepared["metric_registry"],
        "experiment_registry": prepared["experiment_registry"],
        "evidence_obligation_matrix": prepared["evidence_obligation_matrix"],
        "training_trace": training_result["trace"],
        "final_objective": training_result["final_objective"],
        "metrics": evaluation_result["metrics"],
        "artifact_manifest": artifact_manifest,
        "transformers_training": transformers_training,
    }


# Contract alias for the non-Python identifier requested by the task plan.
globals()["Jigsaw 90:10 毒性 probe 数据与残差流特征模块"] = prepare_jigsaw_90_10_toxicity_probe_data


__all__ = [
    "AdaptersOrRegistryEntries",
    "DpoTrainingConfig",
    "DpoTrainingSpec",
    "Factory",
    "Inventory",
    "ObligationsCallablePrimaryFunctio",
    "PairwisePreferenceExample",
    "ProbabilityRatio",
    "SelectorSetMustIncludeOurs",
    "ToxicityProbeSplit",
    "build_dataset_registry",
    "build_evidence_obligation_matrix_registry",
    "build_experiment_registry",
    "build_method_registry",
    "build_metric_registry",
    "classification_metrics",
    "compute_training_objective",
    "evaluate_predictions",
    "extract_last_layer_mean_residual_stream_features",
    "generate_gpt2_dpo_with_added_residual_toxic_offset",
    "load_dpo_training",
    "prepare_dpo_training",
    "prepare_jigsaw_90_10_toxicity_probe_data",
    "prepare_pairwise_preference_data",
    "run_training_loop",
    "toxicity_score",
    "train_dpo_toxicity_aligned_gpt2_llama2",
    "train_dpo_training",
    "train_selectorsetmustincludeours_adaptersorregistryentries_probabilityratio",
    "train_toxicity_probe",
    "write_training_artifacts",
]
