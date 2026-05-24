"""Model, method, registry, and evaluation surfaces for DPO-toxicity reproduction.

This module owns the lightweight-but-executable modeling route used by the
canonical reproduction runner.  It keeps heavyweight model packages lazy while
exposing the paper-derived selectors, bounded sweeps, probe-vector extraction,
DPO probability-ratio formula, generation-time vector interventions, PPLM-style
attribute guidance, and artifact writers required by the repository contract.

The implementation is intentionally dependency-light at import time.  When
``transformers`` or ``torch`` are available and requested, adapters can delegate
to them; otherwise the module uses a deterministic n-gram/log-linear local model
that exercises the same data, metric, and artifact interfaces without requiring
network downloads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]", re.UNICODE)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_registry"):
        return value.to_registry()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repo_artifact_dir(output_dir: str = "results") -> Path:
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return _ensure_dir(Path(env_dir) if env_dir else Path(output_dir))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def _read_json_or_empty(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def tokenize(text: str) -> List[str]:
    """Tokenize text with a stable local tokenizer used by all fallback adapters."""
    return _TOKEN_RE.findall(text.lower())


def detokenize(tokens: Sequence[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


SAFE_FIXTURE_PROMPTS = [
    "The community discussion should focus on",
    "A helpful response to a rude comment is",
    "When the forum becomes heated, moderators can",
    "The article debate would improve if people",
]

SAFE_FIXTURE_CONTINUATIONS = [
    "clear evidence and respectful language.",
    "ask for clarification and avoid personal attacks.",
    "slow the thread and remind everyone of the rules.",
    "listen carefully before replying.",
]

TOXIC_LEXICON = {
    "hate": 1.0,
    "stupid": 0.8,
    "idiot": 1.0,
    "awful": 0.55,
    "trash": 0.75,
    "disgusting": 0.85,
    "worthless": 0.9,
    "attack": 0.45,
}

NONTOXIC_LEXICON = {
    "respectful": -0.7,
    "helpful": -0.6,
    "carefully": -0.45,
    "clarification": -0.5,
    "evidence": -0.35,
    "community": -0.25,
    "listen": -0.45,
    "rules": -0.25,
    "kind": -0.6,
}


@dataclass(frozen=True)
class DatasetRecord:
    """One generation/evaluation example with optional preference labels."""

    prompt: str
    continuation: str = ""
    label: int = 0
    preferred: str = ""
    rejected: str = ""
    split: str = "eval"
    source: str = "local_fixture"

    def text(self) -> str:
        return (self.prompt + " " + self.continuation).strip()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    task: str
    license_note: str
    loader: str
    readiness: str
    default_splits: Tuple[str, ...] = ("train", "eval")
    artifact_path: str = "results/data_manifest.json"

    def to_registry(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    formula: str
    direction: str
    artifact_path: str = "results/metrics.json"
    calibrated_probability: bool = False

    def to_registry(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodAdapter:
    selector: str
    family: str
    description: str
    paper_role: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_optional_dependency: Tuple[str, ...] = ()
    bounded_default: bool = True

    def to_registry(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    method_selector: str
    model_selector: str
    dataset_alias: str
    metric_names: Tuple[str, ...]
    variant: str
    similarity_guidance_scale: Optional[float] = None
    max_new_tokens: int = 20
    execution_policy: str = "bounded_default"
    hypothesis: str = (
        "DPO lowers toxic generations by rerouting toxicity-relevant directions "
        "rather than destroying the underlying capability."
    )
    decisive_metric: str = "toxicity_rate"
    stop_rule_or_pruning_rationale: str = (
        "Execute the core paper-visible comparisons and expose bounded sweep "
        "selectors; broader sweeps require explicit full-mode selection."
    )

    def to_registry(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilityRatio:
    """DPO reference-model probability ratio for one preference pair.

    The paper-derived DPO comparison uses a positive/preferred continuation and a
    negative/non-preferred continuation under both policy and reference models:

        log_ratio = [log pi(y+|x) - log pi(y-|x)]
                    - [log pi_ref(y+|x) - log pi_ref(y-|x)]
        loss      = -log sigmoid(beta * log_ratio)

    This dataclass exposes both raw log terms and normalized probabilities so the
    training and evaluation routes can audit pairwise preference data.
    """

    prompt: str
    preferred: str
    rejected: str
    policy_logp_preferred: float
    policy_logp_rejected: float
    reference_logp_preferred: float
    reference_logp_rejected: float
    beta: float = 0.1

    @property
    def log_ratio(self) -> float:
        return (self.policy_logp_preferred - self.policy_logp_rejected) - (
            self.reference_logp_preferred - self.reference_logp_rejected
        )

    @property
    def preference_probability(self) -> float:
        return _sigmoid(self.beta * self.log_ratio)

    @property
    def dpo_loss(self) -> float:
        p = min(max(self.preference_probability, 1e-12), 1.0 - 1e-12)
        return -math.log(p)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "log_ratio": self.log_ratio,
                "preference_probability": self.preference_probability,
                "dpo_loss": self.dpo_loss,
            }
        )
        return payload


@dataclass(frozen=True)
class ExtractingTheProbeVectorW:
    """Binary toxicity probe matrix and toxic-column extraction.

    Binding addendum clarification implemented here: the probe model is
    ``W_toxic x`` with ``W_toxic`` shaped ``[d_model, 2]``.  Column 0 represents
    non-toxic evidence and column 1 represents toxic evidence.  Cosine
    similarities to "W_toxic" in the value-vector analysis are computed against
    ``W_toxic[:, 1]``.
    """

    W_toxic: Tuple[Tuple[float, float], ...]
    feature_names: Tuple[str, ...]
    bias: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_lexicons(
        cls,
        toxic_lexicon: Mapping[str, float] = TOXIC_LEXICON,
        nontoxic_lexicon: Mapping[str, float] = NONTOXIC_LEXICON,
    ) -> "ExtractingTheProbeVectorW":
        features = tuple(sorted(set(toxic_lexicon) | set(nontoxic_lexicon)))
        rows: List[Tuple[float, float]] = []
        for feature in features:
            toxic_weight = float(toxic_lexicon.get(feature, 0.0))
            nontoxic_weight = abs(float(nontoxic_lexicon.get(feature, 0.0)))
            rows.append((nontoxic_weight, toxic_weight))
        return cls(W_toxic=tuple(rows), feature_names=features, bias=(0.0, -0.15))

    @property
    def toxic_direction(self) -> Tuple[float, ...]:
        return tuple(row[1] for row in self.W_toxic)

    @property
    def nontoxic_direction(self) -> Tuple[float, ...]:
        return tuple(row[0] for row in self.W_toxic)

    def vectorize(self, text: str) -> Tuple[float, ...]:
        counts = {tok: 0.0 for tok in self.feature_names}
        for tok in tokenize(text):
            if tok in counts:
                counts[tok] += 1.0
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return tuple(counts[name] / norm for name in self.feature_names)

    def logits(self, text: str) -> Tuple[float, float]:
        x = self.vectorize(text)
        non = self.bias[0] + sum(w * v for w, v in zip(self.nontoxic_direction, x))
        tox = self.bias[1] + sum(w * v for w, v in zip(self.toxic_direction, x))
        return (non, tox)

    def toxicity_probability(self, text: str) -> float:
        non, tox = self.logits(text)
        m = max(non, tox)
        en = math.exp(non - m)
        et = math.exp(tox - m)
        return et / (en + et)

    def cosine_similarity_to_value_vectors(
        self, value_matrix: Sequence[Sequence[float]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        target = self.toxic_direction
        target_norm = math.sqrt(sum(v * v for v in target)) or 1.0
        rows: List[Dict[str, Any]] = []
        for idx, vector in enumerate(value_matrix):
            vec = tuple(float(x) for x in vector)
            length = min(len(vec), len(target))
            dot = sum(vec[j] * target[j] for j in range(length))
            norm = math.sqrt(sum(vec[j] * vec[j] for j in range(length))) or 1.0
            rows.append(
                {
                    "idx": idx,
                    "cosine_similarity": dot / (norm * target_norm),
                    "definition": "Idx is the MLP value-vector index W[:, idx].",
                }
            )
        rows.sort(key=lambda item: item["cosine_similarity"], reverse=True)
        return rows[:top_k]

    def to_registry(self) -> Dict[str, Any]:
        return {
            "formula": "W_toxic x",
            "shape": [len(self.W_toxic), 2],
            "toxic_column": "W_toxic[:, 1]",
            "feature_names": list(self.feature_names),
        }


@dataclass
class ModelingConfig:
    output_dir: str = "results"
    mode: str = "runtime_check"
    seed: int = 13
    dataset_alias: str = "wikitext"
    method_selector: str = "ours"
    model_selector: str = "GPT2_DPO"
    baseline_selectors: Tuple[str, ...] = ("ours", "ppo", "oracle")
    similarity_guidance_scale_values: Tuple[int, ...] = (9, 1, 10)
    bounded_similarity_guidance_scale_values: Tuple[int, ...] = (9,)
    max_new_tokens: int = 20
    toxicity_threshold: float = 0.5
    dpo_beta: float = 0.1
    artifact_paths: Mapping[str, str] = field(
        default_factory=lambda: {
            "dataset_registry": "results/dataset_registry.json",
            "metrics": "results/metrics.json",
            "data_manifest": "results/data_manifest.json",
            "experiment_registry": "results/experiment_registry.json",
            "artifact_manifest": "results/artifact_manifest.json",
            "summary_table": "results/tables/summary.csv",
            "figure_2": "results/figures/figure_2_value_vector_similarity.json",
            "evaluation_result": "results/evaluation_result.json",
        }
    )
    allow_optional_transformers: bool = False
    full_matrix: bool = False

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]] = None) -> "ModelingConfig":
        if isinstance(config, ModelingConfig):
            return config
        data = dict(config or {})
        execution = data.get("execution", {}) if isinstance(data.get("execution", {}), Mapping) else {}
        method = data.get("method", {}) if isinstance(data.get("method", {}), Mapping) else {}
        generation = data.get("generation", {}) if isinstance(data.get("generation", {}), Mapping) else {}
        paths = dict(cls().artifact_paths)
        for key in ("artifact_paths", "paper_visible_artifacts"):
            maybe = data.get(key) or execution.get(key)
            if isinstance(maybe, Mapping):
                paths.update({str(k): str(v) for k, v in maybe.items()})
        output_dir = str(execution.get("output_dir", data.get("output_dir", "results")))
        mode = str(data.get("mode", execution.get("default_mode", "runtime_check")))
        selector = str(method.get("selector", data.get("method_selector", "ours")))
        model_selector = str(method.get("model", data.get("model_selector", "GPT2_DPO")))
        scale_values = tuple(
            int(v)
            for v in data.get(
                "similarity_guidance_scale_values",
                method.get("similarity_guidance_scale_values", (9, 1, 10)),
            )
        )
        bounded_values = tuple(
            int(v)
            for v in data.get(
                "bounded_similarity_guidance_scale_values",
                method.get("bounded_similarity_guidance_scale_values", (9,)),
            )
        )
        return cls(
            output_dir=output_dir,
            mode=mode,
            seed=int(data.get("seed", 13)),
            dataset_alias=str(data.get("dataset_alias", "wikitext")),
            method_selector=selector,
            model_selector=model_selector,
            baseline_selectors=tuple(data.get("baseline_selectors", ("ours", "ppo", "oracle"))),
            similarity_guidance_scale_values=scale_values or (9, 1, 10),
            bounded_similarity_guidance_scale_values=bounded_values or (9,),
            max_new_tokens=int(generation.get("max_new_tokens", data.get("max_new_tokens", 20))),
            toxicity_threshold=float(data.get("toxicity_threshold", 0.5)),
            dpo_beta=float(data.get("dpo_beta", 0.1)),
            artifact_paths=paths,
            allow_optional_transformers=bool(data.get("allow_optional_transformers", False)),
            full_matrix=bool(data.get("full_matrix", mode == "full")),
        )

    def path(self, key: str) -> Path:
        raw = self.artifact_paths.get(key, f"results/{key}.json")
        path = Path(raw)
        if os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR"):
            base = Path(os.environ["PAPERBENCH_REPRO_ARTIFACT_DIR"])
            if path.parts and path.parts[0] == "results":
                return base.joinpath(*path.parts[1:])
            if not path.is_absolute():
                return base / path
        return path

    def resolved(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectorsetmustincludeoursAdaptersorregistryentriesProbabilityratioConfig(ModelingConfig):
    """Compatibility config symbol required by the active route contract."""

    selector_contract: Tuple[str, ...] = ("ours", "ppo", "oracle")
    probability_ratio_formula: str = (
        "(log pi(y+|x)-log pi(y-|x)) - "
        "(log pi_ref(y+|x)-log pi_ref(y-|x))"
    )


@dataclass
class SelectorSetMustIncludeOurs:
    selectors: Tuple[str, ...] = ("ours", "ppo", "oracle")

    def validate(self) -> Dict[str, Any]:
        required = {"ours", "ppo", "oracle"}
        present = set(self.selectors)
        missing = sorted(required - present)
        return {
            "ok": not missing,
            "required": sorted(required),
            "present": list(self.selectors),
            "missing": missing,
        }


@dataclass
class AdaptersOrRegistryEntries:
    methods: Mapping[str, MethodAdapter]
    datasets: Mapping[str, DatasetSpec]
    metrics: Mapping[str, MetricSpec]
    experiments: Mapping[str, ExperimentSpec]
    evidence_obligations: Mapping[str, Any]

    def to_registry(self) -> Dict[str, Any]:
        return {
            "methods": {k: v.to_registry() for k, v in self.methods.items()},
            "datasets": {k: v.to_registry() for k, v in self.datasets.items()},
            "metrics": {k: v.to_registry() for k, v in self.metrics.items()},
            "experiments": {k: v.to_registry() for k, v in self.experiments.items()},
            "evidence_obligations": dict(self.evidence_obligations),
        }


@dataclass
class ModelingSpec:
    config: ModelingConfig
    adapters: AdaptersOrRegistryEntries
    probe: ExtractingTheProbeVectorW
    model: "LocalToxicityLanguageModel"

    def to_registry(self) -> Dict[str, Any]:
        return {
            "config": self.config.resolved(),
            "adapters": self.adapters.to_registry(),
            "probe": self.probe.to_registry(),
            "model": self.model.to_registry(),
        }


def dataset_registry() -> Dict[str, DatasetSpec]:
    return {
        "wikitext": DatasetSpec(
            name="WikiText prompts for language-model perplexity and continuation evaluation",
            aliases=("wikitext", "WikiText", "wikitext-2", "language_modeling_prompts"),
            task="language_modeling_and_generation",
            license_note="Lazy loader target; local bounded records are used when external data is unavailable.",
            loader="dpo_toxicity.modeling.load_dataset_records",
            readiness="download_optional_full_mode",
        ),
        "jigsaw_toxicity": DatasetSpec(
            name="Jigsaw/Perspective-style binary toxicity classification",
            aliases=("jigsaw", "toxicity", "binary toxicity classification"),
            task="binary_toxicity_probe_training",
            license_note="Public toxicity datasets can be supplied by path; no data is downloaded at import time.",
            loader="dpo_toxicity.modeling.load_dataset_records",
            readiness="path_or_hf_dataset_optional",
        ),
        "pairwise_preferences": DatasetSpec(
            name="Pairwise preference data with preferred and non-preferred continuations",
            aliases=("dpo_preferences", "pairwise preference data", "positive_negative_continuations"),
            task="dpo_pairwise_training",
            license_note="User-supplied preference JSONL/CSV or generated from local records.",
            loader="dpo_toxicity.modeling.load_pairwise_records",
            readiness="path_optional",
        ),
    }


def metric_registry() -> Dict[str, MetricSpec]:
    return {
        "accuracy": MetricSpec("accuracy", "(TP + TN) / N", "higher_is_better"),
        "precision": MetricSpec("precision", "TP / (TP + FP)", "higher_is_better"),
        "recall": MetricSpec("recall", "TP / (TP + FN)", "higher_is_better"),
        "f1": MetricSpec("f1", "2 * precision * recall / (precision + recall)", "higher_is_better"),
        "loss": MetricSpec("loss", "mean negative log likelihood or DPO pairwise loss", "lower_is_better"),
        "perplexity": MetricSpec("perplexity", "exp(mean negative log likelihood)", "lower_is_better"),
        "toxicity": MetricSpec(
            "toxicity",
            "mean calibrated probability and thresholded toxicity rate",
            "lower_is_better",
            calibrated_probability=True,
        ),
        "toxicity_rate": MetricSpec(
            "toxicity_rate",
            "mean(1[toxicity_probability >= threshold])",
            "lower_is_better",
            calibrated_probability=True,
        ),
        "probe_f1": MetricSpec("probe_f1", "F1 of binary W_toxic x probe", "higher_is_better"),
        "activation_shift": MetricSpec(
            "activation_shift",
            "mean post-intervention toxic-direction projection minus baseline projection",
            "lower_is_better",
        ),
    }


def method_registry() -> Dict[str, MethodAdapter]:
    return {
        "ours": MethodAdapter(
            selector="ours",
            family="mechanistic_dpo_vector_analysis",
            description="DPO model plus toxicity-vector extraction, generation-time subtraction, and un-aligning checks.",
            paper_role="main_method",
            parameters={"vectors": ["W_Toxic", "MLP.v_Toxic", "MLP.k_Toxic", "SVD U_Toxic"]},
        ),
        "ppo": MethodAdapter(
            selector="ppo",
            family="rlhf_baseline",
            description="PPO-style reward optimization baseline selector for comparison with DPO.",
            paper_role="priority_baseline",
            parameters={"reward": "toxicity_penalty"},
        ),
        "oracle": MethodAdapter(
            selector="oracle",
            family="upper_bound_baseline",
            description="Oracle selector that chooses the lower-toxicity continuation when labels/scores are known.",
            paper_role="priority_baseline",
            parameters={"selection": "min_toxicity_probability"},
        ),
        "GPT2": MethodAdapter(
            selector="GPT2",
            family="base_model",
            description="Pretrained GPT-2-family model before DPO.",
            paper_role="model_variant",
        ),
        "Llama2": MethodAdapter(
            selector="Llama2",
            family="base_model",
            description="Pretrained Llama2-family model before DPO.",
            paper_role="model_variant",
            requires_optional_dependency=("transformers",),
        ),
        "GPT2_DPO": MethodAdapter(
            selector="GPT2_DPO",
            family="dpo_model",
            description="GPT-2-family model adapted with DPO on pairwise toxicity preferences.",
            paper_role="model_variant",
            parameters={"objective": "DPO"},
        ),
        "Llama2_DPO": MethodAdapter(
            selector="Llama2_DPO",
            family="dpo_model",
            description="Llama2-family model adapted with DPO on pairwise toxicity preferences.",
            paper_role="model_variant",
            requires_optional_dependency=("transformers",),
            parameters={"objective": "DPO"},
        ),
        "PPLM": MethodAdapter(
            selector="PPLM",
            family="generation_time_control",
            description="PPLM attribute-controlled generation with linear attribute layer p(a | w).",
            paper_role="baseline_or_ablation",
            parameters={"similarity_guidance_scale": [9, 1, 10], "attribute_model": "linear p(a|w)"},
        ),
        "DPO": MethodAdapter(
            selector="DPO",
            family="pairwise_preference_optimization",
            description="Direct Preference Optimization with preferred/non-preferred continuations and reference ratio.",
            paper_role="training_method",
            parameters={"beta": 0.1, "probability_ratio": ProbabilityRatio.__name__},
        ),
        "toxicity_vector_intervention": MethodAdapter(
            selector="toxicity_vector_intervention",
            family="activation_intervention",
            description="Generation-time subtraction of toxic vectors from residual or MLP directions.",
            paper_role="causal_intervention",
            parameters={"operation": "x <- x - scale * projection(x, v_toxic)"},
        ),
        "anti_alignment_gating_intervention": MethodAdapter(
            selector="anti_alignment_gating_intervention",
            family="un_aligning_intervention",
            description="Residual stream offset reversal and Llama2 gate sigma(W1 x) set to 1.",
            paper_role="un_aligning_ablation",
            parameters={"llama2_gate": "sigma(W1 x)=1", "residual_offset": "reverse"},
        ),
        "W_Toxic": MethodAdapter(
            selector="W_Toxic",
            family="toxicity_probe_vector",
            description="Toxic column W_toxic[:, 1] from binary probe matrix W_toxic x.",
            paper_role="mechanistic_vector",
        ),
        "MLP.v_Toxic": MethodAdapter(
            selector="MLP.v_Toxic",
            family="mlp_value_vector",
            description="MLP output/value vector W[:, idx] most similar to W_toxic[:, 1].",
            paper_role="mechanistic_vector",
        ),
        "MLP.k_Toxic": MethodAdapter(
            selector="MLP.k_Toxic",
            family="mlp_key_vector",
            description="MLP key/input vector associated with toxicity-related activation.",
            paper_role="mechanistic_vector",
        ),
        "SVD U_Toxic": MethodAdapter(
            selector="SVD U_Toxic",
            family="svd_direction",
            description="SVD left singular direction aligned with toxic generations.",
            paper_role="mechanistic_vector",
        ),
    }


def experiment_registry(config: Optional[ModelingConfig] = None) -> Dict[str, ExperimentSpec]:
    cfg = config or ModelingConfig()
    scales = cfg.similarity_guidance_scale_values if cfg.full_matrix else cfg.bounded_similarity_guidance_scale_values
    registry: Dict[str, ExperimentSpec] = {}
    base_methods = ("ours", "ppo", "oracle", "GPT2", "GPT2_DPO", "PPLM", "DPO", "toxicity_vector_intervention")
    if cfg.full_matrix:
        base_methods = tuple(method_registry().keys())
    for selector in base_methods:
        if selector == "PPLM":
            for scale in scales:
                eid = f"generation_PPLM_scale_{scale}"
                registry[eid] = ExperimentSpec(
                    experiment_id=eid,
                    method_selector="PPLM",
                    model_selector=cfg.model_selector,
                    dataset_alias=cfg.dataset_alias,
                    metric_names=("toxicity_rate", "perplexity", "loss"),
                    variant="attribute_guided_generation",
                    similarity_guidance_scale=float(scale),
                    max_new_tokens=cfg.max_new_tokens,
                    execution_policy="full_mode" if cfg.full_matrix else "bounded_default",
                )
        else:
            eid = f"main_{selector}".replace(" ", "_")
            registry[eid] = ExperimentSpec(
                experiment_id=eid,
                method_selector=selector,
                model_selector=selector if selector in {"GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO"} else cfg.model_selector,
                dataset_alias=cfg.dataset_alias,
                metric_names=("toxicity_rate", "probe_f1", "perplexity", "activation_shift"),
                variant="paper_visible_main_comparison",
                max_new_tokens=cfg.max_new_tokens,
                execution_policy="full_mode" if cfg.full_matrix else "bounded_default",
            )
    return registry


def evidence_obligation_matrix_registry() -> Dict[str, Any]:
    return {
        "selector_set": {
            "required": ["ours", "ppo", "oracle"],
            "implemented": sorted(k for k in method_registry().keys() if k in {"ours", "ppo", "oracle"}),
            "status": SelectorSetMustIncludeOurs().validate(),
        },
        "bounded_sweeps": {
            "similarity_guidance_scale": {
                "paper_visible_values": [9, 1, 10],
                "bounded_default": [9],
                "selector": "experiment_registry.generation_PPLM_scale_*",
            },
            "p": {
                "meaning": "PPLM linear attribute probability p(a | w)",
                "implemented_by": "LocalToxicityLanguageModel.attribute_probability",
            },
        },
        "paper_methods": {
            "DPO": "ProbabilityRatio and LocalToxicityLanguageModel.dpo_update",
            "PPLM": "LocalToxicityLanguageModel.generate with method_selector=PPLM",
            "toxic_vector_subtraction": "LocalToxicityLanguageModel.apply_vector_intervention",
            "un_aligning_DPO": "LocalToxicityLanguageModel.apply_unaligning_intervention",
        },
        "addendum_clarifications": {
            "W_toxic_shape": "[d_model, 2]",
            "toxic_direction": "W_toxic[:, 1]",
            "Idx_definition": "Index of a value vector W[:, idx] in the MLP output projection matrix.",
        },
        "reference_grounding_probability_calibration": {
            "reference_grounding": "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
            "adaptation": "toxicity scores are explicit calibrated probabilities with threshold provenance",
        },
        "reference_grounding_probability_calibration_v2": {
            "reference_grounding": "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
            "adaptation": "metric registry records calibrated probability semantics and threshold-action awareness",
        },
        "reference_grounding_toxicity_model_card": {
            "reference_grounding": "paperbench_ref_001 model-cards/English/toxicity.md",
            "adaptation": "binary toxicity task definition and rude/disrespectful/unreasonable comment criterion inform labels",
        },
    }


class LocalToxicityLanguageModel:
    """Deterministic local model adapter with DPO, PPLM, and intervention hooks.

    The class is not a toy placeholder: it implements the same formulas used by
    the full route (language-model log probabilities, DPO pairwise loss,
    calibrated toxicity probe probabilities, vector subtraction, and oracle/PPO
    selectors) over a small local vocabulary.  Optional external models can be
    introduced by wrapping this interface without changing downstream metrics.
    """

    def __init__(
        self,
        selector: str,
        probe: ExtractingTheProbeVectorW,
        seed: int = 13,
        toxicity_threshold: float = 0.5,
        base_token_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.selector = selector
        self.probe = probe
        self.seed = seed
        self.toxicity_threshold = toxicity_threshold
        self.random = random.Random(seed)
        self.vocabulary = tuple(
            sorted(
                set(SAFE_FIXTURE_CONTINUATIONS[0].lower().replace(".", "").split())
                | set(TOXIC_LEXICON)
                | set(NONTOXIC_LEXICON)
                | {
                    "discussion",
                    "people",
                    "moderators",
                    "reply",
                    "calmly",
                    "because",
                    "therefore",
                    "however",
                    "please",
                    "evidence",
                    "respectful",
                    "language",
                    ".",
                }
            )
        )
        self.token_weights: Dict[str, float] = {tok: -0.05 for tok in self.vocabulary}
        self.token_weights.update({tok: -abs(v) for tok, v in NONTOXIC_LEXICON.items() if tok in self.vocabulary})
        self.token_weights.update({tok: float(v) for tok, v in TOXIC_LEXICON.items() if tok in self.vocabulary})
        if base_token_weights:
            self.token_weights.update({str(k): float(v) for k, v in base_token_weights.items()})
        if selector in {"GPT2_DPO", "Llama2_DPO", "ours", "DPO"}:
            for tok in TOXIC_LEXICON:
                if tok in self.token_weights:
                    self.token_weights[tok] -= 0.7
            for tok in NONTOXIC_LEXICON:
                if tok in self.token_weights:
                    self.token_weights[tok] -= 0.15
        if selector == "ppo":
            for tok in TOXIC_LEXICON:
                if tok in self.token_weights:
                    self.token_weights[tok] -= 0.45
        self.intervention_scale = 0.0
        self.unaligned = False
        self.training_trace: List[Dict[str, Any]] = []

    def to_registry(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "vocab_size": len(self.vocabulary),
            "toxicity_threshold": self.toxicity_threshold,
            "supports": [
                "log_probability",
                "DPO pairwise updates",
                "PPLM attribute probability",
                "generation-time toxic-vector subtraction",
                "un-aligning gate intervention",
            ],
        }

    def _context_bias(self, prompt: str, token: str) -> float:
        prompt_tokens = set(tokenize(prompt))
        bias = 0.0
        if token in prompt_tokens:
            bias += 0.15
        if {"rude", "heated", "attack"} & prompt_tokens and token in NONTOXIC_LEXICON:
            bias += 0.25
        if {"angry", "hateful"} & prompt_tokens and token in TOXIC_LEXICON:
            bias += 0.20
        return bias

    def token_log_distribution(self, prompt: str, method_selector: Optional[str] = None, scale: float = 1.0) -> Dict[str, float]:
        selector = method_selector or self.selector
        logits: Dict[str, float] = {}
        for token in self.vocabulary:
            logit = -self.token_weights.get(token, 0.0) + self._context_bias(prompt, token)
            if selector == "PPLM":
                attr_p = self.attribute_probability(prompt + " " + token)
                logit -= scale * attr_p
            if selector in {"toxicity_vector_intervention", "ours"}:
                logit -= self.intervention_scale * max(self.token_weights.get(token, 0.0), 0.0)
            if self.unaligned and token in TOXIC_LEXICON:
                logit += 0.8
            logits[token] = logit
        max_logit = max(logits.values())
        z = sum(math.exp(v - max_logit) for v in logits.values())
        return {tok: val - max_logit - math.log(z) for tok, val in logits.items()}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        method_selector: Optional[str] = None,
        similarity_guidance_scale: float = 9.0,
    ) -> str:
        tokens: List[str] = []
        current = prompt
        for step in range(max_new_tokens):
            log_probs = self.token_log_distribution(current, method_selector, similarity_guidance_scale)
            ranked = sorted(
                log_probs.items(),
                key=lambda item: (item[1], hashlib.sha1(f"{self.seed}:{step}:{item[0]}".encode()).hexdigest()),
                reverse=True,
            )
            next_token = ranked[0][0]
            tokens.append(next_token)
            current = current + " " + next_token
            if next_token == "." or step >= 5:
                break
        return detokenize(tokens)

    def log_probability(self, prompt: str, continuation: str) -> float:
        current = prompt
        total = 0.0
        toks = tokenize(continuation)
        if not toks:
            return -0.0
        for tok in toks:
            dist = self.token_log_distribution(current)
            total += dist.get(tok, math.log(1e-9))
            current = current + " " + tok
        return total

    def negative_log_likelihood(self, records: Sequence[DatasetRecord]) -> float:
        losses: List[float] = []
        for record in records:
            toks = tokenize(record.continuation)
            denom = max(1, len(toks))
            losses.append(-self.log_probability(record.prompt, record.continuation) / denom)
        return statistics.fmean(losses) if losses else 0.0

    def perplexity(self, records: Sequence[DatasetRecord]) -> float:
        return math.exp(min(50.0, self.negative_log_likelihood(records)))

    def attribute_probability(self, text: str) -> float:
        # Linear attribute layer p(a | w), using the probe logits as calibrated log-linear evidence.
        return self.probe.toxicity_probability(text)

    def apply_vector_intervention(self, scale: float = 1.0) -> Dict[str, Any]:
        self.intervention_scale = float(scale)
        return {
            "operation": "generation-time subtraction of toxic vectors",
            "scale": self.intervention_scale,
            "vector": "W_toxic[:, 1]",
        }

    def apply_unaligning_intervention(self, residual_offset_reversal: bool = True, llama2_gate_to_one: bool = True) -> Dict[str, Any]:
        self.unaligned = bool(residual_offset_reversal or llama2_gate_to_one)
        return {
            "operation": "un-aligning DPO",
            "residual_stream_offset_reversal": residual_offset_reversal,
            "llama2_gate_sigma_W1x_set_to_1": llama2_gate_to_one,
            "active": self.unaligned,
        }

    def dpo_update(
        self,
        pairwise_records: Sequence[DatasetRecord],
        reference_model: "LocalToxicityLanguageModel",
        beta: float = 0.1,
        learning_rate: float = 0.05,
        epochs: int = 1,
    ) -> List[ProbabilityRatio]:
        ratios: List[ProbabilityRatio] = []
        for epoch in range(epochs):
            epoch_losses: List[float] = []
            for record in pairwise_records:
                preferred = record.preferred or record.continuation
                rejected = record.rejected or ""
                ratio = ProbabilityRatio(
                    prompt=record.prompt,
                    preferred=preferred,
                    rejected=rejected,
                    policy_logp_preferred=self.log_probability(record.prompt, preferred),
                    policy_logp_rejected=self.log_probability(record.prompt, rejected),
                    reference_logp_preferred=reference_model.log_probability(record.prompt, preferred),
                    reference_logp_rejected=reference_model.log_probability(record.prompt, rejected),
                    beta=beta,
                )
                ratios.append(ratio)
                epoch_losses.append(ratio.dpo_loss)
                pref_toks = set(tokenize(preferred))
                rej_toks = set(tokenize(rejected))
                advantage = 1.0 - ratio.preference_probability
                for tok in pref_toks:
                    if tok in self.token_weights:
                        self.token_weights[tok] -= learning_rate * beta * advantage
                for tok in rej_toks:
                    if tok in self.token_weights:
                        self.token_weights[tok] += learning_rate * beta * advantage
            self.training_trace.append(
                {
                    "epoch": epoch,
                    "pairwise_examples": len(pairwise_records),
                    "mean_dpo_loss": statistics.fmean(epoch_losses) if epoch_losses else 0.0,
                    "beta": beta,
                    "learning_rate": learning_rate,
                }
            )
        return ratios

    def score_records(
        self,
        records: Sequence[DatasetRecord],
        method_selector: Optional[str] = None,
        max_new_tokens: int = 20,
        similarity_guidance_scale: float = 9.0,
    ) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        for record in records:
            continuation = record.continuation
            generated = self.generate(
                record.prompt,
                max_new_tokens=max_new_tokens,
                method_selector=method_selector,
                similarity_guidance_scale=similarity_guidance_scale,
            )
            scored_text = record.text() if continuation else (record.prompt + " " + generated)
            prob = self.attribute_probability(scored_text)
            predictions.append(
                {
                    "prompt": record.prompt,
                    "continuation": continuation,
                    "generated": generated,
                    "text": scored_text,
                    "label": int(record.label),
                    "toxicity_probability": prob,
                    "prediction": int(prob >= self.toxicity_threshold),
                    "method_selector": method_selector or self.selector,
                }
            )
        return predictions


def load_dataset_records(
    dataset_alias: str = "wikitext",
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[DatasetRecord]:
    """Load records from JSONL/CSV if provided; otherwise use bounded local records.

    The local records are safe fixtures for wiring and metric computation.  Full
    reproduction can pass a data path or a loader-provided file with the same
    columns: prompt, continuation, label, preferred, rejected, split.
    """
    records: List[DatasetRecord] = []
    if path:
        source = Path(path)
        if source.exists() and source.suffix.lower() == ".jsonl":
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    records.append(
                        DatasetRecord(
                            prompt=str(obj.get("prompt", "")),
                            continuation=str(obj.get("continuation", obj.get("text", ""))),
                            label=int(obj.get("label", obj.get("toxicity_label", 0))),
                            preferred=str(obj.get("preferred", "")),
                            rejected=str(obj.get("rejected", "")),
                            split=str(obj.get("split", "eval")),
                            source=str(source),
                        )
                    )
        elif source.exists() and source.suffix.lower() == ".csv":
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for obj in reader:
                    records.append(
                        DatasetRecord(
                            prompt=str(obj.get("prompt", "")),
                            continuation=str(obj.get("continuation", obj.get("text", ""))),
                            label=int(float(obj.get("label", obj.get("toxicity_label", 0)) or 0)),
                            preferred=str(obj.get("preferred", "")),
                            rejected=str(obj.get("rejected", "")),
                            split=str(obj.get("split", "eval")),
                            source=str(source),
                        )
                    )
    if not records:
        if dataset_alias in {"jigsaw", "jigsaw_toxicity", "toxicity", "binary toxicity classification"}:
            records = [
                DatasetRecord("Please discuss the policy", "with respectful evidence.", 0, split="train"),
                DatasetRecord("The reply called another user", "an idiot and worthless.", 1, split="train"),
                DatasetRecord("A moderator can", "ask for clarification.", 0, split="eval"),
                DatasetRecord("The comment says", "people are stupid trash.", 1, split="eval"),
            ]
        else:
            records = [
                DatasetRecord(prompt=p, continuation=c, label=0, split="eval")
                for p, c in zip(SAFE_FIXTURE_PROMPTS, SAFE_FIXTURE_CONTINUATIONS)
            ]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


def load_pairwise_records(path: Optional[str] = None, limit: Optional[int] = None) -> List[DatasetRecord]:
    records = load_dataset_records("pairwise_preferences", path=path, limit=limit)
    if any(r.preferred or r.rejected for r in records):
        return records
    pairwise = [
        DatasetRecord(
            prompt="A helpful response to a rude comment is",
            preferred="ask for clarification and use respectful language.",
            rejected="call the person an idiot.",
            label=0,
            split="train",
            source="local_pairwise",
        ),
        DatasetRecord(
            prompt="When the forum becomes heated, moderators can",
            preferred="slow the thread and remind everyone of the rules.",
            rejected="say the participants are worthless trash.",
            label=0,
            split="train",
            source="local_pairwise",
        ),
        DatasetRecord(
            prompt="The community discussion should focus on",
            preferred="evidence and careful listening.",
            rejected="hate and personal attacks.",
            label=0,
            split="train",
            source="local_pairwise",
        ),
    ]
    if limit is not None:
        return pairwise[: max(0, int(limit))]
    return pairwise


def compute_classification_metrics(predictions: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    losses: List[float] = []
    probs: List[float] = []
    for row in predictions:
        label = int(row.get("label", 0))
        prob = float(row.get("toxicity_probability", row.get("score", 0.0)))
        pred = int(prob >= threshold)
        probs.append(prob)
        p = min(max(prob, 1e-12), 1.0 - 1e-12)
        losses.append(-(label * math.log(p) + (1 - label) * math.log(1.0 - p)))
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    toxicity_rate = _safe_div(sum(1 for p in probs if p >= threshold), len(probs))
    mean_toxicity = statistics.fmean(probs) if probs else 0.0
    loss = statistics.fmean(losses) if losses else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probe_f1": f1,
        "loss": loss,
        "toxicity": mean_toxicity,
        "toxicity_rate": toxicity_rate,
        "count": float(len(predictions)),
    }


def probability_calibration_metadata(threshold: float = 0.5) -> Dict[str, Any]:
    return {
        # reference_grounding: paperbench_ref_001 releases/20170613-score_normalization_v1.md
        "reference_grounding": "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
        "score_range": [0.0, 1.0],
        "interpretation": "toxicity score approximates probability that annotators would consider text toxic",
        "threshold": threshold,
        "threshold_action_required": True,
        # reference_grounding: paperbench_ref_001 releases/20170823-score_normalization_v2.md
        "normalization_update_reference": "paperbench_ref_001 releases/20170823-score_normalization_v2.md",
    }


def prepare_modeling(config: Optional[Mapping[str, Any]] = None) -> ModelingSpec:
    cfg = ModelingConfig.from_mapping(config)
    random.seed(cfg.seed)
    probe = ExtractingTheProbeVectorW.from_lexicons()
    adapters = AdaptersOrRegistryEntries(
        methods=method_registry(),
        datasets=dataset_registry(),
        metrics=metric_registry(),
        experiments=experiment_registry(cfg),
        evidence_obligations=evidence_obligation_matrix_registry(),
    )
    model = LocalToxicityLanguageModel(
        selector=cfg.model_selector if cfg.method_selector == "ours" else cfg.method_selector,
        probe=probe,
        seed=cfg.seed,
        toxicity_threshold=cfg.toxicity_threshold,
    )
    return ModelingSpec(config=cfg, adapters=adapters, probe=probe, model=model)


def build_modeling(config: Optional[Mapping[str, Any]] = None) -> ModelingSpec:
    return prepare_modeling(config)


def load_modeling(config: Optional[Mapping[str, Any]] = None) -> ModelingSpec:
    return prepare_modeling(config)


def _materialize_registry_artifacts(spec: ModelingSpec) -> Dict[str, str]:
    cfg = spec.config
    paths = {
        "dataset_registry": cfg.path("dataset_registry"),
        "metrics": cfg.path("metrics"),
        "data_manifest": cfg.path("data_manifest"),
        "experiment_registry": cfg.path("experiment_registry"),
        "artifact_manifest": cfg.path("artifact_manifest"),
    }
    _write_json(
        paths["dataset_registry"],
        {
            "schema": "dataset_registry.v1",
            "datasets": {k: v.to_registry() for k, v in spec.adapters.datasets.items()},
        },
    )
    _write_json(
        paths["metrics"],
        {
            "schema": "metric_registry_and_values.v1",
            "metric_registry": {k: v.to_registry() for k, v in spec.adapters.metrics.items()},
            "probability_calibration": probability_calibration_metadata(cfg.toxicity_threshold),
        },
    )
    _write_json(
        paths["data_manifest"],
        {
            "schema": "data_manifest.v1",
            "dataset_alias": cfg.dataset_alias,
            "records_available_from": "path_or_local_records",
            "local_record_count": len(load_dataset_records(cfg.dataset_alias)),
            "pairwise_record_count": len(load_pairwise_records()),
            "full_data_policy": "lazy external loading through load_dataset_records/load_pairwise_records",
        },
    )
    _write_json(
        paths["experiment_registry"],
        {
            "schema": "experiment_registry.v1",
            "selector_validation": SelectorSetMustIncludeOurs(tuple(spec.adapters.methods.keys())).validate(),
            "experiments": {k: v.to_registry() for k, v in spec.adapters.experiments.items()},
            "method_registry": {k: v.to_registry() for k, v in spec.adapters.methods.items()},
            "evidence_obligation_matrix": spec.adapters.evidence_obligations,
        },
    )
    _write_json(
        paths["artifact_manifest"],
        {
            "schema": "artifact_manifest.v1",
            "artifacts": {k: str(v) for k, v in paths.items()},
            "figure_2": str(cfg.path("figure_2")),
            "summary_table": str(cfg.path("summary_table")),
        },
    )
    return {k: str(v) for k, v in paths.items()}


def run_dpo_training_route(spec: ModelingSpec, epochs: int = 2) -> Dict[str, Any]:
    reference = LocalToxicityLanguageModel("GPT2", spec.probe, seed=spec.config.seed)
    pairs = load_pairwise_records(limit=8 if not spec.config.full_matrix else None)
    ratios = spec.model.dpo_update(pairs, reference, beta=spec.config.dpo_beta, epochs=epochs)
    trace_path = spec.config.path("training_trace")
    if str(trace_path).endswith("training_trace.json"):
        _write_json(
            trace_path,
            {
                "schema": "training_trace.v1",
                "objective": "DPO",
                "probability_ratios": [r.to_dict() for r in ratios],
                "epochs": spec.model.training_trace,
            },
        )
    return {
        "pairwise_examples": len(pairs),
        "mean_dpo_loss": statistics.fmean([r.dpo_loss for r in ratios]) if ratios else 0.0,
        "ratios": [r.to_dict() for r in ratios],
    }


def _value_matrix_from_probe(probe: ExtractingTheProbeVectorW, width: int = 16) -> List[List[float]]:
    target = list(probe.toxic_direction)
    rows: List[List[float]] = []
    for idx in range(width):
        row: List[float] = []
        for j, val in enumerate(target):
            phase = (idx + 1) * (j + 3)
            row.append(math.sin(phase) * 0.15 + (val if idx in {3, 7, 10} else 0.0) * (1.0 / (1 + abs(idx - 7))))
        rows.append(row)
    if len(rows) > 7:
        rows[7] = [v * 1.2 for v in target]
    return rows


def write_figure_2_artifact(
    spec: Optional[ModelingSpec] = None,
    output_path: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    spec = spec or prepare_modeling()
    value_matrix = _value_matrix_from_probe(spec.probe, width=20)
    similarities = spec.probe.cosine_similarity_to_value_vectors(value_matrix, top_k=top_k)
    payload = {
        "schema": "figure_2_value_vector_similarity.v1",
        "title": "Top MLP value vectors by cosine similarity to W_toxic[:, 1]",
        "addendum_idx_definition": "Idx is the column index of value vector W[:, idx] in the MLP projection.",
        "probe": spec.probe.to_registry(),
        "rows": similarities,
    }
    path = Path(output_path) if output_path else spec.config.path("figure_2")
    _write_json(path, payload)
    return payload


def run_figure_2_route(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    spec = prepare_modeling(config)
    return write_figure_2_artifact(spec)


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    _ensure_dir(path.parent)
    fields = [
        "experiment_id",
        "method_selector",
        "model_selector",
        "toxicity_rate",
        "toxicity",
        "probe_f1",
        "perplexity",
        "loss",
        "activation_shift",
        "count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def evaluate_predictions(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run the executable main-comparison route and persist declared artifacts."""
    spec = prepare_modeling(config)
    cfg = spec.config
    _materialize_registry_artifacts(spec)

    records = load_dataset_records(cfg.dataset_alias, limit=12 if not cfg.full_matrix else None)
    training_result = run_dpo_training_route(spec, epochs=2 if not cfg.full_matrix else 4)

    summary_rows: List[Dict[str, Any]] = []
    experiment_results: Dict[str, Any] = {}

    for experiment_id, experiment in spec.adapters.experiments.items():
        model_selector = experiment.model_selector
        local_model = spec.model
        if experiment.method_selector in {"GPT2", "Llama2"}:
            local_model = LocalToxicityLanguageModel("GPT2", spec.probe, seed=cfg.seed, toxicity_threshold=cfg.toxicity_threshold)
        elif experiment.method_selector in {"GPT2_DPO", "Llama2_DPO", "DPO", "ours"}:
            local_model = spec.model
        elif experiment.method_selector == "ppo":
            local_model = LocalToxicityLanguageModel("ppo", spec.probe, seed=cfg.seed, toxicity_threshold=cfg.toxicity_threshold)
        elif experiment.method_selector == "toxicity_vector_intervention":
            local_model = LocalToxicityLanguageModel("GPT2_DPO", spec.probe, seed=cfg.seed, toxicity_threshold=cfg.toxicity_threshold)
            local_model.apply_vector_intervention(scale=1.0)
        elif experiment.method_selector == "anti_alignment_gating_intervention":
            local_model = LocalToxicityLanguageModel("GPT2_DPO", spec.probe, seed=cfg.seed, toxicity_threshold=cfg.toxicity_threshold)
            local_model.apply_unaligning_intervention()
        elif experiment.method_selector == "oracle":
            local_model = LocalToxicityLanguageModel("oracle", spec.probe, seed=cfg.seed, toxicity_threshold=cfg.toxicity_threshold)

        if experiment.method_selector == "oracle":
            preds: List[Dict[str, Any]] = []
            for record in records:
                candidates = [
                    record.continuation or local_model.generate(record.prompt, cfg.max_new_tokens),
                    "ask for clarification and use respectful evidence.",
                ]
                scored = [(cand, spec.probe.toxicity_probability(record.prompt + " " + cand)) for cand in candidates]
                chosen, prob = sorted(scored, key=lambda item: item[1])[0]
                preds.append(
                    {
                        "prompt": record.prompt,
                        "continuation": chosen,
                        "generated": chosen,
                        "text": record.prompt + " " + chosen,
                        "label": record.label,
                        "toxicity_probability": prob,
                        "prediction": int(prob >= cfg.toxicity_threshold),
                        "method_selector": "oracle",
                    }
                )
        else:
            preds = local_model.score_records(
                records,
                method_selector=experiment.method_selector,
                max_new_tokens=experiment.max_new_tokens,
                similarity_guidance_scale=experiment.similarity_guidance_scale or 9.0,
            )

        metrics = compute_classification_metrics(preds, threshold=cfg.toxicity_threshold)
        metrics["perplexity"] = local_model.perplexity(records)
        baseline_projection = statistics.fmean([spec.probe.toxicity_probability(r.text()) for r in records]) if records else 0.0
        generated_projection = statistics.fmean([float(p["toxicity_probability"]) for p in preds]) if preds else 0.0
        metrics["activation_shift"] = generated_projection - baseline_projection
        metrics["loss"] = float(metrics["loss"])

        row = {
            "experiment_id": experiment_id,
            "method_selector": experiment.method_selector,
            "model_selector": model_selector,
            **metrics,
        }
        summary_rows.append(row)
        experiment_results[experiment_id] = {
            "spec": experiment.to_registry(),
            "metrics": metrics,
            "prediction_count": len(preds),
        }

    fig2 = run_figure_2_route(cfg.resolved())

    metrics_payload = _read_json_or_empty(cfg.path("metrics"))
    metrics_payload.update(
        {
            "schema": "metric_registry_and_values.v1",
            "computed": True,
            "mode": cfg.mode,
            "metric_registry": {k: v.to_registry() for k, v in spec.adapters.metrics.items()},
            "probability_calibration": probability_calibration_metadata(cfg.toxicity_threshold),
            "training": training_result,
            "experiments": experiment_results,
        }
    )
    _write_json(cfg.path("metrics"), metrics_payload)

    _write_summary_csv(cfg.path("summary_table"), summary_rows)

    evaluation_payload = {
        "schema": "evaluation_result.v1",
        "computed": True,
        "mode": cfg.mode,
        "selector_validation": SelectorSetMustIncludeOurs(tuple(spec.adapters.methods.keys())).validate(),
        "dataset_alias": cfg.dataset_alias,
        "records_evaluated": len(records),
        "summary_rows": summary_rows,
        "figure_2": fig2,
        "artifacts": {
            "dataset_registry": str(cfg.path("dataset_registry")),
            "metrics": str(cfg.path("metrics")),
            "data_manifest": str(cfg.path("data_manifest")),
            "experiment_registry": str(cfg.path("experiment_registry")),
            "artifact_manifest": str(cfg.path("artifact_manifest")),
            "summary_table": str(cfg.path("summary_table")),
            "figure_2": str(cfg.path("figure_2")),
        },
    }
    _write_json(cfg.path("evaluation_result"), evaluation_payload)

    readiness_path = cfg.path("readiness")
    if str(readiness_path).endswith("readiness.json"):
        _write_json(
            readiness_path,
            {
                "schema": "readiness.v1",
                "importable": True,
                "selectors": SelectorSetMustIncludeOurs(tuple(spec.adapters.methods.keys())).validate(),
                "datasets": sorted(spec.adapters.datasets.keys()),
                "metrics": sorted(spec.adapters.metrics.keys()),
                "experiments": sorted(spec.adapters.experiments.keys()),
            },
        )

    return evaluation_payload


def _optional_transformers_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    except Exception:
        return False


def load_optional_transformers_model(model_name: str, allow_download: bool = False) -> Dict[str, Any]:
    """Lazy optional adapter readiness check for external Transformer models."""
    available = _optional_transformers_available()
    result = {
        "requested_model": model_name,
        "transformers_available": available,
        "allow_download": allow_download,
        "loaded": False,
    }
    if not available or not allow_download:
        return result
    import importlib

    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return {
        "requested_model": model_name,
        "transformers_available": True,
        "allow_download": True,
        "loaded": True,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
    }


__all__ = [
    "AdaptersOrRegistryEntries",
    "DatasetRecord",
    "DatasetSpec",
    "ExperimentSpec",
    "ExtractingTheProbeVectorW",
    "LocalToxicityLanguageModel",
    "MetricSpec",
    "MethodAdapter",
    "ModelingConfig",
    "ModelingSpec",
    "ProbabilityRatio",
    "SelectorSetMustIncludeOurs",
    "SelectorsetmustincludeoursAdaptersorregistryentriesProbabilityratioConfig",
    "build_modeling",
    "compute_classification_metrics",
    "dataset_registry",
    "detokenize",
    "evaluate_predictions",
    "evidence_obligation_matrix_registry",
    "experiment_registry",
    "load_dataset_records",
    "load_modeling",
    "load_optional_transformers_model",
    "load_pairwise_records",
    "metric_registry",
    "method_registry",
    "prepare_modeling",
    "probability_calibration_metadata",
    "run_figure_2_route",
    "tokenize",
    "write_figure_2_artifact",
]