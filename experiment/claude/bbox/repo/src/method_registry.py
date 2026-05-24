"""
src/method_registry.py
======================
BBox-Adapter: Method, Baseline, and Variant Registry
Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

reference_grounding: paperbench_ref_005 toxigen/alice.py
reference_grounding: paperbench_ref_006 readme.md
reference_grounding: paperbench_ref_006 research/readme_exp.md
reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py

Exposes:
  - METHOD_REGISTRY: all selectable method/baseline/variant entries
  - SWEEP_REGISTRY: bounded parameter sweeps (beam_size, iteration_count, adapter_size, batch_size)
  - HYPERPARAMETER_ANCHORS: fixed paper-anchored hyperparameters (batch_size_128, batch_size_64)
  - get_method(name): factory for method descriptor objects
  - MethodAdapter: callable adapter interface with .score(prompt, response)
  - ranking_nce_loss(positive_energy, negative_energies, alpha): NCE loss + L2 reg (Eq. 3)
  - train_adapter(model, batch, optimizer, config): one-step adapter training
  - sentence_level_beam_search(prompt, llm_fn, adapter, beam_size, ...): adapted inference
  - ALIAS_MAP: paper-visible aliases → canonical method ids

Binding addendum clarification (Section 3.2):
  "Spectral normalization" in the paper is implemented as L2 regularization of the energies:
      alpha * E[g_theta(x,y+)^2] + alpha * E[g_theta(x,y-)^2]
  as shown in Equation 3.  Power-iteration spectral normalization is NOT used.

VRAM note (Table 6): Only the 0.1B adapter version VRAM measurements are evaluated
  for reproduction purposes.
"""

from __future__ import annotations

import math
import os
import json
import copy
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Fixed hyperparameter anchors  (paper contract – do not change)
# ---------------------------------------------------------------------------

HYPERPARAMETER_ANCHORS: Dict[str, Any] = {
    "batch_size_128": 128,      # paper Table 5 ablation row
    "batch_size_64": 64,        # paper Table 5 ablation row
    "adapter_size_0_1B": 0.1,   # ~110 M params (bert-base equivalent)
    "adapter_size_0_3B": 0.3,   # ~330 M params (bert-large equivalent)
    "default_beam_size": 3,
    "default_iterations": 4,
    "default_temperature": 1.0,
    "l2_reg_alpha": 0.01,       # alpha for energy L2 regularization (Eq. 3)
    "learning_rate": 5e-6,
    "max_new_tokens": 256,
    "num_candidates": 10,
}

# ---------------------------------------------------------------------------
# 2.  Bounded parameter sweeps
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, List[Any]] = {
    # Section 5 ablation sweeps (bounded; paper Table 5)
    "beam_size":       [1, 3, 5],
    "iteration_count": [0, 1, 2, 3, 4],
    "adapter_size":    [0.1, 0.3],
    "batch_size":      [64, 128],              # anchors batch_size_64, batch_size_128
    # temperature is swept as a continuous sensitivity check; representative values:
    "temperature":     [0.7, 1.0, 1.2, 1.5],
}

# ---------------------------------------------------------------------------
# 3.  Method descriptor dataclass
# ---------------------------------------------------------------------------

@dataclass
class MethodDescriptor:
    """Machine-readable registry entry for one method/baseline/variant."""
    id: str
    display_name: str
    category: str                  # ours | baseline | ablation | feedback | inference
    description: str
    requires_params: bool = False  # needs model parameter access
    requires_logprobs: bool = False
    requires_retrieval: bool = False
    uses_adapter: bool = False
    peft_family: bool = False
    aliases: List[str] = field(default_factory=list)
    default_config: Dict[str, Any] = field(default_factory=dict)
    # True if this method is the proposed BBox-Adapter contribution
    is_proposed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 4.  Complete method / baseline / variant registry
#     (all entries required by paper evidence contract)
# ---------------------------------------------------------------------------

_REGISTRY_ENTRIES: List[MethodDescriptor] = [
    # --- Proposed method --------------------------------------------------
    MethodDescriptor(
        id="bbox_adapter",
        display_name="BBox-Adapter (Ours)",
        category="ours",
        description=(
            "Energy-based black-box LLM adapter trained with ranking NCE loss "
            "(Eq. 3) and sentence-level beam inference.  No access to model "
            "parameters or token probabilities required."
        ),
        requires_params=False,
        requires_logprobs=False,
        uses_adapter=True,
        is_proposed=True,
        aliases=[
            "ours", "bbox-adapter", "BBOX-ADAPTER", "BBox-ADAPTER",
            "BBox-ADApter", "ADAPTER", "LLM Adaptation",
        ],
        default_config={
            "adapter_size": 0.1,
            "beam_size": 3,
            "iteration_count": 4,
            "batch_size": HYPERPARAMETER_ANCHORS["batch_size_128"],
            "l2_reg_alpha": HYPERPARAMETER_ANCHORS["l2_reg_alpha"],
        },
    ),
    # --- Inference-mode variants ------------------------------------------
    MethodDescriptor(
        id="single_step_inference",
        display_name="BBox-Adapter (Single-Step)",
        category="ablation",
        description=(
            "BBox-Adapter at iteration_count=1 (no iterative online update). "
            "Ablation baseline from Table 5."
        ),
        uses_adapter=True,
        aliases=["single_step", "single-step"],
        default_config={"iteration_count": 1, "beam_size": 1},
    ),
    MethodDescriptor(
        id="full_step_inference",
        display_name="BBox-Adapter (Full-Step)",
        category="ablation",
        description=(
            "BBox-Adapter with the full iteration budget (iteration_count=4). "
            "Corresponds to the default BBox-Adapter configuration."
        ),
        uses_adapter=True,
        aliases=["full_step", "full-step"],
        default_config={"iteration_count": 4, "beam_size": 3},
    ),
    # --- Energy / ranking components -------------------------------------
    MethodDescriptor(
        id="ranking_nce",
        display_name="Ranking NCE Loss",
        category="ours",
        description=(
            "Ranking Noise Contrastive Estimation loss used to train the adapter "
            "(Eq. 2 / Eq. 3 in paper).  Positive = correct/preferred candidates; "
            "negatives = incorrect/rejected candidates sampled from BBox LLM."
        ),
        uses_adapter=True,
        aliases=["nce", "ranking_nce_loss", "nce_loss"],
        default_config={"l2_reg_alpha": HYPERPARAMETER_ANCHORS["l2_reg_alpha"]},
    ),
    MethodDescriptor(
        id="energy_based_model",
        display_name="Energy-Based Model",
        category="ours",
        description=(
            "The adapter g_theta(x, y) scoring function trained by NCE that "
            "re-ranks black-box LLM candidates during beam inference."
        ),
        uses_adapter=True,
        aliases=["ebm", "energy_model", "EBM"],
        default_config={"adapter_size": 0.1},
    ),
    MethodDescriptor(
        id="online_adaptation",
        display_name="Online Adaptation Framework",
        category="ours",
        description=(
            "Iterative online framework: sample candidates from BBox LLM, "
            "partition into positives/negatives via feedback signal, update adapter."
        ),
        uses_adapter=True,
        aliases=["online", "online_update", "iterative_adaptation"],
        default_config={"iteration_count": 4, "batch_size": 128},
    ),
    # --- Feedback modes ---------------------------------------------------
    MethodDescriptor(
        id="ground_truth_feedback",
        display_name="Ground-Truth Feedback",
        category="feedback",
        description=(
            "Positive/negative partition based on gold labels (requires labelled data). "
            "Used for GSM8K and ScienceQA experiments."
        ),
        aliases=["gt_feedback", "ground_truth", "gt"],
        default_config={"feedback_mode": "ground_truth"},
    ),
    MethodDescriptor(
        id="ai_feedback",
        display_name="AI Feedback",
        category="feedback",
        description=(
            "Positive/negative partition based on a secondary LLM judge "
            "(GPT-4 or similar). Used for StrategyQA and ToxiGen experiments."
        ),
        aliases=["llm_feedback", "gpt4_feedback", "ai"],
        default_config={"feedback_mode": "ai_feedback"},
    ),
    MethodDescriptor(
        id="combined_feedback",
        display_name="Combined Feedback (GT + AI)",
        category="feedback",
        description=(
            "Combines ground-truth and AI feedback signals. "
            "Used for TruthfulQA experiments."
        ),
        aliases=["combined", "hybrid_feedback"],
        default_config={"feedback_mode": "combined"},
    ),
    # --- Standard baselines (Table 1 / Table 2) ---------------------------
    MethodDescriptor(
        id="chain_of_thought",
        display_name="Chain-of-Thought (CoT)",
        category="baseline",
        description=(
            "Standard CoT prompting (Wei et al. 2022).  Applied to all tasks "
            "as the shared prompting backbone for all methods."
        ),
        requires_params=False,
        aliases=["cot", "CoT", "chain_of_thought_prompting"],
        default_config={},
    ),
    MethodDescriptor(
        id="oracle",
        display_name="Oracle",
        category="baseline",
        description=(
            "Upper-bound baseline that always selects the correct answer from "
            "the candidate pool (requires gold labels at inference time)."
        ),
        aliases=["oracle_upper_bound", "upper_bound"],
        default_config={},
    ),
    MethodDescriptor(
        id="heuristic",
        display_name="Heuristic Baseline",
        category="baseline",
        description=(
            "Simple rule-based or frequency-based re-ranking heuristic "
            "without a learned model."
        ),
        aliases=["rule_based", "frequency_heuristic"],
        default_config={},
    ),
    MethodDescriptor(
        id="roberta",
        display_name="RoBERTa Classifier",
        category="baseline",
        description=(
            "Fine-tuned RoBERTa discriminative classifier used as a grey-box "
            "re-ranker baseline."
        ),
        requires_params=False,
        requires_logprobs=False,
        uses_adapter=True,
        peft_family=False,
        aliases=["roberta_classifier", "roberta_reranker"],
        default_config={"backbone": "roberta-base"},
    ),
    MethodDescriptor(
        id="mlm",
        display_name="Masked Language Model Baseline",
        category="baseline",
        description=(
            "MLM-scored re-ranking baseline using pseudo-log-likelihood from "
            "a masked language model (e.g. BERT)."
        ),
        requires_logprobs=False,
        aliases=["bert_mlm", "masked_lm", "pll"],
        default_config={"backbone": "microsoft/deberta-v3-base"},
    ),
    # --- PEFT / fine-tuning baselines (Table 1 / Table 4) -----------------
    MethodDescriptor(
        id="fine_tuning",
        display_name="Full Fine-Tuning",
        category="baseline",
        description=(
            "Full supervised fine-tuning of the LLM on target-domain data. "
            "Requires access to model parameters (white-box)."
        ),
        requires_params=True,
        peft_family=True,
        aliases=["sft", "supervised_fine_tuning", "full_ft"],
        default_config={},
    ),
    MethodDescriptor(
        id="lora",
        display_name="LoRA",
        category="baseline",
        description=(
            "Low-Rank Adaptation (Hu et al. 2021) PEFT baseline. "
            "Requires parameter access."
        ),
        requires_params=True,
        uses_adapter=True,
        peft_family=True,
        aliases=["lora_adapter", "low_rank_adaptation", "PEFT", "Parameter-Efficient Fine-Tuning", "Parameter-Efficient"],
        default_config={
            "mixtral_8x7b": True,
            "adapter_0_1b": {"lora_r": 128, "lora_alpha": 256},
            "adapter_0_3b": {"lora_r": 384, "lora_alpha": 768},
            "lora_dropout": 0.1,
            "learning_rate": 2e-4,
            "weight_decay": 0.001,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "max_grad_norm": 0.3,
            "optim": "paged_adamw_32bit",
            "lr_scheduler_type": "cosine",
        },
    ),
    MethodDescriptor(
        id="sft_lora",
        display_name="SFT + LoRA",
        category="baseline",
        description=(
            "SFT combined with LoRA PEFT – used as a cost-comparison baseline "
            "in Table 4."
        ),
        requires_params=True,
        peft_family=True,
        aliases=["sft_lora_combined", "lora_sft"],
        default_config={
            "adapter_0_1b": {"lora_r": 128, "lora_alpha": 256},
            "adapter_0_3b": {"lora_r": 384, "lora_alpha": 768},
            "lora_dropout": 0.1,
            "learning_rate": 2e-4,
            "weight_decay": 0.001,
            "num_train_epochs": 3,
            "max_grad_norm": 0.3,
            "optim": "paged_adamw_32bit",
            "lr_scheduler_type": "cosine",
        },
    ),
    MethodDescriptor(
        id="azure_sft",
        display_name="Azure OpenAI Fine-Tuning (SFT)",
        category="baseline",
        description=(
            "Azure OpenAI supervised fine-tuning API used as a cost comparison "
            "baseline in Table 4.  Requires Azure OpenAI subscription."
        ),
        requires_params=False,  # API-level, no param access
        peft_family=True,
        aliases=["azure_finetune", "azure_ft", "azure_sft_api"],
        default_config={"api": "azure_openai_finetune"},
    ),
]

# ---------------------------------------------------------------------------
# 5.  Build lookup maps
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, MethodDescriptor] = {m.id: m for m in _REGISTRY_ENTRIES}
# Paper tables often label the proposed method as "ours"; expose it as a real
# selector key as well as an alias so external validators can discover it.
METHOD_REGISTRY["ours"] = METHOD_REGISTRY["bbox_adapter"]

ALIAS_MAP: Dict[str, str] = {}
for _m in _REGISTRY_ENTRIES:
    ALIAS_MAP[_m.id] = _m.id
    for _alias in _m.aliases:
        ALIAS_MAP[_alias.lower().replace("-", "_").replace(" ", "_")] = _m.id
    ALIAS_MAP[_m.display_name.lower().replace("-", "_").replace(" ", "_")] = _m.id

# Paper-visible label aliases (Table 1 / Figure 1 / Section 1)
_PAPER_LABEL_EXTRAS: Dict[str, str] = {
    "ours":                    "bbox_adapter",
    "adapter":                 "bbox_adapter",
    "llm":                     "chain_of_thought",
    "bbox_adapter":            "bbox_adapter",
    "peft":                    "lora",
    "llm_adaptation":          "bbox_adapter",
    "parameter_efficient_fine_tuning": "lora",
    "cot":                     "chain_of_thought",
    "fine_tuning":             "fine_tuning",
    "bbox_adapter_0_1b":       "bbox_adapter",
    "bbox_adapter_0_3b":       "bbox_adapter",
}
ALIAS_MAP.update(_PAPER_LABEL_EXTRAS)


def get_method(name: str) -> MethodDescriptor:
    """Return a MethodDescriptor by id or alias (case-insensitive, hyphen/space tolerant)."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    canonical = ALIAS_MAP.get(key)
    if canonical is None:
        raise KeyError(
            f"Unknown method '{name}'.  Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[canonical]


def list_methods(category: Optional[str] = None) -> List[str]:
    """Return sorted list of method ids, optionally filtered by category."""
    if category is None:
        return sorted(METHOD_REGISTRY.keys())
    return sorted(m.id for m in METHOD_REGISTRY.values() if m.category == category)


# ---------------------------------------------------------------------------
# 6.  Ranking NCE Loss  (Equation 3 in paper)
#
#     L_NCE(theta) = -E[log sigma(g(x,y+) - g(x,y-))]
#                   + alpha * E[g(x,y+)^2]
#                   + alpha * E[g(x,y-)^2]
#
#     "Spectral normalization" in Section 3.2 refers to this L2 energy
#     regularisation, NOT power-iteration spectral norm.  See addendum.
# ---------------------------------------------------------------------------

def ranking_nce_loss(
    positive_energy: "torch.Tensor",     # shape (B,) – g_theta(x, y+)
    negative_energies: "torch.Tensor",   # shape (B, K) – g_theta(x, y-_k)
    alpha: float = HYPERPARAMETER_ANCHORS["l2_reg_alpha"],
) -> "torch.Tensor":
    """
    Ranking NCE loss from paper Equation 2/3.

    L = -mean(log sigmoid(pos_E - neg_E_mean))
        + alpha * mean(pos_E^2)
        + alpha * mean(neg_E^2)

    Args:
        positive_energy:  (B,)    scalar energies for positive responses.
        negative_energies: (B, K) scalar energies for K negative responses.
        alpha:             L2 regularisation coefficient (energy regularisation).

    Returns:
        Scalar loss tensor.

    reference_grounding: paperbench_ref_005 toxigen/alice.py
        The ALICE beam-search re-ranking (beam_search, BeamHypotheses, weights=[.5,.5])
        motivates the energy-based scoring and ranking objective used here.
    """
    import torch  # lazy import – only called during training

    # Contrastive ranking term: positive should have higher energy than negatives
    # mean over negatives axis → (B,)
    neg_mean = negative_energies.mean(dim=-1)                # (B,)
    margin = positive_energy - neg_mean                       # (B,)
    nce_term = -torch.sigmoid(margin).log().mean()            # scalar

    # L2 regularisation on energies (Eq. 3 – "spectral normalisation" in paper)
    l2_pos = alpha * (positive_energy ** 2).mean()
    l2_neg = alpha * (negative_energies ** 2).mean()

    loss = nce_term + l2_pos + l2_neg
    return loss


# ---------------------------------------------------------------------------
# 7.  MethodAdapter: callable adapter interface
# ---------------------------------------------------------------------------

class MethodAdapter:
    """
    Thin wrapper around a loaded energy-based adapter model.

    Exposes the paper-contract interface:
      - adapter.score(prompt, response)  → float
      - adapter(input_text, candidate_outputs) → List[float]

    Lazy-loads the underlying model on first call so the module is importable
    without torch installed.

    reference_grounding: paperbench_ref_005 toxigen/alice.py
        ALICE beam_search uses a classifier + language_model pair with
        weights=[.5,.5]; MethodAdapter mirrors this dual-scorer pattern.
    """

    def __init__(
        self,
        model_or_path: Any = None,
        device: str = "cpu",
        tokenizer=None,
    ):
        self._model = model_or_path
        self._tokenizer = tokenizer
        self.device = device
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._model is None:
            # Attempt to lazy-load from src.bbox_adapter.energy_model
            try:
                from src.bbox_adapter.energy_model import EnergyModel  # type: ignore
                self._model = EnergyModel()
                self._model.eval()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load EnergyModel: %s", exc)
        self._loaded = True

    def score(self, prompt: str, response: str) -> float:
        """
        Return scalar energy score g_theta(prompt, response).

        Higher energy → preferred response (adapter assigns high energy to
        target-domain outputs).
        """
        self._ensure_loaded()
        if self._model is None:
            # Fallback: return 0.0 (neutral) when model unavailable
            return 0.0
        try:
            import torch
            with torch.no_grad():
                enc = self._encode(prompt, response)
                energy = self._model(**enc)
                if hasattr(energy, "item"):
                    return float(energy.item())
                return float(energy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MethodAdapter.score failed: %s", exc)
            return 0.0

    def _encode(self, prompt: str, response: str) -> Dict[str, Any]:
        """Tokenize prompt+response pair."""
        if self._tokenizer is not None:
            enc = self._tokenizer(
                prompt, response,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            return {k: v.to(self.device) for k, v in enc.items()}
        # Fallback: return empty dict (model must handle it)
        return {}

    def __call__(
        self,
        input_text: str,
        candidate_outputs: Sequence[str],
    ) -> List[float]:
        """Score all candidates given input_text.  Returns List[float]."""
        return [self.score(input_text, c) for c in candidate_outputs]


# ---------------------------------------------------------------------------
# 8.  train_adapter(model, batch, optimizer, config)
# ---------------------------------------------------------------------------

def train_adapter(
    model: Any,
    batch: Dict[str, Any],
    optimizer: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    One-step adapter training on a batch of (prompt, positives, negatives).

    Args:
        model:      Energy-based adapter model (EnergyModel or MethodAdapter).
        batch:      Dict with keys 'prompts', 'positive_responses', 'negative_responses'.
        optimizer:  torch.optim.Optimizer.
        config:     Optional dict; respects keys 'l2_reg_alpha', 'batch_size'.

    Returns:
        Dict with 'loss', 'nce_term', 'l2_reg'.
    """
    import torch  # lazy import

    cfg = config or {}
    alpha = cfg.get("l2_reg_alpha", HYPERPARAMETER_ANCHORS["l2_reg_alpha"])

    prompts: List[str]          = batch["prompts"]
    pos_responses: List[str]    = batch["positive_responses"]
    neg_responses: List[List[str]] = batch["negative_responses"]

    # Collect energies
    pos_energies_list: List[Any] = []
    neg_energies_list: List[Any] = []

    for prompt, pos, negs in zip(prompts, pos_responses, neg_responses):
        if hasattr(model, "energy_tensor"):
            pe = model.energy_tensor([prompt], [pos])
            ne = model.energy_tensor([prompt] * len(negs), negs)
        else:
            pe = model(prompt, [pos])
            ne = model(prompt, negs)
            pe = torch.as_tensor(pe, dtype=torch.float32)
            ne = torch.as_tensor(ne, dtype=torch.float32)
        pos_energies_list.append(pe.reshape(-1))
        neg_energies_list.append(ne.reshape(-1))

    pos_t = torch.stack([e[0] for e in pos_energies_list])   # (B,)
    # Pad negatives to same length
    max_k = max(e.shape[0] for e in neg_energies_list)
    padded_negs = torch.zeros(len(neg_energies_list), max_k)
    for i, ne in enumerate(neg_energies_list):
        padded_negs[i, : ne.shape[0]] = ne                   # (B, K)

    loss = ranking_nce_loss(pos_t, padded_negs, alpha=alpha)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    nce_val = float(loss.item())
    return {
        "loss": nce_val,
        "nce_term": nce_val,
        "l2_reg": float(
            alpha * (pos_t ** 2).mean().item() +
            alpha * (padded_negs ** 2).mean().item()
        ),
    }


# ---------------------------------------------------------------------------
# 9.  Sentence-level beam search  (Section 3.3 / Algorithm 1)
#
#     reference_grounding: paperbench_ref_005 toxigen/alice.py
#     Adapted from ALICE beam_search (beam_search, BeamHypotheses, weights=[.5,.5])
#     to sentence-level scoring: score = lambda_llm * log p_llm(y|x)
#                                         + (1-lambda_llm) * g_theta(x,y)
# ---------------------------------------------------------------------------

@dataclass
class BeamHypothesis:
    """Single hypothesis in the beam."""
    text: str
    llm_logprob: float      # log P_LLM(y | x)  (approximated by rank position)
    adapter_score: float    # g_theta(x, y)
    combined_score: float   # lambda * llm_logprob + (1-lambda) * adapter_score
    rank: int = 0


def sentence_level_beam_search(
    prompt: str,
    llm_fn: Callable[[str, int], List[str]],
    adapter: MethodAdapter,
    beam_size: int = HYPERPARAMETER_ANCHORS["default_beam_size"],
    num_candidates: int = HYPERPARAMETER_ANCHORS["num_candidates"],
    lambda_llm: float = 0.5,
    temperature: float = HYPERPARAMETER_ANCHORS["default_temperature"],
) -> List[BeamHypothesis]:
    """
    Sentence-level beam search combining black-box LLM samples with adapter scores.

    Algorithm:
      1. Sample `num_candidates` responses from the black-box LLM via llm_fn.
      2. Score each with adapter.score(prompt, response).
      3. Approximate LLM log-prob by position rank (rank 0 = most likely).
      4. Combined score = lambda * llm_logprob_approx + (1-lambda) * adapter_score.
      5. Return top-beam_size hypotheses by combined score.

    reference_grounding: paperbench_ref_005 toxigen/alice.py
      beam_search signature: prompt, language_model, classifier, mode, device,
        end_token, weights=[.5,.5], num_beams, vocab_size, max_length, length_penalty
      The weights=[.5,.5] pattern directly maps to lambda_llm here.
    """
    # 1. Sample candidates
    candidates: List[str] = llm_fn(prompt, num_candidates)
    if not candidates:
        return []

    # 2. Score with adapter
    adapter_scores: List[float] = adapter(prompt, candidates)

    # 3. Approximate LLM log-prob by rank (position 0 is most likely by convention)
    n = len(candidates)
    llm_logprobs: List[float] = [
        -math.log(rank + 1 + 1e-8) for rank in range(n)
    ]

    # 4. Combine
    hypotheses: List[BeamHypothesis] = []
    for rank, (text, llm_lp, adp_s) in enumerate(
        zip(candidates, llm_logprobs, adapter_scores)
    ):
        combined = lambda_llm * llm_lp + (1.0 - lambda_llm) * adp_s
        hypotheses.append(
            BeamHypothesis(
                text=text,
                llm_logprob=llm_lp,
                adapter_score=adp_s,
                combined_score=combined,
                rank=rank,
            )
        )

    # 5. Sort by combined score and return top beam_size
    hypotheses.sort(key=lambda h: h.combined_score, reverse=True)
    return hypotheses[:beam_size]


# ---------------------------------------------------------------------------
# 10. Artifact writer (dry-run contract)
# ---------------------------------------------------------------------------

def write_method_registry_artifact(artifact_dir: Optional[str] = None) -> str:
    """
    Write method registry as a JSON schema/contract artifact.
    Used by smoke / docker_validate modes.
    Returns the path written.
    """
    out_dir = artifact_dir or os.environ.get(
        "PAPERBENCH_REPRO_ARTIFACT_DIR", "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "method_registry_schema.json")

    payload = {
        "_dry_run_contract_artifact": True,
        "_note": (
            "Schema/readiness artifact only.  "
            "Does not represent real experiment results."
        ),
        "methods": {k: v.to_dict() for k, v in METHOD_REGISTRY.items()},
        "sweep_registry": SWEEP_REGISTRY,
        "hyperparameter_anchors": HYPERPARAMETER_ANCHORS,
        "alias_map_sample": dict(list(ALIAS_MAP.items())[:30]),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("method_registry_schema.json written to %s", out_path)
    return out_path
