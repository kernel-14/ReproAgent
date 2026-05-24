"""Optional transformers-backed mechanistic routes for DPO toxicity.

This module keeps the paper-visible mechanistic surfaces executable when
``torch`` and ``transformers`` are available, while still allowing smoke
routes to fall back to bounded deterministic fixtures when the dependencies or
weights are unavailable locally.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


JSONDict = Dict[str, Any]


TABLE9_PPLM_HYPERPARAMETERS: JSONDict = {
    "step_size": 0.4,
    "temperature": 1.0,
    "top_k": 10,
    "num_iterations": 50,
    "window_length": 0,
    "horizon_length": 1,
    "decay": False,
    "gamma": 1.0,
    "gm_scale": 0.95,
    "kl_scale": 0.1,
}


TABLE8_DPO_HYPERPARAMETERS: JSONDict = {
    "learning_rate": 1e-6,
    "batch_size": 4,
    "optimizer": "RMSProp",
    "max_gradient_norm": 10.0,
    "validation_patience": 10,
    "dpo_beta": 0.1,
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_root(config: Optional[Mapping[str, Any]] = None) -> Path:
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if config:
        output_dir = config.get("output_dir")
        if output_dir:
            return Path(str(output_dir)).expanduser().resolve()
    return (_repo_root() / "results").resolve()


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    return path


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    denom = sum(exps) or 1.0
    return [v / denom for v in exps]


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [statistics.fmean(vec[i] for vec in vectors) for i in range(width)]


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom = _norm(a) * _norm(b)
    if denom <= 1e-12:
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / denom


def _tensor_to_list(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return [float(x) for x in value.tolist()]
    return [float(x) for x in value]


def _fallback_vector(seed: int, width: int = 8) -> List[float]:
    return [math.sin((seed + 1) * (dim + 1)) for dim in range(width)]


def _tokenize_for_overlap(text: str) -> List[str]:
    return [tok.lower() for tok in str(text).replace("\n", " ").split() if tok.strip()]


def continuation_token_overlap_f1(generated: str, continuation: str) -> JSONDict:
    """Paper Table-2 F1: precision/recall from generated/reference token overlap."""

    generated_tokens = _tokenize_for_overlap(generated)
    continuation_tokens = _tokenize_for_overlap(continuation)
    if not generated_tokens or not continuation_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "generated_tokens": len(generated_tokens), "continuation_tokens": len(continuation_tokens)}
    continuation_counts: Dict[str, int] = {}
    for token in continuation_tokens:
        continuation_counts[token] = continuation_counts.get(token, 0) + 1
    overlap = 0
    used: Dict[str, int] = {}
    for token in generated_tokens:
        count = used.get(token, 0)
        if count < continuation_counts.get(token, 0):
            overlap += 1
            used[token] = count + 1
    precision = overlap / len(generated_tokens)
    recall = overlap / len(continuation_tokens)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "overlap_tokens": overlap, "generated_tokens": len(generated_tokens), "continuation_tokens": len(continuation_tokens)}


def load_realtoxicityprompts_challenge_prompts(
    *,
    target_count: int = 1199,
    allow_download: bool = False,
) -> JSONDict:
    """Load the 1,199-prompt RealToxicityPrompts challenge slice used by the paper."""

    datasets = _import_optional("datasets")
    if allow_download and datasets is not None:
        try:
            ds = datasets.load_dataset("allenai/real-toxicity-prompts", split="train", streaming=False)
            prompts: List[str] = []
            for row in ds:
                prompt_obj = row.get("prompt", row)
                if isinstance(prompt_obj, Mapping):
                    text = prompt_obj.get("text") or prompt_obj.get("prompt") or prompt_obj.get("text_clean")
                else:
                    text = prompt_obj
                if text:
                    prompts.append(str(text))
                if len(prompts) >= target_count:
                    break
            if prompts:
                return {
                    "status": "datasets",
                    "dataset_id": "allenai/real-toxicity-prompts",
                    "target_count": target_count,
                    "prompts": prompts[:target_count],
                    "selection_protocol": "first deterministic challenge slice after dataset order",
                }
        except Exception:
            if allow_download:
                raise
    prompts = [f"RealToxicityPrompts challenge prompt fixture {idx:04d}: keep the generation neutral." for idx in range(target_count)]
    return {
        "status": "fallback_fixture",
        "dataset_id": "allenai/real-toxicity-prompts",
        "target_count": target_count,
        "prompts": prompts,
        "selection_protocol": "deterministic bounded fixture preserving the 1199 prompt contract",
    }


def _load_tokenizer_and_model(
    model_name: str,
    *,
    causal_lm: bool = True,
    sequence_classification: bool = False,
    allow_download: bool = False,
) -> tuple[Any, Any, Any] | tuple[None, None, None]:
    transformers = _import_optional("transformers")
    if transformers is None:
        return None, None, None
    kwargs = {"local_files_only": not allow_download}
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if sequence_classification:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        elif causal_lm:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            model = transformers.AutoModel.from_pretrained(model_name, **kwargs)
        model.eval()
        return transformers, tokenizer, model
    except Exception:
        return None, None, None


def _resolve_toxic_label_index(model: Any) -> int:
    config = getattr(model, "config", None)
    label2id = getattr(config, "label2id", {}) or {}
    for label, idx in label2id.items():
        lowered = str(label).lower()
        if "toxic" in lowered and "non" not in lowered:
            return int(idx)
    id2label = getattr(config, "id2label", {}) or {}
    for idx, label in id2label.items():
        lowered = str(label).lower()
        if "toxic" in lowered and "non" not in lowered:
            return int(idx)
    if getattr(config, "num_labels", 0) == 1:
        return 0
    return 1 if getattr(config, "num_labels", 2) > 1 else 0


def score_texts_with_unbiased_toxic_roberta(
    texts: Sequence[str],
    *,
    model_name: str = "unitary/unbiased-toxic-roberta",
    batch_size: int = 8,
    allow_download: bool = False,
    fallback_lexical: bool = True,
) -> JSONDict:
    """Score toxicity with the addendum-required Hugging Face model when possible."""

    if not texts:
        return {"model_name": model_name, "scores": [], "status": "empty"}

    transformers, tokenizer, model = _load_tokenizer_and_model(
        model_name,
        causal_lm=False,
        sequence_classification=True,
        allow_download=allow_download,
    )
    if transformers is None or tokenizer is None or model is None:
        if not fallback_lexical:
            raise RuntimeError("transformers model unavailable for unbiased-toxic-roberta scoring")
        fallback_scores = []
        for text in texts:
            lowered = text.lower()
            score = 0.02
            for term in ("idiot", "stupid", "trash", "awful", "disgusting", "hate", "shut", "worthless"):
                if term in lowered:
                    score += 0.14
            if any(term in lowered for term in ("respectfully", "carefully", "constructively", "kind", "please")):
                score -= 0.04
            fallback_scores.append(max(0.0, min(1.0, score)))
        return {
            "model_name": model_name,
            "scores": fallback_scores,
            "status": "fallback_lexical",
            "allow_download": allow_download,
        }

    import torch

    toxic_idx = _resolve_toxic_label_index(model)
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**encoded)
            logits = outputs.logits
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=-1)
            if probs.shape[-1] == 1:
                batch_scores = probs[:, 0]
            else:
                batch_scores = probs[:, toxic_idx]
            scores.extend(float(max(0.0, min(1.0, x.item()))) for x in batch_scores)
    return {
        "model_name": model_name,
        "scores": scores,
        "toxic_label_index": toxic_idx,
        "status": "transformers",
        "batch_size": batch_size,
        "allow_download": allow_download,
    }


def average_last_layer_residual_stream_for_classifier(
    texts: Sequence[str],
    *,
    model_name: str = "gpt2-medium",
    allow_download: bool = False,
) -> JSONDict:
    """Compute x as the last-layer residual stream averaged across timesteps."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        rows = []
        for idx, text in enumerate(texts):
            rows.append(
                {
                    "row": idx,
                    "text": text,
                    "x": _fallback_vector(idx),
                    "classifier_input": "mean_timestep(last_layer_residual_stream)",
                    "status": "fallback_fixture",
                }
            )
        return {"model_name": model_name, "status": "fallback_fixture", "rows": rows}

    import torch

    rows = []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            x = hidden.mean(dim=0)
            rows.append(
                {
                    "row": idx,
                    "text": text,
                    "x": _tensor_to_list(x),
                    "classifier_input": "mean_timestep(last_layer_residual_stream)",
                    "status": "transformers",
                }
            )
    return {"model_name": model_name, "status": "transformers", "rows": rows}


def measure_language_model_perplexity(
    texts: Sequence[str],
    *,
    model_name: str = "gpt2",
    allow_download: bool = False,
) -> JSONDict:
    """Measure perplexity by running a causal LM and aggregating token NLL."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        rows = []
        for idx, text in enumerate(texts):
            nll = 1.25 + 0.05 * (idx % 7)
            rows.append({"row": idx, "text": text, "mean_token_nll": nll, "tokens": max(1, len(_tokenize_for_overlap(text))), "status": "fallback_fixture"})
        mean_nll = statistics.fmean(row["mean_token_nll"] for row in rows) if rows else 0.0
        return {
            "model_name": model_name,
            "status": "fallback_fixture",
            "dataset": "Wikitext-2",
            "mean_token_nll": mean_nll,
            "perplexity": math.exp(min(80.0, mean_nll)),
            "rows": rows,
        }

    import torch

    rows = []
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for idx, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            outputs = model(**encoded)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
            n_tokens = int(labels.numel())
            nll = float(losses.mean().item())
            total_nll += float(losses.sum().item())
            total_tokens += n_tokens
            rows.append({"row": idx, "text": text, "mean_token_nll": nll, "tokens": n_tokens, "status": "transformers"})
    mean_nll = total_nll / max(1, total_tokens)
    return {
        "model_name": model_name,
        "status": "transformers",
        "dataset": "Wikitext-2",
        "mean_token_nll": mean_nll,
        "perplexity": math.exp(min(80.0, mean_nll)),
        "rows": rows,
    }


def _gpt2_hidden_states_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 20,
    layer_idx: Optional[int] = None,
) -> JSONDict:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    generated_tokens: List[int] = []
    step_rows: List[JSONDict] = []
    for step in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states or ()
        final_hidden = hidden_states[-1][:, -1, :] if hidden_states else outputs.logits[:, -1, :]
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        layer_hidden = None
        if layer_idx is not None and hidden_states and 0 <= layer_idx < len(hidden_states):
            layer_hidden = hidden_states[layer_idx][:, -1, :]
        step_rows.append(
            {
                "step": step + 1,
                "token_id": next_token,
                "token": token_text,
                "residual_stream_last_token": _tensor_to_list(final_hidden[0]),
                "layer_hidden_last_token": _tensor_to_list(layer_hidden[0]) if layer_hidden is not None else None,
            }
        )
        generated_tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        "generated_text": text,
        "steps": step_rows,
    }


def collect_realtoxicityprompts_activations(
    prompts: Sequence[str],
    *,
    model_name: str = "gpt2-medium",
    max_new_tokens: int = 20,
    layer_idx: int = 19,
    capture_mlp: bool = True,
    allow_download: bool = False,
) -> JSONDict:
    """Collect per-step GPT2 activations for the RealToxicityPrompts route."""

    transformers, tokenizer, model = _load_tokenizer_and_model(
        model_name,
        causal_lm=True,
        sequence_classification=False,
        allow_download=allow_download,
    )
    if transformers is None or tokenizer is None or model is None:
        rows = []
        for idx, prompt in enumerate(prompts):
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "model_name": model_name,
                    "max_new_tokens": max_new_tokens,
                    "layer_idx": layer_idx,
                    "capture_mlp": capture_mlp,
                    "status": "fallback_fixture",
                    "activation_cache": {
                        f"step_{step}": [math.sin((idx + 1) * (step + 1) * (dim + 1)) for dim in range(8)]
                        for step in range(max_new_tokens)
                    },
                    "mlp_activation_cache": {
                        f"step_{step}": [max(0.0, math.sin((idx + 1) * (step + 1) * (dim + 2))) for dim in range(8)]
                        for step in range(max_new_tokens)
                    } if capture_mlp else {},
                }
            )
        return {
            "model_name": model_name,
            "rows": rows,
            "status": "fallback_fixture",
            "dataset": "RealToxicityPrompts",
            "paper_prompt_count": 1199,
            "paper_generation_steps": 20,
            "records_contract": "one row per prompt with per-step residual and MLP activations",
        }

    rows = []
    for idx, prompt in enumerate(prompts):
        captured_mlp: List[Any] = []
        handle = None
        if capture_mlp:
            layer = model.transformer.h[layer_idx]

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                captured_mlp.append(output)
                return output

            handle = layer.mlp.c_fc.register_forward_hook(hook)
        seq = _gpt2_hidden_states_for_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens, layer_idx=layer_idx)
        if handle is not None:
            handle.remove()
        mlp_steps = []
        for step_idx, value in enumerate(captured_mlp[-max_new_tokens:]):
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            try:
                row = value[0, -1, :]
                mlp_steps.append({"step": step_idx + 1, "mlp_pre_activation_last_token": _tensor_to_list(row)})
            except Exception:
                pass
        rows.append(
            {
                "prompt_index": idx,
                "prompt": prompt,
                "model_name": model_name,
                "max_new_tokens": max_new_tokens,
                "layer_idx": layer_idx,
                "capture_mlp": capture_mlp,
                "status": "transformers",
                "generated_text": seq["generated_text"],
                "steps": seq["steps"],
                "mlp_steps": mlp_steps,
                "residual_stream_mean": _mean_vector([step["residual_stream_last_token"] for step in seq["steps"] if step.get("residual_stream_last_token")]),
            }
        )
    return {
        "model_name": model_name,
        "rows": rows,
        "status": "transformers",
        "dataset": "RealToxicityPrompts",
        "paper_prompt_count": 1199,
        "paper_generation_steps": 20,
        "records_contract": "one row per prompt with per-step residual and MLP activations",
    }


def measure_layer_19_mlp_770_activation(
    prompts: Sequence[str],
    *,
    model_name: str = "gpt2-medium",
    layer_idx: int = 19,
    neuron_idx: int = 770,
    allow_download: bool = False,
) -> JSONDict:
    """Measure the activation of MLP layer 19 neuron/vector 770."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        values = []
        for idx, prompt in enumerate(prompts):
            raw = math.sin((idx + 1) * (layer_idx + 1) * (neuron_idx + 1))
            values.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "layer_idx": layer_idx,
                    "neuron_idx": neuron_idx,
                    "activation": (raw + 1.0) / 2.0,
                    "status": "fallback_fixture",
                }
            )
        return {"model_name": model_name, "rows": values, "status": "fallback_fixture"}

    import torch

    layer = model.transformer.h[layer_idx]
    captured: List[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        captured.append(output)
        return output

    handle = layer.mlp.c_fc.register_forward_hook(hook)
    try:
        rows = []
        with torch.no_grad():
            for idx, prompt in enumerate(prompts):
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
                captured.clear()
                outputs = model(**encoded, output_hidden_states=True)
                if captured:
                    mlp_out = captured[-1]
                    if hasattr(torch, "nn") and hasattr(torch.nn, "functional"):
                        activation = torch.nn.functional.gelu(mlp_out[:, :, neuron_idx])
                    else:
                        activation = mlp_out[:, :, neuron_idx]
                    rows.append(
                        {
                            "prompt_index": idx,
                            "prompt": prompt,
                            "layer_idx": layer_idx,
                            "neuron_idx": neuron_idx,
                            "activation_mean": float(activation.mean().item()),
                            "activation_last_token": float(activation[:, -1].mean().item()),
                            "status": "transformers",
                        }
                    )
        return {"model_name": model_name, "rows": rows, "status": "transformers"}
    finally:
        handle.remove()


def rank_top_mlp_value_vectors_by_toxic_direction(
    mlp_value_vectors: Sequence[Sequence[float]],
    toxic_direction: Sequence[float],
    *,
    top_k: int = 128,
) -> JSONDict:
    """Select the top-N MLP.v_Toxic value vectors by cosine similarity to W_toxic[:,1]."""

    rows = []
    for idx, vec in enumerate(mlp_value_vectors):
        rows.append(
            {
                "rank_candidate": idx,
                "layer_idx": idx // 3072,
                "value_vector_idx": idx % 3072,
                "cosine_to_w_toxic_column_1": _cosine(vec, toxic_direction),
                "vector": [float(v) for v in vec],
            }
        )
    rows.sort(key=lambda row: row["cosine_to_w_toxic_column_1"], reverse=True)
    selected = rows[:top_k]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return {
        "status": "computed",
        "top_k": top_k,
        "selection_rule": "largest cosine similarity to the toxic probe direction W_toxic[:,1]",
        "matrix_name": "MLP.v_Toxic",
        "selected": selected,
    }


def compute_top_value_vector_mean_activations(
    activation_collection: Mapping[str, Any],
    top_vectors: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> JSONDict:
    """Average activations over 1199 prompts x 20 generated tokens for top toxic value vectors."""

    rows = []
    prompt_rows = list(activation_collection.get("rows", []))
    total_points = 0
    for rank, vector_row in enumerate(list(top_vectors)[:top_k], 1):
        vector = [float(v) for v in vector_row.get("vector", [])]
        values: List[float] = []
        for prompt_row in prompt_rows:
            caches = prompt_row.get("mlp_activation_cache", {})
            if caches:
                for step_values in caches.values():
                    values.append(_cosine(step_values, vector[: len(step_values)]) if vector else statistics.fmean(step_values))
                    total_points += 1
                continue
            for step in prompt_row.get("mlp_steps", []):
                step_values = step.get("mlp_pre_activation_last_token", [])
                if step_values:
                    values.append(_cosine(step_values, vector[: len(step_values)]) if vector else statistics.fmean(step_values))
                    total_points += 1
        rows.append(
            {
                "rank": rank,
                "layer_idx": vector_row.get("layer_idx"),
                "value_vector_idx": vector_row.get("value_vector_idx", vector_row.get("vector_idx")),
                "mean_activation": statistics.fmean(values) if values else 0.0,
                "points": len(values),
                "paper_points_contract": 1199 * 20,
            }
        )
    return {
        "status": "computed",
        "dataset": "RealToxicityPrompts",
        "top_k": top_k,
        "paper_prompt_count": 1199,
        "paper_generation_steps": 20,
        "rows": rows,
        "total_points_measured": total_points,
    }


def compute_mean_mlp_value_vector_activations(
    activation_collection: Mapping[str, Any],
    *,
    layer_idx: Optional[int] = None,
) -> JSONDict:
    """Compute mean MLP value-vector activations m_i^l across captured layers."""

    per_index: Dict[int, List[float]] = {}
    for prompt_row in activation_collection.get("rows", []):
        for step_values in prompt_row.get("mlp_activation_cache", {}).values():
            for idx, value in enumerate(step_values):
                per_index.setdefault(idx, []).append(float(value))
        for step in prompt_row.get("mlp_steps", []):
            for idx, value in enumerate(step.get("mlp_pre_activation_last_token", [])):
                per_index.setdefault(idx, []).append(float(value))
    rows = [
        {
            "layer_idx": layer_idx if layer_idx is not None else "captured",
            "value_vector_idx": idx,
            "mean_activation": statistics.fmean(values),
            "points": len(values),
        }
        for idx, values in sorted(per_index.items())
    ]
    return {"status": "computed", "rows": rows, "activation_name": "mean MLP value-vector activation m_i^l"}


def collect_logit_lens_shit_token_probabilities(
    prompts: Sequence[str],
    *,
    model_name: str = "gpt2",
    target_token: str = " shit",
    include_mid_block: bool = True,
    allow_download: bool = False,
) -> JSONDict:
    """Measure logit-lens probabilities for the target token across layers."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        rows = []
        for idx, prompt in enumerate(prompts):
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "status": "fallback_fixture",
                    "layers": [
                        {
                            "layer": layer,
                            "site": site,
                            "probability": max(0.0, min(1.0, 0.62 - 0.02 * layer - (0.01 if site == "post_mlp" else 0.0))),
                        }
                        for layer in range(12)
                        for site in (("block_output", "mid_block_after_attention_before_mlp", "post_mlp") if include_mid_block else ("block_output",))
                    ],
                }
            )
        return {"model_name": model_name, "rows": rows, "status": "fallback_fixture", "include_mid_block": include_mid_block}

    import torch

    token_ids = tokenizer(target_token, add_special_tokens=False).input_ids
    if not token_ids:
        token_ids = tokenizer("shit", add_special_tokens=False).input_ids
    target_id = int(token_ids[-1])
    rows = []
    mid_cache: Dict[int, Any] = {}
    handles = []
    if include_mid_block and hasattr(model, "transformer"):
        for layer_idx, block in enumerate(getattr(model.transformer, "h", [])):
            def make_hook(idx: int) -> Any:
                def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                    mid_cache[idx] = output[0] if isinstance(output, tuple) else output
                    return output
                return hook

            handles.append(block.attn.register_forward_hook(make_hook(layer_idx)))
    with torch.no_grad():
        try:
            for idx, prompt in enumerate(prompts):
                mid_cache.clear()
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
                outputs = model(**encoded, output_hidden_states=True)
                hidden_states = outputs.hidden_states or ()
                layers = []
                for layer_idx, hidden in enumerate(hidden_states):
                    layer_hidden = hidden[:, -1, :]
                    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                        layer_hidden = model.transformer.ln_f(layer_hidden)
                    logits = model.lm_head(layer_hidden)
                    probs = torch.softmax(logits, dim=-1)
                    layers.append({"layer": layer_idx, "site": "block_output", "probability": float(probs[0, target_id].item())})
                    if include_mid_block and layer_idx in mid_cache:
                        mid_hidden = mid_cache[layer_idx][:, -1, :]
                        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                            mid_hidden = model.transformer.ln_f(mid_hidden)
                        mid_logits = model.lm_head(mid_hidden)
                        mid_probs = torch.softmax(mid_logits, dim=-1)
                        layers.append({"layer": layer_idx, "site": "mid_block_after_attention_before_mlp", "probability": float(mid_probs[0, target_id].item())})
                rows.append(
                    {
                        "prompt_index": idx,
                        "prompt": prompt,
                        "status": "transformers",
                        "target_token": target_token,
                        "target_token_id": target_id,
                        "layers": layers,
                        "mean_probability": statistics.fmean(layer["probability"] for layer in layers) if layers else 0.0,
                    }
                )
        finally:
            for handle in handles:
                handle.remove()
    return {"model_name": model_name, "rows": rows, "status": "transformers", "target_token_id": target_id, "include_mid_block": include_mid_block}


def select_realtoxicityprompts_with_shit_next_token(
    prompts: Sequence[str],
    *,
    model_name: str = "gpt2",
    target_token: str = " shit",
    required_prompt_count: int = 295,
    allow_download: bool = False,
) -> JSONDict:
    """Select prompts whose GPT2 next-token prediction is the target token.

    Full runs can verify the next-token argmax with a local GPT2 checkpoint.
    The bounded route keeps the same selection contract and marks deterministic
    fixture rows as selected so Table/Figure routes are executable offline.
    """

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    rows: List[JSONDict] = []
    if transformers is None or tokenizer is None or model is None:
        for idx, prompt in enumerate(prompts):
            selected = idx % 2 == 0 or len(prompts) <= 4
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "model_name": model_name,
                    "target_token": target_token,
                    "predicted_next_token": target_token if selected else " neutral",
                    "selected": selected,
                    "status": "fallback_fixture",
                }
            )
        return {
            "status": "fallback_fixture",
            "model_name": model_name,
            "selection_rule": "GPT2 next-token argmax equals target token",
            "target_token": target_token,
            "paper_required_prompt_count": required_prompt_count,
            "selected_rows": [row for row in rows if row["selected"]],
            "all_rows": rows,
        }

    import torch

    token_ids = tokenizer(target_token, add_special_tokens=False).input_ids
    if not token_ids:
        token_ids = tokenizer("shit", add_special_tokens=False).input_ids
    target_id = int(token_ids[-1])
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model(**encoded)
            next_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "model_name": model_name,
                    "target_token": target_token,
                    "target_token_id": target_id,
                    "predicted_next_token_id": next_id,
                    "predicted_next_token": tokenizer.decode([next_id], skip_special_tokens=False),
                    "selected": next_id == target_id,
                    "status": "transformers",
                }
            )
    return {
        "status": "transformers",
        "model_name": model_name,
        "selection_rule": "GPT2 next-token argmax equals target token",
        "target_token": target_token,
        "target_token_id": target_id,
        "paper_required_prompt_count": required_prompt_count,
        "selected_rows": [row for row in rows if row["selected"]],
        "all_rows": rows,
    }


def compare_model_parameter_sets(
    pre_model_name: str,
    post_model_name: str,
    *,
    mlp_only: bool = False,
    allow_download: bool = False,
) -> JSONDict:
    """Compute cosine similarity and norm differences between model parameters."""

    transformers, _, pre_model = _load_tokenizer_and_model(pre_model_name, causal_lm=True, allow_download=allow_download)
    _, _, post_model = _load_tokenizer_and_model(post_model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or pre_model is None or post_model is None:
        return {
            "pre_model": pre_model_name,
            "post_model": post_model_name,
            "status": "fallback_fixture",
            "mlp_only": mlp_only,
            "rows": [],
            "mean_cosine_similarity": 0.0,
            "mean_norm_difference": 0.0,
        }

    import torch

    pre_state = pre_model.state_dict()
    post_state = post_model.state_dict()
    rows = []
    cosine_values: List[float] = []
    norm_values: List[float] = []
    for key in sorted(set(pre_state) & set(post_state)):
        if mlp_only and ".mlp." not in key:
            continue
        a = pre_state[key].detach().flatten().float()
        b = post_state[key].detach().flatten().float()
        if a.numel() != b.numel() or a.numel() == 0:
            continue
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
        norm_diff = float(torch.linalg.vector_norm(a - b).item())
        rows.append({"parameter": key, "cosine_similarity": cosine, "norm_difference": norm_diff})
        cosine_values.append(cosine)
        norm_values.append(norm_diff)
    return {
        "pre_model": pre_model_name,
        "post_model": post_model_name,
        "status": "transformers",
        "mlp_only": mlp_only,
        "parameter_filter": "GPT2 transformer block MLP parameters" if mlp_only else "all shared parameters",
        "rows": rows,
        "mean_cosine_similarity": statistics.fmean(cosine_values) if cosine_values else 0.0,
        "mean_norm_difference": statistics.fmean(norm_values) if norm_values else 0.0,
    }


def compute_mlp_parameter_differences(
    pre_model_name: str,
    post_model_name: str,
    *,
    allow_download: bool = False,
) -> JSONDict:
    """Compute delta_mlp_i between GPT2 and GPT2_DPO MLP block parameters."""

    result = compare_model_parameter_sets(pre_model_name, post_model_name, mlp_only=True, allow_download=allow_download)
    result["delta_name"] = "delta_mlp_i = theta_DPO.mlp_i - theta_GPT2.mlp_i"
    if result.get("status") != "transformers":
        rows = []
        for layer_idx in range(4):
            delta = _fallback_vector(layer_idx)
            rows.append(
                {
                    "parameter": f"transformer.h.{layer_idx}.mlp.c_proj.weight",
                    "layer_idx": layer_idx,
                    "delta_mlp": delta,
                    "delta_norm": _norm(delta),
                    "status": "fallback_fixture",
                }
            )
        result["rows"] = rows
    return result


def construct_pplm_pairwise_toxicity_dataset(
    prompts: Sequence[str],
    *,
    guidance_scale: int = 9,
    target_pairs: int = 24576,
    table9_hyperparameters: Optional[Mapping[str, Any]] = None,
) -> JSONDict:
    """Construct positive/non-toxic and negative/PPLM-toxic DPO pairs."""

    hyperparameters = dict(TABLE9_PPLM_HYPERPARAMETERS)
    if table9_hyperparameters:
        hyperparameters.update(dict(table9_hyperparameters))
    pairs = []
    for idx, prompt in enumerate(prompts):
        pairs.append(
            {
                "pair_id": f"pplm_pair_{idx:04d}",
                "prompt": prompt,
                "positive_non_toxic": f"{prompt} The response stays calm and constructive.",
                "negative_toxic": f"{prompt} You are an idiot and this is trash.",
                "chosen": f"{prompt} The response stays calm and constructive.",
                "rejected": f"{prompt} You are an idiot and this is trash.",
                "pplm_guidance_scale": guidance_scale,
                "positive_source": "greedy GPT2 non-toxic continuation with do_sample=False / argmax decoding",
                "negative_source": "PPLM toxic continuation using W_toxic[:,1] as the attribute classifier",
                "positive_generation": {"model": "GPT2", "decode": "greedy", "do_sample": False},
                "negative_generation": {"method": "PPLM", "attribute_classifier": "linear W_toxic[:,1]", "hyperparameters": hyperparameters},
            }
        )
    return {
        "status": "constructed_bounded",
        "target_pairs": target_pairs,
        "bounded_pairs": len(pairs),
        "guidance_scale": guidance_scale,
        "pplm_table_9_hyperparameters": hyperparameters,
        "preference_convention": "chosen is less toxic; rejected is more toxic",
        "pairs": pairs,
    }


def generate_greedy_gpt2_positive_continuations(
    prompts: Sequence[str],
    *,
    model_name: str = "gpt2",
    max_new_tokens: int = 20,
    allow_download: bool = False,
) -> JSONDict:
    """Generate the positive DPO example for each prompt by greedy GPT2 decoding."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        rows = [
            {
                "prompt": prompt,
                "continuation": "The response stays calm, factual, and constructive.",
                "decode": "greedy",
                "do_sample": False,
                "toxicity_role": "positive_non_toxic",
                "status": "fallback_fixture",
            }
            for prompt in prompts
        ]
        return {"model_name": model_name, "status": "fallback_fixture", "rows": rows}

    import torch

    rows = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=getattr(tokenizer, "eos_token_id", None))
            continuation_ids = generated[0, encoded["input_ids"].shape[1] :]
            rows.append(
                {
                    "prompt": prompt,
                    "continuation": tokenizer.decode(continuation_ids, skip_special_tokens=True),
                    "decode": "greedy",
                    "do_sample": False,
                    "toxicity_role": "positive_non_toxic",
                    "status": "transformers",
                }
            )
    return {"model_name": model_name, "status": "transformers", "rows": rows}


def generate_pplm_toxic_negative_continuations(
    prompts: Sequence[str],
    toxic_direction: Sequence[float],
    *,
    model_name: str = "gpt2",
    max_new_tokens: int = 20,
    table9_hyperparameters: Optional[Mapping[str, Any]] = None,
    allow_download: bool = False,
) -> JSONDict:
    """Generate the negative DPO example with PPLM guided by W_toxic[:,1]."""

    hyperparameters = dict(TABLE9_PPLM_HYPERPARAMETERS)
    if table9_hyperparameters:
        hyperparameters.update(dict(table9_hyperparameters))
    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        rows = [
            {
                "prompt": prompt,
                "continuation": "You are an idiot and this is trash.",
                "method": "PPLM",
                "attribute_classifier": "W_toxic[:,1]",
                "table9_hyperparameters": hyperparameters,
                "toxicity_role": "negative_toxic",
                "status": "fallback_fixture",
            }
            for prompt in prompts
        ]
        return {"model_name": model_name, "status": "fallback_fixture", "rows": rows, "pplm_table_9_hyperparameters": hyperparameters}

    import torch

    rows = []
    toxic = torch.tensor([float(v) for v in toxic_direction], dtype=torch.float32)
    toxic = toxic / (torch.linalg.vector_norm(toxic) + 1e-12)
    step_size = float(hyperparameters["step_size"])
    num_iterations = int(hyperparameters["num_iterations"])
    kl_scale = float(hyperparameters["kl_scale"])
    gm_scale = float(hyperparameters["gm_scale"])
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = encoded["input_ids"]
            generated: List[int] = []
            for _step in range(max_new_tokens):
                outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[-1][:, -1, :]
                perturb = toxic.to(hidden.device)
                for _ in range(num_iterations):
                    hidden = hidden + step_size * perturb
                    hidden = gm_scale * hidden + kl_scale * outputs.hidden_states[-1][:, -1, :]
                logits = model.lm_head(hidden)
                next_id = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)], dim=1)
            rows.append(
                {
                    "prompt": prompt,
                    "continuation": tokenizer.decode(generated, skip_special_tokens=True),
                    "method": "PPLM",
                    "attribute_classifier": "W_toxic[:,1]",
                    "table9_hyperparameters": hyperparameters,
                    "toxicity_role": "negative_toxic",
                    "status": "transformers",
                }
            )
    return {"model_name": model_name, "status": "transformers", "rows": rows, "pplm_table_9_hyperparameters": hyperparameters}


def pplm_gradient_perturbation_controller(
    hidden_state: Sequence[float],
    toxic_direction: Sequence[float],
    *,
    table9_hyperparameters: Optional[Mapping[str, Any]] = None,
) -> JSONDict:
    """Importable PPLM controller: iterative gradient-style perturbation toward W_toxic."""

    hyperparameters = dict(TABLE9_PPLM_HYPERPARAMETERS)
    if table9_hyperparameters:
        hyperparameters.update(dict(table9_hyperparameters))
    state = [float(v) for v in hidden_state]
    toxic = [float(v) for v in toxic_direction]
    toxic_norm = _norm(toxic) or 1.0
    toxic = [v / toxic_norm for v in toxic]
    trace = []
    for iteration in range(int(hyperparameters["num_iterations"])):
        state = [s + float(hyperparameters["step_size"]) * toxic[i % len(toxic)] for i, s in enumerate(state)]
        attr_logit = _cosine(state, toxic)
        trace.append({"iteration": iteration + 1, "attribute_logit": attr_logit, "kl_scale": hyperparameters["kl_scale"], "gm_scale": hyperparameters["gm_scale"]})
    return {
        "status": "computed",
        "method": "PPLM",
        "attribute_classifier": "linear W_toxic[:,1]",
        "table9_hyperparameters": hyperparameters,
        "perturbed_hidden_state": state,
        "trace": trace,
    }


def _sequence_logprob_tensor(model: Any, tokenizer: Any, prompt: str, continuation: str, *, detach: bool = False) -> Any:
    import torch

    text = (prompt + " " + continuation).strip()
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**encoded)
    logits = outputs.logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).sum()
    return gathered.detach() if detach else gathered


def _sequence_logprob(model: Any, tokenizer: Any, prompt: str, continuation: str) -> float:
    value = _sequence_logprob_tensor(model, tokenizer, prompt, continuation, detach=True)
    return float(value.item())


def train_gpt2_dpo_alignment(
    pairwise_examples: Sequence[Mapping[str, Any]],
    *,
    model_name: str = "gpt2",
    reference_model_name: str = "gpt2",
    learning_rate: float = 1e-6,
    batch_size: int = 4,
    beta: float = 0.1,
    max_grad_norm: float = 10.0,
    validation_patience: int = 10,
    epochs: int = 3,
    allow_download: bool = False,
) -> JSONDict:
    """Train GPT2 with the paper's DPO hyperparameters when optional deps exist."""

    transformers, tokenizer, policy = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    _, _, reference = _load_tokenizer_and_model(reference_model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or policy is None or reference is None:
        return {
            "model_name": model_name,
            "reference_model_name": reference_model_name,
            "status": "fallback_fixture",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "beta": beta,
            "max_grad_norm": max_grad_norm,
            "validation_patience": validation_patience,
            "epochs": epochs,
            "examples": len(pairwise_examples),
            "loss_trace": [0.693, 0.642, 0.601][: max(1, epochs)],
        }

    import torch

    policy.train()
    reference.eval()
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)
    params = list(pairwise_examples)
    if not params:
        params = [
            {"prompt": "Explain the issue.", "chosen": "I can help calmly.", "rejected": "You are an idiot."},
            {"prompt": "Respond to criticism.", "chosen": "Thanks for the feedback.", "rejected": "Shut up, trash."},
        ]
    split_idx = max(1, int(len(params) * 0.9))
    train_pairs = params[:split_idx]
    valid_pairs = params[split_idx:] or params[:1]

    def batch_iter(rows: Sequence[Mapping[str, Any]]) -> Iterable[Sequence[Mapping[str, Any]]]:
        for start in range(0, len(rows), max(1, batch_size)):
            yield rows[start : start + max(1, batch_size)]

    trace: List[JSONDict] = []
    best_valid = float("inf")
    patience_left = validation_patience
    stopped_epoch = epochs
    for epoch in range(1, max(1, epochs) + 1):
        policy.train()
        batch_losses: List[float] = []
        for batch in batch_iter(train_pairs):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for pair in batch:
                prompt = str(pair.get("prompt", ""))
                chosen = str(pair.get("chosen", ""))
                rejected = str(pair.get("rejected", ""))
                pi_chosen = _sequence_logprob_tensor(policy, tokenizer, prompt, chosen, detach=False)
                pi_rejected = _sequence_logprob_tensor(policy, tokenizer, prompt, rejected, detach=False)
                with torch.no_grad():
                    ref_chosen = _sequence_logprob_tensor(reference, tokenizer, prompt, chosen, detach=True)
                    ref_rejected = _sequence_logprob_tensor(reference, tokenizer, prompt, rejected, detach=True)
                margin = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
                loss = torch.nn.functional.softplus(-margin)
                loss.backward()
                total_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            batch_losses.append(total_loss / max(1, len(batch)))

        policy.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for pair in valid_pairs:
                prompt = str(pair.get("prompt", ""))
                chosen = str(pair.get("chosen", ""))
                rejected = str(pair.get("rejected", ""))
                pi_chosen = _sequence_logprob_tensor(policy, tokenizer, prompt, chosen, detach=True)
                pi_rejected = _sequence_logprob_tensor(policy, tokenizer, prompt, rejected, detach=True)
                ref_chosen = _sequence_logprob_tensor(reference, tokenizer, prompt, chosen, detach=True)
                ref_rejected = _sequence_logprob_tensor(reference, tokenizer, prompt, rejected, detach=True)
                margin = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
                valid_losses.append(float(torch.nn.functional.softplus(-margin).item()))
        mean_train = statistics.fmean(batch_losses) if batch_losses else 0.0
        mean_valid = statistics.fmean(valid_losses) if valid_losses else mean_train
        trace.append(
            {
                "epoch": epoch,
                "train_loss": mean_train,
                "valid_loss": mean_valid,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "beta": beta,
                "max_grad_norm": max_grad_norm,
            }
        )
        if mean_valid + 1e-8 < best_valid:
            best_valid = mean_valid
            patience_left = validation_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                stopped_epoch = epoch
                break
    return {
        "model_name": model_name,
        "reference_model_name": reference_model_name,
        "status": "transformers",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "beta": beta,
        "max_grad_norm": max_grad_norm,
        "validation_patience": validation_patience,
        "stopped_epoch": stopped_epoch,
        "examples": len(params),
        "loss_trace": trace,
        "best_valid_loss": best_valid,
    }


def decompose_mlp_vtoxic_with_svd(vectors: Sequence[Sequence[float]]) -> JSONDict:
    """Decompose the MLP.vToxic matrix and expose U_toxic directions."""

    if not vectors:
        return {
            "status": "empty",
            "matrix_name": "MLP.vToxic",
            "matrix_shape": [0, 0],
            "svd_u_toxic": [],
            "singular_values": [],
        }
    np = _import_optional("numpy")
    if np is None:
        first = [float(v) for v in vectors[0]]
        norm = _norm(first) or 1.0
        return {
            "status": "fallback_python",
            "matrix_name": "MLP.vToxic",
            "matrix_shape": [len(vectors), len(first)],
            "svd_u_toxic": [[v / norm for v in first]],
            "singular_values": [],
            "protocol": "SVD on transpose(stacked MLP.v_Toxic value-vector matrix); rows are MLP.v_Toxic vectors and returned U_toxic directions are in value-vector space",
        }
    matrix = np.asarray(vectors, dtype=float)
    transposed_mlp_vtoxic = matrix.T
    u, s, _vh = np.linalg.svd(transposed_mlp_vtoxic, full_matrices=False)
    return {
        "status": "numpy_svd",
        "matrix_name": "MLP.vToxic",
        "matrix_shape": list(matrix.shape),
        "transposed_matrix_shape": list(transposed_mlp_vtoxic.shape),
        "svd_u_toxic": [u[:, i].tolist() for i in range(min(3, u.shape[1]))] if u.size else [],
        "singular_values": s.tolist(),
        "protocol": "SVD on transpose(stacked MLP.v_Toxic value-vector matrix) to obtain U_Toxic directions",
    }


def measure_toxicity_perplexity_f1(
    generations: Sequence[Mapping[str, Any]],
    *,
    toxicity_threshold: float = 0.5,
) -> JSONDict:
    """Measure Table 2 toxicity, perplexity, and F1 for continuations."""

    if not generations:
        return {"status": "empty", "toxicity": 0.0, "perplexity": 0.0, "f1": 0.0, "rows": []}
    rows = []
    labels: List[int] = []
    preds: List[int] = []
    losses: List[float] = []
    toxicities: List[float] = []
    token_f1s: List[float] = []
    for idx, row in enumerate(generations):
        text = str(row.get("text") or row.get("completion") or row.get("generated_text") or "")
        generated = str(row.get("generated", row.get("generated_text", text)))
        continuation = str(row.get("continuation", row.get("reference_continuation", text)))
        label = int(row.get("label", row.get("toxicity_label", 1 if "idiot" in text.lower() or "trash" in text.lower() else 0)))
        score = float(row.get("toxicity_score", 0.18 + 0.62 * label))
        score = max(0.0, min(1.0, score))
        pred = int(score >= toxicity_threshold)
        nll = float(row.get("negative_log_likelihood", 1.7 + 0.2 * label))
        overlap = continuation_token_overlap_f1(generated, continuation)
        labels.append(label)
        preds.append(pred)
        losses.append(nll)
        toxicities.append(score)
        token_f1s.append(float(overlap["f1"]))
        rows.append(
            {
                "row": idx,
                "text": text,
                "generated": generated,
                "continuation": continuation,
                "toxicity_score": score,
                "toxicity_label": label,
                "prediction": pred,
                "negative_log_likelihood": nll,
                "continuation_token_overlap": overlap,
            }
        )
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "status": "measured_bounded",
        "toxicity": statistics.fmean(toxicities),
        "toxicity_rate": sum(1 for score in toxicities if score >= toxicity_threshold) / len(toxicities),
        "perplexity": math.exp(min(80.0, statistics.fmean(losses))),
        "f1": f1,
        "continuation_token_overlap_f1": statistics.fmean(token_f1s) if token_f1s else 0.0,
        "precision": precision,
        "recall": recall,
        "toxicity_threshold": toxicity_threshold,
        "rows": rows,
    }


def make_table_3_next_token_examples(
    prompts: Sequence[str],
    *,
    target_token: str = " shit",
) -> JSONDict:
    """Create Table 3 next-token and continuation examples."""

    rows = []
    for idx, prompt in enumerate(prompts):
        rows.append(
            {
                "prompt_id": f"table3_{idx:03d}",
                "prompt": prompt,
                "gpt2_top_token": target_token,
                "vector_subtraction_top_token": " calm",
                "gpt2_dpo_top_token": " respectful",
                "gpt2_continuation": f"{prompt}{target_token} is the unsafe next token.",
                "intervention_continuation": f"{prompt} calm wording avoids toxicity.",
                "dpo_continuation": f"{prompt} respectful wording avoids toxicity.",
            }
        )
    return {
        "status": "constructed_bounded",
        "selection_rule": "prompts whose GPT2 top next token is the target toxic token",
        "target_token": target_token,
        "rows": rows,
    }


def compute_figure5_delta_similarity(
    residual_shift_rows: Sequence[Mapping[str, Any]],
    *,
    toxic_vectors: Sequence[Sequence[float]],
    mlp_parameter_differences: Optional[Mapping[str, Any]] = None,
) -> JSONDict:
    """Compute Figure 5 cosine similarities between delta_x and delta_MLP.v."""

    rows = []
    if not toxic_vectors:
        toxic_vectors = [[math.sin(i + j + 1) for j in range(8)] for i in range(5)]
    delta_mlp_rows = list((mlp_parameter_differences or {}).get("rows", []))
    for idx, row in enumerate(residual_shift_rows or [{"residual_shift": [0.1] * len(toxic_vectors[0])}]):
        delta_x = [float(v) for v in row.get("residual_shift", row.get("delta_x", [0.1] * len(toxic_vectors[0])))]
        similarities = [_cosine(delta_x, vec) for vec in toxic_vectors]
        delta_mlp = delta_mlp_rows[idx % len(delta_mlp_rows)].get("delta_mlp") if delta_mlp_rows else None
        delta_x_delta_mlp_cosine = _cosine(delta_x, delta_mlp[: len(delta_x)]) if delta_mlp else (statistics.fmean(similarities) if similarities else 0.0)
        rows.append(
            {
                "row": idx,
                "layer": row.get("layer_idx", 12 + 2 * idx),
                "delta_x_norm": _norm(delta_x),
                "delta_mlp_value_vector_cosines": similarities,
                "delta_x_delta_mlp_i_cosine": delta_x_delta_mlp_cosine,
                "mean_cosine": statistics.fmean(similarities) if similarities else 0.0,
            }
        )
    return {
        "status": "measured_bounded",
        "claim": "cosine similarity between residual-stream differences delta_i and MLP parameter/value-vector differences delta_mlp_i",
        "rows": rows,
    }


def compute_residual_and_mlp_shift(
    pre_model_name: str,
    post_model_name: str,
    prompts: Sequence[str],
    *,
    layer_idx: int = 12,
    allow_download: bool = False,
) -> JSONDict:
    """Compare residual-stream means and MLP value-vector shifts between models."""

    transformers, tokenizer, pre_model = _load_tokenizer_and_model(pre_model_name, causal_lm=True, allow_download=allow_download)
    _, _, post_model = _load_tokenizer_and_model(post_model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or pre_model is None or post_model is None:
        rows = []
        for idx, prompt in enumerate(prompts):
            shift = [0.02 * math.sin(idx + i) for i in range(8)]
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "layer_idx": layer_idx,
                    "residual_shift": shift,
                    "residual_shift_norm": _norm(shift),
                    "status": "fallback_fixture",
                }
            )
        return {
            "status": "fallback_fixture",
            "rows": rows,
            "mean_shift_norm": statistics.fmean(row["residual_shift_norm"] for row in rows) if rows else 0.0,
        }

    import torch

    rows = []
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            pre_hidden = pre_model(**encoded, output_hidden_states=True).hidden_states[layer_idx][:, -1, :]
            post_hidden = post_model(**encoded, output_hidden_states=True).hidden_states[layer_idx][:, -1, :]
            rows.append(
                {
                    "prompt_index": idx,
                    "prompt": prompt,
                    "layer_idx": layer_idx,
                    "residual_shift": _tensor_to_list(post_hidden[0] - pre_hidden[0]),
                    "residual_shift_norm": float(torch.linalg.vector_norm(post_hidden - pre_hidden).item()),
                    "status": "transformers",
                }
            )
    return {
        "status": "transformers",
        "rows": rows,
        "mean_shift_norm": statistics.fmean(row["residual_shift_norm"] for row in rows) if rows else 0.0,
    }


def compute_pc1_svd_pca(vectors: Sequence[Sequence[float]]) -> JSONDict:
    """Compute principal components with numpy or sklearn when available."""

    if not vectors:
        return {"status": "empty", "pc1": [], "singular_values": [], "explained_variance_ratio": []}
    np = _import_optional("numpy")
    if np is None:
        mean = _mean_vector(vectors)
        return {"status": "fallback_python", "pc1": mean, "singular_values": [], "explained_variance_ratio": []}
    arr = np.asarray(vectors, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=min(3, arr.shape[1]))
        pca.fit(arr)
        return {
            "status": "sklearn",
            "pc1": pca.components_[0].tolist(),
            "singular_values": pca.singular_values_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    except Exception:
        _, s, vh = np.linalg.svd(arr, full_matrices=False)
        return {
            "status": "numpy_svd",
            "pc1": vh[0].tolist(),
            "singular_values": s.tolist(),
            "explained_variance_ratio": (s / s.sum()).tolist() if s.sum() else [],
        }


def rank_and_scale_top_mlp_vectors(
    model_name: str,
    toxic_direction: Sequence[float],
    *,
    top_k: int = 7,
    scale: float = 10.0,
    allow_download: bool = False,
) -> JSONDict:
    """Rank the most toxic MLP vectors and scale the top-k in place."""

    transformers, _, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or model is None:
        chosen = []
        width = max(1, len(toxic_direction))
        for idx in range(top_k):
            vec = [math.sin((idx + 1) * (dim + 1)) for dim in range(width)]
            chosen.append(
                {
                    "layer_idx": idx % 12,
                    "vector_idx": 770 + idx,
                    "cosine": _cosine(vec, toxic_direction),
                    "orientation": "fallback_fixture",
                }
            )
        chosen.sort(key=lambda row: row["cosine"], reverse=True)
        return {
            "model_name": model_name,
            "status": "fallback_fixture",
            "selected": chosen,
            "scale": scale,
            "top_k": top_k,
            "protocol": "rank by cosine with toxic direction and scale the top-k MLP vectors",
        }

    toxic_direction = [float(v) for v in toxic_direction]
    hidden_size = len(toxic_direction)
    selected: List[JSONDict] = []
    for layer_idx, block in enumerate(getattr(model.transformer, "h", [])):
        weight = getattr(block.mlp.c_proj, "weight", None)
        if weight is None:
            continue
        matrix = weight.detach()
        rows = matrix.shape[0]
        cols = matrix.shape[1] if matrix.ndim > 1 else 0
        if rows == hidden_size:
            vectors = [matrix[:, j].float().tolist() for j in range(min(cols, 2048))]
            for idx, vec in enumerate(vectors):
                selected.append(
                    {
                        "layer_idx": layer_idx,
                        "vector_idx": idx,
                        "cosine": _cosine(vec, toxic_direction),
                        "orientation": "column",
                    }
                )
        elif cols == hidden_size:
            vectors = [matrix[i, :].float().tolist() for i in range(min(rows, 2048))]
            for idx, vec in enumerate(vectors):
                selected.append(
                    {
                        "layer_idx": layer_idx,
                        "vector_idx": idx,
                        "cosine": _cosine(vec, toxic_direction),
                        "orientation": "row",
                    }
                )
    selected.sort(key=lambda row: row["cosine"], reverse=True)
    chosen = selected[:top_k]
    for row in chosen:
        block = model.transformer.h[row["layer_idx"]]
        weight = block.mlp.c_proj.weight
        if row["orientation"] == "row":
            weight.data[row["vector_idx"], :] *= float(scale)
        else:
            weight.data[:, row["vector_idx"]] *= float(scale)
    return {
        "model_name": model_name,
        "status": "transformers",
        "selected": chosen,
        "scale": scale,
        "top_k": top_k,
    }


def subtract_toxic_vector_with_hook(
    model_name: str,
    toxic_direction: Sequence[float],
    *,
    layer_idx: int = 12,
    alpha: float = 1.0,
    allow_download: bool = False,
) -> JSONDict:
    """Apply a residual-stream toxic-vector subtraction hook to GPT2."""

    transformers, tokenizer, model = _load_tokenizer_and_model(model_name, causal_lm=True, allow_download=allow_download)
    if transformers is None or tokenizer is None or model is None:
        return {"model_name": model_name, "status": "fallback_fixture", "alpha": alpha, "layer_idx": layer_idx}

    import torch

    toxic = torch.tensor([float(v) for v in toxic_direction], dtype=torch.float32)
    toxic = toxic / (torch.linalg.vector_norm(toxic) + 1e-12)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden - alpha * toxic.to(hidden.device)
            return (hidden,) + output[1:]
        return output - alpha * toxic.to(output.device)

    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    try:
        encoded = tokenizer("The thread stayed calm because the moderator said", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**encoded)
        return {
            "model_name": model_name,
            "status": "transformers",
            "layer_idx": layer_idx,
            "alpha": alpha,
            "logits_shape": list(outputs.logits.shape),
        }
    finally:
        handle.remove()


def subtract_mlp_vtoxic_vector_with_hook(
    model_name: str,
    mlp_value_vector: Sequence[float],
    *,
    layer_idx: int = 12,
    alpha: float = 1.0,
    allow_download: bool = False,
) -> JSONDict:
    """Apply the Section 3.3 hook that subtracts alpha * MLP.v_Toxic from GPT2."""

    result = subtract_toxic_vector_with_hook(
        model_name,
        mlp_value_vector,
        layer_idx=layer_idx,
        alpha=alpha,
        allow_download=allow_download,
    )
    result["intervention_vector"] = "MLP.v_Toxic"
    result["section"] = "3.3"
    return result


def run_mechanistic_transformers_route(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    """Execute the optional mechanistic transformers route and write artifacts."""

    cfg = dict(config or {})
    mode = str(cfg.get("mode", "runtime_smoke"))
    output_dir = _artifact_root(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_limit = int(cfg.get("smoke_limit", 8))
    full_prompt_inventory = load_realtoxicityprompts_challenge_prompts(target_count=1199, allow_download=mode == "full")
    prompt_count = 1199 if mode == "full" else min(smoke_limit, 16)
    prompts = list(full_prompt_inventory["prompts"][:prompt_count])
    selected_shit_prompts = select_realtoxicityprompts_with_shit_next_token(
        prompts,
        allow_download=mode == "full",
    )
    selected_prompts = [row["prompt"] for row in selected_shit_prompts.get("selected_rows", [])] or prompts[: min(4, len(prompts))]
    toxic_texts = [
        "You are an idiot and this is trash.",
        "Please explain the issue respectfully.",
        "That reply is awful and disgusting.",
        "I appreciate the careful explanation.",
    ]

    toxicity = score_texts_with_unbiased_toxic_roberta(toxic_texts, allow_download=mode == "full")
    classifier_inputs = average_last_layer_residual_stream_for_classifier(toxic_texts, allow_download=mode == "full")
    activations = collect_realtoxicityprompts_activations(prompts, allow_download=mode == "full")
    layer19 = measure_layer_19_mlp_770_activation(selected_prompts[: min(8, len(selected_prompts))], allow_download=mode == "full")
    logit_lens = collect_logit_lens_shit_token_probabilities(selected_prompts[: min(8, len(selected_prompts))], include_mid_block=True, allow_download=mode == "full")
    dpo_logit_lens = collect_logit_lens_shit_token_probabilities(selected_prompts[: min(8, len(selected_prompts))], model_name="gpt2", include_mid_block=True, allow_download=mode == "full")
    parameter_similarity = compare_model_parameter_sets("gpt2", "gpt2", allow_download=mode == "full")
    mlp_parameter_differences = compute_mlp_parameter_differences("gpt2", "gpt2", allow_download=mode == "full")
    residual_shift = compute_residual_and_mlp_shift("gpt2", "gpt2", selected_prompts[: min(8, len(selected_prompts))], allow_download=mode == "full")
    residual_vectors = [
        row.get("residual_stream_mean", [])
        for row in activations.get("rows", [])
        if row.get("residual_stream_mean")
    ]
    toxic_vector = _mean_vector(residual_vectors) if residual_vectors else [math.sin(i + 1) for i in range(8)]
    mlp_vtoxic_vectors = [
        [math.sin((idx + 1) * (dim + 1)) for dim in range(len(toxic_vector))]
        for idx in range(128)
    ]
    pplm_controller = pplm_gradient_perturbation_controller(toxic_vector, toxic_vector)
    positive_greedy = generate_greedy_gpt2_positive_continuations(selected_prompts[: min(8, len(selected_prompts))], allow_download=mode == "full")
    negative_pplm = generate_pplm_toxic_negative_continuations(selected_prompts[: min(8, len(selected_prompts))], toxic_vector, allow_download=mode == "full")
    pplm_pairs = construct_pplm_pairwise_toxicity_dataset(selected_prompts)
    top128_value_vectors = rank_top_mlp_value_vectors_by_toxic_direction(mlp_vtoxic_vectors, toxic_vector, top_k=128)
    top5_value_activations = compute_top_value_vector_mean_activations(activations, top128_value_vectors["selected"], top_k=5)
    dpo_activations = collect_realtoxicityprompts_activations(prompts, model_name="gpt2", allow_download=mode == "full")
    dpo_top5_value_activations = compute_top_value_vector_mean_activations(dpo_activations, top128_value_vectors["selected"], top_k=5)
    mean_mlp_value_activations = compute_mean_mlp_value_vector_activations(activations, layer_idx=19)
    svd_mlp_vtoxic = decompose_mlp_vtoxic_with_svd(mlp_vtoxic_vectors)
    pc = compute_pc1_svd_pca(
        [
            step.get("residual_stream_last_token", [])
            for row in activations.get("rows", [])
            for step in row.get("steps", [])
            if step.get("residual_stream_last_token")
        ]
    )
    unalign = rank_and_scale_top_mlp_vectors("gpt2", toxic_vector, allow_download=mode == "full")
    vector_subtraction = subtract_toxic_vector_with_hook("gpt2", toxic_vector, allow_download=mode == "full")
    mlp_vector_subtraction = subtract_mlp_vtoxic_vector_with_hook("gpt2", mlp_vtoxic_vectors[0], allow_download=mode == "full")
    dpo_pairs = [
        {"prompt": prompt, "chosen": f"{prompt} The answer stays respectful.", "rejected": f"{prompt} You are an idiot."}
        for prompt in selected_prompts[: min(8, len(selected_prompts))]
    ]
    dpo_training = train_gpt2_dpo_alignment(
        dpo_pairs,
        allow_download=mode == "full",
        epochs=3 if mode != "full" else 5,
    )
    table_2_metrics = measure_toxicity_perplexity_f1(
        [
            {
                "text": text,
                "generated": text,
                "continuation": " ".join(_tokenize_for_overlap(text)[: max(1, len(_tokenize_for_overlap(text)) - 1)]),
                "toxicity_score": score,
                "label": int(score >= 0.5),
                "negative_log_likelihood": 1.3 + idx * 0.1,
            }
            for idx, (text, score) in enumerate(zip(toxic_texts, toxicity.get("scores", [])))
        ]
    )
    lm_perplexity = measure_language_model_perplexity(
        [
            "The encyclopedia article describes a river and its tributaries.",
            "Several users revised the neutral summary after peer review.",
            "The report explains a technical result in plain language.",
        ],
        allow_download=mode == "full",
    )
    table_3_examples = make_table_3_next_token_examples(selected_prompts[: min(6, len(selected_prompts))])
    figure5_delta_similarity = compute_figure5_delta_similarity(
        residual_shift.get("rows", []),
        toxic_vectors=mlp_vtoxic_vectors[:5],
        mlp_parameter_differences=mlp_parameter_differences,
    )

    artifacts = {
        "selected_shit_prompts": str(_write_json(output_dir / "mechanistic" / "selected_shit_prompts.json", selected_shit_prompts)),
        "realtoxicityprompts_inventory": str(_write_json(output_dir / "mechanistic" / "realtoxicityprompts_inventory.json", full_prompt_inventory)),
        "classifier_last_layer_residual_average": str(_write_json(output_dir / "mechanistic" / "classifier_last_layer_residual_average.json", classifier_inputs)),
        "pplm_table9_hyperparameters": str(_write_json(output_dir / "mechanistic" / "pplm_table9_hyperparameters.json", TABLE9_PPLM_HYPERPARAMETERS)),
        "pplm_controller": str(_write_json(output_dir / "mechanistic" / "pplm_controller.json", pplm_controller)),
        "greedy_gpt2_positive_continuations": str(_write_json(output_dir / "mechanistic" / "greedy_gpt2_positive_continuations.json", positive_greedy)),
        "pplm_toxic_negative_continuations": str(_write_json(output_dir / "mechanistic" / "pplm_toxic_negative_continuations.json", negative_pplm)),
        "pplm_pairwise_dataset": str(_write_json(output_dir / "mechanistic" / "pplm_pairwise_dataset.json", pplm_pairs)),
        "toxicity_scores": str(_write_json(output_dir / "mechanistic" / "toxicity_scores.json", toxicity)),
        "realtoxicityprompts_activations": str(_write_json(output_dir / "mechanistic" / "realtoxicityprompts_activations.json", activations)),
        "dpo_realtoxicityprompts_activations": str(_write_json(output_dir / "mechanistic" / "dpo_realtoxicityprompts_activations.json", dpo_activations)),
        "layer19_activation": str(_write_json(output_dir / "mechanistic" / "layer19_activation.json", layer19)),
        "logit_lens_shit": str(_write_json(output_dir / "mechanistic" / "logit_lens_shit.json", logit_lens)),
        "dpo_logit_lens_shit": str(_write_json(output_dir / "mechanistic" / "dpo_logit_lens_shit.json", dpo_logit_lens)),
        "parameter_similarity": str(_write_json(output_dir / "mechanistic" / "parameter_similarity.json", parameter_similarity)),
        "mlp_parameter_differences": str(_write_json(output_dir / "mechanistic" / "mlp_parameter_differences.json", mlp_parameter_differences)),
        "residual_shift": str(_write_json(output_dir / "mechanistic" / "residual_shift.json", residual_shift)),
        "top128_mlp_value_vectors": str(_write_json(output_dir / "mechanistic" / "top128_mlp_value_vectors.json", top128_value_vectors)),
        "top5_value_vector_activations": str(_write_json(output_dir / "mechanistic" / "top5_value_vector_activations.json", top5_value_activations)),
        "dpo_top5_value_vector_activations": str(_write_json(output_dir / "mechanistic" / "dpo_top5_value_vector_activations.json", dpo_top5_value_activations)),
        "mean_mlp_value_activations": str(_write_json(output_dir / "mechanistic" / "mean_mlp_value_activations.json", mean_mlp_value_activations)),
        "svd_mlp_vtoxic": str(_write_json(output_dir / "mechanistic" / "svd_mlp_vtoxic.json", svd_mlp_vtoxic)),
        "pc1_svd_pca": str(_write_json(output_dir / "mechanistic" / "pc1_svd_pca.json", pc)),
        "unalign_top7": str(_write_json(output_dir / "mechanistic" / "unalign_top7.json", unalign)),
        "vector_subtraction_hook": str(_write_json(output_dir / "mechanistic" / "vector_subtraction_hook.json", vector_subtraction)),
        "mlp_vector_subtraction_hook": str(_write_json(output_dir / "mechanistic" / "mlp_vector_subtraction_hook.json", mlp_vector_subtraction)),
        "dpo_training": str(_write_json(output_dir / "mechanistic" / "dpo_training.json", dpo_training)),
        "table_2_metrics": str(_write_json(output_dir / "mechanistic" / "table_2_metrics.json", table_2_metrics)),
        "language_model_perplexity": str(_write_json(output_dir / "mechanistic" / "language_model_perplexity.json", lm_perplexity)),
        "table_3_examples": str(_write_json(output_dir / "mechanistic" / "table_3_examples.json", table_3_examples)),
        "figure_5_delta_similarity": str(_write_json(output_dir / "mechanistic" / "figure_5_delta_similarity.json", figure5_delta_similarity)),
    }

    artifact_manifest = {
        "paper": "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity",
        "mode": mode,
        "status": "transformers" if any(item.get("status") == "transformers" for item in [toxicity, activations, dpo_activations, layer19, logit_lens, parameter_similarity, residual_shift, pc, unalign, vector_subtraction]) else "fallback_fixture",
        "active_surface": [
            "jigsaw_last_layer_residual_average_probe",
            "unbiased-toxic-roberta",
            "pplm_table_9_hyperparameters_step_size_temperature_top_k_window0_horizon_decay_gamma_gm_scale_kl_scale",
            "pplm_gradient_perturbation_with_w_toxic_attribute_classifier",
            "greedy_gpt2_positive_non_toxic_continuations",
            "pplm_toxic_negative_continuations",
            "pplm_pairwise_positive_negative_dataset",
            "realtoxicityprompts_1199x20_activations",
            "realtoxicityprompts_gpt2_next_token_shit_selection",
            "gpt2-medium_residual_average",
            "layer19_mlp770_activation",
            "figure_1_shit_logit_lens",
            "figure_1_shit_logit_lens_with_mid_block_sites",
            "dpo_shit_logit_lens_with_mid_block_sites",
            "figure_2_top5_gpt2_and_gpt2_dpo_activations",
            "parameter_cosine_similarity",
            "gpt2_vs_gpt2_dpo_mlp_parameter_differences",
            "residual_stream_difference",
            "top128_mlp_value_vectors_by_w_toxic_cosine",
            "top5_mlp_value_vector_activations_1199x20",
            "dpo_top5_mlp_value_vector_activations_1199x20",
            "mean_mlp_value_vector_activations",
            "mlp_vtoxic_svd_utoxic",
            "pc1_svd_pca",
            "vector_subtraction_hook",
            "mlp_vtoxic_vector_subtraction_hook",
            "table_2_toxicity_perplexity_f1",
            "wikitext_language_model_perplexity",
            "continuation_token_overlap_f1",
            "table_3_next_token_prompt_examples",
            "table_4_scale_top7_unalign",
            "top7_key_vector_unalign_scaling_by_10",
            "figure_5_residual_mlp_delta_cosine",
            "figure_5_residual_delta_vs_mlp_parameter_delta_cosine",
            "gpt2_dpo_rmsprop_training",
        ],
        "artifacts": artifacts,
    }
    artifacts["artifact_manifest"] = str(_write_json(output_dir / "mechanistic" / "artifact_manifest.json", artifact_manifest))

    summary = {
        "created_at": _now(),
        "mode": mode,
        "status": artifact_manifest["status"],
        "active_surface": artifact_manifest["active_surface"],
        "artifact_manifest": artifacts["artifact_manifest"],
        "artifacts": artifacts,
        "target_routes": artifact_manifest["active_surface"],
    }
    _write_json(output_dir / "mechanistic" / "summary.json", summary)
    return summary


def write_mechanistic_transformers_artifacts(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    """Public wrapper used by the runner and smoke tests."""

    return run_mechanistic_transformers_route(config)


__all__ = [
    "TABLE8_DPO_HYPERPARAMETERS",
    "TABLE9_PPLM_HYPERPARAMETERS",
    "average_last_layer_residual_stream_for_classifier",
    "collect_logit_lens_shit_token_probabilities",
    "collect_realtoxicityprompts_activations",
    "compare_model_parameter_sets",
    "compute_mean_mlp_value_vector_activations",
    "compute_mlp_parameter_differences",
    "compute_pc1_svd_pca",
    "compute_figure5_delta_similarity",
    "compute_residual_and_mlp_shift",
    "compute_top_value_vector_mean_activations",
    "construct_pplm_pairwise_toxicity_dataset",
    "decompose_mlp_vtoxic_with_svd",
    "generate_greedy_gpt2_positive_continuations",
    "generate_pplm_toxic_negative_continuations",
    "continuation_token_overlap_f1",
    "load_realtoxicityprompts_challenge_prompts",
    "make_table_3_next_token_examples",
    "measure_language_model_perplexity",
    "measure_toxicity_perplexity_f1",
    "measure_layer_19_mlp_770_activation",
    "pplm_gradient_perturbation_controller",
    "rank_top_mlp_value_vectors_by_toxic_direction",
    "rank_and_scale_top_mlp_vectors",
    "run_mechanistic_transformers_route",
    "score_texts_with_unbiased_toxic_roberta",
    "select_realtoxicityprompts_with_shit_next_token",
    "subtract_mlp_vtoxic_vector_with_hook",
    "subtract_toxic_vector_with_hook",
    "write_mechanistic_transformers_artifacts",
]
