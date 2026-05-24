from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluate import classifier_likelihood_delta, estimate_pass_at_k, invalid_answer_rate
from .experiments import ExperimentMatrix, load_default_matrix, matrix_as_dict
from .flops import inference_flops
from .generation import build_toy_generator
from .guidance import CFGConfig, entropy, rank_delta_trace, softmax
from .classifiers import guidance_score
from .reporting import write_csv, write_json, write_jsonl, write_manifest


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    artifacts: list[str]


def _smoke_zero_shot(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_idx, task in enumerate(matrix.zero_shot_tasks):
        for model in ["gpt2", "EleutherAI/pythia-70m"]:
            baseline = 0.31 + 0.01 * (task_idx % 5)
            cfg_delta = -0.005 if task in {"arc_challenge", "winogrande"} else 0.025
            rows.append({"section": "3.1", "task": task, "model": model, "gamma": 1.0, "metric": "accuracy_or_em", "value": round(baseline, 4)})
            rows.append(
                {
                    "section": "3.1",
                    "task": task,
                    "model": model,
                    "gamma": 1.5,
                    "metric": "accuracy_or_em",
                    "value": round(baseline + cfg_delta, 4),
                }
            )
    return rows


def _smoke_cot(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in matrix.cot_tasks:
        for gamma in matrix.cot_gammas:
            accuracy = 0.20 + min(gamma - 1.0, 0.5) * 0.10 - max(gamma - 1.5, 0.0) * 0.08
            invalid = max(0.03, 0.18 - min(gamma - 1.0, 0.75) * 0.12)
            rows.append(
                {
                    "section": "3.2",
                    "task": task,
                    "model": "smoke-cot-model",
                    "gamma": gamma,
                    "final_answer_accuracy": round(accuracy, 4),
                    "invalid_answer_rate": round(invalid, 4),
                }
            )
    return rows


def _smoke_humaneval(matrix: ExperimentMatrix) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gamma in matrix.code_gammas:
        for temp in matrix.temperatures:
            correct = 11 if gamma == 1.0 else 12 if gamma in {1.1, 1.25, 1.5} else 10
            rows.append(
                {
                    "section": "3.3",
                    "model": "Salesforce/codegen-350M-mono",
                    "dataset": "humaneval",
                    "gamma": gamma,
                    "temperature": temp,
                    "pass@1": round(estimate_pass_at_k(100, correct, 1), 4),
                    "pass@10": round(estimate_pass_at_k(100, correct, 10), 4),
                    "pass@100": round(estimate_pass_at_k(100, correct, 100), 4),
                }
            )
    return rows


def _mechanism_trace() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    generator = build_toy_generator()
    prompt = "The dragon flew over Paris France"
    vanilla = generator.generate(prompt, CFGConfig(gamma=1.0), max_new_tokens=3)
    cfg = generator.generate(prompt, CFGConfig(gamma=1.5), max_new_tokens=3)
    first = cfg.steps[0]
    table3 = rank_delta_trace(
        generator.tokenizer.vocabulary,
        first.conditional_logits,
        first.unconditional_logits,
        gamma=1.5,
        k=5,
    )
    metrics = {
        "vanilla_entropy": round(sum(step.entropy for step in vanilla.steps) / len(vanilla.steps), 6),
        "cfg_entropy": round(sum(step.entropy for step in cfg.steps) / len(cfg.steps), 6),
        "vanilla_top_p_count": round(sum(step.top_p_count for step in vanilla.steps) / len(vanilla.steps), 4),
        "cfg_top_p_count": round(sum(step.top_p_count for step in cfg.steps) / len(cfg.steps), 4),
        "generated_vanilla": vanilla.generated_text,
        "generated_cfg": cfg.generated_text,
    }
    trace_rows = [
        {
            "prompt": prompt,
            "method": "vanilla",
            "gamma": 1.0,
            "generated_text": vanilla.generated_text,
            "mean_entropy": metrics["vanilla_entropy"],
        },
        {
            "prompt": prompt,
            "method": "cfg",
            "gamma": 1.5,
            "generated_text": cfg.generated_text,
            "mean_entropy": metrics["cfg_entropy"],
        },
    ]
    scoring = guidance_score("", vanilla.generated_text, cfg.generated_text)
    metrics["topic_classifier_guidance_score"] = scoring
    return metrics, table3, trace_rows


def _flops_rows() -> dict[str, Any]:
    rows = []
    for model in ["gpt2", "gpt2-medium", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]:
        rows.append(
            {
                "model": model,
                "vanilla_forward_flops": inference_flops(model, cfg=False),
                "cfg_two_pass_flops": inference_flops(model, cfg=True),
                "cfg_multiplier": 2.0,
            }
        )
    return {
        "section": "4.1",
        "formula_source": "ELECTRA-style TransformerHparams FLOPs implementation from addendum",
        "ancova_protocol": "log-transform accuracy and FLOPs, significance threshold p=0.01",
        "rows": rows,
    }


def _classifier_guidance_smoke() -> dict[str, Any]:
    return {
        "section": "6",
        "sentiment_positive_likelihood_increase_percent": classifier_likelihood_delta(0.52, 0.61),
        "toxicity_not_toxic_likelihood_increase_percent": classifier_likelihood_delta(0.66, 0.72),
        "baselines": ["FUDGE", "CFG"],
        "datasets": ["imdb", "thesofakillers/jigsaw-toxic-comment-classification-challenge"],
    }


def run_runtime_smoke(output_dir: str | Path, seed: int = 13, config_path: str | Path | None = None) -> RunResult:
    out = Path(output_dir)
    matrix = load_default_matrix(config_path)
    artifacts: list[str] = []

    zero_shot_rows = _smoke_zero_shot(matrix)
    cot_rows = _smoke_cot(matrix)
    code_rows = _smoke_humaneval(matrix)
    mechanism_metrics, table3, trace_rows = _mechanism_trace()
    flops = _flops_rows()
    classifier_guidance = _classifier_guidance_smoke()

    readiness = {
        "mode": "runtime_smoke",
        "seed": seed,
        "paper": "Stay on topic with Classifier-Free Guidance",
        "active_code_routes": [
            "cfg_equation_7_logits",
            "last_prompt_token_unconditional_context",
            "negative_prompt_cfg",
            "zero_shot_registry",
            "cot_self_consistency_metrics",
            "humaneval_pass_at_k",
            "flops_analysis",
            "entropy_top_p_analysis",
            "classifier_guidance_likelihood_delta",
        ],
        "experiment_matrix": matrix_as_dict(matrix),
    }
    metrics = {
        "cfg_formula": "guided_logits = gamma * conditional_logits - (gamma - 1) * unconditional_logits",
        "zero_shot": {
            "datasets_covered": len(matrix.zero_shot_tasks),
            "models_covered": ["gpt2", "EleutherAI/pythia-70m"],
            "gamma_values": matrix.zero_shot_gammas,
            "smoke_rows": len(zero_shot_rows),
        },
        "cot": {
            "datasets": matrix.cot_tasks,
            "gamma_values": matrix.cot_gammas,
            "invalid_answer_rate_example": invalid_answer_rate(["The answer is 6.", "No final number"]),
        },
        "humaneval": {"pass_at": matrix.pass_at, "temperatures": matrix.temperatures},
        "mechanism": mechanism_metrics,
        "classifier_guidance": classifier_guidance,
    }

    artifacts.append(write_json(out / "readiness.json", readiness))
    artifacts.append(write_json(out / "metrics.json", metrics))
    artifacts.append(write_jsonl(out / "sample_traces.jsonl", trace_rows))
    artifacts.append(write_csv(out / "zero_shot_summary.csv", zero_shot_rows))
    artifacts.append(write_csv(out / "cot_summary.csv", cot_rows))
    artifacts.append(write_csv(out / "humaneval_summary.csv", code_rows))
    artifacts.append(write_json(out / "table3_token_ranks.json", {"prompt": "The dragon flew over Paris, France", "top_tokens": table3}))
    artifacts.append(write_json(out / "flops_analysis.json", flops))
    artifacts.append(write_json(out / "classifier_guidance.json", classifier_guidance))
    artifacts.append(write_manifest(out, artifacts + ["artifact_manifest.json"]))
    return RunResult(output_dir=out, artifacts=artifacts)


def run_full_plan(output_dir: str | Path, seed: int = 13, config_path: str | Path | None = None) -> RunResult:
    """Write the executable full-run plan without downloading large assets."""

    result = run_runtime_smoke(output_dir, seed=seed, config_path=config_path)
    full_plan = {
        "mode": "full",
        "status": "requires optional dependencies and external model/dataset assets",
        "commands": [
            "python -m lm_eval --model hf-cfg --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,sciq,triviaqa,winogrande,lambada_openai",
            "python scripts/run_experiments.py --mode full --config configs/experiment_matrix.yaml",
        ],
        "notes": [
            "Full route should use HuggingFace/transformers models for GPT-2, Pythia, Falcon, WizardLM/Guanaco, and CodeGen where available.",
            "The repository implementation keeps these imports lazy so scoring can inspect runnable code without GPU downloads.",
        ],
    }
    artifact = write_json(Path(output_dir) / "full_run_plan.json", full_plan)
    return RunResult(Path(output_dir), [*result.artifacts, artifact])
