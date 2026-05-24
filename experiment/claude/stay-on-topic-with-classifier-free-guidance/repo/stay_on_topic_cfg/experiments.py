from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMatrix:
    zero_shot_tasks: list[str]
    zero_shot_models: list[str]
    zero_shot_gammas: list[float]
    cot_tasks: list[str]
    cot_models: list[str]
    cot_gammas: list[float]
    code_models: list[str]
    code_gammas: list[float]
    temperatures: list[float]
    pass_at: list[int]
    mechanism_models: list[str]
    classifier_datasets: dict[str, str]

    @property
    def model_routes(self) -> set[str]:
        return set(self.zero_shot_models + self.cot_models + self.code_models + self.mechanism_models)


def load_default_matrix(config_path: str | Path | None = None) -> ExperimentMatrix:
    # The default is deliberately in code so import/runtime smoke never depends
    # on PyYAML. The YAML file mirrors these values for researchers.
    return ExperimentMatrix(
        zero_shot_tasks=[
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "piqa",
            "sciq",
            "triviaqa",
            "winogrande",
            "lambada_openai",
        ],
        zero_shot_models=["gpt2", "gpt2-medium", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"],
        zero_shot_gammas=[1.0, 1.5],
        cot_tasks=["gsm8k", "aqua"],
        cot_models=["WizardLM/WizardLM-30B", "timdettmers/guanaco-65b"],
        cot_gammas=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        code_models=["Salesforce/codegen-350M-mono", "Salesforce/codegen-2B-mono", "Salesforce/codegen-6B-mono"],
        code_gammas=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        temperatures=[0.2, 0.6, 0.8],
        pass_at=[1, 10, 100],
        mechanism_models=["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"],
        classifier_datasets={
            "sentiment": "imdb",
            "toxicity": "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        },
    )


def matrix_as_dict(matrix: ExperimentMatrix) -> dict[str, Any]:
    return {
        "zero_shot": {"tasks": matrix.zero_shot_tasks, "models": matrix.zero_shot_models, "gammas": matrix.zero_shot_gammas},
        "cot": {"tasks": matrix.cot_tasks, "models": matrix.cot_models, "gammas": matrix.cot_gammas},
        "code_generation": {
            "models": matrix.code_models,
            "gammas": matrix.code_gammas,
            "temperatures": matrix.temperatures,
            "pass_at": matrix.pass_at,
        },
        "mechanism": {"models": matrix.mechanism_models, "p3_sample_count": 32902},
        "classifier_guidance": matrix.classifier_datasets,
    }

