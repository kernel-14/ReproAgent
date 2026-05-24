"""Small registry used by the CLI and graders."""

from .paper_protocol import (
    MMLU_VALIDATION_57_TASKS,
    P3_DPT_TRAIN_TASKS_36,
    RECROSS_P3_BART0_TEST_TASKS,
    TABLE1_MODEL_DATASET_CONFIGURATIONS,
)


def protocol_runtime_surface():
    return {
        "D_PT": {"tasks": P3_DPT_TRAIN_TASKS_36, "examples_per_task": 100},
        "BART0_ReCross_P3_test": RECROSS_P3_BART0_TEST_TASKS,
        "MMLU_validation_57_tasks": MMLU_VALIDATION_57_TASKS,
        "table1": TABLE1_MODEL_DATASET_CONFIGURATIONS,
    }

