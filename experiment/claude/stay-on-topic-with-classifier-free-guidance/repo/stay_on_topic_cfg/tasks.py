from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    source: str
    metric: str
    prompt_style: str


ZERO_SHOT_TASKS = {
    "arc_challenge": TaskSpec("arc_challenge", "allenai/ai2_arc:ARC-Challenge", "accuracy", "multiple_choice_loglikelihood"),
    "arc_easy": TaskSpec("arc_easy", "allenai/ai2_arc:ARC-Easy", "accuracy", "multiple_choice_loglikelihood"),
    "boolq": TaskSpec("boolq", "google/boolq", "accuracy", "yes_no_loglikelihood"),
    "hellaswag": TaskSpec("hellaswag", "rowanzellers/hellaswag", "accuracy", "multiple_choice_loglikelihood"),
    "piqa": TaskSpec("piqa", "ybisk/piqa", "accuracy", "multiple_choice_loglikelihood"),
    "sciq": TaskSpec("sciq", "allenai/sciq", "accuracy", "multiple_choice_loglikelihood"),
    "triviaqa": TaskSpec("triviaqa", "mandarjoshi/trivia_qa", "exact_match", "short_answer"),
    "winogrande": TaskSpec("winogrande", "allenai/winogrande", "accuracy", "multiple_choice_loglikelihood"),
    "lambada_openai": TaskSpec("lambada_openai", "EleutherAI/lambada_openai", "accuracy", "last_word_prediction"),
}

COT_TASKS = {
    "gsm8k": TaskSpec("gsm8k", "openai/gsm8k", "final_answer_accuracy_and_invalid_rate", "wang_2023_few_shot_cot"),
    "aqua": TaskSpec("aqua", "nguyen-brat/aqua", "final_answer_accuracy_and_invalid_rate", "wang_2023_few_shot_cot"),
}


def format_cot_prompt(question: str, task_id: str = "gsm8k") -> str:
    if task_id not in COT_TASKS:
        raise KeyError(f"unknown CoT task: {task_id}")
    return (
        "Q: There are 15 trees in the grove. Grove workers plant trees today. "
        "After they are done, there are 21 trees. How many trees did they plant?\n"
        "A: There are 21 - 15 = 6 trees planted. The answer is 6.\n\n"
        f"Q: {question}\nA: Let's think step by step."
    )


def format_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"### Instruction:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Response:\n"

