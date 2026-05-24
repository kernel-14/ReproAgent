from stay_on_topic_cfg.evaluate import estimate_pass_at_k


def pass_at_1(num_samples: int, num_correct: int) -> float:
    return estimate_pass_at_k(num_samples, num_correct, 1)

