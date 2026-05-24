from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .guidance import CFGConfig, combine_cfg_logits, entropy, prepare_unconditional_ids, softmax, top_p_token_count


class ToyTokenizer:
    """Small deterministic tokenizer for smoke tests and examples."""

    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary = list(vocabulary)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for raw in text.replace(",", " ,").replace(".", " .").split():
            token = raw.lower()
            ids.append(self.token_to_id.get(token, self.token_to_id["<unk>"]))
        return ids or [self.token_to_id["<bos>"]]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.vocabulary[int(i)] for i in ids)


@dataclass
class ToyTopicLM:
    """Tiny LM that makes topic adherence visible without external weights."""

    vocabulary: Sequence[str]
    topic_boosts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def next_logits(self, context_ids: Sequence[int], tokenizer: ToyTokenizer) -> np.ndarray:
        logits = np.linspace(0.0, -1.5, num=len(self.vocabulary), dtype=float)
        context_tokens = [tokenizer.vocabulary[int(i)] for i in context_ids if 0 <= int(i) < len(tokenizer.vocabulary)]
        for cue, boosts in self.topic_boosts.items():
            if cue in context_tokens:
                for token, boost in boosts.items():
                    if token in tokenizer.token_to_id:
                        logits[tokenizer.token_to_id[token]] += float(boost)
        return logits


@dataclass
class GenerationStep:
    token: str
    token_id: int
    entropy: float
    top_p_count: int
    conditional_logits: list[float]
    unconditional_logits: list[float]
    guided_logits: list[float]


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    generated_ids: list[int]
    steps: list[GenerationStep]


class CFGGenerator:
    """Model-agnostic generation wrapper implementing paper CFG semantics."""

    def __init__(self, model: ToyTopicLM, tokenizer: ToyTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, config: CFGConfig | None = None, max_new_tokens: int = 4) -> GenerationResult:
        cfg = config or CFGConfig()
        prompt_ids = self.tokenizer.encode(prompt)
        generated: list[int] = []
        steps: list[GenerationStep] = []
        for _ in range(max_new_tokens):
            conditional_context = [*prompt_ids, *generated]
            unconditional_context = prepare_unconditional_ids(prompt_ids, generated, cfg)
            conditional_logits = self.model.next_logits(conditional_context, self.tokenizer)
            unconditional_logits = self.model.next_logits(unconditional_context, self.tokenizer)
            guided_logits = combine_cfg_logits(conditional_logits, unconditional_logits, cfg.gamma)
            probs = softmax(guided_logits)
            next_id = int(np.argmax(probs))
            generated.append(next_id)
            steps.append(
                GenerationStep(
                    token=self.tokenizer.vocabulary[next_id],
                    token_id=next_id,
                    entropy=entropy(probs),
                    top_p_count=top_p_token_count(probs, cfg.top_p),
                    conditional_logits=conditional_logits.tolist(),
                    unconditional_logits=unconditional_logits.tolist(),
                    guided_logits=guided_logits.tolist(),
                )
            )
        return GenerationResult(
            prompt=prompt,
            generated_text=self.tokenizer.decode(generated),
            generated_ids=generated,
            steps=steps,
        )


def build_toy_generator() -> CFGGenerator:
    vocabulary = [
        "<bos>",
        "<unk>",
        "the",
        "dragon",
        "flew",
        "over",
        "paris",
        "france",
        "recipe",
        "football",
        "answer",
        "therefore",
        ".",
        ",",
    ]
    tokenizer = ToyTokenizer(vocabulary)
    model = ToyTopicLM(
        vocabulary=vocabulary,
        topic_boosts={
            "dragon": {"dragon": 3.0, "flew": 2.5, "over": 1.5},
            "paris": {"paris": 3.0, "france": 2.4, ",": 0.5},
            "question": {"answer": 2.0, "therefore": 1.2},
        },
    )
    return CFGGenerator(model=model, tokenizer=tokenizer)

