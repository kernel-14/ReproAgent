from __future__ import annotations

from dataclasses import dataclass

SOFTMAX_FLOPS = 5
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
OUTPUT_FRAC = 1.0


@dataclass(frozen=True)
class TransformerHparams:
    h: int
    layers: int
    seq_len: int = 512
    vocab: int = 50257
    intermediate: int | None = None
    heads: int | None = None

    def block_flops(self) -> float:
        i = self.intermediate or self.h * 4
        heads = self.heads or max(self.h // 64, 1)
        per_token = (
            3 * 2 * self.h * self.h
            + 3 * self.h
            + 2 * self.h * self.seq_len
            + SOFTMAX_FLOPS * self.seq_len * heads
            + 2 * self.h * self.seq_len
            + 2 * self.h * self.h
            + 2 * self.h * i
            + ACTIVATION_FLOPS * i
            + 2 * self.h * i
            + LAYER_NORM_FLOPS * self.h * 2
        )
        return per_token * self.seq_len

    def output_flops(self) -> float:
        return OUTPUT_FRAC * (2 * self.h * self.vocab + SOFTMAX_FLOPS * self.vocab + 2 * self.vocab) * self.seq_len

    def infer_flops(self) -> float:
        return self.layers * self.block_flops() + self.output_flops()


MODEL_HPARAMS = {
    "gpt2": TransformerHparams(h=768, layers=12),
    "gpt2-medium": TransformerHparams(h=1024, layers=24),
    "EleutherAI/pythia-70m": TransformerHparams(h=512, layers=6),
    "EleutherAI/pythia-160m": TransformerHparams(h=768, layers=12),
}


def inference_flops(model_name: str, cfg: bool = False) -> float:
    hparams = MODEL_HPARAMS[model_name]
    multiplier = 2.0 if cfg else 1.0
    return multiplier * hparams.infer_flops()

