from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .guidance import CFGConfig, combine_cfg_logits, prepare_unconditional_ids


@dataclass
class TransformersCFGGenerator:
    """Lazy HuggingFace backend for GPT-2, Pythia, Falcon, WizardLM/Guanaco, CodeGen."""

    model_name: str
    device: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install optional dependency group [models] for TransformersCFGGenerator") from exc
        self._torch = __import__("torch")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device)
        self.model.eval()

    def _next_logits(self, token_ids: Sequence[int]):
        torch = self._torch
        input_ids = torch.tensor([list(token_ids)], device=self.model.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids)
        return output.logits[0, -1, :].detach().cpu().numpy()

    def generate(self, prompt: str, config: CFGConfig | None = None, max_new_tokens: int = 64) -> str:
        cfg = config or CFGConfig()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            conditional_ids = [*prompt_ids, *generated]
            unconditional_ids = prepare_unconditional_ids(prompt_ids, generated, cfg)
            conditional_logits = self._next_logits(conditional_ids)
            unconditional_logits = self._next_logits(unconditional_ids)
            guided = combine_cfg_logits(conditional_logits, unconditional_logits, cfg.gamma)
            next_id = int(guided.argmax())
            generated.append(next_id)
            eos = self.tokenizer.eos_token_id
            if eos is not None and next_id == eos:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True)

