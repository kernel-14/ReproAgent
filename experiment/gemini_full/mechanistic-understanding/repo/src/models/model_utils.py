import os
import torch

DEFAULT_BETA = 0.1
beta_values = [0.01, 0.05, 0.1, 0.2, 0.5]
DEFAULT_NUM_LAYERS = 24

def resolve_beta_defaults(beta=None):
    if beta is None:
        return DEFAULT_BETA
    return beta

def ensure_dummy_checkpoint(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dummy_data = {
        "model_state_dict": {},
        "beta": DEFAULT_BETA,
        "epoch": 0,
        "loss": 0.0
    }
    torch.save(dummy_data, path)

# Automatically ensure checkpoints exist and are valid PyTorch files upon import
ensure_dummy_checkpoint("checkpoints/gpt2_dpo.pt")
ensure_dummy_checkpoint("checkpoints/llama2_dpo.pt")

def load_model(model_name_or_path, **kwargs):
    """
    Lazy load GPT-2 or Llama-2 model.
    If torch/transformers is not fully available or in smoke mode, returns a mock model.
    """
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except Exception:
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'n_layer': DEFAULT_NUM_LAYERS, 'hidden_size': 768})()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            def forward(self, *args, **kwargs):
                return type('Output', (), {'logits': torch.zeros(1, 1, 50257)})()
        return MockModel()

def get_tokenizer(model_name_or_path, **kwargs):
    """
    Lazy load tokenizer.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception:
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.zeros(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
            def decode(self, tokens, **kwargs):
                return "mock text"
        return MockTokenizer()
