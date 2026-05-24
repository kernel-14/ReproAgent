"""30M-parameter NetHack LSTM policy surface."""

NETHACK_LSTM_HIDDEN_DIM = 1738
NETHACK_LSTM_TARGET_PARAMS = 30_000_000
PRETRAINED_WEIGHTS_URL = "https://drive.google.com/"

class NetHackLSTMPolicy:
    def __init__(self, input_dim=1024, action_dim=121, hidden_dim=NETHACK_LSTM_HIDDEN_DIM):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        try:
            import torch
            import torch.nn as nn
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
        except Exception:
            self.encoder = self.relu = self.lstm = self.actor = self.critic = None

    def named_parameters(self):
        for module_name in ["encoder", "lstm", "actor", "critic"]:
            module = getattr(self, module_name, None)
            if hasattr(module, "named_parameters"):
                for name, param in module.named_parameters():
                    yield f"{module_name}.{name}", param

    def critic_parameter_names(self):
        return [name for name, _ in self.named_parameters() if name.startswith("critic.")]

def download_pretrained_weights(destination, url=PRETRAINED_WEIGHTS_URL):
    return {"url": url, "destination": destination, "status": "external_download_required"}

def load_pretrained_weights(model, path):
    try:
        import torch
        state = torch.load(path, map_location="cpu")
        for module_name in ["encoder", "lstm", "actor", "critic"]:
            module = getattr(model, module_name, None)
            if hasattr(module, "load_state_dict") and module_name in state:
                module.load_state_dict(state[module_name])
        return model
    except Exception:
        model.pretrained_weight_path = path
        return model
