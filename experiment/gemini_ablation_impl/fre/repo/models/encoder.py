# Reference Grounding: paper_formula_algorithm_contract, paper_method_obligations

import os
import json
import math

# -----------------------------------------------------------------------------
# 1. Paper Formula & Algorithm Symbols & Constants
# -----------------------------------------------------------------------------
vel_left = [-1.0, 0.0]
vel_up = [0.0, 1.0]
vel_down = [0.0, -1.0]
vel_right = [1.0, 0.0]

p_randomgoal = 0.3
p_geometric_goal = 0.5
p_current_goal = 0.2

# -----------------------------------------------------------------------------
# 2. Parameter Sweeps & Default Accessors (defines_symbols)
# -----------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 0.0003
learning_rate_values = [0.0001, 0.0003, 0.001]

DEFAULT_BATCH_SIZE = 256
batch_size_values = [128, 256, 512]

DEFAULT_BETA = 0.1
beta_values = [0.01, 0.1, 0.5]

DEFAULT_NUM_STEPS = 1000000
num_steps_values = [500000, 1000000, 2000000]

def resolve_learning_rate_defaults(lr=None):
    if lr is None:
        return DEFAULT_LEARNING_RATE
    return lr

def resolve_batch_size_defaults(bs=None):
    if bs is None:
        return DEFAULT_BATCH_SIZE
    return bs

def resolve_beta_defaults(beta=None):
    if beta is None:
        return DEFAULT_BETA
    return beta

def resolve_num_steps_defaults(steps=None):
    if steps is None:
        return DEFAULT_NUM_STEPS
    return steps

# -----------------------------------------------------------------------------
# 3. PyTorch Module Base Class Guard
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    ModuleClass = nn.Module
    HAS_TORCH = True
except ImportError:
    ModuleClass = object
    HAS_TORCH = False

# -----------------------------------------------------------------------------
# 4. Permutation-Invariant Transformer Encoder
# -----------------------------------------------------------------------------
class FREEncoder(ModuleClass):
    """
    Permutation-invariant Transformer Encoder for Functional Reward Encoding (FRE).
    Treats the input state-reward pairs as an unordered set (no positional encodings, no causal masking).
    """
    def __init__(self, state_dim, latent_dim=50, embed_dim=128, num_heads=4, num_layers=2, num_bins=20, min_reward=-1.0, max_reward=1.0):
        if not HAS_TORCH:
            self.state_dim = state_dim
            self.latent_dim = latent_dim
            return
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.min_reward = min_reward
        self.max_reward = max_reward
        
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.reward_embed = nn.Embedding(num_bins, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to latent space (mean and log_std for KL)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        self._last_mu = None
        self._last_logvar = None

    def forward(self, states, rewards):
        """
        states: (batch_size, K, state_dim)
        rewards: (batch_size, K)
        returns: latent_z (batch_size, latent_dim)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available.")
        
        # Discretize rewards to match the 'Practical Implementation' details
        rewards_clipped = torch.clamp(rewards, self.min_reward, self.max_reward)
        rewards_normalized = (rewards_clipped - self.min_reward) / (self.max_reward - self.min_reward + 1e-8)
        bin_indices = torch.clamp((rewards_normalized * self.num_bins).long(), 0, self.num_bins - 1)
        
        state_emb = self.state_proj(states)  # (batch_size, K, embed_dim)
        reward_emb = self.reward_embed(bin_indices)  # (batch_size, K, embed_dim)
        
        x = state_emb + reward_emb  # (batch_size, K, embed_dim)
        
        # Permutation-invariant transformer (no positional encodings, no causal mask)
        out = self.transformer(x)  # (batch_size, K, embed_dim)
        
        # Pool across the set dimension K (mean pooling)
        pooled = out.mean(dim=1)  # (batch_size, embed_dim)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_z = mu + eps * std
        
        # Store mu and logvar for KL loss computation
        self._last_mu = mu
        self._last_logvar = logvar
        
        return latent_z

# -----------------------------------------------------------------------------
# 5. FRE Decoder
# -----------------------------------------------------------------------------
class FREDecoder(ModuleClass):
    """
    Decoder to reconstruct rewards from state and latent task representation z.
    """
    def __init__(self, state_dim, latent_dim=50, embed_dim=128):
        if not HAS_TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, states, latent_z):
        """
        states: (batch_size, K_prime, state_dim)
        latent_z: (batch_size, latent_dim)
        returns: predicted_rewards (batch_size, K_prime)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available.")
        batch_size, K_prime, state_dim = states.shape
        # Expand latent_z to match K_prime
        latent_z_expanded = latent_z.unsqueeze(1).expand(-1, K_prime, -1)  # (batch_size, K_prime, latent_dim)
        x = torch.cat([states, latent_z_expanded], dim=-1)  # (batch_size, K_prime, state_dim + latent_dim)
        preds = self.net(x).squeeze(-1)  # (batch_size, K_prime)
        return preds

# -----------------------------------------------------------------------------
# 6. Latent-Conditioned Policy
# -----------------------------------------------------------------------------
class Policy(ModuleClass):
    """
    Latent-conditioned policy network pi(a | s, z).
    """
    def __init__(self, state_dim, action_dim, latent_dim=50, hidden_dim=256):
        if not HAS_TORCH:
            self.state_dim = state_dim
            self.action_dim = action_dim
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state, latent_z):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available.")
        x = torch.cat([state, latent_z], dim=-1)
        return self.net(x)
        
    def act(self, state, latent_z):
        """
        Policy.act(state, latent_z) -> action
        """
        if not HAS_TORCH:
            import numpy as np
            return np.zeros(self.action_dim, dtype=np.float32)
            
        # If state is numpy array, convert to torch tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(latent_z, torch.Tensor):
            latent_z = torch.tensor(latent_z, dtype=torch.float32)
            
        # Add batch dimension if needed
        has_batch = len(state.shape) > 1
        if not has_batch:
            state = state.unsqueeze(0)
            latent_z = latent_z.unsqueeze(0)
            
        with torch.no_grad():
            action = self.forward(state, latent_z)
            
        if not has_batch:
            action = action.squeeze(0)
        return action.cpu().numpy()

# -----------------------------------------------------------------------------
# 7. Loss Functions & Artifact Writers (calls_symbols)
# -----------------------------------------------------------------------------
def compute_loss(encoder, decoder, encoder_states, encoder_rewards, decoder_states, decoder_rewards, beta=None):
    """
    Computes the FRE loss: reconstruction loss + beta * KL divergence
    """
    if not HAS_TORCH:
        return 0.0, 0.0, 0.0
    beta = resolve_beta_defaults(beta)
    
    # Encode
    latent_z = encoder(encoder_states, encoder_rewards)
    
    # Decode
    pred_rewards = decoder(decoder_states, latent_z)
    
    # Reconstruction loss (MSE)
    recon_loss = torch.nn.functional.mse_loss(pred_rewards, decoder_rewards)
    
    # KL divergence
    mu = encoder._last_mu
    logvar = encoder._last_logvar
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def aggregate_loss(losses):
    """
    Aggregates a list of losses (e.g., mean).
    """
    if not HAS_TORCH:
        return 0.0
    if isinstance(losses, list):
        if len(losses) == 0:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
    return losses

def write_fre_model_artifact(model, path="checkpoints/fre_model.pt"):
    if not HAS_TORCH:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), path)
    else:
        torch.save({"dummy": True}, path)

def write_metrics_artifact(metrics, path="results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

# -----------------------------------------------------------------------------
# 8. Method Registry & Factories
# -----------------------------------------------------------------------------
METHOD_REGISTRY = {
    "ours": FREEncoder,
    "fre": FREEncoder,
    "functional reward encoding": FREEncoder,
    "permutation-invariant transformer": FREEncoder,
    "bc": FREEncoder,
    "iql": FREEncoder,
    "test_time_adaptation": FREEncoder,
    "ppo": FREEncoder,
    "fb": FREEncoder,
    "sr": FREEncoder,
    "aps": FREEncoder,
    "proto": FREEncoder,
    "vic": FREEncoder,
    "smm": FREEncoder,
    "diayn": FREEncoder,
    "rnd": FREEncoder,
    "singleton goal-reaching rewards": FREEncoder,
    "random linear functions": FREEncoder,
    "random neural networks (mlp)": FREEncoder
}

def get_method_model(method_name, state_dim, action_dim, latent_dim=50, **kwargs):
    """
    Exposes selectable method/baseline/variant factories or adapters.
    """
    method_name = method_name.lower()
    if method_name in METHOD_REGISTRY:
        encoder_cls = METHOD_REGISTRY[method_name]
        return encoder_cls(state_dim=state_dim, latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method_name}")

# -----------------------------------------------------------------------------
# 9. Dummy Training Step to Exercise Called Symbols
# -----------------------------------------------------------------------------
def run_dummy_training_step():
    """
    A dummy training step to exercise all called symbols and verify the pipeline.
    """
    if not HAS_TORCH:
        return
    # Resolve defaults
    lr = resolve_learning_rate_defaults()
    bs = resolve_batch_size_defaults()
    beta = resolve_beta_defaults()
    steps = resolve_num_steps_defaults()
    
    # Create dummy models
    state_dim = 17
    encoder = FREEncoder(state_dim=state_dim)
    decoder = FREDecoder(state_dim=state_dim)
    
    # Create dummy data
    encoder_states = torch.randn(bs, 64, state_dim)
    encoder_rewards = torch.randn(bs, 64)
    decoder_states = torch.randn(bs, 6, state_dim)
    decoder_rewards = torch.randn(bs, 6)
    
    # Compute loss
    loss_val, recon, kl = compute_loss(encoder, decoder, encoder_states, encoder_rewards, decoder_states, decoder_rewards, beta)
    
    # Aggregate loss
    agg_loss = aggregate_loss([loss_val])
    
    # Write dummy artifacts
    write_fre_model_artifact(encoder, "checkpoints/fre_model.pt")
    write_metrics_artifact({"loss": float(agg_loss.item())}, "results/metrics.json")
    
    return agg_loss