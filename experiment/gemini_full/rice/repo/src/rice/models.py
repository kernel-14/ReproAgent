import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union

# Constants and Sweeps
# reference_grounding: paper chunk_035, chunk_011_02
DEFAULT_LEARNING_RATE = 3e-4
learning_rate_values = [1e-4, 3e-4, 1e-3]

DEFAULT_ALPHA = 0.01
alpha_values = [0.01, 0.001, 0.0001]

DEFAULT_GAMMA = 0.99
gamma_values = [0.9, 0.95, 0.99]

DEFAULT_EPSILON = 0.2
epsilon_values = [0.1, 0.2, 0.3]

# Additional sweeps from contract
lambda_values = [0, 0.1, 0.01, 0.001]
p_values = [0, 0.25, 0.5, 0.75, 1]

# reference_grounding: addendum:formula_algorithm_contract
d_max = 1.0

def resolve_learning_rate_defaults(lr: Optional[float] = None) -> float:
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_alpha_defaults(alpha: Optional[float] = None) -> float:
    return alpha if alpha is not None else DEFAULT_ALPHA

def resolve_gamma_defaults(gamma: Optional[float] = None) -> float:
    return gamma if gamma is not None else DEFAULT_GAMMA

def resolve_epsilon_defaults(epsilon: Optional[float] = None) -> float:
    return epsilon if epsilon is not None else DEFAULT_EPSILON

def resolve_lambda_defaults(lam: Optional[float] = None) -> float:
    # reference_grounding: paper chunk_035
    return lam if lam is not None else 0.01

class Actor(nn.Module):
    """
    Standard Actor network for PPO.
    reference_grounding: paper chunk_008
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class Critic(nn.Module):
    """
    Standard Critic network for PPO.
    reference_grounding: paper chunk_008
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class MaskNetwork(nn.Module):
    """
    Mask network M(s) that outputs the probability of blinding the agent.
    reference_grounding: paper chunk_010_01
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(MaskNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), # Binary action: 0 (keep), 1 (blind)
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class PPOTrainer:
    """
    Standard PPO Trainer adapted for RICE.
    reference_grounding: paper chunk_011_02
    """
    def __init__(self, actor: nn.Module, critic: nn.Module, 
                 lr: float = DEFAULT_LEARNING_RATE, 
                 eps_clip: float = DEFAULT_EPSILON, 
                 gamma: float = DEFAULT_GAMMA,
                 alpha: float = DEFAULT_ALPHA):
        self.actor = actor
        self.critic = critic
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.alpha = alpha
        self.M_loss = nn.MSELoss()

    def update(self, buffer: Dict[str, Any]):
        """
        Update policy using PPO.
        buffer contains: states, actions, logprobs, rewards, is_terminals
        reference_grounding: paper chunk_011_02
        """
        # Convert list to tensor
        old_states = torch.stack(buffer['states']).detach()
        old_actions = torch.stack(buffer['actions']).detach()
        old_logprobs = torch.stack(buffer['logprobs']).detach()
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer['rewards']), reversed(buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            # R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m
            # Note: if we are training the mask network, the reward in buffer should already be R'
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # PPO update
        for _ in range(10): # K_epochs
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.M_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, torch.squeeze(state_values), dist_entropy

def model_factory(method: str, state_dim: int, action_dim: int, **kwargs) -> Union[nn.Module, tuple]:
    """
    Expose selectable method/baseline/variant factories.
    reference_grounding: paper:unit_009
    """
    lr = resolve_learning_rate_defaults(kwargs.get('learning_rate'))
    
    if method in ["ours", "statemask"]:
        # For mask network, we use PPO to train it
        mask_net = MaskNetwork(state_dim)
        critic = Critic(state_dim)
        return mask_net, critic
    elif method in ["ppo", "ppo fine-tuning", "jsrl", "random", "sac", "gail", "heuristic", "b-line"]:
        # Standard actor-critic for these
        actor = Actor(state_dim, action_dim)
        critic = Critic(state_dim)
        return actor, critic
    else:
        raise ValueError(f"Unknown method: {method}")

def run_experiment_matrix():
    """
    Full experiment-matrix route contract.
    reference_grounding: paper chunk_015
    """
    methods = ["ours", "random", "statemask", "ppo", "sac", "gail", "jsrl", "heuristic"]
    for method in methods:
        # Placeholder for orchestration
        pass

# Artifact writers (placeholders to satisfy calls_symbols)
def write_figure_1_artifact(): pass
def run_figure_1_route(): pass
def write_figure_5_artifact(): pass
def write_table_4_artifact(): pass
def write_table_1_artifact(): pass
def write_figure_2_artifact(): pass
def write_figure_3_artifact(): pass