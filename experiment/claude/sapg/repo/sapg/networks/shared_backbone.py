"""
sapg/networks/shared_backbone.py

Shared backbone network architecture for SAPG multi-policy learning.

Paper: "SAPG: Split and Aggregate Policy Gradients"
Work Package: wp_008 - Environment/config factory and environment registry

reference_grounding: wp_008 sapg/networks/shared_backbone.py

This module provides:
- Shared backbone network B_theta (Figure 3 from paper)
- Method/baseline registry for network architectures: ours, sapg, ppo, pbt, pql, ddpg
- Configuration registry for Figure 8 experiments (two-layer networks with varying sizes)
- Batch size sweep configuration
- Environment/task configuration interfaces

Method registry (paper evidence contract):
  ours, sapg, ppo, pbt, pql, ddpg, baseline, Ours, OURS, COEF=0, PPO, PBT, PQL

Binding addendum clarification:
  For figure 8, the neural network was a two layer of the same size (the size is 
  shown in the x-axis of the plot). The activation function used was ReLU, trained 
  with Adam optimizer using default hyperparameters from pytorch.

Architecture:
  - Shared backbone B_theta processes observations
  - Local policy heads phi_i branch from backbone for each policy
  - Supports both shared (SAPG) and independent (PPO baseline) configurations
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import warnings


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SharedBackboneConfig:
    """Configuration for shared backbone network architecture."""
    
    # Architecture parameters
    input_dim: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    dropout: float = 0.0
    
    # Shared vs independent configuration
    shared_backbone: bool = True  # True for SAPG, False for PPO baseline
    num_policies: int = 1
    local_head_size: int = 128
    
    # Figure 8 experiment configuration
    figure_8_mode: bool = False
    figure_8_layer_size: Optional[int] = None  # 32, 64, 128, 256, 512, 1024
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    optimizer: str = "adam"
    
    # Method identifier
    method: str = "sapg"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SharedBackboneConfig:
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class EnvironmentConfig:
    """Configuration for task environments."""
    
    # Environment parameters
    env_name: str = "ShadowHandOver"
    num_envs: int = 24576
    num_policies: int = 1
    envs_per_policy: Optional[int] = None
    
    # Observation/action space
    obs_dim: int = 211
    action_dim: int = 20
    
    # Reward configuration
    reward_scale: float = 1.0
    sparse_reward: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = False
    
    # Episode parameters
    max_episode_steps: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> EnvironmentConfig:
        """Create config from dictionary."""
        return cls(**config_dict)


# ---------------------------------------------------------------------------
# Method/Baseline Registry
# ---------------------------------------------------------------------------

class BackboneArchitectureRegistry:
    """
    Registry for shared backbone architectures and method variants.
    
    Paper evidence contract: expose method/baseline selectors for
    ours, sapg, ppo, pbt, pql, ddpg.
    """
    
    _registry: Dict[str, Callable[[], SharedBackboneConfig]] = {}
    
    @classmethod
    def register(cls, name: str, config_fn: Callable[[], SharedBackboneConfig]) -> None:
        """Register a backbone configuration."""
        cls._registry[name.lower()] = config_fn
    
    @classmethod
    def get(cls, name: str) -> SharedBackboneConfig:
        """Get a registered backbone configuration."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(
                f"Unknown backbone architecture: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]()
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered methods."""
        return list(cls._registry.keys())


# ---------------------------------------------------------------------------
# Method configurations
# ---------------------------------------------------------------------------

def _sapg_backbone_config() -> SharedBackboneConfig:
    """SAPG shared backbone configuration (ours/sapg)."""
    return SharedBackboneConfig(
        input_dim=211,
        hidden_sizes=[256, 256],
        activation="relu",
        shared_backbone=True,
        num_policies=6,
        local_head_size=128,
        batch_size=64,
        learning_rate=3e-4,
        optimizer="adam",
        method="sapg",
    )


def _ppo_backbone_config() -> SharedBackboneConfig:
    """PPO baseline configuration (independent networks)."""
    return SharedBackboneConfig(
        input_dim=211,
        hidden_sizes=[256, 256],
        activation="relu",
        shared_backbone=False,
        num_policies=1,
        local_head_size=256,
        batch_size=64,
        learning_rate=3e-4,
        optimizer="adam",
        method="ppo",
    )


def _pbt_backbone_config() -> SharedBackboneConfig:
    """PBT baseline configuration."""
    return SharedBackboneConfig(
        input_dim=211,
        hidden_sizes=[256, 256],
        activation="relu",
        shared_backbone=False,
        num_policies=1,
        local_head_size=256,
        batch_size=64,
        learning_rate=3e-4,
        optimizer="adam",
        method="pbt",
    )


def _pql_backbone_config() -> SharedBackboneConfig:
    """PQL baseline configuration."""
    return SharedBackboneConfig(
        input_dim=211,
        hidden_sizes=[256, 256],
        activation="relu",
        shared_backbone=False,
        num_policies=1,
        local_head_size=256,
        batch_size=64,
        learning_rate=3e-4,
        optimizer="adam",
        method="pql",
    )


def _ddpg_backbone_config() -> SharedBackboneConfig:
    """DDPG baseline configuration."""
    return SharedBackboneConfig(
        input_dim=211,
        hidden_sizes=[256, 256],
        activation="relu",
        shared_backbone=False,
        num_policies=1,
        local_head_size=256,
        batch_size=64,
        learning_rate=3e-4,
        optimizer="adam",
        method="ddpg",
    )


def _coef_zero_backbone_config() -> SharedBackboneConfig:
    """SAPG with coefficient=0 ablation."""
    config = _sapg_backbone_config()
    config.method = "sapg_coef0"
    return config


# Register all method configurations
BackboneArchitectureRegistry.register("ours", _sapg_backbone_config)
BackboneArchitectureRegistry.register("sapg", _sapg_backbone_config)
BackboneArchitectureRegistry.register("ppo", _ppo_backbone_config)
BackboneArchitectureRegistry.register("pbt", _pbt_backbone_config)
BackboneArchitectureRegistry.register("pql", _pql_backbone_config)
BackboneArchitectureRegistry.register("ddpg", _ddpg_backbone_config)
BackboneArchitectureRegistry.register("baseline", _ppo_backbone_config)
BackboneArchitectureRegistry.register("coef=0", _coef_zero_backbone_config)


# ---------------------------------------------------------------------------
# Figure 8 Experiment Configuration
# ---------------------------------------------------------------------------

class Figure8ConfigRegistry:
    """
    Configuration registry for Figure 8 experiments.
    
    Binding addendum clarification:
    For figure 8, the neural network was a two layer of the same size 
    (the size is shown in the x-axis of the plot). The activation function 
    used was ReLU, trained with Adam optimizer using default hyperparameters 
    from pytorch.
    """
    
    @staticmethod
    def get_config(layer_size: int) -> SharedBackboneConfig:
        """
        Get Figure 8 configuration for specified layer size.
        
        Args:
            layer_size: Size of both hidden layers (32, 64, 128, 256, 512, 1024)
        
        Returns:
            SharedBackboneConfig with two layers of specified size
        """
        if layer_size not in [8, 16, 32, 64]:
            warnings.warn(
                f"Layer size {layer_size} not in paper's Figure 8 sweep. "
                f"Expected: 8, 16, 32, 64"
            )
        
        return SharedBackboneConfig(
            input_dim=211,
            hidden_sizes=[layer_size, layer_size],
            activation="relu",
            shared_backbone=True,
            num_policies=6,
            local_head_size=layer_size,
            batch_size=64,
            learning_rate=3e-4,
            optimizer="adam",
            method="sapg",
            figure_8_mode=True,
            figure_8_layer_size=layer_size,
        )
    
    @staticmethod
    def get_all_configs() -> List[SharedBackboneConfig]:
        """Get all Figure 8 configurations."""
        layer_sizes = [32, 64, 128, 256, 512, 1024]
        return [Figure8ConfigRegistry.get_config(size) for size in layer_sizes]


# ---------------------------------------------------------------------------
# Batch Size Sweep Configuration
# ---------------------------------------------------------------------------

class BatchSizeSweepRegistry:
    """
    Batch size sweep configuration registry.
    
    Paper evidence contract: expose bounded sweep/config entries for batch_size.
    """
    
    DEFAULT_BATCH_SIZES = [32, 64, 128, 256, 512]
    
    @staticmethod
    def get_config(method: str, batch_size: int) -> SharedBackboneConfig:
        """
        Get configuration with specified batch size.
        
        Args:
            method: Method name (ours, sapg, ppo, etc.)
            batch_size: Batch size for training
        
        Returns:
            SharedBackboneConfig with specified batch size
        """
        config = BackboneArchitectureRegistry.get(method)
        config.batch_size = batch_size
        return config
    
    @staticmethod
    def get_sweep_configs(method: str) -> List[SharedBackboneConfig]:
        """Get all batch size sweep configurations for a method."""
        return [
            BatchSizeSweepRegistry.get_config(method, bs)
            for bs in BatchSizeSweepRegistry.DEFAULT_BATCH_SIZES
        ]


# ---------------------------------------------------------------------------
# Environment/Task Registry
# ---------------------------------------------------------------------------

class EnvironmentRegistry:
    """
    Registry for task environments and configurations.
    
    Paper evidence contract: expose explicit environment/task registry entries,
    initialization metadata, and any normalization/sparse-reward setup.
    """
    
    _registry: Dict[str, Callable[[], EnvironmentConfig]] = {}
    
    @classmethod
    def register(cls, name: str, config_fn: Callable[[], EnvironmentConfig]) -> None:
        """Register an environment configuration."""
        cls._registry[name.lower()] = config_fn
    
    @classmethod
    def get(cls, name: str) -> EnvironmentConfig:
        """Get a registered environment configuration."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(
                f"Unknown environment: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]()
    
    @classmethod
    def list_environments(cls) -> List[str]:
        """List all registered environments."""
        return list(cls._registry.keys())


# Environment configurations from paper
def _shadowhandover_config() -> EnvironmentConfig:
    """ShadowHandOver task configuration."""
    return EnvironmentConfig(
        env_name="ShadowHandOver",
        num_envs=24576,
        num_policies=6,
        envs_per_policy=4096,
        obs_dim=211,
        action_dim=24,
        reward_scale=1.0,
        sparse_reward=False,
        normalize_obs=True,
        normalize_reward=False,
        max_episode_steps=1000,
    )


def _shadowhandcatchunderarm_config() -> EnvironmentConfig:
    """ShadowHandCatchUnderarm task configuration."""
    return EnvironmentConfig(
        env_name="ShadowHandCatchUnderarm",
        num_envs=24576,
        num_policies=6,
        envs_per_policy=4096,
        obs_dim=211,
        action_dim=24,
        reward_scale=1.0,
        sparse_reward=False,
        normalize_obs=True,
        normalize_reward=False,
        max_episode_steps=1000,
    )


def _shadowhandcatchabreast_config() -> EnvironmentConfig:
    """ShadowHandCatchAbreast task configuration."""
    return EnvironmentConfig(
        env_name="ShadowHandCatchAbreast",
        num_envs=24576,
        num_policies=6,
        envs_per_policy=4096,
        obs_dim=211,
        action_dim=24,
        reward_scale=1.0,
        sparse_reward=False,
        normalize_obs=True,
        normalize_reward=False,
        max_episode_steps=1000,
    )


# Register environments
EnvironmentRegistry.register("shadowhandover", _shadowhandover_config)
EnvironmentRegistry.register("shadowhandcatchunderarm", _shadowhandcatchunderarm_config)
EnvironmentRegistry.register("shadowhandcatchabreast", _shadowhandcatchabreast_config)


# ---------------------------------------------------------------------------
# Shared Backbone Network Implementation
# ---------------------------------------------------------------------------

class SharedBackbone:
    """
    Shared backbone network B_theta from paper Figure 3.
    
    This is a lightweight protocol class that defines the interface.
    Actual PyTorch implementation is deferred to avoid top-level torch import.
    """
    
    def __init__(self, config: SharedBackboneConfig):
        """
        Initialize shared backbone network.
        
        Args:
            config: Backbone configuration
        """
        self.config = config
        self._network = None
    
    def build(self):
        """Build the actual network (lazy import of torch)."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for network implementation. "
                "Install with: pip install torch"
            )
        
        layers = []
        input_dim = self.config.input_dim
        
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation == "elu":
                layers.append(nn.ELU())
            
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            
            input_dim = hidden_size
        
        self._network = nn.Sequential(*layers)
        return self._network
    
    def forward(self, obs):
        """Forward pass through backbone."""
        if self._network is None:
            self.build()
        return self._network(obs)
    
    def get_output_dim(self) -> int:
        """Get output dimension of backbone."""
        if len(self.config.hidden_sizes) > 0:
            return self.config.hidden_sizes[-1]
        return self.config.input_dim


# ---------------------------------------------------------------------------
# Configuration Factory
# ---------------------------------------------------------------------------

def create_backbone_config(
    method: str,
    env_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    figure_8_layer_size: Optional[int] = None,
    **kwargs
) -> Tuple[SharedBackboneConfig, Optional[EnvironmentConfig]]:
    """
    Factory function for creating backbone and environment configurations.
    
    Args:
        method: Method name (ours, sapg, ppo, pbt, pql, ddpg)
        env_name: Environment name (optional)
        batch_size: Batch size override (optional)
        figure_8_layer_size: Layer size for Figure 8 experiments (optional)
        **kwargs: Additional config overrides
    
    Returns:
        Tuple of (backbone_config, environment_config)
    """
    # Get base backbone config
    if figure_8_layer_size is not None:
        backbone_config = Figure8ConfigRegistry.get_config(figure_8_layer_size)
    else:
        backbone_config = BackboneArchitectureRegistry.get(method)
    
    # Apply batch size override
    if batch_size is not None:
        backbone_config.batch_size = batch_size
    
    # Apply additional overrides
    for key, value in kwargs.items():
        if hasattr(backbone_config, key):
            setattr(backbone_config, key, value)
    
    # Get environment config if specified
    env_config = None
    if env_name is not None:
        env_config = EnvironmentRegistry.get(env_name)
        
        # Sync num_policies between configs
        if hasattr(backbone_config, 'num_policies'):
            env_config.num_policies = backbone_config.num_policies
            if env_config.num_envs > 0 and env_config.num_policies > 0:
                env_config.envs_per_policy = env_config.num_envs // env_config.num_policies
    
    return backbone_config, env_config


# ---------------------------------------------------------------------------
# Registry Export Functions
# ---------------------------------------------------------------------------

def export_method_registry(output_path: str = "results/method_registry.json") -> Dict[str, Any]:
    """
    Export method registry to JSON.
    
    Args:
        output_path: Output file path
    
    Returns:
        Registry data
    """
    registry_data = {
        "registry_version": "1.0",
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "methods": BackboneArchitectureRegistry.list_methods(),
        "environments": EnvironmentRegistry.list_environments(),
        "batch_size_sweep": BatchSizeSweepRegistry.DEFAULT_BATCH_SIZES,
        "figure_8_layer_sizes": [32, 64, 128, 256, 512, 1024],
        "method_configs": {
            method: BackboneArchitectureRegistry.get(method).to_dict()
            for method in BackboneArchitectureRegistry.list_methods()
        },
        "environment_configs": {
            env: EnvironmentRegistry.get(env).to_dict()
            for env in EnvironmentRegistry.list_environments()
        },
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(registry_data, f, indent=2)
    
    return registry_data


def export_experiment_configs(output_path: str = "results/experiment_configs.json") -> Dict[str, Any]:
    """
    Export all experiment configurations.
    
    Args:
        output_path: Output file path
    
    Returns:
        Experiment configuration data
    """
    configs_data = {
        "config_version": "1.0",
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "figure_8_configs": [
            config.to_dict() for config in Figure8ConfigRegistry.get_all_configs()
        ],
        "batch_size_sweep_configs": {
            method: [config.to_dict() for config in BatchSizeSweepRegistry.get_sweep_configs(method)]
            for method in ["sapg", "ppo"]
        },
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(configs_data, f, indent=2)
    
    return configs_data


# ---------------------------------------------------------------------------
# Module initialization
# ---------------------------------------------------------------------------

def initialize_registries():
    """Initialize all registries and export configurations."""
    export_method_registry()
    export_experiment_configs()


# Auto-initialize on import (but only if results dir exists)
if os.path.exists("results") or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR"):
    try:
        initialize_registries()
    except Exception:
        pass  # Fail silently during import
