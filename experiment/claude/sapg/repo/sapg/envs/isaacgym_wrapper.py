# sapg/envs/isaacgym_wrapper.py
# IsaacGym Environment Wrapper for SAPG Reproduction
# reference_grounding: wp_015 sapg/envs/isaacgym_wrapper.py
#
# Paper evidence contract: Complete environment/task registry for manipulation tasks
# including Shadow Hand (Over, CatchUnderarm, CatchAbreast, Reorientation) and
# Allegro Hand (Reorientation, AllegroKuka, harder_AllegroKuka) plus Throw, Regrasping.
#
# This module provides:
# - Lazy IsaacGym availability checks and imports
# - Task registry with paper-derived task metadata
# - Environment factory: make_environment(task_id, config)
# - Metric computation: compute_task_metric(task_id, trajectories)
# - Smoke fixtures for GPU-free validation
# - Artifact writers: environment_registry.json, metrics.json, scope_report.json

import os
import json
import importlib.util
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


ALLEGRO_HAND_DOF = 16
KUKA_ARM_DOF = 7
ALLEGRO_KUKA_ACTION_DIM = ALLEGRO_HAND_DOF + KUKA_ARM_DOF


# ---------------------------------------------------------------------------
# IsaacGym Availability Check (lazy, no top-level import)
# ---------------------------------------------------------------------------

def check_isaacgym_available() -> Tuple[bool, Optional[str]]:
    """
    Check if IsaacGym is available without importing it.
    
    Returns:
        (available, error_message): True if available, False with reason otherwise
    """
    # Check for isaacgym package
    if importlib.util.find_spec("isaacgym") is None:
        return False, "isaacgym package not found"
    
    # Check for isaacgymenvs package
    if importlib.util.find_spec("isaacgymenvs") is None:
        return False, "isaacgymenvs package not found"
    
    # Check for torch (required by IsaacGym)
    if importlib.util.find_spec("torch") is None:
        return False, "torch package not found (required by IsaacGym)"
    
    return True, None


def get_isaacgym_modules():
    """
    Lazy import of IsaacGym modules.
    
    Returns:
        dict with 'gym', 'gymapi', 'gymutil', 'gymtorch', 'isaacgymenvs'
    
    Raises:
        ImportError: If IsaacGym is not available
    """
    available, error = check_isaacgym_available()
    if not available:
        raise ImportError(f"IsaacGym not available: {error}")
    
    import isaacgym
    from isaacgym import gymapi, gymutil, gymtorch
    import isaacgymenvs
    
    return {
        'isaacgym': isaacgym,
        'gymapi': gymapi,
        'gymutil': gymutil,
        'gymtorch': gymtorch,
        'isaacgymenvs': isaacgymenvs,
    }


# ---------------------------------------------------------------------------
# Paper-Derived Task Registry
# ---------------------------------------------------------------------------

# Task registry with paper-derived metadata
# Paper evidence: Shadow Hand (Over, CatchUnderarm, CatchAbreast, Reorientation)
# Allegro Hand (Reorientation, AllegroKuka, harder_AllegroKuka)
# Additional tasks: Throw, Regrasping
TASK_REGISTRY = {
    # Shadow Hand tasks (hard difficulty)
    "ShadowHandOver": {
        "task_id": "ShadowHandOver",
        "aliases": ["Shadow Hand", "ShadowHand", "HandOver"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "ShadowHand",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 24,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Shadow Hand object manipulation - pass object between hands",
        "paper_section": "Table 1, Figure 5",
    },
    "ShadowHandCatchUnderarm": {
        "task_id": "ShadowHandCatchUnderarm",
        "aliases": ["CatchUnderarm", "Underarm"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "ShadowHandCatchUnderarm",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 24,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Shadow Hand catch object thrown underarm",
        "paper_section": "Table 1",
    },
    "ShadowHandCatchAbreast": {
        "task_id": "ShadowHandCatchAbreast",
        "aliases": ["CatchAbreast", "Abreast"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "ShadowHandCatchAbreast",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 24,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Shadow Hand catch object thrown abreast",
        "paper_section": "Table 1",
    },
    "ShadowHandReOrientation": {
        "task_id": "ShadowHandReOrientation",
        "aliases": ["Reorientation", "HandReorient", "ShadowReorient"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "ShadowHandReOrientation",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 24,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Shadow Hand object reorientation",
        "paper_section": "Table 1",
    },
    
    # Allegro Hand tasks (hard difficulty)
    "AllegroHandReOrientation": {
        "task_id": "AllegroHandReOrientation",
        "aliases": ["Allegro Kuka Reorientation", "AllegroReorient", "Allegro"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroHand",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 42,
        "action_dim": 16,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Allegro Hand object reorientation",
        "paper_section": "Table 1, Figure 5",
    },
    "AllegroKuka": {
        "task_id": "AllegroKuka",
        "aliases": ["AllegroKuka", "Allegro Kuka"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroKuka",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 95,
        "action_dim": ALLEGRO_KUKA_ACTION_DIM,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "16-DoF Allegro Hand mounted on a 7-DoF Kuka arm for manipulation",
        "paper_section": "Table 1, Figure 5",
    },
    "harder_AllegroKuka": {
        "task_id": "harder_AllegroKuka",
        "aliases": ["harder AllegroKuka", "HarderAllegroKuka"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroKuka",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 95,
        "action_dim": ALLEGRO_KUKA_ACTION_DIM,
        "success_threshold": 0.9,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "16-DoF Allegro Hand mounted on a 7-DoF Kuka arm for harder manipulation",
        "paper_section": "Table 1",
        "config_override": {"task_difficulty": "hard"},
    },
    
    # Additional manipulation tasks
    "Throw": {
        "task_id": "Throw",
        "aliases": ["Throw", "ThrowTask", "AllegroKukaThrow"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroKukaThrow",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 95,
        "action_dim": ALLEGRO_KUKA_ACTION_DIM,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "AllegroKuka Throw: 16-DoF Allegro hand mounted on 7-DoF Kuka arm throws an object",
        "paper_section": "Table 1",
    },
    "Regrasping": {
        "task_id": "Regrasping",
        "aliases": ["Regrasping", "Regrasp", "AllegroKukaRegrasping"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroKukaRegrasping",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 95,
        "action_dim": ALLEGRO_KUKA_ACTION_DIM,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "AllegroKuka Regrasping: 16-DoF Allegro hand mounted on 7-DoF Kuka arm regrasps a table object",
        "paper_section": "Table 1",
    },
    
    # Easy difficulty variants (for ablation studies)
    "ShadowHandReOrientation_easy": {
        "task_id": "ShadowHandReOrientation_easy",
        "aliases": ["Easy Reorientation", "EasyReorient"],
        "difficulty": "easy",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "ShadowHandReOrientation",
        "max_episode_length": 800,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 24,
        "success_threshold": 0.6,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Shadow Hand object reorientation (easy variant)",
        "paper_section": "Ablation studies",
        "config_override": {"task_difficulty": "easy"},
    },
    "AllegroHandReOrientation_easy": {
        "task_id": "AllegroHandReOrientation_easy",
        "aliases": ["Easy Allegro", "EasyAllegro"],
        "difficulty": "easy",
        "environment_type": "isaacgym",
        "isaacgym_task_name": "AllegroHand",
        "max_episode_length": 800,
        "num_envs": 24576,
        "observation_dim": 42,
        "action_dim": 16,
        "success_threshold": 0.6,
        "metric_type": "success_rate",
        "metric_key": "consecutive_successes",
        "description": "Allegro Hand object reorientation (easy variant)",
        "paper_section": "Ablation studies",
        "config_override": {"task_difficulty": "easy"},
    },
}


def get_task_metadata(task_id: str) -> Dict[str, Any]:
    """
    Get task metadata from registry.
    
    Args:
        task_id: Task identifier or alias
    
    Returns:
        Task metadata dictionary
    
    Raises:
        ValueError: If task_id not found in registry
    """
    # Direct lookup
    if task_id in TASK_REGISTRY:
        return TASK_REGISTRY[task_id].copy()
    
    # Alias lookup
    for task_name, metadata in TASK_REGISTRY.items():
        if task_id in metadata.get("aliases", []):
            return metadata.copy()
    
    raise ValueError(f"Task '{task_id}' not found in registry. Available tasks: {list(TASK_REGISTRY.keys())}")


def list_available_tasks(difficulty: Optional[str] = None) -> List[str]:
    """
    List available tasks, optionally filtered by difficulty.
    
    Args:
        difficulty: Optional difficulty filter ('easy', 'hard')
    
    Returns:
        List of task IDs
    """
    tasks = []
    for task_id, metadata in TASK_REGISTRY.items():
        if difficulty is None or metadata.get("difficulty") == difficulty:
            tasks.append(task_id)
    return tasks


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------

class IsaacGymEnvironmentWrapper:
    """
    Wrapper for IsaacGym environments with unified interface.
    
    Provides:
    - Lazy loading of IsaacGym
    - Unified observation/action interface
    - Metric tracking (success rate, episode reward)
    - Smoke fixture mode for GPU-free validation
    """
    
    def __init__(self, task_id: str, config: Dict[str, Any], smoke_mode: bool = False):
        """
        Initialize environment wrapper.
        
        Args:
            task_id: Task identifier from registry
            config: Environment configuration
            smoke_mode: If True, use lightweight smoke fixture instead of real env
        """
        self.task_id = task_id
        self.config = config
        self.smoke_mode = smoke_mode
        self.metadata = get_task_metadata(task_id)
        
        # Environment state
        self.env = None
        self.num_envs = config.get("num_envs", self.metadata["num_envs"])
        self.observation_dim = self.metadata["observation_dim"]
        self.action_dim = self.metadata["action_dim"]
        
        # Metric tracking
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_rewards = None
        self.current_episode_lengths = None
        
        if smoke_mode:
            self._init_smoke_fixture()
        else:
            self._init_isaacgym_env()
    
    def _init_smoke_fixture(self):
        """Initialize lightweight smoke fixture for validation."""
        import numpy as np
        
        self.env = None  # No real environment
        self.current_episode_rewards = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._smoke_step_count = 0
        self._smoke_max_steps = 10  # Bounded for smoke validation
    
    def _init_isaacgym_env(self):
        """Initialize real IsaacGym environment."""
        # Lazy import
        modules = get_isaacgym_modules()
        isaacgymenvs = modules['isaacgymenvs']
        
        # Build IsaacGym config
        isaacgym_config = self._build_isaacgym_config()
        
        # Create environment
        from isaacgymenvs.tasks import isaacgym_task_map
        task_name = self.metadata["isaacgym_task_name"]
        
        if task_name not in isaacgym_task_map:
            raise ValueError(f"IsaacGym task '{task_name}' not found in task map")
        
        # Initialize environment
        self.env = isaacgym_task_map[task_name](
            cfg=isaacgym_config,
            sim_device=self.config.get("sim_device", "cuda:0"),
            graphics_device_id=self.config.get("graphics_device_id", 0),
            headless=self.config.get("headless", True),
        )
        
        # Initialize tracking
        import numpy as np
        self.current_episode_rewards = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    
    def _build_isaacgym_config(self) -> Dict[str, Any]:
        """Build IsaacGym-compatible configuration."""
        config = {
            "name": self.metadata["isaacgym_task_name"],
            "physics_engine": self.config.get("physics_engine", "physx"),
            "env": {
                "numEnvs": self.num_envs,
                "envSpacing": self.config.get("env_spacing", 0.75),
                "episodeLength": self.metadata["max_episode_length"],
                "enableDebugVis": False,
            },
            "sim": {
                "dt": 1.0 / 60.0,
                "substeps": 2,
                "up_axis": "z",
                "use_gpu_pipeline": self.config.get("use_gpu_pipeline", True),
                "gravity": [0.0, 0.0, -9.81],
            },
        }
        
        # Apply task-specific config overrides
        if "config_override" in self.metadata:
            config.update(self.metadata["config_override"])
        
        return config
    
    def reset(self):
        """Reset environment."""
        if self.smoke_mode:
            import numpy as np
            self._smoke_step_count = 0
            obs = np.random.randn(self.num_envs, self.observation_dim).astype(np.float32)
            return obs
        else:
            obs = self.env.reset()
            self.current_episode_rewards.fill(0)
            self.current_episode_lengths.fill(0)
            return obs
    
    def step(self, actions):
        """
        Step environment.
        
        Args:
            actions: Action array [num_envs, action_dim]
        
        Returns:
            (observations, rewards, dones, info)
        """
        if self.smoke_mode:
            import numpy as np
            
            self._smoke_step_count += 1
            obs = np.random.randn(self.num_envs, self.observation_dim).astype(np.float32)
            rewards = np.random.randn(self.num_envs).astype(np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            
            # Simulate some episodes finishing
            if self._smoke_step_count >= self._smoke_max_steps:
                dones[:] = True
            
            info = {
                "consecutive_successes": np.random.randint(0, 10, size=self.num_envs),
                "episode_reward": rewards,
            }
            
            return obs, rewards, dones, info
        else:
            obs, rewards, dones, info = self.env.step(actions)
            
            # Track metrics
            self.current_episode_rewards += rewards
            self.current_episode_lengths += 1
            
            # Record completed episodes
            for i in range(self.num_envs):
                if dones[i]:
                    self.episode_rewards.append(self.current_episode_rewards[i])
                    
                    # Extract success metric
                    if "consecutive_successes" in info:
                        success = info["consecutive_successes"][i] > 0
                        self.episode_successes.append(success)
                    
                    # Reset tracking
                    self.current_episode_rewards[i] = 0
                    self.current_episode_lengths[i] = 0
            
            return obs, rewards, dones, info
    
    def close(self):
        """Close environment."""
        if self.env is not None and hasattr(self.env, 'close'):
            self.env.close()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metric_type as key and computed value
        """
        import numpy as np
        
        metrics = {}
        
        if self.metadata["metric_type"] == "success_rate":
            if len(self.episode_successes) > 0:
                metrics["success_rate"] = np.mean(self.episode_successes)
            else:
                metrics["success_rate"] = 0.0
        
        if len(self.episode_rewards) > 0:
            metrics["episode_reward_mean"] = np.mean(self.episode_rewards)
            metrics["episode_reward_std"] = np.std(self.episode_rewards)
        else:
            metrics["episode_reward_mean"] = 0.0
            metrics["episode_reward_std"] = 0.0
        
        metrics["num_episodes"] = len(self.episode_rewards)
        
        return metrics


def make_environment(task_id: str, config: Dict[str, Any]) -> IsaacGymEnvironmentWrapper:
    """
    Factory function to create environment.
    
    Args:
        task_id: Task identifier from registry
        config: Environment configuration
    
    Returns:
        IsaacGymEnvironmentWrapper instance
    """
    # Check if smoke mode
    smoke_mode = config.get("smoke_mode", False)
    
    # Check IsaacGym availability if not smoke mode
    if not smoke_mode:
        available, error = check_isaacgym_available()
        if not available:
            raise ImportError(
                f"Cannot create real IsaacGym environment: {error}. "
                f"Use smoke_mode=True for validation without GPU."
            )
    
    return IsaacGymEnvironmentWrapper(task_id, config, smoke_mode=smoke_mode)


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_task_metric(task_id: str, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute task-specific metrics from trajectories.
    
    Args:
        task_id: Task identifier
        trajectories: List of trajectory dictionaries with keys:
            - observations: [T, obs_dim]
            - actions: [T, action_dim]
            - rewards: [T]
            - dones: [T]
            - info: [T] (list of info dicts)
    
    Returns:
        Dictionary with computed metrics
    """
    import numpy as np
    
    metadata = get_task_metadata(task_id)
    metric_type = metadata["metric_type"]
    metric_key = metadata.get("metric_key", "consecutive_successes")
    
    metrics = {}
    
    # Compute episode rewards
    episode_rewards = []
    for traj in trajectories:
        episode_reward = np.sum(traj["rewards"])
        episode_rewards.append(episode_reward)
    
    if len(episode_rewards) > 0:
        metrics["episode_reward_mean"] = float(np.mean(episode_rewards))
        metrics["episode_reward_std"] = float(np.std(episode_rewards))
        metrics["episode_reward_min"] = float(np.min(episode_rewards))
        metrics["episode_reward_max"] = float(np.max(episode_rewards))
    else:
        metrics["episode_reward_mean"] = 0.0
        metrics["episode_reward_std"] = 0.0
        metrics["episode_reward_min"] = 0.0
        metrics["episode_reward_max"] = 0.0
    
    # Compute success rate if applicable
    if metric_type == "success_rate":
        successes = []
        for traj in trajectories:
            # Check if trajectory has success info
            if "info" in traj and len(traj["info"]) > 0:
                # Look for success in final step info
                final_info = traj["info"][-1]
                if metric_key in final_info:
                    success = final_info[metric_key] > 0
                    successes.append(success)
        
        if len(successes) > 0:
            metrics["success_rate"] = float(np.mean(successes))
            metrics["success_count"] = int(np.sum(successes))
        else:
            metrics["success_rate"] = 0.0
            metrics["success_count"] = 0
    
    metrics["num_trajectories"] = len(trajectories)
    metrics["task_id"] = task_id
    metrics["metric_type"] = metric_type
    
    return metrics


# ---------------------------------------------------------------------------
# Artifact Writers
# ---------------------------------------------------------------------------

def write_environment_registry(output_path: str = "results/environment_registry.json"):
    """
    Write environment registry to JSON artifact.
    
    Args:
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    registry_artifact = {
        "task_registry": TASK_REGISTRY,
        "num_tasks": len(TASK_REGISTRY),
        "difficulty_breakdown": {
            "easy": len([t for t in TASK_REGISTRY.values() if t["difficulty"] == "easy"]),
            "hard": len([t for t in TASK_REGISTRY.values() if t["difficulty"] == "hard"]),
        },
        "environment_types": list(set(t["environment_type"] for t in TASK_REGISTRY.values())),
        "paper_coverage": {
            "shadow_hand_tasks": [t for t in TASK_REGISTRY.keys() if "Shadow" in t],
            "allegro_hand_tasks": [t for t in TASK_REGISTRY.keys() if "Allegro" in t],
            "manipulation_tasks": [t for t in TASK_REGISTRY.keys() if t in ["Throw", "Regrasping"]],
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(registry_artifact, f, indent=2)


def write_metrics_artifact(metrics: Dict[str, Any], output_path: str = "results/metrics.json"):
    """
    Write metrics to JSON artifact.
    
    Args:
        metrics: Metrics dictionary
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def write_scope_report(output_path: str = "results/scope_report.json"):
    """
    Write scope report artifact.
    
    Args:
        output_path: Output file path
    """
    available, error = check_isaacgym_available()
    
    scope_report = {
        "isaacgym_available": available,
        "isaacgym_error": error,
        "num_registered_tasks": len(TASK_REGISTRY),
        "registered_tasks": list(TASK_REGISTRY.keys()),
        "hard_difficulty_tasks": list_available_tasks(difficulty="hard"),
        "easy_difficulty_tasks": list_available_tasks(difficulty="easy"),
        "paper_evidence_coverage": {
            "shadow_hand": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast", "ShadowHandReOrientation"],
            "allegro_hand": ["AllegroHandReOrientation", "AllegroKuka", "harder_AllegroKuka"],
            "additional_tasks": ["Throw", "Regrasping"],
        },
        "smoke_mode_available": True,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scope_report, f, indent=2)


# ---------------------------------------------------------------------------
# Smoke Validation
# ---------------------------------------------------------------------------

def run_smoke_validation():
    """
    Run smoke validation to verify wrapper functionality.
    
    Creates all declared artifacts with dry-run/schema content.
    """
    print("Running IsaacGym wrapper smoke validation...")
    
    # Write environment registry
    write_environment_registry()
    print("✓ Written results/environment_registry.json")
    
    # Write scope report
    write_scope_report()
    print("✓ Written results/scope_report.json")
    
    # Test smoke environment creation
    config = {
        "num_envs": 16,
        "smoke_mode": True,
    }
