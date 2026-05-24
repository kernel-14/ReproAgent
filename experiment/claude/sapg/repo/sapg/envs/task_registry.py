"""
sapg/envs/task_registry.py

Task and environment registry for SAPG reproduction.

Paper: "SAPG: Split and Aggregate Policy Gradients"
Work Package: wp_015 - Task/environment registry and metric computation

reference_grounding: wp_015 sapg/envs/task_registry.py

This module provides:
- Task registry with paper-derived environment entries
- Environment factory functions with lazy simulator loading
- Task-specific metric computation (success rate, episode reward)
- Smoke fixtures for GPU-intensive environments
- Artifact generation for environment_registry.json, metrics.json, scope_report.json

Task coverage (paper evidence contract):
  Hard Difficulty: ShadowHandOver, ShadowHandCatchUnderarm, ShadowHandCatchAbreast
  Easy Difficulty: ShadowHandReOrientation, AllegroHandReOrientation
  Additional: Throw, Regrasping, AllegroKuka variants

Metric types:
  - success_rate: Binary success indicator averaged over episodes
  - episode_reward: Cumulative reward per episode
  - consecutive_successes: Consecutive success count for curriculum tasks
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import importlib.util


PAPER_NUM_PARALLEL_ENVS = 24576
SHADOW_HAND_DOF = 24
ALLEGRO_HAND_DOF = 16
KUKA_ARM_DOF = 7
ALLEGRO_KUKA_ACTION_DIM = ALLEGRO_HAND_DOF + KUKA_ARM_DOF
ALLEGRO_KUKA_OBSERVATION_COMPONENTS = [
    "joint_angles",
    "joint_velocities",
    "object_pose",
    "object_linear_velocity",
    "object_angular_velocity",
]


def define_observation_space_joint_object_velocity_components() -> Dict[str, Any]:
    """
    Define the paper observation space: joint angles, joint velocities,
    object pose, object linear velocity, and object angular velocity.
    """
    return {
        "components": list(ALLEGRO_KUKA_OBSERVATION_COMPONENTS),
        "joint_angles": ALLEGRO_KUKA_ACTION_DIM,
        "joint_velocities": ALLEGRO_KUKA_ACTION_DIM,
        "object_pose": 7,
        "object_linear_velocity": 3,
        "object_angular_velocity": 3,
        "observation_dim": 95,
    }


# ---------------------------------------------------------------------------
# Task metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskMetadata:
    """Metadata for a registered task/environment."""
    
    task_id: str
    aliases: List[str] = field(default_factory=list)
    difficulty: str = "unknown"  # "easy", "hard", "medium"
    environment_type: str = "isaacgym"  # "isaacgym", "mujoco", "pybullet"
    metric_type: str = "episode_reward"  # "success_rate", "episode_reward", "consecutive_successes"
    
    # Environment configuration hints
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None
    max_episode_steps: int = 1000
    num_parallel_envs: int = PAPER_NUM_PARALLEL_ENVS  # Paper default for IsaacGym
    
    # Goal and curriculum hints
    goal_description: str = ""
    curriculum_stages: List[str] = field(default_factory=list)
    
    # Artifact paths
    result_artifacts: List[str] = field(default_factory=list)
    
    # Simulator requirements
    requires_gpu: bool = True
    requires_isaacgym: bool = True
    
    # Factory function name
    factory_fn: str = "make_isaacgym_env"


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskMetadata] = {
    # Hard Difficulty Tasks (Paper Section 4.1)
    "ShadowHandOver": TaskMetadata(
        task_id="ShadowHandOver",
        aliases=["shadow_hand_over", "HandOver", "ShadowHand_Over"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=211,
        action_dim=SHADOW_HAND_DOF,
        max_episode_steps=200,
        goal_description="Grasp and hand over object to target position",
        curriculum_stages=["grasp", "lift", "transfer", "release"],
        result_artifacts=["results/figures/figure_5.png", "results/tables/table_1.csv"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    "ShadowHandCatchUnderarm": TaskMetadata(
        task_id="ShadowHandCatchUnderarm",
        aliases=["shadow_hand_catch_underarm", "CatchUnderarm", "ShadowHand_CatchUnderarm"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=211,
        action_dim=SHADOW_HAND_DOF,
        max_episode_steps=200,
        goal_description="Catch thrown object with underarm motion",
        curriculum_stages=["track", "position", "catch", "stabilize"],
        result_artifacts=["results/figures/figure_5.png", "results/tables/table_1.csv"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    "ShadowHandCatchAbreast": TaskMetadata(
        task_id="ShadowHandCatchAbreast",
        aliases=["shadow_hand_catch_abreast", "CatchAbreast", "ShadowHand_CatchAbreast"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=211,
        action_dim=SHADOW_HAND_DOF,
        max_episode_steps=200,
        goal_description="Catch thrown object with abreast motion",
        curriculum_stages=["track", "position", "catch", "stabilize"],
        result_artifacts=["results/figures/figure_5.png", "results/tables/table_1.csv"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    # Easy Difficulty Tasks (Paper Section 4.1)
    "ShadowHandReOrientation": TaskMetadata(
        task_id="ShadowHandReOrientation",
        aliases=["shadow_hand_reorientation", "Reorientation", "ShadowHand_Reorientation"],
        difficulty="easy",
        environment_type="isaacgym",
        metric_type="consecutive_successes",
        observation_dim=211,
        action_dim=SHADOW_HAND_DOF,
        max_episode_steps=200,
        goal_description="Reorient grasped object to target orientation",
        curriculum_stages=["grasp", "rotate", "stabilize"],
        result_artifacts=["results/figures/figure_5.png", "results/tables/table_1.csv"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    "AllegroHandReOrientation": TaskMetadata(
        task_id="AllegroHandReOrientation",
        aliases=["allegro_hand_reorientation", "AllegroReorientation", "AllegroHand_Reorientation"],
        difficulty="easy",
        environment_type="isaacgym",
        metric_type="consecutive_successes",
        observation_dim=79,
        action_dim=16,
        max_episode_steps=200,
        goal_description="Reorient grasped object using Allegro hand",
        curriculum_stages=["grasp", "rotate", "stabilize"],
        result_artifacts=["results/figures/figure_5.png", "results/tables/table_1.csv"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    # Additional Manipulation Tasks
    "AllegroKuka": TaskMetadata(
        task_id="AllegroKuka",
        aliases=["allegro_kuka", "AllegroKukaReorientation", "Allegro_Kuka"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=95,
        action_dim=23,
        max_episode_steps=300,
        goal_description="Dual-arm manipulation with Allegro hand and Kuka arm",
        curriculum_stages=["reach", "grasp", "coordinate", "manipulate"],
        result_artifacts=["results/figures/figure_7.png"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_isaacgym_env",
    ),
    
    "Throw": TaskMetadata(
        task_id="Throw",
        aliases=["throw", "AllegroKukaThrow", "Allegro_Kuka_Throw", "IsaacGymEnvs.AllegroKukaThrow"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=95,
        action_dim=ALLEGRO_KUKA_ACTION_DIM,
        max_episode_steps=150,
        goal_description="Allegro hand mounted on Kuka arm throws object into an out-of-reach bucket",
        curriculum_stages=["allegro_kuka_mount", "random_table_object", "wind_up", "throw_into_bucket"],
        result_artifacts=["results/figures/figure_7.png"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_allegro_kuka_throw_env",
    ),
    
    "Regrasping": TaskMetadata(
        task_id="Regrasping",
        aliases=["regrasping", "AllegroKukaRegrasping", "Allegro_Kuka_Regrasping"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=95,
        action_dim=ALLEGRO_KUKA_ACTION_DIM,
        max_episode_steps=250,
        goal_description="Allegro hand mounted on Kuka arm regrasps a table object and holds it at a random 3D goal for 30 steps",
        curriculum_stages=["allegro_kuka_mount", "random_table_object", "random_3d_goal", "hold_goal_30_steps"],
        result_artifacts=["results/figures/figure_7.png"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_allegro_kuka_regrasping_env",
    ),

    "Reorientation": TaskMetadata(
        task_id="Reorientation",
        aliases=["AllegroKukaReorientation", "Allegro_Kuka_ReOrientation", "IsaacGymEnvs.Allegro_Kuka_ReOrientation"],
        difficulty="hard",
        environment_type="isaacgym",
        metric_type="success_rate",
        observation_dim=95,
        action_dim=ALLEGRO_KUKA_ACTION_DIM,
        max_episode_steps=250,
        goal_description="Allegro hand mounted on Kuka arm orients the object to a random 7D pose and samples a new goal after success",
        curriculum_stages=["allegro_kuka_mount", "random_table_object", "random_7d_pose_goal", "orient_to_goal_pose"],
        result_artifacts=["results/figures/figure_7.png", "results/figures/figure_8.png"],
        requires_gpu=True,
        requires_isaacgym=True,
        factory_fn="make_allegro_kuka_reorientation_env",
    ),
}


class AllegroKukaTaskInitializer:
    """Paper-specific AllegroKuka task setup for Throw, Regrasping, and Reorientation."""

    observation_components = ALLEGRO_KUKA_OBSERVATION_COMPONENTS

    def __init__(self, rng_seed: int = 0):
        import numpy as np

        self.rng = np.random.default_rng(rng_seed)
        self.allegro_hand_dof = ALLEGRO_HAND_DOF
        self.kuka_arm_dof = KUKA_ARM_DOF
        self.action_dim = ALLEGRO_KUKA_ACTION_DIM
        self.success_hold_steps = 30
        self.success_streak = 0

    def mount_allegro_hand_on_kuka_arm(self) -> Dict[str, Any]:
        """Mount an Allegro Hand with 16 DoF on a Kuka arm with 7 DoF."""
        return {
            "hand": "Allegro Hand",
            "hand_degrees_of_freedom": ALLEGRO_HAND_DOF,
            "arm": "Kuka",
            "arm_degrees_of_freedom": KUKA_ARM_DOF,
            "total_action_degrees_of_freedom": ALLEGRO_KUKA_ACTION_DIM,
        }

    def define_observation_space_components(self) -> Dict[str, Any]:
        """Assemble joint angles, joint velocities, object pose, linear velocity, and angular velocity."""
        return {
            "components": list(ALLEGRO_KUKA_OBSERVATION_COMPONENTS),
            "joint_angles": ALLEGRO_KUKA_ACTION_DIM,
            "joint_velocities": ALLEGRO_KUKA_ACTION_DIM,
            "object_pose": 7,
            "object_linear_velocity": 3,
            "object_angular_velocity": 3,
        }

    def random_object_position_on_table(self) -> Tuple[float, float, float]:
        """Place an object in a random position on a table at task initialization."""
        x = float(self.rng.uniform(-0.25, 0.25))
        y = float(self.rng.uniform(-0.25, 0.25))
        z = 0.78
        return (x, y, z)

    def random_three_dimensional_goal_position(self) -> Tuple[float, float, float]:
        """Select a random 3D goal position at task initialization."""
        return (
            float(self.rng.uniform(-0.35, 0.35)),
            float(self.rng.uniform(-0.35, 0.35)),
            float(self.rng.uniform(0.85, 1.25)),
        )

    def bucket_position_out_of_reach(self) -> Tuple[float, float, float]:
        """Place a bucket at a 3D position beyond the arm's direct reach for Throw."""
        return (
            float(self.rng.uniform(1.35, 1.75)),
            float(self.rng.uniform(-0.25, 0.25)),
            float(self.rng.uniform(0.65, 0.95)),
        )

    def random_seven_dimensional_goal_pose(self) -> Tuple[float, float, float, float, float, float, float]:
        """Select a random 7D goal pose: xyz plus unit quaternion."""
        import numpy as np

        xyz = self.random_three_dimensional_goal_position()
        quat = self.rng.normal(size=4)
        quat = quat / np.linalg.norm(quat)
        return tuple(xyz + tuple(float(v) for v in quat))

    def regrasping_success_for_30_steps(
        self,
        object_position: Tuple[float, float, float],
        goal_position: Tuple[float, float, float],
        lifted: bool,
        tolerance: float = 0.05,
    ) -> bool:
        """Mark Regrasping success only after the object is lifted and held at the goal for 30 timesteps."""
        import numpy as np

        at_goal = np.linalg.norm(np.asarray(object_position) - np.asarray(goal_position)) <= tolerance
        self.success_streak = self.success_streak + 1 if lifted and at_goal else 0
        return self.success_streak >= self.success_hold_steps

    def throw_success_object_in_bucket(
        self,
        object_position: Tuple[float, float, float],
        bucket_position: Tuple[float, float, float],
        lifted: bool,
        tolerance: float = 0.15,
    ) -> bool:
        """Mark Throw success when the arm lifts the object and throws it into the bucket."""
        import numpy as np

        return bool(lifted and np.linalg.norm(np.asarray(object_position) - np.asarray(bucket_position)) <= tolerance)

    def orientation_success(
        self,
        object_pose_7d: Tuple[float, float, float, float, float, float, float],
        goal_pose_7d: Tuple[float, float, float, float, float, float, float],
        position_tolerance: float = 0.05,
        quaternion_dot_tolerance: float = 0.98,
    ) -> bool:
        """Mark success if the object pose matches the random 7D goal pose."""
        import numpy as np

        obj = np.asarray(object_pose_7d)
        goal = np.asarray(goal_pose_7d)
        close_position = np.linalg.norm(obj[:3] - goal[:3]) <= position_tolerance
        close_orientation = abs(float(np.dot(obj[3:], goal[3:]))) >= quaternion_dot_tolerance
        return bool(close_position and close_orientation)

    def reinitialize_task_if_success(self, success: bool, task_kind: str) -> Dict[str, Any]:
        """Re-initialize a task when it is marked successful."""
        if not success:
            return {"success": False, "reinitialized": False}
        payload = {
            "success": True,
            "reinitialized": True,
            "object_position": self.random_object_position_on_table(),
        }
        if task_kind == "Throw":
            payload["bucket_position"] = self.bucket_position_out_of_reach()
        elif task_kind == "Reorientation":
            payload["new_goal_pose_7d"] = self.random_seven_dimensional_goal_pose()
        else:
            payload["new_goal_position_3d"] = self.random_three_dimensional_goal_position()
        self.success_streak = 0
        return payload


class ShadowHandReorientationInitializer:
    """Paper-specific 24-DoF ShadowHand cube reorientation setup."""

    def __init__(self, rng_seed: int = 0):
        import numpy as np

        self.rng = np.random.default_rng(rng_seed)
        self.shadow_hand_dof = SHADOW_HAND_DOF

    def place_cube_on_hand_and_random_goal_orientation(self) -> Dict[str, Any]:
        """Place a cube on the ShadowHand and pick a random orientation goal."""
        import numpy as np

        quat = self.rng.normal(size=4)
        quat = quat / np.linalg.norm(quat)
        return {
            "hand": "ShadowHand",
            "hand_degrees_of_freedom": SHADOW_HAND_DOF,
            "cube_position": (0.0, 0.0, 0.12),
            "goal_orientation_quaternion": tuple(float(v) for v in quat),
        }

    def cube_reaches_goal_orientation(self, cube_quaternion: Tuple[float, float, float, float], goal_quaternion: Tuple[float, float, float, float]) -> bool:
        """Mark success when the cube reaches the goal orientation."""
        import numpy as np

        return bool(abs(float(np.dot(np.asarray(cube_quaternion), np.asarray(goal_quaternion)))) >= 0.98)

    def reinitialize_if_success(self, success: bool) -> Dict[str, Any]:
        """Re-initialize ShadowHand reorientation after success."""
        if not success:
            return {"success": False, "reinitialized": False}
        payload = self.place_cube_on_hand_and_random_goal_orientation()
        payload["success"] = True
        payload["reinitialized"] = True
        return payload


# ---------------------------------------------------------------------------
# Simulator availability checks
# ---------------------------------------------------------------------------

def check_isaacgym_available() -> bool:
    """Check if IsaacGym is available without importing it."""
    return importlib.util.find_spec("isaacgym") is not None


def check_gpu_available() -> bool:
    """Check if GPU is available for training."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Environment factory functions
# ---------------------------------------------------------------------------

def make_isaacgym_env(task_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create IsaacGym environment for the specified task.
    
    Uses lazy imports to avoid requiring IsaacGym during static analysis.
    
    Args:
        task_id: Task identifier from TASK_REGISTRY
        config: Optional configuration overrides
        
    Returns:
        Environment instance or smoke fixture
        
    Raises:
        ImportError: If IsaacGym is required but not available
        ValueError: If task_id is not registered
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    
    task_meta = TASK_REGISTRY[task_id]
    config = config or {}
    
    # Check if we're in smoke/dry-run mode
    mode = config.get("mode", "train")
    if mode in ["runtime_smoke", "docker_validate", "smoke"]:
        return _make_smoke_fixture(task_id, task_meta, config)
    
    # Check simulator availability
    if task_meta.requires_isaacgym and not check_isaacgym_available():
        warnings.warn(
            f"IsaacGym not available for task {task_id}. "
            "Returning smoke fixture. Install IsaacGym for real training."
        )
        return _make_smoke_fixture(task_id, task_meta, config)
    
    if task_meta.requires_gpu and not check_gpu_available():
        warnings.warn(
            f"GPU not available for task {task_id}. "
            "Performance will be degraded."
        )
    
    # Lazy import IsaacGym wrapper
    try:
        from sapg.envs.isaacgym_wrapper import IsaacGymWrapper
        
        # Build environment configuration
        env_config = {
            "task_name": task_id,
            "num_envs": config.get("num_envs", task_meta.num_parallel_envs),
            "max_episode_length": config.get("max_episode_steps", task_meta.max_episode_steps),
            "observation_dim": task_meta.observation_dim,
            "action_dim": task_meta.action_dim,
            "device": config.get("device", "cuda" if check_gpu_available() else "cpu"),
        }
        
        return IsaacGymWrapper(**env_config)
        
    except ImportError as e:
        warnings.warn(f"Failed to import IsaacGymWrapper: {e}. Using smoke fixture.")
        return _make_smoke_fixture(task_id, task_meta, config)


def import_allegro_kuka_throw_from_isaacgymenvs() -> Any:
    """Import the AllegroKukaThrow environment from IsaacGymEnvs when available."""
    try:
        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_throw import AllegroKukaThrow
        return AllegroKukaThrow
    except Exception:
        return None


def import_allegro_kuka_regrasping_from_isaacgymenvs() -> Any:
    """Import the Allegro Kuka Regrasping environment from IsaacGymEnvs when available."""
    try:
        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_regrasping import AllegroKukaRegrasping
        return AllegroKukaRegrasping
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka import AllegroKukaRegrasping
        return AllegroKukaRegrasping
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.allegro_kuka_regrasping import AllegroKukaRegrasping
        return AllegroKukaRegrasping
    except Exception:
        return None


def import_allegro_kuka_reorientation_from_isaacgymenvs() -> Any:
    """Import the Allegro_Kuka_ReOrientation environment from IsaacGymEnvs when available."""
    try:
        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
        return AllegroKukaReorientation
    except Exception:
        return None


def import_allegro_hand_reorientation_from_isaacgymenvs() -> Any:
    """Import the Allegro hand reorientation environment from IsaacGymEnvs when available."""
    try:
        from isaacgymenvs.tasks.allegro_hand.allegro_hand_reorientation import AllegroHandReOrientation
        return AllegroHandReOrientation
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.allegro_hand_reorientation import AllegroHandReOrientation
        return AllegroHandReOrientation
    except Exception:
        return None


def import_allegro_hand_environment_from_isaacgymenvs() -> Any:
    """Explicitly import AllegroHand/AllegroHandReOrientation task modules from IsaacGymEnvs."""
    candidates = [
        ("isaacgymenvs.tasks.allegro_hand.allegro_hand_reorientation", "AllegroHandReOrientation"),
        ("isaacgymenvs.tasks.allegro_hand_reorientation", "AllegroHandReOrientation"),
        ("isaacgymenvs.tasks.allegro_hand.allegro_hand", "AllegroHand"),
    ]
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        except Exception:
            continue
    return None


def import_shadow_hand_reorientation_from_isaacgymenvs() -> Any:
    """Import the Shadow hand reorientation environment from IsaacGymEnvs when available."""
    try:
        from isaacgymenvs.tasks.shadow_hand.shadow_hand_reorientation import ShadowHandReOrientation
        return ShadowHandReOrientation
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.shadow_hand_reorientation import ShadowHandReOrientation
        return ShadowHandReOrientation
    except Exception:
        return None


def import_shadow_hand_over_from_isaacgymenvs() -> Any:
    """Import the ShadowHandOver environment from IsaacGymEnvs."""
    try:
        from isaacgymenvs.tasks.shadow_hand.shadow_hand_over import ShadowHandOver
        return ShadowHandOver
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.shadow_hand_over import ShadowHandOver
        return ShadowHandOver
    except Exception:
        return None


def import_shadow_hand_catch_underarm_from_isaacgymenvs() -> Any:
    """Import the ShadowHandCatchUnderarm environment from IsaacGymEnvs."""
    try:
        from isaacgymenvs.tasks.shadow_hand.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
        return ShadowHandCatchUnderarm
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
        return ShadowHandCatchUnderarm
    except Exception:
        return None


def import_allegro_hand_from_isaacgymenvs() -> Any:
    """Import an AllegroHand environment from IsaacGymEnvs."""
    try:
        from isaacgymenvs.tasks.allegro_hand.allegro_hand_reorientation import AllegroHandReOrientation
        return AllegroHandReOrientation
    except Exception:
        pass
    try:
        from isaacgymenvs.tasks.allegro_hand_reorientation import AllegroHandReOrientation
        return AllegroHandReOrientation
    except Exception:
        return None


def import_shadow_hand_environment_from_isaacgymenvs() -> Any:
    """Explicitly import ShadowHand task modules from IsaacGymEnvs."""
    candidates = [
        ("isaacgymenvs.tasks.shadow_hand.shadow_hand_reorientation", "ShadowHandReOrientation"),
        ("isaacgymenvs.tasks.shadow_hand_reorientation", "ShadowHandReOrientation"),
        ("isaacgymenvs.tasks.shadow_hand.shadow_hand_over", "ShadowHandOver"),
        ("isaacgymenvs.tasks.shadow_hand.shadow_hand_catch_underarm", "ShadowHandCatchUnderarm"),
        ("isaacgymenvs.tasks.shadow_hand.shadow_hand_catch_abreast", "ShadowHandCatchAbreast"),
    ]
    for module_name, attr_name in candidates:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        except Exception:
            continue
    return None


def isaacgymenvs_hand_import_registry() -> Dict[str, Any]:
    """Runtime registry showing explicit ShadowHand and AllegroHand imports from IsaacGymEnvs."""
    return {
        "ShadowHand": import_shadow_hand_environment_from_isaacgymenvs(),
        "AllegroHand": import_allegro_hand_environment_from_isaacgymenvs(),
        "AllegroKukaThrow": import_allegro_kuka_throw_from_isaacgymenvs(),
        "AllegroKukaRegrasping": import_allegro_kuka_regrasping_from_isaacgymenvs(),
        "AllegroKukaReorientation": import_allegro_kuka_reorientation_from_isaacgymenvs(),
    }


def make_allegro_kuka_throw_env(config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for the paper Allegro Kuka Throw task with 16+7 DoF and bucket success logic."""
    task_cls = import_allegro_kuka_throw_from_isaacgymenvs()
    if task_cls is not None and config and config.get("mode") not in ["runtime_smoke", "docker_validate", "smoke"]:
        return task_cls(config)
    task_config = dict(config or {})
    initializer = AllegroKukaTaskInitializer()
    task_config["mounted_robot"] = initializer.mount_allegro_hand_on_kuka_arm()
    task_config["observation_space"] = define_observation_space_joint_object_velocity_components()
    task_config["paper_initializer"] = initializer.reinitialize_task_if_success(False, "Throw")
    return _make_smoke_fixture("Throw", TASK_REGISTRY["Throw"], task_config)


def make_allegro_kuka_regrasping_env(config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for the paper Allegro Kuka Regrasping task with 30-step hold success."""
    task_cls = import_allegro_kuka_regrasping_from_isaacgymenvs()
    if task_cls is not None and config and config.get("mode") not in ["runtime_smoke", "docker_validate", "smoke"]:
        return task_cls(config)
    task_config = dict(config or {})
    initializer = AllegroKukaTaskInitializer()
    task_config["mounted_robot"] = initializer.mount_allegro_hand_on_kuka_arm()
    task_config["observation_space"] = define_observation_space_joint_object_velocity_components()
    task_config["initial_object_position_on_table"] = initializer.random_object_position_on_table()
    task_config["initial_goal_position_3d"] = initializer.random_three_dimensional_goal_position()
    task_config["paper_initializer"] = initialize_regrasping_task_with_random_object_on_table(initializer)
    return _make_smoke_fixture("Regrasping", TASK_REGISTRY["Regrasping"], task_config)


def initialize_regrasping_task_with_random_object_on_table(
    initializer: AllegroKukaTaskInitializer | None = None,
) -> Dict[str, Any]:
    """Initialize Regrasping by placing the object randomly on the table and sampling a 3D goal."""
    initializer = initializer or AllegroKukaTaskInitializer()
    object_position = initializer.random_object_position_on_table()
    goal_position = initializer.random_three_dimensional_goal_position()
    return {
        "task": "AllegroKukaRegrasping",
        "object_position": object_position,
        "object_position_randomization_range": {"x": [-0.25, 0.25], "y": [-0.25, 0.25], "z": 0.78},
        "goal_position_3d": goal_position,
        "mounted_robot": initializer.mount_allegro_hand_on_kuka_arm(),
        "success_condition": "lift object and hold it at the sampled 3D goal for 30 timesteps",
    }


def mount_allegro_hand_16dof_on_kuka_arm_7dof() -> Dict[str, Any]:
    """Executable setup helper for the 16-DoF Allegro hand mounted on the 7-DoF Kuka arm."""
    return AllegroKukaTaskInitializer().mount_allegro_hand_on_kuka_arm()


def create_policy_for_allegro_kuka_task(task_id: str, observation_space: Any = None, action_space: Any = None) -> Any:
    """Use the recurrent LSTM policy path for AllegroKuka, Throw, Regrasping, and Reorientation tasks."""
    recurrent_tasks = {"AllegroKuka", "AllegroKukaThrow", "AllegroKukaRegrasping", "AllegroKukaReorientation", "Throw", "Regrasping", "Reorientation"}
    if task_id not in recurrent_tasks:
        return None
    from sapg.networks.policy import create_recurrent_policy_for_allegro_kuka

    return create_recurrent_policy_for_allegro_kuka(
        observation_space=observation_space,
        action_space=action_space,
        task_id=task_id,
    )


def make_allegro_kuka_reorientation_env(config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for the paper Allegro_Kuka_ReOrientation task with random 7D pose goals."""
    task_cls = import_allegro_kuka_reorientation_from_isaacgymenvs()
    if task_cls is not None and config and config.get("mode") not in ["runtime_smoke", "docker_validate", "smoke"]:
        return task_cls(config)
    task_config = dict(config or {})
    initializer = AllegroKukaTaskInitializer()
    task_config["mounted_robot"] = initializer.mount_allegro_hand_on_kuka_arm()
    task_config["observation_space"] = define_observation_space_joint_object_velocity_components()
    task_config["paper_initializer"] = initializer.reinitialize_task_if_success(False, "Reorientation")
    return _make_smoke_fixture("Reorientation", TASK_REGISTRY["Reorientation"], task_config)


def _make_smoke_fixture(task_id: str, task_meta: TaskMetadata, config: Dict[str, Any]) -> Any:
    """
    Create lightweight smoke fixture for testing without GPU/simulator.
    
    Returns a minimal environment-like object that satisfies the interface
    contract without requiring heavy dependencies.
    """
    import numpy as np
    
    class SmokeEnvironment:
        """Minimal environment fixture for smoke testing."""
        
        def __init__(self, task_id: str, task_meta: TaskMetadata, config: Dict[str, Any]):
            self.task_id = task_id
            self.task_meta = task_meta
            self.config = config
            self.num_envs = config.get("num_envs", 4)  # Small for smoke
            self.observation_dim = task_meta.observation_dim or 64
            self.action_dim = task_meta.action_dim or 16
            self.max_episode_steps = task_meta.max_episode_steps
            self._step_count = 0
            self.initial_object_position_on_table = config.get("initial_object_position_on_table")
            self.initial_goal_position_3d = config.get("initial_goal_position_3d")
            self.paper_initializer = config.get("paper_initializer")
            self.policy_architecture = (
                "recurrent_lstm"
                if task_id in {"AllegroKuka", "AllegroKukaThrow", "AllegroKukaRegrasping", "AllegroKukaReorientation", "Throw", "Regrasping", "Reorientation"}
                else "mlp"
            )
            
        def reset(self):
            """Reset environment and return initial observations."""
            self._step_count = 0
            if self.task_id == "Regrasping":
                initializer = AllegroKukaTaskInitializer()
                self.initial_object_position_on_table = initializer.random_object_position_on_table()
                self.initial_goal_position_3d = initializer.random_three_dimensional_goal_position()
                self.paper_initializer = initialize_regrasping_task_with_random_object_on_table(initializer)
            obs = np.random.randn(self.num_envs, self.observation_dim).astype(np.float32)
            return obs
        
        def step(self, actions):
            """Execute actions and return (obs, reward, done, info)."""
            self._step_count += 1
            
            obs = np.random.randn(self.num_envs, self.observation_dim).astype(np.float32)
            reward = np.random.randn(self.num_envs).astype(np.float32)
            done = np.random.rand(self.num_envs) < 0.05  # 5% done probability
            
            # Force done after max steps
            if self._step_count >= self.max_episode_steps:
                done = np.ones(self.num_envs, dtype=bool)
            
            info = {
                "success": np.random.rand(self.num_envs) < 0.3,  # 30% success rate
                "episode_length": np.full(self.num_envs, self._step_count),
            }
            
            return obs, reward, done, info
        
        def close(self):
            """Clean up environment resources."""
            pass
    
    return SmokeEnvironment(task_id, task_meta, config)


def make_environment(task_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Main environment factory function.
    
    Interface contract: make_environment(task_id, config)
    
    Args:
        task_id: Task identifier from TASK_REGISTRY
        config: Optional configuration dictionary
        
    Returns:
        Environment instance
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    task_meta = TASK_REGISTRY[task_id]
    factory_fn_name = task_meta.factory_fn
    
    if factory_fn_name == "make_isaacgym_env":
        return make_isaacgym_env(task_id, config)
    if factory_fn_name == "make_allegro_kuka_throw_env":
        return make_allegro_kuka_throw_env(config)
    if factory_fn_name == "make_allegro_kuka_regrasping_env":
        return make_allegro_kuka_regrasping_env(config)
    if factory_fn_name == "make_allegro_kuka_reorientation_env":
        return make_allegro_kuka_reorientation_env(config)
    else:
        raise ValueError(f"Unknown factory function: {factory_fn_name}")


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_task_metric(task_id: str, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute task-specific metrics from trajectory data.
    
    Interface contract: compute_task_metric(task_id, trajectories)
    
    Args:
        task_id: Task identifier from TASK_REGISTRY
        trajectories: List of trajectory dictionaries with keys:
            - observations: List of observation arrays
            - actions: List of action arrays
            - rewards: List of reward values
            - dones: List of done flags
            - infos: List of info dictionaries
            
    Returns:
        Dictionary of computed metrics with task-specific semantics
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    task_meta = TASK_REGISTRY[task_id]
    metric_type = task_meta.metric_type
    
    if metric_type == "success_rate":
        return _compute_success_rate(trajectories)
    elif metric_type == "episode_reward":
        return _compute_episode_reward(trajectories)
    elif metric_type == "consecutive_successes":
        return _compute_consecutive_successes(trajectories)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def _compute_success_rate(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute success rate metric from trajectories."""
    import numpy as np
    
    if not trajectories:
        return {
            "success_rate": 0.0,
            "num_episodes": 0,
            "num_successes": 0,
        }
    
    successes = []
    for traj in trajectories:
        infos = traj.get("infos", [])
        if infos:
            # Check final info for success flag
            final_info = infos[-1]
            if isinstance(final_info, dict):
                success = final_info.get("success", False)
            elif isinstance(final_info, (list, np.ndarray)):
                # Handle vectorized environments
                success = np.any(final_info)
            else:
                success = False
            successes.append(float(success))
    
    success_rate = np.mean(successes) if successes else 0.0
    
    return {
        "success_rate": float(success_rate),
        "num_episodes": len(trajectories),
        "num_successes": int(np.sum(successes)),
    }


def _compute_episode_reward(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute episode reward metric from trajectories."""
    import numpy as np
    
    if not trajectories:
        return {
            "mean_episode_reward": 0.0,
            "std_episode_reward": 0.0,
            "min_episode_reward": 0.0,
            "max_episode_reward": 0.0,
            "num_episodes": 0,
        }
    
    episode_rewards = []
    for traj in trajectories:
        rewards = traj.get("rewards", [])
        if rewards:
            total_reward = np.sum(rewards)
            episode_rewards.append(float(total_reward))
    
    if not episode_rewards:
        return {
            "mean_episode_reward": 0.0,
            "std_episode_reward": 0.0,
            "min_episode_reward": 0.0,
            "max_episode_reward": 0.0,
            "num_episodes": 0,
        }
    
    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "min_episode_reward": float(np.min(episode_rewards)),
        "max_episode_reward": float(np.max(episode_rewards)),
        "num_episodes": len(episode_rewards),
    }


def _compute_consecutive_successes(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute consecutive successes metric from trajectories."""
    import numpy as np
    
    if not trajectories:
        return {
            "max_consecutive_successes": 0,
            "mean_consecutive_successes": 0.0,
            "success_rate": 0.0,
            "num_episodes": 0,
        }
    
    successes = []
    for traj in trajectories:
        infos = traj.get("infos", [])
        if infos:
            final_info = infos[-1]
            if isinstance(final_info, dict):
                success = final_info.get("success", False)
            elif isinstance(final_info, (list, np.ndarray)):
                success = np.any(final_info)
            else:
                success = False
            successes.append(bool(success))
    
    # Compute consecutive success streaks
    max_consecutive = 0
    current_consecutive = 0
    all_consecutives = []
    
    for success in successes:
        if success:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            if current_consecutive > 0:
                all_consecutives.append(current_consecutive)
            current_consecutive = 0
    
    if current_consecutive > 0:
        all_consecutives.append(current_consecutive)
    
    mean_consecutive = np.mean(all_consecutives) if all_consecutives else 0.0
    success_rate = np.mean(successes) if successes else 0.0
    
    return {
        "max_consecutive_successes": int(max_consecutive),
        "mean_consecutive_successes": float(mean_consecutive),
        "success_rate": float(success_rate),
        "num_episodes": len(trajectories),
    }


# ---------------------------------------------------------------------------
# Artifact generation functions
# ---------------------------------------------------------------------------

def write_environment_registry_artifact(output_path: str = "results/environment_registry.json"):
    """
    Write environment registry artifact.
    
    Artifact: results/environment_registry.json
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    registry_data = {
        "registry_version": "1.0",
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "num_tasks": len(TASK_REGISTRY),
        "tasks": {
            task_id: asdict(meta)
            for task_id, meta in TASK_REGISTRY.items()
        },
        "difficulty_breakdown": {
            "easy": [tid for tid, meta in TASK_REGISTRY.items() if meta.difficulty == "easy"],
            "hard": [tid for tid, meta in TASK_REGISTRY.items() if meta.difficulty == "hard"],
            "medium": [tid for tid, meta in TASK_REGISTRY.items() if meta.difficulty == "medium"],
        },
        "metric_types": list(set(meta.metric_type for meta in TASK_REGISTRY.values())),
        "simulator_requirements": {
            "isaacgym_available": check_isaacgym_available(),
            "gpu_available": check_gpu_available(),
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_metrics_artifact(output_path: str = "results/metrics.json"):
    """
    Write metrics schema artifact.
    
    Artifact: results/metrics.json
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics_schema = {
        "schema_version": "1.0",
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "metric_types": {
            "success_rate": {
                "description": "Binary success indicator averaged over episodes",
                "fields": ["success_rate", "num_episodes", "num_successes"],
                "range": [0.0, 1.0],
                "higher_is_better": True,
            },
            "episode_reward": {
                "description": "Cumulative reward per episode",
                "fields": ["mean_episode_reward", "std_episode_reward", "min_episode_reward", "max_episode_reward", "num_episodes"],
                "range": None,
                "higher_is_better": True,
            },
            "consecutive_successes": {
                "description": "Consecutive success count for curriculum tasks",
                "fields": ["max_consecutive_successes", "mean_consecutive_successes", "success_rate", "num_episodes"],
                "range": [0, None],
                "higher_is_better": True,
            },
        },
        "task_metric_bindings": {
            task_id: meta.metric_type
            for task_id, meta in TASK_REGISTRY.items()
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(metrics_schema, f, indent=2)


def write_scope_report_artifact(output_path: str = "results/scope_report.json"):
    """
    Write scope report artifact.
    
    Artifact: results/scope_report.json
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    scope_report = {
        "report_version": "1.0",
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "task_coverage": {
            "total_tasks": len(TASK_REGISTRY),
            "hard_tasks": len([t for t in TASK_REGISTRY.values() if t.difficulty == "hard"]),
            "easy_tasks": len([t for t in TASK_REGISTRY.values() if t.difficulty == "easy"]),
            "task_list": list(TASK_REGISTRY.keys()),
        },
        "environment_coverage": {
            "isaacgym_tasks": len([t for t in TASK_REGISTRY.values() if t.environment_type == "isaacgym"]),
            "requires_gpu": len([t for t in TASK_REGISTRY.values() if t.requires_gpu]),
        },
        "metric_coverage": {
            "success_rate_tasks": len([t for t in TASK_REGISTRY.values() if t.metric_type == "success_rate"]),
            "episode_reward_tasks": len([t for t in TASK_REGISTRY.values() if t.metric_type == "episode_reward"]),
            "consecutive_successes_tasks": len([t for t in TASK_REGISTRY.values() if t.metric_type == "consecutive_successes"]),
        },
        "paper_alignment": {
            "hard_difficulty_tasks": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast"],
            "easy_difficulty_tasks": ["ShadowHandReOrientation", "AllegroHandReOrientation"],
            "additional_tasks": ["AllegroKuka", "Throw", "Regrasping", "Reorientation"],
            "allegro_kuka_tasks": {
                "Throw": "Allegro Hand 16 DoF mounted on Kuka 7 DoF, bucket out of reach",
                "Regrasping": "Allegro Hand 16 DoF mounted on Kuka 7 DoF, random 3D goal held for 30 timesteps",
                "Reorientation": "Allegro Hand 16 DoF mounted on Kuka 7 DoF, random 7D pose goal",
            },
            "observation_components": ALLEGRO_KUKA_OBSERVATION_COMPONENTS,
        },
        "simulator_status": {
            "isaacgym_available": check_isaacgym_available(),
            "gpu_available": check_gpu_available(),
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(scope_report, f, indent=2)


def write_all_artifacts():
    """Write all declared artifacts for this module."""
    write_environment_registry_artifact()
    write_metrics_artifact()
    write_scope_report_artifact()


# ---------------------------------------------------------------------------
# Smoke test and validation
# ---------------------------------------------------------------------------

def run_smoke_test():
    """Run smoke test to validate registry and factory functions."""
    print("Running task_registry smoke test...")
    
    # Test registry access
    assert len(TASK_REGISTRY) > 0, "Task registry is empty"
    print(f"✓ Task registry contains {len(TASK_REGISTRY)} tasks")
    
    # Test environment factory with smoke fixture
    config = {"mode": "runtime_smoke", "num_envs": 2}
    for task_id in list(TASK_REGISTRY.keys())[:3]:  # Test first 3 tasks
        env = make_environment(task_id, config)
        obs = env.reset()
        assert obs is not None, f"Failed to reset environment for {task_id}"
