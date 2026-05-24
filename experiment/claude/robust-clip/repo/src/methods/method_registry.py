"""
Method registry for Robust CLIP reproduction.

This module provides:
- Comprehensive method/baseline/attack registry
- Method selectors and configuration builders
- Parameter sweep definitions
- Evidence contract matrix generation
- Experiment protocol registration

Paper evidence contract methods:
- Vision encoders: clip, robust_clip, vit, fare, tecoa
- LVLMs: llava, openflamingo
- Attacks: pgd, apgd, autoattack, random
- Training: fine_tuning, adapter, baseline
- Evaluation: pope, cot, cli

Implementation surfaces: baseline_or_ablation, evaluation, artifact_writer, config, tests
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Method Configuration Dataclasses
# ============================================================================

@dataclass
class MethodConfig:
    """Configuration for a single method."""
    name: str
    category: str  # vision_encoder, lvlm, attack, training, evaluation
    description: str
    enabled: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    baseline: bool = False
    paper_reference: Optional[str] = None
    implementation_notes: Optional[str] = None


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    name: str
    epsilon: float
    steps: int
    step_size: float
    norm: str  # l2, linf
    targeted: bool = False
    random_start: bool = True
    precision: str = "float32"  # float16, float32 (from addendum)
    momentum: float = 0.0
    
    def get_dtype_str(self) -> str:
        """Get dtype string for attack precision (from addendum clarification)."""
        if self.precision == "float16":
            return "16-bit"
        elif self.precision == "float32":
            return "32-bit"
        return "32-bit"


@dataclass
class SweepConfig:
    """Parameter sweep configuration."""
    parameter: str
    values: List[Any]
    method: str
    description: str
    

# ============================================================================
# Method Registry (Paper Evidence Contract)
# ============================================================================

class MethodRegistry:
    """
    Central registry for all methods, baselines, and attacks.
    
    Paper evidence contract obligations:
    - Vision encoders: clip, robust_clip, vit, fare, tecoa, ours
    - LVLMs: llava, openflamingo
    - Attacks: pgd, apgd, autoattack, random, baseline
    - Training: fine_tuning, adapter
    - Evaluation: pope, cot, cli, fare_loss
    """
    
    def __init__(self):
        """Initialize method registry with paper-defined methods."""
        self._methods: Dict[str, MethodConfig] = {}
        self._attacks: Dict[str, AttackConfig] = {}
        self._sweeps: Dict[str, SweepConfig] = {}
        self._register_all_methods()
        
    def _register_all_methods(self):
        """Register all methods from paper evidence contract."""
        self._register_vision_encoders()
        self._register_lvlm_methods()
        self._register_attack_methods()
        self._register_training_methods()
        self._register_evaluation_methods()
        self._register_parameter_sweeps()
        
    def _register_vision_encoders(self):
        """Register vision encoder methods."""
        # CLIP (baseline)
        self.register_method(MethodConfig(
            name="clip",
            category="vision_encoder",
            description="Standard OpenAI CLIP ViT-L/14 baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "model_name": "ViT-L/14",
                "pretrained": "openai",
                "image_size": 224,
                "embed_dim": 768,
            },
            paper_reference="Radford et al. 2021",
            implementation_notes="OpenCLIP implementation"
        ))
        
        # Robust CLIP (alias for FARE)
        self.register_method(MethodConfig(
            name="robust_clip",
            category="vision_encoder",
            description="Robust CLIP (FARE-finetuned)",
            enabled=True,
            baseline=False,
            hyperparameters={
                "base_model": "clip",
                "finetuning_method": "fare",
                "lambda_preserve": 1.0,
            },
            paper_reference="This paper",
            implementation_notes="Alias for FARE method"
        ))
        
        # ViT (baseline)
        self.register_method(MethodConfig(
            name="vit",
            category="vision_encoder",
            description="Standard Vision Transformer baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "model_name": "vit_large_patch14_224",
                "pretrained": True,
                "num_classes": 1000,
            },
            paper_reference="Dosovitskiy et al. 2021"
        ))
        
        # FARE (ours)
        self.register_method(MethodConfig(
            name="fare",
            category="vision_encoder",
            description="Feature-Alignment Robust Embedding (ours)",
            enabled=True,
            baseline=False,
            hyperparameters={
                "loss_type": "fare",
                "alignment_target": "class_token",
                "distance_metric": "l2",
                "lambda_preserve": 1.0,
                "attack_steps": 10,
                "step_size": 0.01,
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
            },
            paper_reference="This paper (Algorithm 1)",
            implementation_notes="Unsupervised adversarial fine-tuning"
        ))
        
        # FARE-CLIP (alias)
        self.register_method(MethodConfig(
            name="fare_clip",
            category="vision_encoder",
            description="FARE-finetuned CLIP",
            enabled=True,
            baseline=False,
            hyperparameters={
                "base_model": "clip",
                "finetuning_method": "fare",
            },
            paper_reference="This paper"
        ))
        
        # TeCoA (baseline)
        self.register_method(MethodConfig(
            name="tecoa",
            category="vision_encoder",
            description="Text-guided Contrastive Adversarial baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "loss_type": "tecoa",
                "alignment_target": "text_guided",
                "attack_steps": 10,
                "step_size": 0.01,
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
                "supervised": True,
            },
            paper_reference="Mao et al. 2023",
            implementation_notes="Supervised adversarial fine-tuning baseline"
        ))
        
        # Ours (alias for FARE)
        self.register_method(MethodConfig(
            name="ours",
            category="vision_encoder",
            description="Our method (FARE)",
            enabled=True,
            baseline=False,
            hyperparameters=self._methods.get("fare", MethodConfig(
                name="fare", category="vision_encoder", description="FARE"
            )).hyperparameters,
            paper_reference="This paper",
            implementation_notes="Primary contribution"
        ))
        
    def _register_lvlm_methods(self):
        """Register Large Vision-Language Model methods."""
        # LLaVA
        self.register_method(MethodConfig(
            name="llava",
            category="lvlm",
            description="LLaVA-1.5 7B with CLIP ViT-L/14@224",
            enabled=True,
            baseline=False,
            hyperparameters={
                "model_name": "llava-v1.5-7b",
                "vision_encoder": "clip",
                "vision_resolution": 224,  # Not 336 (from addendum)
                "llm_backbone": "vicuna-7b-v1.5",
                "use_openclip": True,  # Modified to use OpenCLIP (from addendum)
            },
            paper_reference="https://github.com/haotian-liu/LLaVA/tree/main",
            implementation_notes="Modified to use OpenCLIP instead of HuggingFace"
        ))
        
        # OpenFlamingo
        self.register_method(MethodConfig(
            name="openflamingo",
            category="lvlm",
            description="OpenFlamingo 9B",
            enabled=True,
            baseline=False,
            hyperparameters={
                "model_name": "openflamingo-9b",
                "vision_encoder": "clip",
                "llm_backbone": "mpt-7b",
            },
            paper_reference="https://github.com/mlfoundations/open_flamingo/tree/main",
            implementation_notes="Few-shot LVLM baseline"
        ))
        
    def _register_attack_methods(self):
        """Register adversarial attack methods."""
        # PGD (Projected Gradient Descent)
        self.register_method(MethodConfig(
            name="pgd",
            category="attack",
            description="Projected Gradient Descent attack",
            enabled=True,
            baseline=True,
            hyperparameters={
                "steps": 10,
                "step_size": 0.01,
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
                "random_start": True,
                "targeted": False,
            },
            paper_reference="Madry et al. 2018"
        ))
        
        self.register_attack(AttackConfig(
            name="pgd",
            epsilon=4.0 / 255.0,
            steps=10,
            step_size=0.01,
            norm="linf",
            targeted=False,
            random_start=True,
            precision="float32",
            momentum=0.0,
        ))
        
        # APGD (Auto-PGD)
        self.register_method(MethodConfig(
            name="apgd",
            category="attack",
            description="Auto-PGD attack",
            enabled=True,
            baseline=False,
            hyperparameters={
                "steps": 100,
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
                "n_restarts": 1,
                "loss": "ce",
            },
            paper_reference="https://github.com/fra31/robust-finetuning",
            implementation_notes="APGD from robust-finetuning repository"
        ))
        
        self.register_attack(AttackConfig(
            name="apgd",
            epsilon=4.0 / 255.0,
            steps=100,
            step_size=0.01,
            norm="linf",
            targeted=False,
            random_start=False,
            precision="float32",
            momentum=0.0,
        ))
        
        # AutoAttack
        self.register_method(MethodConfig(
            name="autoattack",
            category="attack",
            description="AutoAttack ensemble",
            enabled=True,
            baseline=False,
            hyperparameters={
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
                "version": "standard",
            },
            paper_reference="Croce & Hein 2020"
        ))
        
        self.register_attack(AttackConfig(
            name="autoattack",
            epsilon=4.0 / 255.0,
            steps=100,
            step_size=0.01,
            norm="linf",
            targeted=False,
            random_start=False,
            precision="float32",
            momentum=0.0,
        ))
        
        # Random (baseline)
        self.register_method(MethodConfig(
            name="random",
            category="attack",
            description="Random noise baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "epsilon": 4.0 / 255.0,
                "norm": "linf",
            },
            paper_reference="Baseline"
        ))
        
        self.register_attack(AttackConfig(
            name="random",
            epsilon=4.0 / 255.0,
            steps=1,
            step_size=0.0,
            norm="linf",
            targeted=False,
            random_start=True,
            precision="float32",
            momentum=0.0,
        ))
        
        # Jailbreak attack (PGD-based)
        self.register_method(MethodConfig(
            name="jailbreak",
            category="attack",
            description="Visual adversarial jailbreak attack",
            enabled=True,
            baseline=False,
            hyperparameters={
                "steps": 5000,
                "step_size": 1.0 / 255.0,  # alpha = 1/255 (from addendum)
                "epsilon": 16.0 / 255.0,
                "norm": "linf",
                "targeted": True,
                "momentum": 0.0,  # No momentum (from addendum)
            },
            paper_reference="https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models",
            implementation_notes="Adapted from Qi et al. 2023 for LLaVA-1.5 7B"
        ))
        
        self.register_attack(AttackConfig(
            name="jailbreak",
            epsilon=16.0 / 255.0,
            steps=5000,
            step_size=1.0 / 255.0,
            norm="linf",
            targeted=True,
            random_start=False,
            precision="float32",
            momentum=0.0,
        ))
        
    def _register_training_methods(self):
        """Register training method variants."""
        # Fine-tuning
        self.register_method(MethodConfig(
            name="fine_tuning",
            category="training",
            description="Standard supervised fine-tuning",
            enabled=True,
            baseline=True,
            hyperparameters={
                "learning_rate": 1e-5,
                "batch_size": 32,
                "epochs": 10,
                "optimizer": "adamw",
            },
            paper_reference="Standard"
        ))
        
        # Adapter
        self.register_method(MethodConfig(
            name="adapter",
            category="training",
            description="Adapter-based fine-tuning",
            enabled=True,
            baseline=False,
            hyperparameters={
                "adapter_dim": 64,
                "learning_rate": 1e-4,
                "adapter_position": "post_attention",
            },
            paper_reference="Houlsby et al. 2019"
        ))
        
        # Baseline (standard training)
        self.register_method(MethodConfig(
            name="baseline",
            category="training",
            description="Standard training baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "learning_rate": 1e-4,
                "batch_size": 256,
                "epochs": 30,
            },
            paper_reference="Standard"
        ))
        
    def _register_evaluation_methods(self):
        """Register evaluation protocol methods."""
        # POPE (hallucination benchmark)
        self.register_method(MethodConfig(
            name="pope",
            category="evaluation",
            description="Polling-based Object Probing Evaluation",
            enabled=True,
            baseline=False,
            hyperparameters={
                "dataset": "coco",
                "split": "val",
                "categories": ["random", "popular", "adversarial"],
            },
            paper_reference="Li et al. 2023 (POPE)",
            implementation_notes="From LLaVA repository"
        ))
        
        # CoT (Chain of Thought)
        self.register_method(MethodConfig(
            name="cot",
            category="evaluation",
            description="Chain of Thought reasoning evaluation",
            enabled=True,
            baseline=False,
            hyperparameters={
                "benchmark": "sqa_i",
                "prompt_style": "cot",
            },
            paper_reference="SQA-I benchmark"
        ))
        
        # CLI (Command Line Interface / Clean-Image baseline)
        self.register_method(MethodConfig(
            name="cli",
            category="evaluation",
            description="Clean-image evaluation baseline",
            enabled=True,
            baseline=True,
            hyperparameters={
                "attack": None,
                "epsilon": 0.0,
            },
            paper_reference="Standard"
        ))
        
        # FARE-loss (evaluation metric)
        self.register_method(MethodConfig(
            name="fare_loss",
            category="evaluation",
            description="FARE alignment loss metric",
            enabled=True,
            baseline=False,
            hyperparameters={
                "distance_metric": "l2",
                "alignment_target": "class_token",
            },
            paper_reference="This paper (Equation 3)"
        ))
        
    def _register_parameter_sweeps(self):
        """Register parameter sweep configurations."""
        # Lambda (preservation weight) sweep for FARE
        self.register_sweep(SweepConfig(
            parameter="lambda_preserve",
            values=[0.1, 0.5, 1.0, 2.0, 5.0],
            method="fare",
            description="FARE preservation weight ablation (Table 6)"
        ))
        
        # Epsilon sweep (adversarial budget)
        self.register_sweep(SweepConfig(
            parameter="epsilon",
            values=[2.0/255.0, 4.0/255.0, 8.0/255.0, 16.0/255.0],
            method="all",
            description="Adversarial perturbation budget sweep"
        ))
        
        # Attack steps sweep
        self.register_sweep(SweepConfig(
            parameter="attack_steps",
            values=[10, 20, 50, 100],
            method="pgd",
            description="PGD attack steps ablation"
        ))
        
        # Alignment target sweep (FARE ablation)
        self.register_sweep(SweepConfig(
            parameter="alignment_target",
            values=["class_token", "patch_tokens", "both"],
            method="fare",
            description="FARE alignment target ablation (Table 6)"
        ))
        
        # Distance metric sweep
        self.register_sweep(SweepConfig(
            parameter="distance_metric",
            values=["l2", "cosine", "kl"],
            method="fare",
            description="Distance metric ablation"
        ))
        
    def register_method(self, method: MethodConfig):
        """Register a method in the registry."""
        self._methods[method.name] = method
        
    def register_attack(self, attack: AttackConfig):
        """Register an attack configuration."""
        self._attacks[attack.name] = attack
        
    def register_sweep(self, sweep: SweepConfig):
        """Register a parameter sweep."""
        key = f"{sweep.method}_{sweep.parameter}"
        self._sweeps[key] = sweep
        
    def get_method(self, name: str) -> Optional[MethodConfig]:
        """Get method configuration by name."""
        return self._methods.get(name)
        
    def get_attack(self, name: str) -> Optional[AttackConfig]:
        """Get attack configuration by name."""
        return self._attacks.get(name)
        
    def get_sweep(self, method: str, parameter: str) -> Optional[SweepConfig]:
        """Get parameter sweep configuration."""
        key = f"{method}_{parameter}"
        return self._sweeps.get(key)
        
    def list_methods(self, category: Optional[str] = None) -> List[str]:
        """List all registered methods, optionally filtered by category."""
        if category is None:
            return list(self._methods.keys())
        return [
            name for name, method in self._methods.items()
            if method.category == category
        ]
        
    def list_baselines(self) -> List[str]:
        """List all baseline methods."""
        return [
            name for name, method in self._methods.items()
            if method.baseline
        ]
        
    def list_attacks(self) -> List[str]:
        """List all registered attacks."""
        return list(self._attacks.keys())
        
    def get_evidence_matrix(self) -> Dict[str, Any]:
        """
        Generate evidence obligation matrix for paper reproduction.
        
        Returns contract matrix mapping paper claims to implementation.
        """
        matrix = {
            "vision_encoders": {
                "required": ["clip", "fare", "tecoa", "robust_clip", "vit"],
                "implemented": self.list_methods(category="vision_encoder"),
                "baselines": [m for m in self.list_methods(category="vision_encoder") 
                             if self._methods[m].baseline],
            },
            "lvlms": {
                "required": ["llava", "openflamingo"],
                "implemented": self.list_methods(category="lvlm"),
            },
            "attacks": {
                "required": ["pgd", "apgd", "autoattack", "random"],
                "implemented": self.list_attacks(),
                "baselines": ["random"],
            },
            "training_methods": {
                "required": ["fine_tuning", "adapter", "baseline"],
                "implemented": self.list_methods(category="training"),
            },
            "evaluation_methods": {
                "required": ["pope", "cot", "cli", "fare_loss"],
                "implemented": self.list_methods(category="evaluation"),
            },
            "parameter_sweeps": {
                "implemented": list(self._sweeps.keys()),
            },
            "coverage": {
                "total_methods": len(self._methods),
                "total_attacks": len(self._attacks),
                "total_sweeps": len(self._sweeps),
            },
        }
        
        # Compute coverage scores
        for category in ["vision_encoders", "lvlms", "attacks", "training_methods", "evaluation_methods"]:
            required = set(matrix[category]["required"])
            implemented = set(matrix[category]["implemented"])
            matrix[category]["coverage_score"] = len(required & implemented) / len(required) if required else 1.0
            matrix[category]["missing"] = list(required - implemented)
            
        return matrix
        
    def get_experiment_registry(self) -> Dict[str, Any]:
        """
        Generate experiment protocol registry.
        
        Returns mapping of experiments to method/dataset/metric configurations.
        """
        experiments = {
            "table_1_lvlm_robustness": {
                "methods": ["llava", "openflamingo"],
                "vision_encoders": ["clip", "fare", "tecoa"],
                "attacks": ["pgd", "autoattack"],
                "datasets": ["coco", "flickr30k", "vqav2", "textvqa"],
                "metrics": ["clean_accuracy", "robust_accuracy"],
            },
            "table_2_fare_ablation": {
                "methods": ["fare"],
                "vision_encoders": ["fare"],
                "parameter_sweep": "lambda_preserve",
                "datasets": ["imagenet"],
                "metrics": ["clean_accuracy", "robust_accuracy"],
            },
            "table_3_zero_shot_classification": {
                "methods": ["clip", "tecoa", "fare"],
                "attacks": ["pgd", "apgd", "autoattack"],
                "datasets": ["imagenet", "cifar10", "cifar100", "caltech101", "dtd", "eurosat", "fgvc", "food101", "oxford_pets", "stanford_cars", "sun397"],
                "metrics": ["clean_accuracy", "robust_accuracy"],
            },
            "table_4_pope_hallucination": {
                "methods": ["llava"],
                "vision_encoders": ["clip", "fare", "tecoa"],
                "datasets": ["pope_random", "pope_popular", "pope_adversarial"],
                "metrics": ["accuracy", "f1", "precision"],
            },
            "table_5_jailbreak": {
                "methods": ["llava"],
                "vision_encoders": ["clip", "fare", "tecoa"],
                "attacks": ["jailbreak"],
                "datasets": ["harmful_instructions"],
                "metrics": ["attack_success_rate"],
            },
            "figure_1_radar": {
                "methods": ["clip", "tecoa", "fare"],
                "datasets": ["imagenet", "cifar10", "cifar100"],
                "metrics": ["clean_accuracy", "robust_accuracy"],
                "visualization": "radar_plot",
            },
        }
        
        return experiments
        
    def export_to_json(self, output_dir: str = "results") -> Dict[str, str]:
        """
        Export registry artifacts to JSON files.
        
        Returns mapping of artifact names to output paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Export evidence matrix
        evidence_matrix_path = output_dir / "evidence_contract_matrix.json"
        with open(evidence_matrix_path, 'w') as f:
            json.dump(self.get_evidence_matrix(), f, indent=2)
        artifacts["evidence_matrix"] = str(evidence_matrix_path)
        
        # Export experiment registry
        experiment_registry_path = output_dir / "experiment_registry.json"
        with open(experiment_registry_path, 'w') as f:
            json.dump(self.get_experiment_registry(), f, indent=2)
        artifacts["experiment_registry"] = str(experiment_registry_path)
        
        # Export method configurations
        methods_path = output_dir / "method_configurations.json"
        methods_data = {
            name: asdict(method) for name, method in self._methods.items()
        }
        with open(methods_path, 'w') as f:
            json.dump(methods_data, f, indent=2)
        artifacts["methods"] = str(methods_path)
        
        # Export attack configurations
        attacks_path = output_dir / "attack_configurations.json"
        attacks_data = {
            name: asdict(attack) for name, attack in self._attacks.items()
        }
        with open(attacks_path, 'w') as f:
            json.dump(attacks_data, f, indent=2)
        artifacts["attacks"] = str(attacks_path)
        
        # Export sweep configurations
        sweeps_path = output_dir / "parameter_sweeps.json"
        sweeps_data = {
            key: asdict(sweep) for key, sweep in self._sweeps.items()
        }
        with open(sweeps_path, 'w') as f:
            json.dump(sweeps_data, f, indent=2)
        artifacts["sweeps"] = str(sweeps_path)
        
        # Export artifact manifest
        manifest_path = output_dir / "artifact_manifest.json"
        manifest = {
            "registry_version": "1.0.0",
            "total_methods": len(self._methods),
            "total_attacks": len(self._attacks),
            "total_sweeps": len(self._sweeps),
            "artifacts": artifacts,
            "generated_at": self._get_timestamp(),
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        artifacts["manifest"] = str(manifest_path)
        
        return artifacts
        
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        try:
            from datetime import datetime
            return datetime.utcnow().isoformat() + "Z"
        except:
            return "unknown"


# ============================================================================
# Method Selector Functions
# ============================================================================

def get_registry() -> MethodRegistry:
    """Get the global method registry instance."""
    return MethodRegistry()


def select_method(method_name: str, registry: Optional[MethodRegistry] = None) -> Optional[MethodConfig]:
    """
    Select a method by name from the registry.
    
    Args:
        method_name: Name of the method to select
        registry: Method registry (creates new if None)
        
    Returns:
        Method configuration or None if not found
    """
    if registry is None:
        registry = get_registry()
    return registry.get_method(method_name)


def select_attack(attack_name: str, epsilon: Optional[float] = None, registry: Optional[MethodRegistry] = None) -> Optional[AttackConfig]:
    """
    Select an attack by name from the registry.
    
    Args:
        attack_name: Name of the attack to select
        epsilon: Override epsilon value (optional)
        registry: Method registry (creates new if None)
        
    Returns:
        Attack configuration or None if not found
    """
    if registry is None:
        registry = get_registry()
    
    attack = registry.get_attack(attack_name)
    if attack is not None and epsilon is not None:
        # Create modified copy with new epsilon
        import copy
        attack = copy.deepcopy(attack)
        attack.epsilon = epsilon
    
    return attack