#!/usr/bin/env python3
"""
LCA-on-the-Line: Environment and Task Registry

Paper-derived environment/task registry with ids, aliases, setup metadata, and factory/config hooks.
Exposes datasets, benchmarks, evaluation protocols, and paper evidence obligation matrix.

Method obligations:
- Environment/task registry for: OOD Top-1, well-trained model, imagenet, laion, Fig 3, Fig 5,
  ImgN-v2, ImageNet-v2, VM+VLM, 75 models, Correlating OOD Top-1, OOD Top-5
- Dataset/benchmark registry for: imagenet, laion, imagenet_1k, imagenet_c, imagenet_r,
  imagenet_v2, imagenet_sketch
- Paper evidence obligation matrix binding experiments to datasets, methods, parameters, trends, artifacts

Binding addendum clarifications:
- ImageNet download via HuggingFace using trust_remote_code=True
- ImageNet-v2: https://imagenetv2.org/

reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
reference_grounding: paperbench_ref_001 references/depth/stereo/README.md
reference_grounding: paperbench_ref_001 torchvision/models/detection/keypoint_rcnn.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Registry - Paper Evidence Contract
# reference_grounding: paperbench_ref_001 references/depth/stereo/README.md
# =============================================================================

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ID Datasets
    "imagenet": {
        "dataset_id": "imagenet",
        "aliases": ["imagenet", "imagenet_1k", "ILSVRC2012", "ImageNet-1K"],
        "name": "ImageNet-1K",
        "num_classes": 1000,
        "splits": ["train", "val"],
        "source_type": "huggingface",
        "source_path": "ILSVRC/imagenet-1k",
        "data_dir": "data/imagenet",
        "trust_remote_code": True,
        "description": "ImageNet-1K (ILSVRC2012) in-distribution dataset",
        "paper_usage": "ID dataset for LCA distance computation and model training",
        "download_instructions": "Use HuggingFace datasets.load_dataset with trust_remote_code=True",
    },
    "imagenet_1k": {
        "dataset_id": "imagenet_1k",
        "aliases": ["imagenet_1k", "imagenet-1k"],
        "name": "ImageNet-1K",
        "num_classes": 1000,
        "splits": ["train", "val"],
        "source_type": "huggingface",
        "source_path": "ILSVRC/imagenet-1k",
        "data_dir": "data/imagenet",
        "trust_remote_code": True,
        "description": "ImageNet-1K alias for consistency",
        "paper_usage": "ID dataset alias",
    },
    "laion": {
        "dataset_id": "laion",
        "aliases": ["laion", "laion-400m", "LAION"],
        "name": "LAION-400M",
        "num_classes": None,
        "splits": ["train"],
        "source_type": "external",
        "source_path": "https://laion.ai/",
        "data_dir": "data/laion",
        "description": "LAION-400M pretraining dataset",
        "paper_usage": "Pretraining dataset for VLMs (CLIP, OpenCLIP)",
    },
    # OOD Datasets
    "imagenet_v2": {
        "dataset_id": "imagenet_v2",
        "aliases": ["imagenet_v2", "imagenet-v2", "ImgN-v2", "ImageNet-v2"],
        "name": "ImageNet-v2 Matched Frequency",
        "num_classes": 1000,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://imagenetv2.org/",
        "data_dir": "data/imagenet-v2",
        "variant": "matched-frequency",
        "description": "ImageNet-v2 test set with matched frequency variant",
        "paper_usage": "OOD dataset for evaluating distribution shift robustness",
        "download_instructions": "Download from https://imagenetv2.org/",
    },
    "imagenet_a": {
        "dataset_id": "imagenet_a",
        "aliases": ["imagenet_a", "imagenet-a", "ImageNet-A"],
        "name": "ImageNet-A",
        "num_classes": 200,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://github.com/hendrycks/natural-adv-examples",
        "data_dir": "data/imagenet-a",
        "description": "ImageNet-A natural adversarial examples",
        "paper_usage": "OOD dataset for evaluating adversarial robustness",
    },
    "imagenet_r": {
        "dataset_id": "imagenet_r",
        "aliases": ["imagenet_r", "imagenet-r", "ImageNet-R"],
        "name": "ImageNet-R",
        "num_classes": 200,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://github.com/hendrycks/imagenet-r",
        "data_dir": "data/imagenet-r",
        "description": "ImageNet-R renditions",
        "paper_usage": "OOD dataset for evaluating style shift robustness",
    },
    "imagenet_sketch": {
        "dataset_id": "imagenet_sketch",
        "aliases": ["imagenet_sketch", "imagenet-sketch", "ImageNet-Sketch"],
        "name": "ImageNet-Sketch",
        "num_classes": 1000,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://github.com/HaohanWang/ImageNet-Sketch",
        "data_dir": "data/imagenet-sketch",
        "description": "ImageNet-Sketch black-and-white sketches",
        "paper_usage": "OOD dataset for evaluating sketch domain shift",
    },
    "imagenet_c": {
        "dataset_id": "imagenet_c",
        "aliases": ["imagenet_c", "imagenet-c", "ImageNet-C"],
        "name": "ImageNet-C",
        "num_classes": 1000,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://github.com/hendrycks/robustness",
        "data_dir": "data/imagenet-c",
        "corruption_types": 19,
        "severity_levels": 5,
        "description": "ImageNet-C common corruptions",
        "paper_usage": "OOD dataset for evaluating corruption robustness",
    },
    "objectnet": {
        "dataset_id": "objectnet",
        "aliases": ["objectnet", "ObjectNet"],
        "name": "ObjectNet",
        "num_classes": 113,
        "splits": ["test"],
        "source_type": "external",
        "source_path": "https://objectnet.dev/",
        "data_dir": "data/objectnet",
        "description": "ObjectNet test set with pose/background variations",
        "paper_usage": "OOD dataset for evaluating viewpoint robustness",
    },
}


# =============================================================================
# Environment/Task Registry - Paper Evidence Contract
# reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
# =============================================================================

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ood_top1_evaluation": {
        "task_id": "ood_top1_evaluation",
        "aliases": ["OOD Top-1", "ood_top1", "ood_accuracy"],
        "name": "OOD Top-1 Accuracy Evaluation",
        "task_type": "classification",
        "metric": "top1_accuracy",
        "datasets": ["imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "description": "Evaluate model Top-1 accuracy on OOD test sets",
        "paper_usage": "Core metric for LCA-on-the-Line correlation analysis (Fig 3, Fig 5)",
    },
    "ood_top5_evaluation": {
        "task_id": "ood_top5_evaluation",
        "aliases": ["OOD Top-5", "ood_top5"],
        "name": "OOD Top-5 Accuracy Evaluation",
        "task_type": "classification",
        "metric": "top5_accuracy",
        "datasets": ["imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "description": "Evaluate model Top-5 accuracy on OOD test sets",
        "paper_usage": "Additional OOD generalization metric",
    },
    "id_lca_distance": {
        "task_id": "id_lca_distance",
        "aliases": ["ID LCA", "lca_distance", "lca_mistake"],
        "name": "ID LCA Distance Computation",
        "task_type": "hierarchical_metric",
        "metric": "lca_distance",
        "datasets": ["imagenet"],
        "description": "Compute average LCA distance for model mistakes on ImageNet validation",
        "paper_usage": "Key predictor of OOD performance in LCA-on-the-Line phenomenon",
    },
    "lca_ood_correlation": {
        "task_id": "lca_ood_correlation",
        "aliases": ["Correlating OOD Top-1", "lca_correlation", "Fig 3", "Fig 5"],
        "name": "LCA-OOD Correlation Analysis",
        "task_type": "correlation_analysis",
        "metrics": ["lca_distance", "top1_accuracy", "pearson_r", "spearman_rho"],
        "id_dataset": "imagenet",
        "ood_datasets": ["imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "description": "Correlate ID LCA distance with OOD Top-1 accuracy across models",
        "paper_usage": "Core contribution: demonstrate LCA-on-the-Line phenomenon (Fig 3, Table 1)",
    },
    "vm_vlm_benchmark": {
        "task_id": "vm_vlm_benchmark",
        "aliases": ["VM+VLM", "75 models", "model_benchmark"],
        "name": "Vision Model + Vision-Language Model Benchmark",
        "task_type": "multi_model_evaluation",
        "model_categories": ["vision_models", "vision_language_models"],
        "num_models": 75,
        "num_vms": 36,
        "num_vlms": 39,
        "datasets": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "description": "Evaluate 75 pretrained models (36 VMs + 39 VLMs) on ID and OOD datasets",
        "paper_usage": "Comprehensive benchmark for LCA-on-the-Line validation (Table 1, Table 2)",
    },
    "imagenet_training": {
        "task_id": "imagenet_training",
        "aliases": ["imagenet", "well-trained model", "supervised_training"],
        "name": "ImageNet Supervised Training",
        "task_type": "training",
        "dataset": "imagenet",
        "description": "Standard supervised training on ImageNet-1K",
        "paper_usage": "Train models with soft labels and hierarchy-aware methods",
    },
    "laion_pretraining": {
        "task_id": "laion_pretraining",
        "aliases": ["laion", "vlm_pretraining"],
        "name": "LAION Pretraining",
        "task_type": "pretraining",
        "dataset": "laion",
        "description": "Vision-language pretraining on LAION dataset",
        "paper_usage": "Pretrain VLMs with contrastive learning",
    },
}


# =============================================================================
# Paper Evidence Obligation Matrix
# Binding experiments to datasets, methods, parameters, trends, and artifacts
# =============================================================================

PAPER_EVIDENCE_MATRIX = [
    {
        "experiment_id": "exp_lca_correlation",
        "name": "LCA-on-the-Line Correlation",
        "paper_reference": "Fig 3, Fig 5, Table 1",
        "environments": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "datasets": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "methods": ["75_pretrained_models"],
        "baselines": ["id_accuracy", "aline_s", "aline_d"],
        "metrics": ["lca_distance", "top1_accuracy", "pearson_r", "spearman_rho"],
        "parameters": {},
        "expected_trend": "Strong positive correlation between ID LCA distance and OOD Top-1 accuracy",
        "decision_claim": "ID LCA is superior predictor of OOD performance compared to ID accuracy",
        "result_artifacts": ["results/figures/figure3_lca_on_the_line.pdf", "results/correlations.json"],
    },
    {
        "experiment_id": "exp_prediction_comparison",
        "name": "OOD Performance Prediction Comparison",
        "paper_reference": "Table 3",
        "environments": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "datasets": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "methods": ["id_lca", "id_accuracy", "aline_s", "aline_d"],
        "baselines": ["id_accuracy", "aline_s", "aline_d"],
        "metrics": ["mae", "rmse"],
        "parameters": {},
        "expected_trend": "ID LCA achieves lowest MAE for predicting OOD accuracy",
        "decision_claim": "ID LCA outperforms baselines in OOD performance prediction",
        "result_artifacts": ["results/tables/table3_prediction_mae.json"],
    },
    {
        "experiment_id": "exp_latent_taxonomy",
        "name": "Latent Taxonomy Inference",
        "paper_reference": "Section 5, Table 11",
        "environments": ["imagenet"],
        "datasets": ["imagenet"],
        "methods": ["kmeans_clustering"],
        "baselines": ["wordnet_hierarchy"],
        "metrics": ["lca_distance", "silhouette_score"],
        "parameters": {"num_layers": [2, 3, 4], "num_clusters_per_layer": [10, 20, 50]},
        "expected_trend": "K-means latent hierarchy achieves comparable LCA predictive power to WordNet",
        "decision_claim": "Pretrained models encode implicit hierarchical structure",
        "result_artifacts": ["results/latent_taxonomy.json", "results/tables/table11_latent_hierarchy.json"],
    },
    {
        "experiment_id": "exp_soft_labels",
        "name": "Soft Label Training",
        "paper_reference": "Section 6, Table 5",
        "environments": ["imagenet", "imagenet_v2", "imagenet_r"],
        "datasets": ["imagenet", "imagenet_v2", "imagenet_r"],
        "methods": ["soft_labels_wordnet", "soft_labels_latent"],
        "baselines": ["hard_labels"],
        "metrics": ["top1_accuracy", "lca_distance"],
        "parameters": {"temperature": [1.0, 2.0, 5.0], "lca_loss_weight": [0.1, 0.5, 1.0]},
        "expected_trend": "Soft labels with hierarchy improve OOD accuracy",
        "decision_claim": "Hierarchy-aware training improves generalization",
        "result_artifacts": ["results/tables/table5_soft_labels.json", "checkpoints/resnet18_soft_labels.pth"],
    },
    {
        "experiment_id": "exp_vlm_prompting",
        "name": "Hierarchy-Aware VLM Prompting",
        "paper_reference": "Section 6, Table 12",
        "environments": ["imagenet", "imagenet_v2"],
        "datasets": ["imagenet", "imagenet_v2"],
        "methods": ["hierarchical_prompts"],
        "baselines": ["standard_prompts"],
        "metrics": ["top1_accuracy", "top5_accuracy"],
        "parameters": {"prompt_template": ["standard", "hierarchical", "hierarchical_with_examples"]},
        "expected_trend": "Hierarchical prompts improve zero-shot VLM accuracy",
        "decision_claim": "Embedding hierarchy in prompts benefits VLMs",
        "result_artifacts": ["results/tables/table12_vlm_prompts.json"],
    },
]


# =============================================================================
# Experiment Registry - Named Experiment Protocols
# =============================================================================

EXPERIMENT_REGISTRY = {
    "lca_correlation": {
        "experiment_id": "lca_correlation",
        "name": "LCA-on-the-Line Correlation Analysis",
        "environments": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "models": "all_75_models",
        "metrics": ["lca_distance", "top1_accuracy", "pearson_r"],
        "paper_figures": ["Fig 3", "Fig 5"],
    },
    "prediction_comparison": {
        "experiment_id": "prediction_comparison",
        "name": "OOD Performance Prediction Comparison",
        "environments": ["imagenet", "imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch", "objectnet"],
        "predictors": ["id_lca", "id_accuracy", "aline_s", "aline_d"],
        "metrics": ["mae", "rmse"],
        "paper_tables": ["Table 3"],
    },
}


# =============================================================================
# Parameter Sweep Configuration
# =============================================================================

PARAMETER_SWEEPS = {
    "latent_taxonomy_layers": {
        "sweep_id": "latent_taxonomy_layers",
        "parameter": "num_layers",
        "values": [2, 3, 4],
        "description": "Number of hierarchy layers for K-means clustering",
        "experiment": "exp_latent_taxonomy",
    },
    "soft_label_temperature": {
        "sweep_id": "soft_label_temperature",
        "parameter": "temperature",
        "values": [1.0, 2.0, 5.0],
        "description": "Temperature for soft label smoothing",
        "experiment": "exp_soft_labels",
    },
    "lca_loss_weight": {
        "sweep_id": "lca_loss_weight",
        "parameter": "lca_loss_weight",
        "values": [0.0, 0.1, 0.5, 1.0],
        "description": "Weight of LCA loss in training objective",
        "experiment": "exp_soft_labels",
    },
}


# =============================================================================
# Registry Access Functions
# =============================================================================

def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Get dataset configuration by ID or alias."""
    # Direct lookup
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]
    
    # Search aliases
    for ds_id, config in DATASET_REGISTRY.items():
        if dataset_id in config.get("aliases", []):
            return config
    
    raise ValueError(f"Dataset '{dataset_id}' not found in registry")


def get_environment(task_id: str) -> Dict[str, Any]:
    """Get environment/task configuration by ID or alias."""
    # Direct lookup
    if task_id in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[task_id]
    
    # Search aliases
    for env_id, config in ENVIRONMENT_REGISTRY.items():
        if task_id in config.get("aliases", []):
            return config
    
    raise ValueError(f"Environment/task '{task_id}' not found in registry")


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """Get experiment configuration by ID."""
    if experiment_id in EXPERIMENT_REGISTRY:
        return EXPERIMENT_REGISTRY[experiment_id]
    
    raise ValueError(f"Experiment '{experiment_id}' not found in registry")


def list_datasets(filter_type: Optional[str] = None) -> List[str]:
    """List all registered datasets, optionally filtered by type."""
    datasets = list(DATASET_REGISTRY.keys())
    if filter_type:
        datasets = [
            ds_id for ds_id in datasets
            if DATASET_REGISTRY[ds_id].get("paper_usage", "").lower().find(filter_type.lower()) >= 0
        ]
    return datasets


def list_environments() -> List[str]:
    """List all registered environments/tasks."""
    return list(ENVIRONMENT_REGISTRY.keys())


def list_experiments() -> List[str]:
    """List all registered experiments."""
    return list(EXPERIMENT_REGISTRY.keys())


def get_paper_evidence_matrix() -> List[Dict[str, Any]]:
    """Get the complete paper evidence obligation matrix."""
    return PAPER_EVIDENCE_MATRIX


# =============================================================================
# Artifact Writing Functions
# =============================================================================

def write_registry_artifacts(output_dir: str = "results", dry_run: bool = False) -> None:
    """
    Write registry artifacts to output directory.
    
    Args:
        output_dir: Output directory path
        dry_run: If True, mark artifacts as dry-run/readiness artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write dataset registry
    dataset_registry_path = output_path / "dataset_registry.json"
    with open(dataset_registry_path, 'w') as f:
        json.dump({
            "dry_run": dry_run,
            "artifact_type": "dataset_registry" if not dry_run else "dataset_registry_schema",
            "datasets": DATASET_REGISTRY,
            "count": len(DATASET_REGISTRY),
        }, f, indent=2)
    logger.info(f"Written dataset registry to {dataset_registry_path}")
    
    # Write environment registry
    environment_registry_path = output_path / "environment_registry.json"
    with open(environment_registry_path, 'w') as f:
        json.dump({
            "dry_run": dry_run,
            "artifact_type": "environment_registry" if not dry_run else "environment_registry_schema",
            "environments": ENVIRONMENT_REGISTRY,
            "count": len(ENVIRONMENT_REGISTRY),
        }, f, indent=2)
    logger.info(f"Written environment registry to {environment_registry_path}")
    
    # Write experiment registry
    experiment_registry_path = output_path / "experiment_registry.json"
    with open(experiment_registry_path, 'w') as f:
        json.dump({
            "dry_run": dry_run,
            "artifact_type": "experiment_registry" if not dry_run else "experiment_registry_schema",
            "experiments": EXPERIMENT_REGISTRY,
            "count": len(EXPERIMENT_REGISTRY),
        }, f, indent=2)
    logger.info(f"Written experiment registry to {experiment_registry_path}")
    
    # Write paper evidence obligation matrix
    evidence_matrix_path = output_path / "evidence_contract_matrix.json"
    with open(evidence_matrix_path, 'w') as f:
        json.dump({
            "dry_run": dry_run,
            "artifact_type": "evidence_contract_matrix" if not dry_run else "evidence_contract_matrix_schema",
            "paper_evidence_matrix": PAPER_EVIDENCE_MATRIX,
            "count": len(PAPER_EVIDENCE_MATRIX),
            "parameter_sweeps": PARAMETER_SWEEPS,
        }, f, indent=2)
    logger.info(f"Written evidence contract matrix to {evidence_matrix_path}")
    
    # Write artifact manifest
    manifest_path = output_path / "artifact_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            "dry_run": dry_run,
            "artifact_type": "artifact_manifest" if not dry_run else "artifact_manifest_schema",
            "artifacts": {
                "dataset_registry": str(dataset_registry_path),
                "environment_registry": str(environment_registry_path),
                "experiment_registry": str(experiment_registry_path),
                "evidence_contract_matrix": str(evidence_matrix_path),
            },
            "registry_counts": {
                "datasets": len(DATASET_REGISTRY),
                "environments": len(ENVIRONMENT_REGISTRY),
                "experiments": len(EXPERIMENT_REGISTRY),
                "evidence_matrix_rows": len(PAPER_EVIDENCE_MATRIX),
                "parameter_sweeps": len(PARAMETER_SWEEPS),
            },
        }, f, indent=2)
    logger.info(f"Written artifact manifest to {manifest_path}")


def validate_registry_completeness() -> Dict[str, Any]:
    """
    Validate that all paper-stated environments, datasets, and experiments are registered.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    
    # Check required datasets
    required_datasets = ["imagenet", "laion", "imagenet_1k", "imagenet_c", "imagenet_r", 
                        "imagenet_v2", "imagenet_sketch"]
    for ds in required_datasets:
        try:
            get_dataset(ds)
        except ValueError:
            validation["valid"] = False
            validation["errors"].append(f"Required dataset '{ds}' not found in registry")
    
    # Check required environments
    required_environments = ["imagenet", "laion"]  # Paper evidence contract aliases
    for env in required_environments:
        try:
            get_environment(env)
        except ValueError:
            validation["warnings"].append(f"Environment alias '{env}' not found (may be OK if dataset exists)")
    
    # Check paper evidence matrix completeness
    if not PAPER_EVIDENCE_MATRIX:
        validation["valid"] = False
        validation["errors"].append("Paper evidence matrix is empty")
    
    for exp in PAPER_EVIDENCE_MATRIX:
        required_fields = ["experiment_id", "name", "paper_reference", "environments", 
                          "datasets", "methods", "metrics", "expected_trend", "result_artifacts"]
        for field in required_fields:
            if field not in exp:
                validation["warnings"].append(f"Experiment '{exp.get('experiment_id', 'unknown')}' missing field '{field}'")
    
    return validation


# =============================================================================
# Module-level execution for smoke testing
# =============================================================================

if __name__ == "__main__":
    # Smoke test: validate registry and write artifacts
    print("LCA-on-the-Line Environment Registry - Smoke Test")
    print("=" * 80)
    
    # List registries
    print(f"\nDatasets: {len(list_datasets())} registered")
    print(f"  ID datasets: {list_datasets('ID')}")
    print(f"  OOD datasets: {list_datasets('OOD')}")
    
    print(f"\nEnvironments: {len(list_environments())} registered")
    for env_id in list_environments():
        env = ENVIRONMENT_REGISTRY[env_id]
        print(f"  - {env_id}: {env['name']}")
    
    print(f"\nExperiments: {len(list_experiments())} registered")
    for exp_id in list_experiments():
        exp = EXPERIMENT_REGISTRY[exp_id]
        print(f"  - {exp_id}: {exp['name']}")
    
    print(f"\nPaper Evidence Matrix: {len(PAPER_EVIDENCE_MATRIX)} rows")
    for exp in PAPER_EVIDENCE_MATRIX:
        print(f"  - {exp['experiment_id']}: {exp['name']} ({exp['paper_reference']})")
    
    # Validate completeness
    print("\nValidating registry completeness...")
    validation = validate_registry_completeness()
    if validation["valid"]:
        print("✓ Registry validation passed")
    else:
        print("✗ Registry validation failed")
    
    if validation["errors"]:
        print("\nErrors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    # Write artifacts
    print("\nWriting registry artifacts...")
    write_registry_artifacts(output_dir="results", dry_run=True)
    print("\n✓ Smoke test complete")