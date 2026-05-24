"""
LCA-on-the-Line: Soft Label Training Sweep Registry

This module exposes bounded parameter sweeps for soft label training experiments
(Table 5: WordNet soft labels, Table 6: Latent hierarchy soft labels, Table 9: Ablation study).

Paper evidence context:
- Table 5: Soft labeling with WordNet for Linear Probing
  - Baseline: Cross Entropy only
  - Ours: CE + LCA soft loss + weight interpolation
- Table 6: Soft Labeling with Latent Hierarchies for Linear Probing
  - Using K-means constructed hierarchies from pretrained models
- Table 9: Ablation Study on Soft Loss Labels
  - CE-only, Soft Loss, Interpolation, No ID Accuracy Drop

Implementation surfaces: training_loop, model_or_method, metric_formula

reference_grounding: paperbench_ref_006 configs/imagenet.py
reference_grounding: paperbench_ref_006 configs/imagenet_linear.py
reference_grounding: paperbench_ref_006 configs/imagenet_short.py
reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SoftLabelTrainingConfig:
    """Configuration for soft label training with hierarchical taxonomies."""
    
    # Experiment identification
    experiment_id: str
    description: str
    
    # Model configuration
    model_name: str
    pretrained: bool = True
    freeze_backbone: bool = True  # For linear probing
    
    # Dataset configuration
    dataset: str = "imagenet"
    num_classes: int = 1000
    
    # Hierarchy configuration
    hierarchy_source: str = "wordnet"  # Options: wordnet, latent_kmeans, custom
    hierarchy_path: Optional[str] = None
    
    # Clustering configuration (for latent hierarchies)
    clustering_layers: int = 2  # Paper: 2 or 3 layers
    clustering_nums: List[int] = field(default_factory=lambda: [10, 100])  # Root: 10, Leaf: 100
    
    # Soft label loss configuration
    use_soft_loss: bool = True
    soft_label_temperature: float = 1.0  # Temperature for soft label distribution
    lca_loss_weight: float = 0.5  # Weight for LCA soft loss
    
    # Training hyperparameters
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Weight interpolation (following Wortsman et al., 2022)
    use_weight_interpolation: bool = True
    interpolation_alpha: float = 0.5  # Alpha for CE-only and CE+soft models
    
    # Ablation flags
    disable_id_accuracy_drop: bool = False
    use_ce_only: bool = False


# =============================================================================
# Table 5: Soft Labeling with WordNet for Linear Probing
# Paper: "Results show that integrating soft loss consistently improves model 
# OOD performance, without causing accuracy drop on in-distribution ImageNet."
# =============================================================================

TABLE5_WORDNET_SOFT_LABEL_SWEEPS: Dict[str, SoftLabelTrainingConfig] = {
    # Baseline: Cross Entropy only
    "resnet18_ce_only": SoftLabelTrainingConfig(
        experiment_id="table5_resnet18_baseline",
        description="Table 5 Baseline: ResNet-18 trained with Cross Entropy only",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        use_soft_loss=False,
        use_weight_interpolation=False,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    "resnet50_ce_only": SoftLabelTrainingConfig(
        experiment_id="table5_resnet50_baseline",
        description="Table 5 Baseline: ResNet-50 trained with Cross Entropy only",
        model_name="resnet50",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        use_soft_loss=False,
        use_weight_interpolation=False,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    "vit_b_16_ce_only": SoftLabelTrainingConfig(
        experiment_id="table5_vit_b_16_baseline",
        description="Table 5 Baseline: ViT-B/16 trained with Cross Entropy only",
        model_name="vit_b_16",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        use_soft_loss=False,
        use_weight_interpolation=False,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # Ours: CE + LCA soft loss + weight interpolation
    "resnet18_soft_wordnet": SoftLabelTrainingConfig(
        experiment_id="table5_resnet18_ours",
        description="Table 5 Ours: ResNet-18 with WordNet soft labels + interpolation",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    "resnet50_soft_wordnet": SoftLabelTrainingConfig(
        experiment_id="table5_resnet50_ours",
        description="Table 5 Ours: ResNet-50 with WordNet soft labels + interpolation",
        model_name="resnet50",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    "vit_b_16_soft_wordnet": SoftLabelTrainingConfig(
        experiment_id="table5_vit_b_16_ours",
        description="Table 5 Ours: ViT-B/16 with WordNet soft labels + interpolation",
        model_name="vit_b_16",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
}


# =============================================================================
# Table 6: Soft Labeling with Latent Hierarchies for Linear Probing
# Paper: "Using latent hierarchies constructed from pretrained models using 
# K-means clustering also delivers a generalization boost"
# =============================================================================

TABLE6_LATENT_HIERARCHY_SWEEPS: Dict[str, SoftLabelTrainingConfig] = {
    # Baseline: CE only
    "resnet18_ce_baseline": SoftLabelTrainingConfig(
        experiment_id="table6_resnet18_baseline",
        description="Table 6 Baseline: ResNet-18 with CE only",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="latent_kmeans",
        use_soft_loss=False,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # Latent hierarchy with 2 layers: Root 10 clusters, Leaf 100 clusters
    "resnet18_latent_2layer": SoftLabelTrainingConfig(
        experiment_id="table6_resnet18_latent_2layer",
        description="Table 6: ResNet-18 with 2-layer latent hierarchy (10->100 clusters)",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="latent_kmeans",
        clustering_layers=2,
        clustering_nums=[10, 100],
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # Latent hierarchy with 3 layers
    "resnet18_latent_3layer": SoftLabelTrainingConfig(
        experiment_id="table6_resnet18_latent_3layer",
        description="Table 6: ResNet-18 with 3-layer latent hierarchy (10->50->100 clusters)",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="latent_kmeans",
        clustering_layers=3,
        clustering_nums=[10, 50, 100],
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
}


# =============================================================================
# Table 9: Ablation Study on Soft Loss Labels
# Paper ablation components:
# - CE-only: baseline
# - Soft Loss: without interpolation
# - Interpolation: with weight interpolation
# - No ID Accuracy Drop: maintaining ID accuracy
# =============================================================================

TABLE9_ABLATION_SWEEPS: Dict[str, SoftLabelTrainingConfig] = {
    # CE-only baseline
    "ablation_ce_only": SoftLabelTrainingConfig(
        experiment_id="table9_ce_only",
        description="Table 9 Ablation: CE-only baseline",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        use_soft_loss=False,
        use_weight_interpolation=False,
        use_ce_only=True,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # Soft Loss without interpolation
    "ablation_soft_loss_only": SoftLabelTrainingConfig(
        experiment_id="table9_soft_loss_only",
        description="Table 9 Ablation: Soft loss without weight interpolation",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=False,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # Full method: Soft Loss + Interpolation
    "ablation_full_method": SoftLabelTrainingConfig(
        experiment_id="table9_full_method",
        description="Table 9 Ablation: Soft loss + weight interpolation (full method)",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.5,
        use_weight_interpolation=True,
        interpolation_alpha=0.5,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
    
    # No ID accuracy drop variant
    "ablation_no_id_drop": SoftLabelTrainingConfig(
        experiment_id="table9_no_id_drop",
        description="Table 9 Ablation: Optimized to maintain ID accuracy",
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=True,
        hierarchy_source="wordnet",
        hierarchy_path="resources/hierarchy/imagenet_fiveai.csv",
        use_soft_loss=True,
        soft_label_temperature=1.0,
        lca_loss_weight=0.3,  # Lower weight to maintain ID accuracy
        use_weight_interpolation=True,
        interpolation_alpha=0.7,  # Higher alpha favors CE model
        disable_id_accuracy_drop=True,
        batch_size=256,
        num_epochs=50,
        learning_rate=0.1,
    ),
}


# =============================================================================
# Bounded Parameter Sweep Space
# Expose bounded config entries for paper experiments without exhaustive execution
# =============================================================================

PARAMETER_SWEEP_SPACE: Dict[str, Any] = {
    # Clustering configuration for latent hierarchies
    "clustering": {
        "layers": [2, 3],  # Paper: 2 or 3 layers
        "root_clusters": [5, 10, 20],  # Root layer cluster counts
        "intermediate_clusters": [30, 50, 75],  # For 3-layer hierarchies
        "leaf_clusters": [50, 100, 200],  # Leaf layer cluster counts
    },
    
    # Soft label loss hyperparameters
    "soft_label": {
        "temperature": [0.5, 1.0, 2.0, 4.0],  # Temperature for soft distribution
        "lca_loss_weight": [0.1, 0.3, 0.5, 0.7, 0.9],  # Weight for LCA soft loss
    },
    
    # Weight interpolation parameters
    "interpolation": {
        "alpha": [0.3, 0.5, 0.7, 0.9],  # Interpolation between CE-only and CE+soft
    },
    
    # Training hyperparameters
    "training": {
        "batch_size": [128, 256, 512],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_epochs": [30, 50, 100],
    },
}


# =============================================================================
# Registry Access Functions
# =============================================================================

def get_table5_configs() -> Dict[str, SoftLabelTrainingConfig]:
    """
    Get configurations for Table 5 experiments (WordNet soft labels).
    
    Returns:
        Dictionary mapping experiment IDs to training configurations
    """
    return TABLE5_WORDNET_SOFT_LABEL_SWEEPS


def get_table6_configs() -> Dict[str, SoftLabelTrainingConfig]:
    """
    Get configurations for Table 6 experiments (Latent hierarchy soft labels).
    
    Returns:
        Dictionary mapping experiment IDs to training configurations
    """
    return TABLE6_LATENT_HIERARCHY_SWEEPS


def get_table9_configs() -> Dict[str, SoftLabelTrainingConfig]:
    """
    Get configurations for Table 9 ablation study.
    
    Returns:
        Dictionary mapping experiment IDs to training configurations
    """
    return TABLE9_ABLATION_SWEEPS


def get_all_configs() -> Dict[str, SoftLabelTrainingConfig]:
    """
    Get all soft label training configurations.
    
    Returns:
        Combined dictionary of all configurations
    """
    all_configs = {}
    all_configs.update(TABLE5_WORDNET_SOFT_LABEL_SWEEPS)
    all_configs.update(TABLE6_LATENT_HIERARCHY_SWEEPS)
    all_configs.update(TABLE9_ABLATION_SWEEPS)
    return all_configs


def get_config_by_id(experiment_id: str) -> Optional[SoftLabelTrainingConfig]:
    """
    Retrieve a specific configuration by experiment ID.
    
    Args:
        experiment_id: Unique identifier for the experiment
        
    Returns:
        SoftLabelTrainingConfig if found, None otherwise
    """
    all_configs = get_all_configs()
    return all_configs.get(experiment_id)


def get_parameter_sweep_space() -> Dict[str, Any]:
    """
    Get bounded parameter sweep space for soft label training experiments.
    
    Returns:
        Dictionary containing bounded sweep ranges for all parameters
    """
    return PARAMETER_SWEEP_SPACE


def create_custom_config(
    model_name: str,
    hierarchy_source: str = "wordnet",
    use_soft_loss: bool = True,
    **kwargs: Any
) -> SoftLabelTrainingConfig:
    """
    Create a custom training configuration with specified parameters.
    
    Args:
        model_name: Name of the model (e.g., "resnet18", "vit_b_16")
        hierarchy_source: Source of hierarchy ("wordnet", "latent_kmeans")
        use_soft_loss: Whether to use soft label loss
        **kwargs: Additional configuration parameters to override defaults
        
    Returns:
        SoftLabelTrainingConfig instance
    """
    config = SoftLabelTrainingConfig(
        experiment_id=f"custom_{model_name}_{hierarchy_source}",
        description=f"Custom config for {model_name} with {hierarchy_source}",
        model_name=model_name,
        hierarchy_source=hierarchy_source,
        use_soft_loss=use_soft_loss,
    )
    
    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# =============================================================================
# Dry-run / Smoke Test Configuration
# For runtime_smoke and docker_validate modes
# =============================================================================

SMOKE_TEST_CONFIG = SoftLabelTrainingConfig(
    experiment_id="smoke_test",
    description="Minimal configuration for smoke testing",
    model_name="resnet18",
    pretrained=True,
    freeze_backbone=True,
    hierarchy_source="wordnet",
    use_soft_loss=True,
    soft_label_temperature=1.0,
    lca_loss_weight=0.5,
    use_weight_interpolation=True,
    interpolation_alpha=0.5,
    batch_size=8,  # Small batch for smoke test
    num_epochs=1,  # Single epoch for smoke test
    learning_rate=0.1,
)


def get_smoke_test_config() -> SoftLabelTrainingConfig:
    """
    Get minimal configuration for smoke testing.
    
    Returns:
        Smoke test configuration with reduced parameters
    """
    return SMOKE_TEST_CONFIG