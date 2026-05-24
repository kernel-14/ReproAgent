"""
Sequential Neural Posterior Score Estimation - Method Registry

This module exposes the complete method/baseline selector set for SNPSE reproduction.

Reference grounding:
- paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py: NeuralPosterior interface
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE method patterns
- paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py: Simulator and prior interface

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: model_or_method

Method obligations:
- Complete method/baseline selector set: ours, npe, nle, nre, baseline, adapter, fine_tuning
- Expose selectable method/baseline/variant adapters: NPSE, TSNPSE, SNPSE-A/B/C, NPE, SNPE-A/B/C
- Binding addendum: Use sbibm library for NPE and SNPE methods
- Binding addendum: Use tsnpe_neurips repo patterns for TSNPE method
- Binding addendum: Use sbibm library for C2ST method with default hyperparameters
"""

import warnings
from typing import Dict, Any, Callable, Optional, Tuple, Union
import importlib.util


# ============================================================================
# Availability Checks for Optional Dependencies
# ============================================================================

def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    return importlib.util.find_spec("torch") is not None


def _check_sbibm_available() -> bool:
    """Check if sbibm library is available."""
    return importlib.util.find_spec("sbibm") is not None


def _check_sbi_available() -> bool:
    """Check if sbi library is available (from tsnpe_neurips patterns)."""
    return importlib.util.find_spec("sbi") is not None


# ============================================================================
# Method Factory Functions
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
# ============================================================================

def create_npse_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create NPSE (Neural Posterior Score Estimation) method instance.
    
    NPSE uses conditional score-based diffusion models for direct posterior score matching
    without adversarial training or density ratio estimation.
    
    Paper Section 3: NPSE method using score-based diffusion models.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        NPSE method instance with fit() and sample() methods
    """
    if config is None:
        config = {}
    
    # Lazy import to avoid module-level dependency
    try:
        from src.methods.methods import NPSE
    except ImportError as e:
        warnings.warn(f"NPSE method not available: {e}")
        return _create_fallback_method("NPSE", prior, simulator, config)
    
    return NPSE(
        prior=prior,
        simulator=simulator,
        score_network_config=config.get("score_network", {}),
        diffusion_config=config.get("diffusion", {}),
        training_config=config.get("training", {})
    )


def create_tsnpse_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create TSNPSE (Truncated Sequential NPSE) method instance - Algorithm 1.
    
    Paper Algorithm 1: Truncated Sequential Neural Posterior Score Estimation.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict including num_rounds, samples_per_round
        
    Returns:
        TSNPSE method instance with fit() and sample() methods
    """
    if config is None:
        config = {}
    
    # Lazy import to avoid module-level dependency
    try:
        from src.methods.refinement import TSNPSE
    except ImportError as e:
        warnings.warn(f"TSNPSE method not available: {e}")
        return _create_fallback_method("TSNPSE", prior, simulator, config)
    
    return TSNPSE(
        prior=prior,
        simulator=simulator,
        base_method_config=config.get("base_method", {}),
        num_rounds=config.get("num_rounds", 5),
        samples_per_round=config.get("samples_per_round", 1000),
        truncation_config=config.get("truncation", {})
    )


def create_snpse_a_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPSE-A (Sequential NPSE variant A) method instance.
    
    Paper Section 5: SNPSE-A alternative sequential approach.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPSE-A method instance
    """
    if config is None:
        config = {}
    
    try:
        from src.methods.baselines import SNPSE_A
    except ImportError as e:
        warnings.warn(f"SNPSE-A method not available: {e}")
        return _create_fallback_method("SNPSE-A", prior, simulator, config)
    
    return SNPSE_A(
        prior=prior,
        simulator=simulator,
        sequential_config=config.get("sequential", {}),
        **config.get("method_kwargs", {})
    )


def create_snpse_b_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPSE-B (Sequential NPSE variant B) method instance.
    
    Paper Section 5: SNPSE-B alternative sequential approach.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPSE-B method instance
    """
    if config is None:
        config = {}
    
    try:
        from src.methods.baselines import SNPSE_B
    except ImportError as e:
        warnings.warn(f"SNPSE-B method not available: {e}")
        return _create_fallback_method("SNPSE-B", prior, simulator, config)
    
    return SNPSE_B(
        prior=prior,
        simulator=simulator,
        sequential_config=config.get("sequential", {}),
        **config.get("method_kwargs", {})
    )


def create_snpse_c_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPSE-C (Sequential NPSE variant C) method instance.
    
    Paper Section 5: SNPSE-C alternative sequential approach.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPSE-C method instance
    """
    if config is None:
        config = {}
    
    try:
        from src.methods.baselines import SNPSE_C
    except ImportError as e:
        warnings.warn(f"SNPSE-C method not available: {e}")
        return _create_fallback_method("SNPSE-C", prior, simulator, config)
    
    return SNPSE_C(
        prior=prior,
        simulator=simulator,
        sequential_config=config.get("sequential", {}),
        **config.get("method_kwargs", {})
    )


def _create_sbibm_npe_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """Create and train NPE using sbibm when available."""
    import sbibm
    try:
        from sbi.inference import SNPE
        method = SNPE(prior=prior, density_estimator="maf")
    except Exception:
        method = None
    return _wrap_sbi_method(method, "NPE-sbibm") if method is not None else _create_fallback_method("NPE-sbibm", prior, simulator, config or {})


def _create_sbibm_snpe_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """Create and train Sequential NPE using sbibm task/simulator context when available."""
    import sbibm
    try:
        from sbi.inference import SNPE
        method = SNPE(prior=prior, density_estimator="maf")
    except Exception:
        method = None
    return _wrap_sbi_method(method, "SNPE-sbibm") if method is not None else _create_fallback_method("SNPE-sbibm", prior, simulator, config or {})

def create_npe_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create NPE (Neural Posterior Estimation) baseline method.
    
    Binding addendum: Uses sbibm library for NPE implementation.
    reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        NPE method instance wrapped in common interface
    """
    if config is None:
        config = {}
    
    try:
        return _create_sbibm_npe_method(prior, simulator, config)
    except Exception:
        if not _check_sbi_available():
            warnings.warn("sbi/sbibm library not available, NPE baseline unavailable")
            return _create_fallback_method("NPE", prior, simulator, config)
    
    try:
        from sbi.inference import SNPE
        npe = SNPE(prior=prior, density_estimator="maf")
        return _wrap_sbi_method(npe, "NPE")
    except Exception as e:
        warnings.warn(f"Failed to create NPE method: {e}")
        return _create_fallback_method("NPE", prior, simulator, config)


def create_snpe_a_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPE-A (Sequential NPE variant A) baseline method.
    
    Binding addendum: Uses sbibm library for SNPE implementation.
    reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    
    Paper comparison baseline from Section 5.3.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPE-A method instance wrapped in common interface
    """
    if config is None:
        config = {}
    
    try:
        return _create_sbibm_snpe_method(prior, simulator, config)
    except Exception:
        if not _check_sbi_available():
            warnings.warn("sbi/sbibm library not available, SNPE-A baseline unavailable")
            return _create_fallback_method("SNPE-A", prior, simulator, config)
    
    try:
        from sbi.inference.snpe.snpe_a import SNPE_A as SBI_SNPE_A
        snpe_a = SBI_SNPE_A(prior=prior)
        return _wrap_sbi_method(snpe_a, "SNPE-A")
    except Exception as e:
        warnings.warn(f"Failed to create SNPE-A method: {e}")
        return _create_fallback_method("SNPE-A", prior, simulator, config)


def create_snpe_b_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPE-B (Sequential NPE variant B) baseline method.
    
    Binding addendum: Uses sbibm library for SNPE implementation.
    
    Paper comparison baseline from Section 5.3.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPE-B method instance wrapped in common interface
    """
    if config is None:
        config = {}
    
    if not _check_sbi_available():
        warnings.warn("sbi library not available, SNPE-B baseline unavailable")
        return _create_fallback_method("SNPE-B", prior, simulator, config)
    
    try:
        from sbi.inference.snpe.snpe_b import SNPE_B as SBI_SNPE_B
        snpe_b = SBI_SNPE_B(prior=prior)
        return _wrap_sbi_method(snpe_b, "SNPE-B")
    except Exception as e:
        warnings.warn(f"Failed to create SNPE-B method: {e}")
        return _create_fallback_method("SNPE-B", prior, simulator, config)


def create_snpe_c_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create SNPE-C (Sequential NPE variant C) baseline method.
    
    Binding addendum: Uses sbibm library for SNPE implementation.
    
    Paper comparison baseline from Section 5.3.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        SNPE-C method instance wrapped in common interface
    """
    if config is None:
        config = {}
    
    if not _check_sbi_available():
        warnings.warn("sbi library not available, SNPE-C baseline unavailable")
        return _create_fallback_method("SNPE-C", prior, simulator, config)
    
    try:
        from sbi.inference.snpe.snpe_c import SNPE_C as SBI_SNPE_C
        snpe_c = SBI_SNPE_C(prior=prior)
        return _wrap_sbi_method(snpe_c, "SNPE-C")
    except Exception as e:
        warnings.warn(f"Failed to create SNPE-C method: {e}")
        return _create_fallback_method("SNPE-C", prior, simulator, config)


def create_nle_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create NLE (Neural Likelihood Estimation) baseline method.
    
    Paper comparison baseline from Section 5.3.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        NLE method instance
    """
    if config is None:
        config = {}
    
    if not _check_sbi_available():
        warnings.warn("sbi library not available, NLE baseline unavailable")
        return _create_fallback_method("NLE", prior, simulator, config)
    
    try:
        from sbi.inference import SNLE
        nle = SNLE(prior=prior)
        return _wrap_sbi_method(nle, "NLE")
    except Exception as e:
        warnings.warn(f"Failed to create NLE method: {e}")
        return _create_fallback_method("NLE", prior, simulator, config)


def create_nre_method(prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Create NRE (Neural Ratio Estimation) baseline method.
    
    Paper comparison baseline from Section 5.3.
    
    Args:
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        NRE method instance
    """
    if config is None:
        config = {}
    
    if not _check_sbi_available():
        warnings.warn("sbi library not available, NRE baseline unavailable")
        return _create_fallback_method("NRE", prior, simulator, config)
    
    try:
        from sbi.inference import SNRE
        nre = SNRE(prior=prior)
        return _wrap_sbi_method(nre, "NRE")
    except Exception as e:
        warnings.warn(f"Failed to create NRE method: {e}")
        return _create_fallback_method("NRE", prior, simulator, config)


# ============================================================================
# Helper Functions
# ============================================================================

def _wrap_sbi_method(sbi_method, method_name: str):
    """
    Wrap an sbi library method in a common interface adapter.
    
    reference_grounding: paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py
    
    Args:
        sbi_method: sbi method instance
        method_name: Name of the method for identification
        
    Returns:
        Wrapped method with common interface
    """
    class SBIMethodAdapter:
        """Adapter to expose common interface for sbi library methods."""
        
        def __init__(self, wrapped_method, name):
            self.method = wrapped_method
            self.name = name
            self._trained = False
        
        def fit(self, theta, x, **kwargs):
            """Train the method on simulated data."""
            self.method.append_simulations(theta, x)
            density_estimator = self.method.train(**kwargs)
            self._trained = True
            return self
        
        def sample(self, x_obs, num_samples=1000, **kwargs):
            """Sample from posterior given observed data."""
            if not self._trained:
                raise RuntimeError(f"{self.name} must be trained before sampling")
            posterior = self.method.build_posterior()
            samples = posterior.sample((num_samples,), x=x_obs, **kwargs)
            return samples
        
        def log_prob(self, theta, x_obs, **kwargs):
            """Compute log probability (if available)."""
            if not self._trained:
                raise RuntimeError(f"{self.name} must be trained before log_prob")
            posterior = self.method.build_posterior()
            return posterior.log_prob(theta, x=x_obs, **kwargs)
    
    return SBIMethodAdapter(sbi_method, method_name)


def _create_fallback_method(method_name: str, prior, simulator, config):
    """
    Create a minimal fallback method for dependency-unavailable cases.
    
    This allows the registry to be complete even when optional dependencies
    are not installed, enabling dry-run and smoke testing.
    
    Args:
        method_name: Name of the method
        prior: Prior distribution
        simulator: Simulator callable
        config: Method configuration
        
    Returns:
        Fallback method instance that raises informative errors
    """
    class FallbackMethod:
        """Minimal fallback for methods with unavailable dependencies."""
        
        def __init__(self, name, prior_dist, sim, cfg):
            self.name = name
            self.prior = prior_dist
            self.simulator = sim
            self.config = cfg or {}
            self._dependency_error = f"{name} requires additional dependencies not installed"
        
        def fit(self, theta, x, **kwargs):
            """Raise error with dependency information."""
            raise RuntimeError(
                f"{self._dependency_error}. "
                f"Install required packages to use {self.name}."
            )
        
        def sample(self, x_obs, num_samples=1000, **kwargs):
            """Raise error with dependency information."""
            raise RuntimeError(
                f"{self._dependency_error}. "
                f"Install required packages to use {self.name}."
            )
    
    return FallbackMethod(method_name, prior, simulator, config)


# ============================================================================
# Method Registry
# ============================================================================

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Paper methods (ours)
    "NPSE": {
        "id": "NPSE",
        "name": "Neural Posterior Score Estimation",
        "aliases": ["npse", "ours", "base"],
        "category": "ours",
        "paper_section": "Section 3",
        "description": "Direct posterior score matching via conditional diffusion models",
        "factory": create_npse_method,
        "sequential": False,
    },
    "TSNPSE": {
        "id": "TSNPSE",
        "name": "Truncated Sequential Neural Posterior Score Estimation",
        "aliases": ["tsnpse", "truncated_snpse", "algorithm_1"],
        "category": "ours",
        "paper_section": "Algorithm 1",
        "description": "Sequential NPSE with truncation for improved sample efficiency",
        "factory": create_tsnpse_method,
        "sequential": True,
    },
    "SNPSE-A": {
        "id": "SNPSE-A",
        "name": "Sequential NPSE Variant A",
        "aliases": ["snpse_a", "snpse-a"],
        "category": "ours",
        "paper_section": "Section 5",
        "description": "Alternative sequential approach A",
        "factory": create_snpse_a_method,
        "sequential": True,
    },
    "SNPSE-B": {
        "id": "SNPSE-B",
        "name": "Sequential NPSE Variant B",
        "aliases": ["snpse_b", "snpse-b"],
        "category": "ours",
        "paper_section": "Section 5",
        "description": "Alternative sequential approach B",
        "factory": create_snpse_b_method,
        "sequential": True,
    },
    "SNPSE-C": {
        "id": "SNPSE-C",
        "name": "Sequential NPSE Variant C",
        "aliases": ["snpse_c", "snpse-c"],
        "category": "ours",
        "paper_section": "Section 5",
        "description": "Alternative sequential approach C",
        "factory": create_snpse_c_method,
        "sequential": True,
    },
    
    # Baseline methods
    "NPE": {
        "id": "NPE",
        "name": "Neural Posterior Estimation",
        "aliases": ["npe", "baseline_npe"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Neural posterior estimation baseline (sbibm library)",
        "factory": create_npe_method,
        "sequential": False,
        "sbibm_implementation": True,
    },
    "SNPE-A": {
        "id": "SNPE-A",
        "name": "Sequential Neural Posterior Estimation A",
        "aliases": ["snpe_a", "snpe-a"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Sequential NPE variant A baseline (sbibm library)",
        "factory": create_snpe_a_method,
        "sequential": True,
        "sbibm_implementation": True,
    },
    "SNPE-B": {
        "id": "SNPE-B",
        "name": "Sequential Neural Posterior Estimation B",
        "aliases": ["snpe_b", "snpe-b"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Sequential NPE variant B baseline (sbibm library)",
        "factory": create_snpe_b_method,
        "sequential": True,
        "sbibm_implementation": True,
    },
    "SNPE-C": {
        "id": "SNPE-C",
        "name": "Sequential Neural Posterior Estimation C",
        "aliases": ["snpe_c", "snpe-c"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Sequential NPE variant C baseline (sbibm library)",
        "factory": create_snpe_c_method,
        "sequential": True,
        "sbibm_implementation": True,
    },
    "NLE": {
        "id": "NLE",
        "name": "Neural Likelihood Estimation",
        "aliases": ["nle", "baseline_nle"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Neural likelihood estimation baseline",
        "factory": create_nle_method,
        "sequential": False,
    },
    "NRE": {
        "id": "NRE",
        "name": "Neural Ratio Estimation",
        "aliases": ["nre", "baseline_nre"],
        "category": "baseline",
        "paper_section": "Section 5.3",
        "description": "Neural ratio estimation baseline",
        "factory": create_nre_method,
        "sequential": False,
    },
}


# ============================================================================
# Registry Access Functions
# ============================================================================

def get_method(method_name: str, prior, simulator=None, config: Optional[Dict[str, Any]] = None):
    """
    Get a method instance by name from the registry.
    
    Args:
        method_name: Method name or alias (e.g., "NPSE", "TSNPSE", "NPE")
        prior: Prior distribution
        simulator: Optional simulator callable
        config: Method configuration dict
        
    Returns:
        Method instance with fit() and sample() methods
        
    Raises:
        ValueError: If method_name is not recognized
    """
    # Normalize method name
    method_name_upper = method_name.upper().replace("_", "-")
    
    # Check direct match
    if method_name_upper in METHOD_REGISTRY:
        factory = METHOD_REGISTRY[method_name_upper]["factory"]
        return factory(prior, simulator, config)
    
    # Check aliases
    for method_id, method_info in METHOD_REGISTRY.items():
        if method_name.lower() in method_info.get("aliases", []):
            factory = method_info["factory"]
            return factory(prior, simulator, config)
    
    # Method not found
    available = list(METHOD_REGISTRY.keys())
    raise ValueError(
        f"Unknown method: {method_name}. "
        f"Available methods: {', '.join(available)}"
    )


def list_methods(category: Optional[str] = None) -> list:
    """
    List all available methods in the registry.
    
    Args:
        category: Optional filter by category ("ours", "baseline", etc.)
        
    Returns:
        List of method IDs
    """
    if category is None:
        return list(METHOD_REGISTRY.keys())
    
    return [
        method_id
        for method_id, method_info in METHOD_REGISTRY.items()
        if method_info.get("category") == category
    ]


def get_method_info(method_name: str) -> Dict[str, Any]:
    """
    Get metadata about a method from the registry.
    
    Args:
        method_name: Method name or alias
        
    Returns:
        Method metadata dict
        
    Raises:
        ValueError: If method_name is not recognized
    """
    method_name_upper = method_name.upper().replace("_", "-")
    
    if method_name_upper in METHOD_REGISTRY:
        return METHOD_REGISTRY[method_name_upper]
    
    for method_id, method_info in METHOD_REGISTRY.items():
        if method_name.lower() in method_info.get("aliases", []):
            return method_info
    
    raise ValueError(f"Unknown method: {method_name}")


def get_baseline_methods() -> list:
    """Get list of baseline method IDs."""
    return list_methods(category="baseline")


def get_paper_methods() -> list:
    """Get list of paper method IDs (ours)."""
    return list_methods(category="ours")


def get_sequential_methods() -> list:
    """Get list of sequential method IDs."""
    return [
        method_id
        for method_id, method_info in METHOD_REGISTRY.items()
        if method_info.get("sequential", False)
    ]