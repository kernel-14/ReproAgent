"""Method-agent adapters for SNPSE repair validation.

reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
"""
from typing import Any, Dict
from src.methods.methods import NPSE, TSNPSE, SNPSE_A, SNPSE_B, SNPSE_C

METHOD_AGENT_REGISTRY = {
    "NPSE": NPSE,
    "TSNPSE": TSNPSE,
    "SNPSE-A": SNPSE_A,
    "SNPSE-B": SNPSE_B,
    "SNPSE-C": SNPSE_C,
}

def make_agent(method: str, **kwargs: Dict[str, Any]) -> Any:
    cls = METHOD_AGENT_REGISTRY.get(method, TSNPSE)
    try:
        return cls(**kwargs)
    except TypeError:
        return {"method": method, "adapter": cls.__name__, "kwargs": kwargs}
