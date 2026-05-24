"""Grouped source package with legacy module aliases."""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_MODULE_ALIASES = {'artifact_contract': 'core.artifact_contract', 'data': 'data.data', 'dataset_registry': 'data.dataset_registry', 'environments': 'data.environments', 'environment_registry': 'data.environment_registry', 'baselines': 'methods.baselines', 'explainers': 'methods.explainers', 'method_registry': 'methods.method_registry', 'methods': 'methods.methods', 'models': 'methods.models', 'refinement': 'methods.refinement', 'evaluation': 'experiments.evaluation', 'experiments': 'experiments.experiments', 'training': 'experiments.training', 'artifacts': 'reporting.artifacts', 'plotting': 'reporting.plotting', 'trend_assertions': 'reporting.trend_assertions'}

for _legacy_name, _target_name in _MODULE_ALIASES.items():
    try:
        _module = _importlib.import_module(f"{__name__}.{_target_name}")
    except Exception:
        continue
    _sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)
    globals().setdefault(_legacy_name, _module)


_GROUP_EXPORTS = {
    "core": "core",
    "data": "data",
    "methods": "methods",
    "experiments": "experiments",
    "reporting": "reporting",
}

for _legacy_name, _target_name in _GROUP_EXPORTS.items():
    try:
        _module = _importlib.import_module(f"{__name__}.{_target_name}")
    except Exception:
        continue
    _sys.modules.setdefault(f"{__name__}.{_legacy_name}", _module)
    globals().setdefault(_legacy_name, _module)

del _importlib, _sys
