"""Compatibility package for data modules."""

from __future__ import annotations

from importlib import import_module as _import_module

_MODULES = ['data', 'dataset_registry', 'environments', 'environment_registry']

for _name in _MODULES:
    try:
        _module = _import_module(f"{__name__}.{_name}")
    except Exception:
        continue
    for _key, _value in vars(_module).items():
        if not _key.startswith("_"):
            globals().setdefault(_key, _value)

del _import_module
