"""Reporting wrappers for the PINN reproduction package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src import artifact_contract


PAPER_ROUTE_FUNCTIONS = artifact_contract.runtime_routes()


def write_dry_run_artifacts(
    output_root: Path | str = artifact_contract.DEFAULT_OUTPUT_ROOT,
    mode: str = "runtime_smoke",
    records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    return artifact_contract.write_dry_run_artifacts(output_root=output_root, mode=mode, records=records)


def write_report(output_root: Path | str = artifact_contract.DEFAULT_OUTPUT_ROOT, mode: str = "runtime_smoke", records: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    return write_dry_run_artifacts(output_root=output_root, mode=mode, records=records)


def route_path(route_name: str) -> str:
    return artifact_contract.artifact_specs()[route_name].path


def runtime_route(route_name: str, output_root: Path | str = artifact_contract.DEFAULT_OUTPUT_ROOT, mode: str = "runtime_smoke", records: Optional[Sequence[Mapping[str, Any]]] = None) -> Path:
    route = PAPER_ROUTE_FUNCTIONS[route_name]
    return route(output_root=output_root, mode=mode, records=records)


def plot_component_spectra(output_root: Path | str = artifact_contract.DEFAULT_OUTPUT_ROOT, mode: str = "runtime_smoke", records: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, str]:
    artifact_contract.write_figure_3(output_root=output_root, records=records, mode=mode)
    artifact_contract.write_figure_7(output_root=output_root, records=records, mode=mode)
    return {
        "figure_3": route_path("figure_3"),
        "figure_7": route_path("figure_7"),
    }


def report_summary(records: Optional[Sequence[Mapping[str, Any]]] = None, mode: str = "runtime_smoke") -> Dict[str, Any]:
    return {
        "analysis": artifact_contract.analysis(records),
        "artifact_writer": artifact_contract.artifact_writer(mode=mode),
    }


__all__ = [
    "PAPER_ROUTE_FUNCTIONS",
    "write_dry_run_artifacts",
    "write_report",
    "route_path",
    "runtime_route",
    "plot_component_spectra",
    "report_summary",
]
