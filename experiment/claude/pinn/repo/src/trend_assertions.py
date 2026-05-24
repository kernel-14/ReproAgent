"""Trend assertion helpers for the PINN loss-landscape reproduction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping

from src.artifact_contract import TREND_ASSERTIONS as PAPER_TREND_ASSERTIONS
from src.artifact_contract import validate_expected_trends


TREND_ASSERTIONS = PAPER_TREND_ASSERTIONS
TREND_ASSERTION_NAMES = tuple(trend.name for trend in TREND_ASSERTIONS)
TREND_OBLIGATIONS = {
    trend.name: {
        "statement": trend.statement,
        "polarity": trend.polarity,
        "artifact_ids": list(trend.artifact_ids),
        "note": trend.note,
    }
    for trend in TREND_ASSERTIONS
}


def trend_obligations() -> Dict[str, Dict[str, Any]]:
    return dict(TREND_OBLIGATIONS)


def validate_trend_obligations(trend_checks: Mapping[str, bool]) -> Dict[str, bool]:
    return validate_expected_trends(trend_checks)


def trend_payload() -> Dict[str, Any]:
    return {
        "trend_names": list(TREND_ASSERTION_NAMES),
        "trends": [asdict(trend) for trend in TREND_ASSERTIONS],
    }


__all__ = [
    "TREND_ASSERTIONS",
    "TREND_ASSERTION_NAMES",
    "TREND_OBLIGATIONS",
    "trend_obligations",
    "validate_trend_obligations",
    "trend_payload",
]
