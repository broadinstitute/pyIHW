"""Benjamini–Hochberg and Bonferroni adjustment of (weighted) p-values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    return value


def get_bh_threshold(
    pvalues: Sequence[float], alpha: float, mtests: int | None = None
) -> float:
    """Return the largest p-value rejected by the BH procedure."""
    alpha = validate_alpha(alpha)
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(p) if mtests is None else int(mtests)
    if m < len(p) or m < 0:
        raise ValueError("mtests must be at least the number of observed p-values")
    if not len(p) or m == 0:
        return 0.0
    ordered = np.sort(p)
    rejected = np.flatnonzero(ordered <= alpha * np.arange(1, len(p) + 1) / m)
    return float(ordered[rejected[-1]]) if len(rejected) else 0.0


def adjust_pvalues(pvalues: np.ndarray, method: str, mtests: int) -> np.ndarray:
    """BH or Bonferroni adjustment over ``mtests`` hypotheses; NaNs stay missing."""
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = ~np.isnan(p)
    values = p[valid]
    if not len(values):
        return result
    if method == "bonferroni":
        result[valid] = np.minimum(1.0, values * mtests)
        return result
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * mtests / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def weighted_pvalues(pvalues: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``p / w`` capped at 1; a zero weight never rejects unless ``p == 0``."""
    result = np.ones_like(pvalues, dtype=float)
    result[pvalues == 0] = 0.0
    positive = weights > 0
    result[positive] = np.minimum(1.0, pvalues[positive] / weights[positive])
    return result
