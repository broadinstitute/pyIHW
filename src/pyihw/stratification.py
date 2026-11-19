"""Turning a covariate into the strata IHW learns one weight for."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAX_AUTO_BINS = 40
TESTS_PER_AUTO_BIN = 1500


@dataclass(frozen=True)
class Strata:
    """Zero-based stratum code per hypothesis, the stratum labels and the penalty."""

    codes: np.ndarray
    labels: np.ndarray
    penalty: str

    @property
    def count(self) -> int:
        return len(self.labels)


def groups_by_filter(
    covariate: Sequence[float], nbins: int, seed: int | None = None
) -> np.ndarray:
    """Assign approximately equal-sized ordinal covariate bins (1-based).

    Ties are broken at random, so only the ordering of the covariate matters.
    """
    values = np.asarray(covariate, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("covariate must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("covariates must be finite")
    if int(nbins) != nbins or nbins < 1:
        raise ValueError("nbins must be a positive integer")
    nbins = int(nbins)
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(len(values))
    order = np.lexsort((tie_breaker, values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return np.minimum(nbins, (ranks * nbins + len(values) - 1) // len(values))


def auto_nbins(n_tests: int) -> int:
    return max(1, min(MAX_AUTO_BINS, n_tests // TESTS_PER_AUTO_BIN))


def stratify(
    covariate: np.ndarray,
    original: Any,
    covariate_type: str,
    nbins: int | str,
    seed: int | None,
    has_m_groups: bool,
) -> Strata:
    """Stratify the covariates of the tested hypotheses.

    ``original`` is the user's covariate object, consulted only for the
    category order of a pandas ``Categorical``.
    """
    is_numeric = np.issubdtype(covariate.dtype, np.number)
    if covariate_type == "ordinal" and is_numeric:
        return _numeric_strata(covariate, nbins, seed, has_m_groups)
    if is_numeric:
        raise ValueError("nominal covariates must be categorical")
    return _categorical_strata(covariate, original, covariate_type, nbins)


def _numeric_strata(
    covariate: np.ndarray, nbins: int | str, seed: int | None, has_m_groups: bool
) -> Strata:
    if not np.all(np.isfinite(covariate.astype(float))):
        raise ValueError("covariates for non-missing p-values must be finite")
    if has_m_groups:
        raise ValueError("m_groups is only supported for categorical covariates")
    if nbins == "auto":
        n_bins = auto_nbins(len(covariate))
    else:
        if int(nbins) != nbins or nbins < 1:
            raise ValueError("nbins must be a positive integer or 'auto'")
        n_bins = int(nbins)
    codes = groups_by_filter(covariate.astype(float), n_bins, seed=seed) - 1
    return Strata(codes, np.arange(1, n_bins + 1), "total variation")


def _categorical_strata(
    covariate: np.ndarray, original: Any, covariate_type: str, nbins: int | str
) -> Strata:
    if isinstance(original, pd.Series) and isinstance(original.dtype, pd.CategoricalDtype):
        categories = [c for c in original.cat.categories if np.any(covariate == c)]
    else:
        categories = list(pd.unique(pd.Series(covariate)))
        try:
            categories = sorted(categories)
        except TypeError:
            pass
    if not categories:
        raise ValueError("at least one covariate stratum is required")
    if nbins != "auto" and int(nbins) != len(categories):
        warnings.warn("nbins is ignored for categorical covariates", UserWarning)
    lookup = {label: idx for idx, label in enumerate(categories)}
    codes = np.asarray([lookup[value] for value in covariate], dtype=int)
    penalty = "uniform deviation" if covariate_type == "nominal" else "total variation"
    return Strata(codes, np.asarray(categories, dtype=object), penalty)
