"""Cross-fitting: fold assignment and choosing the regularization strength."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .multiple_testing import adjust_pvalues, weighted_pvalues
from .optimization import fit_weights


def assign_folds(count: int, nfolds: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform fold draws, as R's ``sample(1:nfolds, m, replace=TRUE)``."""
    if nfolds == 1:
        return np.zeros(count, dtype=int)
    return rng.integers(0, nfolds, size=count)


def holdout_counts(
    population: np.ndarray,
    available_counts: np.ndarray,
    train_counts: np.ndarray,
    nfolds: int,
) -> np.ndarray:
    """Stratum sizes of a held-out fold, spreading untested hypotheses evenly."""
    missing_counts = population - available_counts
    if np.any(missing_counts < -1e-8):
        raise ValueError("m_groups cannot be smaller than the observed stratum counts")
    return np.maximum(missing_counts, 0) / nfolds + available_counts - train_counts


def default_lambdas(n_bins: int) -> np.ndarray:
    return np.asarray(
        [0.0, 1.0, n_bins / 8, n_bins / 4, n_bins / 2, float(n_bins), np.inf],
        dtype=float,
    )


def resolve_lambdas(lambdas: Sequence[float] | str, n_bins: int) -> np.ndarray:
    if isinstance(lambdas, str):
        if lambdas != "auto":
            raise ValueError("lambdas must be 'auto' or a sequence of numbers")
        return default_lambdas(n_bins)
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or not len(grid):
        raise ValueError("lambdas must be a non-empty one-dimensional sequence")
    if np.any(np.isnan(grid)) or np.any(grid < 0):
        raise ValueError("lambdas must be non-negative and cannot contain NaN")
    return grid


def select_lambda(
    pvalues: np.ndarray,
    groups: np.ndarray,
    population: np.ndarray,
    lambdas: np.ndarray,
    penalty: str,
    adjustment_type: str,
    alpha: float,
    nfolds: int,
    nsplits: int,
    rng: np.random.Generator,
) -> float:
    """Pick the regularization with the most rejections in nested cross-fitting."""
    if len(lambdas) == 1:
        return float(lambdas[0])
    totals = np.zeros(len(lambdas), dtype=float)
    for _ in range(nsplits):
        totals += cross_validated_rejections(
            pvalues, groups, population, lambdas, penalty, adjustment_type, alpha,
            min(nfolds, len(pvalues)), rng,
        )
    return float(lambdas[np.argmax(totals)])


def cross_validated_rejections(
    pvalues: np.ndarray,
    groups: np.ndarray,
    population: np.ndarray,
    lambdas: np.ndarray,
    penalty: str,
    adjustment_type: str,
    alpha: float,
    nfolds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Number of rejections each candidate regularization achieves when cross-fitted."""
    fold_ids = assign_folds(len(pvalues), nfolds, rng)
    available = np.bincount(groups, minlength=len(population)).astype(float)
    mtests = int(np.sum(population))
    scores = np.zeros(len(lambdas), dtype=float)
    for lambda_idx, regularization in enumerate(lambdas):
        candidate_weights = np.ones(len(pvalues), dtype=float)
        for fold in np.unique(fold_ids):
            held = fold_ids == fold
            training = ~held if nfolds > 1 else np.ones(len(pvalues), dtype=bool)
            if nfolds > 1:
                train_counts = np.bincount(
                    groups[training], minlength=len(population)
                ).astype(float)
                held_counts = holdout_counts(population, available, train_counts, nfolds)
                training_counts = population - held_counts
            else:
                held_counts = training_counts = population
            weights = fit_weights(
                pvalues[training],
                groups[training],
                len(population),
                held_counts,
                training_counts,
                penalty,
                float(regularization),
                adjustment_type,
                alpha,
            )
            candidate_weights[held] = weights[groups[held]]
        adjusted = adjust_pvalues(
            weighted_pvalues(pvalues, candidate_weights), adjustment_type, mtests
        )
        scores[lambda_idx] = np.count_nonzero(adjusted <= alpha)
    return scores
