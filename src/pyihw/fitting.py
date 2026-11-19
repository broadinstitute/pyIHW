"""``ihw``: learn weights by cross-fitting, then apply weighted BH/Bonferroni."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .crossfit import assign_folds, holdout_counts, resolve_lambdas, select_lambda
from .multiple_testing import adjust_pvalues, validate_alpha, weighted_pvalues
from .optimization import fit_weights
from .result import IHWResult
from .stratification import stratify


def ihw(
    pvalues: Sequence[float],
    covariates: Sequence[Any],
    alpha: float,
    *,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    m_groups: Sequence[float] | np.ndarray | None = None,
    folds: Sequence[Any] | None = None,
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas: Sequence[float] | str = "auto",
    seed: int | None = 1,
    adjustment_type: str = "BH",
    null_proportion: bool = False,
    null_proportion_level: float = 0.5,
    quiet: bool = True,
) -> IHWResult:
    """Fit independent hypothesis weighting and return weighted adjusted p-values.

    Args:
        pvalues: one p-value per hypothesis; NaN marks an untested hypothesis.
        covariates: a covariate per hypothesis, independent of the p-value
            under the null. Numeric covariates are binned into ``nbins``
            equal-sized strata; categorical ones define the strata directly.
        alpha: target FDR (``"BH"``) or FWER (``"bonferroni"``) level.
        covariate_type: ``"ordinal"`` (neighboring strata are smoothed towards
            each other) or ``"nominal"`` (strata are shrunk towards uniform).
        nbins: number of strata for numeric covariates; ``"auto"`` uses about
            1,500 hypotheses per stratum, at most 40.
        m_groups: total hypotheses per stratum, including untested ones, for
            categorical covariates. Two-dimensional ``(nbins, nfolds)`` counts
            require explicit ``folds``.
        folds: explicit fold labels; otherwise ``nfolds`` random folds.
        nfolds_internal: folds of the nested cross-fitting that picks the
            regularization strength from ``lambdas``.
        nsplits_internal: how many times that nested split is repeated.
        seed: seeds stratum tie-breaking and fold assignment.
        null_proportion: rescale weights by a weighted Storey estimate of the
            null proportion at ``null_proportion_level``.
        quiet: if False, print the regularization chosen in each fold.
    """
    alpha = validate_alpha(alpha)
    p_all, cov_all = _check_inputs(pvalues, covariates)
    _check_options(covariate_type, adjustment_type, nfolds, nfolds_internal, nsplits_internal)

    valid = ~np.isnan(p_all)
    if not valid.any():
        raise ValueError("at least one non-missing p-value is required")
    if pd.isna(cov_all[valid]).any():
        raise ValueError("covariates for non-missing p-values cannot be missing")
    if folds is not None and len(folds) != len(p_all):
        raise ValueError("folds must match the p-value vector")
    p = p_all[valid]
    rng = np.random.default_rng(seed)

    strata = stratify(
        cov_all[valid], covariates, covariate_type, nbins, seed, m_groups is not None
    )
    groups, n_bins = strata.codes, strata.count
    available = np.bincount(groups, minlength=n_bins).astype(float)
    m_groups_arr = _check_m_groups(m_groups, available, n_bins, folds)
    if n_bins == 1:
        nfolds = 1

    fold_values = None if folds is None else np.asarray(folds)[valid]
    fold_ids, n_folds, population, population_by_fold = _resolve_folds(
        fold_values, groups, n_bins, available, m_groups_arr, int(nfolds), rng
    )
    if np.any(population <= 0):
        raise ValueError("each stratum must represent a positive number of tests")
    lambda_grid = resolve_lambdas(lambdas, n_bins)

    if n_bins == 1:
        fold_ids = np.zeros(len(p), dtype=int)
        n_folds = 1
        weight_matrix = np.ones((1, 1))
        fold_lambdas = np.zeros(1)
        population_by_fold = population[:, None]
    else:
        weight_matrix, fold_lambdas = _fit_fold_weights(
            p,
            groups,
            fold_ids,
            n_bins,
            n_folds,
            population,
            available,
            population_by_fold if m_groups_arr.ndim == 2 else None,
            lambda_grid,
            strata.penalty,
            adjustment_type,
            alpha,
            int(nfolds_internal),
            int(nsplits_internal),
            rng,
        )

    weights = weight_matrix[groups, fold_ids]
    if null_proportion:
        _rescale_by_null_proportion(
            p, weights, weight_matrix, fold_ids, n_folds, population,
            population_by_fold, null_proportion_level,
        )

    weighted_p = weighted_pvalues(p, weights)
    adjusted = adjust_pvalues(weighted_p, adjustment_type, int(np.sum(population)))
    frame = _result_frame(
        p_all, cov_all, valid, adjusted, weights, weighted_p,
        strata.labels[groups], groups, fold_ids,
    )
    if not quiet and n_bins > 1:
        print(f"Selected regularization per fold: {fold_lambdas.tolist()}")
    return IHWResult(
        _frame=frame,
        _weight_matrix=weight_matrix,
        alpha=alpha,
        nbins=n_bins,
        nfolds=n_folds,
        regularization_term=fold_lambdas,
        m_groups=np.asarray(population, dtype=int),
        penalty=strata.penalty,
        covariate_type=covariate_type,
        adjustment_type=adjustment_type,
    )


def _check_inputs(pvalues, covariates) -> tuple[np.ndarray, np.ndarray]:
    p_all = np.asarray(pvalues, dtype=float)
    cov_all = np.asarray(covariates)
    if p_all.ndim != 1 or cov_all.ndim != 1 or len(p_all) != len(cov_all):
        raise ValueError("pvalues and covariates must be one-dimensional and equally sized")
    if not len(p_all):
        raise ValueError("at least one hypothesis is required")
    observed = p_all[~np.isnan(p_all)]
    if np.any(np.isinf(p_all)) or np.any((observed < 0) | (observed > 1)):
        raise ValueError("non-missing p-values must be finite and in [0, 1]")
    return p_all, cov_all


def _check_options(covariate_type, adjustment_type, nfolds, nfolds_internal, nsplits_internal):
    if covariate_type not in ("ordinal", "nominal"):
        raise ValueError("covariate_type must be 'ordinal' or 'nominal'")
    if adjustment_type not in ("BH", "bonferroni"):
        raise ValueError("adjustment_type must be 'BH' or 'bonferroni'")
    for name, value in (
        ("nfolds", nfolds),
        ("nfolds_internal", nfolds_internal),
        ("nsplits_internal", nsplits_internal),
    ):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _check_m_groups(m_groups, available: np.ndarray, n_bins: int, folds) -> np.ndarray:
    if m_groups is None:
        return available.copy()
    counts = np.asarray(m_groups, dtype=float)
    if counts.ndim == 2:
        if folds is None:
            raise ValueError("a two-dimensional m_groups requires pre-specified folds")
        if counts.shape != (n_bins, len(np.unique(folds))):
            raise ValueError("m_groups dimensions must be (nbins, number of folds)")
    elif counts.ndim != 1 or len(counts) != n_bins:
        raise ValueError("m_groups must have one count per stratum")
    if not np.all(np.isfinite(counts)):
        raise ValueError("m_groups must be finite")
    if counts.ndim == 1 and np.any(counts < available):
        raise ValueError("m_groups cannot be smaller than the observed stratum counts")
    return counts


def _resolve_folds(
    fold_values: np.ndarray | None,
    groups: np.ndarray,
    n_bins: int,
    available: np.ndarray,
    m_groups: np.ndarray,
    nfolds: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray | None]:
    """Fold ids, fold count, stratum population and, with explicit folds, its split by fold."""
    if fold_values is None:
        return assign_folds(len(groups), nfolds, rng), nfolds, m_groups, None

    labels = list(pd.unique(fold_values))
    lookup = {value: idx for idx, value in enumerate(labels)}
    fold_ids = np.asarray([lookup[value] for value in fold_values], dtype=int)
    n_folds = len(labels)
    observed = np.zeros((n_bins, n_folds), dtype=float)
    np.add.at(observed, (groups, fold_ids), 1)
    if m_groups.ndim == 2:
        if np.any(m_groups < observed):
            raise ValueError("m_groups cannot be smaller than observed group/fold counts")
        by_fold = m_groups
    else:
        by_fold = observed + (m_groups[:, None] - available[:, None]) / n_folds
    return fold_ids, n_folds, np.sum(by_fold, axis=1), by_fold


def _fit_fold_weights(
    p: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
    n_bins: int,
    n_folds: int,
    population: np.ndarray,
    available: np.ndarray,
    held_population: np.ndarray | None,
    lambda_grid: np.ndarray,
    penalty: str,
    adjustment_type: str,
    alpha: float,
    nfolds_internal: int,
    nsplits_internal: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn each fold's weights from the other folds only."""
    weight_matrix = np.ones((n_bins, n_folds))
    fold_lambdas = np.zeros(n_folds)
    for fold in range(n_folds):
        training = fold_ids != fold if n_folds > 1 else np.ones(len(p), dtype=bool)
        training_p, training_groups = p[training], groups[training]
        if n_folds == 1:
            held_counts = population.copy()
            training_counts = population.copy()
        else:
            if held_population is not None:
                held_counts = held_population[:, fold]
            else:
                observed = np.bincount(training_groups, minlength=n_bins).astype(float)
                held_counts = holdout_counts(population, available, observed, n_folds)
            training_counts = population - held_counts

        chosen = select_lambda(
            training_p, training_groups, training_counts, lambda_grid, penalty,
            adjustment_type, alpha, nfolds_internal, nsplits_internal, rng,
        )
        weight_matrix[:, fold] = fit_weights(
            training_p, training_groups, n_bins, held_counts, training_counts,
            penalty, chosen, adjustment_type, alpha,
        )
        fold_lambdas[fold] = chosen
    if n_folds == 1:
        warnings.warn(
            "Using one fold fits weights and tests them on the same data; "
            "do not use this for confirmatory inference.",
            UserWarning,
        )
    return weight_matrix, fold_lambdas


def _rescale_by_null_proportion(
    p: np.ndarray,
    weights: np.ndarray,
    weight_matrix: np.ndarray,
    fold_ids: np.ndarray,
    n_folds: int,
    population: np.ndarray,
    population_by_fold: np.ndarray | None,
    level: float,
) -> None:
    """Divide each fold's weights, in place, by its weighted Storey null proportion."""
    if not 0 < level < 1:
        raise ValueError("null_proportion_level must be strictly between 0 and 1")
    for fold in range(n_folds):
        selected = fold_ids == fold
        fold_total = (
            population_by_fold[:, fold].sum()
            if population_by_fold is not None
            else population.sum() / n_folds
        )
        fold_weights = weights[selected]
        pi0 = (
            np.max(fold_weights) + np.sum(fold_weights * (p[selected] > level))
        ) / (fold_total * (1 - level))
        pi0 = max(pi0, np.finfo(float).tiny)
        weights[selected] /= pi0
        weight_matrix[:, fold] /= pi0


def _result_frame(
    p_all, cov_all, valid, adjusted, weights, weighted_p, labels, groups, fold_ids
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pvalue": p_all,
            "adj_pvalue": np.nan,
            "weight": np.nan,
            "weighted_pvalue": np.nan,
            "group": None,
            "covariate": cov_all,
            "fold": None,
            "group_code": np.nan,
            "fold_code": np.nan,
        }
    )
    frame.loc[valid, "adj_pvalue"] = adjusted
    frame.loc[valid, "weight"] = weights
    frame.loc[valid, "weighted_pvalue"] = weighted_p
    frame.loc[valid, "group"] = labels
    frame.loc[valid, "fold"] = fold_ids + 1
    frame.loc[valid, "group_code"] = groups
    frame.loc[valid, "fold_code"] = fold_ids
    return frame
