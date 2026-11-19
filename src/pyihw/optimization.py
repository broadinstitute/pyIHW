"""The linear program that turns Grenander estimates into stratum weights.

Variables are laid out as ``[F_1..F_G, t_1..t_G, aux...]``: ``F_g`` is the
estimated rejection probability of stratum ``g`` at threshold ``t_g``, and
the auxiliary variables carry the regularization (Supplementary Note of
Ignatiadis et al., 2016).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .grenander import grenander_lines

PENALTIES = ("total variation", "uniform deviation")
# R uses 1e-20; HiGHS refuses coefficients above 1e15 and Grenander slopes are <= 1 / TINY_PVALUE.
TINY_PVALUE = 1e-14


class _InequalityRows:
    """Sparse ``A_ub x <= b_ub`` assembled one row at a time."""

    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.rhs: list[float] = []

    def add(self, cols, vals, rhs: float = 0.0) -> None:
        cols = np.atleast_1d(cols)
        self.rows.extend([len(self.rhs)] * len(cols))
        self.cols.extend(cols.tolist())
        self.vals.extend(np.broadcast_to(vals, cols.shape).tolist())
        self.rhs.append(float(rhs))

    def build(self, nvars: int):
        shape = (len(self.rhs), nvars)
        matrix = coo_matrix((self.vals, (self.rows, self.cols)), shape=shape).tocsr()
        return matrix, np.asarray(self.rhs)


def fit_weights(
    pvalues: np.ndarray,
    groups: np.ndarray,
    group_count: int,
    held_counts: np.ndarray,
    training_counts: np.ndarray,
    penalty: str,
    regularization: float,
    adjustment_type: str,
    alpha: float,
) -> np.ndarray:
    """Learn one weight per stratum from training p-values.

    Weights are normalized so that they average to one over the held-out
    hypotheses (``held_counts``) they will be applied to.
    """
    if regularization == 0:
        return np.ones(group_count)
    if penalty not in PENALTIES:
        raise ValueError("penalty must be 'total variation' or 'uniform deviation'")
    total_held = float(np.sum(held_counts))
    if total_held <= 0 or float(np.sum(training_counts)) <= 0:
        raise ValueError("the training and held-out populations must be non-empty")

    F = np.arange(group_count)
    t = group_count + F
    constraints = _InequalityRows()
    # Tiny p-values are zeroed for the weight fit only; adjustment uses the originals.
    pvalues = np.where(pvalues > TINY_PVALUE, pvalues, 0.0)

    # F_g <= intercept + slope * t_g for every supporting line of the Grenander CDF.
    for group in range(group_count):
        slopes, intercepts = grenander_lines(
            pvalues[groups == group], training_counts[group]
        )
        for slope, intercept in zip(slopes, intercepts):
            constraints.add([F[group], t[group]], [1.0, -float(slope)], intercept)

    nvars = 2 * group_count
    if np.isfinite(regularization):
        if penalty == "total variation":
            nvars = _add_total_variation(
                constraints, t, nvars, regularization * held_counts / total_held
            )
        else:
            nvars = _add_uniform_deviation(
                constraints, t, nvars, held_counts, regularization
            )

    # Error-rate budget: BH controls sum(m_g t_g) <= alpha * sum(m_g F_g);
    # Bonferroni controls sum(m_g t_g) <= alpha.
    if adjustment_type == "BH":
        constraints.add(
            np.concatenate([F, t]),
            np.concatenate([-alpha * held_counts, held_counts]),
        )
    else:
        constraints.add(t, held_counts, alpha)

    objective = np.zeros(nvars)
    objective[F] = -held_counts / total_held * group_count
    matrix, rhs = constraints.build(nvars)
    solution = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=(0, None), method="highs")
    if not solution.success:
        raise RuntimeError(f"IHW weight optimization failed: {solution.message}")

    thresholds = np.maximum(solution.x[t], 0.0)
    denominator = float(np.dot(held_counts, thresholds))
    if denominator <= 0:
        return np.ones(group_count)
    return thresholds * total_held / denominator


def _add_total_variation(
    constraints: _InequalityRows, t: np.ndarray, start: int, budget: np.ndarray
) -> int:
    """``sum |t_{g+1} - t_g| <= sum(budget_g * t_g)`` for ordered strata."""
    diffs = start + np.arange(len(t) - 1)
    for idx, diff in enumerate(diffs):
        cols = [t[idx], t[idx + 1], diff]
        constraints.add(cols, [1.0, -1.0, -1.0])
        constraints.add(cols, [-1.0, 1.0, -1.0])
    constraints.add(np.concatenate([diffs, t]), np.concatenate([np.ones(len(diffs)), -budget]))
    return start + len(diffs)


def _add_uniform_deviation(
    constraints: _InequalityRows,
    t: np.ndarray,
    start: int,
    held_counts: np.ndarray,
    regularization: float,
) -> int:
    """``sum_g |m t_g - sum_h m_h t_h| <= lambda * sum(m_g t_g)`` for unordered strata."""
    total_held = float(held_counts.sum())
    weighted_sum = start
    deviations = start + 1 + np.arange(len(t))
    # Define the weighted sum once instead of repeating it in each deviation.
    cols = np.append(t, weighted_sum)
    coeffs = np.append(held_counts, -1.0)
    constraints.add(cols, coeffs)
    constraints.add(cols, -coeffs)
    for threshold, deviation in zip(t, deviations):
        constraints.add([threshold, weighted_sum, deviation], [total_held, -1.0, -1.0])
        constraints.add([threshold, weighted_sum, deviation], [-total_held, 1.0, -1.0])
    constraints.add(
        np.append(deviations, weighted_sum),
        np.append(np.ones(len(deviations)), -regularization),
    )
    return start + 1 + len(deviations)
