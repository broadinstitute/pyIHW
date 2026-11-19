"""Grenander estimate of a stratum's p-value distribution.

The Grenander estimator is the least concave majorant (LCM) of the empirical
CDF. Being concave and piecewise linear, it is the minimum of its supporting
lines, which is the form the weight optimization consumes.
"""

from __future__ import annotations

import numpy as np


def least_concave_majorant(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the least concave majorant of points sorted by ``x``."""
    hull: list[int] = []
    for index in range(len(x)):
        hull.append(index)
        while len(hull) >= 3:
            a, b, c = hull[-3:]
            slope_ab = (y[b] - y[a]) / (x[b] - x[a])
            slope_bc = (y[c] - y[b]) / (x[c] - x[b])
            if slope_ab >= slope_bc:
                break
            hull.pop(-2)
    return x[hull], y[hull]


def grenander_lines(
    pvalues: np.ndarray, total: float
) -> tuple[np.ndarray, np.ndarray]:
    """Slopes and intercepts of the LCM of the ECDF of ``pvalues``.

    The ECDF is scaled by ``total`` rather than ``len(pvalues)`` so that a
    stratum's untested hypotheses (``m_groups``) count towards it.
    """
    if len(pvalues):
        x, counts = np.unique(pvalues, return_counts=True)
        y = np.cumsum(counts) / total
        if x[0] > 0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 0.0)
        if x[-1] < 1:
            x = np.append(x, 1.0)
            y = np.append(y, len(pvalues) / total)
    else:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
    hx, hy = least_concave_majorant(x, y)
    slopes = np.diff(hy) / np.diff(hx)
    intercepts = hy[:-1] - slopes * hx[:-1]
    return slopes, intercepts
