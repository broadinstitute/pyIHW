"""The fitted IHW result and its accessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .multiple_testing import get_bh_threshold


@dataclass(eq=False)
class IHWResult:
    """Fitted IHW weights and multiple-testing results.

    Per-hypothesis arrays follow the input order, including hypotheses with a
    missing p-value, which get missing weights and adjusted p-values.
    """

    _frame: pd.DataFrame
    _weight_matrix: np.ndarray
    alpha: float
    nbins: int
    nfolds: int
    regularization_term: np.ndarray
    m_groups: np.ndarray
    penalty: str
    covariate_type: str
    adjustment_type: str

    def adj_pvalues(self) -> np.ndarray:
        return self._frame["adj_pvalue"].to_numpy(copy=True)

    def weights(self, levels_only: bool = False) -> np.ndarray:
        """Per-hypothesis weights, or the ``(nbins, nfolds)`` matrix with ``levels_only``."""
        if levels_only:
            return self._weight_matrix.copy()
        return self._frame["weight"].to_numpy(copy=True)

    def thresholds(self, levels_only: bool = False) -> np.ndarray:
        """Raw p-value cutoffs that reproduce the fitted tests' rejection flags.

        Cutoffs account for floating-point rounding at observed p-values.
        """
        mtests = int(np.sum(self.m_groups))
        if self.adjustment_type == "bonferroni":
            threshold = self.alpha / mtests
        else:
            threshold = get_bh_threshold(self.weighted_pvalues(), self.alpha, mtests)
        level_thresholds = threshold * self._weight_matrix
        groups = self._frame["group_code"].to_numpy(dtype=float)
        folds = self._frame["fold_code"].to_numpy(dtype=float)
        coded = ~np.isnan(groups) & ~np.isnan(folds)
        indices = groups[coded].astype(int), folds[coded].astype(int)
        pvalues = self.pvalues()[coded]
        rejected = self.adj_pvalues()[coded] <= self.alpha
        cutoffs = level_thresholds[indices]
        # Inverting the adjustment can round differently from computing it.
        too_low = rejected & (pvalues > cutoffs)
        too_high = ~rejected & (pvalues <= cutoffs)
        np.maximum.at(
            level_thresholds, (indices[0][too_low], indices[1][too_low]), pvalues[too_low],
        )
        np.minimum.at(
            level_thresholds, (indices[0][too_high], indices[1][too_high]),
            np.nextafter(pvalues[too_high], 0),
        )
        if levels_only:
            return level_thresholds
        result = np.full(len(self._frame), np.nan)
        result[coded] = level_thresholds[indices]
        return result

    def pvalues(self) -> np.ndarray:
        return self._frame["pvalue"].to_numpy(copy=True)

    def weighted_pvalues(self) -> np.ndarray:
        return self._frame["weighted_pvalue"].to_numpy(copy=True)

    def covariates(self) -> np.ndarray:
        return self._frame["covariate"].to_numpy(copy=True)

    def groups_factor(self) -> np.ndarray:
        return self._frame["group"].to_numpy(copy=True)

    def stratification_breaks(self) -> np.ndarray:
        """Maximum ordinal covariate value in each stratum.

        Nominal covariates have no ordered breakpoints.
        """
        if self.covariate_type != "ordinal":
            raise ValueError("stratification breaks require an ordinal covariate")

        covariates = pd.to_numeric(self._frame["covariate"], errors="coerce").to_numpy(
            dtype=float
        )
        groups = self._frame["group_code"].to_numpy(dtype=float)
        breaks = np.full(self.nbins, np.nan)
        for group in range(self.nbins):
            in_group = (groups == group) & np.isfinite(covariates)
            if in_group.any():
                breaks[group] = np.max(covariates[in_group])
        return breaks

    def stratum_sizes(self) -> np.ndarray:
        """Number of observed, non-missing p-values in each stratum."""
        groups = self._frame["group_code"].to_numpy(dtype=float)
        valid_groups = groups[np.isfinite(groups)].astype(int)
        return np.bincount(valid_groups, minlength=self.nbins)[: self.nbins]

    def rejected_hypotheses(self) -> pd.arrays.BooleanArray:
        adjusted = self._frame["adj_pvalue"].to_numpy()
        rejected = pd.array(adjusted <= self.alpha, dtype="boolean")
        rejected[np.isnan(adjusted)] = pd.NA
        return rejected

    def rejections(self) -> int:
        return int(np.count_nonzero(self._frame["adj_pvalue"].to_numpy() <= self.alpha))

    def to_dataframe(self) -> pd.DataFrame:
        """One row per hypothesis with p-value, adjusted p-value, weight, stratum and fold."""
        return self._frame.drop(columns=["group_code", "fold_code"]).copy()

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        control = "FDR" if self.adjustment_type == "BH" else "FWER"
        return (
            f"IHWResult({len(self)} tests, {self.rejections()} rejections, "
            f"{control} level={self.alpha}, {self.nbins} bins, "
            f"{self.nfolds} folds)"
        )

    def plot(
        self,
        what: str = "weights",
        x_axis: str | None = None,
        scale: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Draw one diagnostic plot; see :mod:`pyihw.plots` (requires Matplotlib).

        ``what`` is one of ``"weights"``, ``"decisionboundary"``,
        ``"pvalues"``, ``"rejections"`` or ``"qc"``.
        """
        from . import plots

        if what == "weights":
            return plots.plot_weights(self, scale=scale, ax=ax)
        if what == "decisionboundary":
            if (scale or self.covariate_type) == "nominal":
                raise NotImplementedError(
                    "decision-boundary plots require an ordinal covariate"
                )
            return plots.plot_decision_boundary(
                self, x_axis=x_axis or "covariate", ax=ax
            )
        if what == "pvalues":
            return plots.plot_pvalue_histograms(self)
        if what == "rejections":
            return plots.plot_rejections(self, ax=ax)
        if what == "qc":
            return plots.qc_report(self)
        raise ValueError(
            "what must be 'weights', 'decisionboundary', 'pvalues', 'rejections' or 'qc'"
        )
