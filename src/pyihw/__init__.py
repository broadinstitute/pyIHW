"""Independent hypothesis weighting.

Modules:
    fitting           ``ihw``, the main entry point
    result            ``IHWResult`` and its accessors
    plots             QC plots (requires Matplotlib)
    sceptre           IHW on pysceptre results with analytical power
    adapters          formula and DESeq2-style front ends
    stratification    covariate binning
    grenander         Grenander estimate of each stratum's p-value CDF
    optimization      the linear program that learns weights
    crossfit          folds and regularization selection
    multiple_testing  BH / Bonferroni adjustment
"""

from .adapters import ihw_deseq_results, ihw_formula
from .fitting import ihw
from .multiple_testing import get_bh_threshold
from .plots import (
    plot_decision_boundary,
    plot_pvalue_densities,
    plot_pvalue_histogram,
    plot_pvalue_histograms,
    plot_rejections,
    plot_stratum_summary,
    plot_weights,
    qc_report,
)
from .result import IHWResult
from .sceptre import calibration_cells_per_grna, ihw_with_power
from .stratification import groups_by_filter

__all__ = [
    "IHWResult",
    "calibration_cells_per_grna",
    "get_bh_threshold",
    "groups_by_filter",
    "ihw",
    "ihw_deseq_results",
    "ihw_formula",
    "ihw_with_power",
    "plot_decision_boundary",
    "plot_pvalue_densities",
    "plot_pvalue_histogram",
    "plot_pvalue_histograms",
    "plot_rejections",
    "plot_stratum_summary",
    "plot_weights",
    "qc_report",
]
