"""Convenience front ends for formula strings and DESeq2-style result tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .fitting import ihw
from .result import IHWResult


def ihw_formula(formula: str, data: pd.DataFrame, **kwargs: Any) -> IHWResult:
    """Formula convenience interface, for example ``ihw_formula('pvalue ~ baseMean', df)``."""
    terms = formula.split("~")
    if len(terms) != 2 or not all(term.strip() for term in terms):
        raise ValueError("formula must have the form 'pvalue ~ covariate'")
    p_name, cov_name = (term.strip() for term in terms)
    if p_name not in data or cov_name not in data:
        raise KeyError("formula columns must exist in data")
    return ihw(data[p_name], data[cov_name], **kwargs)


def ihw_deseq_results(
    results: pd.DataFrame,
    filter: str | Sequence[float] = "baseMean",
    alpha: float = 0.1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return a DESeq2-like results frame with IHW ``padj`` and ``weight`` columns."""
    if "pvalue" not in results:
        raise KeyError("results must contain a 'pvalue' column")
    covariate = results[filter] if isinstance(filter, str) else filter
    fitted = ihw(results["pvalue"], covariate, alpha=alpha, **kwargs)
    output = results.copy()
    output["padj"] = fitted.adj_pvalues()
    output["weight"] = fitted.weights()
    output.attrs["ihw_result"] = fitted
    output.attrs["alpha"] = alpha
    return output
