"""IHW for pysceptre discovery results, with analytical power as the covariate.

pyihw does not import pysceptre: these functions only need the data frames
returned by ``pysceptre.run_discovery_analysis`` / ``run_calibration_check``
and ``pysceptre.compute_power``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .fitting import ihw

PAIR_COLUMNS = ("response_id", "grna_target")


def ihw_with_power(
    results: pd.DataFrame,
    power_results: pd.DataFrame,
    alpha: float,
    *,
    pair_cols: Sequence[str] = PAIR_COLUMNS,
    pvalue_col: str = "p_value",
    power_col: str = "power",
    **ihw_kwargs: Any,
) -> pd.DataFrame:
    """Apply IHW to pysceptre results using matched analytical power as covariate.

    Rows of ``results`` are kept in order; ``power_results`` is matched to
    them on ``pair_cols``. Adds ``ihw_weight``, ``ihw_weighted_p_value``,
    ``ihw_adj_p_value`` and ``ihw_rejected``; the fitted
    :class:`~pyihw.IHWResult` is stored in ``attrs["ihw_result"]``.
    Untested pairs (NaN p-value) are kept with missing IHW columns.

    The usual IHW independence assumption still applies to the power estimates.
    """
    keys = list(pair_cols)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("pair_cols must contain distinct key column names")
    for frame, name, columns in (
        (results, "results", [*keys, pvalue_col]),
        (power_results, "power_results", [*keys, power_col]),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(f"{name} is missing column(s): {missing}")

    if power_results.duplicated(keys).any():
        raise ValueError("power_results must have at most one row per pair")

    pvalues = pd.to_numeric(results[pvalue_col], errors="raise").to_numpy(dtype=float)
    tested = ~np.isnan(pvalues)
    if results.loc[tested, keys].isna().any().any():
        raise ValueError("tested results must have non-missing pair identifiers")

    aligned_power = results[keys].merge(
        power_results.loc[:, [*keys, power_col]],
        on=keys,
        how="left",
        sort=False,
        validate="many_to_one",
    )[power_col]
    covariate = pd.to_numeric(aligned_power, errors="coerce").to_numpy(dtype=float)
    invalid_power = tested & (~np.isfinite(covariate) | (covariate < 0) | (covariate > 1))
    if invalid_power.any():
        missing_pairs = results.loc[invalid_power, keys].drop_duplicates().head(5)
        raise ValueError(
            "each tested pair needs a finite power estimate in [0, 1]; "
            f"invalid or missing pairs include {missing_pairs.to_dict('records')}"
        )

    fitted = ihw(pvalues, covariate, alpha, covariate_type="ordinal", **ihw_kwargs)
    output = results.copy()
    output["ihw_weight"] = fitted.weights()
    output["ihw_weighted_p_value"] = fitted.weighted_pvalues()
    output["ihw_adj_p_value"] = fitted.adj_pvalues()
    rejected = pd.array(fitted.rejected_hypotheses(), dtype="boolean")
    rejected[~tested] = pd.NA
    output["ihw_rejected"] = rejected
    output.attrs["ihw_result"] = fitted
    output.attrs["ihw_alpha"] = alpha
    return output


def calibration_cells_per_grna(
    calibration: pd.DataFrame,
    ntc_grna_cells: Mapping[str, np.ndarray],
    separator: str = "&",
) -> pd.DataFrame:
    """``cells_per_grna`` for the synthetic targets of a pysceptre calibration check.

    Calibration targets are ``separator``-joined non-targeting gRNA ids, so
    ``compute_power`` needs each target's member gRNAs and their cell counts
    to score the negative-control pairs.
    """
    if "grna_target" not in calibration.columns:
        raise KeyError("calibration is missing column 'grna_target'")
    rows = []
    for target in pd.unique(calibration["grna_target"]):
        for grna in str(target).split(separator):
            if grna not in ntc_grna_cells:
                raise KeyError(f"gRNA {grna!r} of target {target!r} is not in ntc_grna_cells")
            rows.append((grna, target, int(np.asarray(ntc_grna_cells[grna]).size)))
    return pd.DataFrame(rows, columns=["grna_id", "grna_target", "num_cells"])
