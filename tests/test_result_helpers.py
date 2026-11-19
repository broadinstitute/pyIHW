import numpy as np
import pandas as pd
import pytest

from pyihw import ihw, ihw_deseq_results


@pytest.mark.parametrize(("method", "cutoff"), [("BH", 0.04), ("bonferroni", 0.1 / 3)])
def test_thresholds_agree_with_rejection_flags(method, cutoff):
    result = ihw([0.03, 0.04, 0.9], [1, 1, 1], 0.1, adjustment_type=method)

    np.testing.assert_allclose(result.thresholds(levels_only=True), [[cutoff]])
    np.testing.assert_array_equal(
        result.pvalues() <= result.thresholds(), result.rejected_hypotheses()
    )


def test_independently_fitted_adapter_outputs_can_be_concatenated():
    data = pd.DataFrame({"pvalue": [0.03, 0.5], "baseMean": [2, 3]})

    combined = pd.concat([ihw_deseq_results(data), ihw_deseq_results(data)])

    np.testing.assert_allclose(combined["padj"], [0.06, 0.5, 0.06, 0.5])


@pytest.mark.parametrize(
    ("method", "alpha", "ntests", "pvalue"),
    [
        pytest.param("BH", 0.03, 7, 0.03 / 7, id="bh-lower"),
        pytest.param("bonferroni", 0.03, 7, 0.03 / 7, id="bonferroni-lower"),
        pytest.param(
            "bonferroni", 0.001, 3, np.nextafter(0.001 / 3, 1), id="bonferroni-raise"
        ),
    ],
)
def test_cutoffs_match_adjusted_values_at_rounding_boundary(method, alpha, ntests, pvalue):
    result = ihw(
        [pvalue] + [0.9] * (ntests - 1),
        [1] * ntests,
        alpha,
        adjustment_type=method,
    )

    flags = result.rejected_hypotheses()
    np.testing.assert_array_equal(flags, result.adj_pvalues() <= alpha)
    np.testing.assert_array_equal(result.pvalues() <= result.thresholds(), flags)
    np.testing.assert_array_equal(
        result.pvalues() <= result.thresholds(levels_only=True)[0, 0], flags
    )


def test_stratification_breaks_and_sizes_exclude_missing_pvalues():
    result = ihw(
        [0.01, 0.4, 0.6, 0.8, np.nan],
        [0.1, 0.3, 0.6, 0.9, 0.5],
        alpha=0.1,
        nbins=2,
        folds=[1, 2, 1, 2, 1],
        lambdas=[0.5],
    )

    np.testing.assert_allclose(result.stratification_breaks(), [0.3, 0.9])
    np.testing.assert_array_equal(result.stratum_sizes(), [2, 2])


def test_stratification_breaks_require_ordinal_covariate():
    result = ihw(
        [0.01, 0.4, 0.8],
        pd.Categorical(["a", "b", "c"]),
        alpha=0.1,
        covariate_type="nominal",
        nbins=3,
    )

    with pytest.raises(ValueError, match="ordinal covariate"):
        result.stratification_breaks()
