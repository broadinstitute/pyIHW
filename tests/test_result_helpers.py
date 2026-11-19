import numpy as np
import pandas as pd
import pytest

from pyihw import ihw


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
