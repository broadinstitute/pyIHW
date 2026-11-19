import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyihw import ihw

_R_RESULTS = json.loads(
    (Path(__file__).parent / "fixtures" / "ihw_r_results.json").read_text()
)


def _example_data():
    n_per_group = 40
    groups = np.repeat(np.arange(1, 5), n_per_group)
    covariates = groups + np.tile(
        np.arange(1, n_per_group + 1) / (n_per_group + 1) / 10, 4
    )
    signal_counts = [2, 8, 16, 24]
    pvalues = np.concatenate(
        [
            np.concatenate(
                [
                    np.linspace(0.0001, 0.02, signal_counts[group - 1]),
                    np.linspace(0.2, 0.99, n_per_group - signal_counts[group - 1]),
                ]
            )
            for group in range(1, 5)
        ]
    )
    return pvalues, covariates


@pytest.mark.parametrize(
    ("reference", "covariate_type", "regularization"),
    [
        ("ordinal_auto", "ordinal", "auto"),
        ("ordinal_regularized", "ordinal", [0.5]),
        ("nominal_regularized", "nominal", [0.5]),
    ],
)
def test_results_match_bioconductor_ihw(
    reference, covariate_type, regularization
):
    pvalues, covariates = _example_data()
    folds = np.tile(np.arange(1, 6), len(pvalues) // 5)
    if covariate_type == "nominal":
        covariates = pd.Categorical(
            np.repeat(list("abcd"), 40), categories=list("abcd")
        )

    result = ihw(
        pvalues,
        covariates,
        alpha=0.1,
        covariate_type=covariate_type,
        nbins=4,
        folds=folds,
        nfolds_internal=1,
        lambdas=regularization,
        seed=1,
    )
    expected = _R_RESULTS[reference]

    np.testing.assert_allclose(
        result.adj_pvalues(), expected["adj_pvalues"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.weighted_pvalues(), expected["weighted_pvalues"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.weights(), expected["weights"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.weights(levels_only=True),
        expected["weight_matrix"],
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.thresholds(), expected["thresholds"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.thresholds(levels_only=True),
        expected["threshold_matrix"],
        rtol=1e-6,
        atol=1e-8,
    )
    frame = result.to_dataframe()
    np.testing.assert_array_equal(frame["group"].astype(str), expected["groups"])
    np.testing.assert_array_equal(frame["fold"].astype(int), expected["folds"])
    np.testing.assert_array_equal(
        result.rejected_hypotheses(),
        np.asarray(expected["adj_pvalues"]) <= result.alpha,
    )
    np.testing.assert_array_equal(result.regularization_term, expected["lambdas"])
    assert result.rejections() == expected["rejections"]


def test_rejected_hypotheses_preserves_missing_pvalues():
    result = ihw([0.001, np.nan, 0.8], [1.0, 2.0, 3.0], alpha=0.1, nbins=1)

    rejected = result.rejected_hypotheses()
    assert bool(rejected[0])
    assert pd.isna(rejected[1])
    assert not bool(rejected[2])
    assert result.rejections() == 1


def test_single_fold_learns_weights_from_all_hypotheses_like_r():
    pvalues, covariates = _example_data()
    with pytest.warns(UserWarning, match="one fold"):
        result = ihw(pvalues, covariates, alpha=0.1, nbins=4, nfolds=1, lambdas=[0.5])
    expected = _R_RESULTS["ordinal_one_fold"]

    np.testing.assert_allclose(result.weights(), expected["weights"], rtol=1e-6)
    np.testing.assert_allclose(
        result.adj_pvalues(), expected["adj_pvalues"], rtol=1e-6, atol=1e-8
    )
    assert result.rejections() == expected["rejections"]


def test_null_proportion_with_random_folds():
    pvalues, covariates = _example_data()
    result = ihw(pvalues, covariates, alpha=0.1, nbins=4, null_proportion=True)

    assert np.all(np.isfinite(result.weights()))


def test_tiny_pvalues_do_not_break_the_linear_program():
    pvalues, covariates = _example_data()
    pvalues[[120, 121]] = [1e-30, 1e-25]
    folds = np.tile(np.arange(1, 6), len(pvalues) // 5)
    result = ihw(pvalues, covariates, alpha=0.1, nbins=4, folds=folds, lambdas=[0.5])
    expected = _R_RESULTS["tiny_pvalues"]

    np.testing.assert_allclose(result.weights(), expected["weights"], rtol=1e-6)
    np.testing.assert_allclose(
        result.adj_pvalues(), expected["adj_pvalues"], rtol=1e-6, atol=1e-30
    )
    assert result.rejections() == expected["rejections"]
