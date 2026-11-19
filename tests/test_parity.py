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


@pytest.mark.parametrize("null_proportion", [False, True])
def test_small_fit_has_no_empty_random_folds(null_proportion):
    result = ihw(
        [0.001, 0.7], [1, 2], alpha=0.1, nbins=2, seed=1,
        null_proportion=null_proportion,
    )

    assert result.nfolds == 2
    assert result.to_dataframe()["fold"].nunique() == 2
    assert np.isfinite(result.adj_pvalues()).all()


@pytest.mark.parametrize("counts", [[1.8, 2.8], [[1, -1], [1, 1]], [[1, 0.5], [0, 1]]])
def test_population_counts_must_be_nonnegative_integers(counts):
    with pytest.raises(ValueError, match="non-negative integers"):
        ihw([0.03, 0.04], ["a", "b"], alpha=0.1, folds=[1, 2], m_groups=counts)


def test_fold_count_columns_follow_sorted_labels_including_untested_folds():
    pvalues = np.array([0.04, 0.001, 0.9, np.nan])
    covariates = np.array(["a", "b", "b", "a"])
    folds = np.array([20, 20, 10, 30])
    counts = [[0, 1, 3], [1, 1, 0]]
    order = np.array([2, 0, 3, 1])

    result = ihw(pvalues, covariates, 0.1, folds=folds, m_groups=counts, lambdas=[0.5])
    reordered = ihw(
        pvalues[order], covariates[order], 0.1, folds=folds[order],
        m_groups=counts, lambdas=[0.5],
    )
    np.testing.assert_array_equal(result.m_groups, [4, 2])
    assert result.to_dataframe()["fold"].iloc[:3].tolist() == [2, 2, 1]
    np.testing.assert_allclose(result.adj_pvalues()[order], reordered.adj_pvalues())


def test_counted_fold_with_no_observed_tests_is_not_used_for_training():
    with pytest.warns(UserWarning, match="No observed training p-values"):
        result = ihw(
            [0.03, 0.04, np.nan], ["a", "b", "a"], 0.1,
            folds=[1, 1, 2], m_groups=[[1, 2], [1, 0]],
            adjustment_type="bonferroni",
        )

    np.testing.assert_allclose(result.weights(), [1, 1, np.nan])
    np.testing.assert_allclose(result.adj_pvalues(), [0.12, 0.16, np.nan])


@pytest.mark.parametrize("as_series", [False, True])
def test_nominal_numeric_categories_preserve_declared_order(as_series):
    covariates = pd.Categorical([1, 2, 1, 2], categories=[2, 1], ordered=True)
    if as_series:
        covariates = pd.Series(covariates)
    result = ihw(
        [0.01, 0.04, 0.5, 0.9], covariates, 0.1,
        covariate_type="nominal", folds=[1, 1, 2, 2], m_groups=[3, 2], lambdas=[0],
    )

    np.testing.assert_allclose(result.adj_pvalues(), [0.05, 0.1, 5 / 6, 1])


@pytest.mark.parametrize("as_series", [False, True])
def test_ordered_categorical_keeps_non_alphabetical_order(as_series):
    covariates = pd.Categorical(
        ["high", "low", "mid"], categories=["low", "mid", "high"], ordered=True,
    )
    if as_series:
        covariates = pd.Series(covariates)
    result = ihw(
        [0.01, 0.04, 0.9], covariates, 0.1, folds=[1, 2, 1],
        m_groups=[1, 1, 2], lambdas=[0],
    )

    np.testing.assert_allclose(result.adj_pvalues(), [0.04, 0.08, 1])


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
