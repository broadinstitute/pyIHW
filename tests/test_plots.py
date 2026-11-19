import numpy as np
import pandas as pd
import pytest

from pyihw import (
    ihw,
    plot_decision_boundary,
    plot_pvalue_densities,
    plot_pvalue_histogram,
    plot_pvalue_histograms,
    plot_stratum_summary,
    qc_report,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture
def result():
    rng = np.random.default_rng(0)
    covariate = rng.uniform(0, 4, 3000)
    pvalues = np.where(rng.random(3000) < 0.1, rng.uniform(0, 1e-3, 3000), rng.random(3000))
    pvalues[:5] = np.nan
    return ihw(pvalues, covariate, alpha=0.1, nbins=6)


@pytest.mark.parametrize("bins", [1, 20])
def test_qc_report_draws_every_panel(result, bins):
    fig = qc_report(result, bins=bins)

    titles = [ax.get_title() for ax in fig.axes]
    assert titles[:6] == [
        "All p-values",
        "p-value density by stratum",
        "Signal by stratum",
        "Learned weights",
        "Decision boundary",
        "Rejections vs level",
    ]


@pytest.mark.parametrize(
    "plot", [plot_pvalue_histogram, plot_pvalue_densities, plot_stratum_summary]
)
def test_standalone_qc_plots_draw_on_supplied_axis(result, plot):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    try:
        assert plot(result, ax=ax) is ax
        assert ax.has_data()
    finally:
        plt.close(fig)


def test_qc_report_skips_decision_boundary_for_nominal_covariates():
    rng = np.random.default_rng(1)
    covariate = pd.Categorical(rng.choice(list("abc"), 900))
    result = ihw(rng.random(900), covariate, alpha=0.1, covariate_type="nominal")

    fig = qc_report(result)

    assert "Decision boundary" not in [ax.get_title() for ax in fig.axes]


def test_pvalue_histograms_has_one_panel_per_stratum(result):
    fig = plot_pvalue_histograms(result, ncols=4)

    assert sum(ax.get_visible() for ax in fig.axes) == result.nbins


def test_decision_boundary_defaults_to_covariate_values(result):
    ax = plot_decision_boundary(result)

    assert ax.get_xlabel() == "Covariate"


def test_result_decision_boundary_defaults_to_covariate_values(result):
    ax = result.plot("decisionboundary")

    assert ax.get_xlabel() == "Covariate"
