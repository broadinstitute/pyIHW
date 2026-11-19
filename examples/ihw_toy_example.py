# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
#     "ihw-py",
# ]
#
# [tool.uv.sources]
# ihw-py = { path = "..", editable = true }
# ///
# ruff: noqa: PLR1711, B018  (marimo cells end in bare returns and display expressions)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys

    import marimo as mo

    # In the browser (Pyodide), install the ihw-py wheel served next to this page.
    if sys.platform == "emscripten":
        import micropip
        from pyodide.http import pyfetch

        base = str(mo.notebook_location()).rstrip("/")
        wheel = (await (await pyfetch(f"{base}/wheel.txt")).string()).strip()
        await micropip.install(f"{base}/{wheel}")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import false_discovery_control, norm

    import pyihw

    return false_discovery_control, mo, norm, np, pd, plt, pyihw


@app.cell
def _(mo):
    mo.md(r"""
    # Independent Hypothesis Weighting: a toy example

    When we run thousands of tests, the **Benjamini–Hochberg (BH)** procedure
    controls the false discovery rate (FDR) by treating every hypothesis the
    same. Often, though, we know something about each test *before* looking at
    its p-value, such as the number of observations used in the test. IHW
    uses that extra information to prioritize tests that are more likely to
    yield discoveries.

    **IHW** uses such a covariate to give each test a weight $w_i$, with
    weights averaging 1, and runs BH on the weighted p-values $p_i / w_i$.
    Tests in informative strata get $w_i > 1$ and are easier to reject. Tests
    in uninformative strata get $w_i < 1$.

    **The one requirement:** under the null hypothesis, the covariate must be
    *independent of the p-value*. The covariate may predict *whether* a test is
    a true signal, but must not change the distribution of null p-values.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Simulate tests with a known truth

    Each test has a covariate $x \in [0, 1]$. A test is a true
    signal with probability $0.4\,x^3$, so signals are more common at high $x$.
    Signals get a z-score shift equal to the effect size. Null p-values are
    uniform whatever $x$ is, so the covariate is valid.
    """)
    return


@app.cell
def _(mo):
    n_tests = mo.ui.slider(2_000, 40_000, step=2_000, value=20_000, label="Number of tests")
    signal_strength = mo.ui.slider(1.0, 4.0, step=0.25, value=2.5, label="Effect size (z)")
    seed = mo.ui.number(0, 10_000, value=1, label="Seed")
    mo.hstack([n_tests, signal_strength, seed])
    return n_tests, seed, signal_strength


@app.cell
def _(n_tests, norm, np, pd, seed, signal_strength):
    rng = np.random.default_rng(seed.value)
    _n = n_tests.value
    covariate = rng.uniform(0, 1, _n)
    is_signal = rng.random(_n) < 0.4 * covariate**3
    z = rng.normal(np.where(is_signal, signal_strength.value, 0.0))
    pvalues = norm.sf(z)
    tests = pd.DataFrame({"covariate": covariate, "pvalue": pvalues, "is_signal": is_signal})
    tests.head()
    return covariate, is_signal, pvalues, tests


@app.cell
def _(mo, np, plt, tests):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _null = ~tests["is_signal"]
    _axes[0].hist(
        [tests.loc[_null, "pvalue"], tests.loc[~_null, "pvalue"]],
        bins=40, stacked=True, label=["null", "signal"], color=["0.7", "C3"],
    )
    _axes[0].set(xlabel="p-value", ylabel="Tests", title="All p-values")
    _axes[0].legend()
    _axes[1].scatter(
        tests["covariate"], -np.log10(tests["pvalue"].clip(lower=1e-300)),
        c=tests["is_signal"].map({True: "C3", False: "0.7"}), s=3, rasterized=True,
    )
    _axes[1].set(xlabel="Covariate", ylabel="-log10 p", title="Signals concentrate at high covariate")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(
            "The left panel shows the usual mixture: a flat null with a spike near 0. "
            "The right panel shows that the spike comes mostly from high-covariate tests. "
            "BH cannot use this information, but IHW can."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Fit IHW

    `pyihw.ihw` does four things:

    1. **Stratify** the tests into `nbins` equal-sized bins of the covariate.
    2. **Split** them into `nfolds` folds. The weights for one fold are learned
       *only* from the other folds (cross-fitting), so a test's weight never
       depends on its own p-value.
    3. In each bin, estimate the p-value distribution with the **Grenander
       estimator**, a concave CDF fit. Then solve a **linear program** for
       the per-bin thresholds that maximize expected discoveries at level
       $\alpha$. A regularization term $\lambda$, chosen by internal
       cross-validation, keeps neighboring bins' weights similar.
    4. Turn the thresholds into weights and apply **weighted BH**.
    """)
    return


@app.cell
def _(mo):
    alpha = mo.ui.slider(0.01, 0.3, step=0.01, value=0.1, label="FDR level α")
    nbins = mo.ui.slider(1, 20, value=10, label="Covariate bins")
    mo.hstack([alpha, nbins])
    return alpha, nbins


@app.cell
def _(alpha, covariate, nbins, pvalues, pyihw):
    result = pyihw.ihw(pvalues, covariate, alpha=alpha.value, nbins=nbins.value, seed=1)
    result
    return (result,)


@app.cell
def _(alpha, false_discovery_control, is_signal, mo, np, pd, pvalues, result):
    def _summary(rejected):
        n_rejected = int(rejected.sum())
        false = int((rejected & ~is_signal).sum())
        return {
            "rejections": n_rejected,
            "false discoveries": false,
            "FDP": false / max(n_rejected, 1),
            "power (true signals found)": (rejected & is_signal).sum() / is_signal.sum(),
        }

    bh_rejected = false_discovery_control(pvalues, method="bh") <= alpha.value
    ihw_rejected = np.asarray(result.rejected_hypotheses(), dtype=bool)
    comparison = pd.DataFrame({"BH": _summary(bh_rejected), "IHW": _summary(ihw_rejected)}).T
    mo.vstack([
        mo.md(f"### BH vs IHW at α = {alpha.value:g}"),
        mo.ui.table(comparison.round(3), selection=None),
        mo.md(
            "Because this is a simulation, we know which tests are true signals. "
            "FDP is the fraction of this run’s rejections that are false. "
            "FDR is the expected FDP across runs, so a single run can exceed α."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Explore the quality-control plots

    Choose a panel to see how to read it and what to try. The display controls
    below do not refit IHW; the simulation and fit controls above do.
    To see all six together in your own analysis, call `pyihw.qc_report(result)`.
    """)
    return


@app.cell
def _(mo):
    qc_panel = mo.ui.dropdown(
        ["All p-values", "Density by stratum", "Signal by stratum",
         "Learned weights", "Decision boundary", "Rejections vs level"],
        value="All p-values",
        label="QC panel",
    )
    qc_panel
    return (qc_panel,)


@app.cell(hide_code=True)
def _(mo):
    hist_bins = mo.ui.slider(10, 50, step=10, value=20, label="Histogram bins", show_value=True)
    boundary_axis = mo.ui.radio(
        {"Covariate values": "covariate", "Covariate ranks": "ranks"},
        value="Covariate values",
        label="Horizontal axis",
        inline=True,
    )
    preview_level = mo.ui.slider(
        0.01, 0.30, step=0.01, value=0.10, label="Preview level (fixed weights)", show_value=True,
    )
    return boundary_axis, hist_bins, preview_level


@app.cell(hide_code=True)
def _(boundary_axis, hist_bins, preview_level, qc_panel):
    {
        "All p-values": hist_bins,
        "Density by stratum": hist_bins,
        "Decision boundary": boundary_axis,
        "Rejections vs level": preview_level,
    }.get(qc_panel.value)
    return


@app.cell(hide_code=True)
def _(
    boundary_axis,
    false_discovery_control,
    hist_bins,
    mo,
    np,
    plt,
    preview_level,
    pvalues,
    pyihw,
    qc_panel,
    result,
):
    from pyihw import (
        plot_pvalue_densities,
        plot_pvalue_histogram,
        plot_stratum_summary,
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _panel = qc_panel.value

    if _panel == "All p-values":
        plot_pvalue_histogram(result, bins=hist_bins.value, ax=_ax)
        _guide = r"""
        **Read it:** The bars show the density of p-values pooled across tests.
        A spike near zero can reflect real signal. The dashed line is Storey's
        estimate of the uniform-null contribution, $\hat\pi_0$.

        **Try it:** Change the histogram resolution. Do bumps persist with wider
        bins? A flat pooled tail can hide differences between strata; check the
        next panel.
        """
    elif _panel == "Density by stratum":
        plot_pvalue_densities(result, bins=hist_bins.value, ax=_ax)
        _guide = """
        **Read it:** Each line is a covariate stratum. Compare tail shapes, not
        heights: strata with different proportions of true nulls can have flat
        tails at different heights. Systematic slopes or bumps are worth checking,
        but flat tails do not prove the covariate is valid. The vertical scale can
        clip the spike near zero to make the tails easier to compare.

        **Try it:** Increase *Covariate bins* or decrease *Number of tests* above.
        With fewer observations per stratum, the lines tend to get noisier.
        """
    elif _panel == "Signal by stratum":
        plot_stratum_summary(result, ax=_ax)
        _guide = r"""
        **Read it:** Gray bars estimate the fraction of true nulls ($\hat\pi_0$);
        red points show the fraction of tests rejected, not their false discovery
        proportion. In this simulation, higher strata tend to have fewer true
        nulls and more rejections.

        **Try it:** Lower the effect size. The simulated signal labels stay the
        same, but weaker signals can be harder to separate from nulls. The bars
        are estimates, not the known truth.
        """
    elif _panel == "Learned weights":
        pyihw.plot_weights(result, ax=_ax)
        _guide = """
        **Read it:** Each line shows the weights applied to one held-out fold.
        A weight above 1 reduces the weighted p-value; below 1 it increases it.
        Large differences between fold lines can indicate unstable estimates.

        **Try it:** Reduce *Covariate bins* above and compare how closely the fold
        lines agree. With one stratum, all weights are 1 and IHW reduces to BH.
        """
    elif _panel == "Decision boundary":
        pyihw.plot_decision_boundary(result, x_axis=boundary_axis.value, ax=_ax)
        _guide = r"""
        **Read it:** Each dot is a test; smaller p-values sit higher on the
        $-\log_{10}(p)$ scale. Red dots are rejections. Compare a test only to its
        own fold's threshold. A lower line lets that fold reject larger raw
        p-values in that stratum.

        **Try it:** Toggle values and ranks; the same tests are rejected. Then
        change *FDR level* above to refit IHW and see the thresholds move.
        """
    else:
        _level = preview_level.value
        _grid = np.linspace(0.005, max(0.30, 2 * result.alpha), 60)
        _levels = np.unique(np.r_[_grid, _level, result.alpha])
        pyihw.plot_rejections(result, alphas=_levels, ax=_ax)
        _ihw_count = int(np.sum(result.adj_pvalues() <= _level))
        _bh_count = int(np.sum(false_discovery_control(pvalues, method="bh") <= _level))
        _ax.scatter([_level, _level], [_ihw_count, _bh_count], color=["C3", "0.4"], zorder=3)
        _guide = f"""
        **Read it:** Each curve counts rejections as the cutoff on adjusted
        p-values increases. The dotted line marks where IHW learned its weights
        ({result.alpha:g}). Away from that line, the curve reuses those weights;
        it does not show what a refit would discover.

        **Try it:** Move the preview slider. At {_level:g}, there are
        **{_ihw_count:,} IHW** and **{_bh_count:,} BH** rejections. Use *FDR level*
        above to refit rather than preview.
        """

    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.md(_guide), _fig])
    return


@app.cell
def _(mo, np, pd, result):
    strata = pd.DataFrame({
        "covariate up to": result.stratification_breaks(),
        "tests": result.stratum_sizes(),
        "mean weight": np.nanmean(result.weights(levels_only=True), axis=1),
    })
    strata.index.name = "stratum"
    mo.vstack([
        mo.md(
            "The strata behind these plots: `result.stratification_breaks()` is the "
            "largest covariate value in each stratum, `result.stratum_sizes()` the "
            "number of tests, and `result.weights(levels_only=True)` the "
            "(stratum × fold) weight matrix, averaged over folds here."
        ),
        mo.ui.table(strata.round(3), selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. What if the covariate carries no information?

    Refit with an independent random covariate while keeping the same p-values.
    There should be no systematic pattern in the weights and no reliable gain
    in discoveries over BH. A single simulation can still show differences.
    """)
    return


@app.cell
def _(alpha, false_discovery_control, nbins, np, plt, pvalues, pyihw):
    _noise = np.random.default_rng(123).uniform(size=len(pvalues))
    noise_result = pyihw.ihw(pvalues, _noise, alpha=alpha.value, nbins=nbins.value, seed=1)
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    pyihw.plot_weights(noise_result, ax=_ax)
    _ax.set_title(
        f"Uninformative covariate: IHW {noise_result.rejections()} vs "
        f"BH {(false_discovery_control(pvalues, method='bh') <= alpha.value).sum()} rejections"
    )
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
