# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyihw",
# ]
#
# [tool.uv.sources]
# pyihw = { path = "..", editable = true }
# ///
# ruff: noqa: PLR1711, B018  (marimo cells end in bare returns and display expressions)

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
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
    its p-value: how deeply a gene is sequenced, how many cells carry a gRNA,
    or the analytical **power** of a CRISPR-screen pair. Tests with high power
    are more likely to be discoveries.

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

    Each test has a covariate $x \in [0, 1]$ (think: power). A test is a true
    signal with probability $0.4\,x^3$, so almost all signals sit at high $x$.
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
       depends on its own p-value. This is what preserves FDR control.
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
            "Both methods keep the **false discovery proportion** near or below α, "
            "and IHW usually finds **more true signals**."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Quality-control plots

    `pyihw.qc_report(result)` draws a six-panel summary. Read it as follows:

    * **All p-values / density by stratum**: the right half of each histogram
      should be flat. That is the null part. A slope there, especially one that
      differs between strata, suggests the covariate may not be independent of
      null p-values.
    * **Signal by stratum**: Storey's null-proportion estimate $\hat\pi_0$ per
      bin, with the fraction of tests rejected. An informative covariate makes
      them move in opposite directions.
    * **Learned weights**: one line per fold. Folds should broadly agree. Wild
      disagreement means there are too few tests per bin, so reduce `nbins`.
    * **Decision boundary**: each fold's rejection threshold on the
      $-\log_{10} p$ scale, against the covariate value (use
      `result.plot("decisionboundary", x_axis="ranks")` for ranks). Points
      above the line are rejected.
    * **Rejections vs level**: IHW versus unweighted BH across $\alpha$. The
      weights were learned at the chosen α, so the curve is exact only there.
    """)
    return


@app.cell
def _(pyihw, result):
    pyihw.qc_report(result)
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
def _(mo, pyihw, result):
    mo.vstack([
        mo.md(
            "Per-stratum p-value histograms. The dashed line is $\\hat\\pi_0$. "
            "The spike at 0 grows with the covariate."
        ),
        pyihw.plot_pvalue_histograms(result),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. What if the covariate carries no information?

    Replace the covariate with random noise. IHW should then learn weights
    close to 1 and behave like BH. It loses a little power to noise but does
    not inflate false discoveries.
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Using it on a pysceptre screen

    For a CRISPR screen, analytical power from `pysceptre.compute_power` is a
    natural covariate:

    ```python
    from pyihw import ihw_with_power, qc_report

    weighted = ihw_with_power(discovery, power, alpha=0.1, nbins=10)
    weighted["ihw_rejected"]                  # discoveries
    qc_report(weighted.attrs["ihw_result"])   # the same QC plots as above
    ```

    See `docs/pysceptre.md` for the full pipeline, including the null check on
    calibration pairs.
    """)
    return


if __name__ == "__main__":
    app.run()
