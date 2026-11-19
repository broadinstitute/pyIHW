# QC plots

All plots require Matplotlib (`pip install 'pyihw[plot]'`). Each function takes
an `IHWResult`. Single-axis plots accept `ax=` and return the axis; multi-panel
plots return a figure.

| function | `result.plot(what=...)` | shows |
| --- | --- | --- |
| `qc_report(result)` | `"qc"` | six-panel overview (below) |
| `plot_weights(result)` | `"weights"` | learned weight per stratum, one line per fold |
| `plot_decision_boundary(result)` | `"decisionboundary"` | rejection threshold on -log10 p vs covariate value (ordinal only); pass `x_axis="ranks"` for rank |
| `plot_pvalue_histograms(result)` | `"pvalues"` | one p-value histogram per stratum with Storey's $\hat\pi_0$ |
| `plot_rejections(result)` | `"rejections"` | IHW vs unweighted BH rejections across α |
| `plot_pvalue_histogram(result)` | | histogram of all p-values |
| `plot_pvalue_densities(result)` | | overlaid p-value densities per stratum |
| `plot_stratum_summary(result)` | | $\hat\pi_0$ and fraction rejected per stratum |

## The `qc_report` panels

1. **All p-values**: a flat right half with a spike near 0. A bump near 1 or
   a slope in the right half points to miscalibrated p-values. Fix that first.
2. **Density by stratum**: the null part (p > 0.5) should be flat at the same
   height in every stratum. Strata whose right halves differ suggest the
   covariate is *not* independent of null p-values, which would break FDR
   control.
3. **Signal by stratum**: $\hat\pi_0$, the estimated null fraction, should fall
   as the covariate becomes more informative, and the fraction rejected should
   rise. If both are flat, the covariate carries no information and IHW will
   behave like BH.
4. **Learned weights**: folds should agree roughly. Large disagreement
   between folds means too few tests per stratum, so lower `nbins`.
5. **Decision boundary**: the per-fold rejection threshold against the covariate value. Tests above it
   are rejected.
6. **Rejections vs α**: weights are learned at the fitted α. The curve away
   from that α re-applies the same weights, so read it as indicative only.
