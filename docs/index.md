# pyIHW

`pyihw` is a Python implementation of Independent Hypothesis Weighting (IHW),
based on the Bioconductor R package [IHW](https://bioconductor.org/packages/IHW/).
It learns covariate-dependent hypothesis weights using cross-fitting, then
applies weighted Benjamini–Hochberg (FDR) or Bonferroni (FWER) adjustment.

!!! warning
    The covariate must be independent of the p-value under the null hypothesis.
    A covariate that violates this assumption can invalidate error-rate control.

## Install

```bash
python -m pip install "pyihw[plot] @ git+https://github.com/broadinstitute/pyIHW"
```

!!! note
    `pip install pyihw` from PyPI installs an unrelated project with the same
    name. Install from the GitHub URL above.

## Quick start

```python
import numpy as np
from scipy.stats import norm
from pyihw import ihw, qc_report

rng = np.random.default_rng(42)
covariate = rng.uniform(0, 4, size=10_000)
is_signal = rng.binomial(1, 0.1, size=10_000)
pvalues = norm.sf(rng.normal(is_signal * covariate))

result = ihw(pvalues, covariate, alpha=0.1, nbins=10, seed=1)
print(result.rejections())
qc_report(result).savefig("ihw_qc.png")
```

## Where next

* **[Interactive notebook](notebook.md)**: a toy example that runs in your
  browser, with plots and an explanation of each step.
* **[pysceptre workflow](pysceptre.md)**: use analytical power as the covariate
  for multiple-testing correction of a pysceptre screen, with a null check.
* **[QC plots](qc.md)**: what each diagnostic panel should look like.
* **[API reference](api.md)**.

## Parity with R

Results are tested against Bioconductor IHW 1.40.0 (`tests/test_parity.py`).
With identical folds and a single λ, weights agree to about 1e-12 and
rejections are identical. Python's `seed` does not reproduce R's random number
stream, so supply explicit `folds` when comparing across implementations.

## Attribution

A Python reimplementation of the method of Ignatiadis, Klaus, Zaugg, and Huber
(2016), *Data-driven hypothesis weighting increases detection power in
genome-scale multiple testing*, Nature Methods 13, 577–580,
[doi:10.1038/nmeth.3885](https://doi.org/10.1038/nmeth.3885). The upstream R
package is maintained by Nikos Ignatiadis and Wolfgang Huber under the Artistic
License 2.0. This project is independent and not affiliated with Bioconductor.
