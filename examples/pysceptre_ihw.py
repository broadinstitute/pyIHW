"""IHW on a synthetic high-MOI pysceptre screen, with analytical power as covariate.

Run with ``python examples/pysceptre_ihw.py``; it writes QC figures next to it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from pysceptre import compute_power, run_calibration_check, run_discovery_analysis
from pysceptre.analytical_power import (
    baseline_expression_stats_from_fits,
    cells_per_grna_from_assignments,
)
from pysceptre.pipeline.discovery import fit_all_genes
from scipy import sparse
from scipy.stats import false_discovery_control

from pyihw import (
    calibration_cells_per_grna,
    ihw_with_power,
    plot_pvalue_histograms,
    qc_report,
)

ALPHA = 0.1
OUT = Path(__file__).parent

# --- a synthetic screen: 3,000 cells, 400 genes, 40 targets x 2 gRNAs, 10 NT gRNAs ---
rng = np.random.default_rng(0)
n_cells, n_genes, n_targets = 3000, 400, 40
covariate_matrix = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
guides = [f"t{k}_g{j}" for k in range(n_targets) for j in range(2)]
ntc_guides = [f"nt{j}" for j in range(10)]
grna_cells = {
    g: np.flatnonzero(rng.random(n_cells) < rng.uniform(0.01, 0.06))
    for g in guides + ntc_guides
}
targeting_grna_cells = {g: grna_cells[g] for g in guides}
ntc_grna_cells = {g: grna_cells[g] for g in ntc_guides}
grna_target_cells = {
    f"t{k}": np.union1d(grna_cells[f"t{k}_g0"], grna_cells[f"t{k}_g1"]) for k in range(n_targets)
}
grna_target_data_frame = pd.DataFrame(
    {"grna_id": guides, "grna_target": [g.split("_")[0] for g in guides]}
)

gene_ids = [f"gene{i}" for i in range(n_genes)]
base = np.exp(rng.uniform(-1.5, 2.5, n_genes))[:, None]
mu = base * np.exp(0.3 * covariate_matrix[:, 1])[None, :]
for i in rng.choice(n_genes, 60, replace=False):  # 60 true links, 40% knockdown
    mu[i, grna_target_cells[f"t{rng.integers(n_targets)}"]] *= 0.6
counts = rng.negative_binomial(5, 5 / (5 + mu))
response_matrix = sparse.csr_matrix(counts.astype(float))
pairs = pd.DataFrame(
    [(g, t) for g in gene_ids for t in grna_target_cells], columns=["response_id", "grna_target"]
)

# --- 1. discovery ---
discovery = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs, seed=0
)

# --- 2. analytical power for the same pairs ---
fits = fit_all_genes(response_matrix, gene_ids, covariate_matrix)
baseline = baseline_expression_stats_from_fits(covariate_matrix, fits)
cells_per_grna = cells_per_grna_from_assignments(grna_target_data_frame, targeting_grna_cells)
cutoff = ALPHA / len(pairs) / 2  # pre-specified, not read off the discovery p-values
power = compute_power(
    pairs, cells_per_grna, baseline,
    fold_change_mean=0.85, fold_change_sd=0.13,
    cutoff=cutoff, num_total_cells=n_cells,
)

# --- 3. IHW: weighted BH with power as the covariate ---
weighted = ihw_with_power(discovery, power, alpha=ALPHA, nbins=8)
fit = weighted.attrs["ihw_result"]
bh = false_discovery_control(discovery["p_value"], method="bh")
print(fit)
print("BH rejections:", int(np.sum(bh <= ALPHA)), "| IHW rejections:", fit.rejections())
qc_report(fit).savefig(OUT / "qc_report.png", dpi=90)

# --- 4. null check: the same pipeline on negative-control (calibration) pairs ---
calibration = run_calibration_check(
    response_matrix, gene_ids, covariate_matrix, ntc_grna_cells,
    n_calibration_pairs=len(pairs), calibration_group_size=2,
    n_nonzero_trt_thresh=0, n_nonzero_cntrl_thresh=0, seed=0,
)
calibration_power = compute_power(
    calibration, calibration_cells_per_grna(calibration, ntc_grna_cells), baseline,
    fold_change_mean=0.85, fold_change_sd=0.13,
    cutoff=cutoff, num_total_cells=n_cells,
)
null_fit = ihw_with_power(calibration, calibration_power, alpha=ALPHA, nbins=8).attrs[
    "ihw_result"
]
print("Calibration (should be ~0):", null_fit)
plot_pvalue_histograms(null_fit, ncols=4).savefig(OUT / "calibration_histograms.png", dpi=90)
