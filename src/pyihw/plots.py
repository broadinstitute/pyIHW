"""Diagnostic plots for an :class:`~pyihw.IHWResult` (requires Matplotlib).

Every ``plot_*`` function draws on ``ax`` (or a new axes) and returns it;
``plot_pvalue_histograms`` and ``qc_report`` return a whole figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .multiple_testing import adjust_pvalues
from .result import IHWResult

STOREY_LEVEL = 0.5
PI0 = r"$\hat{\pi}_0$"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install pyihw with the 'plot' extra to draw plots.") from exc
    return plt


def _axes(ax):
    return _pyplot().subplots()[1] if ax is None else ax


def _tested(result: IHWResult) -> pd.DataFrame:
    return result._frame.loc[result._frame["pvalue"].notna()]


def _stratum_pvalues(result: IHWResult) -> list[np.ndarray]:
    frame = _tested(result)
    codes = frame["group_code"].to_numpy(dtype=int)
    p = frame["pvalue"].to_numpy(dtype=float)
    return [p[codes == g] for g in range(result.nbins)]


def _numeric_covariate(frame: pd.DataFrame) -> np.ndarray | None:
    values = pd.to_numeric(frame["covariate"], errors="coerce").to_numpy(dtype=float)
    return None if np.isnan(values).any() else values


def stratum_labels(result: IHWResult) -> list[str]:
    """Covariate range of each numeric stratum, or the category name."""
    frame = _tested(result)
    codes = frame["group_code"].to_numpy(dtype=int)
    covariate = _numeric_covariate(frame)
    if covariate is None or result.covariate_type == "nominal":
        labels = frame.groupby(codes)["group"].first()
        return [str(labels.get(g, g + 1)) for g in range(result.nbins)]
    labels = []
    for g in range(result.nbins):
        values = covariate[codes == g]
        labels.append(f"{values.min():.2g}–{values.max():.2g}" if len(values) else "empty")
    return labels


def storey_pi0(pvalues: np.ndarray, level: float = STOREY_LEVEL) -> float:
    """Storey's estimate of the proportion of true nulls, capped at 1."""
    if not len(pvalues):
        return float("nan")
    return float(min(1.0, np.mean(pvalues > level) / (1 - level)))


def plot_weights(result: IHWResult, *, scale: str | None = None, ax: Any = None) -> Any:
    """Learned weight of each stratum, one line (or bar group) per fold."""
    scale = scale or result.covariate_type
    if scale not in ("ordinal", "nominal"):
        raise ValueError("scale must be 'ordinal' or 'nominal'")
    ax = _axes(ax)
    matrix = result.weights(levels_only=True)
    positions = np.arange(result.nbins)
    labels = stratum_labels(result)
    if scale == "nominal":
        width = 0.8 / result.nfolds
        for fold in range(result.nfolds):
            offset = (fold - (result.nfolds - 1) / 2) * width
            ax.bar(positions + offset, matrix[:, fold], width=width, label=f"Fold {fold + 1}")
    else:
        for fold in range(result.nfolds):
            ax.plot(positions, matrix[:, fold], marker="o", label=f"Fold {fold + 1}")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    _stratum_ticks(ax, labels)
    ax.set_ylabel("Weight")
    ax.set_title("Learned weights")
    if result.nfolds > 1:
        ax.legend(fontsize="small")
    return ax


def plot_decision_boundary(
    result: IHWResult, *, x_axis: str = "covariate", ax: Any = None
) -> Any:
    """Tests on (covariate, -log10 p) with each fold's rejection threshold.

    A hypothesis is rejected when its point lies above its fold's line.
    """
    if x_axis not in ("ranks", "covariate"):
        raise ValueError("x_axis must be 'ranks' or 'covariate'")
    if result.covariate_type == "nominal":
        raise NotImplementedError("decision-boundary plots require an ordinal covariate")
    ax = _axes(ax)
    frame = _tested(result)
    codes = frame["group_code"].to_numpy(dtype=int)
    covariate = _numeric_covariate(frame)
    if covariate is None:
        jitter = np.random.default_rng(0).uniform(-0.4, 0.4, len(codes))
        x, xlabel = codes + 1 + jitter, "Stratum"
    elif x_axis == "ranks":
        x, xlabel = pd.Series(covariate).rank(method="first").to_numpy(), "Covariate rank"
    else:
        x, xlabel = covariate, "Covariate"
    y = -np.log10(np.maximum(frame["pvalue"].to_numpy(dtype=float), 1e-300))
    rejected = frame["adj_pvalue"].to_numpy(dtype=float) <= result.alpha

    ax.scatter(x[~rejected], y[~rejected], s=4, color="0.7", rasterized=True, label="Not rejected")
    ax.scatter(x[rejected], y[rejected], s=6, color="C3", rasterized=True, label="Rejected")
    thresholds = result.thresholds(levels_only=True)
    with np.errstate(divide="ignore"):
        boundary = np.where(thresholds > 0, -np.log10(thresholds), np.nan)
    for fold in range(result.nfolds):
        xs, ys = [], []
        for g in range(result.nbins):
            in_stratum = codes == g
            if in_stratum.any():
                xs += [x[in_stratum].min(), x[in_stratum].max()]
                ys += [boundary[g, fold]] * 2
        ax.plot(xs, ys, color=f"C{fold}", linewidth=1.2, label=f"Fold {fold + 1} threshold")
    if not np.isfinite(boundary).any():
        ax.set_title("Decision boundary (no rejections)")
    else:
        ax.set_title("Decision boundary")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("-log10(p-value)")
    ax.legend(fontsize="small", markerscale=2)
    return ax


def plot_pvalue_histograms(
    result: IHWResult, *, bins: int = 20, ncols: int = 5, figsize: tuple | None = None
) -> Any:
    """One p-value histogram per stratum, with Storey's null level dashed.

    Under a valid covariate, the right-hand side of every histogram is flat.
    """
    plt = _pyplot()
    nrows = int(np.ceil(result.nbins / ncols))
    ncols = min(ncols, result.nbins)
    figsize = figsize or (2.6 * ncols, 2.2 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    edges = np.linspace(0, 1, bins + 1)
    for ax, p, label in zip(axes.flat, _stratum_pvalues(result), stratum_labels(result)):
        ax.hist(p, bins=edges, density=True, color="C0", alpha=0.8)
        pi0 = storey_pi0(p)
        ax.axhline(pi0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{label}\nn={len(p)}, {PI0}={pi0:.2f}", fontsize="small")
        ax.tick_params(labelsize="x-small")
    for ax in list(axes.flat)[result.nbins :]:
        ax.set_visible(False)
    fig.supxlabel("p-value")
    fig.supylabel("Density")
    fig.suptitle("p-values by covariate stratum")
    fig.tight_layout()
    return fig


def plot_pvalue_histogram(result: IHWResult, *, bins: int = 40, ax: Any = None) -> Any:
    """Histogram of all tested p-values."""
    ax = _axes(ax)
    p = _tested(result)["pvalue"].to_numpy(dtype=float)
    ax.hist(p, bins=np.linspace(0, 1, bins + 1), density=True, color="C0", alpha=0.8)
    ax.axhline(storey_pi0(p), color="k", linestyle="--", linewidth=1, label=f"Storey {PI0}")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("All p-values")
    ax.legend(fontsize="small")
    return ax


def plot_pvalue_densities(result: IHWResult, *, bins: int = 20, ax: Any = None) -> Any:
    """Stratum p-value densities overlaid, coloured from low to high stratum.

    The y-axis is clipped to keep the null (right-hand) part readable.
    """
    plt = _pyplot()
    ax = _axes(ax)
    edges = np.linspace(0, 1, bins + 1)
    cmap = plt.get_cmap("viridis")
    tail_max = 1.0
    for g, p in enumerate(_stratum_pvalues(result)):
        if not len(p):
            continue
        density, _ = np.histogram(p, bins=edges, density=True)
        tail_max = max(tail_max, density[edges[:-1] >= STOREY_LEVEL].max())
        colour = cmap(g / max(result.nbins - 1, 1))
        ax.stairs(density, edges, baseline=None, color=colour, linewidth=1.2)
    ax.set_ylim(0, 2 * tail_max)
    norm = plt.Normalize(1, max(result.nbins, 2))
    colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Stratum")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("p-value density by stratum")
    return ax


def plot_stratum_summary(result: IHWResult, *, ax: Any = None) -> Any:
    """Per-stratum null proportion (Storey) and fraction of tests rejected."""
    ax = _axes(ax)
    frame = _tested(result)
    codes = frame["group_code"].to_numpy(dtype=int)
    rejected = frame["adj_pvalue"].to_numpy(dtype=float) <= result.alpha
    positions = np.arange(result.nbins)
    pi0 = [storey_pi0(p) for p in _stratum_pvalues(result)]
    rate = [rejected[codes == g].mean() if np.any(codes == g) else np.nan for g in positions]
    ax.bar(positions, pi0, color="0.8", label=f"Null proportion {PI0}")
    ax.plot(positions, rate, color="C3", marker="o", label="Fraction rejected")
    ax.set_ylim(0, 1.05)
    _stratum_ticks(ax, stratum_labels(result))
    ax.set_ylabel("Proportion")
    ax.set_title("Signal by stratum")
    ax.legend(fontsize="small")
    return ax


def plot_rejections(
    result: IHWResult, *, alphas: np.ndarray | None = None, ax: Any = None
) -> Any:
    """Rejections of IHW and the unweighted procedure across nominal levels.

    IHW's weights were learned at ``result.alpha`` and are not refitted for
    other levels, so the curve is exact only at that level.
    """
    ax = _axes(ax)
    if alphas is None:
        alphas = np.linspace(0.005, min(0.99, max(0.2, 2 * result.alpha)), 60)
    frame = _tested(result)
    mtests = int(np.sum(result.m_groups))
    unweighted = adjust_pvalues(
        frame["pvalue"].to_numpy(dtype=float), result.adjustment_type, mtests
    )
    weighted = frame["adj_pvalue"].to_numpy(dtype=float)
    name = "BH" if result.adjustment_type == "BH" else "Bonferroni"
    ax.plot(alphas, [np.sum(weighted <= a) for a in alphas], color="C3", label="IHW")
    ax.plot(alphas, [np.sum(unweighted <= a) for a in alphas], color="0.4", label=name)
    ax.axvline(
        result.alpha, color="k", linestyle=":", linewidth=1,
        label=f"Weights learned at {result.alpha:g}",
    )
    ax.set_xlabel("Nominal level")
    ax.set_ylabel("Rejections")
    ax.set_title("Rejections vs level")
    ax.legend(fontsize="small")
    return ax


def qc_report(result: IHWResult, *, bins: int = 20, figsize: tuple = (16, 9)) -> Any:
    """Six-panel overview of an IHW fit: p-values, signal by stratum, weights,
    decision boundary and rejections."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    plot_pvalue_histogram(result, bins=2 * bins, ax=axes[0, 0])
    plot_pvalue_densities(result, bins=bins, ax=axes[0, 1])
    plot_stratum_summary(result, ax=axes[0, 2])
    plot_weights(result, ax=axes[1, 0])
    if result.covariate_type == "ordinal":
        plot_decision_boundary(result, ax=axes[1, 1])
    else:
        axes[1, 1].set_axis_off()
    plot_rejections(result, ax=axes[1, 2])
    fig.suptitle(repr(result))
    fig.tight_layout()
    return fig


def _stratum_ticks(ax, labels: list[str]) -> None:
    positions = np.arange(len(labels))
    step = max(1, int(np.ceil(len(labels) / 12)))
    ax.set_xticks(positions[::step], labels[::step], rotation=45, ha="right", fontsize="x-small")
    ax.set_xlabel("Covariate stratum")
