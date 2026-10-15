"""Generate every portfolio figure from the persisted metric CSVs.

No value is hardcoded: each figure reads `docs/portfolio/data/*.csv`, which is produced by
`docs/portfolio/scripts/compute_metrics.py`. `src/` is never touched.

Each figure is written four ways:
    figures/NN_name.png          article width, 300 dpi
    figures/NN_name.svg          article width, vector
    figures/square/NN_name.png   1080x1080 for a social feed
    figures/square/NN_name.svg   1080x1080, vector

Design rules: white background, sans-serif, no titles inside the figure (captions live in the
text), labelled axes, numeric values annotated, no gridlines beyond a light reference, and
hatching in addition to colour so the figures survive black-and-white printing.

Usage:
    python docs/portfolio/figures/scripts/make_figures.py          # all
    python docs/portfolio/figures/scripts/make_figures.py 2 3      # selected
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "docs" / "portfolio" / "data"
FIG = ROOT / "docs" / "portfolio" / "figures"
(FIG / "square").mkdir(parents=True, exist_ok=True)

# --- palette: one accent, grey for reference, red only for what does not work -------------
INK = "#2B2B2B"
GREY = "#9E9E9E"
GREY_SOFT = "#D8D8D8"
ACCENT = "#1F4E79"
RED = "#A32E25"
HATCH_NEWS = "//"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#BBBBBB",
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def _save(fig, name: str, square: bool) -> None:
    if square:
        # Resize first, then recompute the layout. Calling tight_layout before the resize
        # leaves margins sized for the wide canvas, which clips long tick labels.
        fig.set_size_inches(5.4, 5.4)
        fig.tight_layout()
        png, svg, dpi, bbox = FIG / "square" / f"{name}.png", FIG / "square" / f"{name}.svg", 200, None
    else:
        png, svg, dpi, bbox = FIG / f"{name}.png", FIG / f"{name}.svg", 300, "tight"
    fig.savefig(png, dpi=dpi, bbox_inches=bbox, facecolor="white")
    fig.savefig(svg, bbox_inches=bbox, facecolor="white")
    plt.close(fig)
    print(f"  {png.relative_to(ROOT)}")
    print(f"  {svg.relative_to(ROOT)}")


def _fs(square: bool, base: float) -> float:
    """Font size: slightly smaller on the square canvas, which is more crowded."""
    return base * (0.82 if square else 1.0)


# ------------------------------------------------------------------------------------- #
# Figure 1 — architecture / pipeline
# ------------------------------------------------------------------------------------- #
def fig01(square: bool = False) -> None:
    counts = pd.read_csv(DATA / "dataset_counts.csv").set_index("claim")["value"]
    shape = pd.read_csv(DATA / "energy_panel_shape.csv").set_index("stage")["value"]
    wf = pd.read_csv(DATA / "energy_level_walkforward.csv")
    n_folds = int((~wf["skipped"]).sum())

    def n(key, src=counts):
        return f"{int(src[key]):,}"

    stages = [
        ("Data sources", "EIA spot prices  ·  FRED rates  ·  EODHD news", False),
        ("Ingestion", "resumable pulls, raw payloads kept immutable on disk", False),
        ("Point-in-time cleaning",
         f"{n('price_panel_rows')} price rows  ·  no forward-fill  ·  native log returns", True),
        ("News corpus",
         f"{n('corpus_unique_articles')} unique articles  "
         f"({n('corpus_rows_article_x_asset')} article-asset rows)", False),
        ("FinBERT scoring",
         f"{n('articles_scored_finbert_energy')} articles scored, title and body", False),
        ("Daily aggregation",
         f"{n('daily_sentiment_rows_energy')} asset-days  ·  features at t, target at t+1", True),
        ("Feature matrix",
         f"{n('feature_matrix_rows', shape)} rows x {shape['n_features_M3']} features  "
         f"({shape['panel_start']} to {shape['panel_end']})", False),
        ("Ablation M0 - M3", "price / +volume / +third-party tone / +FinBERT", False),
        ("Walk-forward evaluation",
         f"{n_folds} expanding folds, 6-month test windows, train strictly in the past", True),
        ("Metrics", "R², RMSE, ROC-AUC, average precision, permutation importance", False),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.055, 1.0)   # bottom strip reserved for the legend line
    ax.axis("off")

    n = len(stages)
    pad = 0.012
    h = (1.0 - pad * (n - 1)) / n
    x0, w = 0.16, 0.80

    for i, (label, detail, pit) in enumerate(stages):
        y = 1.0 - (i + 1) * h - i * pad
        ax.add_patch(FancyBboxPatch(
            (x0, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=1.1, edgecolor="#B9C4CE", facecolor="#F4F7FA"))
        ax.text(x0 + 0.022, y + h * 0.63, label, fontsize=_fs(square, 11.5),
                fontweight="bold", va="center", color=INK)
        ax.text(x0 + 0.022, y + h * 0.26, detail, fontsize=_fs(square, 8.6),
                va="center", color="#5A6672")
        if pit:
            ax.add_patch(FancyBboxPatch(
                (0.012, y + h * 0.28), 0.125, h * 0.44,
                boxstyle="round,pad=0.003,rounding_size=0.010",
                linewidth=0, facecolor=ACCENT))
            ax.text(0.0745, y + h * 0.50, "point-in-time", fontsize=_fs(square, 6.6),
                    ha="center", va="center", color="white", fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(x0 + w / 2, y - pad), xytext=(x0 + w / 2, y),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.1,
                                        shrinkA=0, shrinkB=0))

    ax.text(0.012, -0.032, "Blue badge = stage where point-in-time discipline is enforced",
            fontsize=_fs(square, 7.4), color="#5A6672")
    fig.tight_layout()
    _save(fig, "01_architecture", square)


# ------------------------------------------------------------------------------------- #
# Figure 2 — persistence vs models
# ------------------------------------------------------------------------------------- #
def fig02(square: bool = False) -> None:
    d = pd.read_csv(DATA / "energy_level_single_split.csv")
    labels = {"persistence": "Persistence baseline\n(tomorrow = today)",
              "M0 precio": "M0  price only",
              "M1 +volumen": "M1  + news volume",
              "M2 +EODHD": "M2  + third-party tone",
              "M3 +FinBERT": "M3  + FinBERT sentiment"}
    d["label"] = d["model"].map(labels)
    pers = float(d.loc[d["model"] == "persistence", "r2_oos"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ypos = range(len(d))
    colors = [ACCENT if m == "persistence" else GREY for m in d["model"]]
    ax.barh(list(ypos), d["r2_oos"], height=0.62, color=colors,
            edgecolor="white", linewidth=0.8)
    ax.axvline(pers, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1.3, zorder=3)

    for y, v in zip(ypos, d["r2_oos"]):
        ax.text(v + 0.012, y, f"{v:.4f}", va="center",
                fontsize=_fs(square, 9.5), color=INK)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(d["label"], fontsize=_fs(square, 9.5))
    ax.invert_yaxis()
    # Headroom above the first bar so the reference label does not collide with the axis.
    ax.set_ylim(len(d) - 0.5, -1.15)
    ax.text(pers, -0.78, f"the trivial baseline wins:  {pers:.4f}", color=ACCENT,
            fontsize=_fs(square, 9.5), ha="right", va="center", fontweight="bold")
    ax.set_xlabel("Out-of-sample $R^2$  —  next-day realized volatility (5-day window)",
                  fontsize=_fs(square, 10))
    ax.set_xlim(0, max(d["r2_oos"]) * 1.16)
    ax.xaxis.grid(True, color=GREY_SOFT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "02_persistence_vs_models", square)


# ------------------------------------------------------------------------------------- #
# Figure 3 — permutation importance, symlog
# ------------------------------------------------------------------------------------- #
def fig03(square: bool = False) -> None:
    d = pd.read_csv(DATA / "energy_permutation_level.csv").sort_values(
        "importance_mean", ascending=True).reset_index(drop=True)

    def style_of(row):
        if row["feature"] == "vol_t":
            return ACCENT, ""
        if row["block"] == "price":
            return GREY, ""
        return RED, HATCH_NEWS

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, row in d.iterrows():
        c, hatch = style_of(row)
        ax.barh(i, row["importance_mean"], height=0.62, color=c, hatch=hatch,
                edgecolor="white", linewidth=0.8)

    ax.set_xscale("symlog", linthresh=1e-4)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=_fs(square, 9))
    ax.set_xlabel("Permutation importance: $R^2$ drop when the feature is shuffled\n"
                  "(symmetric log scale, which also shows negative values)",
                  fontsize=_fs(square, 9.5))
    ax.xaxis.grid(True, color=GREY_SOFT, linewidth=0.6)
    ax.set_axisbelow(True)

    # Notes live in the empty upper-left quadrant. No connector line: a long diagonal arrow
    # across a bar chart reads as plotted data.
    top = d.iloc[-1]
    best_fb = d[d["block"] == "finbert"]["importance_mean"].max()
    ax.text(0.035, 0.93,
            f"{top['feature']} = {top['importance_mean']:.4f}\n"
            f"{top['importance_mean'] / best_fb:.0f}x the best FinBERT feature",
            transform=ax.transAxes, fontsize=_fs(square, 9.2), color=ACCENT,
            fontweight="bold", va="top", ha="left", linespacing=1.5)
    ax.text(0.035, 0.74,
            "Hatched red = news-derived features\n(volume, third-party tone, FinBERT)",
            transform=ax.transAxes, fontsize=_fs(square, 8.2), color="#5A6672",
            va="top", ha="left", linespacing=1.4)
    fig.tight_layout()
    _save(fig, "03_permutation_importance_log", square)


# ------------------------------------------------------------------------------------- #
# Figure 4 — walk-forward by fold
# ------------------------------------------------------------------------------------- #
def fig04(square: bool = False) -> None:
    d = pd.read_csv(DATA / "energy_level_walkforward.csv")
    d = d[~d["skipped"]].reset_index(drop=True)
    delta = d["delta_M3_minus_M0"]
    mean, pos = delta.mean(), int((delta > 0).sum())

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    colors = [ACCENT if v > 0 else RED for v in delta]
    hatches = ["" if v > 0 else HATCH_NEWS for v in delta]
    for i, (v, c, hh) in enumerate(zip(delta, colors, hatches)):
        ax.bar(i, v, width=0.56, color=c, hatch=hh, edgecolor="white", linewidth=0.8)
        off = 0.0004 if v > 0 else -0.0004
        ax.text(i, v + off, f"{v:+.4f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=_fs(square, 8.8), color=INK)

    ax.axhline(0, color=INK, linewidth=1.0)
    ax.axhline(mean, color=GREY, linestyle=(0, (4, 3)), linewidth=1.2)
    # Summary inside the axes: an annotation hanging off the right edge gets clipped on the
    # square canvas, where the figure is saved without a tight bounding box.
    ax.text(0.02, 0.97, f"mean {mean:+.4f}  ·  positive in {pos} of {len(delta)} folds",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=_fs(square, 9), color="#5A6672", fontweight="bold")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"fold {int(f)}\n{t}" for f, t in zip(d["fold"], d["test_start"])],
                       fontsize=_fs(square, 8.6))
    ax.set_ylabel("$\\Delta R^2$  (M3 +FinBERT  -  M0 price)", fontsize=_fs(square, 10))
    ax.set_xlabel("Expanding walk-forward folds, 6-month test window each",
                  fontsize=_fs(square, 9.5))
    lim = max(abs(delta)) * 1.45
    ax.set_ylim(-lim, lim)
    ax.yaxis.grid(True, color=GREY_SOFT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "04_walkforward_by_fold", square)


# ------------------------------------------------------------------------------------- #
# Figure 5 — generalization across sectors
# ------------------------------------------------------------------------------------- #
def fig05(square: bool = False) -> None:
    """Per-fold FinBERT delta in every universe where the block could be isolated.

    Folds are shown individually rather than as a mean bar: a single mean hides that the
    effect wins a minority of folds and that its sign is not stable. Both panels use
    macro-AUC so the universes are measured with the same estimator.
    """
    en = pd.read_csv(DATA / "energy_jumps_walkforward_macro.csv")
    se = pd.read_csv(DATA / "sectors_walkforward.csv")
    allf = pd.concat([en, se], ignore_index=True)
    allf = allf[~allf["skipped"]]

    order = ["energy", "tech", "biotech", "financials"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    # Per-sector summaries live in the tick labels, not inside the axes: text positioned in
    # data coordinates runs off the square canvas, which is saved without a tight bounding box.
    labels, spread = [], []
    for y, sec in enumerate(order):
        g = allf[(allf["sector"] == sec) & (allf["delta_finbert_M3_minus_M2"].notna())]
        if g.empty:
            labels.append(f"{sec.capitalize()}\nnot evaluated")
            continue
        v = g["delta_finbert_M3_minus_M2"]
        spread.extend(v.tolist())
        ax.scatter(v, [y] * len(v), s=62,
                   c=[ACCENT if x > 0 else RED for x in v],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([v.mean()], [y], marker="D", s=52, c=INK, zorder=4)
        labels.append(f"{sec.capitalize()}\nmean {v.mean():+.4f}  ·  "
                      f"{int((v > 0).sum())}/{len(v)} folds +")

    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=_fs(square, 8.8))
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ macro-AUC from adding the FinBERT block\n"
                  "(one dot per walk-forward fold; diamond = mean)\n"
                  "Financials: news corpus unavailable, block not estimated.",
                  fontsize=_fs(square, 9))
    pad = (max(spread) - min(spread)) * 0.12
    ax.set_xlim(min(spread) - pad, max(spread) + pad)
    ax.xaxis.grid(True, color=GREY_SOFT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "05_sector_generalization", square)


FIGS = {"1": fig01, "2": fig02, "3": fig03, "4": fig04, "5": fig05}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for k in which:
        if k not in FIGS:
            print(f"skip unknown figure {k}")
            continue
        print(f"[figure {k}]")
        FIGS[k](square=False)
        FIGS[k](square=True)
