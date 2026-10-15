"""Recompute and PERSIST every metric used in the portfolio writeup.

Why this file exists: nothing in `src/` writes metrics to disk - the experiment scripts print
to stdout and return dicts. This script re-runs them with their existing seeds and
hyperparameters and writes structured CSVs, so that (a) every published number traces to a
file and (b) no figure hardcodes a value.

`src/` is NOT modified. This script only imports it. Walk-forward cut dates, fold thresholds,
model hyperparameters and the jump threshold all come from the source modules (or are copied
verbatim from them where they are function-local; verified identical across the three modules).

Usage:
    python docs/portfolio/scripts/compute_metrics.py counts
    python docs/portfolio/scripts/compute_metrics.py energy
    python docs/portfolio/scripts/compute_metrics.py sectors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src import config                                        # noqa: E402
from src.model import features                                # noqa: E402
from src.model.train import FEATURE_SETS, SPLIT_DATE, _model  # noqa: E402
from src.model.train_jumps import _clf, _label, _load         # noqa: E402
from src.sectors import ablation, baskets, panel              # noqa: E402

OUT = ROOT / "docs" / "portfolio" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Walk-forward grid. Identical in src/model/train.py, src/model/train_jumps.py and
# src/sectors/ablation.py (verified by reading all three). Taken from the one module that
# exposes it at module level, so there is a single source of truth.
CUTS = ablation.CUTS
HORIZON = ablation.HORIZON
MIN_ROWS_LEVEL_FOLD = 30      # src/model/train.py:111      -> if len(te) < 30: continue
MIN_JUMPS_ENERGY_FOLD = 10    # src/model/train_jumps.py:111 -> if te[jump].sum() < 10
MIN_JUMPS_SECTOR_FOLD = 20    # src/sectors/ablation.py:101  -> if b[jump].sum() < 20

BLOCK_OF = {}
for _f in features.PRICE_FEATURES:
    BLOCK_OF[_f] = "price"
for _f in features.NEWS_VOLUME:
    BLOCK_OF[_f] = "news_volume"
for _f in features.EODHD_FEATURES:
    BLOCK_OF[_f] = "third_party_tone"
for _f in features.FINBERT_FEATURES:
    BLOCK_OF[_f] = "finbert"


def _save(df: pd.DataFrame, name: str) -> None:
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  wrote docs/portfolio/data/{name}  ({len(df)} rows)")


# --------------------------------------------------------------------------- #
# Dataset counts
# --------------------------------------------------------------------------- #
def counts() -> None:
    print("[counts]")
    rows = []

    def add(claim, value, src, how):
        rows.append({"claim": claim, "value": value, "source_file": src, "how": how})

    p = config.PRICES_DAILY_PARQUET
    d = pd.read_parquet(p)
    add("price_panel_rows", pq.ParquetFile(p).metadata.num_rows, p.name, "parquet metadata")
    add("price_panel_start", str(d["date"].min().date()), p.name, "min(date)")
    add("price_panel_end", str(d["date"].max().date()), p.name, "max(date)")
    add("price_panel_assets", "|".join(sorted(d["asset"].astype(str).unique())), p.name,
        "unique(asset)")
    add("price_log_return_nonnull", int(d["log_return"].notna().sum()), p.name, "notna().sum()")

    p = config.NEWS_CORPUS_PARQUET
    nc = pd.read_parquet(p, columns=["url", "trading_day", "asset"])
    add("corpus_rows_article_x_asset", pq.ParquetFile(p).metadata.num_rows, p.name,
        "parquet metadata")
    add("corpus_unique_articles", int(nc["url"].nunique()), p.name, "url.nunique()")
    add("corpus_start", str(nc["trading_day"].min().date()), p.name, "min(trading_day)")
    add("corpus_end", str(nc["trading_day"].max().date()), p.name, "max(trading_day)")

    p = config.SENTIMENT_ARTICLES_PARQUET
    sa = pd.read_parquet(p)
    add("articles_scored_finbert_energy", int(sa["url"].nunique()), p.name, "url.nunique()")

    p = config.SENTIMENT_DAILY_PARQUET
    add("daily_sentiment_rows_energy", pq.ParquetFile(p).metadata.num_rows, p.name,
        "parquet metadata")

    for sec in baskets.SECTORS:
        tp = baskets.titles_scored_path(sec)
        if tp.exists():
            t = pd.read_parquet(tp)
            add(f"finbert_daily_rows_{sec}", len(t), tp.name, "row count")
            add(f"tickers_{sec}", "|".join(sorted(t["asset"].unique())), tp.name,
                "unique(asset)")
        else:
            add(f"finbert_daily_rows_{sec}", "UNAVAILABLE", "-", "no scored parquet on disk")
        add(f"tickers_configured_{sec}", "|".join(baskets.SECTORS[sec]),
            "src/sectors/baskets.py", "SECTORS constant")

    add("modeling_window_start", str(features.CORPUS_START.date()),
        "src/model/features.py", "CORPUS_START constant")
    add("vol_window_days", features.VOL_WINDOW, "src/model/features.py", "VOL_WINDOW constant")
    add("jump_quantile", ablation.TOP_Q, "src/sectors/ablation.py", "TOP_Q constant")
    add("split_date", str(SPLIT_DATE.date()), "src/model/train.py", "SPLIT_DATE constant")

    _save(pd.DataFrame(rows), "dataset_counts.csv")


# --------------------------------------------------------------------------- #
# Energy
# --------------------------------------------------------------------------- #
def energy() -> None:
    print("[energy] building panel")
    df = features.build()
    tr, te = df[df["date"] < SPLIT_DATE], df[df["date"] >= SPLIT_DATE]
    print(f"  train={len(tr)} test={len(te)}")

    # Panel shape, needed for the volume annotations in the architecture figure.
    _save(pd.DataFrame([
        {"stage": "feature_matrix_rows", "value": len(df)},
        {"stage": "train_rows", "value": len(tr)},
        {"stage": "test_rows", "value": len(te)},
        {"stage": "n_features_M3", "value": len(FEATURE_SETS["M3 +FinBERT"])},
        {"stage": "panel_start", "value": str(df["date"].min().date())},
        {"stage": "panel_end", "value": str(df["date"].max().date())},
    ]), "energy_panel_shape.csv")

    # ---- single split, level ----
    pv = te.dropna(subset=["vol_t"])
    rows = [{"model": "persistence", "r2_oos": r2_score(pv["target"], pv["vol_t"]),
             "rmse": np.nan, "rank_corr": np.nan, "n_test": len(pv),
             "note": "predict target = vol_t (realized vol of today)"}]
    fitted = {}
    for name, feats in FEATURE_SETS.items():
        m = _model().fit(tr[feats], tr["target"])
        pred = m.predict(te[feats])
        fitted[name] = (m, feats, pred)
        rows.append({"model": name, "r2_oos": r2_score(te["target"], pred),
                     "rmse": mean_squared_error(te["target"], pred) ** 0.5,
                     "rank_corr": spearmanr(te["target"], pred).statistic,
                     "n_test": len(te), "note": ""})
    _save(pd.DataFrame(rows), "energy_level_single_split.csv")

    # ---- permutation importance, level, M3 ----
    m3, feats3, _ = fitted["M3 +FinBERT"]
    imp = permutation_importance(m3, te[feats3], te["target"], n_repeats=10, random_state=0)
    _save(pd.DataFrame({"feature": feats3,
                        "block": [BLOCK_OF.get(f, "other") for f in feats3],
                        "importance_mean": imp.importances_mean,
                        "importance_std": imp.importances_std}
                       ).sort_values("importance_mean", ascending=False),
          "energy_permutation_level.csv")

    # ---- per-asset R2 (M3) ----
    t2 = te.copy()
    t2["pred"] = fitted["M3 +FinBERT"][2]
    pa = [{"asset": str(a), "r2_oos": r2_score(s["target"], s["pred"]), "n_test": len(s)}
          for a, s in t2.groupby("asset", observed=True) if len(s) > 5]
    _save(pd.DataFrame(pa), "energy_level_by_asset.csv")

    # ---- walk-forward, level (replicates src/model/train.py walk_forward) ----
    wf = []
    for i, cut in enumerate(CUTS, start=1):
        a = df[df["date"] < cut]
        b = df[(df["date"] >= cut) & (df["date"] < cut + HORIZON)]
        if len(b) < MIN_ROWS_LEVEL_FOLD:
            wf.append({"fold": i, "test_start": str(cut.date()), "n_test": len(b),
                       "skipped": True})
            continue
        pvb = b.dropna(subset=["vol_t"])
        r2 = {}
        for name in ("M0 precio", "M3 +FinBERT"):
            m = _model().fit(a[FEATURE_SETS[name]], a["target"])
            r2[name] = r2_score(b["target"], m.predict(b[FEATURE_SETS[name]]))
        wf.append({"fold": i, "test_start": str(cut.date()), "n_test": len(b),
                   "skipped": False,
                   "r2_persistence": r2_score(pvb["target"], pvb["vol_t"]),
                   "r2_M0": r2["M0 precio"], "r2_M3": r2["M3 +FinBERT"],
                   "delta_M3_minus_M0": r2["M3 +FinBERT"] - r2["M0 precio"]})
    _save(pd.DataFrame(wf), "energy_level_walkforward.csv")

    # ---- jumps: single split (replicates src/model/train_jumps.py run) ----
    dj = _load()
    dj["jump"] = _label(dj, dj["date"] < SPLIT_DATE)
    tj, ej = dj[dj["date"] < SPLIT_DATE], dj[dj["date"] >= SPLIT_DATE]
    bt = ej.dropna(subset=["vol_t"])
    jrows = [{"model": "baseline_vol_t", "roc_auc": roc_auc_score(bt["jump"], bt["vol_t"]),
              "avg_precision": average_precision_score(bt["jump"], bt["vol_t"]),
              "n_test": len(bt), "positives": int(bt["jump"].sum())}]
    for name, feats in FEATURE_SETS.items():
        prob = _clf().fit(tj[feats], tj["jump"]).predict_proba(ej[feats])[:, 1]
        jrows.append({"model": name, "roc_auc": roc_auc_score(ej["jump"], prob),
                      "avg_precision": average_precision_score(ej["jump"], prob),
                      "n_test": len(ej), "positives": int(ej["jump"].sum())})
    _save(pd.DataFrame(jrows), "energy_jumps_single_split.csv")

    # ---- jumps: walk-forward (replicates src/model/train_jumps.py walk_forward) ----
    jwf = []
    for i, cut in enumerate(CUTS, start=1):
        d = dj.copy()
        d["jump"] = _label(d, d["date"] < cut)
        a = d[d["date"] < cut]
        b = d[(d["date"] >= cut) & (d["date"] < cut + HORIZON)]
        if b["jump"].sum() < MIN_JUMPS_ENERGY_FOLD:
            jwf.append({"fold": i, "test_start": str(cut.date()), "n_test": len(b),
                        "positives": int(b["jump"].sum()), "skipped": True})
            continue
        auc = {}
        for name in ("M0 precio", "M2 +EODHD", "M3 +FinBERT"):
            m = _clf().fit(a[FEATURE_SETS[name]], a["jump"])
            auc[name] = roc_auc_score(b["jump"], m.predict_proba(b[FEATURE_SETS[name]])[:, 1])
        jwf.append({"fold": i, "test_start": str(cut.date()), "n_test": len(b),
                    "positives": int(b["jump"].sum()), "skipped": False,
                    "auc_M0": auc["M0 precio"], "auc_M2": auc["M2 +EODHD"],
                    "auc_M3": auc["M3 +FinBERT"],
                    "delta_news_M3_minus_M0": auc["M3 +FinBERT"] - auc["M0 precio"],
                    "delta_finbert_M3_minus_M2": auc["M3 +FinBERT"] - auc["M2 +EODHD"]})
    _save(pd.DataFrame(jwf), "energy_jumps_walkforward.csv")


# --------------------------------------------------------------------------- #
# Energy, jump walk-forward re-scored with macro-AUC
# --------------------------------------------------------------------------- #
def energy_macro() -> None:
    """Energy jump walk-forward measured with macro-AUC instead of pooled ROC-AUC.

    Reason: the sector experiments score with macro-AUC (AUC per asset, then averaged),
    because the jump label is a per-asset quantile. The original energy script scores with
    pooled ROC-AUC. Both are valid, but they are different estimators, so putting them on one
    axis would be wrong. This function recomputes energy with the sector estimator so the
    cross-sector figure compares like with like. The experiment itself is unchanged: same
    panel, same labels, same model, same folds - only the metric differs.
    """
    print("[energy_macro]")
    dj = _load()
    rows = []
    for i, cut in enumerate(CUTS, start=1):
        d = dj.copy()
        d["jump"] = _label(d, d["date"] < cut)
        a = d[d["date"] < cut]
        b = d[(d["date"] >= cut) & (d["date"] < cut + HORIZON)]
        if b["jump"].sum() < MIN_JUMPS_ENERGY_FOLD:
            rows.append({"sector": "energy", "fold": i, "test_start": str(cut.date()),
                         "n_test": len(b), "positives": int(b["jump"].sum()), "skipped": True})
            continue
        au = {}
        for name in ("M0 precio", "M2 +EODHD", "M3 +FinBERT"):
            m = _clf().fit(a[FEATURE_SETS[name]], a["jump"])
            prob = m.predict_proba(b[FEATURE_SETS[name]])[:, 1]
            au[name] = ablation._macro_auc(b["jump"].values, prob, b["asset"].values)
        rows.append({"sector": "energy", "fold": i, "test_start": str(cut.date()),
                     "n_test": len(b), "positives": int(b["jump"].sum()), "skipped": False,
                     "auc_M0": au["M0 precio"], "auc_M2": au["M2 +EODHD"],
                     "auc_M3": au["M3 +FinBERT"],
                     "delta_finbert_M3_minus_M2": au["M3 +FinBERT"] - au["M2 +EODHD"]})
    _save(pd.DataFrame(rows), "energy_jumps_walkforward_macro.csv")


# --------------------------------------------------------------------------- #
# Sectors (uses the ablation module own helpers, unmodified)
# --------------------------------------------------------------------------- #
def sectors() -> None:
    summary, folds = [], []
    for sec in baskets.SECTORS:
        print(f"[sectors] {sec}")
        df = panel.build_panel(sec)
        sets = ablation._sets(df)
        has_fb = "M3" in sets
        tr = df[df["date"] < ablation.SPLIT_DATE]
        te = df[df["date"] >= ablation.SPLIT_DATE]

        pv = te.dropna(subset=["vol_t"])
        row = {"sector": sec, "n_rows": len(df), "n_assets": df["asset"].nunique(),
               "finbert_available": has_fb,
               "level_r2_persistence": r2_score(pv["target"], pv["vol_t"])}
        for n, f in sets.items():
            row[f"level_r2_{n}"] = r2_score(
                te["target"], ablation._reg().fit(tr[f], tr["target"]).predict(te[f]))

        dj = df.copy()
        dj["jump"] = ablation._jump_label(dj, dj["date"] < ablation.SPLIT_DATE)
        tj = dj[dj["date"] < ablation.SPLIT_DATE]
        ej = dj[dj["date"] >= ablation.SPLIT_DATE]
        row["jump_macro_auc_baseline_vol_t"] = ablation._macro_auc(
            ej["jump"].values, ej["vol_t"].values, ej["asset"].values)
        for n, f in sets.items():
            row[f"jump_macro_auc_{n}"] = ablation._fit_auc(tj, ej, f)
        summary.append(row)

        for i, cut in enumerate(ablation.CUTS, start=1):
            d = df.copy()
            d["jump"] = ablation._jump_label(d, d["date"] < cut)
            a = d[d["date"] < cut]
            b = d[(d["date"] >= cut) & (d["date"] < cut + ablation.HORIZON)]
            if b["jump"].sum() < MIN_JUMPS_SECTOR_FOLD:
                folds.append({"sector": sec, "fold": i, "test_start": str(cut.date()),
                              "n_test": len(b), "positives": int(b["jump"].sum()),
                              "skipped": True})
                continue
            au = {n: ablation._fit_auc(a, b, sets[n]) for n in sets}
            rec = {"sector": sec, "fold": i, "test_start": str(cut.date()),
                   "n_test": len(b), "positives": int(b["jump"].sum()), "skipped": False,
                   "auc_M0": au["M0"], "auc_M1": au["M1"], "auc_M2": au["M2"],
                   "delta_volume_M1_minus_M0": au["M1"] - au["M0"],
                   "delta_tone_M2_minus_M1": au["M2"] - au["M1"]}
            if has_fb:
                rec["auc_M3"] = au["M3"]
                rec["delta_finbert_M3_minus_M2"] = au["M3"] - au["M2"]
            folds.append(rec)

    _save(pd.DataFrame(summary), "sectors_single_split.csv")
    _save(pd.DataFrame(folds), "sectors_walkforward.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("counts", "all"):
        counts()
    if cmd in ("energy", "all"):
        energy()
    if cmd in ("energy_macro", "all"):
        energy_macro()
    if cmd in ("sectors", "all"):
        sectors()
