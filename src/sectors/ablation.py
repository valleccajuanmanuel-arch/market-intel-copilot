"""Ablation multi-sector (nivel + saltos) con walk-forward, por sector.

Reusa el diseño de `model/train.py`/`train_jumps.py` parametrizado por sector.
Para SALTOS usa **macro-AUC** (AUC por ticker, promediada): las labels de salto
son top-decil *relativo a cada ticker*, así que una AUC pooled con tickers de
vol muy distinta (TSLA vs AAPL) sería engañosa.

Aísla los bloques: volumen (M1-M0), sentimiento EODHD (M2-M1) y FinBERT (M3-M2),
cada uno validado en walk-forward. Correr: `python -m src.sectors.ablation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import r2_score, roc_auc_score

from src.sectors import baskets, panel

SPLIT_DATE = pd.Timestamp("2024-06-01", tz="UTC")
CUTS = [pd.Timestamp(c, tz="UTC") for c in (
    "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01")]
HORIZON = pd.DateOffset(months=6)
TOP_Q = 0.90


def _reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                         l2_regularization=1.0, random_state=0)


def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                          l2_regularization=1.0, random_state=0)


def _sets(df: pd.DataFrame) -> dict[str, list[str]]:
    m0 = panel.PRICE_FEATURES
    m1 = m0 + panel.NEWS_VOLUME
    m2 = m1 + panel.EODHD_FEATURES
    sets = {"M0": m0, "M1": m1, "M2": m2}
    if df.attrs.get("has_finbert"):
        sets["M3"] = m2 + panel.FINBERT_FEATURES
    return sets


def _jump_label(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    thr = df[train_mask].groupby("asset")["absret_next"].quantile(TOP_Q)
    cutoff = df["asset"].map(thr).astype(float)
    return (df["absret_next"] >= cutoff).astype(int)


def _macro_auc(y, score, asset) -> float:
    """AUC por ticker (ignora NaN en score), promediada. Nan si nada evaluable."""
    y, score, asset = np.asarray(y), np.asarray(score, float), np.asarray(asset)
    ok = ~np.isnan(score)
    aucs = []
    for a in np.unique(asset):
        m = (asset == a) & ok
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], score[m]))
    return float(np.mean(aucs)) if aucs else np.nan


def _fit_auc(train, test, feats) -> float:
    m = _clf().fit(train[feats], train["jump"])
    return _macro_auc(test["jump"].values, m.predict_proba(test[feats])[:, 1], test["asset"].values)


def run_sector(sector: str) -> dict:
    df = panel.build_panel(sector)
    sets = _sets(df)
    has_fb = "M3" in sets
    tr, te = df[df["date"] < SPLIT_DATE], df[df["date"] >= SPLIT_DATE]

    # ---------- NIVEL de volatilidad (regresión, pooled — sanity check) ----------
    pv = te.dropna(subset=["vol_t"])
    pers = r2_score(pv["target"], pv["vol_t"])
    r2 = {n: r2_score(te["target"], _reg().fit(tr[f], tr["target"]).predict(te[f]))
          for n, f in sets.items()}

    # ---------- SALTOS (clasificación, macro-AUC) split único ----------
    dj = df.copy()
    dj["jump"] = _jump_label(dj, dj["date"] < SPLIT_DATE)
    tj, ej = dj[dj["date"] < SPLIT_DATE], dj[dj["date"] >= SPLIT_DATE]
    base = _macro_auc(ej["jump"].values, ej["vol_t"].values, ej["asset"].values)
    auc = {n: _fit_auc(tj, ej, f) for n, f in sets.items()}

    # ---------- WALK-FORWARD: deltas por bloque ----------
    d_vol, d_eod, d_fb = [], [], []
    for cut in CUTS:
        d = df.copy()
        d["jump"] = _jump_label(d, d["date"] < cut)
        a = d[d["date"] < cut]
        b = d[(d["date"] >= cut) & (d["date"] < cut + HORIZON)]
        if b["jump"].sum() < 20:
            continue
        av = {n: _fit_auc(a, b, sets[n]) for n in sets}
        d_vol.append(av["M1"] - av["M0"])
        d_eod.append(av["M2"] - av["M1"])
        if has_fb:
            d_fb.append(av["M3"] - av["M2"])

    def _s(x):
        arr = np.array(x, float)
        return arr.mean(), int((arr > 0).sum()), len(arr)

    vm, vp, vn = _s(d_vol)
    em, ep, en_ = _s(d_eod)
    fm, fp, fn = _s(d_fb) if d_fb else (np.nan, 0, 0)

    print(f"\n===== {sector.upper()} =====  (filas {len(df)}, activos {df['asset'].nunique()}, "
          f"finbert={has_fb})")
    print("  NIVEL vol : persist R2 {:+.3f} | M0 {:+.3f} | M2 {:+.3f}{}".format(
        pers, r2["M0"], r2["M2"], " | M3 {:+.3f}".format(r2["M3"]) if has_fb else ""))
    print("  SALTOS macro-AUC : base {:.3f} | M0 {:.3f} | M1 {:.3f} | M2 {:.3f}{}".format(
        base, auc["M0"], auc["M1"], auc["M2"], " | M3 {:.3f}".format(auc["M3"]) if has_fb else ""))
    print("  WF dAUC  volumen (M1-M0): {:+.4f}  pos {}/{}".format(vm, vp, vn))
    print("  WF dAUC  EODHD   (M2-M1): {:+.4f}  pos {}/{}".format(em, ep, en_))
    if has_fb:
        print("  WF dAUC  FinBERT (M3-M2): {:+.4f}  pos {}/{}".format(fm, fp, fn))

    return {"sector": sector, "finbert": has_fb,
            "auc_base": base, "auc_M0": auc["M0"], "auc_M1": auc["M1"],
            "auc_M2": auc["M2"], "auc_M3": auc.get("M3", np.nan),
            "vol_mean": vm, "vol_pos": f"{vp}/{vn}",
            "eodhd_mean": em, "eodhd_pos": f"{ep}/{en_}",
            "fb_mean": fm, "fb_pos": f"{fp}/{fn}" if has_fb else "-"}


def run_all() -> pd.DataFrame:
    rows = [run_sector(s) for s in baskets.SECTORS]
    summ = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("RESUMEN CROSS-SECTOR (saltos, macro-AUC; deltas = aporte de cada bloque en WF)")
    cols = ["sector", "auc_base", "auc_M0", "auc_M2", "auc_M3",
            "vol_mean", "vol_pos", "eodhd_mean", "eodhd_pos", "fb_mean", "fb_pos"]
    print(summ[cols].round(4).to_string(index=False))
    return summ


if __name__ == "__main__":
    run_all()
