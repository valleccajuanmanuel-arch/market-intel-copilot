"""Test de saltos (Módulo 3): ¿FinBERT ayuda a *anticipar shocks* de precio?

Complemento del test de nivel (`train.py`): en vez de predecir el nivel suave de
volatilidad —donde la persistencia es imbatible— acá clasificamos el evento que la
persistencia NO ve: un **día de shock** = |retorno de t+1| en el decil superior.
Ese target NO está mecánicamente determinado por la vol de hoy, así que es el
escenario donde noticias/FinBERT tienen su mejor chance de aportar.

Mismo panel de features y mismo ablation M0->M3, pero con clasificador y métricas
de ranking (ROC-AUC y average-precision, robustas al desbalance ~10% positivos).
Baseline análogo a persistencia: rankear por `vol_t` (la vol de hoy).

Correr:  python -m src.model.train_jumps
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model import features
from src.model.train import FEATURE_SETS, SPLIT_DATE

TOP_Q = 0.90  # salto = |ret_{t+1}| en el decil superior (umbral por activo, SOLO train)


def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=0,
    )


def _load() -> pd.DataFrame:
    df = features.build()
    return df.dropna(subset=["absret_next"]).reset_index(drop=True)


def _label(df: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    """Etiqueta de salto con umbral top-decil calculado SOLO sobre train (PiT)."""
    thr = df[train_mask].groupby("asset", observed=True)["absret_next"].quantile(TOP_Q)
    cutoff = df["asset"].astype(str).map(thr).astype(float)
    return (df["absret_next"] >= cutoff).astype(int)


def run(split: pd.Timestamp = SPLIT_DATE) -> None:
    df = _load()
    df["jump"] = _label(df, df["date"] < split)
    train = df[df["date"] < split]
    test = df[df["date"] >= split]
    print(f"train: {len(train)} | test: {len(test)} | positivos test: "
          f"{test['jump'].mean():.1%}")

    # Baseline "persistencia": rankear los saltos por la vol de hoy (vol_t).
    bt = test.dropna(subset=["vol_t"])
    print(f"\nBaseline VOL_T (rankear por vol de hoy):  "
          f"AUC={roc_auc_score(bt['jump'], bt['vol_t']):.3f}  "
          f"AP={average_precision_score(bt['jump'], bt['vol_t']):.3f}  (n={len(bt)})")

    print(f"\n{'Modelo':<14}{'ROC-AUC':>10}{'avg-prec':>10}")
    results: dict[str, tuple] = {}
    for name, feats in FEATURE_SETS.items():
        m = _clf()
        m.fit(train[feats], train["jump"])
        prob = m.predict_proba(test[feats])[:, 1]
        auc = roc_auc_score(test["jump"], prob)
        ap = average_precision_score(test["jump"], prob)
        results[name] = (m, feats, auc)
        print(f"{name:<14}{auc:>10.3f}{ap:>10.3f}")

    print("\nDelta-AUC al agregar cada bloque:")
    names = list(FEATURE_SETS)
    for i in range(1, len(names)):
        d = results[names[i]][2] - results[names[i - 1]][2]
        print(f"  {names[i]:<14} vs {names[i - 1]:<12}: {d:+.4f}")

    # ¿El clasificador usa FinBERT? Importancia por permutación (scoring ROC-AUC).
    m3, feats3, _ = results["M3 +FinBERT"]
    imp = permutation_importance(m3, test[feats3], test["jump"],
                                 scoring="roc_auc", n_repeats=10, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nImportancia (permutacion, ROC-AUC OOS) en M3:")
    for j in order:
        tag = "  <- FinBERT" if feats3[j] in features.FINBERT_FEATURES else ""
        print(f"  {feats3[j]:<26} {imp.importances_mean[j]:+.4f}{tag}")


def walk_forward() -> None:
    """Robustez: ¿el dAUC(M3-M0) es positivo consistente fold a fold, o ruido?"""
    df = _load()
    cuts = [pd.Timestamp(c, tz="UTC") for c in (
        "2023-07-01", "2024-01-01", "2024-07-01",
        "2025-01-01", "2025-07-01", "2026-01-01",
    )]
    horizon = pd.DateOffset(months=6)

    print("\n" + "=" * 60)
    print("WALK-FORWARD (expansivo, test = 6 meses) -- ROC-AUC")
    print(f"\n{'fold (test desde)':<18}{'n':>5}{'M0':>7}{'M2':>7}{'M3':>7}"
          f"{'news':>9}{'finbert':>9}")
    print(f"{'':18}{'':5}{'':7}{'':7}{'':7}{'(M3-M0)':>9}{'(M3-M2)':>9}")
    d_news, d_fb = [], []
    for cut in cuts:
        d = df.copy()
        d["jump"] = _label(d, d["date"] < cut)
        tr = d[d["date"] < cut]
        te = d[(d["date"] >= cut) & (d["date"] < cut + horizon)]
        if te["jump"].sum() < 10:
            continue
        auc = {}
        for name in ("M0 precio", "M2 +EODHD", "M3 +FinBERT"):
            m = _clf()
            m.fit(tr[FEATURE_SETS[name]], tr["jump"])
            auc[name] = roc_auc_score(te["jump"], m.predict_proba(te[FEATURE_SETS[name]])[:, 1])
        dn = auc["M3 +FinBERT"] - auc["M0 precio"]
        df_ = auc["M3 +FinBERT"] - auc["M2 +EODHD"]
        d_news.append(dn)
        d_fb.append(df_)
        print(f"{str(cut.date()):<18}{len(te):>5}{auc['M0 precio']:>7.3f}"
              f"{auc['M2 +EODHD']:>7.3f}{auc['M3 +FinBERT']:>7.3f}{dn:>+9.4f}{df_:>+9.4f}")

    an, af = np.array(d_news), np.array(d_fb)
    print(f"\nnoticias (M3-M0):  media {an.mean():+.4f} | mediana {np.median(an):+.4f} "
          f"| positivo {int((an > 0).sum())}/{len(an)}")
    print(f"FinBERT  (M3-M2):  media {af.mean():+.4f} | mediana {np.median(af):+.4f} "
          f"| positivo {int((af > 0).sum())}/{len(af)}")


if __name__ == "__main__":
    run()
    walk_forward()
