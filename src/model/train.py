"""Súper test (Módulo 3): ¿FinBERT aporta sobre el baseline para predecir volatilidad?

Ablation incremental con validación **temporal** (train pasado, test futuro — sin
look-ahead), sobre `HistGradientBoostingRegressor` (no-lineal, capta interacciones):
    M0 precio  ->  M1 +volumen  ->  M2 +EODHD  ->  M3 +FinBERT

Reporta R2 out-of-sample, RMSE, rank-corr, el ΔR2 de cada bloque, la importancia
por permutación en M3 (¿el modelo usa FinBERT?), y R2 por activo. Compara contra
un baseline de **persistencia** (predecir la vol de mañana = la de hoy).

Correr:  python -m src.model.train
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from src.model import features

SPLIT_DATE = pd.Timestamp("2024-06-01", tz="UTC")  # train < split, test >= split

FEATURE_SETS = {
    "M0 precio": features.PRICE_FEATURES,
    "M1 +volumen": features.PRICE_FEATURES + features.NEWS_VOLUME,
    "M2 +EODHD": features.PRICE_FEATURES + features.NEWS_VOLUME + features.EODHD_FEATURES,
    "M3 +FinBERT": (features.PRICE_FEATURES + features.NEWS_VOLUME
                    + features.EODHD_FEATURES + features.FINBERT_FEATURES),
}


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=0,
    )


def run(split: pd.Timestamp = SPLIT_DATE) -> None:
    df = features.build()
    train = df[df["date"] < split]
    test = df[df["date"] >= split]
    print(f"train: {len(train)} filas (< {split.date()}) | test: {len(test)} (>= {split.date()})")

    # Baseline de persistencia: vol de mañana = vol de hoy (sobre filas con vol_t válido).
    pv = test.dropna(subset=["vol_t"])
    pers_r2 = r2_score(pv["target"], pv["vol_t"])
    print(f"\nBaseline PERSISTENCIA (target = vol_hoy):  R2_oos = {pers_r2:+.4f}  (n={len(pv)})")

    print(f"\n{'Modelo':<14}{'R2_oos':>10}{'RMSE':>12}{'rankcorr':>10}")
    results: dict[str, tuple] = {}
    for name, feats in FEATURE_SETS.items():
        m = _model()
        m.fit(train[feats], train["target"])
        pred = m.predict(test[feats])
        r2v = r2_score(test["target"], pred)
        rmse = mean_squared_error(test["target"], pred) ** 0.5
        rc = spearmanr(test["target"], pred).statistic
        results[name] = (m, feats, pred, r2v)
        print(f"{name:<14}{r2v:>10.4f}{rmse:>12.5f}{rc:>10.3f}")

    print("\nDelta-R2 al agregar cada bloque:")
    names = list(FEATURE_SETS)
    for i in range(1, len(names)):
        d = results[names[i]][3] - results[names[i - 1]][3]
        print(f"  {names[i]:<14} vs {names[i - 1]:<12}: {d:+.5f}")

    # ¿El modelo usa FinBERT? Importancia por permutación en M3.
    m3, feats3, _, _ = results["M3 +FinBERT"]
    imp = permutation_importance(m3, test[feats3], test["target"], n_repeats=10, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nImportancia (permutación, OOS) en M3 — features ordenadas:")
    for j in order:
        tag = "  <- FinBERT" if feats3[j] in features.FINBERT_FEATURES else ""
        print(f"  {feats3[j]:<26} {imp.importances_mean[j]:+.5f}{tag}")

    # R2 out-of-sample por activo (modelo M3).
    test2 = test.copy()
    test2["pred"] = results["M3 +FinBERT"][2]
    print("\nR2_oos por activo (M3):")
    for a in ("WTI", "BRENT", "HENRY_HUB"):
        sub = test2[test2["asset"] == a]
        if len(sub) > 5:
            print(f"  {a:<10} {r2_score(sub['target'], sub['pred']):+.4f}  (n={len(sub)})")


def walk_forward() -> None:
    """Validación robusta: varios folds expansivos (train pasado -> test 6 meses).

    Si FinBERT aporta señal real, el ΔR2(M3 - M0) debe ser positivo *consistente*
    fold a fold; si cambia de signo, era ruido del split único.
    """
    df = features.build()
    cuts = [pd.Timestamp(c, tz="UTC") for c in (
        "2023-07-01", "2024-01-01", "2024-07-01",
        "2025-01-01", "2025-07-01", "2026-01-01",
    )]
    horizon = pd.DateOffset(months=6)

    print("\n" + "=" * 60)
    print("WALK-FORWARD (expansivo, test = 6 meses)")
    print(f"\n{'fold (test desde)':<20}{'n':>6}{'pers':>8}{'M0':>8}{'M3':>8}{'d(M3-M0)':>10}")
    deltas = []
    for cut in cuts:
        tr = df[df["date"] < cut]
        te = df[(df["date"] >= cut) & (df["date"] < cut + horizon)]
        if len(te) < 30:
            continue
        pv = te.dropna(subset=["vol_t"])
        pers = r2_score(pv["target"], pv["vol_t"])
        r2 = {}
        for name in ("M0 precio", "M3 +FinBERT"):
            m = _model()
            m.fit(tr[FEATURE_SETS[name]], tr["target"])
            r2[name] = r2_score(te["target"], m.predict(te[FEATURE_SETS[name]]))
        d = r2["M3 +FinBERT"] - r2["M0 precio"]
        deltas.append(d)
        print(f"{str(cut.date()):<20}{len(te):>6}{pers:>8.3f}"
              f"{r2['M0 precio']:>8.3f}{r2['M3 +FinBERT']:>8.3f}{d:>+10.4f}")

    arr = np.array(deltas)
    pos = int((arr > 0).sum())
    print(f"\ndR2(M3-M0): media {arr.mean():+.4f} | mediana {np.median(arr):+.4f} "
          f"| positivo en {pos}/{len(arr)} folds")


if __name__ == "__main__":
    run()
    walk_forward()
