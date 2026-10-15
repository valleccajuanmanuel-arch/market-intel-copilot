"""Matriz de features para el modelo de volatilidad (Módulo 3).

Panel por (asset, trading_day) desde 2020: features de **precio** (vol rezagada,
retornos — la volatilidad es autocorrelada, este es el baseline a batir) +
features de **noticias** (volumen, sentimiento EODHD y FinBERT). Target = realized
vol del día siguiente (t+1), PiT-limpio.

Los bloques de features se exponen como listas para el ablation en `train.py`.
"""

from __future__ import annotations

import pandas as pd

from src import config

VOL_WINDOW = 5
CORPUS_START = pd.Timestamp("2020-01-01", tz="UTC")

# Bloques de features (para el ablation incremental).
PRICE_FEATURES = ["vol_t", "abs_ret_t", "ret_t", "vol_lag5", "abs_ret_lag1"]
NEWS_VOLUME = ["n_articles"]
EODHD_FEATURES = ["eodhd_mean"]
FINBERT_FEATURES = [
    "finbert_title_mean", "finbert_body_mean", "finbert_body_std",
    "body_pos_minus_neg_ratio", "abs_body",
]


def build(vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    prices = pd.read_parquet(config.PRICES_DAILY_PARQUET)
    sent = pd.read_parquet(config.SENTIMENT_DAILY_PARQUET)

    # Features de precio/volatilidad (sobre toda la historia, para que los lags
    # en el borde 2020 sean válidos).
    p = prices[["date", "asset", "log_return"]].sort_values(["asset", "date"]).copy()
    p["ret_t"] = p["log_return"]
    p["abs_ret_t"] = p["log_return"].abs()
    p["vol_t"] = p.groupby("asset", observed=True)["log_return"].transform(
        lambda s: s.rolling(vol_window).std()
    )
    p["vol_lag5"] = p.groupby("asset", observed=True)["vol_t"].shift(vol_window)
    p["abs_ret_lag1"] = p.groupby("asset", observed=True)["abs_ret_t"].shift(1)
    p["target"] = p.groupby("asset", observed=True)["vol_t"].shift(-1)  # vol de t+1
    p["absret_next"] = p.groupby("asset", observed=True)["abs_ret_t"].shift(-1)  # |ret| de t+1 (target de salto)

    p = p[p["date"] >= CORPUS_START].copy()

    # Features de noticias (left join: días sin noticias -> volumen 0, sentimiento NaN).
    s = sent.rename(columns={"trading_day": "date"}).copy()
    s["abs_body"] = s["finbert_body_mean"].abs()
    news_cols = ["date", "asset"] + NEWS_VOLUME + EODHD_FEATURES + FINBERT_FEATURES
    df = p.merge(s[news_cols], on=["date", "asset"], how="left")
    df["n_articles"] = df["n_articles"].fillna(0)  # sin noticias = 0 artículos

    # El target del último día por activo es NaN (no hay t+1) -> se descarta.
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df
