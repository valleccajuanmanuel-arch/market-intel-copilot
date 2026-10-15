"""Dataset supervisado: features de sentimiento(t) → volatilidad(t+1).

Alinea `sentiment_daily.parquet` (features del día `t`) con la volatilidad del
**día de trading siguiente** (`t+1`), respetando disciplina Point-in-Time: el
target es SIEMPRE estrictamente futuro (nunca el mismo día `t`).

Volatilidad (proxy) por activo, del `log_return` de `prices_daily`:
- `abs_ret`  = |log_return|                (movimiento absoluto del día)
- `rvol`     = std móvil de `log_return`    (realized vol, ventana configurable)

El target es la volatilidad de `t+1`: `target_abs_ret`, `target_rvol`.
"""

from __future__ import annotations

import pandas as pd

from src import config


def _volatility(prices: pd.DataFrame, roll: int) -> pd.DataFrame:
    """Por activo: |log_return| y realized vol (std móvil)."""
    v = prices[["date", "asset", "log_return"]].sort_values(["asset", "date"]).copy()
    v["abs_ret"] = v["log_return"].abs()
    v["rvol"] = v.groupby("asset", observed=True)["log_return"].transform(
        lambda s: s.rolling(roll).std()
    )
    return v


def build_supervised(roll: int = 5) -> pd.DataFrame:
    """Une sentimiento(t) con la volatilidad de t+1. Una fila por (t, asset)."""
    sent = pd.read_parquet(config.SENTIMENT_DAILY_PARQUET)
    prices = pd.read_parquet(config.PRICES_DAILY_PARQUET)

    vol = _volatility(prices, roll).sort_values(["asset", "date"])
    # Target = volatilidad del PRÓXIMO día de trading (shift -1 por activo).
    grp = vol.groupby("asset", observed=True)
    vol["target_abs_ret"] = grp["abs_ret"].shift(-1)
    vol["target_rvol"] = grp["rvol"].shift(-1)
    vol["target_date"] = grp["date"].shift(-1)

    df = sent.merge(
        vol[["date", "asset", "target_abs_ret", "target_rvol", "target_date"]],
        left_on=["trading_day", "asset"],
        right_on=["date", "asset"],
        how="inner",
    ).drop(columns="date")
    return df


# Señales de sentimiento a evaluar como predictoras.
SIGNAL_COLS = [
    "finbert_title_mean",
    "finbert_body_mean",
    "finbert_body_mean_trusted",
    "eodhd_mean",
    "body_pos_minus_neg_ratio",
    "finbert_body_std",
]
