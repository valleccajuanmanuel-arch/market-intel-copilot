"""Construye el panel (asset, día) por sector desde los crudos EODHD.

Mismo esquema de features que `model/features.py` (precio: vol rezagada + retornos;
noticias: volumen + sentimiento EODHD + FinBERT-título opcional) y mismos targets
PiT: vol realizada de t+1 y |ret| de t+1 (target de salto). Cada ticker es un activo.

Mapeo trading-day: cada noticia se asigna al **primer día de trading >= su fecha**
(fin de semana/feriado -> próximo día hábil). Como el target es t+1, no hay leak.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.ingestion import eodhd_news as en
from src.sectors import baskets

VOL_WINDOW = 5

PRICE_FEATURES = ["vol_t", "abs_ret_t", "ret_t", "vol_lag5", "abs_ret_lag1"]
NEWS_VOLUME = ["n_articles"]
EODHD_FEATURES = ["eodhd_mean"]
FINBERT_FEATURES = ["finbert_title_mean"]


def _load_prices(sector: str) -> pd.DataFrame:
    frames = []
    for ticker in baskets.SECTORS[sector]:
        path = baskets.prices_dir(sector) / f"{en._safe(ticker)}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")  # ajustado por splits
        df = (df[["date", "close"]].dropna().drop_duplicates("date")
              .sort_values("date").reset_index(drop=True))
        df["asset"] = baskets.asset_of(ticker)
        ratio = df["close"] / df["close"].shift(1)
        df["log_return"] = np.log(ratio.where(ratio > 0))
        frames.append(df[["date", "asset", "log_return"]])
    return pd.concat(frames, ignore_index=True)


def _news_daily_from_sentiments(sector: str, prices: pd.DataFrame) -> pd.DataFrame:
    """Sentimiento diario agregado de EODHD (count, normalized) -> (asset, date).

    `count` -> n_articles (volumen); `normalized` -> eodhd_mean (tono). Cada día se
    mapea al primer día de trading >= su fecha (fin de semana/feriado -> próximo
    hábil), consistente con el mapeo PiT del resto del panel.
    """
    sdir = baskets.SECTORS_RAW / sector / "sentiments"
    recs = []
    for f in sorted(sdir.glob("*.json")):
        asset = f.stem.split("-")[0]
        for d in en._read_raw(f):
            recs.append({"asset": asset, "cal_date": d.get("date"),
                         "n_articles": d.get("count"), "eodhd_mean": d.get("normalized")})
    df = pd.DataFrame(recs)
    df["cal_date"] = pd.to_datetime(df["cal_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["cal_date"])

    out = []
    for asset, g in df.groupby("asset"):
        tdays = np.sort(prices.loc[prices["asset"] == asset, "date"].unique())
        if len(tdays) == 0:
            continue
        idx = np.searchsorted(tdays, g["cal_date"].to_numpy(), side="left")
        valid = idx < len(tdays)
        gg = g.loc[valid].copy()
        gg["date"] = tdays[idx[valid]]
        out.append(gg)
    mapped = pd.concat(out, ignore_index=True)
    # Varias fechas-calendario pueden caer en el mismo día de trading (fin de semana).
    return (mapped.groupby(["asset", "date"])
            .agg(n_articles=("n_articles", "sum"),
                 eodhd_mean=("eodhd_mean", "mean")).reset_index())


def build_panel(sector: str, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    prices = _load_prices(sector)
    nd = _news_daily_from_sentiments(sector, prices)

    # FinBERT-título opcional (Fase 2): parquet diario (asset, date, finbert_title_mean).
    tp = baskets.titles_scored_path(sector)
    has_fb = tp.exists()
    if has_fb:
        fb = pd.read_parquet(tp)  # asset, date, finbert_title_mean
        nd = nd.merge(fb, on=["asset", "date"], how="left")

    # Features de precio/vol + targets (mismo diseño que model/features.py).
    p = prices.sort_values(["asset", "date"]).copy()
    p["ret_t"] = p["log_return"]
    p["abs_ret_t"] = p["log_return"].abs()
    p["vol_t"] = p.groupby("asset")["log_return"].transform(lambda s: s.rolling(vol_window).std())
    p["vol_lag5"] = p.groupby("asset")["vol_t"].shift(vol_window)
    p["abs_ret_lag1"] = p.groupby("asset")["abs_ret_t"].shift(1)
    p["target"] = p.groupby("asset")["vol_t"].shift(-1)          # vol realizada t+1
    p["absret_next"] = p.groupby("asset")["abs_ret_t"].shift(-1)  # |ret| t+1 (salto)

    df = p.merge(nd, on=["date", "asset"], how="left")
    df["n_articles"] = df["n_articles"].fillna(0)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df.attrs["has_finbert"] = has_fb
    return df


if __name__ == "__main__":
    for sec in baskets.SECTORS:
        d = build_panel(sec)
        na = d["n_articles"].gt(0).mean()
        print(f"{sec:<12} filas={len(d):>6} activos={d['asset'].nunique()} "
              f"dias_con_news={na:.0%} finbert={d.attrs['has_finbert']}")
