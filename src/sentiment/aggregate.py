"""Agrega el sentimiento por artículo a features diarias por (trading_day, asset).

Junta tres señales por artículo (unidas al corpus por `url`):
- **FinBERT título** (`title_score`) y **FinBERT cuerpo** (`body_score`)  ← el diferencial.
- **EODHD `polarity`** (sentimiento propio de EODHD)                      ← el baseline.

y las agrega a una fila por `(trading_day, asset)` → `sentiment_daily.parquet`.
Estas son las features de sentimiento del día `t`; el Módulo 3 las alinea con la
volatilidad de `t+1` (nunca del mismo día — disciplina PiT heredada del Módulo 1).
"""

from __future__ import annotations

import pandas as pd  # aggregate no usa torch → no hay conflicto de orden

from src import config
from src.ingestion import eodhd_news


def _eodhd_sentiment() -> pd.DataFrame:
    """url -> `polarity` de EODHD (baseline). Sólo para artículos de EODHD."""
    rows: list[dict] = []
    for f in sorted(config.RAW_NEWS_EODHD_DIR.glob("*.json")):
        for a in eodhd_news._read_raw(f):
            s = a.get("sentiment") or {}
            rows.append({"url": a.get("link") or "", "eodhd_polarity": s.get("polarity")})
    return pd.DataFrame(rows).dropna(subset=["url"]).drop_duplicates("url")


def build_daily() -> pd.DataFrame:
    corpus = pd.read_parquet(
        config.NEWS_CORPUS_PARQUET, columns=["url", "trading_day", "asset", "source_quality"]
    )
    art = pd.read_parquet(config.SENTIMENT_ARTICLES_PARQUET)  # url, title/body _label/_score
    eod = _eodhd_sentiment()

    # Unir señales a cada fila del corpus (por url).
    df = (corpus
          .merge(art[["url", "title_score", "body_score", "body_label"]], on="url", how="left")
          .merge(eod, on="url", how="left"))

    # Agregado por (trading_day, asset).
    g = df.groupby(["trading_day", "asset"], observed=True)
    daily = g.agg(
        n_articles=("url", "size"),
        finbert_title_mean=("title_score", "mean"),
        finbert_body_mean=("body_score", "mean"),
        finbert_body_std=("body_score", "std"),
        eodhd_mean=("eodhd_polarity", "mean"),
        n_body_pos=("body_label", lambda s: (s == "positive").sum()),
        n_body_neg=("body_label", lambda s: (s == "negative").sum()),
    ).reset_index()
    daily["body_pos_minus_neg_ratio"] = (
        (daily["n_body_pos"] - daily["n_body_neg"]) / daily["n_articles"]
    )

    # Variante sólo fuentes trusted (por el sesgo a PR — feature alternativa).
    tr = (df[df["source_quality"] == "trusted"]
          .groupby(["trading_day", "asset"], observed=True)
          .agg(n_trusted=("url", "size"),
               finbert_body_mean_trusted=("body_score", "mean"))
          .reset_index())
    daily = daily.merge(tr, on=["trading_day", "asset"], how="left")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(config.SENTIMENT_DAILY_PARQUET, index=False)
    return daily


if __name__ == "__main__":
    daily = build_daily()
    print(f"sentiment_daily: {daily.shape} -> {config.SENTIMENT_DAILY_PARQUET}")
    print("\nPor activo (filas = días con noticias):")
    print(daily.groupby("asset", observed=True).size().to_string())
    print("\nRango:", daily["trading_day"].min().date(), "->", daily["trading_day"].max().date())
    print("\nCorrelación de las 3 señales diarias (title/body/eodhd):")
    print(daily[["finbert_title_mean", "finbert_body_mean", "eodhd_mean"]].corr().round(3).to_string())
    print("\nMuestra:")
    print(daily.head(4).to_string())
