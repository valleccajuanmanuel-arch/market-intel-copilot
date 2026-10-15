"""Scorea el corpus completo con FinBERT (título + cuerpo). Batched y resumible.

Estrategia:
- Se scorea a nivel de **artículo único** (por `url`), no las filas explotadas del
  corpus (WTI/Brent comparten el mismo artículo de oil). El join a
  `(trading_day, asset)` lo hace `aggregate.py`.
- Se procesa en **chunks**; cada chunk se guarda en
  `data/interim/sentiment_chunks/chunk_NNNN.parquet`. Resumible: si el chunk ya
  existe, se saltea. Al final se consolidan en `sentiment_articles.parquet`.
- Por artículo se guardan dos scores: título y cuerpo (`*_label`, `*_score`).

Correr:  python -m src.sentiment.score_corpus   (retoma donde quedó)
"""

from __future__ import annotations

from src.sentiment import finbert  # importa torch primero (fix Windows)

import pandas as pd

from src import config

CHUNK_SIZE = 2000
CHUNK_DIR = config.INTERIM_DIR / "sentiment_chunks"
# Tope de tokens por texto (CPU): títulos son cortos; cuerpos se truncan al lead.
TITLE_MAX = 128
BODY_MAX = 256


def _unique_articles() -> pd.DataFrame:
    """Artículos únicos del corpus (por url), orden determinístico para el resume."""
    news = pd.read_parquet(config.NEWS_CORPUS_PARQUET, columns=["url", "title", "body"])
    return (news.drop_duplicates("url")
                .sort_values("url")
                .reset_index(drop=True))


def score_corpus(chunk_size: int = CHUNK_SIZE) -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    uniq = _unique_articles()
    n = len(uniq)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"Articulos unicos: {n} | chunks: {n_chunks} (x {chunk_size})", flush=True)

    for i in range(n_chunks):
        path = CHUNK_DIR / f"chunk_{i:04d}.parquet"
        if path.exists():
            continue  # resume: chunk ya scoreado
        part = uniq.iloc[i * chunk_size:(i + 1) * chunk_size].reset_index(drop=True)
        ts = finbert.score_texts(part["title"].tolist(), max_length=TITLE_MAX)
        bs = finbert.score_texts(part["body"].tolist(), max_length=BODY_MAX)
        out = pd.DataFrame({
            "url": part["url"].to_numpy(),
            "title_label": ts["label"].to_numpy(),
            "title_score": ts["sentiment_score"].to_numpy(),
            "body_label": bs["label"].to_numpy(),
            "body_score": bs["sentiment_score"].to_numpy(),
        })
        out.to_parquet(path, index=False)
        print(f"chunk {i + 1}/{n_chunks}: {len(out)} scoreados", flush=True)


def consolidate() -> pd.DataFrame:
    """Junta todos los chunks en `sentiment_articles.parquet`."""
    files = sorted(CHUNK_DIR.glob("chunk_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SENTIMENT_ARTICLES_PARQUET, index=False)
    return df


if __name__ == "__main__":
    score_corpus()
    df = consolidate()
    print(f"\nListo: {len(df)} articulos -> {config.SENTIMENT_ARTICLES_PARQUET}")
    print("\nLabels (titulo):")
    print(df["title_label"].value_counts().to_string())
    print("\nLabels (cuerpo):")
    print(df["body_label"].value_counts().to_string())
