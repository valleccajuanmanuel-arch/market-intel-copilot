"""FinBERT sobre títulos por sector, con MUESTREO por día (Fase 2 multi-sector).

Los mega-caps tienen cientos de miles de noticias; scorearlas todas es horas de
CPU inútil. Para el promedio diario de sentimiento alcanza con muestrear ~K
títulos por (ticker, día): la media diaria queda bien estimada y el volumen cae
un orden de magnitud. Titles-only (MAX_LEN=64), chunked/resumible.

Produce `{sector}_titles_scored.parquet` (asset, date, finbert_title_mean) que
`panel.build_panel` levanta para habilitar M3.

Orden de imports Windows: finbert (torch) PRIMERO. Correr:
    python -m src.sectors.score_titles [tech|biotech|financials|all] [per_day]
"""

from __future__ import annotations

import sys

from src.sentiment import finbert  # importa torch primero (gotcha Windows)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ingestion import eodhd_news as en  # noqa: E402
from src.sectors import baskets, panel  # noqa: E402

MAX_LEN = 64
CHUNK = 5000
PER_DAY = 8


def _load_titles(sector: str) -> pd.DataFrame:
    recs = []
    for f in sorted(baskets.news_dir(sector).glob("*.json")):
        asset = f.stem.split("-")[0]
        for a in en._read_raw(f):
            title = (a.get("title") or "").strip()
            link = a.get("link") or ""
            if title and link:
                recs.append({"asset": asset, "ts": a.get("date"), "title": title, "link": link})
    df = pd.DataFrame(recs)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).drop_duplicates(["asset", "link"])


def _map_trading_day(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cal_date"] = df["ts"].dt.normalize()
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
    return pd.concat(out, ignore_index=True)


def build_scoring_set(sector: str, per_day: int = PER_DAY, seed: int = 0) -> pd.DataFrame:
    prices = panel._load_prices(sector)
    df = _map_trading_day(_load_titles(sector), prices)
    df = df.sample(frac=1.0, random_state=seed)                       # baraja
    df = df.groupby(["asset", "date"]).head(per_day).reset_index(drop=True)  # cap por día
    return df[["asset", "date", "title"]]


def score_sector(sector: str, per_day: int = PER_DAY) -> pd.DataFrame:
    ss = build_scoring_set(sector, per_day)
    n = len(ss)
    print(f"[{sector}] {n} titulos a scorear (cap {per_day}/dia)", flush=True)

    cdir = baskets.SECTORS_PROCESSED / "titles_chunks" / sector
    cdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n, CHUNK):
        cid = start // CHUNK
        cpath = cdir / f"chunk_{cid:04d}.parquet"
        if cpath.exists():
            continue
        block = ss.iloc[start:start + CHUNK]
        scored = finbert.score_texts(block["title"].tolist(), batch_size=64, max_length=MAX_LEN)
        out = block[["asset", "date"]].copy()
        out["title_score"] = scored["sentiment_score"].to_numpy()
        out.to_parquet(cpath, index=False)
        print(f"  chunk {cid} ({start}-{min(start + CHUNK, n)}) OK", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(cdir.glob("chunk_*.parquet"))]
    allrows = pd.concat(parts, ignore_index=True)
    daily = (allrows.groupby(["asset", "date"])
             .agg(finbert_title_mean=("title_score", "mean")).reset_index())
    baskets.SECTORS_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(baskets.titles_scored_path(sector), index=False)
    print(f"[{sector}] -> {baskets.titles_scored_path(sector).name} "
          f"({len(daily)} dias-activo)", flush=True)
    return daily


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    per_day = int(sys.argv[2]) if len(sys.argv) > 2 else PER_DAY
    targets = list(baskets.SECTORS) if arg == "all" else [arg]
    for s in targets:
        score_sector(s, per_day)
