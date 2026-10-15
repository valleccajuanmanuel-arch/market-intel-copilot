"""Limpieza del corpus de noticias → `news_corpus.parquet` (esquema sección 7).

Fuentes integradas:
- **GDELT** (`data/raw/news/gdelt/`): dedup + filtro de relevancia D3.
- **EIA Weekly** (vía `eia_weekly.to_documents`): releases con timestamp oficial.
- **EODHD** (vía `eodhd_news.to_documents`): título + CUERPO completo, dedup entre
  tickers, sin drop de relevancia (relevante por fuente).
- (Apify: reservado, `origin=APIFY`, aún no integrado.)

Decisiones (D3, acordadas):
- Se **dropean** los títulos de GDELT sin ninguna keyword de energía (relevancia:
  FinBERT solo ve el titular; un titular off-topic es ruido). Los docs de EIA
  siempre se conservan (son releases curados y siempre on-topic).
- Se **conservan todas las fuentes** con flag `source_quality` (trusted/other);
  el filtrado a trusted-only queda como decisión downstream, reversible.
- asset por origen: crudo→{WTI,BRENT}, gas→{HENRY_HUB}. Explode.
- cutoff PiT 20:00 UTC; `trading_day` = próximo día de trading del calendario de
  precios del activo.

Reproducible: re-correr tras bajar el histórico completo de GDELT regenera el
corpus definitivo sin cambios de código.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pandera as pa

from src import config
from src.ingestion import eia_weekly, eodhd_news

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
QUERY_ASSETS: dict[str, list[str]] = {
    "crude": ["WTI", "BRENT"],
    "gas": ["HENRY_HUB"],
}
ASSETS = ["WTI", "BRENT", "HENRY_HUB"]
ORIGINS = ["GDELT", "EIA_WEEKLY", "EODHD", "APIFY"]
QUALITY = ["trusted", "other"]

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com",
    "oilprice.com", "spglobal.com", "argusmedia.com", "cnbc.com",
    "marketwatch.com", "rigzone.com", "naturalgasintel.com", "eia.gov",
}

KEYWORDS = [
    "oil", "crude", "WTI", "Brent", "OPEC", "natural gas", "Henry Hub", "LNG",
    "refinery", "crude inventory", "shale", "oil price", "pipeline", "barrel",
    "gas storage", "gas price",
]

CUTOFF_HOUR_UTC = 20
DEDUP_WINDOW = pd.Timedelta(hours=24)
# Ventana del corpus (scope validación 2020+): alinea todas las fuentes y evita
# amontonar releases viejos de EIA (que llega a 1982) en el primer día.
CORPUS_START = pd.Timestamp("2020-01-01", tz="UTC")

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Helpers de texto / fuente
# --------------------------------------------------------------------------- #
def _normalize_title(title: str) -> str:
    t = _PUNCT.sub(" ", str(title).lower())
    return _WS.sub(" ", t).strip()


def _matched_keywords(title: str) -> list[str]:
    tl = str(title).lower()
    return [kw for kw in KEYWORDS if kw.lower() in tl]


def _source_quality(domain: str) -> str:
    d = (domain or "").lower()
    for w in TRUSTED_DOMAINS:
        if d == w or d.endswith("." + w):
            return "trusted"
    return "other"


# --------------------------------------------------------------------------- #
# Calendario de trading + asignación de trading_day
# --------------------------------------------------------------------------- #
def _load_calendar(assets: list[str]) -> dict[str, pd.DatetimeIndex]:
    prices = pd.read_parquet(config.PRICES_DAILY_PARQUET)
    cal: dict[str, pd.DatetimeIndex] = {}
    for a in assets:
        d = prices.loc[(prices["asset"] == a) & (prices["close"].notna()), "date"]
        d = pd.DatetimeIndex(pd.to_datetime(d, utc=True)).tz_convert("UTC").tz_localize(None)
        cal[a] = pd.DatetimeIndex(sorted(set(d)))
    return cal


def _trading_day_for(pub_ts: pd.Timestamp, asset: str, cal: dict[str, pd.DatetimeIndex]):
    pub_utc = pub_ts.tz_convert("UTC")
    eff = pub_utc.tz_localize(None).normalize()
    if pub_utc.hour >= CUTOFF_HOUR_UTC:
        eff = eff + pd.Timedelta(days=1)
    caldx = cal.get(asset)
    if caldx is None or len(caldx) == 0:
        return pd.NaT
    pos = caldx.searchsorted(eff)
    if pos < len(caldx):
        return caldx[pos].tz_localize("UTC")
    return pd.NaT


# --------------------------------------------------------------------------- #
# Carga de fuentes
# --------------------------------------------------------------------------- #
def _load_gdelt() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(config.RAW_NEWS_GDELT_DIR / "*.json"))):
        tag = Path(f).name.split("_")[0]  # 'crude' / 'gas'
        if tag not in QUERY_ASSETS:
            continue
        txt = Path(f).read_text(encoding="utf-8").strip()
        if not txt:
            continue  # ventana vacía (guardada por gdelt_news para el resume)
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError:
            continue
        for art in payload.get("articles", []) or []:
            rows.append({**art, "_tag": tag})
    return rows


_FRAME_COLS = ["published_at", "title", "body", "url", "domain", "matched_keywords", "assets", "origin"]


def _gdelt_frame(stats: dict) -> pd.DataFrame:
    """GDELT → frame article-level, con dedup (24h) y filtro de relevancia D3."""
    raw = _load_gdelt()
    stats["gdelt_raw"] = len(raw)
    if not raw:
        stats["gdelt_dedup"] = stats["gdelt_dropped"] = stats["gdelt_kept"] = 0
        return pd.DataFrame(columns=_FRAME_COLS)

    g = pd.DataFrame(raw)
    g["published_at"] = pd.to_datetime(g["seendate"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    g = g[g["published_at"].notna()].copy()
    g["title"] = g["title"].fillna("").astype(str)
    g["url"] = g["url"].fillna("").astype(str)
    g["domain"] = g["domain"].fillna("").astype(str)

    # Dedup hash de título normalizado + ventana 24h (conserva el más antiguo).
    g["_hash"] = g["title"].map(lambda t: hashlib.sha1(_normalize_title(t).encode("utf-8")).hexdigest())
    g = g.sort_values("published_at").reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    keep: list[bool] = []
    for h, t in zip(g["_hash"], g["published_at"]):
        prev = last_seen.get(h)
        if prev is None or (t - prev) > DEDUP_WINDOW:
            keep.append(True)
            last_seen[h] = t
        else:
            keep.append(False)
    g = g[pd.Series(keep, index=g.index)].copy()
    stats["gdelt_dedup"] = len(g)

    g["matched_keywords"] = g["title"].map(_matched_keywords)
    # D3: filtro de relevancia — dropear títulos sin ninguna keyword.
    before = len(g)
    g = g[g["matched_keywords"].map(len) > 0].copy()
    stats["gdelt_dropped"] = before - len(g)
    stats["gdelt_kept"] = len(g)

    g["assets"] = g["_tag"].map(lambda tg: list(QUERY_ASSETS[tg]))
    g["origin"] = "GDELT"
    g["body"] = ""
    return g[_FRAME_COLS]


def _eia_frame(stats: dict) -> pd.DataFrame:
    """EIA Weekly → frame article-level (docs con timestamp oficial)."""
    docs: list[dict] = []
    for kind in ("crude", "gas"):
        docs.extend(eia_weekly.to_documents(kind))
    stats["eia_docs"] = len(docs)
    if not docs:
        return pd.DataFrame(columns=_FRAME_COLS)
    e = pd.DataFrame(docs)
    e["published_at"] = pd.to_datetime(e["published_at"], utc=True)
    e["matched_keywords"] = e["title"].map(_matched_keywords)
    e["origin"] = "EIA_WEEKLY"
    e["body"] = ""
    return e[_FRAME_COLS]


def _dedup_by_title_24h(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup por hash de título normalizado + ventana 24h (conserva el más antiguo)."""
    df = df.copy()
    df["_hash"] = df["title"].map(lambda t: hashlib.sha1(_normalize_title(t).encode("utf-8")).hexdigest())
    df = df.sort_values("published_at").reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    keep: list[bool] = []
    for h, t in zip(df["_hash"], df["published_at"]):
        prev = last_seen.get(h)
        if prev is None or (t - prev) > DEDUP_WINDOW:
            keep.append(True)
            last_seen[h] = t
        else:
            keep.append(False)
    return df[pd.Series(keep, index=df.index)].drop(columns="_hash").reset_index(drop=True)


def _eodhd_frame(stats: dict) -> pd.DataFrame:
    """EODHD → frame article-level (título + CUERPO completo). Dedup entre tickers.

    No aplica el drop de relevancia D3: las noticias EODHD vienen de tickers del
    sector energético (relevantes por fuente), a diferencia de GDELT.
    """
    docs = eodhd_news.to_documents()
    stats["eodhd_raw"] = len(docs)
    if not docs:
        stats["eodhd_dedup"] = 0
        return pd.DataFrame(columns=_FRAME_COLS)
    e = pd.DataFrame(docs)
    e["published_at"] = pd.to_datetime(e["published_at"], utc=True, errors="coerce")
    e = e[e["published_at"].notna()].copy()
    for col in ("title", "body", "url", "domain"):
        e[col] = e[col].fillna("").astype(str)
    e = _dedup_by_title_24h(e)  # colapsa la misma noticia repetida entre tickers
    stats["eodhd_dedup"] = len(e)
    e["matched_keywords"] = e["title"].map(_matched_keywords)
    e["origin"] = "EODHD"
    return e[_FRAME_COLS]


# --------------------------------------------------------------------------- #
# Construcción del corpus
# --------------------------------------------------------------------------- #
def build_corpus() -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    frames = [
        f for f in (_gdelt_frame(stats), _eia_frame(stats), _eodhd_frame(stats))
        if not f.empty
    ]
    if not frames:
        raise FileNotFoundError(
            "No hay datos de noticias (ni GDELT ni EIA Weekly). Corré las ingestas primero."
        )
    df = pd.concat(frames, ignore_index=True)

    # Acotar a la ventana del corpus (D1: 2017+). Descarta releases de EIA previos
    # a 2017 (la serie llega a 1982) para alinear con GDELT y evitar amontonarlos
    # en el primer día de trading. La variación w/w de EIA ya se computó sobre la
    # serie completa, así que el primer doc de 2017 conserva su cambio correcto.
    before_win = len(df)
    df = df[df["published_at"] >= CORPUS_START].copy()
    stats["pre_window_dropped"] = before_win - len(df)

    # Explode a una fila por (artículo × activo).
    df = df.rename(columns={"assets": "asset"}).explode("asset", ignore_index=True)

    # Red de seguridad: sin duplicados exactos de (url, asset) entre fuentes.
    df = df.drop_duplicates(subset=["url", "asset"], keep="first").reset_index(drop=True)

    df["source"] = df["domain"].fillna("").astype(str)
    df["source_quality"] = df["source"].map(_source_quality)
    df["body"] = df["body"].fillna("").astype(str)

    cal = _load_calendar(sorted(df["asset"].dropna().unique()))
    df["trading_day"] = [
        _trading_day_for(p, a, cal) for p, a in zip(df["published_at"], df["asset"])
    ]
    stats["no_trading_day"] = int(df["trading_day"].isna().sum())
    df = df[df["trading_day"].notna()].copy()

    df["asset"] = pd.Categorical(df["asset"], categories=ASSETS)
    df["source_quality"] = pd.Categorical(df["source_quality"], categories=QUALITY)
    df["origin"] = pd.Categorical(df["origin"], categories=ORIGINS)
    df["source"] = df["source"].astype("category")
    df["trading_day"] = pd.to_datetime(df["trading_day"], utc=True)

    cols = [
        "published_at", "trading_day", "title", "body", "source",
        "source_quality", "url", "asset", "origin", "matched_keywords",
    ]
    df = df[cols].sort_values(["asset", "published_at"]).reset_index(drop=True)
    stats["rows"] = len(df)
    return df, stats


# --------------------------------------------------------------------------- #
# Validación Pandera (esquema sección 7)
# --------------------------------------------------------------------------- #
def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "published_at": pa.Column("datetime64[ns, UTC]", nullable=False),
            "trading_day": pa.Column("datetime64[ns, UTC]", nullable=False),
            "title": pa.Column(object, nullable=False),
            "body": pa.Column(object, nullable=True),
            "source": pa.Column(pd.CategoricalDtype(), nullable=False),
            "source_quality": pa.Column(pd.CategoricalDtype(QUALITY), nullable=False),
            "url": pa.Column(object, nullable=False),
            "asset": pa.Column(pd.CategoricalDtype(ASSETS), nullable=False),
            "origin": pa.Column(pd.CategoricalDtype(ORIGINS), nullable=False),
            "matched_keywords": pa.Column(object, nullable=True),
        },
        unique=["url", "asset"],
        strict=True,
        ordered=False,
        checks=pa.Check(
            lambda d: bool((d["trading_day"] >= d["published_at"].dt.normalize()).all()),
            error="trading_day anterior al día de published_at (look-ahead)",
        ),
    )


def run() -> tuple[pd.DataFrame, dict]:
    df, stats = build_corpus()
    validated = build_schema().validate(df, lazy=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(config.NEWS_CORPUS_PARQUET, index=False)
    return validated, stats


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def _report(df: pd.DataFrame, stats: dict) -> None:
    print("\n===== PIPELINE =====")
    print(f"GDELT crudos        : {stats.get('gdelt_raw', 0)}")
    print(f"  tras dedup (24h)  : {stats.get('gdelt_dedup', 0)}")
    print(f"  dropeados D3 (sin keyword): {stats.get('gdelt_dropped', 0)}")
    print(f"  GDELT conservados : {stats.get('gdelt_kept', 0)}")
    print(f"EIA Weekly docs     : {stats.get('eia_docs', 0)}")
    print(f"EODHD crudos        : {stats.get('eodhd_raw', 0)}")
    print(f"  tras dedup (24h)  : {stats.get('eodhd_dedup', 0)}")
    print(f"Dropeados < {CORPUS_START.date()} (fuera de ventana): {stats.get('pre_window_dropped', 0)}")
    print(f"Sin trading_day     : {stats.get('no_trading_day', 0)}")
    print(f"Filas finales (x activo): {stats.get('rows', 0)}")

    print("\n===== POR ORIGEN =====")
    print(df.groupby("origin", observed=True).size().to_string())
    print("\n===== POR ACTIVO =====")
    print(df.groupby("asset", observed=True).size().to_string())
    print("\n===== CALIDAD DE FUENTE =====")
    print(df.groupby("source_quality", observed=True).size().to_string())

    print("\n===== RANGOS TEMPORALES =====")
    print(f"published_at: {df['published_at'].min()}  ->  {df['published_at'].max()}")
    print(f"trading_day : {df['trading_day'].min().date()}  ->  {df['trading_day'].max().date()}")

    print("\n===== MUESTRA (2 por origen si hay) =====")
    for origin in ("GDELT", "EIA_WEEKLY", "EODHD"):
        sub = df[df["origin"] == origin].head(2)
        for _, r in sub.iterrows():
            print(f"- [{origin}/{r['asset']}] pub={r['published_at']} -> td={r['trading_day'].date()} "
                  f"| {r['source']} ({r['source_quality']}) | body={len(r['body'])}c\n    {r['title'][:95]}")


if __name__ == "__main__":
    validated, stats = run()
    print(f"\nGuardado: {config.NEWS_CORPUS_PARQUET}")
    _report(validated, stats)
