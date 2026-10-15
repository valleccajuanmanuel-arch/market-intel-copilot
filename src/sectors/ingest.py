"""Ingesta EODHD para la expansión multi-sector: precios EOD + noticias por ticker.

- Precios: `/api/eod/{ticker}` -> JSON crudo por ticker (usa `adjusted_close`,
  imprescindible en equities por splits/dividendos).
- Noticias: `/api/news?s={ticker}` por ventanas mensuales, guardando registros
  **lean** (date, title, link, polarity) -> liviano y listo para FinBERT-títulos.

Todo resumible (salta archivos ya bajados). Reusa el retry/ventanas de
`eodhd_news`. Correr: `python -m src.sectors.ingest`.
"""

from __future__ import annotations

import json
import time
from datetime import date

import requests

from src import config
from src.ingestion import eodhd_news as en
from src.sectors import baskets

EOD_URL = "https://eodhd.com/api/eod/{ticker}"
SENT_URL = "https://eodhd.com/api/sentiments"


def _today() -> date:
    return date.today()


def _start() -> date:
    y, m, d = (int(x) for x in baskets.START.split("-"))
    return date(y, m, d)


# --------------------------------------------------------------------------- #
# Precios EOD
# --------------------------------------------------------------------------- #
def download_prices(sector: str) -> None:
    tok = config.require_key("EODHD_API_TOKEN")
    out = baskets.prices_dir(sector)
    out.mkdir(parents=True, exist_ok=True)
    for ticker in baskets.SECTORS[sector]:
        path = out / f"{en._safe(ticker)}.json"
        if path.exists():
            continue
        r = requests.get(
            EOD_URL.format(ticker=ticker),
            params={"api_token": tok, "from": baskets.START,
                    "to": _today().isoformat(), "fmt": "json"},
            timeout=en.REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"  [{sector}] {ticker}: {len(rows)} barras EOD", flush=True)
        time.sleep(en.SLEEP_BETWEEN_CALLS)


# --------------------------------------------------------------------------- #
# Noticias (lean)
# --------------------------------------------------------------------------- #
def _fetch_news_page(ticker: str, tok: str, frm: str, to: str, offset: int) -> list[dict]:
    params = {"api_token": tok, "s": ticker, "from": frm, "to": to,
              "limit": en.PAGE_LIMIT, "offset": offset, "fmt": "json"}
    data = en._get(params).json()
    return data if isinstance(data, list) else []


def _lean(a: dict) -> dict:
    s = a.get("sentiment") or {}
    return {"date": a.get("date"), "title": (a.get("title") or "").strip(),
            "link": a.get("link") or "", "polarity": s.get("polarity")}


def download_news(sector: str) -> None:
    """Un archivo por ticker, paginando sobre TODO el rango (pocos requests).

    Sólo para Fase 2 (FinBERT sobre títulos): guarda date+title+link. El volumen
    y el tono EODHD ya salen del endpoint agregado `download_sentiments`.
    """
    tok = config.require_key("EODHD_API_TOKEN")
    out = baskets.news_dir(sector)
    out.mkdir(parents=True, exist_ok=True)
    frm, to = baskets.START, _today().isoformat()
    for ticker in baskets.SECTORS[sector]:
        path = out / f"{en._safe(ticker)}.json"
        if path.exists():
            continue
        arts: list[dict] = []
        offset = 0
        while True:
            page = _fetch_news_page(ticker, tok, frm, to, offset)
            arts.extend(page)
            if len(page) < en.PAGE_LIMIT:
                break
            offset += en.PAGE_LIMIT
            time.sleep(en.SLEEP_BETWEEN_CALLS)
        lean = [{"date": a.get("date"), "title": (a.get("title") or "").strip(),
                 "link": a.get("link") or ""} for a in arts]
        path.write_text(json.dumps(lean, ensure_ascii=False), encoding="utf-8")
        print(f"  [{sector}] {ticker}: {len(arts)} noticias", flush=True)
        time.sleep(en.SLEEP_BETWEEN_CALLS)


# --------------------------------------------------------------------------- #
# Sentiment agregado diario (liviano: count + normalized, SIN cuerpos)
# --------------------------------------------------------------------------- #
def download_sentiments(sector: str) -> None:
    """Un request por ticker: sentimiento diario agregado (count, normalized).

    Es lo que necesita Fase 1 (M0-M2): volumen de noticias + tono EODHD, sin bajar
    el texto de los artículos. Rapidísimo comparado con `/api/news`.
    """
    tok = config.require_key("EODHD_API_TOKEN")
    out = baskets.SECTORS_RAW / sector / "sentiments"
    out.mkdir(parents=True, exist_ok=True)
    for ticker in baskets.SECTORS[sector]:
        path = out / f"{en._safe(ticker)}.json"
        if path.exists():
            continue
        r = requests.get(SENT_URL, params={"s": ticker, "from": baskets.START,
                         "to": _today().isoformat(), "api_token": tok, "fmt": "json"},
                         timeout=en.REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        series = data.get(ticker, []) if isinstance(data, dict) else []
        path.write_text(json.dumps(series, ensure_ascii=False), encoding="utf-8")
        print(f"  [{sector}] {ticker}: {len(series)} dias con sentimiento", flush=True)
        time.sleep(en.SLEEP_BETWEEN_CALLS)


def download_sector(sector: str) -> None:
    print(f"[{sector}] precios...", flush=True)
    download_prices(sector)
    print(f"[{sector}] noticias...", flush=True)
    download_news(sector)


if __name__ == "__main__":
    for sec in baskets.SECTORS:
        print(f"\n===== {sec.upper()} =====", flush=True)
        download_sector(sec)
    print("\nListo: precios + noticias de todos los sectores.", flush=True)
