"""Descarga de noticias financieras de energía vía EODHD News API.

Fuente de **texto** para FinBERT: a diferencia de GDELT DOC (solo titular), EODHD
devuelve el **cuerpo completo** del artículo, y además su propio score de
sentimiento (útil como baseline adicional). Requiere plan pago de EODHD
(token en `EODHD_API_TOKEN`).

--- Estrategia (VERIFICADA contra la API) ---
EODHD indexa noticias **por ticker de empresa**, no por símbolo de commodity
(los futuros tipo CL.COMM dan 404; los ETFs y tags genéricos dan 0). Por eso se
consulta una **canasta de tickers del sector energético** por grupo:
    - oil  -> majors/servicios/refinadores -> activos WTI y BRENT
    - gas  -> productoras de gas natural    -> activo HENRY_HUB
Una misma noticia de mercado aparece bajo varios tickers; el dedup del
`news_cleaner` (hash de título) la colapsa.

Respuesta por artículo (campos confirmados): date, title, content, link,
symbols, tags, sentiment. Guarda un JSON por (grupo, ticker, mes) en
`data/raw/news/eodhd/` (`{grupo}_{ticker}_{YYYYMM}.json`), escrito de forma
atómica al terminar el mes -> resumible limpio. Este módulo SOLO descarga +
expone `to_documents()`; la limpieza es de `news_cleaner`.

LICENCIA: plan personal EODHD = uso NO comercial (confirmado por soporte: OK
mientras no generes ingresos *proveyendo su data*). Válido para la fase de
validación; al comercializar, revisar términos / pasar al plan comercial.

INTEGRACIÓN al cleaner (hacer cuando confirmemos el corpus real):
  1. Agregar "EODHD" a `ORIGINS` en `news_cleaner.py`.
  2. Agregar un `_eodhd_frame()` (dedup + relevancia) que use `to_documents()`.
  3. Hacer que el cleaner arrastre `body` por fuente (EODHD trae cuerpo).
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import config

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
EODHD_NEWS_URL = "https://eodhd.com/api/news"

# Canastas de tickers por grupo (todas verificadas con noticias). Tuneables.
OIL_TICKERS = [
    "XOM.US", "CVX.US", "COP.US", "OXY.US", "SLB.US",
    "EOG.US", "MPC.US", "VLO.US", "PSX.US", "HAL.US",
]
GAS_TICKERS = ["EQT.US", "LNG.US", "SWN.US", "CHK.US", "AR.US", "RRC.US", "CTRA.US"]

# grupo -> (tickers, activos a los que mapea)
QUERY_GROUPS: dict[str, dict] = {
    "oil": {"tickers": OIL_TICKERS, "assets": ["WTI", "BRENT"]},
    "gas": {"tickers": GAS_TICKERS, "assets": ["HENRY_HUB"]},
}

DEFAULT_START = date(2020, 1, 1)     # ventana de validación
DEFAULT_END = date.today()
PAGE_LIMIT = 1000                    # máximo de artículos por request
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.3            # EODHD permite ~1000 req/min; sin throttle real


# --------------------------------------------------------------------------- #
# HTTP con reintentos
# --------------------------------------------------------------------------- #
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get(params: dict) -> requests.Response:
    resp = requests.get(EODHD_NEWS_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _month_windows(start: date, end: date) -> list[tuple[str, str]]:
    """Lista de (from, to) por mes en formato YYYY-MM-DD, cubriendo [start, end]."""
    out: list[tuple[str, str]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        out.append((max(cur, start).isoformat(), min(nxt - timedelta(days=1), end).isoformat()))
        cur = nxt
    return out


def _domain(url: str) -> str:
    try:
        net = urlparse(url or "").netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except ValueError:
        return ""


def _safe(ticker: str) -> str:
    return ticker.replace(".", "-")


# --------------------------------------------------------------------------- #
# Descarga + persistencia del crudo
# --------------------------------------------------------------------------- #
def _fetch_page(ticker: str, api_token: str, from_s: str, to_s: str, offset: int) -> list[dict]:
    params = {
        "api_token": api_token,
        "s": ticker,
        "from": from_s,
        "to": to_s,
        "limit": PAGE_LIMIT,
        "offset": offset,
        "fmt": "json",
    }
    data = _get(params).json()
    return data if isinstance(data, list) else []


def download_ticker(
    group: str,
    ticker: str,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    api_token: str | None = None,
) -> int:
    """Baja las noticias de un ticker por ventanas mensuales (resumible por mes)."""
    api_token = api_token or config.require_key("EODHD_API_TOKEN")
    config.RAW_NEWS_EODHD_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for from_s, to_s in _month_windows(start, end):
        yyyymm = from_s[:7].replace("-", "")
        raw_path = config.RAW_NEWS_EODHD_DIR / f"{group}_{_safe(ticker)}_{yyyymm}.json"
        if raw_path.exists():
            total += len(_read_raw(raw_path))
            continue
        try:
            articles: list[dict] = []
            offset = 0
            while True:
                page = _fetch_page(ticker, api_token, from_s, to_s, offset)
                articles.extend(page)
                if len(page) < PAGE_LIMIT:
                    break
                offset += PAGE_LIMIT
                time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as exc:  # noqa: BLE001 - tolerante: se reintenta en próxima corrida
            print(f"  [{group}] {ticker} {yyyymm}: FALLO ({type(exc).__name__}) -> reintenta luego", flush=True)
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue
        raw_path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
        total += len(articles)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return total


def download_group(group: str, start: date = DEFAULT_START, end: date = DEFAULT_END) -> int:
    total = 0
    for ticker in QUERY_GROUPS[group]["tickers"]:
        n = download_ticker(group, ticker, start, end)
        total += n
        print(f"  [{group}] {ticker}: {n} noticias", flush=True)
    return total


def download_all(start: date = DEFAULT_START, end: date = DEFAULT_END) -> dict[str, int]:
    results: dict[str, int] = {}
    for group in QUERY_GROUPS:
        print(f"[EODHD] grupo '{group}' -> activos {QUERY_GROUPS[group]['assets']}", flush=True)
        results[group] = download_group(group, start, end)
    return results


# --------------------------------------------------------------------------- #
# Conversión a documentos del corpus (lo consumirá news_cleaner)
# --------------------------------------------------------------------------- #
def _read_raw(path: Path) -> list[dict]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def to_documents() -> list[dict]:
    """Todos los crudos EODHD -> documentos article-level del corpus.

    Cada doc: published_at (str ISO), title, body (cuerpo completo — ventaja de
    EODHD), url, domain, assets (según el grupo del ticker), origin='EODHD'.
    El dedup entre tickers lo hace el cleaner. El sentimiento propio de EODHD
    queda en el crudo para usarlo como baseline.
    """
    docs: list[dict] = []
    for group, cfg in QUERY_GROUPS.items():
        assets = cfg["assets"]
        for f in sorted(config.RAW_NEWS_EODHD_DIR.glob(f"{group}_*.json")):
            for a in _read_raw(f):
                docs.append(
                    {
                        "published_at": a.get("date"),
                        "title": (a.get("title") or "").strip(),
                        "body": (a.get("content") or "").strip(),
                        "url": a.get("link") or "",
                        "domain": _domain(a.get("link") or ""),
                        "assets": list(assets),
                        "origin": "EODHD",
                    }
                )
    return docs


if __name__ == "__main__":
    print(f"EODHD News: {DEFAULT_START} -> {DEFAULT_END}")
    print(f"Grupos: { {g: c['tickers'] for g, c in QUERY_GROUPS.items()} }")
    results = download_all()
    print(f"\nResumen (noticias nuevas por grupo): {results}")
