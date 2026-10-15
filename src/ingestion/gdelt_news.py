"""Descarga de noticias de energía vía GDELT DOC 2.0 API (sin API key).

Estrategia definitiva (decisiones D1/D2):
- **Rango**: 2017-01-01 → hoy. La DOC 2.0 API NO cubre antes de 2017 (devuelve
  "Invalid query start date"); 2015-2016 requeriría GKG (diferido).
- **Muestreo estratificado por franjas horarias**: cada día se parte en N
  ventanas UTC iguales (crudo 3×8h, gas 2×12h). Esto elimina el sesgo nocturno
  del cap de 250 (con 1 sola ventana diaria + sort=datedesc se perdía la mañana)
  y mantiene la descarga acotada y respetuosa del rate limit.
- **Resumible**: cada ventana se guarda como archivo; si ya existe, se saltea.
  Las ventanas que fallan (rate limit persistente) no dejan archivo y se
  reintentan en la próxima corrida. Esto permite correr en varias sesiones.

Guarda el JSON crudo verbatim en `data/raw/news/gdelt/` como
`{tag}_{YYYYMMDD}_{HH}.json`. Sin limpieza/dedup (eso es news_cleaner).

Caveat PiT: el timestamp es `seendate` (detección, granularidad 15 min, UTC),
usado como `published_at`.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERY_CRUDE = '(WTI OR Brent OR "crude oil" OR OPEC OR "oil price") sourcelang:english'
QUERY_GAS = '("natural gas" OR "Henry Hub" OR LNG OR "gas storage" OR "gas price") sourcelang:english'

# Rango (scope validación): ventana densa reciente 2020+ (COVID + crisis
# energética 2022 = oro para volatilidad). La DOC API cubre desde 2017 si más
# adelante se quiere extender.
HIST_START = date(2020, 1, 1)
HIST_END = date.today()

# Estratificación (D2): menos ventanas = tratable; con 2 franjas de 12h ya se
# mata el sesgo nocturno del cap de 250.
CRUDE_WINDOWS = 2   # 2 franjas de 12h (00-12, 12-24 UTC)
GAS_WINDOWS = 1     # 1 franja de 24h (gas tiene mucho menos volumen)

MAXRECORDS = 250            # tope por llamada
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 10.0  # 2x el límite de GDELT (1 req/5s): más margen -> menos 429/fallos

HEADERS = {"User-Agent": "market-intel-copilot/0.1 (research; volatility NLP)"}


# --------------------------------------------------------------------------- #
# Capa HTTP con reintentos
# --------------------------------------------------------------------------- #
class GdeltTransientError(Exception):
    """Respuesta transitoria (no-JSON: rate limit / cuelgue puntual)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, GdeltTransientError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def _get(params: dict) -> requests.Response:
    resp = requests.get(GDELT_DOC_URL, params=params, timeout=REQUEST_TIMEOUT, headers=HEADERS)
    resp.raise_for_status()
    if resp.text.strip():
        try:
            resp.json()
        except json.JSONDecodeError as exc:
            raise GdeltTransientError(f"Respuesta no-JSON: {resp.text[:160]!r}") from exc
    return resp


# --------------------------------------------------------------------------- #
# Ventanas / persistencia
# --------------------------------------------------------------------------- #
def _day_windows(day: date, n: int) -> list[tuple[datetime, datetime]]:
    """Parte un día en n franjas UTC contiguas e iguales (24 % n == 0)."""
    step = 24 // n
    base = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    return [
        (base + timedelta(hours=i * step), base + timedelta(hours=(i + 1) * step) - timedelta(seconds=1))
        for i in range(n)
    ]


def _read_raw(path: Path) -> list[dict]:
    """Lee artículos de un crudo guardado (tolerante a archivo vacío/corrupto)."""
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        return json.loads(txt).get("articles", []) or []
    except json.JSONDecodeError:
        return []


def fetch_window(
    start_dt: datetime,
    end_dt: datetime,
    query: str,
    tag: str,
    maxrecords: int = MAXRECORDS,
    sort: str = "datedesc",
) -> tuple[list[dict], Path]:
    """Descarga una ventana y guarda el crudo verbatim (incluso si vino vacío)."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords": maxrecords,
        "sort": sort,
    }
    resp = _get(params)
    raw_text = resp.text
    config.RAW_NEWS_GDELT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = config.RAW_NEWS_GDELT_DIR / f"{tag}_{start_dt.strftime('%Y%m%d_%H')}.json"
    raw_path.write_text(raw_text, encoding="utf-8")  # verbatim
    articles = (resp.json().get("articles", []) or []) if raw_text.strip() else []
    return articles, raw_path


# --------------------------------------------------------------------------- #
# Descarga histórica estratificada (resumible)
# --------------------------------------------------------------------------- #
def download_historical(
    tag: str,
    query: str,
    n_windows: int,
    start_date: date = HIST_START,
    end_date: date = HIST_END,
    sleep: float = SLEEP_BETWEEN_CALLS,
) -> dict:
    """Descarga 2017+ por ventanas estratificadas. Resumible y tolerante a fallos."""
    config.RAW_NEWS_GDELT_DIR.mkdir(parents=True, exist_ok=True)
    day = start_date
    total_new = 0
    n_fetched = 0
    n_failed = 0
    while day <= end_date:
        fetched_today = 0
        cached_today = 0
        for s, e in _day_windows(day, n_windows):
            raw_path = config.RAW_NEWS_GDELT_DIR / f"{tag}_{s.strftime('%Y%m%d_%H')}.json"
            if raw_path.exists():
                cached_today += 1
                continue
            try:
                articles, _ = fetch_window(s, e, query=query, tag=tag)
            except Exception as exc:  # noqa: BLE001 - tolerante: se reintenta en próxima corrida
                n_failed += 1
                print(f"  {s:%Y-%m-%d} {s:%H}h {tag}: FALLO ({type(exc).__name__}) -> reintenta luego", flush=True)
                time.sleep(sleep)
                continue
            fetched_today += 1
            n_fetched += 1
            total_new += len(articles)
            time.sleep(sleep)
        print(
            f"{day.isoformat()} {tag}: +{fetched_today} ventanas nuevas, {cached_today} cache "
            f"| cum nuevos: {total_new} art, {n_fetched} llamadas, {n_failed} fallos",
            flush=True,
        )
        day += timedelta(days=1)
    return {"total_new": total_new, "fetched": n_fetched, "failed": n_failed}


if __name__ == "__main__":
    print(f"GDELT histórico estratificado: {HIST_START} -> {HIST_END}", flush=True)
    print(f"[CRUDO] {CRUDE_WINDOWS} ventanas/día | {QUERY_CRUDE}", flush=True)
    r_crude = download_historical("crude", QUERY_CRUDE, CRUDE_WINDOWS)
    print(f"=== CRUDO: {r_crude} ===\n", flush=True)
    print(f"[GAS] {GAS_WINDOWS} ventanas/día | {QUERY_GAS}", flush=True)
    r_gas = download_historical("gas", QUERY_GAS, GAS_WINDOWS)
    print(f"=== GAS: {r_gas} ===", flush=True)
