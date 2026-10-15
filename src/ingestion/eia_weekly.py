"""Descarga de los reportes semanales de EIA (texto estructurado de alto impacto).

Dos releases programados que mueven el mercado de energía (spec sección 4):
- **Weekly Petroleum Status Report (WPSR)** — inventarios de crudo. Release:
  miércoles 10:30 ET, reporta la semana que terminó el viernes anterior.
- **Weekly Natural Gas Storage Report** — almacenamiento de gas (clave para
  Henry Hub). Release: jueves 10:30 ET.

A diferencia de GDELT (seendate ≈ detección), acá el timestamp es **exacto y
oficial**: lo computamos como la hora de release (ET → UTC con DST vía zoneinfo),
no el `period` (que es la fecha de referencia del dato). Eso hace el cutoff PiT
trivial y robusto.

Este módulo descarga + guarda el crudo, y expone `to_documents()` que sintetiza
los documentos del corpus (lo consume `news_cleaner`). No produce el parquet.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import config

EIA_V2_SERIESID_URL = "https://api.eia.gov/v2/seriesid/{series_id}"

# kind -> serie semanal de EIA.
SERIES: dict[str, str] = {
    "crude": "PET.WCESTUS1.W",                 # U.S. ending stocks excl. SPR (MBBL)
    "gas": "NG.NW2_EPG0_SWO_R48_BCF.W",        # Lower-48 working gas storage (BCF)
}
# Activos que cada reporte informa.
ASSETS_FOR: dict[str, list[str]] = {"crude": ["WTI", "BRENT"], "gas": ["HENRY_HUB"]}
# Texto base por reporte.
LABEL = {
    "crude": ("EIA Weekly Petroleum Status Report", "U.S. crude oil inventories"),
    "gas": ("EIA Weekly Natural Gas Storage Report", "U.S. working gas in storage"),
}
# Offset (días desde el viernes de `period`) y hora de release oficial.
RELEASE_OFFSET_DAYS = {"crude": 5, "gas": 6}   # miércoles / jueves siguientes
RELEASE_TIME_ET = (10, 30)
ET = ZoneInfo("America/New_York")

REQUEST_TIMEOUT = 30


# --------------------------------------------------------------------------- #
# HTTP con reintentos (igual criterio que eia_prices)
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
def _get(url: str, params: dict) -> requests.Response:
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Descarga + persistencia del crudo
# --------------------------------------------------------------------------- #
def fetch_report(kind: str, api_key: str | None = None) -> Path:
    """Descarga la serie semanal completa y guarda el JSON crudo verbatim."""
    api_key = api_key or config.require_key("EIA_API_KEY")
    config.RAW_NEWS_EIA_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    url = EIA_V2_SERIESID_URL.format(series_id=SERIES[kind])
    params = {
        "api_key": api_key,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    resp = _get(url, params)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.RAW_NEWS_EIA_WEEKLY_DIR / f"{kind}_{download_date}.json"
    path.write_text(resp.text, encoding="utf-8")
    return path


def download_all() -> dict[str, Path]:
    paths = {}
    for kind in SERIES:
        print(f"Descargando EIA Weekly [{kind}] ({SERIES[kind]}) ...")
        paths[kind] = fetch_report(kind)
        print(f"  -> {paths[kind].name}")
    return paths


# --------------------------------------------------------------------------- #
# Síntesis de documentos del corpus (lo consume news_cleaner)
# --------------------------------------------------------------------------- #
def _release_dt(period_str: str, kind: str) -> datetime:
    """Hora oficial de release (UTC) a partir del `period` (viernes ref.)."""
    y, m, d = (int(x) for x in period_str.split("-"))
    rel = datetime(y, m, d) + timedelta(days=RELEASE_OFFSET_DAYS[kind])
    dt_et = datetime(rel.year, rel.month, rel.day, *RELEASE_TIME_ET, tzinfo=ET)
    return dt_et.astimezone(timezone.utc)


def _latest_raw(kind: str) -> Path | None:
    files = sorted(config.RAW_NEWS_EIA_WEEKLY_DIR.glob(f"{kind}_*.json"))
    return files[-1] if files else None


def to_documents(kind: str) -> list[dict]:
    """Convierte el crudo de un reporte en documentos del corpus.

    Cada release semanal = un documento con `published_at` = hora oficial de
    release (UTC), título sintetizado con la variación semana-a-semana, y los
    activos que informa. `origin=EIA_WEEKLY`, fuente `eia.gov` (trusted).
    """
    path = _latest_raw(kind)
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get("response", {}).get("data", [])
    rows = [r for r in rows if r.get("value") is not None and r.get("period")]
    rows.sort(key=lambda r: r["period"])

    report_name, subject = LABEL[kind]
    docs: list[dict] = []
    prev = None
    for r in rows:
        value = float(r["value"])
        units = r.get("units", "")
        chg = None if prev is None else value - prev
        prev = value
        if chg is None:
            direction, chg_txt = "reported at", ""
        elif chg > 0:
            direction, chg_txt = "rose to", f" (+{chg:,.0f} {units} w/w)"
        elif chg < 0:
            direction, chg_txt = "fell to", f" ({chg:,.0f} {units} w/w)"
        else:
            direction, chg_txt = "were unchanged at", ""
        title = (
            f"{report_name}: {subject} {direction} {value:,.0f} {units}"
            f"{chg_txt} (week ending {r['period']})"
        )
        docs.append(
            {
                "published_at": _release_dt(r["period"], kind),
                "title": title,
                "url": f"https://www.eia.gov/petroleum/weekly/#{r['period']}" if kind == "crude"
                else f"https://ir.eia.gov/ngs/ngs.html#{r['period']}",
                "domain": "eia.gov",
                "assets": ASSETS_FOR[kind],
                "value": value,
                "wow_change": chg,
                "units": units,
            }
        )
    return docs


if __name__ == "__main__":
    download_all()
    for kind in SERIES:
        docs = to_documents(kind)
        print(f"\n=== {kind}: {len(docs)} documentos ===")
        for d in docs[-3:]:
            print(f"  pub={d['published_at'].isoformat()} | assets={d['assets']}")
            print(f"    {d['title']}")
