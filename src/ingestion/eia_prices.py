"""Descarga de precios spot diarios de EIA (WTI, Brent, Henry Hub).

Usa la EIA API v2 vía el endpoint `/v2/seriesid/{id}`, que acepta los series
IDs estilo v1 del spec. Guarda la respuesta JSON **cruda y sin modificar** en
`data/raw/prices/eia/`, con el nombre `{serie}_{fecha_descarga}.json` para
trazabilidad.

Este módulo SOLO descarga y persiste el crudo. No hace limpieza, alineación
de calendarios ni cálculo de retornos: eso vive en `src/cleaning/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
EIA_V2_SERIESID_URL = "https://api.eia.gov/v2/seriesid/{series_id}"

# Series IDs (estilo v1, soportados por el endpoint /v2/seriesid/).
SERIES: dict[str, str] = {
    "WTI": "PET.RWTC.D",
    "BRENT": "PET.RBRTE.D",
    "HENRY_HUB": "NG.RNGWHHD.D",
}

DEFAULT_START = "2014-01-01"   # cubrimos "desde al menos 2014" (spec sección 9)
PAGE_LENGTH = 5000             # máximo de filas por request que permite la API
REQUEST_TIMEOUT = 30           # segundos


# --------------------------------------------------------------------------- #
# Capa HTTP con reintentos
# --------------------------------------------------------------------------- #
def _is_retryable(exc: BaseException) -> bool:
    """Solo reintentar fallos transitorios; nunca un 4xx de cliente.

    Reintentar un 400/403 (p. ej. key inválida o serie mal escrita) es inútil:
    el error es permanente. Sí reintentamos timeouts, cortes de red y 5xx/429.
    """
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
    """GET con backoff exponencial ante fallos transitorios."""
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Descarga + persistencia del crudo
# --------------------------------------------------------------------------- #
def _raw_path(series_name: str, download_date: str, page: int) -> Path:
    """Ruta del archivo crudo. Agrega sufijo de página solo si hay paginación."""
    suffix = "" if page == 0 else f"_p{page}"
    return config.RAW_PRICES_EIA_DIR / f"{series_name}_{download_date}{suffix}.json"


def fetch_series(
    series_name: str,
    series_id: str,
    start: str = DEFAULT_START,
    api_key: str | None = None,
) -> tuple[list[dict], list[Path]]:
    """Descarga una serie completa desde `start` y guarda el/los crudo(s).

    Devuelve `(rows, raw_paths)` donde `rows` son los registros combinados de
    todas las páginas (para verificación) y `raw_paths` los archivos guardados.
    La API tope a 5000 filas por request; si el rango pedido excede eso,
    paginamos por `offset` y guardamos cada página como un crudo independiente.
    """
    api_key = api_key or config.require_key("EIA_API_KEY")
    config.RAW_PRICES_EIA_DIR.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = EIA_V2_SERIESID_URL.format(series_id=series_id)

    rows: list[dict] = []
    raw_paths: list[Path] = []
    offset = 0
    page = 0

    while True:
        params = {
            "api_key": api_key,
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_LENGTH,
        }
        resp = _get(url, params)

        # Guardar el crudo VERBATIM (texto tal cual lo devolvió la API).
        path = _raw_path(series_name, download_date, page)
        path.write_text(resp.text, encoding="utf-8")
        raw_paths.append(path)

        payload = resp.json()
        response = payload.get("response", {})
        data = response.get("data", [])
        rows.extend(data)

        total = int(response.get("total", len(rows)))
        offset += len(data)
        page += 1

        # Cortar cuando ya trajimos todo o la página vino vacía.
        if not data or offset >= total:
            break

    return rows, raw_paths


def download_all(start: str = DEFAULT_START) -> dict[str, tuple[list[dict], list[Path]]]:
    """Descarga las tres series spot del spec."""
    results: dict[str, tuple[list[dict], list[Path]]] = {}
    for name, sid in SERIES.items():
        print(f"Descargando {name} ({sid}) desde {start} ...")
        rows, raw_paths = fetch_series(name, sid, start=start)
        print(f"  -> {len(rows)} filas | crudo: {', '.join(p.name for p in raw_paths)}")
        results[name] = (rows, raw_paths)
    return results


# --------------------------------------------------------------------------- #
# Verificación rápida (solo lectura — no escribe nada procesado)
# --------------------------------------------------------------------------- #
def _preview(name: str, rows: list[dict]) -> None:
    """Muestra primeras/últimas filas y rango de fechas de una serie."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{name}] SIN DATOS")
        return

    # 'period' = fecha, 'value' = precio. Ordenamos por las dudas.
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period").reset_index(drop=True)
    units = df["units"].iloc[0] if "units" in df.columns else "?"

    print(f"\n===== {name} =====")
    print(f"Filas: {len(df)} | Rango: {df['period'].min().date()} -> {df['period'].max().date()} | Unidad: {units}")
    cols = [c for c in ("period", "value", "units") if c in df.columns]
    print("Primeras filas:")
    print(df[cols].head().to_string(index=False))
    print("Últimas filas:")
    print(df[cols].tail().to_string(index=False))


if __name__ == "__main__":
    results = download_all()
    for name, (rows, _paths) in results.items():
        _preview(name, rows)
