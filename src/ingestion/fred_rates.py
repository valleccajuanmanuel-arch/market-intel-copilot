"""Descarga de la tasa libre de riesgo desde FRED (vía `fredapi`).

Baja SOFR como tasa principal y DGS3MO (T-Bill 3 meses) como respaldo. La
tasa libre de riesgo es el "piso" contra el que se mide toda estrategia
(Sharpe/Sortino); la consume el Módulo 5.

`fredapi` devuelve una `pandas.Series` (no JSON crudo), así que persistimos
esa serie **tal cual la entrega la librería**: a CSV, sin dropear NaN, sin
rellenar y sin resamplear. La limpieza/alineación es trabajo de `cleaning/`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

import pandas as pd
from fredapi import Fred
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
# id -> rol. SOFR es la principal; DGS3MO el respaldo (cubre historia previa a
# 2018, cuando SOFR todavía no existía).
SERIES: dict[str, str] = {
    "SOFR": "principal",
    "DGS3MO": "respaldo",
}

REQUEST_TIMEOUT = 30  # segundos (lo usa fredapi internamente vía urllib)


# --------------------------------------------------------------------------- #
# Descarga con reintentos
# --------------------------------------------------------------------------- #
@retry(
    retry=retry_if_exception_type((URLError, ConnectionError, TimeoutError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _fetch(fred: Fred, series_id: str) -> pd.Series:
    """Trae una serie de FRED, reintentando solo ante fallos de red."""
    return fred.get_series(series_id)


def fetch_series(series_id: str, fred: Fred | None = None) -> pd.Series:
    """Descarga una serie de FRED y la devuelve sin modificar."""
    fred = fred or Fred(api_key=config.require_key("FRED_API_KEY"))
    serie = _fetch(fred, series_id)
    serie.index = pd.to_datetime(serie.index)
    serie.name = series_id
    return serie


# --------------------------------------------------------------------------- #
# Persistencia del crudo
# --------------------------------------------------------------------------- #
def _save_raw(series_id: str, serie: pd.Series) -> Path:
    """Guarda la serie verbatim en data/raw/prices/fred/ como CSV (date,value)."""
    config.RAW_PRICES_FRED_DIR.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.RAW_PRICES_FRED_DIR / f"{series_id}_{download_date}.csv"
    # NaN se escribe como celda vacía: no lo tocamos, es parte del crudo.
    serie.rename("value").rename_axis("date").to_csv(path)
    return path


def download_all() -> dict[str, tuple[pd.Series, Path]]:
    """Descarga SOFR (principal) y DGS3MO (respaldo)."""
    fred = Fred(api_key=config.require_key("FRED_API_KEY"))
    results: dict[str, tuple[pd.Series, Path]] = {}
    for series_id, rol in SERIES.items():
        print(f"Descargando {series_id} ({rol}) desde FRED ...")
        serie = fetch_series(series_id, fred=fred)
        path = _save_raw(series_id, serie)
        print(f"  -> {len(serie)} observaciones | crudo: {path.name}")
        results[series_id] = (serie, path)
    return results


# --------------------------------------------------------------------------- #
# Verificación rápida (solo lectura)
# --------------------------------------------------------------------------- #
def _preview(series_id: str, serie: pd.Series) -> None:
    rol = SERIES.get(series_id, "?")
    no_nulas = serie.dropna()
    print(f"\n===== {series_id} ({rol}) =====")
    if no_nulas.empty:
        print("SIN DATOS")
        return
    print(
        f"Observaciones: {len(serie)} | No nulas: {len(no_nulas)} | "
        f"Rango: {serie.index.min().date()} -> {serie.index.max().date()}"
    )
    print(f"Último valor: {no_nulas.iloc[-1]:.4f} % ({no_nulas.index[-1].date()})")
    print("Primeras filas:")
    print(serie.head().to_string())
    print("Últimas filas:")
    print(serie.tail().to_string())


if __name__ == "__main__":
    results = download_all()
    for series_id, (serie, _path) in results.items():
        _preview(series_id, serie)
