"""Limpieza y alineación de precios spot de EIA → `prices_daily.parquet`.

Implementa la sección 4.1 del spec y produce el esquema de la sección 6:
columnas `date, asset, close, log_return, source` en formato largo (una fila
por día de trading y activo).

Decisiones clave (ver sección 4.1):
- **Alineación sin relleno ingenuo**: se construye un índice de fechas común
  (unión de los calendarios de los tres activos) y se deja NaN donde un activo
  no operó. NO se hace forward-fill: un día sin dato es un día sin trading, y
  rellenar fabricaría retornos falsos de cero.
- **Log-retornos sobre la serie nativa**: `r_t = ln(P_t / P_{t-1})` se calcula
  sobre las observaciones reales consecutivas de cada activo (antes del
  recorte de ventana), de modo que el primer retorno de la ventana use el
  cierre previo real y no quede NaN de borde.
- **Precios no positivos** (ej. WTI negativo de abril 2020, evento real): el
  ratio `P_t/P_{t-1}` se vuelve negativo y `ln` queda indefinido → log_return
  NaN, pero el precio se conserva. No se borra nada automáticamente.
- **Outliers** (|r_t| > 50%): se detectan y reportan, NO se eliminan. El
  esquema de la sección 6 no tiene columna de flag, así que no se persiste:
  downstream puede recalcularlo desde `log_return`.

El crudo en `data/raw/` queda intacto.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa

from src import config

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
ASSETS: tuple[str, ...] = ("WTI", "BRENT", "HENRY_HUB")
SOURCES: tuple[str, ...] = ("EIA", "EODHD")

# Ventana del procesado. None => historia completa de los crudos.
START_DATE: str | None = "2014-01-01"

# |r_t| > 0.5 (~50% en términos de log-retorno) se considera outlier.
OUTLIER_LOG_RETURN_THRESHOLD: float = 0.5


# --------------------------------------------------------------------------- #
# Carga de crudos
# --------------------------------------------------------------------------- #
def _load_raw_asset(asset: str) -> pd.Series:
    """Carga y unifica todas las páginas crudas de un activo.

    Devuelve una Series `close` indexada por fecha (UTC), historia nativa
    completa, ordenada y sin duplicados de fecha.
    """
    pattern = str(config.RAW_PRICES_EIA_DIR / f"{asset}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No hay crudos para {asset} en {config.RAW_PRICES_EIA_DIR}. "
            f"Corré primero src.ingestion.eia_prices."
        )

    records: list[dict] = []
    for f in files:
        payload = json.loads(Path(f).read_text(encoding="utf-8"))
        records.extend(payload.get("response", {}).get("data", []))

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"], utc=True)
    df["close"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df[["date", "close"]]
        .dropna(subset=["date"])
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .set_index("date")
    )
    return df["close"]


# --------------------------------------------------------------------------- #
# Construcción del dataset largo
# --------------------------------------------------------------------------- #
def _log_returns(close: pd.Series) -> pd.Series:
    """`r_t = ln(P_t / P_{t-1})`, NaN donde el ratio no es positivo.

    Si `P_t` o `P_{t-1}` no son positivos (ej. precio negativo), el ratio es
    ≤ 0 y el log queda indefinido: devolvemos NaN sin tocar el precio.
    """
    ratio = close / close.shift(1)
    return np.log(ratio.where(ratio > 0))


def build_prices_daily() -> pd.DataFrame:
    """Arma el DataFrame largo final (sin validar todavía)."""
    # 1. Cargar historia nativa de cada activo.
    native = {asset: _load_raw_asset(asset) for asset in ASSETS}

    # 2. Log-retornos sobre la serie nativa completa (consecutivos reales).
    returns = {asset: _log_returns(close) for asset, close in native.items()}

    # 3. Índice de fechas común (unión de calendarios), recortado a la ventana.
    common_index = pd.DatetimeIndex(
        sorted(set().union(*[s.index for s in native.values()]))
    )
    if START_DATE is not None:
        common_index = common_index[common_index >= pd.Timestamp(START_DATE, tz="UTC")]

    # 4. Reindexar cada activo al índice común (deja NaN donde no operó) y
    #    apilar en formato largo.
    frames = []
    for asset in ASSETS:
        close = native[asset].reindex(common_index)
        logret = returns[asset].reindex(common_index)
        frames.append(
            pd.DataFrame(
                {
                    "date": common_index,
                    "asset": asset,
                    "close": close.to_numpy(dtype="float64"),
                    "log_return": logret.to_numpy(dtype="float64"),
                    "source": "EIA",
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)

    # 5. Tipos exactos de la sección 6.
    df["asset"] = pd.Categorical(df["asset"], categories=list(ASSETS))
    df["source"] = pd.Categorical(df["source"], categories=list(SOURCES))

    # 6. Orden: por activo y fecha (índice temporal creciente dentro de c/activo).
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    return df


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara booleana de outliers: |log_return| > umbral."""
    return df["log_return"].abs() > OUTLIER_LOG_RETURN_THRESHOLD


# --------------------------------------------------------------------------- #
# Validación Pandera (esquema sección 6)
# --------------------------------------------------------------------------- #
def _dates_strictly_increasing(df: pd.DataFrame) -> bool:
    """True si dentro de cada activo las fechas son estrictamente crecientes."""
    per_asset = df.groupby("asset", observed=True)["date"].apply(
        lambda s: s.is_monotonic_increasing
    )
    return bool(per_asset.all())


def build_schema() -> pa.DataFrameSchema:
    """Esquema de validación del dataset final (sección 6)."""
    return pa.DataFrameSchema(
        columns={
            "date": pa.Column("datetime64[ns, UTC]", nullable=False),
            "asset": pa.Column(pd.CategoricalDtype(list(ASSETS)), nullable=False),
            "close": pa.Column(float, nullable=True),
            "log_return": pa.Column(float, nullable=True),
            "source": pa.Column(pd.CategoricalDtype(list(SOURCES)), nullable=False),
        },
        # Sin duplicados de fecha+activo.
        unique=["date", "asset"],
        # Sin columnas extra ni faltantes.
        strict=True,
        ordered=False,
        # Índice temporal estrictamente creciente por activo.
        checks=pa.Check(
            _dates_strictly_increasing,
            error="Las fechas no son estrictamente crecientes dentro de cada activo",
        ),
    )


def run() -> tuple[pd.DataFrame, pd.Series]:
    """Construye, valida y persiste `prices_daily.parquet`.

    Devuelve `(df_validado, mascara_outliers)`.
    """
    df = build_prices_daily()
    schema = build_schema()
    validated = schema.validate(df, lazy=True)  # lanza si algo no cumple

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(config.PRICES_DAILY_PARQUET, index=False)
    return validated, outlier_mask(validated)


# --------------------------------------------------------------------------- #
# Verificación
# --------------------------------------------------------------------------- #
def _report(df: pd.DataFrame, outliers: pd.Series) -> None:
    print("\n===== ESQUEMA DEL PARQUET =====")
    print(df.dtypes.to_string())

    print("\n===== FILAS POR ACTIVO =====")
    resumen = (
        df.groupby("asset", observed=True)
        .agg(
            filas=("close", "size"),
            close_no_nulos=("close", "count"),
            log_ret_no_nulos=("log_return", "count"),
        )
    )
    print(resumen.to_string())
    print(f"\nTotal de filas: {len(df)}")
    print(f"Rango de fechas: {df['date'].min().date()} -> {df['date'].max().date()}")

    nonpos = df[df["close"] <= 0]
    print(f"\n===== PRECIOS NO POSITIVOS (conservados, log_return=NaN) =====")
    print(f"Cantidad: {len(nonpos)}")
    if not nonpos.empty:
        print(nonpos[["date", "asset", "close"]].to_string(index=False))

    print(f"\n===== OUTLIERS (|log_return| > {OUTLIER_LOG_RETURN_THRESHOLD:.0%}) =====")
    flagged = df[outliers]
    print(f"Cantidad flaggeada (no borrada): {len(flagged)}")
    if not flagged.empty:
        print(
            flagged[["date", "asset", "close", "log_return"]]
            .sort_values("date")
            .to_string(index=False)
        )

    print("\n===== MUESTRA (WTI, primeras y últimas) =====")
    wti = df[df["asset"] == "WTI"]
    print(wti.head().to_string(index=False))
    print("...")
    print(wti.tail().to_string(index=False))


if __name__ == "__main__":
    validated, outliers = run()
    print(f"\nGuardado: {config.PRICES_DAILY_PARQUET}")
    _report(validated, outliers)
