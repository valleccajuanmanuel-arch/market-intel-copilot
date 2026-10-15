"""Canastas de tickers por sector + rutas de datos de la expansión multi-sector.

Elección de sectores (para barrer el espectro de la hipótesis sentimiento->precio):
- tech       : narrativa/hype/atención retail   -> a priori el mejor caso.
- biotech    : eventos binarios (FDA/trials)     -> shocks por noticia (frame de saltos).
- financials : titulares + macro, alta cobertura -> caso intermedio.
La energía (Módulo 3) queda como referencia negativa (mercado físico).
"""

from __future__ import annotations

from src import config

SECTORS: dict[str, list[str]] = {
    "tech": ["NVDA.US", "AAPL.US", "MSFT.US", "AMD.US", "META.US", "TSLA.US"],
    "biotech": ["MRNA.US", "BNTX.US", "PFE.US", "GILD.US", "REGN.US", "VRTX.US"],
    "financials": ["JPM.US", "GS.US", "BAC.US", "MS.US", "WFC.US", "C.US"],
}

START = "2020-01-01"   # misma ventana que el corpus de energía (comparabilidad)

SECTORS_RAW = config.DATA_DIR / "raw" / "sectors"
SECTORS_PROCESSED = config.DATA_DIR / "processed" / "sectors"


def prices_dir(sector: str):
    return SECTORS_RAW / sector / "prices"


def news_dir(sector: str):
    return SECTORS_RAW / sector / "news"


def titles_scored_path(sector: str):
    """FinBERT sobre títulos (opcional; si no existe, se corre M0-M2 nomás)."""
    return SECTORS_PROCESSED / f"{sector}_titles_scored.parquet"


def asset_of(ticker: str) -> str:
    """`NVDA.US` -> `NVDA` (nombre de activo en el panel)."""
    return ticker.split(".")[0]
