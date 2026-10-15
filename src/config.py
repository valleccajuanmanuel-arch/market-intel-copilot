"""Configuración central del proyecto.

Carga las API keys desde el archivo .env (vía python-dotenv) y expone las
rutas de las carpetas de datos como constantes. Ningún otro módulo debería
construir paths a mano ni leer os.environ directamente: todo pasa por acá.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Rutas base
# --------------------------------------------------------------------------- #
# config.py vive en src/, así que la raíz del proyecto es un nivel más arriba.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargar variables de entorno desde el .env de la raíz del proyecto.
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Carpetas de datos (ver sección 1 del spec)
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Subcarpetas de raw/ por fuente.
RAW_PRICES_EIA_DIR = RAW_DIR / "prices" / "eia"
RAW_PRICES_FRED_DIR = RAW_DIR / "prices" / "fred"
RAW_PRICES_EODHD_DIR = RAW_DIR / "prices" / "eodhd"
RAW_NEWS_GDELT_DIR = RAW_DIR / "news" / "gdelt"
RAW_NEWS_EODHD_DIR = RAW_DIR / "news" / "eodhd"
RAW_NEWS_APIFY_DIR = RAW_DIR / "news" / "apify"
RAW_NEWS_EIA_WEEKLY_DIR = RAW_DIR / "news" / "eia_weekly"

# Datasets finales (sección 6 del spec).
PRICES_DAILY_PARQUET = PROCESSED_DIR / "prices_daily.parquet"
NEWS_CORPUS_PARQUET = PROCESSED_DIR / "news_corpus.parquet"

# Módulo 2 — Sentimiento
SENTIMENT_ARTICLES_PARQUET = PROCESSED_DIR / "sentiment_articles.parquet"
SENTIMENT_DAILY_PARQUET = PROCESSED_DIR / "sentiment_daily.parquet"

# --------------------------------------------------------------------------- #
# API keys (ver secciones 3 y 8 del spec)
# --------------------------------------------------------------------------- #
EIA_API_KEY = os.getenv("EIA_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")


def require_key(name: str) -> str:
    """Devuelve la API key pedida o lanza un error claro si falta.

    Útil para que los módulos de ingestion fallen temprano y con un mensaje
    entendible en vez de mandar un request con key=None.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno {name!r}. "
            f"Copiá .env.example a .env y completá las keys."
        )
    return value


if __name__ == "__main__":
    # Diagnóstico rápido: ¿se cargaron las keys y existen las carpetas?
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}  (existe: {DATA_DIR.exists()})")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}  (existe: {PROCESSED_DIR.exists()})")
    print()
    print("Estado de las API keys (no se imprime el valor por seguridad):")
    for key_name in ("EIA_API_KEY", "FRED_API_KEY", "EODHD_API_TOKEN", "APIFY_API_TOKEN"):
        cargada = bool(os.getenv(key_name))
        print(f"  {key_name:<18}: {'OK (cargada)' if cargada else 'FALTA'}")
