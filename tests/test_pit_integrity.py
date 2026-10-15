"""Tests de integridad Point-in-Time (spec sección 5).

Validan el dataset de precios ya generado y dejan documentado que los
chequeos de noticias están inactivos hasta tener el corpus.

Correr desde la raíz del proyecto:
    python -m pytest tests/test_pit_integrity.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.validation import pit_checks

# Si todavía no se generó el parquet de precios, saltear todo el módulo con un
# motivo claro en vez de fallar de forma confusa.
pytestmark = pytest.mark.skipif(
    not Path(config.PRICES_DAILY_PARQUET).exists(),
    reason="Falta prices_daily.parquet — correr primero src.cleaning.price_cleaner",
)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return pd.read_parquet(config.PRICES_DAILY_PARQUET)


@pytest.fixture(scope="module")
def raw_close() -> dict:
    return pit_checks.load_raw_close()


# --------------------------------------------------------------------------- #
# Precios
# --------------------------------------------------------------------------- #
def test_indice_unico_y_estrictamente_creciente(prices):
    result = pit_checks.check_unique_and_strictly_increasing(prices)
    assert result.passed, result.detail


def test_sin_nan_silenciosos_en_columnas_criticas(prices, raw_close):
    result = pit_checks.check_no_silent_nans(prices, raw_close)
    assert result.passed, result.detail


def test_log_return_usa_pasado_no_futuro(prices, raw_close):
    result = pit_checks.check_log_return_lag(prices, raw_close)
    assert result.passed, result.detail


def test_todos_los_chequeos_de_precios_en_verde(prices, raw_close):
    results = pit_checks.run_price_checks(prices, raw_close)
    fallidos = [r for r in results if not r.passed]
    assert not fallidos, "; ".join(str(r) for r in fallidos)


# --------------------------------------------------------------------------- #
# Noticias (inactivos por ahora)
# --------------------------------------------------------------------------- #
def test_news_pit_checks():
    """Si existe news_corpus.parquet, los chequeos PiT de noticias deben pasar;
    si todavía no existe, quedan en SKIP (también passed=True)."""
    results = pit_checks.run_news_checks()
    assert results, "run_news_checks no devolvió ningún resultado"
    fallidos = [r for r in results if not r.passed]
    assert not fallidos, "; ".join(str(r) for r in fallidos)
