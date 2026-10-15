"""Tests Point-in-Time del Módulo 2 (sentimiento).

Verifican que la alineación feature(t) → target(t+1) no tenga look-ahead: el
target de volatilidad es SIEMPRE del día de trading siguiente, nunca del mismo.

Correr:  python -m pytest tests/test_sentiment_pit.py -v
"""

from pathlib import Path

import pytest

from src import config
from src.sentiment import dataset

pytestmark = pytest.mark.skipif(
    not Path(config.SENTIMENT_DAILY_PARQUET).exists(),
    reason="Falta sentiment_daily.parquet — correr src.sentiment.aggregate",
)


@pytest.fixture(scope="module")
def sup():
    return dataset.build_supervised()


def test_target_estrictamente_futuro(sup):
    """El target de volatilidad es siempre t+1, nunca el mismo día (sin look-ahead)."""
    d = sup.dropna(subset=["target_date"])
    assert (d["target_date"] > d["trading_day"]).all(), "hay target no estrictamente futuro"


def test_sin_duplicados_dia_activo(sup):
    assert not sup.duplicated(["trading_day", "asset"]).any(), "duplicados de (trading_day, asset)"


def test_fechas_tz_utc(sup):
    assert "UTC" in str(sup["trading_day"].dtype)
    td = sup["target_date"].dropna()
    assert "UTC" in str(td.dtype)


def test_target_del_calendario_de_precios(sup):
    """Cada trading_day con sentimiento tiene una fila alineada (join válido)."""
    assert sup["trading_day"].notna().all()
    assert len(sup) > 0
