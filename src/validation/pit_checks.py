"""Chequeos de integridad Point-in-Time (spec sección 5).

Cada chequeo es una función pura que devuelve un `CheckResult` (no lanza
excepciones), de modo que sirve igual para los tests de pytest y para correr
el módulo standalone.

Filosofía: los chequeos **re-derivan desde el crudo** de `data/raw/` y
**recalculan** los log-retornos de forma independiente, sin reutilizar la
lógica del cleaner. Así la validación es un control real contra la fuente de
verdad y no una tautología sobre el propio código que la generó.

Estado:
- Precios: activos (prices_daily.parquet).
- Noticias: estructura preparada pero INACTIVA hasta tener news_corpus.parquet
  (ver `check_news_published_before_target` / `run_news_checks`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
# Reutilizamos SOLO el lector de crudos y las constantes; los retornos los
# recalculamos acá de forma independiente.
from src.cleaning.price_cleaner import ASSETS, START_DATE, _load_raw_asset

# Tolerancia para comparar log-retornos recalculados vs. guardados.
LOG_RETURN_TOL: float = 1e-9


@dataclass
class CheckResult:
    """Resultado de un chequeo individual."""

    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    @property
    def mark(self) -> str:
        return "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")

    def __str__(self) -> str:
        return f"[{self.mark}] {self.name}: {self.detail}"


# --------------------------------------------------------------------------- #
# Carga de la fuente de verdad (crudo)
# --------------------------------------------------------------------------- #
def load_raw_close() -> dict[str, pd.Series]:
    """Series de cierre nativas (historia completa) por activo, desde el crudo."""
    return {asset: _load_raw_asset(asset) for asset in ASSETS}


def _window_start() -> pd.Timestamp | None:
    return pd.Timestamp(START_DATE, tz="UTC") if START_DATE is not None else None


# --------------------------------------------------------------------------- #
# Chequeos de PRECIOS
# --------------------------------------------------------------------------- #
def check_unique_and_strictly_increasing(df: pd.DataFrame) -> CheckResult:
    """Índice temporal estrictamente creciente y sin duplicados de fecha+activo."""
    problems: list[str] = []

    dups = int(df.duplicated(subset=["date", "asset"]).sum())
    if dups:
        problems.append(f"{dups} duplicados de (date, asset)")

    for asset in ASSETS:
        fechas = df.loc[df["asset"] == asset, "date"]
        if not fechas.is_unique:
            problems.append(f"{asset}: fechas duplicadas")
        if not fechas.is_monotonic_increasing:
            problems.append(f"{asset}: fechas no estrictamente crecientes")

    passed = not problems
    detail = (
        "sin duplicados y estrictamente creciente dentro de cada activo"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("indice_unico_y_creciente", passed, detail)


def check_no_silent_nans(df: pd.DataFrame, raw_close: dict[str, pd.Series]) -> CheckResult:
    """No hay NaN silenciosos en columnas críticas.

    - `date`, `asset`, `source` no pueden ser NaN nunca.
    - `close` solo puede ser NaN en días en que el activo NO operó. Si el crudo
      tiene un valor para esa fecha (dentro de la ventana) y el procesado quedó
      NaN, es un NaN silencioso (dato perdido en la limpieza) → falla.
    """
    problems: list[str] = []

    for col in ("date", "asset", "source"):
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"{n} NaN en columna crítica '{col}'")

    win_start = _window_start()
    for asset in ASSETS:
        sub = df[df["asset"] == asset]
        proc_con_dato = set(sub.loc[sub["close"].notna(), "date"])

        raw = raw_close[asset].dropna()
        if win_start is not None:
            raw = raw[raw.index >= win_start]
        # Acotar al rango efectivo del procesado.
        raw = raw[(raw.index >= sub["date"].min()) & (raw.index <= sub["date"].max())]

        faltantes = set(raw.index) - proc_con_dato
        if faltantes:
            ejemplos = [d.date().isoformat() for d in sorted(faltantes)[:3]]
            problems.append(
                f"{asset}: {len(faltantes)} fechas con dato en crudo pero close NaN "
                f"(ej. {ejemplos})"
            )

    passed = not problems
    detail = (
        "columnas críticas sin NaN; todo NaN de close coincide con un día sin trading"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("sin_nan_silenciosos", passed, detail)


def check_log_return_lag(
    df: pd.DataFrame,
    raw_close: dict[str, pd.Series],
    tol: float = LOG_RETURN_TOL,
) -> CheckResult:
    """Los log-retornos usan P_t y P_{t-1}, nunca P_{t+1}.

    Recalcula desde el crudo la versión correcta (backward) `ln(P_t/P_{t-1})`
    y la versión con look-ahead (forward) `ln(P_{t+1}/P_t)`. Verifica que lo
    guardado coincide con la backward y que difiere de la forward (prueba de
    que NO mira el futuro).
    """
    backward: dict[str, pd.Series] = {}
    forward: dict[str, pd.Series] = {}
    for asset, serie in raw_close.items():
        serie = serie.sort_index()
        rb = serie / serie.shift(1)          # P_t / P_{t-1}
        rf = serie.shift(-1) / serie         # P_{t+1} / P_t  (look-ahead)
        backward[asset] = np.log(rb.where(rb > 0))
        forward[asset] = np.log(rf.where(rf > 0))

    problems: list[str] = []
    max_diff = 0.0
    comparadas = 0
    difiere_de_forward = 0

    for asset in ASSETS:
        sub = df[df["asset"] == asset].sort_values("date")
        idx = pd.DatetimeIndex(sub["date"])
        guardado = pd.Series(sub["log_return"].to_numpy(), index=idx)

        esperado_b = backward[asset].reindex(idx)
        esperado_f = forward[asset].reindex(idx)

        comparables = guardado.notna() & esperado_b.notna()
        comparadas += int(comparables.sum())
        if comparables.any():
            d = float((guardado[comparables] - esperado_b[comparables]).abs().max())
            max_diff = max(max_diff, d)
            if d > tol:
                problems.append(f"{asset}: no coincide con ln(P_t/P_(t-1)) (max dif {d:.2e})")

        # Evidencia anti look-ahead: debe diferir de la versión forward.
        cmp_f = guardado.notna() & esperado_f.notna()
        if cmp_f.any():
            difiere_de_forward += int(((guardado[cmp_f] - esperado_f[cmp_f]).abs() > tol).sum())

        # Un retorno guardado sin predecesor válido sería sospechoso (borde/look-ahead).
        huerfanos = int((guardado.notna() & esperado_b.isna()).sum())
        if huerfanos:
            problems.append(f"{asset}: {huerfanos} retornos sin P_(t-1) válido")

    passed = not problems
    detail = (
        f"log_return == ln(P_t/P_(t-1)) en {comparadas} filas (max dif {max_diff:.2e}); "
        f"difiere de la versión forward en {difiere_de_forward} filas -> no usa P_(t+1)"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("log_return_desfasado_correcto", passed, detail)


def run_price_checks(
    df: pd.DataFrame | None = None,
    raw_close: dict[str, pd.Series] | None = None,
) -> list[CheckResult]:
    """Corre todos los chequeos PiT de precios."""
    if df is None:
        df = pd.read_parquet(config.PRICES_DAILY_PARQUET)
    if raw_close is None:
        raw_close = load_raw_close()
    return [
        check_unique_and_strictly_increasing(df),
        check_no_silent_nans(df, raw_close),
        check_log_return_lag(df, raw_close),
    ]


# --------------------------------------------------------------------------- #
# Chequeos de NOTICIAS  (PREPARADO PERO INACTIVO)
# --------------------------------------------------------------------------- #
def check_news_published_before_target(
    news_df: pd.DataFrame,
    *,
    cutoff_hour_utc: int = 20,
) -> CheckResult:
    """PiT del corpus: `published_at` nunca posterior al target que predice.

    Verifica los invariantes PiT a nivel corpus:
      - `published_at` y `trading_day` UTC, sin NaT.
      - Ninguna noticia se asigna a un `trading_day` anterior al día de su
        `published_at` (eso sería look-ahead).
      - El cutoff se respetó: una noticia con `published_at` >= cutoff (20:00
        UTC) va a un `trading_day` estrictamente posterior a su día.

    (La regla feature(t) -> target(t+1) se verifica al construir el dataset de
    entrenamiento en el Módulo 2, no acá: este corpus aún no tiene targets.)
    """
    problems: list[str] = []

    if news_df["published_at"].isna().any():
        problems.append("published_at con NaT")
    if news_df["trading_day"].isna().any():
        problems.append("trading_day con NaT")

    pub_day = news_df["published_at"].dt.tz_convert("UTC").dt.normalize()
    trad_day = news_df["trading_day"].dt.tz_convert("UTC").dt.normalize()

    look_ahead = int((trad_day < pub_day).sum())
    if look_ahead:
        problems.append(f"{look_ahead} filas con trading_day ANTERIOR a published_at (look-ahead)")

    post_cutoff = news_df["published_at"].dt.tz_convert("UTC").dt.hour >= cutoff_hour_utc
    mal_cutoff = int((post_cutoff & (trad_day <= pub_day)).sum())
    if mal_cutoff:
        problems.append(
            f"{mal_cutoff} filas post-cutoff ({cutoff_hour_utc}:00 UTC) no movidas al día siguiente"
        )

    passed = not problems
    detail = (
        f"{len(news_df)} filas: published_at <= trading_day y cutoff {cutoff_hour_utc}:00 UTC respetado (sin look-ahead)"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("news_pit_published_before_target", passed, detail)


def run_news_checks(news_path: Path | None = None) -> list[CheckResult]:
    """Corre los chequeos PiT de noticias si el corpus existe; si no, los saltea."""
    news_path = Path(news_path) if news_path else config.NEWS_CORPUS_PARQUET
    if not news_path.exists():
        return [
            CheckResult(
                name="news_pit_published_before_target",
                passed=True,
                skipped=True,
                detail=f"inactivo: todavía no existe {news_path.name}",
            )
        ]
    news = pd.read_parquet(news_path)
    return [check_news_published_before_target(news)]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_all() -> list[CheckResult]:
    return run_price_checks() + run_news_checks()


if __name__ == "__main__":
    import sys

    resultados = run_all()
    todo_ok = True
    for r in resultados:
        print(r)
        if not r.passed and not r.skipped:
            todo_ok = False

    print("\n[OK] TODOS LOS CHECKS PiT EN VERDE" if todo_ok else "\n[X] HAY CHECKS PiT EN ROJO")
    sys.exit(0 if todo_ok else 1)
