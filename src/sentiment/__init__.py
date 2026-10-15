"""Módulo 2 — Sentimiento (FinBERT).

Nota (Windows): `torch` debe importarse ANTES que pandas/numpy, o falla la carga
de c10.dll (WinError 1114, conflicto de OpenMP). Por eso:
- `finbert.py` importa torch en su primera línea.
- Los scripts que USAN FinBERT (ej. `score_corpus.py`) importan `finbert` antes
  que pandas.
- Los módulos que NO usan torch (`dataset.py`, `aggregate.py`) no lo importan →
  sin conflicto. (Este `__init__` NO importa torch a propósito.)
"""
