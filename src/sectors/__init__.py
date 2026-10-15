"""Expansión multi-sector del súper test (tech, biotech, financials).

Generaliza el Módulo 3 a acciones: cada ticker es su propio activo (serie de
precio EOD de EODHD + sus propias noticias). Mismo ablation M0->M3 y walk-forward
que energía, para decidir POR SECTOR si el sentimiento / FinBERT aporta.

Sin torch salvo el scorer de títulos (`score_titles.py`), que lo importa primero.
"""
