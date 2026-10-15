"""Completa FinBERT (Fase 2) para biotech + financials y corre la tabla final.

Descarga títulos -> scorea (cap 8/día) -> ablation cross-sector con las tres
columnas FinBERT. Job largo (descargas de mega-corpus + scoring CPU). Resumible:
si se corta, re-correr retoma (news por ticker, scoring por chunk).

Orden Windows: score_titles importa torch primero. Correr:
    python -m src.sectors.run_finbert_rest
"""

from __future__ import annotations

from src.sectors import score_titles  # importa torch primero

from src.sectors import ablation, baskets, ingest  # noqa: E402

REST = ["biotech", "financials"]

if __name__ == "__main__":
    for sec in REST:
        print(f"\n########## {sec.upper()} — descarga de títulos ##########", flush=True)
        ingest.download_news(sec)
        print(f"########## {sec.upper()} — scoring FinBERT ##########", flush=True)
        score_titles.score_sector(sec, 8)

    print("\n########## TABLA FINAL CROSS-SECTOR (con FinBERT) ##########", flush=True)
    ablation.run_all()
