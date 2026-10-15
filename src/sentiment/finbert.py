"""Sentimiento financiero con FinBERT (ProsusAI/finbert, Apache-2.0) — Módulo 2.

Scorea texto financiero (título y/o cuerpo de los artículos del corpus) y
devuelve, por texto:
- `label`: positive / negative / neutral (argmax de la softmax)
- `prob_positive` / `prob_negative` / `prob_neutral`: probabilidades
- `sentiment_score`: señal continua en [-1, 1] = P(pos) - P(neg), lista para
  agregar por (trading_day, asset) y alimentar el modelo de volatilidad.

FinBERT es BERT (límite 512 tokens): los textos largos se **truncan** a 512.
El modelo se carga una sola vez (`lru_cache`). Inferencia en CPU.

Decisión pendiente (Módulo 2): scorear el **título**, el **cuerpo** (truncado), o
ambos. `score_texts` es agnóstico; el caller decide qué texto pasarle.
"""

from __future__ import annotations

# IMPORTANTE (Windows): torch DEBE importarse antes que pandas/numpy, o falla la
# carga de c10.dll (WinError 1114, conflicto de runtime OpenMP). No reordenar.
import torch

from functools import lru_cache

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _load():
    """Carga tokenizer + modelo una sola vez (se cachea)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def score_texts(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_TOKENS,
) -> pd.DataFrame:
    """Scorea una lista de textos. Devuelve un DataFrame (una fila por texto).

    `max_length`: tope de tokens (los textos más largos se truncan). Bajarlo
    acelera mucho en CPU (ej. 256 para cuerpos captura el lead del artículo).
    """
    tokenizer, model = _load()
    # id2label del modelo (ProsusAI: 0=positive, 1=negative, 2=neutral), sin hardcodear.
    id2label = {i: lbl.lower() for i, lbl in model.config.id2label.items()}

    out: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = [
            t if isinstance(t, str) and t.strip() else "[EMPTY]"
            for t in texts[start:start + batch_size]
        ]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        for row in probs:
            d = {id2label[j]: float(row[j]) for j in range(row.shape[0])}
            out.append(
                {
                    "label": max(d, key=d.get),
                    "prob_positive": d.get("positive", 0.0),
                    "prob_negative": d.get("negative", 0.0),
                    "prob_neutral": d.get("neutral", 0.0),
                    "sentiment_score": d.get("positive", 0.0) - d.get("negative", 0.0),
                }
            )
    return pd.DataFrame(out)


if __name__ == "__main__":
    # Demo mínima (dispara la descarga del modelo la primera vez, ~440 MB).
    demo = [
        "Oil prices surged after OPEC announced deeper production cuts.",
        "Crude inventories rose sharply, pressuring oil prices lower.",
        "The company reported quarterly earnings in line with expectations.",
    ]
    res = score_texts(demo)
    for text, (_, r) in zip(demo, res.iterrows()):
        print(f"[{r['label']:>8}] score={r['sentiment_score']:+.3f}  {text}")
