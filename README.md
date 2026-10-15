# Market Intelligence Copilot

**Does news sentiment improve volatility forecasting? A negative result.**

I tested whether news sentiment, measured with FinBERT, improves next-day volatility forecasts
beyond what price history already provides. It does not. This repository holds the data pipeline,
the experiment, and the writeup.

The result is negative on purpose. What the project demonstrates is the design: point-in-time
discipline, an ablation that isolates each feature block, temporal validation, and a deliberate
attempt to falsify my own conclusion across four asset universes.

| Measure | Value |
|---|---|
| Persistence baseline, out-of-sample R² | **0.8061** |
| Best model including FinBERT | 0.7403 |
| Permutation importance: realized volatility vs. strongest FinBERT feature | 0.5558 vs. 0.0014 (**408x**) |
| Walk-forward ΔR² from the FinBERT block | **−0.0016**, positive in 3 of 6 folds |
| Universes tested | energy, technology, biotechnology, financials |

📄 **Full writeup:** [docs/portfolio/article.md](docs/portfolio/article.md)

---

## The finding in two figures

![Persistence versus models](docs/portfolio/figures/02_persistence_vs_models.png)

Out-of-sample R² on next-day realized volatility. A persistence baseline — tomorrow's volatility
equals today's — beats every model, including the one with sentiment.

![Walk-forward by fold](docs/portfolio/figures/04_walkforward_by_fold.png)

ΔR² from adding the FinBERT block, fold by fold. The sign alternates. That is what noise looks
like, and it is why a single favourable split is not evidence.

---

## Pipeline

![Pipeline architecture](docs/portfolio/figures/01_architecture.png)

End to end, with the data volume at each stage. The blue badges mark where point-in-time
discipline is enforced: no forward-fill across trading calendars, features built at day *t* with
the target at *t+1*, and temporal validation rather than random cross-validation.

---

## What is in this repository

| Path | Contents |
|---|---|
| `src/ingestion` | Data acquisition from EIA, FRED and EODHD. Resumable pulls, raw payloads kept immutable. |
| `src/cleaning` | Point-in-time cleaning. No forward-fill, log returns on the native series, non-positive prices preserved, outliers flagged rather than dropped. |
| `src/sentiment` | FinBERT scoring of 45,834 unique articles (title and body) and daily aggregation to asset-days. |
| `src/model` | Feature matrix, the M0–M3 ablation, and the walk-forward evaluation. |
| `src/sectors` | The same experiment generalized to three equity universes. |
| `src/validation` | Point-in-time checks. |
| `notebooks/01`–`04` | Data exploration, corpus, sentiment, and the volatility study. |
| `docs/portfolio/article.md` | The writeup. |
| `docs/portfolio/figures/` | Figures, 300 dpi PNG and SVG, plus 1080×1080 versions. |
| `docs/portfolio/data/` | The metric CSVs behind every figure. |
| `docs/portfolio/scripts/` | Scripts that regenerate the metrics and verify every published claim. |

---

## Reproducing the numbers

No figure hardcodes a value. Every one reads from `docs/portfolio/data/*.csv`, which is
regenerated from the parquet datasets by a single script, which in turn imports `src/` without
modifying it.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python docs/portfolio/scripts/compute_metrics.py all     # metrics  -> docs/portfolio/data/
python docs/portfolio/figures/scripts/make_figures.py    # figures  -> docs/portfolio/figures/
python docs/portfolio/scripts/verify.py                  # checks every published claim
```

`verify.py` cross-checks each numeric claim in the writeup against the CSV it came from, enforces
the length limits, and scans for over-claiming — for example, it fails if the text says the model
works, or attributes the FinBERT result to four sectors when the block was isolated in three.

### Running the experiments directly

```bash
python -m src.model.train         # level target: ablation + walk-forward
python -m src.model.train_jumps   # jump target: classification + walk-forward
python -m src.sectors.ablation    # multi-sector generalization
```

---

## Data is not versioned

`data/` is excluded from the repository. The news corpus alone is 194 MB, past GitHub's file
limit, and the raw payloads are redownloadable from their sources. To rebuild it you need API
credentials in a `.env` file — copy `.env.example` and fill in your own keys:

```
EIA_API_KEY=...
FRED_API_KEY=...
EODHD_API_TOKEN=...
```

The EODHD news endpoints require a paid plan. The financials basket in the generalization test was
never downloaded for that reason, which the writeup states explicitly rather than working around.

---

## Scope of the claim

No effect was found, with this design, on these data. That is not the same as "sentiment is
uninformative".

The corpus skews toward press releases and trade coverage, so tone is compressed. Source breadth
is narrow. I used one volatility definition and one model family. Several walk-forward folds carry
fewer than thirty positive cases, so per-fold estimates are noisy by construction. The FinBERT
block itself was isolated in three universes, not four.

The writeup lists these limitations before a reader has to find them, and says what I would do
differently: the current work in this area measures *change between documents over time* rather
than tone, which is a different variable than the one I tested.

---

## Stack

Python 3.12 · pandas · scikit-learn (`HistGradientBoosting`) · PyTorch + Transformers
(FinBERT, `ProsusAI/finbert`) · pyarrow · matplotlib
