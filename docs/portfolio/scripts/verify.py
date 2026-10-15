"""Step 3 verification: every published claim must survive an automated check.

Checks performed:
  1. Length limits (headline <= 120 chars per option, project entry <= 2,000, article 900-1,300 words).
  2. Banned marketing vocabulary.
  3. Spanish leakage into English deliverables.
  4. Numeric claims cross-checked against the persisted CSVs, not against prose.

Usage: python docs/portfolio/scripts/verify.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "docs" / "portfolio"
DATA = P / "data"

BANNED = ["revolutionary", "cutting-edge", "cutting edge", "powerful", "leverage",
          "game-changing", "game changing", "deep dive", "unlock", "seamless",
          "robust solution", "state-of-the-art", "harness", "empower"]

SPANISH_HINTS = [" el ", " la ", " los ", " las ", " que ", " para ", " con ", " una ",
                 " del ", " por ", "ción ", " más ", " también "]

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def read(name: str) -> str:
    return (P / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
print("[1] length limits")
hl = read("headline.md")
sections = {}
for block in re.split(r"\n## ", hl)[1:]:
    title = block.split("\n", 1)[0].strip()
    sections[title] = re.findall(r"```\n(.+?)\n```", block, flags=re.S)

name_sec = next((k for k in sections if k.startswith("Project name")), None)
head_sec = next((k for k in sections if k.startswith("Profile headline")), None)
check(name_sec is not None and head_sec is not None,
      "headline.md has both a project-name and a profile-headline section")

for sec, limit in ((name_sec, 255), (head_sec, 120)):
    opts = sections.get(sec, [])
    check(len(opts) == 3, f"{sec}: 3 options (found {len(opts)})")
    for i, o in enumerate(opts, 1):
        n = len(o.strip())
        check(n <= limit, f"{sec} option {i}: {n} chars (limit {limit})")

pe = read("project_entry.md")
body = pe.split("\n", 2)[2] if pe.count("\n") > 1 else pe
check(len(body) <= 2000, f"project_entry.md body: {len(body)} chars (limit 2,000)")

art = read("article.md")
words = len(re.findall(r"\b[\w'-]+\b", re.sub(r"!\[.*?\]\(.*?\)", "", art)))
check(900 <= words <= 1300, f"article.md: {words} words (target 900-1,300)")

# --------------------------------------------------------------------------- #
print("\n[2] banned marketing vocabulary")
for f in ("headline.md", "project_entry.md", "article.md", "README_section.md"):
    txt = read(f).lower()
    hits = [w for w in BANNED if w in txt]
    check(not hits, f"{f}: no banned terms" + (f" — found {hits}" if hits else ""))

# --------------------------------------------------------------------------- #
print("\n[3] no Spanish in the English deliverables")
for f in ("headline.md", "project_entry.md", "article.md", "README_section.md"):
    txt = " " + read(f).lower().replace("\n", " ") + " "
    hits = [h.strip() for h in SPANISH_HINTS if h in txt]
    check(not hits, f"{f}: English only" + (f" — found {hits}" if hits else ""))

# --------------------------------------------------------------------------- #
print("\n[4] numeric claims vs persisted CSVs")
lvl = pd.read_csv(DATA / "energy_level_single_split.csv").set_index("model")["r2_oos"]
perm = pd.read_csv(DATA / "energy_permutation_level.csv")
wf = pd.read_csv(DATA / "energy_level_walkforward.csv")
wf = wf[~wf["skipped"]]
jmp = pd.read_csv(DATA / "energy_jumps_single_split.csv").set_index("model")
jwf = pd.read_csv(DATA / "energy_jumps_walkforward.csv")
jwf_ok = jwf[~jwf["skipped"]]
emac = pd.read_csv(DATA / "energy_jumps_walkforward_macro.csv")
emac_ok = emac[~emac["skipped"]]
sec = pd.read_csv(DATA / "sectors_walkforward.csv")
sec_ok = sec[~sec["skipped"]]
shape = pd.read_csv(DATA / "energy_panel_shape.csv").set_index("stage")["value"]
cnt = pd.read_csv(DATA / "dataset_counts.csv").set_index("claim")["value"]

vol_t = float(perm.loc[perm["feature"] == "vol_t", "importance_mean"].iloc[0])
best_fb = float(perm[perm["block"] == "finbert"]["importance_mean"].max())
fb_pool = jwf_ok["delta_finbert_M3_minus_M2"]
fb_macro = emac_ok["delta_finbert_M3_minus_M2"]
news_pool = jwf_ok["delta_news_M3_minus_M0"]

claims = [
    ("persistence R2 = 0.8061", round(float(lvl["persistence"]), 4), 0.8061),
    ("M0 = 0.7357", round(float(lvl["M0 precio"]), 4), 0.7357),
    ("M1 = 0.7310", round(float(lvl["M1 +volumen"]), 4), 0.7310),
    ("M2 = 0.7301", round(float(lvl["M2 +EODHD"]), 4), 0.7301),
    ("M3 = 0.7403", round(float(lvl["M3 +FinBERT"]), 4), 0.7403),
    ("M3-M2 single-split delta = 0.0102",
     round(float(lvl["M3 +FinBERT"] - lvl["M2 +EODHD"]), 4), 0.0102),
    ("vol_t importance = 0.5558", round(vol_t, 4), 0.5558),
    ("best FinBERT importance = 0.0014", round(best_fb, 4), 0.0014),
    ("ratio = 408x", round(vol_t / best_fb), 408),
    ("negative-importance features = 3", int((perm["importance_mean"] < 0).sum()), 3),
    ("level WF mean = -0.0016", round(float(wf["delta_M3_minus_M0"].mean()), 4), -0.0016),
    ("level WF positive folds = 3 of 6",
     f"{int((wf['delta_M3_minus_M0'] > 0).sum())} of {len(wf)}", "3 of 6"),
    ("jump baseline AUC = 0.6984", round(float(jmp.loc["baseline_vol_t", "roc_auc"]), 4), 0.6984),
    ("jump M1 AUC = 0.7384", round(float(jmp.loc["M1 +volumen", "roc_auc"]), 4), 0.7384),
    ("jump M3 AUC = 0.7524", round(float(jmp.loc["M3 +FinBERT", "roc_auc"]), 4), 0.7524),
    ("jump WF usable folds = 5", len(jwf_ok), 5),
    ("dropped fold positives = 9", int(jwf[jwf["skipped"]]["positives"].iloc[0]), 9),
    ("news WF mean = +0.0094", round(float(news_pool.mean()), 4), 0.0094),
    ("news WF positive = 3 of 5",
     f"{int((news_pool > 0).sum())} of {len(news_pool)}", "3 of 5"),
    ("FinBERT pooled mean = -0.0071", round(float(fb_pool.mean()), 4), -0.0071),
    ("FinBERT pooled positive = 2 of 5",
     f"{int((fb_pool > 0).sum())} of {len(fb_pool)}", "2 of 5"),
    ("FinBERT macro mean = +0.0037", round(float(fb_macro.mean()), 4), 0.0037),
    ("feature matrix rows = 4335", int(shape["feature_matrix_rows"]), 4335),
    ("features = 12", int(shape["n_features_M3"]), 12),
    ("train rows = 2965", int(shape["train_rows"]), 2965),
    ("test rows = 1370", int(shape["test_rows"]), 1370),
    ("panel start = 2020-01-02", str(shape["panel_start"]), "2020-01-02"),
    ("panel end = 2026-06-19", str(shape["panel_end"]), "2026-06-19"),
    ("price rows = 9633", int(cnt["price_panel_rows"]), 9633),
    ("unique articles = 45834", int(cnt["corpus_unique_articles"]), 45834),
    ("article-asset rows = 84692", int(cnt["corpus_rows_article_x_asset"]), 84692),
]

for sector, expected_pos, expected_n in (("tech", 1, 6), ("biotech", 1, 6)):
    g = sec_ok[(sec_ok["sector"] == sector) & sec_ok["delta_finbert_M3_minus_M2"].notna()]
    v = g["delta_finbert_M3_minus_M2"]
    claims.append((f"{sector} FinBERT positive folds = {expected_pos} of {expected_n}",
                   f"{int((v > 0).sum())} of {len(v)}", f"{expected_pos} of {expected_n}"))

fin = sec_ok[sec_ok["sector"] == "financials"]["delta_volume_M1_minus_M0"]
claims.append(("financials volume mean = +0.0315", round(float(fin.mean()), 4), 0.0315))
claims.append(("financials volume positive = 3 of 5",
               f"{int((fin > 0).sum())} of {len(fin)}", "3 of 5"))

for label, got, want in claims:
    check(got == want, f"{label}  (file says {got})")

# --------------------------------------------------------------------------- #
print("\n[5] over-claim scan")
for f in ("headline.md", "project_entry.md", "article.md", "README_section.md"):
    t = read(f).lower()
    bad = []
    if "sentiment does not predict volatility" in t or "sentiment is useless" in t:
        bad.append("over-generalized claim")
    if "four sectors" in t and "finbert" in t.split("four sectors")[0][-240:]:
        bad.append("FinBERT attributed to four sectors (isolated in three)")
    if re.search(r"the model (works|succeeds|beats)", t):
        bad.append("claims the model works")
    check(not bad, f"{f}: scope is correct" + (f" — {bad}" if bad else ""))

print("\n" + "=" * 62)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
