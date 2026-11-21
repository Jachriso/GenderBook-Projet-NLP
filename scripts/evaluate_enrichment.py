"""Valide l'enrichissement P2 contre l'annotation manuelle (gold_60_clean.csv).
Usage : python scripts/evaluate_enrichment.py"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
gold = pd.read_csv(ROOT / "data/reference/gold_60_clean.csv", dtype=str, keep_default_na=False)
enr = pd.read_parquet(ROOT / "data/processed/books_enriched.parquet")
enr["book_id"] = enr["book_id"].astype(str)
enr = enr.set_index("book_id")

def split(s):
    return {x.strip() for x in s.split("|") if x.strip()}

rows = []
for _, g in gold.iterrows():
    if g["book_id"] not in enr.index:
        print("book_id absent du catalogue :", g["book_id"]); continue
    e = enr.loc[g["book_id"]]
    row = {"book_id": g["book_id"], "title": g["title"][:40]}
    for col, gcol in [("genres", "gold_genres"), ("themes", "gold_themes")]:
        pred, truth = set(e[col]), split(g[gcol])
        inter = pred & truth
        row[f"{col}_pred"] = " | ".join(sorted(pred))
        row[f"{col}_gold"] = " | ".join(sorted(truth))
        row[f"{col}_precision"] = len(inter) / len(pred) if pred else 0
        row[f"{col}_recall"] = len(inter) / len(truth) if truth else 0
        row[f"{col}_hit"] = int(len(inter) > 0)
    rows.append(row)

r = pd.DataFrame(rows)
r.to_csv(ROOT / "data/reference/gold_eval_report.csv", index=False)
print(f"{len(r)} livres évalués")
for col in ["genres", "themes"]:
    print(f"{col:7s} precision={r[col+'_precision'].mean():.2f}  "
          f"recall={r[col+'_recall'].mean():.2f}  au moins 1 bon label={r[col+'_hit'].mean():.0%}")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 55)
print("\nLivres sans aucun genre correct :")
print(r[r["genres_hit"] == 0][["title", "genres_pred", "genres_gold"]].to_string())
print("\nLivres sans aucun thème correct (10 premiers) :")
print(r[r["themes_hit"] == 0][["title", "themes_pred", "themes_gold"]].head(10).to_string())
