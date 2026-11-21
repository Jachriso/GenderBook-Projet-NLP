"""Balayage de z_min et top_k pour le bloc Themes, évalué sur gold_60_clean.csv.
Réutilise les cosinus bruts stockés dans label_scores : pas de ré-encodage."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en

BLOCK = "Themes"
enr = pd.read_parquet(ROOT / "data/processed/books_enriched.parquet")
enr["book_id"] = enr["book_id"].astype(str)
labels = list(en.load_blocks()[BLOCK].keys())

def parse(x):
    return json.loads(x) if isinstance(x, str) else x
first = parse(enr["label_scores"].iloc[0])
print("blocs dans label_scores :", list(first.keys()))
sims = np.array([[parse(s)[BLOCK][l] for l in labels] for s in enr["label_scores"]])
z = en.zscore_columns(sims)

gold = pd.read_csv(ROOT / "data/reference/gold_60_clean.csv", dtype=str, keep_default_na=False)
pos = {b: i for i, b in enumerate(enr["book_id"])}
gold = gold[gold["book_id"].isin(pos)]
truth = [{x.strip() for x in g.split("|") if x.strip()} for g in gold["gold_themes"]]
idx = [pos[b] for b in gold["book_id"]]

rows = []
for top_k in (1, 2, 3):
    for z_min in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        chosen = en.select_labels(z[idx], labels, top_k, z_min, 1)
        P = np.mean([len(set(c) & t) / len(c) for c, t in zip(chosen, truth)])
        R = np.mean([len(set(c) & t) / len(t) for c, t in zip(chosen, truth)])
        H = np.mean([len(set(c) & t) > 0 for c, t in zip(chosen, truth)])
        rows.append(dict(top_k=top_k, z_min=z_min, labels_par_livre=round(np.mean([len(c) for c in chosen]), 2),
                         precision=round(P, 2), rappel=round(R, 2), F1=round(2 * P * R / (P + R + 1e-9), 2),
                         au_moins_1=f"{H:.0%}"))
print(pd.DataFrame(rows).sort_values("F1", ascending=False).to_string(index=False))
