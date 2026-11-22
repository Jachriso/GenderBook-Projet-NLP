"""
Applique les corrections manuelles (relecture des 100 catégories les plus fréquentes)
dans data/reference/category_genre_map.csv, colonne genre_manual.

    python scripts/apply_category_fixes.py

Ensuite : python -m scripts.run_enrichment   (la colonne genre_manual est conservée à chaque relance)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "reference" / "category_genre_map.csv"
BLOCKS = json.loads((ROOT / "data" / "reference" / "blocks.json").read_text(encoding="utf-8"))
VALID = set(BLOCKS["Genre"])

# catégorie -> genre(s) corrigé(s), séparés par « | ». Raison de chaque correction en commentaire.
FIXES = {
    "Adventure Books": "Thriller & Suspense",                      # adulte, pas jeunesse (710 livres)
    "Short Story Books": "Literary Fiction",                       # nouvelles, pas jeunesse
    "Erotic Fiction": "Romance",
    "Military Fiction": "Historical Fiction",
    "Anthologies (non-poetry)": "Literary Fiction",
    "Biography: Literary": "Biography & Memoir",                   # retire Literary Fiction
    "Graphic Novels: Literary & Memoirs": "Graphic Novels & Comics|Biography & Memoir",
    "Family": "Literary Fiction",                                  # romans familiaux (catégorie sans genre)
    "Relationships": "Self-help & Psychology",
    "Family & Relationships": "Self-help & Psychology",
    "Political & Legal": "Thriller & Suspense",                    # thrillers politiques / judiciaires
    "Politics & Government": "History",
    "Political Science & Theory": "History",
    "True Crime Books": "Biography & Memoir",                      # non-fiction
    "English Language: Reading Schemes": "Children's",
    "Advice On Careers & Achieving Success": "Business & Economics",
    "Historical Crime": "Mystery & Detective|Historical Fiction",
    "People & Places": "History",
}

bad = {g for v in FIXES.values() for g in v.split("|")} - VALID
assert not bad, f"genres inconnus : {bad}"

df = pd.read_csv(CSV, keep_default_na=False)
missing = [c for c in FIXES if c not in set(df["category"])]
for cat, genres in FIXES.items():
    df.loc[df["category"] == cat, "genre_manual"] = genres
df.to_csv(CSV, index=False)
print(f"{len(FIXES) - len(missing)} catégories corrigées dans {CSV.name}")
if missing:
    print("introuvables (ignorées) :", missing)
