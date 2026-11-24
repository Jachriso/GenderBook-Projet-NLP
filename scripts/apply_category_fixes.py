"""
Apply the manual corrections (review of the 100 most frequent categories)
in data/reference/category_genre_map.csv, genre_manual column.

    python scripts/apply_category_fixes.py

Then: python -m scripts.run_enrichment   (the genre_manual column is kept on every rerun)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "reference" / "category_genre_map.csv"
BLOCKS = json.loads((ROOT / "data" / "reference" / "blocks.json").read_text(encoding="utf-8"))
VALID = set(BLOCKS["Genre"])

# category -> corrected genre(s), separated by "|". The reason for each correction is in a comment.
FIXES = {
    "Adventure Books": "Thriller & Suspense",                      # adult, not children's (710 books)
    "Short Story Books": "Literary Fiction",                       # short stories, not children's
    "Erotic Fiction": "Romance",
    "Military Fiction": "Historical Fiction",
    "Anthologies (non-poetry)": "Literary Fiction",
    "Biography: Literary": "Biography & Memoir",                   # removes Literary Fiction
    "Graphic Novels: Literary & Memoirs": "Graphic Novels & Comics|Biography & Memoir",
    "Family": "Literary Fiction",                                  # family novels (category without a genre)
    "Relationships": "Self-help & Psychology",
    "Family & Relationships": "Self-help & Psychology",
    "Political & Legal": "Thriller & Suspense",                    # political / legal thrillers
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
