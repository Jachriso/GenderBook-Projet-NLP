"""
Fabrique data/processed/books_enriched_sample.parquet comme SOUS-ENSEMBLE EXACT de books_enriched.parquet.

Pourquoi : l'ancien échantillon était ré-enrichi sur 500 livres isolés ; la normalisation par label dépend du catalogue
entier, donc ses labels différaient de ceux du fichier complet (seulement 32-38 % identiques). Ici on ne recalcule rien :
on extrait les 500 livres de books_sample.parquet (P1) du fichier complet, puis on vérifie l'égalité.

    python scripts/make_sample.py          # à lancer APRES python -m scripts.run_enrichment
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en  # noqa: E402

PROC = ROOT / "data" / "processed"
LABEL_COLS = ["genres", "themes", "styles", "atmospheres", "keywords", "label_scores"]


def main() -> None:
    sample_ids = pd.read_parquet(PROC / "books_sample.parquet", columns=["book_id"])["book_id"].astype(str).tolist()
    full = pd.read_parquet(PROC / "books_enriched.parquet")
    sub = en.subset_enriched(full, sample_ids)
    out = PROC / "books_enriched_sample.parquet"
    en.save_enriched(sub, out)

    # vérification : relecture du fichier écrit et comparaison cellule par cellule avec le fichier complet
    chk = pd.read_parquet(out)
    ref = full.assign(book_id=full["book_id"].astype(str)).set_index("book_id").loc[chk["book_id"].astype(str)]
    norm = lambda v: v if isinstance(v, str) else [str(x) for x in v]
    for col in LABEL_COLS:
        same = sum(norm(a) == norm(b) for a, b in zip(chk[col], ref[col]))
        assert same == len(chk), f"{col} : {len(chk) - same} lignes différentes du fichier complet"
    print(f"OK -> {out} : {len(chk)} livres, labels identiques à books_enriched.parquet (100 %).")


if __name__ == "__main__":
    main()
