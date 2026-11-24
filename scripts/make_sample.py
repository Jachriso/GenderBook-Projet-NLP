"""
Build data/processed/books_enriched_sample.parquet as an EXACT SUBSET of books_enriched.parquet.

Why: the old sample was re-enriched on 500 isolated books; the per-label normalisation depends on the whole
catalogue, so its labels differed from those of the full file (only 32-38% identical). Here nothing is recomputed:
the 500 books of books_sample.parquet (P1) are extracted from the full file, then equality is checked.

    python scripts/make_sample.py          # run AFTER python -m scripts.run_enrichment
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

    # check: re-read the written file and compare it cell by cell with the full file
    chk = pd.read_parquet(out)
    ref = full.assign(book_id=full["book_id"].astype(str)).set_index("book_id").loc[chk["book_id"].astype(str)]
    norm = lambda v: v if isinstance(v, str) else [str(x) for x in v]
    for col in LABEL_COLS:
        same = sum(norm(a) == norm(b) for a, b in zip(chk[col], ref[col]))
        assert same == len(chk), f"{col} : {len(chk) - same} lignes différentes du fichier complet"
    print(f"OK -> {out} : {len(chk)} livres, labels identiques à books_enriched.parquet (100 %).")


if __name__ == "__main__":
    main()
