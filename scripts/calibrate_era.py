"""
Helps choose ERA_Z_MIN (threshold to assign a period to the story). The period is no longer forced: it is only kept
if its z-score is high and clearly higher than that of the 2nd best period.

    python scripts/calibrate_era.py
    python scripts/calibrate_era.py --titles "Harry Potter" "Dune" "Pride and Prejudice" --era-z-min 2.5

1) The table gives the share of books that receive a period depending on the threshold (aim for about 20 to 40%).
2) For books whose period you know, it shows the best periods, their z-score and the decision.
   Pick the lowest threshold for which these books no longer get a wrong period, then put it in
   src/enrichment.py (ERA_Z_MIN) and rerun run_enrichment.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en  # noqa: E402

DEFAULT_TITLES = ["Harry Potter and the Sorcerer", "Dune", "The Great Gatsby", "Pride and Prejudice",
                  "The Hunger Games", "Nineteen Eighty-Four", "The Hobbit"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--titles", nargs="*", default=DEFAULT_TITLES)
    ap.add_argument("--era-z-min", type=float, default=en.ERA_Z_MIN)
    ap.add_argument("--model", default=en.DEFAULT_MODEL)
    a = ap.parse_args()

    df = pd.read_parquet(ROOT / "data/processed/books_clean.parquet")
    enc = en.SbertEncoder(a.model)
    book_emb = en.encode_cached(enc, df["embed_text"].fillna("").tolist(), df["book_id"].astype(str).tolist(),
                                f"p2_{a.model.split('/')[-1]}_clean_books")
    labels, lab_emb = en.encode_blocks(enc, en.load_blocks())["Atmosphere"]
    sims = en.score_block(book_emb, lab_emb)
    z = en.zscore_columns(sims)

    print("Part des livres avec une époque selon le seuil :")
    print(en.era_coverage_report(sims, labels).to_string(index=False))

    era_idx = [j for j, l in enumerate(labels) if l in en.ERA_LABELS]
    print(f"\nLivres de contrôle (seuil = {a.era_z_min}, marge = {en.ERA_MARGIN}) :")
    for t in a.titles:
        hit = df[df["title"].str.contains(t, case=False, regex=False)].head(1)
        if hit.empty:
            print(f"  {t!r} : absent du catalogue")
            continue
        i = hit.index[0]
        top = sorted(era_idx, key=lambda j: -z[i, j])[:3]
        decision = en.select_atmospheres(z[i:i + 1], labels, en.Z_MIN, a.era_z_min)[0]
        print(f"  {hit.iloc[0]['title'][:45]:45s} " + "  ".join(f"{labels[j]} {z[i, j]:+.2f}" for j in top)
              + f"   -> {decision}")


if __name__ == "__main__":
    main()
