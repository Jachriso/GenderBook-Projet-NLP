"""
Run the semantic enrichment (P2).

    python -m scripts.run_enrichment --sample          # 500 books, ~1 min: quick test
    python -m scripts.run_enrichment                   # full catalogue (12,000 books)
    python -m scripts.run_enrichment --annotation-sheet  # generates the sheet of the 60 books to annotate

Outputs:
    data/processed/books_enriched.parquet          (full)  or  books_enriched_sample.parquet (--sample)
    data/reference/category_genre_map.csv          categories -> genres table (genre_manual column to review)
    data/reference/enrichment_report.json          enrichment statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allows "python scripts/run_enrichment.py" as well as "python -m scripts..."

import pandas as pd  # noqa: E402

from src import enrichment as en  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="travailler sur books_sample.parquet (500 livres)")
    ap.add_argument("--model", default=en.DEFAULT_MODEL)
    ap.add_argument("--z-min", type=float, default=en.Z_MIN, help="seuil z-score pour garder un label")
    ap.add_argument("--no-keywords", action="store_true", help="sauter l'extraction de mots-clés")
    ap.add_argument("--annotation-sheet", action="store_true",
                    help="écrit data/reference/gold_60_to_annotate.csv puis s'arrête")
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    src_path = proc / ("books_sample.parquet" if args.sample else "books_clean.parquet")
    df = pd.read_parquet(src_path)
    print(f"Chargé {src_path.name} : {df.shape}")

    if args.annotation_sheet:
        sheet = en.make_annotation_sheet(pd.read_parquet(proc / "books_sample.parquet")
                                         if (proc / "books_sample.parquet").exists() else df)
        out = ROOT / "data" / "reference" / "gold_60_to_annotate.csv"
        sheet.to_csv(out, index=False)
        print(f"Feuille d'annotation écrite : {out} (remplir gold_genres et gold_themes, séparés par « | »)")
        return

    encoder = en.SbertEncoder(args.model)
    blocks = en.load_blocks()

    # Book embeddings (disk cache: rerunning is instantaneous)
    tag = "sample" if args.sample else "clean"
    name = f"p2_{args.model.split('/')[-1]}_{tag}_books"
    book_emb = en.encode_cached(encoder, df["embed_text"].fillna("").tolist(),
                                df["book_id"].astype(str).tolist(), name)

    # Categories -> genres table: the manual review is kept if the file already exists
    existing = pd.read_csv(en.CATEGORY_MAP_PATH, keep_default_na=False) if en.CATEGORY_MAP_PATH.exists() else None
    cat_map = en.build_category_map(en.category_counts(df), encoder, blocks, existing)
    cat_map.to_csv(en.CATEGORY_MAP_PATH, index=False)
    print(f"Table de correspondance écrite : {en.CATEGORY_MAP_PATH} ({len(cat_map)} catégories)")

    enriched, report = en.enrich_catalogue(df, encoder, blocks, cat_map, book_emb,
                                           z_min=args.z_min, with_keywords=not args.no_keywords)

    out_name = "books_enriched_sample.parquet" if args.sample else "books_enriched.parquet"
    en.save_enriched(enriched, proc / out_name)
    (ROOT / "data" / "reference" / "enrichment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"OK -> {proc / out_name}")
    print(enriched[["title", "genres", "themes", "styles", "atmospheres", "keywords"]].head(5).to_string())


if __name__ == "__main__":
    main()
