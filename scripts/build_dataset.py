"""
Construit le catalogue de livres à partir du dataset Kaggle brut (personne 1).

Usage (depuis la racine du repo) :
    python -m scripts.build_dataset            # catalogue complet + échantillon
    python -m scripts.build_dataset --no-langdetect

Sorties :
    data/processed/books_clean.parquet   catalogue final (~12 000 livres)
    data/processed/books_sample.parquet  échantillon stratifié de 500 livres
    data/reference/filtering_report.csv  nombre de lignes après chaque filtre
    data/reference/category_stats.csv    fréquence des catégories (pour la personne 2)
"""
import argparse
import time

from src import config
from src.data_loader import build_catalogue, download_raw_dataset, make_sample, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=int, default=config.TARGET_CATALOGUE_SIZE)
    parser.add_argument("--no-langdetect", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    download_raw_dataset()  # ne fait rien si les CSV sont déjà là
    catalogue, report, stats = build_catalogue(
        target_size=args.target_size, detect_language=not args.no_langdetect
    )
    save_parquet(catalogue, config.BOOKS_CLEAN)
    save_parquet(make_sample(catalogue), config.BOOKS_SAMPLE)
    config.DATA_REFERENCE.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.FILTERING_REPORT, index=False)
    stats.to_csv(config.CATEGORY_STATS, index=False)
    print(f"\n{len(catalogue):,} books saved to {config.BOOKS_CLEAN} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
