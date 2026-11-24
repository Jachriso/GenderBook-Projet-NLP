"""
Build the book catalogue from the raw Kaggle dataset (person 1).

Usage (from the repo root):
    python -m scripts.build_dataset            # full catalogue + sample
    python -m scripts.build_dataset --no-langdetect

Outputs:
    data/processed/books_clean.parquet   final catalogue (~12,000 books)
    data/processed/books_sample.parquet  stratified sample of 500 books
    data/reference/filtering_report.csv  number of rows after each filter
    data/reference/category_stats.csv    category frequencies (for person 2)
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
    download_raw_dataset()  # does nothing if the CSVs are already there
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
