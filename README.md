# GenderBook: semantic book recommendation (ECE ING5 NLP project)

Book recommendation system based on the **semantic analysis** of the reader's preferences expressed in natural language. It is theme 3 of the assignment *Semantic Analysis and Intelligent Contextual Recommendation Systems*. The full plan and the split of tasks are in [PLAN_PROJET.md](PLAN_PROJET.md).

## Installation

```bash
python3 -m venv .venv          # Python 3.11+ (tested with 3.12)
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (person 1: done ✅)

The clean catalogue is **already versioned** in the repo. There is no need to download the raw dataset to work:

| File | Content |
|---|---|
| `data/processed/books_clean.parquet` | **final catalogue: 12,000 books** (one row per work) |
| `data/processed/books_sample.parquet` | 500 stratified books, **full final schema** (P2 columns present but empty) |
| `data/reference/category_stats.csv` | frequency of the catalogue's 1,078 Book Depository categories (for P2's categories → genres mapping) |
| `data/reference/filtering_report.csv` | number of books after each filtering step |

To regenerate everything from the raw Kaggle dataset (automatic download, ≈ 1.1 GB in `data/raw/`, ≈ 2 min):

```bash
python -m scripts.build_dataset
```

### Usage in the code

```python
from src.data_loader import load_books, get_books_by_author, apply_constraints

books = load_books()                       # enriched > clean > sample (the best available)
get_books_by_author(books, "jk rowling")   # all spellings -> the 7 Harry Potter books + The Cuckoo's Calling
get_books_by_author(books, "stephen kng")  # fuzzy matching -> Stephen King's books
mask = apply_constraints(books, audience="Young Adult", formats=["Paperback"],
                         excluded_authors=["Suzanne Collins"], fiction_only=True)
```

### Columns of `books_clean.parquet`

| Column | Description |
|---|---|
| `book_id` | identifier (ISBN-13 of the edition kept) |
| `title` | display title, without edition mentions ("Deluxe Edition"…) |
| `authors` / `authors_str` / `main_author` | list of authors (**a single spelling per author**) / string / main author (excluding illustrator, translator) |
| `categories` | Book Depository categories (list of names) |
| `description` | **text for NLP**: cleaned, without generic text, press quotes, commercial banners or "Praise for…" sections |
| `description_raw` | description with minimal cleaning only (traceability) |
| `embed_text` | **text to encode** = `title. description`, cleanly truncated at ≈ 1,500 characters |
| `pub_year` | publication year of the **edition** (not the period of the story!) |
| `rating_avg`, `rating_count` | Goodreads ratings |
| `popularity` | score ∈ [0,1] = (rating/5) × log(1+number of ratings)/log(1+max) |
| `audience` | `Children` / `Young Adult` / `Adult` (for-ages + categories) |
| `domain`, `is_fiction` | `Fiction` / `Non-fiction` |
| `format`, `for_ages`, `bestsellers_rank`, `image_url`, `url`, `isbn13`, `n_editions`, `n_words` | metadata |

## Repo structure

```
data/          raw/ (not versioned) · processed/ · reference/ · embeddings/
notebooks/     01_data_preparation.ipynb (P1) … 06_evaluation.ipynb
src/           config.py · data_loader.py + marketing.py (P1) · enrichment.py · embeddings.py · scoring.py · …
scripts/       build_dataset.py
tests/         unit tests (python -m pytest tests -q)
docs/          figures/ · report/ (sections of the PDF report) · slides/
app/           Streamlit application
```

## Progress

- [x] **P1: Data**: catalogue, EDA, utilities, 52 tests, report section (`docs/report/01_dataset.md`), slides (`docs/slides/01_dataset_slides.md`)
- [ ] P2: Block taxonomy and enrichment (`books_enriched.parquet`)
- [ ] P3: Semantic engine (embeddings)
- [ ] P4: Scoring and recommendation
- [ ] P5: Streamlit application and visualisations
- [ ] P6: Evaluation, report, slides
