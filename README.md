# GenderBook: semantic book recommendation (ECE ING5 NLP project)

GenderBook recommends books from preferences written in **natural language**. A reader describes the book they would like to read, a book they loved and the atmosphere they are looking for; the system compares the **meaning** of these answers with the blurbs of 12,000 books (SBERT embeddings, cosine similarity), scores each book with an explainable formula, and returns a **top 3** with a score, the criteria behind it and a template-based explanation.

It is theme 3 of the assignment *Semantic Analysis and Intelligent Contextual Recommendation Systems* (*Book Recommendation Based on Literary Preferences*). The original plan and the split of tasks are in [PLAN_PROJET.md](PLAN_PROJET.md); the full documentation is the report in [docs/report/](docs/report/).

```
questionnaire ──► answers.json ──► build_profile() ──► filters ──► retrieval (top 300)
                                   texts + Likert          ──► re-ranking (blurb + coverage of 4 blocks
                                   → SBERT vectors              + popularity − "to avoid") ──► top 3 + explanations
```

## 1. Installation

Python **3.11 or 3.12** is required (some dependencies, e.g. `gensim`, do not build on newer versions yet).

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # needed by the TF-IDF / GloVe preprocessing
python -m pytest tests -q                 # all tests should pass
```

The first time a model is used, `sentence-transformers` downloads it (≈ 90 MB for MiniLM) and the catalogue is encoded once (≈ 7 min on CPU), then cached in `data/embeddings/`.

## 2. Quick start

### Get recommendations in Python

```python
from src.recommender import recommend

answers = {
    "free_text": {
        "ideal_book": "A slow-building supernatural terror in an old house where something evil watches a family.",
        "loved_book": "A writer looking after an empty mountain hotel in winter slowly goes mad.",
        "atmosphere": "Creepy, claustrophobic and unsettling.",
        "avoid": "No zombies, no splatter.",
    },
    "genres": ["Horror"], "themes": ["Madness & the mind"], "period": "No preference",
    "audience": "Adult", "formats": [], "favorite_author": "Stephen King",
    "guided": {"focus": "Characters", "standalone_only": False, "exclude_favorite_author": True},
    "likert": {"complexity": 3, "pace": 2, "darkness": 5, "popularity": 2,
               "w_genre": 5, "w_themes": 3, "w_style": 2, "w_atmosphere": 5},
}
profile, recs = recommend(answers, top_k=3)
for r in recs:
    print(r.rank, r.title, r.final_score, r.match_label)
    print(r.explanation)
```

The answer format is documented in [PLAN_PROJET.md §4.3](PLAN_PROJET.md); ready-made examples are in `data/eval/personas.json` and `data/reference/personas_p4.json`.

### Run the web application

<!-- TO COMPLETE when P5's pull request is merged (app/streamlit_app.py). -->

```bash
streamlit run app/streamlit_app.py
```

## 3. Reproduce the whole pipeline

Every intermediate file is versioned, so **nothing needs to be regenerated** to use the system. To rebuild everything from the raw data, run the steps in this order from the repo root:

| Step | Command | Output | Owner |
|---|---|---|---|
| 1. Catalogue | `python -m scripts.build_dataset` (downloads the Kaggle dataset, ≈ 1.1 GB, ≈ 2 min) | `data/processed/books_clean.parquet` (12,000 books) | P1 |
| 2. Enrichment | `python -m scripts.run_enrichment` | `data/processed/books_enriched.parquet` (genres, themes, styles, atmospheres, keywords) | P2 |
| 3. Embeddings | `python -X utf8 -m scripts.p3_build_embeddings` (add `--model mpnet` for MPNet) | `data/embeddings/` cache | P3 |
| 4. Semantic map | `python -X utf8 -m scripts.p3_semantic_map` | `data/embeddings/umap_2d.npy`, figures | P3 |
| 5. Encoder demo | `python -X utf8 -m scripts.p3_compare_encoders` | TF-IDF vs GloVe vs SBERT on 500 books | P3 |

The notebooks document each step with comments and explanations and can be run in the same order:

| Notebook | Content | Report section |
|---|---|---|
| `notebooks/01_data_preparation.ipynb` | dataset origin, cleaning, EDA | 2 |
| `notebooks/02_reference_enrichment.ipynb` | block taxonomy, zero-shot labels, keywords, validation | 3 |
| `notebooks/03_semantic_engine.ipynb` | preprocessing, TF-IDF / GloVe / SBERT, cosine, model choice | 4 |
| `notebooks/04_scoring_recommendation.ipynb` | user profile, scoring formula, calibration, explanations | 5 |
| `notebooks/05_app_and_visualization.ipynb` | questionnaire and visualisations (P5, to be merged) | 6 |
| `notebooks/06_evaluation.ipynb` | end-to-end evaluation on 10 test personas | 7 |

## 4. Evaluation

The evaluation of the complete system is in `notebooks/06_evaluation.ipynb` (metric code in `src/evaluation.py`). Run it from Jupyter or VS Code with the `.venv` kernel, or from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/06_evaluation.ipynb
```

- Test personas: `data/eval/personas.json` (expected books fixed by hand before running the system).
- Results: `data/eval/results/*.csv`; figures: `docs/figures/06_*.png`.
- Blind manual judgments of the recommended books: `data/eval/judgments.csv` (fill the `relevant` column with 1 / 0, then re-run the last cells).
- MPNet (`RUN_MPNET` in the first cell) needs the MPNet embeddings of the catalogue: encoding them takes ≈ 80 min on CPU (≈ 10 min on a GPU, e.g. Google Colab) with `python -X utf8 -m scripts.p3_build_embeddings --model mpnet`, or copy the three `all-mpnet-base-v2_*` cache files from a teammate into `data/embeddings/` (they are not versioned).
- Main results (report section 7): with MiniLM, genre-precision@3 0.97 and theme-overlap@3 0.97 on the 10 test personas; the top 3 is twice as stable under rephrasing as with the lexical baseline (Jaccard 0.63 vs 0.28).

## 5. Tests

```bash
python -m pytest tests -q
```

Unit tests cover data cleaning, enrichment, preprocessing, the embedding engine, encoders, scoring and evaluation metrics. They use small fake models and run in a few seconds without downloading anything.

## 6. Repository structure

```
data/
  raw/                 Kaggle dataset (not versioned, downloaded by build_dataset)
  processed/           books_clean.parquet (P1) · books_enriched.parquet (P2) · samples
  reference/           blocks.json (taxonomy) · gold annotations · tuning personas · reports
  embeddings/          embedding caches (not versioned) · umap_2d.npy (versioned)
  eval/                evaluation personas, judgments and results (P6)
notebooks/             01 … 06, one per part, commented step by step
src/
  config.py            paths and constants shared by all modules
  data_loader.py, marketing.py                         P1: catalogue, cleaning, filters
  enrichment.py                                        P2: genres, themes, styles, keywords
  preprocessing.py, embeddings.py, encoders.py, semantic_map.py   P3: semantic engine
  schemas.py, profile.py, scoring.py, recommender.py, explain.py, fallback_engine.py   P4
  evaluation.py                                        P6: end-to-end evaluation
scripts/               command-line versions of the pipeline steps
tests/                 unit tests
docs/
  report/              sections of the PDF report (00 introduction … 11 appendix)
  slides/              slide plans per person
  figures/             figures used in the report and the slides
app/                   Streamlit application (P5)
```

## 7. Data

The catalogue comes from the public **Book Depository Dataset** (Kaggle, `sp1thas/book-depository-dataset`, 1.1 million editions), reduced to 12,000 well-known works with an informative English blurb. Origin, cleaning and limitations are described in report section 2.

| Main column | Description |
|---|---|
| `book_id`, `title`, `authors`, `main_author` | identifier (ISBN-13), display title, authors (one spelling per author) |
| `description`, `embed_text` | cleaned blurb (no press quotes or banners) / text encoded by SBERT (`title. description`, ≈ 1,500 characters) |
| `genres`, `themes`, `styles`, `atmospheres`, `keywords` | labels derived by semantic analysis (P2) |
| `audience`, `domain`, `format`, `pub_year` | filters: Children / Young Adult / Adult, Fiction / Non-fiction, Paperback / Hardback, year of the **edition** |
| `rating_avg`, `rating_count`, `popularity` | Goodreads ratings and a popularity score in [0, 1] |

```python
from src.data_loader import load_books, get_books_by_author, apply_constraints

books = load_books()                       # enriched > clean > sample (the best available)
get_books_by_author(books, "stephen kng")  # fuzzy matching on author names
mask = apply_constraints(books, audience="Young Adult", formats=["Paperback"], fiction_only=True)
```

## 8. Team

| Role | Responsibility |
|---|---|
| P1: Data engineering | dataset cleaning, catalogue, EDA |
| P2: Taxonomy and enrichment | block taxonomy, genres, themes, styles, keywords, validation |
| P3: Semantic engine | preprocessing, embeddings, encoder comparison, semantic map |
| P4: Scoring and recommendation | user profile, scoring formula, calibration, explanations |
| P5: Application and visualisations | Streamlit questionnaire, results page, charts |
| P6: Evaluation and integration | code reviews and merges, evaluation, README, report, slides |

<!-- TO COMPLETE: add the name of each member next to their role. -->
