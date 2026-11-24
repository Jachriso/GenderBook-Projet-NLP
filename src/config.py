"""
Central project configuration (paths, filtering parameters, NLP constants).

Every module imports its constants from here: changing a value here
changes it everywhere (notebooks, scripts, Streamlit application).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
# Repo root = parent folder of src/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"                # raw Kaggle CSVs (not versioned)
DATA_PROCESSED = DATA_DIR / "processed"    # clean parquet files (versioned)
DATA_REFERENCE = DATA_DIR / "reference"    # block reference, side tables
DATA_EMBEDDINGS = DATA_DIR / "embeddings"  # embedding matrices (.npy)
DATA_RESPONSES = DATA_DIR / "responses"    # questionnaire answers (app)
FIGURES_DIR = ROOT / "docs" / "figures"

# Raw files of the Kaggle dataset "sp1thas/book-depository-dataset"
KAGGLE_DATASET = "sp1thas/book-depository-dataset"
RAW_DATASET = DATA_RAW / "dataset.csv"
RAW_AUTHORS = DATA_RAW / "authors.csv"
RAW_CATEGORIES = DATA_RAW / "categories.csv"
RAW_FORMATS = DATA_RAW / "formats.csv"

# Outputs of person 1 (data), then person 2 (enrichment)
BOOKS_SAMPLE = DATA_PROCESSED / "books_sample.parquet"
BOOKS_CLEAN = DATA_PROCESSED / "books_clean.parquet"
BOOKS_ENRICHED = DATA_PROCESSED / "books_enriched.parquet"
CATEGORY_STATS = DATA_REFERENCE / "category_stats.csv"
FILTERING_REPORT = DATA_REFERENCE / "filtering_report.csv"
BLOCKS_JSON = DATA_REFERENCE / "blocks.json"

# ---------------------------------------------------------------------------
# 2. Catalogue construction parameters (person 1)
# ---------------------------------------------------------------------------
# CSV columns that are actually useful (dimensions, weight, image checksums,
# etc. are not used for recommendation and are not loaded).
RAW_USECOLS = [
    "id", "title", "authors", "categories", "description", "lang",
    "format", "for-ages", "publication-date", "rating-avg", "rating-count",
    "bestsellers-rank", "image-url", "url", "isbn13", "edition", "imprint",
]

KEEP_LANG = "en"                     # the reference SBERT model is English
KEEP_FORMATS = ("Paperback", "Hardback")  # drop audio CDs, calendars, games...
MIN_DESC_CHARS = 300                 # below this: not enough text for semantic analysis
MIN_RATING_COUNT = 50                # "known" book = reliable Goodreads ratings
MAX_BOOKS_PER_AUTHOR = 8             # diversity: a prolific author must not saturate the catalogue
TARGET_CATALOGUE_SIZE = 12_000       # final catalogue size (assignment: at least 60 books)
EMBED_TEXT_MAX_CHARS = 1_500         # ≈ 300 tokens, beyond that the model truncates anyway
SAMPLE_SIZE = 500                    # development sample for the whole team
RANDOM_SEED = 42

# Target audience (inferred from for-ages + categories)
AUDIENCES = ("Children", "Young Adult", "Adult")

# Periods offered in the questionnaire.
# WARNING: pub_year = publication year of the edition, NOT the period of the story.
# This mapping is only used as an optional "books published in this period" filter.
# The period of the story is handled semantically (Atmosphere block, person 2).
PUBLICATION_PERIODS = {
    "Before 1950": (0, 1949),
    "1950-1999": (1950, 1999),
    "2000-2009": (2000, 2009),
    "2010 and later": (2010, 2100),
}

# ---------------------------------------------------------------------------
# 3. NLP / scoring constants (persons 3 and 4): default values from the plan
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_ALT = "sentence-transformers/all-mpnet-base-v2"
BLOCKS = ("Genre", "Themes", "Style", "Atmosphere")

RETRIEVAL_TOP_K = 300      # step 1: candidates retrieved by global similarity
ALPHA_DESC = 0.5           # weight of the query ↔ blurb similarity
BETA_COVERAGE = 0.4        # weight of the multi-block coverage score
LAMBDA_POP_MAX = 0.2       # max popularity bonus (set by the "best-sellers ↔ hidden gems" Likert)
MU_AVOID = 0.5             # "to avoid" penalty
TAU_AVOID = 0.3            # similarity threshold above which we penalise
MATCH_THRESHOLDS = {"Excellent": 0.75, "Good": 0.55}

# ---------------------------------------------------------------------------
# 4. Scoring and recommendation
# ---------------------------------------------------------------------------
# The constants above (ALPHA_DESC, BETA_COVERAGE, LAMBDA_POP_MAX, MU_AVOID,
# TAU_AVOID, MATCH_THRESHOLDS, RETRIEVAL_TOP_K) remain the reference of the formula.
# Below: the additional settings used by src/profile.py,
# src/scoring.py and src/recommender.py.

CALIBRATION_JSON = DATA_EMBEDDINGS / "calibration_p4.json"  # p5/p95 computed by P4 (cache)
CALIBRATION_JSON_P3 = DATA_EMBEDDINGS / "calibration.json"   # optional file from P3
# "p4": percentiles over the personas' candidates (default, see scoring.compute_calibration)
# "p3": use P3's calibration.json if it exists (automatic fallback to "p4")
CALIBRATION_SOURCE = "p4"
CALIBRATION_PERCENTILES = (5, 95)   # s̃ = clip((s - p5) / (p95 - p5), 0, 1)

# Profile: mix of free text / favourite author's books in q_global
W_TEXT_QUERY = 0.7                  # share of the free-text answers
W_AUTHOR_CENTROID = 0.3             # share of the centroid of the favourite author's books
QUERY_COMBINATION = "mean"          # "mean" (average of embeddings) or "concat" (concatenated text)

# Likert: 3 = "no preference" -> no sentence generated (avoids diluting the block)
LIKERT_NEUTRAL = 3

# "popularity" Likert: 1 = hidden gems ... 5 = best-sellers
# λ = LAMBDA_POP_MAX * (popularity - 1) / 4  ∈ [0 ; LAMBDA_POP_MAX]
DEFAULT_BLOCK_WEIGHT = 3            # weight W_b if the importance Likert is missing

# Diversity of the top: at most N books per main author
MAX_PER_AUTHOR = 1
# Number of shared labels shown per block in the explanation
N_MATCHED_LABELS = 2
# A book label is only cited as "shared" if its similarity with one of the
# user's preferences exceeds this threshold (avoids "Humorous & witty (0.04)")
MATCHED_LABEL_MIN_SIM = 0.35
# Reference personas (answers.json format): calibration + notebook 04 demos
PERSONAS_JSON = DATA_REFERENCE / "personas_p4.json"

# Fiction genres: if the user ticks ONLY genres from this list
# (and does not answer "fiction only"), fiction_only is enabled automatically.
# Ambiguous genres (Classics, Young Adult, Children's, Humour, Poetry & Drama,
# Graphic Novels & Comics) neither trigger nor block anything.
FICTION_GENRES = ("Fantasy", "Science Fiction", "Mystery & Detective", "Thriller & Suspense",
                  "Horror", "Romance", "Historical Fiction", "Literary Fiction")
NONFICTION_GENRES = ("Biography & Memoir", "History", "Science & Nature", "Self-help & Psychology",
                     "Philosophy & Religion", "Business & Economics", "Cooking & Lifestyle")
# Categories classified as "fiction" by data_loader that are not narratives
# (joke collections, confessions, catalogues): excluded when fiction_only = True.
NON_NARRATIVE_CATEGORY_RE = r"humour collections|jokes & humour|exhibition catalogues"
# Number of strongest affinities that may be cited in the explanation (in addition to ticked labels)
N_TOP_AFFINITIES = 3
