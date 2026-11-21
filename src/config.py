"""
Configuration centrale du projet (chemins, paramètres de filtrage, constantes NLP).

Tous les modules importent leurs constantes d'ici : changer une valeur ici
la change partout (notebooks, scripts, application Streamlit).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Chemins
# ---------------------------------------------------------------------------
# Racine du repo = dossier parent de src/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"                # CSV Kaggle bruts (non versionnés)
DATA_PROCESSED = DATA_DIR / "processed"    # parquet propres (versionnés)
DATA_REFERENCE = DATA_DIR / "reference"    # référentiel des blocs, tables annexes
DATA_EMBEDDINGS = DATA_DIR / "embeddings"  # matrices d'embeddings (.npy)
DATA_RESPONSES = DATA_DIR / "responses"    # réponses du questionnaire (app)
FIGURES_DIR = ROOT / "docs" / "figures"

# Fichiers bruts du dataset Kaggle "sp1thas/book-depository-dataset"
KAGGLE_DATASET = "sp1thas/book-depository-dataset"
RAW_DATASET = DATA_RAW / "dataset.csv"
RAW_AUTHORS = DATA_RAW / "authors.csv"
RAW_CATEGORIES = DATA_RAW / "categories.csv"
RAW_FORMATS = DATA_RAW / "formats.csv"

# Sorties de la personne 1 (data) puis de la personne 2 (enrichissement)
BOOKS_SAMPLE = DATA_PROCESSED / "books_sample.parquet"
BOOKS_CLEAN = DATA_PROCESSED / "books_clean.parquet"
BOOKS_ENRICHED = DATA_PROCESSED / "books_enriched.parquet"
CATEGORY_STATS = DATA_REFERENCE / "category_stats.csv"
FILTERING_REPORT = DATA_REFERENCE / "filtering_report.csv"
BLOCKS_JSON = DATA_REFERENCE / "blocks.json"

# ---------------------------------------------------------------------------
# 2. Paramètres de construction du catalogue (personne 1)
# ---------------------------------------------------------------------------
# Colonnes du CSV réellement utiles (les dimensions, le poids, les checksums
# d'image, etc. ne servent pas à la recommandation et ne sont pas chargées).
RAW_USECOLS = [
    "id", "title", "authors", "categories", "description", "lang",
    "format", "for-ages", "publication-date", "rating-avg", "rating-count",
    "bestsellers-rank", "image-url", "url", "isbn13", "edition", "imprint",
]

KEEP_LANG = "en"                     # le modèle SBERT de référence est anglais
KEEP_FORMATS = ("Paperback", "Hardback")  # on retire CD audio, calendriers, jeux...
MIN_DESC_CHARS = 300                 # en dessous : pas assez de texte pour l'analyse sémantique
MIN_RATING_COUNT = 50                # livre « connu » = avis Goodreads fiables
MAX_BOOKS_PER_AUTHOR = 8             # diversité : un auteur prolifique ne doit pas saturer le catalogue
TARGET_CATALOGUE_SIZE = 12_000       # taille finale du catalogue (sujet : minimum 60 livres)
EMBED_TEXT_MAX_CHARS = 1_500         # ≈ 300 tokens, au-delà le modèle tronque de toute façon
SAMPLE_SIZE = 500                    # échantillon de développement pour toute l'équipe
RANDOM_SEED = 42

# Public visé (déduit de for-ages + catégories)
AUDIENCES = ("Children", "Young Adult", "Adult")

# Époques proposées dans le questionnaire.
# ATTENTION : pub_year = année de publication de l'édition, PAS l'époque du récit.
# Ce mapping sert uniquement de filtre optionnel « livres publiés à telle période ».
# L'époque du récit est traitée sémantiquement (bloc Atmosphere, personne 2).
PUBLICATION_PERIODS = {
    "Before 1950": (0, 1949),
    "1950-1999": (1950, 1999),
    "2000-2009": (2000, 2009),
    "2010 and later": (2010, 2100),
}

# ---------------------------------------------------------------------------
# 3. Constantes NLP / scoring (personnes 3 et 4) : valeurs par défaut du plan
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_ALT = "sentence-transformers/all-mpnet-base-v2"
BLOCKS = ("Genre", "Themes", "Style", "Atmosphere")

RETRIEVAL_TOP_K = 300      # étape 1 : candidats récupérés par similarité globale
ALPHA_DESC = 0.5           # poids de la similarité requête ↔ résumé
BETA_COVERAGE = 0.4        # poids du score de couverture multi-blocs
LAMBDA_POP_MAX = 0.2       # bonus popularité max (réglé par le Likert « best-sellers ↔ pépites »)
MU_AVOID = 0.5             # pénalité « à éviter »
TAU_AVOID = 0.3            # seuil de similarité à partir duquel on pénalise
MATCH_THRESHOLDS = {"Excellent": 0.75, "Good": 0.55}
