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

# ---------------------------------------------------------------------------
# 4. Scoring et recommandation 
# ---------------------------------------------------------------------------
# Les constantes ci-dessus (ALPHA_DESC, BETA_COVERAGE, LAMBDA_POP_MAX, MU_AVOID,
# TAU_AVOID, MATCH_THRESHOLDS, RETRIEVAL_TOP_K) restent la référence de la formule.
# Ci-dessous : les réglages complémentaires utilisés par src/profile.py,
# src/scoring.py et src/recommender.py.

CALIBRATION_JSON = DATA_EMBEDDINGS / "calibration_p4.json"  # p5/p95 calculés par P4 (cache)
CALIBRATION_JSON_P3 = DATA_EMBEDDINGS / "calibration.json"   # fichier éventuel de P3
# "p4" : percentiles sur les candidats de personas (défaut, voir scoring.compute_calibration)
# "p3" : utiliser calibration.json de P3 s'il existe (repli automatique sur "p4")
CALIBRATION_SOURCE = "p4"
CALIBRATION_PERCENTILES = (5, 95)   # s̃ = clip((s - p5) / (p95 - p5), 0, 1)

# Profil : mélange texte libre / livres de l'auteur préféré dans q_global
W_TEXT_QUERY = 0.7                  # part des réponses libres
W_AUTHOR_CENTROID = 0.3             # part du centroïde des livres de l'auteur préféré
QUERY_COMBINATION = "mean"          # "mean" (moyenne des embeddings) ou "concat" (texte concaténé)

# Likert : 3 = « pas de préférence » -> aucune phrase générée (évite de diluer le bloc)
LIKERT_NEUTRAL = 3

# Likert « popularity » : 1 = pépites méconnues ... 5 = best-sellers
# λ = LAMBDA_POP_MAX * (popularity - 1) / 4  ∈ [0 ; LAMBDA_POP_MAX]
DEFAULT_BLOCK_WEIGHT = 3            # poids W_b si le Likert d'importance est absent

# Diversité du top : au plus N livres par auteur principal
MAX_PER_AUTHOR = 1
# Nombre de labels communs affichés par bloc dans l'explication
N_MATCHED_LABELS = 2
# Un label du livre n'est cité comme « commun » que si sa similarité avec une
# préférence de l'utilisateur dépasse ce seuil (évite « Humorous & witty (0.04) »)
MATCHED_LABEL_MIN_SIM = 0.35
# Personas de référence (format answers.json) : calibration + démos du notebook 04
PERSONAS_JSON = DATA_REFERENCE / "personas_p4.json"

# Genres de fiction : si l'utilisateur ne coche QUE des genres de cette liste
# (et ne répond pas à « fiction only »), on active fiction_only automatiquement.
# Les genres ambigus (Classics, Young Adult, Children's, Humour, Poetry & Drama,
# Graphic Novels & Comics) ne déclenchent rien et ne bloquent rien.
FICTION_GENRES = ("Fantasy", "Science Fiction", "Mystery & Detective", "Thriller & Suspense",
                  "Horror", "Romance", "Historical Fiction", "Literary Fiction")
NONFICTION_GENRES = ("Biography & Memoir", "History", "Science & Nature", "Self-help & Psychology",
                     "Philosophy & Religion", "Business & Economics", "Cooking & Lifestyle")
# Catégories classées « fiction » par data_loader mais qui ne sont pas des récits
# (recueils de blagues, confessions, catalogues) : exclues quand fiction_only = True.
NON_NARRATIVE_CATEGORY_RE = r"humour collections|jokes & humour|exhibition catalogues"
# Nombre d'affinités les plus fortes citables dans l'explication (en plus des labels cochés)
N_TOP_AFFINITIES = 3
