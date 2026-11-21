# GenderBook : recommandation sémantique de livres (Projet NLP ECE ING5)

Système de recommandation de livres fondé sur l'**analyse sémantique** des préférences du lecteur exprimées en langage naturel. C'est le thème 3 du sujet *Semantic Analysis and Intelligent Contextual Recommendation Systems*. Le plan complet et la répartition des tâches sont dans [PLAN_PROJET.md](PLAN_PROJET.md).

## Installation

```bash
python3 -m venv .venv          # Python 3.11+ (testé en 3.12)
source .venv/bin/activate
pip install -r requirements.txt
```

## Données (personne 1 : terminé ✅)

Le catalogue propre est **déjà versionné** dans le repo. Inutile de télécharger le dataset brut pour travailler :

| Fichier | Contenu |
|---|---|
| `data/processed/books_clean.parquet` | **catalogue final : 12 000 livres** (une ligne par œuvre) |
| `data/processed/books_sample.parquet` | 500 livres stratifiés, **schéma final complet** (colonnes de P2 présentes mais vides) |
| `data/reference/category_stats.csv` | fréquence des 1 078 catégories Book Depository du catalogue (pour le mapping catégories → genres de P2) |
| `data/reference/filtering_report.csv` | nombre de livres après chaque étape de filtrage |

Pour tout régénérer depuis le dataset Kaggle brut (téléchargement automatique, ≈ 1,1 Go dans `data/raw/`, ≈ 2 min) :

```bash
python -m scripts.build_dataset
```

### Utilisation dans le code

```python
from src.data_loader import load_books, get_books_by_author, apply_constraints

books = load_books()                       # enrichi > propre > échantillon (le meilleur disponible)
get_books_by_author(books, "jk rowling")   # toutes les orthographes -> les 7 Harry Potter + The Cuckoo's Calling
get_books_by_author(books, "stephen kng")  # matching flou -> livres de Stephen King
mask = apply_constraints(books, audience="Young Adult", formats=["Paperback"],
                         excluded_authors=["Suzanne Collins"], fiction_only=True)
```

### Colonnes de `books_clean.parquet`

| Colonne | Description |
|---|---|
| `book_id` | identifiant (ISBN-13 de l'édition retenue) |
| `title` | titre d'affichage, sans mentions d'édition (« Deluxe Edition »…) |
| `authors` / `authors_str` / `main_author` | liste des auteurs (**une seule orthographe par auteur**) / chaîne / auteur principal (hors illustrateur, traducteur) |
| `categories` | catégories Book Depository (liste de noms) |
| `description` | **texte pour le NLP** : nettoyé, sans texte générique, citations de presse, bandeaux commerciaux ni sections « Praise for… » |
| `description_raw` | description avec nettoyage minimal seulement (traçabilité) |
| `embed_text` | **texte à encoder** = `title. description`, tronqué proprement à ≈ 1 500 caractères |
| `pub_year` | année de publication de l'**édition** (pas l'époque du récit !) |
| `rating_avg`, `rating_count` | notes Goodreads |
| `popularity` | score ∈ [0,1] = (note/5) × log(1+nb notes)/log(1+max) |
| `audience` | `Children` / `Young Adult` / `Adult` (for-ages + catégories) |
| `domain`, `is_fiction` | `Fiction` / `Non-fiction` |
| `format`, `for_ages`, `bestsellers_rank`, `image_url`, `url`, `isbn13`, `n_editions`, `n_words` | métadonnées |

## Structure du repo

```
data/          raw/ (non versionné) · processed/ · reference/ · embeddings/
notebooks/     01_data_preparation.ipynb (P1) … 06_evaluation.ipynb
src/           config.py · data_loader.py + marketing.py (P1) · enrichment.py · embeddings.py · scoring.py · …
scripts/       build_dataset.py
tests/         tests unitaires (python -m pytest tests -q)
docs/          figures/ · report/ (sections du rapport PDF) · slides/
app/           application Streamlit
```

## Avancement

- [x] **P1 : Données** : catalogue, EDA, utilitaires, 52 tests, section du rapport (`docs/report/01_dataset.md`), slides (`docs/slides/01_dataset_slides.md`)
- [ ] P2 : Référentiel des blocs et enrichissement (`books_enriched.parquet`)
- [ ] P3 : Moteur sémantique (embeddings)
- [ ] P4 : Scoring et recommandation
- [ ] P5 : Application Streamlit et visualisations
- [ ] P6 : Évaluation, rapport, slides
