# Projet NLP : système de recommandation sémantique de livres
**ECE ING5 Data & IA, 2026-27 (Sarah Malaeb). Thème 3 de l'Annexe I : « Book Recommendation Based on Literary Preferences »**
**Dataset : Book Depository (Kaggle, `sp1thas/book-depository-dataset`)**

> Le thème 3 fait déjà partie de l'Annexe I, donc **pas besoin de validation par la prof**. Le dataset est libre (« Students are free to select their dataset »), mais il faudra **documenter son origine, sa structure et son preprocessing** (Requirement 2).

---

## 0. TL;DR : ce qu'on doit rendre

| Livrable | Forme | Notation |
|---|---|---|
| Code source | Notebooks Jupyter **commentés à chaque étape** + scripts Python (`src/`) + app web | Groupe |
| Documentation | **PDF** : choix de conception, algorithmes, librairies, justification de la formule de score | Groupe |
| Présentation + démo | Slides + démo live de l'app, **les 6 parlent** | **Individuelle** |

L'application doit faire, dans l'ordre :
1. **Questionnaire web** (Streamlit) avec les **5 types de questions** : texte libre, Likert, guidées, choix multiple, cases à cocher.
2. **Analyse sémantique** des réponses : embeddings SBERT, avec Word2Vec/GloVe/TF-IDF en baselines de comparaison.
3. **Score d'affinité sémantique** par bloc (Genre, Thèmes, Style, Atmosphère/Époque), plus un **score de couverture** pondéré.
4. **Top 3 des livres** recommandés, chacun avec son score, les critères qui ont contribué et une **explication générée par template** (pas de GenAI obligatoire).
5. **Visualisations** : radar, barres, carte de similarité 2D.
6. *(Bonus facultatif)* explication par LLM (Gemini free tier ou Ollama). L'app doit **fonctionner sans**.

---

## 1. Lecture des consignes : les pièges à éviter

1. **« Not purely numerical or keyword matching »** : le cœur du système, c'est la similarité cosinus entre embeddings contextuels. Les Likert ne doivent **pas** être juste des nombres : on les **convertit en phrases** (ex. complexité 5/5 → « I want a complex, dense, literary narrative ») encodées elles aussi, ou on s'en sert comme **poids** des blocs.
2. **« Sufficient free-text input »** : prévoir au moins 3 questions ouvertes, dont 1 obligatoire avec un minimum de caractères.
3. **« Competency coverage score that aggregates alignment across multiple blocks »** : adapté aux livres, ça donne la *couverture des préférences du lecteur* sur 4 blocs (Genre, Thèmes, Style, Atmosphère). Formule obligatoire : `Σ Wi·Si / Σ Wi`.
4. **Chaque recommandation** doit contenir : l'item, son score, les critères principaux et une explication. Il manque souvent un de ces quatre éléments.
5. **Justifier** le choix du modèle d'embedding et de la mesure de similarité. On le fait par une **comparaison chiffrée** TF-IDF vs GloVe vs SBERT (c'est ce qui donne un rapport solide).
6. **« Demonstrate how results are derived from semantic analysis »** : dans l'app, afficher la décomposition du score et la phrase du résumé la plus proche de la demande de l'utilisateur.
7. **Commentaires à chaque étape du code** : c'est explicitement noté.
8. **Le dataset ne contient PAS les colonnes Themes / Style / Keywords** demandées par la structure suggérée `BookID | Genre | Title | Author | Synopsis | Themes | Style | Keywords`. On les **dérive par NLP** (zero-shot par embeddings + KeyBERT). C'est un vrai plus à mettre en avant en soutenance.
9. **Le dataset est énorme** (plus d'un million de lignes, fichier CSV lourd). On travaille sur un **sous-ensemble propre d'environ 10 000 livres** en anglais avec une description riche, et on le justifie dans le rapport. Le minimum demandé est de 60 livres.

---

## 2. Le dataset Book Depository

Fichiers du Kaggle (noms exacts à vérifier à l'ouverture) :
- `dataset.csv` : un livre par ligne. Colonnes principales : `id`, `title`, `authors` (liste d'IDs, ex. `"[1, 2]"`), `categories` (liste d'IDs), `description`, `lang`, `publication-date`, `rating-avg`, `rating-count`, `format`, `for-ages`, `bestsellers-rank`, `image-url`, `isbn13`, `edition`, `imprint`, `publication-place`, plus des dimensions et le poids (inutiles ici).
- `authors.csv` (`author_id`, `author_name`), `categories.csv` (`category_id`, `category_name`), `formats.csv`, `places.csv` : tables de correspondance des IDs.

Téléchargement :
```bash
pip install kaggle   # puis mettre son token kaggle.json dans ~/.kaggle/
kaggle datasets download -d sp1thas/book-depository-dataset -p data/raw --unzip
```

**Filtrage recommandé** (à justifier dans le rapport) :
- `lang == "en"` (le modèle SBERT de référence est anglais).
- `description` non vide, **≥ 300 caractères** après nettoyage HTML (sinon il n'y a rien de sémantique à comparer).
- `rating-count ≥ 20` (ou un autre seuil) pour garder des livres « connus », ce qui rend la démo plus parlante.
- **Dédoublonnage** des éditions multiples : clé `(titre normalisé, premier auteur)`, on garde la ligne avec le plus de `rating-count`.
- Retirer les catégories hors fiction ou non pertinentes si on veut un catalogue « lecture loisir » (manuels scolaires, dictionnaires, etc.), ou les garder et les mapper en « Non-fiction ». **À décider en équipe au kick-off.**
- Cible : **5 000 à 15 000 livres**. Ça suffit largement et les embeddings se calculent en quelques minutes.

---

## 3. Architecture technique (décidée une fois pour toutes)

**Stack** : Python 3.11, pandas, pyarrow, numpy, scikit-learn, `sentence-transformers`, `gensim` (GloVe via `gensim.downloader`), `keybert`, `umap-learn`, `plotly`, `streamlit`, `spacy` (preprocessing des baselines). Optionnel : `faiss-cpu`, `google-generativeai` ou `ollama`.

**Modèle d'embedding** : `sentence-transformers/all-MiniLM-L6-v2` par défaut (rapide, 384 dimensions, celui de l'exemple de la prof), comparé à `all-mpnet-base-v2` (meilleur, 768 dimensions). On garde le meilleur selon l'évaluation de la personne 6. **Interface et saisie en anglais**, cohérent avec le dataset. Option : `paraphrase-multilingual-MiniLM-L12-v2` si on veut accepter le français.

**Pipeline :**
```
Questionnaire (Streamlit) ──► answers.json
                                  │
                                  ▼
                        build_profile()  ── encode les textes utilisateur (SBERT)
                                  │          + convertit les Likert en phrases/poids
                                  ▼
  books_enriched.parquet ──► Étape 1 : retrieval (cosinus requête globale vs résumés) → top 300
  book_embeddings.npy           │
  label_embeddings              ▼
                         Étape 2 : re-ranking par blocs (Genre/Thèmes/Style/Atmosphère)
                                  + popularité + pénalité « à éviter » + filtres + diversité
                                  ▼
                         Top 3 + scores + critères + explication (template)
                                  ▼
                         Visualisations (radar, barres, carte UMAP)
```

**Arborescence du repo GitHub :**
```
book-reco/
├── README.md
├── requirements.txt
├── .gitignore                 # data/raw/, *.npy lourds, .env
├── data/
│   ├── raw/                   # CSV Kaggle (jamais commités)
│   ├── processed/
│   │   ├── books_sample.parquet       # 500 livres, livré en 2 h par P1 pour débloquer tout le monde
│   │   ├── books_clean.parquet        # P1
│   │   └── books_enriched.parquet     # P2
│   ├── reference/
│   │   └── blocks.json                # P2 : taxonomie des 4 blocs
│   └── embeddings/                    # P3 : .npy + metadata
├── notebooks/
│   ├── 01_data_preparation.ipynb      # P1
│   ├── 02_reference_enrichment.ipynb  # P2
│   ├── 03_semantic_engine.ipynb       # P3
│   ├── 04_scoring_recommendation.ipynb# P4
│   ├── 05_app_and_visualization.ipynb # P5 (captures + explication de l'UI)
│   └── 06_evaluation.ipynb            # P6
├── src/
│   ├── config.py           # chemins, nom du modèle, poids par défaut (tout le monde)
│   ├── schemas.py          # dataclasses partagées (P4 le crée en premier)
│   ├── data_loader.py      # P1
│   ├── enrichment.py       # P2
│   ├── preprocessing.py    # P3
│   ├── embeddings.py       # P3
│   ├── profile.py          # P4
│   ├── scoring.py          # P4
│   ├── recommender.py      # P4 (point d'entrée unique pour l'app)
│   ├── explain.py          # P4
│   ├── viz.py              # P5
│   └── llm_explain.py      # P6 (bonus)
├── app/
│   └── streamlit_app.py    # P5
└── docs/
    ├── report.md → report.pdf   # P6 (chacun écrit sa section)
    ├── slides.pdf               # P6 (chacun fait ses slides)
    └── figures/
```

---

## 4. Les contrats d'interface (À FIXER AU KICK-OFF, c'est ce qui permet de travailler en parallèle)

### 4.1 `books_clean.parquet` (P1) puis `books_enriched.parquet` (P2)
| Colonne | Type | Rempli par |
|---|---|---|
| `book_id` | str | P1 |
| `title` | str | P1 |
| `authors` | list[str] (noms) | P1 |
| `authors_str` | str | P1 |
| `categories` | list[str] (noms Book Depository) | P1 |
| `description` | str (nettoyée, sans HTML) | P1 |
| `pub_year` | int / NaN | P1 |
| `rating_avg`, `rating_count` | float / int | P1 |
| `format`, `for_ages` | str | P1 |
| `image_url`, `url` | str | P1 |
| `popularity` | float ∈ [0,1] | P1 |
| `embed_text` | str = `title + ". " + description` (tronqué à environ 1 500 caractères) | P1 |
| `main_author`, `description_raw`, `audience` (Children/Young Adult/Adult), `domain` (Fiction/Non-fiction), `is_fiction`, `n_editions`, `n_words` | **ajoutées par P1** (voir README) | P1 |
| `genres` | list[str] (labels du bloc Genre) | P2 |
| `themes` | list[str] (top 3 labels du bloc Thèmes) | P2 |
| `styles` | list[str] (top 2) | P2 |
| `atmospheres` | list[str] (top 2, époque/ambiance) | P2 |
| `keywords` | list[str] (top 5 KeyBERT) | P2 |
| `label_scores` | dict[block → dict[label → float]] (stocké en **chaîne JSON** dans le parquet ; `load_books()` le reconvertit en dict) | P2 |

### 4.2 `data/reference/blocks.json` (P2)
```json
{
  "Genre":      {"Fantasy": "Fantasy fiction with magic, mythical creatures, imaginary worlds and epic quests", "...": "..."},
  "Themes":     {"Identity": "A story about self-discovery, identity and finding who you really are", "...": "..."},
  "Style":      {"Fast-paced page-turner": "A gripping, fast-paced, suspenseful plot that is hard to put down", "...": "..."},
  "Atmosphere": {"Victorian era": "Set in 19th century Victorian England", "Dark and unsettling": "A dark, disturbing, eerie atmosphere", "...": "..."}
}
```
Chaque label a une **phrase descriptive en langage naturel**, et c'est cette phrase qu'on encode. C'est ce qui rend le matching sémantique (« boy wizard school » matche « Fantasy » sans que le mot apparaisse).

### 4.3 `answers.json` (sortie du questionnaire, P5 → P4)
```json
{
  "free_text": {
    "ideal_book":  "I enjoy psychological novels exploring identity and moral dilemmas...",
    "loved_book":  "I loved The Secret History because of the dark academic atmosphere...",
    "atmosphere":  "Something melancholic, rainy, set in an old university",
    "avoid":       "No romance-centered plots, no gore"
  },
  "genres": ["Literary Fiction", "Thriller & Suspense"],
  "themes": ["Identity", "Morality & Guilt", "Friendship"],
  "period": "Contemporary",
  "audience": "Adult",
  "formats": ["Paperback", "Hardback"],
  "favorite_author": "Donna Tartt",
  "guided": {"focus": "Characters", "standalone_only": false, "exclude_favorite_author": true},
  "likert": {"complexity": 4, "pace": 2, "darkness": 4, "popularity": 3,
             "w_genre": 4, "w_themes": 5, "w_style": 3, "w_atmosphere": 4}
}
```

### 4.4 API Python (`src/schemas.py` et `src/recommender.py`)
```python
@dataclass
class UserProfile:
    query_texts: dict[str, str]          # {"global","Genre","Themes","Style","Atmosphere","avoid"}
    query_vecs: dict[str, np.ndarray]    # vecteurs normalisés
    weights: dict[str, float]            # {"Genre":..,"Themes":..,"Style":..,"Atmosphere":..}
    block_affinities: dict[str, dict[str, float]]  # bloc → label → affinité utilisateur (pour le radar profil)
    constraints: dict                    # audience, formats, période stricte, auteurs exclus
    summary: str                         # résumé texte du profil lecteur

@dataclass
class Recommendation:
    rank: int
    book_id: str
    title: str
    authors: str
    image_url: str
    final_score: float                   # ∈ [0,1]
    match_label: str                     # "Excellent" / "Good" / "Partial"
    block_scores: dict[str, float]       # S_desc, S_Genre, S_Themes, S_Style, S_Atmosphere
    contributions: dict[str, float]      # poids × score (pour barres empilées)
    coverage: float
    matched_labels: dict[str, list[tuple[str, float]]]  # bloc → [(label, sim)]
    best_sentence: str                   # phrase du résumé la plus proche de la requête
    best_sentence_score: float
    explanation: str

# Point d'entrée unique utilisé par l'app :
def recommend(answers: dict, top_k: int = 3) -> tuple[UserProfile, list[Recommendation]]: ...
```
```python
# src/embeddings.py (P3)
class EmbeddingEngine:
    def encode(self, texts: list[str]) -> np.ndarray: ...          # L2-normalisé
    def book_matrix(self) -> np.ndarray: ...                         # (N, d), aligné sur books_enriched
    def label_matrix(self, block: str) -> tuple[list[str], np.ndarray]: ...
    def search(self, query_vec, k=300) -> tuple[np.ndarray, np.ndarray]: ...  # indices, scores
    def best_sentence(self, query_vec, text: str) -> tuple[str, float]: ...
```

---

## 5. La formule de score (proposition, à justifier dans le rapport)

Notations : `u` est l'utilisateur, `j` un livre, `cos` la similarité cosinus sur des vecteurs normalisés.

**1. Score global de description**
`S_desc(j) = cos(q_global, e_j)` où `q_global` est l'embedding de la concaténation des réponses libres (ideal_book + loved_book + atmosphere) et `e_j` l'embedding de `embed_text`.

**2. Score par bloc** (même logique « max puis moyenne » que le pseudo-code de la prof)
Pour chaque bloc `b ∈ {Genre, Themes, Style, Atmosphere}` :
- `P_b(u)` = phrases de préférence de l'utilisateur pour ce bloc, soit les descriptions des labels cochés + le texte libre lié + les phrases issues des Likert.
- `L_b(j)` = labels attribués au livre `j` (par P2) et leur embedding.
- `S_b(j) = moyenne_{p ∈ P_b(u)} [ max_{l ∈ L_b(j) ∪ {desc_j}} cos(p, l) ]`

**3. Calibration** : les cosinus SBERT tombent souvent entre 0,1 et 0,7, ce qui est peu lisible. On les ramène sur [0,1] :
`s̃ = clip((s − p5) / (p95 − p5), 0, 1)`, où p5/p95 sont les percentiles calculés sur le catalogue.

**4. Score de couverture** (exigé par le sujet)
`Coverage(j) = Σ_b W_b · s̃_b(j) / Σ_b W_b`, avec `W_b` = l'importance donnée par l'utilisateur en Likert (1 à 5).

**5. Score final**
`Final(j) = 0.5 · s̃_desc(j) + 0.4 · Coverage(j) + λ · Pop(j) − μ · max(0, cos(q_avoid, e_j) − τ)`
- `Pop(j) = (rating_avg/5) · log(1+rating_count)/log(1+max_count)`, avec `λ ∈ [0 ; 0.2]` réglé par le Likert « populaire ↔ pépites ».
- `μ = 0.5`, `τ = 0.3` (pénalité « à éviter »).
- **Filtres durs** : public (for_ages), format, période si « strict », auteur exclu.
- **Diversité** : au maximum 1 livre par auteur dans le top 3 (ou MMR).

**6. Libellés** (comme l'étape 5 de la prof) : `Final ≥ 0.75` → « Excellent match », `0.55–0.75` → « Good match », `< 0.55` → « Partial match ».

Toutes les constantes (0.5 / 0.4 / λ / μ / τ) vont dans `config.py`, et **P6 fait une ablation** pour les justifier.

---

## 6. Le questionnaire (contenu exact proposé)

| # | Type (exigé) | Question | Usage NLP |
|---|---|---|---|
| Q1 | **Ouverte** (obligatoire, ≥ 30 car.) | *Describe the kind of book you'd love to read right now.* | `q_global` |
| Q2 | **Ouverte** | *Tell us about a book you loved and why.* | `q_global` + bloc Thèmes |
| Q3 | **Ouverte** | *What atmosphere or setting are you in the mood for?* | bloc Atmosphère |
| Q4 | **Ouverte** (optionnelle) | *Anything you want to avoid?* | `q_avoid` (pénalité) |
| Q5 | **Cases à cocher** | Genres préférés (≈ 15 labels de `blocks.json`) | bloc Genre |
| Q6 | **Cases à cocher** | Thèmes favoris (≈ 20 labels) | bloc Thèmes |
| Q7 | **Choix multiple (radio)** | Époque : Ancient / Medieval / 18–19th c. / 20th c. / Contemporary / Future / No preference | bloc Atmosphère + filtre |
| Q8 | **Choix multiple (selectbox)** | Public : Children / Young Adult / Adult | filtre `for_ages` |
| Q9 | **Multiselect** | Formats acceptés | filtre |
| Q10 | **Likert 1–5** | Complexité narrative (simple → dense/littéraire) | **converti en phrase** → bloc Style |
| Q11 | **Likert 1–5** | Rythme (lent et contemplatif → page-turner) | phrase → bloc Style |
| Q12 | **Likert 1–5** | Tonalité (légère/feel-good → sombre) | phrase → bloc Atmosphère |
| Q13 | **Likert 1–5** | Best-sellers ↔ pépites méconnues | λ |
| Q14 | **Likert 1–5** ×4 | Importance de : Genre / Thèmes / Style / Atmosphère | poids `W_b` |
| Q15 | **Guidée** | *Do you have a favourite author?* → si oui, champ texte | centroïde des livres de l'auteur, ajouté à `q_global` |
| Q16 | **Guidée** | *Do you prefer character-driven or plot-driven stories?* | phrase → bloc Style |
| Q17 | **Guidée** | *Should we exclude books by your favourite author?* | filtre |

---

## 7. Répartition du travail (6 personnes, charge équivalente d'environ 15 à 18 h chacune)

| | Rôle | Livrables code | Notebook | Section du PDF | Slides |
|---|---|---|---|---|---|
| **P1** | Data Engineer | `data_loader.py`, `books_sample/clean.parquet` | 01 | Dataset : origine, structure, nettoyage, EDA | 2 |
| **P2** | Référentiel & enrichissement NLP | `blocks.json`, `enrichment.py`, `books_enriched.parquet` | 02 | Construction du référentiel, zero-shot, KeyBERT | 2 |
| **P3** | Moteur sémantique | `preprocessing.py`, `embeddings.py`, `.npy` | 03 | Preprocessing, embeddings, comparaison des modèles, similarité | 2–3 |
| **P4** | Scoring & recommandation | `schemas.py`, `profile.py`, `scoring.py`, `recommender.py`, `explain.py` | 04 | Formule de score, couverture, explications | 2–3 |
| **P5** | App web & visualisation | `streamlit_app.py`, `viz.py` | 05 | Questionnaire, UI, visualisations | 2 + **démo live** |
| **P6** | Évaluation, intégration & livrables | `06_evaluation`, README, PDF, slides, `llm_explain.py` (bonus) | 06 | Évaluation, ablations, limites, défis + assemblage | intro + conclusion |

**Pourquoi c'est équitable** : chacun a un module de code, un notebook documenté, une section du rapport et 2–3 slides. P1 est moins chargé côté modèle mais porte l'EDA et le nettoyage d'un dataset lourd. P6 a moins de code « produit » mais porte l'évaluation, l'intégration et toute la mise en forme finale.

**Coordination** : P6 joue le rôle de chef de projet/intégrateur (suivi du planning, merge des PR, tests de bout en bout).

---

## 8. Planning le plus court possible (environ 7 jours, à adapter à la deadline)

L'astuce : **tout le monde code dès le jour 1 contre des données factices** et contre les contrats de la section 4.

| Jour | P1 Data | P2 Référentiel | P3 Moteur | P4 Scoring | P5 App | P6 Éval/Intég |
|---|---|---|---|---|---|---|
| **J0 (1 h, tous ensemble)** | Kick-off : valider les contrats §4, la formule §5, le questionnaire §6, créer le repo, `requirements.txt`, `config.py` | | | | | |
| **J1** | Télécharger, explorer, livrer **`books_sample.parquet` (500 livres) en 2 h** | Écrire `blocks.json` v1 (4 blocs, ≈ 60 labels avec descriptions) | Preprocessing + encodage SBERT sur le sample | `schemas.py` en 1er, puis `profile.py` avec des vecteurs aléatoires | Squelette Streamlit : les 17 questions → `answers.json` | Repo, README, template du rapport, **10 personas de test** |
| **J2** | `books_clean.parquet` complet + EDA | Mapping catégories → genres, zero-shot thèmes/style/atmosphère | Baselines TF-IDF + GloVe, `EmbeddingEngine`, `search`, `best_sentence` | `scoring.py` (formule complète) + calibration | Page résultats avec un **stub** `recommend()` | Annoter à la main 60 livres (thèmes/genre) pour valider P2 |
| **J3** | Utilitaires (auteur → livres, filtres), aide P2 | KeyBERT, `books_enriched.parquet` final, validation sur les 60 annotés | Embeddings finaux du catalogue + labels, comparaison MiniLM vs MPNet | `recommender.py` + `explain.py` (templates) | `viz.py` : radar, barres empilées, carte UMAP | Script d'évaluation (precision@3, comparaison des encodeurs) |
| **J4 : INTÉGRATION** | Tous : brancher le vrai `recommend()` dans l'app, corriger, tester les 10 personas | | | | | |
| **J5** | Section du PDF + slides | idem | idem | idem + ajuster les poids selon l'ablation | idem + polish UI | Ablations, bonus LLM si le temps le permet |
| **J6** | Relecture croisée des notebooks (commentaires partout) | | | | | Assemblage PDF + slides |
| **J7** | **Répétition générale** de la soutenance (chrono, démo live, plan B en vidéo) | | | | | |

**Règles d'équipe :**
- Une branche Git par personne, PR vers `main`, P6 merge.
- Aucune donnée brute dans Git. Les `.parquet` (quelques Mo) peuvent y aller. Si les `.npy` sont trop gros, on les partage sur un Drive ou on les régénère avec un script.
- Toute modification d'un contrat (§4) est annoncée sur le groupe **avant** d'être faite.
- Chacun commente son code **à chaque étape** (c'est noté) et écrit des cellules Markdown explicatives dans son notebook.

---

## 9. Présentation (notation individuelle)

Durée à confirmer avec la prof. Ordre proposé (≈ 2–3 min par personne) :
1. **P6** : problème, objectifs, architecture globale (schéma du pipeline).
2. **P1** : dataset, filtrage, EDA (pourquoi 10 000 livres et pas 1 million).
3. **P2** : référentiel de blocs et enrichissement sémantique (zero-shot par embeddings, KeyBERT, validation).
4. **P3** : preprocessing et embeddings ; **démo « sémantique vs mots-clés »** (une requête sans mot-clé commun qui trouve quand même le bon livre) ; comparaison TF-IDF / GloVe / SBERT.
5. **P4** : formule de score, couverture, exemple calculé à la main sur un livre.
6. **P5** : **démo live** (persona préparée) + visualisations.
7. **P6** : évaluation, défis rencontrés et solutions, limites, bonus, conclusion.

**Défis à raconter** (le sujet demande « challenges faced and how you solved them ») : taille du dataset, éditions en double, descriptions vides ou en HTML, absence de colonnes thèmes/style (résolue par zero-shot), cosinus compressés (résolu par calibration), troncature des longues descriptions (256 tokens), biais de popularité, saisies utilisateur trop courtes.

**Plan B** : enregistrer une vidéo de la démo au cas où le Wi-Fi ou le chargement du modèle lâche.

---

## 10. Checklist de conformité (à cocher avant de rendre)

- [ ] Questionnaire web avec **texte libre, Likert, guidées, choix multiple, cases à cocher** (Req. 1)
- [ ] Assez de texte libre (≥ 3 questions ouvertes, 1 obligatoire) (Req. 1)
- [ ] Réponses sauvegardées en **JSON/CSV** (consigne « store responses in a structured format »)
- [ ] Dataset avec **catégories/blocs, attributs, descriptions en langage naturel, mots-clés, relations** (Req. 2)
- [ ] Origine, structure et preprocessing du dataset **documentés** (Req. 2)
- [ ] Preprocessing + **embeddings de mots (GloVe/Word2Vec/fastText)** + **embeddings contextuels (SBERT)** + cosinus (Req. 3)
- [ ] **Justification** du modèle et de la mesure de similarité (Req. 3)
- [ ] Formule de score **définie et justifiée** avec poids, agrégation par bloc, seuils, contraintes (Req. 4)
- [ ] **Score de couverture** multi-blocs `Σ Wi·Si / Σ Wi` (Expected outcomes)
- [ ] **Top 3**, chacun avec score, critères principaux et explication par template (Req. 5)
- [ ] Profil/résumé utilisateur, scores par bloc, radar + barres + carte de similarité (Req. 6)
- [ ] Dans l'app, on **montre** d'où vient le score (décomposition + phrase la plus proche) (Req. 6)
- [ ] L'app fonctionne **sans LLM** (si bonus présent)
- [ ] Notebooks commentés à **chaque étape**
- [ ] **PDF** de documentation
- [ ] Slides + démo, **les 6 présentent**
- [ ] Dépôt sur **BOOSTCAMP** avant la deadline

---

## 11. Prompts à envoyer à Claude (un par personne)

**Mode d'emploi** : copier d'abord le **bloc CONTEXTE COMMUN**, puis le prompt de sa personne, dans **le même message**. Joindre si possible le PDF du sujet et ce fichier `PLAN_PROJET.md`. Si Claude Code est utilisé dans le repo, il lira directement les fichiers.

### 11.0 CONTEXTE COMMUN (à coller avant chaque prompt)

```text
CONTEXTE DU PROJET
Je suis étudiant en 5e année (ING5 Data & IA, ECE). Projet NLP en groupe de 6 : « Semantic Analysis and Intelligent Contextual Recommendation Systems », thème 3 de l'annexe : recommandation de livres selon les préférences littéraires.
Dataset : Book Depository (Kaggle sp1thas/book-depository-dataset) : dataset.csv (plus d'1M de livres, colonnes id, title, authors [liste d'IDs], categories [liste d'IDs], description, lang, publication-date, rating-avg, rating-count, format, for-ages, image-url, url…) + authors.csv, categories.csv, formats.csv (tables d'IDs). Vérifie les noms exacts des colonnes en lisant les fichiers.

Exigences du sujet (à respecter strictement) :
- Analyse SÉMANTIQUE et contextuelle, pas du keyword matching ni du scoring purement numérique.
- Questionnaire web (Streamlit) avec questions ouvertes, Likert, guidées, choix multiple, cases à cocher ; assez de texte libre.
- Référentiel structuré avec blocs/catégories, attributs, descriptions en langage naturel, mots-clés, relations.
- Pipeline NLP : preprocessing, word embeddings (GloVe/Word2Vec/fastText), embeddings contextuels (SBERT), similarité cosinus ; justifier les choix.
- Score d'affinité sémantique par bloc + score de couverture pondéré Σ Wi·Si / Σ Wi + formule justifiée (poids, seuils, contraintes).
- Top 3 recommandations, chacune avec : item, score, critères principaux, explication générée par template (sans GenAI).
- Visualisations (radar, barres, carte de similarité) montrant comment le résultat découle de l'analyse sémantique.
- Code commenté À CHAQUE ÉTAPE + notebooks Jupyter avec cellules Markdown explicatives + doc PDF. LLM = bonus optionnel, l'app doit marcher sans.

ARCHITECTURE FIXÉE PAR L'ÉQUIPE (ne pas modifier les contrats sans prévenir) :
- Python 3.11, pandas, pyarrow, numpy, scikit-learn, sentence-transformers (modèle par défaut all-MiniLM-L6-v2, comparé à all-mpnet-base-v2), gensim, keybert, umap-learn, plotly, streamlit, spacy.
- Repo : data/{raw,processed,reference,embeddings}, notebooks/01..06, src/{config,schemas,data_loader,enrichment,preprocessing,embeddings,profile,scoring,recommender,explain,viz,llm_explain}.py, app/streamlit_app.py, docs/.
- books_enriched.parquet colonnes : book_id, title, authors (list[str]), authors_str, categories (list[str]), description, pub_year, rating_avg, rating_count, format, for_ages, image_url, url, popularity [0-1], embed_text (= title + ". " + description, ~1500 car. max), main_author, audience (Children/Young Adult/Adult), domain (Fiction/Non-fiction), is_fiction, description_raw, n_editions, genres, themes, styles, atmospheres, keywords (list[str]), label_scores (dict bloc→label→score).
- data/reference/blocks.json : {"Genre": {label: phrase descriptive}, "Themes": {...}, "Style": {...}, "Atmosphere": {...}}.
- answers.json (sortie du questionnaire) : {"free_text": {"ideal_book","loved_book","atmosphere","avoid"}, "genres": [...], "themes": [...], "period": str, "audience": str, "formats": [...], "favorite_author": str, "guided": {"focus": "Characters"|"Plot"|"Both", "standalone_only": bool, "exclude_favorite_author": bool}, "likert": {"complexity","pace","darkness","popularity","w_genre","w_themes","w_style","w_atmosphere"} (1-5)}.
- Dataclasses (src/schemas.py) : UserProfile(query_texts, query_vecs, weights, block_affinities, constraints, summary) et Recommendation(rank, book_id, title, authors, image_url, final_score, match_label, block_scores, contributions, coverage, matched_labels, best_sentence, best_sentence_score, explanation).
- Point d'entrée : src/recommender.py → recommend(answers: dict, top_k=3) -> tuple[UserProfile, list[Recommendation]].
- EmbeddingEngine (src/embeddings.py) : encode(texts)->np.ndarray normalisé ; book_matrix() ; label_matrix(block)->(labels, matrix) ; search(query_vec, k=300)->(indices, scores) ; best_sentence(query_vec, text)->(phrase, score).
- Formule : Final = 0.5·S̃_desc + 0.4·Coverage + λ·Pop − μ·max(0, cos(avoid, livre) − τ) ; Coverage = Σ W_b·S̃_b / Σ W_b sur les blocs Genre/Themes/Style/Atmosphere ; S_b = moyenne sur les phrases de préférence utilisateur du max de cosinus avec les labels du livre ; S̃ = calibration percentile p5/p95 ; constantes dans src/config.py.

CONSIGNES DE TRAVAIL POUR TOI (Claude) :
- Code propre, typé, docstrings, et un commentaire explicatif à chaque étape (c'est noté).
- Notebook avec cellules Markdown qui expliquent le POURQUOI de chaque choix (servira de base au rapport).
- Donne-moi aussi un paragraphe de 300-500 mots prêt à coller dans le rapport PDF pour ma partie, et un plan de 2-3 slides.
- Avance par étapes, fais tourner le code et vérifie les sorties ; si quelque chose dans les données ne correspond pas au contrat, signale-le.
```

### 11.1 Prompt PERSONNE 1 : Data Engineer

```text
MA PARTIE : PERSONNE 1, préparation des données (notebook 01_data_preparation.ipynb + src/data_loader.py).

Ce que je dois produire :
1. Script/cellules de téléchargement (kaggle CLI) vers data/raw/, puis lecture EFFICACE de dataset.csv (gros fichier) : usecols pour ne charger que les colonnes utiles, dtypes explicites, lecture par chunks si nécessaire. Afficher la taille, les colonnes, le taux de valeurs manquantes.
2. Jointures : convertir les colonnes authors et categories (chaînes du type "[1, 2]") en listes d'IDs (ast.literal_eval sécurisé), puis en listes de NOMS via authors.csv et categories.csv. Idem pour format via formats.csv si c'est un ID.
3. Nettoyage du texte de description : retirer le HTML (BeautifulSoup ou regex), décoder les entités, normaliser les espaces, retirer les mentions marketing répétitives si utile. NE PAS retirer les stopwords ni lemmatiser ici (SBERT a besoin du texte naturel ; le preprocessing des baselines est fait par P3).
4. Filtrage justifié : lang == "en" ; description nettoyée >= 300 caractères ; rating-count >= 20 (expérimenter 10/20/50 et montrer combien de livres restent pour chaque seuil) ; dédoublonnage des éditions (clé titre normalisé + premier auteur, garder le max rating-count) ; décider (et documenter) le sort des catégories non-fiction/manuels. Cible : 5 000 à 15 000 livres. Afficher un tableau « étape de filtrage → nombre de lignes restantes ».
5. Colonnes dérivées : book_id (str), pub_year (int depuis publication-date), authors_str, popularity = (rating_avg/5) * log(1+rating_count)/log(1+max_rating_count), embed_text = title + ". " + description tronquée proprement à ~1500 caractères (fin de phrase).
6. EXPORTS :
   a. EN PREMIER ET VITE : data/processed/books_sample.parquet = 500 livres avec EXACTEMENT les colonnes du contrat (les colonnes de P2 genres/themes/styles/atmospheres/keywords/label_scores peuvent être vides) pour débloquer l'équipe.
   b. data/processed/books_clean.parquet complet.
7. EDA avec figures plotly/matplotlib sauvegardées dans docs/figures/ : distribution des longueurs de description (en mots et en tokens approximatifs, pour montrer l'impact de la limite de 256 tokens de MiniLM), top 30 catégories, distribution rating_avg et rating_count (log), livres par année, top auteurs, formats, for_ages.
8. src/data_loader.py : load_books(path) -> DataFrame ; get_books_by_author(df, name) avec matching flou (rapidfuzz ou difflib) ; apply_constraints(df, audience, formats, period, excluded_authors) -> masque booléen ; mapping période → intervalle d'années (pub_year n'est PAS l'époque du récit, le signaler : l'époque du récit est traitée sémantiquement par P2).
9. Documentation pour le rapport : origine (Kaggle, Book Depository, scraping), licence, structure (tableau des colonnes gardées/supprimées et pourquoi), étapes de preprocessing, limites (biais anglophone, biais de popularité, descriptions marketing).

Livre-moi le notebook complet commenté, le module data_loader.py, le paragraphe pour le rapport et le plan de mes 2 slides.
```

### 11.2 Prompt PERSONNE 2 : Référentiel & enrichissement sémantique

```text
MA PARTIE : PERSONNE 2, construction du référentiel de blocs et enrichissement NLP du catalogue (notebook 02_reference_enrichment.ipynb + src/enrichment.py + data/reference/blocks.json).

Problème à résoudre : le sujet suggère la structure BookID | Genre | Title | Author | Synopsis | Themes | Style | Keywords, mais le dataset Book Depository n'a ni thèmes, ni style, ni mots-clés, et ses catégories sont très nombreuses et bruitées. Je dois les DÉRIVER par analyse sémantique.

Ce que je dois produire :
1. data/reference/blocks.json avec 4 blocs ; chaque label a une phrase descriptive riche en anglais (c'est cette phrase qui est encodée, pas juste le mot) :
   - Genre (~15) : Fantasy, Science Fiction, Mystery & Detective, Thriller & Suspense, Horror, Romance, Historical Fiction, Literary Fiction, Classics, Young Adult, Children's, Humour, Biography & Memoir, History, Science & Nature, Self-help & Psychology, Philosophy & Religion, Poetry & Drama, Graphic Novels…
   - Themes (~20) : Identity & self-discovery, Family relationships, Friendship, Love & heartbreak, Coming of age, Morality & guilt, Power & politics, War & conflict, Survival, Grief & loss, Social injustice, Revenge, Redemption, Freedom & oppression, Technology & humanity, Nature & environment, Faith & spirituality, Madness & the mind, Adventure & exploration, Crime & justice…
   - Style (~10) : Fast-paced page-turner, Slow and contemplative, Dense literary prose, Lyrical/poetic, Humorous & witty, Simple and accessible, Character-driven, Plot-driven/twisty, Epic multi-perspective, Dark and gritty voice…
   - Atmosphere (~15, époque + ambiance) : Ancient world, Medieval, Victorian/19th century, World Wars era, Contemporary, Futuristic/dystopian, Dark & unsettling, Cozy & heartwarming, Melancholic, Whimsical & magical, Tense & suspenseful, Romantic, Academic/campus, Small town, Urban…
   Justifie la taxonomie (inspirée des genres standards de l'édition / BISAC, etc.).
2. Mapping catégories Book Depository → Genre : encoder les noms de catégories et les descriptions de genres avec SBERT, assigner par cosinus (+ quelques règles mots-clés pour les cas évidents), produire une table de correspondance relue manuellement sur les 100 catégories les plus fréquentes. Un livre reçoit les genres de ses catégories ; à défaut, on prend le genre le plus proche de sa description.
3. Zero-shot par embeddings (rapide, CPU-friendly) pour Themes, Style et Atmosphere : cos(embedding de description, embedding de la phrase du label) → garder le top 3 thèmes, le top 2 styles et le top 2 atmosphères au-dessus d'un seuil (calibrer le seuil). Stocker tous les scores dans label_scores. Optionnel : comparer sur 200 livres avec un vrai zero-shot NLI (facebook/bart-large-mnli via pipeline zero-shot-classification) et discuter le compromis qualité/temps.
4. Mots-clés : KeyBERT (même modèle SBERT, keyphrase_ngram_range=(1,2), use_mmr=True, diversity=0.5, top_n=5). Si c'est trop lent sur tout le catalogue, le faire en batch ou se rabattre sur TF-IDF pour une partie et le documenter.
5. Validation qualité : sur 60 livres annotés à la main par P6 et moi (genre + 2-3 thèmes), mesurer la precision@k des labels prédits, montrer 10 exemples bons et 5 erreurs commentés.
6. Export data/processed/books_enriched.parquet (contrat exact) + src/enrichment.py avec des fonctions réutilisables : load_blocks(), assign_genres(df, engine), assign_labels(df, engine, block, top_k, threshold), extract_keywords(df).
7. Visualisations pour le rapport : distribution des genres et thèmes dans le catalogue, heatmap genre × thème (co-occurrences) qui illustre les « relations entre catégories ».
Pour encoder, utilise le modèle all-MiniLM-L6-v2 (ou EmbeddingEngine de P3 s'il est prêt ; sinon un encodeur minimal compatible).

Livre-moi blocks.json, le notebook complet commenté, enrichment.py, le paragraphe pour le rapport et le plan de mes 2 slides.
```

### 11.3 Prompt PERSONNE 3 : Moteur sémantique (preprocessing + embeddings)

```text
MA PARTIE : PERSONNE 3, moteur sémantique (notebook 03_semantic_engine.ipynb + src/preprocessing.py + src/embeddings.py + data/embeddings/).

Ce que je dois produire :
1. src/preprocessing.py :
   - preprocess_for_transformer(text) : nettoyage minimal (espaces, caractères parasites) ; expliquer pourquoi on garde stopwords et ponctuation pour SBERT (le contexte compte).
   - preprocess_for_classic(text) : minuscules, tokenisation, stopwords, lemmatisation spaCy (en_core_web_sm) pour TF-IDF et GloVe.
   - split_sentences(text) pour l'explicabilité.
2. Trois représentations, avec l'explication théorique de chacune dans le notebook (statique vs contextuel) :
   a. Baseline lexicale TF-IDF (sklearn) : montrer qu'elle échoue sans mot commun.
   b. Word embeddings GloVe (gensim.downloader 'glove-wiki-gigaword-100' ou fastText) : moyenne (ou moyenne pondérée TF-IDF) des vecteurs de mots.
   c. Embeddings contextuels SBERT : all-MiniLM-L6-v2 ET all-mpnet-base-v2, normalize_embeddings=True, batch_size adapté, barre de progression. Gérer la limite de tokens (256 / 384) : comparer troncature vs moyenne de chunks sur un échantillon.
3. src/embeddings.py : classe EmbeddingEngine (contrat : encode, book_matrix, label_matrix, search, best_sentence), avec cache disque (data/embeddings/{model_name}_books.npy + book_ids.json pour garantir l'alignement), chargement paresseux du modèle. search = produit scalaire numpy sur vecteurs normalisés (= cosinus), FAISS IndexFlatIP en option si N est grand. best_sentence = découpe le résumé en phrases, encode, renvoie la phrase la plus proche + son score.
4. Justification de la similarité cosinus (invariance à la norme, standard pour SBERT, équivalence avec le produit scalaire une fois normalisé), comparée rapidement à la distance euclidienne.
5. Démonstrations dans le notebook (elles serviront en soutenance) :
   - « Semantic vs keyword » : 5 requêtes paraphrasées sans mot commun avec le livre cible (ex. "a boy discovers he is a wizard and goes to a magic school" → Harry Potter si présent) ; montrer le rang du bon livre avec TF-IDF vs GloVe vs SBERT.
   - Distribution des cosinus utilisateur ↔ catalogue (pour justifier la calibration percentile p5/p95 utilisée par P4) ; calculer et sauver ces percentiles dans data/embeddings/calibration.json.
   - Projection UMAP 2D des embeddings du catalogue colorée par genre (fichier data/embeddings/umap_2d.npy réutilisé par P5 pour la carte de similarité) + clustering KMeans optionnel avec un score silhouette.
6. Benchmark de temps (encodage du catalogue, temps de requête) pour MiniLM vs MPNet.
7. Fournir à P6 une fonction get_encoder(name) -> callable(texts)->np.ndarray pour qu'il puisse comparer les 4 encodeurs sur les personas.
Travaille d'abord sur books_sample.parquet, puis sur books_enriched.parquet quand il est prêt.

Livre-moi les modules, le notebook complet commenté, les fichiers d'embeddings générés, le paragraphe pour le rapport et le plan de mes 2-3 slides.
```

### 11.4 Prompt PERSONNE 4 : Scoring, recommandation & explications

```text
MA PARTIE : PERSONNE 4, profil utilisateur, scoring sémantique, recommandation et explications (notebook 04_scoring_recommendation.ipynb + src/schemas.py, src/profile.py, src/scoring.py, src/recommender.py, src/explain.py, src/config.py pour les constantes).

Ce que je dois produire (dans cet ordre) :
1. src/schemas.py : dataclasses UserProfile et Recommendation EXACTEMENT selon le contrat. À livrer EN PREMIER (l'app de P5 en dépend).
2. src/profile.py → build_profile(answers, engine) -> UserProfile :
   - q_global = encodage de ideal_book + loved_book + atmosphere (expérimenter : concaténation vs moyenne des embeddings de chaque réponse) ; si favorite_author est trouvé dans le catalogue (data_loader.get_books_by_author de P1), mélanger avec le centroïde de ses livres (ex. 0.7 texte / 0.3 auteur).
   - Phrases de préférence par bloc P_b(u) : descriptions des labels cochés (via blocks.json) + texte libre lié + LIKERT CONVERTIS EN PHRASES (ex. complexity 5 → "a complex, dense, literary and demanding narrative" ; pace 1 → "a slow, contemplative, quiet story" ; darkness 5 → "a dark, disturbing and intense atmosphere" ; focus Characters → "a character-driven story with deep psychological portraits"). C'est ce qui garde l'analyse sémantique même pour les questions numériques.
   - weights W_b depuis likert w_genre/w_themes/w_style/w_atmosphere ; λ depuis likert popularity ; q_avoid si fourni.
   - block_affinities : pour chaque label de chaque bloc, l'affinité de l'utilisateur (cos entre q_global/P_b et la phrase du label) → sert au radar « profil lecteur ».
   - summary : résumé texte du profil généré par template (top genres, top thèmes, style, ambiance).
3. src/scoring.py : implémenter la formule du contrat :
   S_desc = cos(q_global, e_livre) ; S_b = moyenne_{p∈P_b} max_{l∈labels_b(livre) ∪ {desc}} cos(p,l) ; calibration percentile (calibration.json de P3) ; Coverage = Σ W_b·S̃_b/Σ W_b ; Final = 0.5·S̃_desc + 0.4·Coverage + λ·Pop − μ·max(0, cos(q_avoid,e)−τ) ; match_label avec seuils 0.75/0.55. Calcul VECTORISÉ en numpy (matrices), pas de boucles Python sur les livres.
4. src/recommender.py → recommend(answers, top_k=3) :
   étape 1 : retrieval top-300 par S_desc (engine.search) ; étape 2 : filtres durs (data_loader.apply_constraints) ; étape 3 : re-ranking avec la formule ; étape 4 : diversité (max 1 livre par auteur, ou MMR) ; étape 5 : construire les Recommendation (block_scores, contributions = poids_normalisé × score par composante, matched_labels = top 2 labels communs par bloc avec sim, best_sentence via engine.best_sentence). Mettre en cache le chargement (fonction get_recommender() singleton).
5. src/explain.py : explication TRANSPARENTE par template à partir des résultats NLP, par exemple :
   "Recommended because its synopsis is semantically close to your description (similarity 0.71) — especially: «<best_sentence>». It shares the themes you care about: Identity (0.82), Morality & guilt (0.77). Its style (Dense literary prose) matches your wish for a complex narrative. Atmosphere match: Melancholic (0.69). Overall preference coverage: 81 %."
   Plusieurs variantes de phrases selon le niveau de score, mentionner le critère le plus contributif et le plus faible.
6. Notebook : dérouler à la main l'exemple d'un utilisateur (afficher chaque S_b, la calibration, la couverture, le score final pour 3 livres) comme l'exemple « Step 1 à 5 » du sujet ; justifier chaque terme de la formule et chaque constante ; tester 3 personas contrastés (fantasy jeunesse / thriller sombre / essai philosophique) et vérifier la cohérence.
7. Tests unitaires simples (pytest) : scores dans [0,1], top_k respecté, filtres respectés, déterminisme.
Avant que P2/P3 aient fini, travaille avec books_sample.parquet et des embeddings calculés à la volée.

Livre-moi tous les modules, le notebook complet commenté, les tests, le paragraphe pour le rapport (formule en LaTeX + justification) et le plan de mes 2-3 slides.
```

### 11.5 Prompt PERSONNE 5 : Application web & visualisations

```text
MA PARTIE : PERSONNE 5, application Streamlit (app/streamlit_app.py) + visualisations (src/viz.py) + notebook 05_app_and_visualization.ipynb.

Ce que je dois produire :
1. Questionnaire multi-étapes (st.session_state pour naviguer Précédent/Suivant, barre de progression) avec EXACTEMENT ces questions (tous les types exigés) :
   Étape 1 « Tell us about your reading » : Q1 text_area obligatoire min 30 caractères (Describe the kind of book you'd love to read right now), Q2 text_area (A book you loved and why), Q3 text_area (Atmosphere/setting you're in the mood for), Q4 text_area optionnel (Anything to avoid?). Placeholders avec des exemples.
   Étape 2 « Genres & themes » : Q5 cases à cocher genres, Q6 cases à cocher thèmes (labels lus depuis data/reference/blocks.json, avec la phrase descriptive en tooltip via help=).
   Étape 3 « Context » : Q7 radio époque, Q8 selectbox public (Children/Young Adult/Adult), Q9 multiselect formats.
   Étape 4 « Your taste (1-5) » : sliders Likert avec libellés aux extrémités : complexité, rythme, tonalité légère↔sombre, best-sellers↔pépites ; importance de Genre/Thèmes/Style/Atmosphère.
   Étape 5 « A few guided questions » : favourite author (radio oui/non → text_input conditionnel), character-driven vs plot-driven (radio), exclude favourite author (checkbox).
   Validation des champs, puis sauvegarde des réponses en JSON horodaté dans data/responses/ (format answers.json du contrat) + bouton de téléchargement.
2. Page résultats (après appel à recommender.recommend(answers) ; tant que P4 n'a pas fini, utiliser un STUB qui renvoie des Recommendation factices mais conformes au contrat) :
   - Carte « Your reader profile » : summary + radar plotly des affinités par bloc (block_affinities).
   - Top 3 sous forme de cartes : couverture (image_url, avec fallback), titre, auteurs, badge match_label, final_score en %, explication, best_sentence surlignée, chips des thèmes/genres communs.
   - Pour chaque livre : radar « vous vs ce livre » sur les 4 blocs + barres horizontales empilées de la décomposition du score (contributions), expander « How was this score computed? » montrant chaque S_b, les poids, la formule.
   - Carte de similarité : scatter UMAP 2D du catalogue (data/embeddings/umap_2d.npy de P3) coloré par genre, avec le point utilisateur (projection via umap transform ou plus proches voisins) et les 3 livres mis en évidence.
   - Heatmap phrases utilisateur × labels/phrases du livre (similarité cosinus) pour montrer l'analyse sémantique.
   - Bouton « Start again ».
3. src/viz.py : fonctions pures qui renvoient des figures plotly (radar_profile, radar_compare, score_breakdown, similarity_map, similarity_heatmap), réutilisables dans les notebooks et le rapport (export PNG via kaleido dans docs/figures/).
4. UX : st.cache_resource pour le modèle et le recommender (le premier chargement est lent : afficher un spinner), mise en page responsive (colonnes), thème cohérent, textes d'aide, gestion des erreurs (aucun résultat après filtres → message + relâcher les filtres).
5. Préparer 2 personas « de démo » pré-remplissables par un bouton (mode démo) pour la soutenance, et le script de la démo (ce que je clique et dis, en 3 minutes).
6. Notebook 05 : captures d'écran et explication de chaque écran et de chaque visualisation (ce qu'elle montre, pourquoi c'est interprétable).
Commande de lancement : streamlit run app/streamlit_app.py depuis la racine (gérer sys.path pour importer src/).

Livre-moi l'app complète commentée, viz.py, le stub, le notebook, le script de démo, le paragraphe pour le rapport et le plan de mes 2 slides.
```

### 11.6 Prompt PERSONNE 6 : Évaluation, intégration, livrables (+ bonus)

```text
MA PARTIE : PERSONNE 6, évaluation, intégration, documentation finale et bonus (notebook 06_evaluation.ipynb, README.md, docs/report.md → PDF, slides, src/llm_explain.py en bonus).

Ce que je dois produire :
1. Jeu de test : 10 personas variés écrits en answers.json (fantasy jeunesse, thriller nordique sombre, romance historique, SF dystopique, roman psychologique littéraire, polar cosy, mémoires/biographie, horreur, classiques XIXe, livre feel-good), chacun avec les genres/thèmes ATTENDUS (vérité terrain) + 2 ou 3 livres du catalogue jugés pertinents à la main.
2. Annotation manuelle de 60 livres (genre + 2-3 thèmes) avec P2 pour valider l'enrichissement → fichier data/reference/gold_60.csv.
3. Évaluation quantitative :
   - Comparaison des encodeurs (TF-IDF, GloVe moyenne, MiniLM, MPNet via get_encoder de P3) : genre-precision@3 (part des top 3 dont un genre ∈ genres attendus), theme-overlap@3, MRR des livres jugés pertinents, temps de réponse. Tableau + barplot.
   - Test « paraphrase/robustesse » : même persona reformulé 3 fois → stabilité du top 3 (Jaccard).
   - Ablation de la formule : S_desc seul / Coverage seule / sans calibration / sans pénalité avoid / sans popularité / formule complète → justifier les poids 0.5/0.4/λ/μ choisis par P4 (proposer des ajustements si besoin).
   - Sensibilité aux poids Likert : montrer que monter W_Themes fait bien remonter les livres thématiquement proches.
   - Évaluation qualitative : 3 exemples commentés de bonnes recos + 2 échecs analysés (pourquoi ?).
4. Intégration : tester recommend() de bout en bout sur les 10 personas, lister et faire corriger les bugs, vérifier la conformité aux contrats, gérer requirements.txt (versions figées), README (installation, téléchargement des données, génération des embeddings dans l'ordre 01→02→03, lancement de l'app, structure du repo, rôles de chacun).
5. Rapport PDF (docs/report.md → PDF via pandoc ou export) : plan = 1. Introduction & objectifs ; 2. Dataset (P1) ; 3. Référentiel & enrichissement (P2) ; 4. Moteur sémantique & justification des modèles (P3) ; 5. Scoring & recommandation (P4) ; 6. Application & visualisations (P5) ; 7. Évaluation (moi) ; 8. Défis rencontrés et solutions ; 9. Limites & perspectives ; 10. Bonus GenAI ; Annexe : tableau de conformité exigence → où c'est implémenté. Harmoniser le style des sections fournies par chacun.
6. Slides (~15-18) : ordre de passage P6 intro → P1 → P2 → P3 → P4 → P5 démo → P6 éval/défis/conclusion ; mettre l'accent sur la partie sémantique (la prof le demande) ; préparer une vidéo de secours de la démo.
7. BONUS (seulement si tout le reste est fini) : src/llm_explain.py, explication personnalisée par LLM (Gemini API free tier via GEMINI_API_KEY dans .env, ou Ollama local) qui reçoit le profil, les scores, les livres recommandés et les critères, avec la consigne de rester fidèle aux scores et de ne rien inventer ; activable par un toggle dans l'app, avec retour automatique à l'explication template si pas de clé ou en cas d'erreur. Optionnel : petit chat « why this book? » en RAG sur les descriptions du top 10.

Livre-moi les personas, le notebook d'évaluation complet commenté, le README, le squelette complet du rapport (avec mes sections rédigées), le plan des slides et le bonus si on a le temps.
```

---

## 12. Conseils pour aller vite

- **Colab GPU** pour encoder le catalogue (environ 1 min au lieu de 10 à 15 en CPU), puis on télécharge les `.npy`.
- **Commencer petit** : tout le pipeline sur 500 livres le J1/J2, on passe à 10 000 seulement le J3.
- **Pas d'over-engineering** : ni base de données ni API REST. Streamlit, des fichiers parquet/npy/json, c'est suffisant.
- **Le bonus LLM en dernier**, et seulement si la checklist §10 est entièrement cochée.
- **Faire relire chaque notebook par un autre membre** (commentaires, Markdown, résultats affichés) : le code commenté est explicitement noté.
