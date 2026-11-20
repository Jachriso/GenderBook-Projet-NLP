# 2. Dataset : origine, structure et préparation

*Section rédigée par la personne 1 (Data Engineering). Code : `src/data_loader.py`, `scripts/build_dataset.py`, `notebooks/01_data_preparation.ipynb`.*

## 2.1 Origine

Le catalogue de référence provient du dataset public **Book Depository Dataset** publié sur Kaggle (`sp1thas/book-depository-dataset`). Il a été obtenu par scraping du site du libraire en ligne britannique Book Depository.
Il contient **1 109 383 lignes**, soit un fichier `dataset.csv` de 1,1 Go, et quatre tables de correspondance (`authors.csv`, `categories.csv`, `formats.csv`, `places.csv`).
Chaque ligne est une **édition** de livre décrite par un titre, des auteurs et des catégories (stockés sous forme de listes d'identifiants), une **description éditoriale** en langage naturel, la langue, le format, l'âge conseillé, la date de publication et les **notes Goodreads** (moyenne et nombre de notes).
Le téléchargement est automatisé avec `kagglehub`, sans clé API. Les données brutes ne sont pas versionnées.

## 2.2 Pourquoi ce dataset convient à une recommandation sémantique

Le sujet impose une analyse **contextuelle** des préférences du lecteur et non un simple appariement de mots-clés. Il faut donc un texte riche décrivant le contenu de chaque livre. La colonne `description` est renseignée pour 93 % des livres, avec une médiane de 441 caractères. Les **catégories** (≈ 2 900 libellés de la nomenclature des libraires) fournissent des blocs thématiques, et les **notes Goodreads** un signal de qualité et de popularité. En revanche, le dataset ne contient ni thèmes, ni style narratif, ni mots-clés, contrairement à la structure suggérée par le sujet (`BookID | Genre | Title | Author | Synopsis | Themes | Style | Keywords`). Ces attributs sont **dérivés par NLP** à l'étape suivante (section 3).

## 2.3 Constats de l'analyse exploratoire

- **Langues** : 89 % des livres sont en anglais.
- **Formats** : environ 105 000 CD audio, ainsi que des calendriers, cartes et jeux.
- **Descriptions génériques** : 12 485 textes sont répétés au moins 3 fois et couvrent 141 070 livres. Il s'agit surtout de réimpressions automatiques du domaine public (« *This book is a facsimile reprint…* », « *Print on Demand* ») et de carnets. Ces textes ne décrivent aucun contenu et produiraient des embeddings identiques.
- **Marketing** : environ 7 % des descriptions commencent par des citations de presse (*"A chilling thriller" --Booklist*) ou des bandeaux (*#1 NEW YORK TIMES BESTSELLER*). Ces phrases décrivent le succès d'un livre, pas son contenu, et rapprocheraient artificiellement les best-sellers entre eux dans l'espace vectoriel.
- **Éditions multiples** : une même œuvre apparaît sous des dizaines d'éditions, jusqu'à 101 pour *Romeo and Juliet*. Les identifiants d'auteur varient (*J.K. Rowling* / *J. K. Rowling* / *J K Rowling*), de même que les titres (*Philosopher's Stone* / *Sorcerer's Stone*, *Dune : Deluxe Edition*).
- **Public visé** : `for-ages` manque pour 93 % des livres.

## 2.4 Pipeline de préparation

| Étape | Livres restants | Justification |
|---|---:|---|
| Dataset brut | 1 109 383 | |
| Langue anglaise | 986 575 | modèle SBERT anglais, interface en anglais |
| Titre et description présents | 952 244 | rien à encoder sinon |
| Format Paperback / Hardback | 832 881 | retire CD audio, calendriers, jouets |
| Sans description générique | 495 766 | 12 motifs regex (facsimile, print on demand…) |
| Description nettoyée ≥ 300 caractères | 494 725 | assez de contenu pour un embedding pertinent |
| ≥ 50 notes Goodreads | 134 946 | notes fiables, métadonnées riches |
| Catégories « lecture loisir » | 110 732 | fiction + non-fiction narrative |
| Sans coffrets / lots | 110 461 | un coffret n'est pas *une* expérience de lecture |
| Fusion des éditions | 96 040 | une œuvre = une ligne |
| Sans descriptions partagées | 93 991 | textes génériques d'éditeur restants |
| ≤ 8 livres par auteur | 80 915 | diversité du catalogue |
| Description réellement en anglais (langdetect, top candidats) | 13 711 | `lang` décrit le livre, pas toujours sa description |
| **Catalogue final** | **12 000** | les œuvres les plus notées |

**Nettoyage du texte.** Le nettoyage est volontairement **minimal** : décodage des entités HTML, suppression des balises et URLs, normalisation Unicode (NFKC) et des espaces. On ne met pas en minuscules et on ne retire ni les stopwords ni ne lemmatise, car les modèles contextuels exploitent la phrase complète (négations, ponctuation, mots grammaticaux). Ce preprocessing classique est réservé aux baselines TF-IDF et GloVe. Les citations de presse et bandeaux commerciaux sont retirés par expressions régulières. Le texte avec nettoyage minimal seulement est conservé dans `description_raw` pour la traçabilité.

**Catégories.** Chaque libellé est classé par des règles regex transparentes en *fiction*, *non-fiction narrative* (biographies, histoire, true crime, vulgarisation, philosophie…), *exclu* (cuisine, informatique, manuels scolaires, droit, médecine, dictionnaires…) ou *neutre*. Un livre est gardé s'il a au moins une catégorie « loisir » et au moins autant de catégories loisir que de catégories exclues. Le domaine (*Fiction* / *Non-fiction*) correspond à la classe majoritaire.

**Fusion des éditions.** Un algorithme **union-find** relie deux lignes si :
- (a) elles ont le même titre normalisé (sans mentions d'édition) et au moins un auteur en commun, selon une clé normalisée (`jkrowling`) ;
- (b) ou leurs notes Goodreads sont quasi identiques (Goodreads agrège les notes par œuvre : nombre de notes à ±2 %, moyenne à ±0,05) **et** leurs titres sont proches (Jaccard des mots significatifs ≥ 0,5, ou un titre inclus dans l'autre).

Dans chaque groupe, on conserve l'édition standard la plus notée. Par exemple, les 25 éditions anglaises de *Harry Potter* tome 1 sont regroupées en une seule œuvre.
Une première version de la règle (b) acceptait « même auteur » sans titre proche. L'inspection a montré qu'elle fusionnait à tort des romans différents d'auteurs prolifiques (Lee Child, Clive Cussler). Avec la règle stricte, les groupes douteux, c'est-à-dire ceux dont les titres n'ont aucun mot en commun, passent de 1 606 à 23 sur ≈ 69 000 groupes, et ces 23 cas sont essentiellement de vraies traductions (*Inferno* / *The Divine Comedy*).

**Public visé.** On utilise `for-ages` quand il existe (âge minimum < 12 → *Children*, 12-17 → *Young Adult*). Sinon, le public est déduit des catégories (*Teenage*, *Young Adult* → *Young Adult* ; *Children's*, *for Kids*, *Picture Books* → *Children*), et *Adult* par défaut.

**Sélection finale.** On garde les 12 000 œuvres les plus notées, avec au plus 8 livres par auteur principal. L'auteur principal est l'auteur de la liste le plus fréquent dans le corpus, pour ne pas compter les illustrateurs et traducteurs. Le sujet recommande au moins 60 livres. 12 000 livres assurent une bonne couverture de tous les profils, un calcul d'embeddings de quelques minutes sur CPU, des descriptions éditoriales riches et des livres reconnaissables, ce qui facilite l'évaluation qualitative et la démo.

## 2.5 Catalogue obtenu

- **12 000 œuvres**, 5 994 auteurs principaux distincts, 1 078 catégories Book Depository.
- **Domaine** : 9 343 *Fiction* (78 %), 2 657 *Non-fiction* (22 %).
- **Public** : 9 321 *Adult*, 1 454 *Children*, 1 225 *Young Adult*.
- **Popularité** : entre 5 189 et 7,4 millions de notes Goodreads (médiane : 14 163).
- **Descriptions** : 143 mots en moyenne. 80 % des textes `embed_text` (titre + description, tronqué proprement à ≈ 1 500 caractères) tiennent entièrement dans les 256 tokens de `all-MiniLM-L6-v2`, et 100 % dans les 384 tokens de `all-mpnet-base-v2`, mesuré avec le tokenizer du modèle.

**Colonnes produites** (`data/processed/books_clean.parquet`) : `book_id`, `title` (sans mentions d'édition), `authors`, `authors_str`, `main_author`, `categories`, `description` (texte NLP), `description_raw`, `pub_year`, `rating_avg`, `rating_count`, `bestsellers_rank`, `format`, `for_ages`, `audience`, `domain`, `is_fiction`, `image_url`, `url`, `isbn13`, `n_editions`, `popularity`, `n_words`, `embed_text`.

**Score de popularité**, utilisé dans la formule de recommandation :

$$\text{Pop}(j) = \frac{\text{rating\_avg}_j}{5}\cdot\frac{\log(1+\text{rating\_count}_j)}{\log(1+\max_k \text{rating\_count}_k)} \in [0,1]$$

Le logarithme évite qu'une poignée de best-sellers à plusieurs millions de notes écrase le reste du catalogue.

## 2.6 Limites

- **Biais de popularité et biais anglophone** : le catalogue reflète les ventes d'un libraire britannique en ligne. Ce biais est atténué par le plafond par auteur et par le réglage « best-sellers ↔ pépites » du questionnaire.
- **Marketing résiduel** : les descriptions restent des textes éditoriaux, et un peu de marketing subsiste (« *#1 bestselling classic* »).
- **Règles imparfaites** : le classement des catégories et le public déduit reposent sur des règles et font quelques erreurs à la marge.
- **`pub_year`** est l'année de publication de l'édition, pas l'époque du récit. L'époque et l'atmosphère sont donc déterminées sémantiquement à partir de la description (bloc *Atmosphere*).
