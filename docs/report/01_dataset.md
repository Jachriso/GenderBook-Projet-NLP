# 2. Dataset : origine, structure et préparation

*Section rédigée par la personne 1 (Data Engineering). Code : `src/data_loader.py`, `src/marketing.py`, `scripts/build_dataset.py`, `notebooks/01_data_preparation.ipynb`.*

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
- **Marketing** : environ 7 % des descriptions commencent par des citations de presse (*"A chilling thriller" --Booklist*) ou des bandeaux (*#1 NEW YORK TIMES BESTSELLER*), et beaucoup se terminent par une section *Praise for…*. Ces passages décrivent le succès d'un livre, pas son contenu, et rapprocheraient artificiellement les best-sellers entre eux dans l'espace vectoriel.
- **Éditions multiples** : une même œuvre apparaît sous des dizaines d'éditions, jusqu'à 101 pour *Romeo and Juliet*. Les titres varient (*Philosopher's Stone* / *Sorcerer's Stone*, *Dune : Deluxe Edition*, *Cat's Eye* / *Cats Eye*).
- **Auteurs** : un même auteur a un identifiant différent pour chaque orthographe (*J.K. Rowling* / *J. K. Rowling* / *J K Rowling*), et certaines éditions contiennent des fautes de saisie (*Jane Austin*).
- **Public visé** : `for-ages` manque pour 93 % des livres.

## 2.4 Pipeline de préparation

| Étape | Livres restants | Justification |
|---|---:|---|
| Dataset brut | 1 109 383 | |
| Langue anglaise | 986 575 | modèle SBERT anglais, interface en anglais |
| Titre et description présents | 952 244 | rien à encoder sinon |
| Format Paperback / Hardback | 832 881 | retire CD audio, calendriers, jouets |
| ≥ 50 notes Goodreads | 232 731 | notes fiables, métadonnées riches ; filtre numérique fait avant les nettoyages de texte coûteux |
| Description brute ≥ 300 caractères | 136 531 | un texte trop court porte trop peu de sens |
| Pas uniquement du texte générique | 135 877 | phrases génériques retirées dans 1 018 textes ; on écarte ceux qui n'avaient que ça |
| ≥ 300 caractères sans marketing | 135 257 | il reste assez de contenu une fois le marketing retiré |
| Catégories « lecture loisir » | 110 905 | fiction + non-fiction narrative |
| Au moins un auteur nommé | 110 883 | nécessaire à l'affichage et au dédoublonnage |
| Sans coffrets / lots | 110 632 | un coffret n'est pas *une* expérience de lecture |
| Sans produits dérivés | 110 303 | livres pop-up, « making of » de films, guides officiels |
| Fusion des éditions | 95 877 | une œuvre = une ligne |
| Sans descriptions partagées | 93 724 | textes génériques d'éditeur restants |
| ≤ 8 livres par auteur | 80 718 | diversité du catalogue |
| Top candidats par popularité | 13 799 | marge de 15 % avant le contrôle de langue |
| Description réellement en anglais (langdetect) | 13 709 | `lang` décrit le livre, pas toujours sa description |
| **Catalogue final** | **12 000** | les œuvres les plus notées |

Dans l'EDA, la majorité des lignes écartées au stade « description » l'est par le **seuil de longueur** (≈ 295 000 descriptions brutes de moins de 300 caractères), et non par la détection de texte générique (≈ 42 000). Les deux filtres sont logués séparément.

**Nettoyage du texte.** Le nettoyage est volontairement **minimal** : décodage des entités HTML, suppression des balises et URLs, normalisation Unicode (NFKC) et des espaces. On ne met pas en minuscules et on ne retire ni les stopwords ni ne lemmatise, car les modèles contextuels exploitent la phrase complète (négations, ponctuation, mots grammaticaux). Ce preprocessing classique est réservé aux baselines TF-IDF et GloVe. Le texte avec nettoyage minimal seulement est conservé dans `description_raw` pour la traçabilité.

**Texte générique, phrase par phrase.** Les motifs génériques (« facsimile reprint », « public domain », « print on demand »…) sont appliqués **phrase par phrase** : on retire la phrase concernée et on garde le reste. La phrase « *The original book has been in the public domain since 1956* » disparaît, mais le résumé du *Magicien d'Oz* qui l'entoure est conservé. Un livre dont la description n'était que du texte générique passe sous le seuil de 300 caractères et est écarté.

**Retrait du marketing (`src/marketing.py`).** La difficulté est de savoir **où s'arrête l'attribution** qui suit une citation de presse. Dans *"A premium delivery." --People Meghan doesn't know Agatha…*, « People » est un magazine mais « Meghan » est déjà l'intrigue. Une règle du type « suite de mots en majuscule » ne peut pas faire la différence. Le module s'appuie donc sur des connaissances tirées du dataset lui-même :
- une liste de **publications de presse** (People, Booklist, Kirkus…) ;
- les **631 000 noms d'auteurs** de `authors.csv`, pour reconnaître « --Stephen King » ;
- les **1,1 million de titres** du dataset, pour savoir où s'arrête « *author of The Statistical Probability of Love at First Sight* ».

Quand rien n'est reconnu, un repli prudent ne retire qu'un nom probable. Deux règles de sécurité complètent le tout :
- un bandeau n'est retiré que s'il forme un **segment autonome**, jamais s'il est suivi d'un mot en minuscule (« *The #1 New York Times bestselling debut novel that…* » est conservé) ;
- une citation sans source identifiable après le tiret est une citation **du récit** et n'est pas touchée.

*Évaluation* : une description qui commence par une minuscule signale une phrase coupée. La première version purement regex en produisait 235. La version actuelle en produit 28, contre 9 dans le texte brut. Les cas restants sont surtout des **restes d'attribution** non retirés (« *via Twitter* »), une erreur volontairement prudente.

**Catégories.** Chaque libellé est classé par des règles regex transparentes en *fiction*, *non-fiction narrative* (biographies, histoire, true crime, vulgarisation, philosophie…), *exclu* (cuisine, informatique, manuels scolaires, droit, médecine, dictionnaires…) ou *neutre*. Un livre est gardé s'il a au moins une catégorie « loisir » et au moins autant de catégories loisir que de catégories exclues. Le domaine (*Fiction* / *Non-fiction*) correspond à la classe majoritaire.

**Orthographe canonique des auteurs.** Chaque auteur est remplacé par l'orthographe la plus fréquente parmi les variantes de même clé normalisée (`jkrowling`). Les utilitaires de recherche comparent aussi ces clés. Sans cela, 119 auteurs du catalogue avaient plusieurs orthographes, et exclure J.K. Rowling ne retirait qu'un Harry Potter sur 7.

**Produits dérivés.** Une liste **précise** de motifs (livre pop-up, *Page to Screen*, guides officiels ou non officiels, livres d'art de films, livres de stickers, journaux à remplir), complétée par les auteurs « corporate » (Warner Bros, Scholastic…), retire 329 produits. Des motifs larges (« guide to », « the art of ») auraient supprimé *The Art of War* ou *The Hitchhiker's Guide to the Galaxy*.

**Fusion des éditions.** Un algorithme **union-find** relie deux lignes si :
- (a) elles ont le même titre normalisé (sans mentions d'édition, apostrophes supprimées, tirets remplacés par des espaces, comparé **sans espaces**) et au moins un auteur en commun ;
- (b) ou leurs notes Goodreads sont quasi identiques (Goodreads agrège les notes par œuvre : nombre de notes à ±2 %, moyenne à ±0,05) **et** leurs titres sont proches : Jaccard des mots significatifs ≥ 0,5, ou un titre inclus dans l'autre si le plus court a au moins 3 mots (ou 2 mots et un auteur en commun).

Dans chaque groupe, on conserve l'édition standard la plus notée. L'auteur principal est l'**auteur majoritaire parmi toutes les éditions**, ce qui corrige les fautes de saisie isolées (*Pride and Prejudice* est bien crédité à Jane Austen). Par exemple, les 25 éditions anglaises de *Harry Potter* tome 1 sont regroupées en une seule œuvre.

Les règles ont été affinées en inspectant les groupes dont les titres n'ont aucun mot en commun :
- une première version acceptait « même auteur » sans titre proche et fusionnait des romans différents de Lee Child (1 606 groupes douteux) ;
- la règle d'inclusion fusionnait ensuite *Civil War* (Lucain) avec *Spider-Man: Miles Morales: Civil War II*.

Il reste aujourd'hui 14 groupes de ce type sur ≈ 70 000, et ce sont de vraies variantes (*Blindside* / *Blind Side*, traductions). Aucun doublon titre + auteur ne subsiste dans le catalogue.

**Public visé.** On utilise `for-ages` quand il existe (âge minimum < 12 → *Children*, 12-17 → *Young Adult*). Sinon, le public est déduit des catégories (*Teenage*, *Young Adult* → *Young Adult* ; *Children's*, *for Kids*, *Picture Books* → *Children*), et *Adult* par défaut.

**Sélection finale.** On garde les 12 000 œuvres les plus notées, avec au plus 8 livres par auteur principal. L'auteur principal est l'auteur de la liste le plus fréquent dans le corpus, pour ne pas compter les illustrateurs et traducteurs. Le sujet recommande au moins 60 livres. 12 000 livres assurent une bonne couverture de tous les profils, un calcul d'embeddings de quelques minutes sur CPU, des descriptions éditoriales riches et des livres reconnaissables, ce qui facilite l'évaluation qualitative et la démo.

## 2.5 Catalogue obtenu

- **12 000 œuvres**, 5 860 auteurs principaux distincts, 1 074 catégories Book Depository.
- **Domaine** : 9 336 *Fiction* (78 %), 2 664 *Non-fiction* (22 %).
- **Public** : 9 339 *Adult*, 1 440 *Children*, 1 221 *Young Adult*.
- **Popularité** : entre 5 185 et 7,4 millions de notes Goodreads (médiane : 14 143).
- **Descriptions** : 143 mots en moyenne. Environ 80 % des textes `embed_text` (titre + description, tronqué proprement à ≈ 1 500 caractères) tiennent entièrement dans les 256 tokens de `all-MiniLM-L6-v2`, et la totalité dans les 384 tokens de `all-mpnet-base-v2`, mesuré avec le tokenizer du modèle.

**Colonnes produites** (`data/processed/books_clean.parquet`) : `book_id`, `title` (sans mentions d'édition), `authors` (orthographe canonique), `authors_str`, `main_author`, `categories`, `description` (texte NLP), `description_raw`, `pub_year`, `rating_avg`, `rating_count`, `bestsellers_rank`, `format`, `for_ages`, `audience`, `domain`, `is_fiction`, `image_url`, `url`, `isbn13`, `n_editions`, `popularity`, `n_words`, `embed_text`.

**Score de popularité**, utilisé dans la formule de recommandation :

$$\text{Pop}(j) = \frac{\text{rating\_avg}_j}{5}\cdot\frac{\log(1+\text{rating\_count}_j)}{\log(1+\max_k \text{rating\_count}_k)} \in [0,1]$$

Le logarithme évite qu'une poignée de best-sellers à plusieurs millions de notes écrase le reste du catalogue.

**Qualité vérifiée par 52 tests unitaires** (`tests/test_data_loader.py`). Ils couvrent notamment les cas signalés en revue : variantes d'auteur, citations de presse, doublons de ponctuation, produits dérivés.

## 2.6 Limites

- **Biais de popularité et biais anglophone** : le catalogue reflète les ventes d'un libraire britannique en ligne. Ce biais est atténué par le plafond par auteur et par le réglage « best-sellers ↔ pépites » du questionnaire.
- **Marketing résiduel** : il reste un peu de marketing intégré aux phrases (« *#1 bestselling classic* »), volontairement conservé quand le retirer couperait la phrase.
- **Règles imparfaites** : le classement des catégories, le public déduit et la détection des produits dérivés reposent sur des règles et font quelques erreurs à la marge.
- **`pub_year`** est l'année de publication de l'édition, pas l'époque du récit. L'époque et l'atmosphère sont donc déterminées sémantiquement à partir de la description (bloc *Atmosphere*).
