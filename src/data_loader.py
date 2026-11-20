"""
Préparation des données Book Depository (personne 1).

Ce module contient :
  * le téléchargement et le chargement des CSV bruts Kaggle ;
  * les fonctions de nettoyage (texte, titres, listes d'IDs) ;
  * la classification des catégories (fiction / non-fiction narrative / exclue) ;
  * la déduction du public visé (Children / Young Adult / Adult) ;
  * le pipeline complet `build_catalogue()` qui produit books_clean.parquet ;
  * des utilitaires pour la suite du projet : load_books(), get_books_by_author(),
    apply_constraints().

Le notebook notebooks/01_data_preparation.ipynb appelle ces fonctions étape
par étape et commente chaque résultat.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

# ===========================================================================
# 1. Téléchargement et chargement des fichiers bruts
# ===========================================================================

def download_raw_dataset(dest: Path = config.DATA_RAW) -> Path:
    """Télécharge le dataset Kaggle dans `dest` (si absent) via kagglehub.

    Le dataset est public : kagglehub le télécharge sans clé API.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "dataset.csv").exists():
        return dest
    import kagglehub  # import local : uniquement nécessaire au premier lancement

    kagglehub.dataset_download(config.KAGGLE_DATASET, output_dir=str(dest))
    return dest


def load_raw_books(path: Path = config.RAW_DATASET,
                   usecols: Iterable[str] = config.RAW_USECOLS) -> pd.DataFrame:
    """Charge dataset.csv en ne gardant que les colonnes utiles.

    On lit tout en `str` pour éviter les inférences de type coûteuses ou
    erronées (ISBN convertis en float, etc.) ; les conversions numériques sont
    faites explicitement ensuite.
    """
    return pd.read_csv(path, usecols=list(usecols), dtype=str)


def load_lookup(path: Path, key: str, value: str) -> dict[int, str]:
    """Charge une table de correspondance (authors/categories/formats) en dict id -> nom."""
    table = pd.read_csv(path, dtype=str).dropna(subset=[key])
    table[value] = table[value].fillna("").str.strip()
    return dict(zip(table[key].astype(int), table[value]))


# ===========================================================================
# 2. Fonctions de nettoyage unitaires
# ===========================================================================

_ID_RE = re.compile(r"\d+")


def parse_id_list(raw: str | float) -> list[int]:
    """Convertit une chaîne du type "[214, 220, 237]" en liste d'entiers.

    On utilise une regex plutôt que `eval`/`ast.literal_eval` : c'est plus
    rapide sur un million de lignes et ça n'exécute jamais de code.
    """
    if not isinstance(raw, str):
        return []
    return [int(x) for x in _ID_RE.findall(raw)]


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:\b(?:visit|see|at)\s+)?(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_description(text: str | float) -> str:
    """Nettoyage MINIMAL d'une description.

    On garde le texte naturel (stopwords, ponctuation, majuscules) car les
    modèles contextuels type SBERT exploitent tout le contexte de la phrase.
    Le preprocessing « lourd » (lemmatisation, stopwords) est réservé aux
    baselines TF-IDF / GloVe (personne 3).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                       # &amp; -> &, &#39; -> '
    text = unicodedata.normalize("NFKC", text)       # unifie les caractères unicode
    text = _TAG_RE.sub(" ", text)                    # retire les balises HTML résiduelles
    text = _URL_RE.sub(" ", text)                    # retire les URLs (bruit pour l'embedding)
    text = text.replace(" ", " ")               # espaces insécables
    text = _MULTI_SPACE_RE.sub(" ", text)            # espaces / retours à la ligne multiples
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)   # "mot ." -> "mot."
    return text.strip()


# Phrases types des réimpressions / produits sans contenu narratif.
# Détectées dans l'EDA : des dizaines de milliers de livres partagent ces textes.
# Appliquées PHRASE PAR PHRASE (voir strip_boilerplate) : une phrase qui contient
# un de ces motifs est retirée, le reste de la description est conservé.
BOILERPLATE_PATTERNS = [
    r"facsimile reprint",
    r"scarce antiquarian book",
    r"culturally important",
    r"knowledge base of civilization",
    r"print on demand",
    r"may (?:have occasional|contain) imperfections",
    r"marks, notations",
    r"due to its age",
    r"true to the original work",
    r"protecting, preserving,? and promoting",
    r"reproduction of (?:an|the) original",
    r"we have (?:scanned|reformatted|recreated)",
    r"this is a reproduction of a book published before",
    r"(?:perfect|blank|lined|ruled) (?:notebook|journal|planner)",
    r"\bnotebook\b.*\b(?:pages|sheets)\b",
    r"click or search",
    r"public domain",
    r"historical books limited",
    r"send an email to",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)


def is_boilerplate(text: str) -> bool:
    """True si la description CONTIENT un texte générique (statistiques de l'EDA)."""
    return bool(_BOILERPLATE_RE.search(text or ""))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“A-Z0-9])")


def strip_boilerplate(text: str) -> str:
    """Retire les PHRASES génériques (réimpressions, « public domain »…) et garde le reste.

    On travaille phrase par phrase plutôt que de jeter toute la description :
    "The original book has been in the public domain in the US since 1956." est
    retirée, mais le résumé du *Magicien d'Oz* qui l'entoure est conservé.
    Un livre dont la description n'était QUE du texte générique tombe ensuite
    sous le seuil de 300 caractères et est écarté.
    """
    if not text or not _BOILERPLATE_RE.search(text):
        return text
    kept = [s for s in _SENTENCE_SPLIT_RE.split(text) if not _BOILERPLATE_RE.search(s)]
    return " ".join(kept).strip()


# --- Texte marketing ---------------------------------------------------------
# La logique (citations de presse, bandeaux « BESTSELLER », attributions) est
# dans src/marketing.py ; on la ré-exporte ici pour compatibilité.
from src.marketing import MarketingStripper, norm_words, strip_marketing  # noqa: E402,F401


# --- Éditions spéciales, coffrets, auteurs -----------------------------------
EDITION_WORDS_RE = re.compile(
    r"edition|illustrated|deluxe|anniversary|movie tie-in|tie-in|slipcase|"
    r"collector|large print|unabridged|annotated|abridged|special|volume \d|"
    r"\bvol\.? ?\d|study guide|thrift|classics? collection|orange collection",
    flags=re.IGNORECASE,
)
BOX_SET_RE = re.compile(
    r"box(?:ed)?[\s-]?set|slipcase set|\bbundle\b|complete \d+[- ]book|"
    r"\d+[- ]book (?:set|collection)|\d+[- ]volume|books? \d+\s*[-–&]\s*\d+",
    flags=re.IGNORECASE,
)


# Produits dérivés / compagnons : ce ne sont pas des expériences de lecture
# (livres pop-up, « making of » de films, guides officiels, livres d'art,
# livres de stickers ou journaux à remplir). Liste volontairement PRÉCISE :
# des motifs larges ("guide to", "the art of", "companion") supprimeraient
# The Art of War, The Hitchhiker's Guide to the Galaxy ou des romans
# « compagnons » comme Lying Out Loud: A Companion to the Duff.
COMPANION_RE = re.compile(
    r"pop-?up\b|page to screen|film wizardry|making the movie|behind the scenes of|"
    r"official (?:movie )?(?:guide|companion|handbook|colou?ring)|"
    r"(?:unofficial|unauthori[sz]ed) (?:guide|companion|handbook)|\[companion book\]|"
    r"the art of the (?:animated|film|movie|series|game)|peel-off stickers|sticker book|"
    r"wreck this journal|journal pages|colou?ring book|activity book|"
    r"visual (?:companion|guide|dictionary)|the world of the hunger games|"
    r"the sorcerer's companion|cats of the clans|the book of the exhibition|"
    r"a journey through a history of magic",
    flags=re.IGNORECASE,
)
# Auteurs « corporate » (studios, éditeurs) : signe d'un produit dérivé
CORPORATE_AUTHOR_RE = re.compile(
    r"^(?:warner bros\.?|scholastic(?: inc\.?)?|insight editions|running press|lucasfilm(?: ltd)?|"
    r"disney(?: book group)?|dreamworks|pixar|nickelodeon|cartoon network|bbc|national geographic|"
    r"lonely planet|dk|universal studios|paramount)$",
    flags=re.IGNORECASE,
)


def is_companion_product(title: str, authors: list[str]) -> bool:
    """True pour un produit dérivé (pop-up, guide officiel, making of...)."""
    return bool(COMPANION_RE.search(title)) or any(CORPORATE_AUTHOR_RE.match(a) for a in authors)


def author_key(name: str) -> str:
    """Clé d'auteur robuste : "J.K. Rowling", "J. K. Rowling", "J K Rowling" -> "jkrowling"."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", n)


def display_title(title: str) -> str:
    """Titre d'affichage débarrassé des mentions d'édition.

    "Dune : Deluxe Edition" -> "Dune" ; "The Crucible : (penguin Orange Collection)" -> "The Crucible".
    Si tout le titre serait supprimé, on garde l'original.
    """
    t = re.sub(r"[\(\[][^\)\]]*[\)\]]",
               lambda m: " " if EDITION_WORDS_RE.search(m.group(0)) else m.group(0), title)
    parts = re.split(r"\s+[:;]\s+|\s+-\s+|\s*:\s+(?=[A-Z(])", t)
    kept = [parts[0]] + [p for p in parts[1:] if not EDITION_WORDS_RE.search(p)]
    out = " : ".join(p.strip(" ,;:") for p in kept if p.strip(" ,;:"))
    return out or title


_TITLE_BRACKETS_RE = re.compile(r"[\(\[].*?[\)\]]")
_TITLE_SUFFIX_RE = re.compile(
    r":\s*(a novel|a memoir|a thriller|a mystery|the novel|stories|a true story)\s*$",
    flags=re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_title(title: str) -> str:
    """Clé de titre pour détecter les éditions multiples d'une même œuvre.

    "Pride & Prejudice (Penguin Classics)" et "Pride and Prejudice" donnent
    la même clé "pride and prejudice" ; "Cat's Eye" et "Cats Eye" -> "cats eye".
    """
    if not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = _TITLE_BRACKETS_RE.sub(" ", t)          # retire "(Penguin Classics)", "[Illustrated]"...
    t = _TITLE_SUFFIX_RE.sub("", t)            # retire ": A Novel", ": A Memoir"...
    t = t.lower().replace("&", " and ")
    # apostrophes SUPPRIMÉES : "Cat's Eye" = "Cats Eye".
    # Les tirets deviennent des espaces ("Water-Babies" -> "water babies") pour que
    # la comparaison mot à mot de la règle (b) fonctionne ; la règle (a) compare
    # les titres SANS espaces, donc "Wall-paper" = "Wallpaper" reste couvert.
    t = re.sub(r"['’`]", "", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return _MULTI_SPACE_RE.sub(" ", t).strip()


def clean_title(title: str) -> str:
    """Titre d'affichage : seulement les espaces et entités HTML."""
    if not isinstance(title, str):
        return ""
    return _MULTI_SPACE_RE.sub(" ", html.unescape(title)).strip()


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Tronque proprement un texte à la dernière fin de phrase avant max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    # si aucune fin de phrase dans la 2e moitié, on coupe au dernier espace
    if last_end > max_chars // 2:
        return cut[: last_end + 1]
    return cut[: cut.rfind(" ")].rstrip(",;:") + "…"


# ===========================================================================
# 3. Classification des catégories Book Depository
# ===========================================================================
# Le dataset contient ~2 900 catégories (nomenclature BIC des libraires).
# Pour un système de recommandation « préférences littéraires », on garde :
#   * la FICTION (romans, BD/manga, poésie, théâtre, jeunesse...) ;
#   * la NON-FICTION NARRATIVE lue par plaisir (biographies, histoire,
#     true crime, vulgarisation, philosophie, développement personnel...).
# On exclut les ouvrages pratiques, scolaires, techniques ou professionnels
# (cuisine, informatique, manuels, droit, médecine, dictionnaires...), dont la
# description ne décrit pas une « expérience de lecture ».

FICTION_RE = re.compile(
    r"fiction|novel|romance|thriller|crime|myster|horror|fantasy|sci[\s-]?fi|"
    r"science fiction|space opera|sagas?\b|graphic novel|manga|comic|cartoon|"
    r"poetry|poets|plays|playscript|drama|short stor|anthologies|classic books|"
    r"adventure|funny books|humou?r|fairy|folk tales|myths|legends|ghost|"
    r"supernatural|witches|story books|stories|espionage|erotic fiction|westerns\b|"
    r"dystopi|picture books|teenage|young adult|chick lit|superhero",
    flags=re.IGNORECASE,
)
# Catégories qui matchent FICTION_RE mais sont de la non-fiction ou de l'étude
FICTION_FALSE_POSITIVE_RE = re.compile(
    r"true crime|true stories|criminolog|crime & |studies|study|criticism|"
    r"writing|history of|theory|guides?\b|reference|business|forensic|advice|"
    r"corporate|confessions",
    flags=re.IGNORECASE,
)

NONFICTION_RE = re.compile(
    r"biograph|autobiograph|memoir|diaries, letters|true crime|true stories|"
    r"histor|travel writing|people & places|popular science|science: general|"
    r"natural history|wildlife|astronomy|evolution|philosoph|psycholog|"
    r"self-esteem|personal development|motivation|mind, body|spirituality|"
    r"religio|christian|buddhis|islam|judaism|hindu|society & culture|"
    r"popular culture|social & cultural|sociology|anthropology|gender studies|"
    r"feminism|black & asian studies|politic|international relations|"
    r"current affairs|essays|journalism|environment|conservation|"
    r"music|films?\b|cinema|television|performers|sport|mythology|"
    r"relationships|family|ethics|cosmology|nature",
    flags=re.IGNORECASE,
)

EXCLUDE_RE = re.compile(
    r"teaching|educat|textbook|exams?\b|revision|study guide|curricul|"
    r"reading schemes|language learning|elt\b|english language|foreign language|"
    r"dictionar|thesaur|grammar|reference books|encyclop|"
    r"cooker|cookery|recipe|cuisine|baking|diets|nutrition|fitness|exercise|"
    r"computing|programming|software|computer|internet|databases|networking|"
    r"accounting|finance|investment|taxation|business|management|marketing|"
    r"sales|careers|job hunting|\blaw\b|legal|medicine|medical|nursing|clinical|"
    r"surgery|pharmac|dentist|veterinar|engineering|technology|mathemat|"
    r"statistics|physics|chemistry|calendars?|stationery|notebook|colou?ring|"
    r"activity books|puzzle|quiz|crossword|maps\b|atlas|guidebook|"
    r"gardening|crafts|diy|interior|antiques|collectables|baby names|"
    r"pregnancy|child care|parenting|early learning|pre-school|"
    r"games|hobbies|vehicles|transport|military equipment|weapons|"
    r"knitting|sewing|photography: techniques|art techniques|drawing",
    flags=re.IGNORECASE,
)


def classify_category(name: str) -> str:
    """Classe un nom de catégorie : 'fiction', 'nonfiction', 'excluded' ou 'neutral'.

    L'ordre des tests compte : l'exclusion est testée en premier pour que
    "Teaching Resources & Education" ne passe jamais en non-fiction, puis la
    fiction (en écartant ses faux positifs comme "True Crime"), puis la
    non-fiction narrative.
    """
    if not name:
        return "neutral"
    if EXCLUDE_RE.search(name) and not FICTION_RE.search(name):
        return "excluded"
    if FICTION_RE.search(name) and not FICTION_FALSE_POSITIVE_RE.search(name):
        return "fiction"
    if NONFICTION_RE.search(name):
        return "nonfiction"
    if EXCLUDE_RE.search(name):
        return "excluded"
    return "neutral"


# ===========================================================================
# 4. Public visé (Children / Young Adult / Adult)
# ===========================================================================
# for-ages n'est renseigné que pour ~7 % des livres : on le complète par les
# catégories ("Children's Fiction", "Romance Books for Teenagers"...).

CHILDREN_CAT_RE = re.compile(
    r"children|for kids|kids’|kids'|picture books|baby books|early learning|"
    r"pre-school|story books|\(children's/ya\)",
    flags=re.IGNORECASE,
)
YA_CAT_RE = re.compile(r"teenage|teenagers|young adult", flags=re.IGNORECASE)
_AGES_RE = re.compile(r"(\d+)")


def parse_for_ages(raw: str | float) -> tuple[int | None, int | None]:
    """'9-12' -> (9, 12) ; '12+' -> (12, None) ; NaN -> (None, None)."""
    if not isinstance(raw, str):
        return None, None
    nums = [int(x) for x in _AGES_RE.findall(raw)]
    if not nums:
        return None, None
    return nums[0], (nums[1] if len(nums) > 1 else None)


def infer_audience(for_ages: str | float, category_names: list[str]) -> str:
    """Déduit le public visé. Priorité à for-ages (info éditeur), puis aux catégories."""
    min_age, _ = parse_for_ages(for_ages)
    if min_age is not None:
        if min_age < 9:
            return "Children"
        if min_age < 18:
            # "9-12" = middle grade (Children), "12-17" / "13+" = Young Adult
            return "Children" if min_age < 12 else "Young Adult"
        return "Adult"
    joined = " | ".join(category_names)
    if YA_CAT_RE.search(joined):
        return "Young Adult"
    if CHILDREN_CAT_RE.search(joined):
        return "Children"
    return "Adult"


# ===========================================================================
# 5. Pipeline complet
# ===========================================================================

def _log(report: list[dict], step: str, df: pd.DataFrame, comment: str) -> None:
    """Ajoute une ligne au rapport de filtrage (étape -> nombre de lignes)."""
    report.append({"step": step, "rows": len(df), "comment": comment})
    print(f"{step:<45} {len(df):>10,}  {comment}")


def build_catalogue(raw: pd.DataFrame | None = None,
                    target_size: int = config.TARGET_CATALOGUE_SIZE,
                    min_rating_count: int = config.MIN_RATING_COUNT,
                    max_per_author: int = config.MAX_BOOKS_PER_AUTHOR,
                    detect_language: bool = True,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construit le catalogue propre à partir du CSV brut.

    Retourne (catalogue, rapport_de_filtrage, statistiques_des_catégories).
    Chaque étape est loguée pour pouvoir justifier les choix dans le rapport.
    """
    report: list[dict] = []
    if raw is None:
        raw = load_raw_books()
    df = raw.copy()
    _log(report, "0. Raw dataset", df, "all rows of dataset.csv")

    # --- 5.1 Langue, titre et description présents -------------------------
    df = df[df["lang"] == config.KEEP_LANG]
    _log(report, "1. lang == 'en'", df, "English SBERT model + English UI")
    df = df[df["title"].notna() & df["description"].notna()]
    _log(report, "2. title & description not null", df, "nothing to embed otherwise")

    # --- 5.2 Format physique : livres papier uniquement --------------------
    formats = load_lookup(config.RAW_FORMATS, "format_id", "format_name")
    df["format"] = pd.to_numeric(df["format"], errors="coerce").map(formats)
    df = df[df["format"].isin(config.KEEP_FORMATS)]
    _log(report, "3. format in Paperback/Hardback", df, "drop audio CDs, calendars, toys...")

    # --- 5.3 Notes Goodreads (filtre numérique bon marché, fait AVANT les
    #         nettoyages de texte coûteux) ------------------------------------
    df["rating_count"] = pd.to_numeric(df["rating-count"], errors="coerce")
    df["rating_avg"] = pd.to_numeric(df["rating-avg"], errors="coerce")
    df = df[df["rating_count"] >= min_rating_count]
    _log(report, f"4. rating_count >= {min_rating_count}", df,
         "known books with reliable ratings")

    # --- 5.4 Qualité de la description --------------------------------------
    df = df[df["description"].str.len() >= config.MIN_DESC_CHARS]
    _log(report, f"5. raw description >= {config.MIN_DESC_CHARS} chars", df,
         "short blurbs carry too little meaning")
    df["description_raw"] = df["description"].map(clean_description)
    n_with_boilerplate = int(df["description_raw"].map(is_boilerplate).sum())
    df["description"] = df["description_raw"].map(strip_boilerplate)
    df = df[df["description"].str.len() >= config.MIN_DESC_CHARS]
    _log(report, "6. not only boilerplate", df,
         f"generic sentences removed in {n_with_boilerplate:,} texts; "
         "drop texts that were only boilerplate")
    # Texte utilisé pour le NLP : sans citations de presse ni bandeaux commerciaux.
    # Le « stripper » connaît les noms d'auteurs et les titres du dataset pour
    # savoir où s'arrête une attribution (voir src/marketing.py).
    stripper = build_marketing_stripper(raw)
    df["description"] = df["description"].map(stripper)
    df = df[df["description"].str.len() >= config.MIN_DESC_CHARS]
    _log(report, f"7. >= {config.MIN_DESC_CHARS} chars without press quotes", df,
         "enough content once marketing is removed")

    # --- 5.5 Catégories : noms + domaine -----------------------------------
    cat_names = load_lookup(config.RAW_CATEGORIES, "category_id", "category_name")
    df["category_ids"] = df["categories"].map(parse_id_list)
    df["category_names"] = df["category_ids"].map(
        lambda ids: list(dict.fromkeys(cat_names.get(i, "") for i in ids if cat_names.get(i)))
    )
    # Classification une seule fois par nom de catégorie (≈ 2 900 noms)
    domain_of = {name: classify_category(name) for name in set(cat_names.values())}
    domains = df["category_names"].map(lambda names: [domain_of[n] for n in names])
    df["n_fiction"] = domains.map(lambda d: d.count("fiction"))
    df["n_nonfiction"] = domains.map(lambda d: d.count("nonfiction"))
    df["n_excluded"] = domains.map(lambda d: d.count("excluded"))
    reader = df["n_fiction"] + df["n_nonfiction"]
    # On garde un livre s'il a au moins une catégorie « lecture loisir » et
    # au moins autant de catégories loisir que de catégories exclues.
    df = df[(reader >= 1) & (reader >= df["n_excluded"])]
    _log(report, "8. leisure-reading categories", df,
         "fiction + narrative non-fiction, no textbooks/cookery/IT...")
    df["is_fiction"] = df["n_fiction"] >= df["n_nonfiction"]
    df["domain"] = np.where(df["is_fiction"], "Fiction", "Non-fiction")

    # --- 5.6 Auteurs : orthographe canonique ---------------------------------
    author_names = load_lookup(config.RAW_AUTHORS, "author_id", "author_name")
    df["author_ids"] = df["authors"].map(parse_id_list)
    df["authors_list"] = df["author_ids"].map(
        lambda ids: [author_names[i] for i in ids if author_names.get(i)]
    )
    df = df[df["authors_list"].map(len) > 0]
    _log(report, "9. at least one named author", df, "needed for display & dedup")
    # Le dataset a un ID (et une orthographe) différent pour "J.K. Rowling",
    # "J. K. Rowling" et "J K Rowling". On unifie : clé normalisée -> orthographe
    # canonique (la plus fréquente), pour que "tous les livres de J.K. Rowling"
    # soient bien retrouvés (auteur favori, auteurs exclus du questionnaire).
    canonical = canonical_author_names(df["authors_list"])
    df["authors_list"] = df["authors_list"].map(lambda lst: canonicalize_authors(lst, canonical))
    df["author_keys"] = df["authors_list"].map(lambda lst: [author_key(a) for a in lst])
    # Auteur principal = l'auteur de la liste le plus fréquent dans le corpus.
    # Les illustrateurs / traducteurs / éditeurs scientifiques sont parfois
    # listés en premier ("Jim Kay, J.K. Rowling") mais apparaissent moins souvent.
    freq = pd.Series([k for ks in df["author_keys"] for k in ks]).value_counts()
    df["main_author_key"] = df["author_keys"].map(lambda ks: max(ks, key=lambda k: freq.get(k, 0)))

    # --- 5.7 Coffrets, lots et produits dérivés ------------------------------
    df["title"] = df["title"].map(clean_title)
    df = df[~df["title"].map(lambda t: bool(BOX_SET_RE.search(t)))]
    _log(report, "10. drop box sets / bundles", df, "a box set is not a single reading experience")
    df = df[~pd.Series([is_companion_product(t, a) for t, a in zip(df["title"], df["authors_list"])],
                       index=df.index)]
    _log(report, "11. drop companion products", df, "pop-up books, film tie-in guides, art books...")

    # --- 5.8 Dédoublonnage des éditions (union-find) ------------------------
    df = dedup_editions(df)
    _log(report, "12. merge editions of the same work", df,
         "same title+author, or same Goodreads ratings + similar title/author")
    # Descriptions identiques restantes = éditions groupées, textes génériques
    dup_counts = df["description"].map(df["description"].value_counts())
    df = df[dup_counts == 1]
    _log(report, "13. drop shared descriptions", df, "generic publisher texts")

    # --- 5.9 Diversité : plafond de livres par auteur -----------------------
    df = df.sort_values("rating_count", ascending=False)
    df["author_rank"] = df.groupby("main_author_key").cumcount()
    df = df[df["author_rank"] < max_per_author]
    _log(report, f"14. max {max_per_author} books per author", df,
         "avoid catalogue saturation by prolific authors / manga series")

    # --- 5.9 Vérification de la langue de la description -------------------
    # lang décrit le livre, mais certaines descriptions sont dans une autre
    # langue. On ne teste que les meilleurs candidats (coût ~5 ms/texte).
    # df est déjà trié par rating_count décroissant : on garde une marge de 15 %
    pool = df.head(int(target_size * 1.15)).copy()
    _log(report, f"15. top {len(pool):,} candidates by rating_count", pool,
         "well-known books: richer descriptions, recognisable in the demo")
    if detect_language:
        pool["desc_lang"] = detect_languages(pool["description"])
        pool = pool[pool["desc_lang"] == "en"]
        _log(report, "16. description really in English", pool,
             "langdetect (seed fixed)")

    # --- 5.10 Sélection finale ------------------------------------------------
    cat_df = pool.head(target_size).copy()
    _log(report, f"17. final catalogue (top {target_size:,})", cat_df,
         "target size reached")

    cat_df = finalize_columns(cat_df)
    report_df = pd.DataFrame(report)
    stats = category_stats(cat_df, domain_of)
    return cat_df, report_df, stats


_TITLE_STOPWORDS = {
    "the", "a", "an", "and", "of", "or", "to", "in", "on", "s", "for", "with",
    # mots génériques de séries / éditions : ne prouvent pas que deux titres sont la même œuvre
    "volume", "vol", "book", "books", "novel", "part", "series", "edition", "collection",
}


def build_marketing_stripper(raw: pd.DataFrame) -> MarketingStripper:
    """Construit le MarketingStripper avec les connaissances du dataset.

    * noms d'auteurs (authors.csv) : pour reconnaître "--Stephen King" ;
    * titres (dataset.csv, titre complet ET partie avant ':') : pour savoir où
      s'arrête "author of The Statistical Probability of Love at First Sight".
    """
    authors = pd.read_csv(config.RAW_AUTHORS, dtype=str)["author_name"].dropna()
    titles = raw["title"].dropna()
    titles = pd.concat([titles, titles.str.split(r"\s*:\s*").str[0]]).drop_duplicates()
    return MarketingStripper(authors, titles)


def canonical_author_names(author_lists: Iterable[list[str]]) -> dict[str, str]:
    """Clé normalisée -> orthographe canonique (la plus fréquente, puis la plus longue).

    {"jkrowling": "J.K. Rowling"} si "J.K. Rowling" est l'orthographe la plus
    fréquente parmi "J.K. Rowling" / "J. K. Rowling" / "J K Rowling".
    """
    counts = pd.Series([a for lst in author_lists for a in lst]).value_counts()
    best: dict[str, tuple[int, int, str]] = {}
    for name, n in counts.items():
        k = author_key(name)
        if not k:
            continue
        cand = (int(n), len(name), name)
        if k not in best or cand > best[k]:
            best[k] = cand
    return {k: v[2] for k, v in best.items()}


def canonicalize_authors(authors: list[str], canonical: dict[str, str]) -> list[str]:
    """Remplace chaque auteur par son orthographe canonique, sans doublons, ordre conservé."""
    out: list[str] = []
    for a in authors:
        name = canonical.get(author_key(a), a)
        if name not in out:
            out.append(name)
    return out


def _title_tokens(key: str) -> set[str]:
    """Mots significatifs d'un titre normalisé (sans mots vides ni numéros)."""
    return {w for w in key.split() if w not in _TITLE_STOPWORDS and not w.isdigit()}


def dedup_editions(df: pd.DataFrame, rel_tol: float = 0.02, avg_tol: float = 0.05,
                   min_count_fuzzy: int = 1000, title_jaccard: float = 0.5,
                   return_groups: bool = False) -> pd.DataFrame:
    """Fusionne les éditions d'une même œuvre et garde un seul représentant.

    Deux lignes sont considérées comme la même œuvre si :
      (a) même titre normalisé (comparé sans espaces) ET au moins un auteur en commun
          (clés normalisées) ;
      (b) OU notes Goodreads quasi identiques (nombre de notes à ±2 %, moyenne à
          ±0.05, pour les œuvres ≥ 1 000 notes : Goodreads agrège les notes par
          œuvre, et le dataset a été scrapé à des dates différentes) ET titres
          proches (Jaccard des mots ≥ 0.5 ou un titre inclus dans l'autre).
          NB : une 1re version acceptait aussi « même auteur » sans titre proche ;
          l'inspection a montré des fusions abusives (romans différents d'un même
          auteur prolifique), d'où cette règle plus stricte.
          Ex. "Harry Potter and the Philosopher's Stone - Gryffindor Edition"
          et "Harry Potter and the Sorcerer's Stone".
    Les liens sont regroupés par union-find (transitivité), puis on garde dans
    chaque groupe l'édition « standard » (sans mot d'édition spéciale) la plus notée.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # clé de titre calculée sur le titre SANS mentions d'édition ("Dune : Deluxe Edition" -> "dune")
    title_keys = df["title"].map(lambda t: normalize_title(display_title(t))).tolist()
    author_sets = [set(k) for k in df["author_keys"]]  # utilisé par les règles (a) et (b)

    # (a) même titre normalisé + un auteur commun. Le titre est comparé SANS
    #     espaces : "Daddy-Long-Legs" = "Daddy Long Legs", "The Wasteland" = "The Waste Land"
    first_seen: dict[tuple[str, str], int] = {}
    for i, (tk, keys) in enumerate(zip(title_keys, author_sets)):
        tk = tk.replace(" ", "")
        for ak in keys:
            j = first_seen.setdefault((tk, ak), i)
            if j != i:
                union(i, j)

    # (b) mêmes notes Goodreads (fenêtre glissante sur les lignes triées par nb de notes)
    counts = df["rating_count"].to_numpy()
    avgs = df["rating_avg"].fillna(-1).to_numpy()
    tokens = [_title_tokens(t) for t in title_keys]
    order = [i for i in np.argsort(counts) if counts[i] >= min_count_fuzzy]
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if counts[j] > counts[i] * (1 + rel_tol):
                break
            if abs(avgs[i] - avgs[j]) > avg_tol:
                continue
            ti, tj = tokens[i], tokens[j]
            inter = len(ti & tj)
            jac = inter / max(1, len(ti | tj))
            # un titre inclus dans l'autre : le plus court doit avoir ≥ 3 mots
            # significatifs, ou 2 mots ET un auteur en commun. Sinon "Out" serait
            # inclus dans "Knocked Out by my Nunga-Nungas", ou "Civil War" (Lucain)
            # dans "Spider-Man: Miles Morales: Civil War II" ; mais "Little Women"
            # reste fusionné avec "Little Women: Or Meg, Jo, Beth and Amy".
            shortest = min(len(ti), len(tj))
            same_author = bool(author_sets[i] & author_sets[j])
            overlap = inter / shortest if shortest >= 3 or (shortest == 2 and same_author) else 0.0
            # Le titre proche est OBLIGATOIRE : un auteur commun ne suffit pas
            # (deux romans de Lee Child peuvent avoir ±2 % de notes d'écart).
            if jac >= title_jaccard or overlap >= 0.8:
                union(i, j)

    df["work_id"] = [find(i) for i in range(n)]
    df["is_special_edition"] = df["title"].map(lambda t: bool(EDITION_WORDS_RE.search(t)))
    if "main_author_key" in df.columns:
        df = _harmonize_main_author(df)
    df["desc_len"] = df["description"].str.len()
    # Représentant : édition standard d'abord, puis la plus notée, puis la plus longue description
    df = df.sort_values(["is_special_edition", "rating_count", "desc_len"],
                        ascending=[True, False, False])
    df["n_editions"] = df.groupby("work_id")["work_id"].transform("size")
    if return_groups:  # toutes les lignes avec leur work_id (pour inspecter les fusions)
        return df
    return df.drop_duplicates("work_id", keep="first")


def _harmonize_main_author(df: pd.DataFrame) -> pd.DataFrame:
    """Auteur principal d'une œuvre = l'auteur principal le plus fréquent parmi ses éditions.

    Corrige les fautes de saisie d'une édition isolée : l'édition la plus notée de
    Pride and Prejudice est créditée à "Jane Austin", mais la grande majorité de
    ses 31 éditions le sont à "Jane Austen".
    """
    mode = df.groupby("work_id")["main_author_key"].agg(lambda s: s.value_counts().index[0])
    name_of: dict[str, str] = {}
    for lst, keys in zip(df["authors_list"], df["author_keys"]):
        for a, k in zip(lst, keys):
            name_of.setdefault(k, a)
    new_key = df["work_id"].map(mode)
    wrong = df["main_author_key"] != new_key
    for idx in df.index[wrong]:
        old_k, good_k = df.at[idx, "main_author_key"], new_key[idx]
        lst, keys = list(df.at[idx, "authors_list"]), list(df.at[idx, "author_keys"])
        if good_k in keys:            # le bon auteur est déjà listé (ex. illustrateur en tête)
            pass
        elif old_k in keys:           # faute de saisie : on remplace l'auteur fautif
            i = keys.index(old_k)
            lst[i], keys[i] = name_of[good_k], good_k
        df.at[idx, "authors_list"], df.at[idx, "author_keys"] = lst, keys
        df.at[idx, "main_author_key"] = good_k
    return df


def detect_languages(texts: pd.Series) -> list[str]:
    """Détecte la langue de chaque texte avec langdetect (graine fixée = déterministe)."""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = config.RANDOM_SEED
    out = []
    for t in texts:
        try:
            out.append(detect(t[:1000]))
        except LangDetectException:
            out.append("unknown")
    return out


# Ordre et noms des colonnes du contrat (PLAN_PROJET.md §4.1)
CLEAN_COLUMNS = [
    "book_id", "title", "authors", "authors_str", "main_author", "categories",
    "description", "description_raw", "pub_year", "rating_avg", "rating_count",
    "bestsellers_rank", "format", "for_ages", "audience", "domain", "is_fiction",
    "image_url", "url", "isbn13", "n_editions", "popularity", "n_words", "embed_text",
]


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les colonnes dérivées et renomme selon le contrat d'équipe."""
    out = pd.DataFrame(index=df.index)
    out["book_id"] = df["id"].astype(str)
    out["title"] = df["title"].map(display_title)      # sans "Deluxe Edition", "(Penguin Classics)"...
    out["authors"] = df["authors_list"]
    out["authors_str"] = df["authors_list"].map(", ".join)
    out["main_author"] = [
        lst[keys.index(k)] for lst, keys, k in zip(df["authors_list"], df["author_keys"], df["main_author_key"])
    ]
    out["categories"] = df["category_names"]
    out["description"] = df["description"]              # nettoyée, sans citations de presse
    out["description_raw"] = df["description_raw"]      # nettoyage minimal (traçabilité)
    year = pd.to_datetime(df["publication-date"], errors="coerce").dt.year
    # dates aberrantes (ex. 2099 = date « à paraître » factice) -> manquantes
    out["pub_year"] = year.where(year.between(1450, 2026)).astype("Int64")
    out["rating_avg"] = df["rating_avg"].astype(float)
    out["rating_count"] = df["rating_count"].astype(int)
    out["bestsellers_rank"] = pd.to_numeric(df["bestsellers-rank"], errors="coerce").astype("Int64")
    out["format"] = df["format"]
    out["for_ages"] = df["for-ages"]
    out["audience"] = [infer_audience(a, c) for a, c in zip(df["for-ages"], df["category_names"])]
    out["domain"] = df["domain"]
    out["is_fiction"] = df["is_fiction"].astype(bool)
    out["image_url"] = df["image-url"]
    out["url"] = df["url"]
    out["isbn13"] = df["isbn13"]
    out["n_editions"] = df["n_editions"].astype(int)    # nb d'éditions fusionnées
    # Popularité ∈ [0,1] : note moyenne × log(nb de notes) normalisé.
    # Le log évite qu'Harry Potter (7 M de notes) écrase tout le reste.
    log_count = np.log1p(out["rating_count"])
    out["popularity"] = (out["rating_avg"].fillna(0) / 5.0) * (log_count / log_count.max())
    out["n_words"] = out["description"].str.split().map(len)
    # Texte encodé par SBERT : le titre apporte du contexte (« Dracula »...)
    out["embed_text"] = [
        truncate_at_sentence(f"{t}. {d}", config.EMBED_TEXT_MAX_CHARS)
        for t, d in zip(out["title"], out["description"])
    ]
    return out[CLEAN_COLUMNS].reset_index(drop=True)


def category_stats(catalogue: pd.DataFrame, domain_of: dict[str, str]) -> pd.DataFrame:
    """Fréquence de chaque catégorie dans le catalogue final (utile à la personne 2)."""
    exploded = catalogue[["categories", "is_fiction"]].explode("categories").dropna()
    stats = (exploded.groupby("categories")
             .agg(n_books=("is_fiction", "size"), share_fiction=("is_fiction", "mean"))
             .sort_values("n_books", ascending=False)
             .reset_index().rename(columns={"categories": "category"}))
    stats["domain_rule"] = stats["category"].map(domain_of)
    stats["share_fiction"] = stats["share_fiction"].round(3)
    return stats


# Colonnes ajoutées plus tard par la personne 2 (vides dans l'échantillon)
ENRICHMENT_COLUMNS = ["genres", "themes", "styles", "atmospheres", "keywords"]


def make_sample(catalogue: pd.DataFrame, n: int = config.SAMPLE_SIZE,
                seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Échantillon stratifié (domaine × public) avec le schéma complet du contrat.

    Les colonnes d'enrichissement de la personne 2 sont présentes mais vides,
    pour que tout le monde code dès maintenant contre le schéma final.
    `label_scores` est stocké en chaîne JSON (un dict vide n'est pas
    sérialisable proprement en parquet).
    """
    # Même fraction tirée dans chaque strate (domaine × public) -> proportions conservées.
    # (GroupBy.sample garde toutes les colonnes, contrairement à GroupBy.apply en pandas 3)
    frac = min(1.0, (n + 20) / len(catalogue))
    sample = catalogue.groupby(["domain", "audience"]).sample(frac=frac, random_state=seed)
    sample = sample.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    for col in ENRICHMENT_COLUMNS:
        sample[col] = [[] for _ in range(len(sample))]
    sample["label_scores"] = "{}"
    return sample


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")  # zstd : ≈ 30 % plus compact que snappy
    return path


# ===========================================================================
# 6. Utilitaires pour le reste de l'équipe
# ===========================================================================

def load_books(path: Path | None = None) -> pd.DataFrame:
    """Charge le meilleur catalogue disponible : enrichi > propre > échantillon.

    Les colonnes liste (numpy arrays après lecture parquet) sont reconverties
    en listes Python, et label_scores (JSON) en dict.
    """
    if path is None:
        for candidate in (config.BOOKS_ENRICHED, config.BOOKS_CLEAN, config.BOOKS_SAMPLE):
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("Aucun catalogue trouvé : lancer scripts/build_dataset.py")
    df = pd.read_parquet(path)
    for col in ["authors", "categories", *ENRICHMENT_COLUMNS]:
        if col in df.columns:
            df[col] = df[col].map(lambda v: list(v) if v is not None else [])
    if "label_scores" in df.columns:
        df["label_scores"] = df["label_scores"].map(
            lambda v: json.loads(v) if isinstance(v, str) else (v or {})
        )
    return df


def _norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return _MULTI_SPACE_RE.sub(" ", re.sub(r"[^a-z ]", " ", n)).strip()


def find_author(df: pd.DataFrame, name: str, min_score: int = 85) -> str | None:
    """Retrouve le nom (orthographe du catalogue) d'un auteur à partir d'une saisie libre.

    1. correspondance EXACTE sur la clé normalisée : "jk rowling", "J. K. Rowling",
       "J.K.Rowling" -> "J.K. Rowling" ;
    2. sinon matching flou (rapidfuzz) pour tolérer les fautes : "stephen kng" -> "Stephen King".
    Retourne None si rien d'assez proche.
    """
    if not name or not name.strip():
        return None
    all_authors = sorted({a for lst in df["authors"] for a in lst})
    by_key = {author_key(a): a for a in all_authors}
    if author_key(name) in by_key:
        return by_key[author_key(name)]
    from rapidfuzz import fuzz, process, utils

    match = process.extractOne(_norm_name(name), [_norm_name(a) for a in all_authors],
                               scorer=fuzz.WRatio, processor=utils.default_process)
    if match is None or match[1] < min_score:
        return None
    return all_authors[match[2]]


def _has_author_key(df: pd.DataFrame, keys: set[str]) -> pd.Series:
    """Masque : le livre a au moins un auteur dont la clé normalisée est dans `keys`."""
    return df["authors"].map(lambda lst: any(author_key(a) in keys for a in lst))


def get_books_by_author(df: pd.DataFrame, name: str, min_score: int = 85) -> pd.DataFrame:
    """Tous les livres du catalogue d'un auteur (saisie libre, matching flou).

    La comparaison se fait sur la clé normalisée : toutes les variantes
    d'orthographe d'un même auteur sont retrouvées.
    """
    author = find_author(df, name, min_score)
    if author is None:
        return df.iloc[0:0]
    return df[_has_author_key(df, {author_key(author)})]


def apply_constraints(df: pd.DataFrame,
                      audience: str | Iterable[str] | None = None,
                      formats: Iterable[str] | None = None,
                      period: str | None = None,
                      excluded_authors: Iterable[str] | None = None,
                      fiction_only: bool | None = None) -> pd.Series:
    """Masque booléen des livres qui respectent les contraintes « dures » de l'utilisateur.

    * audience : "Children" / "Young Adult" / "Adult" (ou une liste) ; None = tous
    * formats : sous-ensemble de ("Paperback", "Hardback") ; vide/None = tous
    * period : clé de config.PUBLICATION_PERIODS (année de PUBLICATION) ; None = toutes
    * excluded_authors : noms (saisie libre acceptée, matching flou)
    * fiction_only : True = fiction, False = non-fiction, None = les deux
    """
    mask = pd.Series(True, index=df.index)
    if audience:
        aud = [audience] if isinstance(audience, str) else list(audience)
        if "Any" not in aud:
            mask &= df["audience"].isin(aud)
    if formats:
        mask &= df["format"].isin(list(formats))
    if period and period in config.PUBLICATION_PERIODS:
        lo, hi = config.PUBLICATION_PERIODS[period]
        mask &= df["pub_year"].between(lo, hi).fillna(False).astype(bool)
    if excluded_authors:
        names = {find_author(df, a) for a in excluded_authors} - {None}
        if names:
            mask &= ~_has_author_key(df, {author_key(n) for n in names})
    if fiction_only is not None:
        mask &= df["is_fiction"] == fiction_only
    return mask
