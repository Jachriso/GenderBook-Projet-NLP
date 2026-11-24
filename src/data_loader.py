"""
Book Depository data preparation (person 1).

This module contains:
  * downloading and loading the raw Kaggle CSVs;
  * cleaning functions (text, titles, ID lists);
  * category classification (fiction / narrative non-fiction / excluded);
  * inference of the target audience (Children / Young Adult / Adult);
  * the full `build_catalogue()` pipeline that produces books_clean.parquet;
  * utilities for the rest of the project: load_books(), get_books_by_author(),
    apply_constraints().

The notebook notebooks/01_data_preparation.ipynb calls these functions step
by step and comments on each result.
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
# 1. Downloading and loading the raw files
# ===========================================================================

def download_raw_dataset(dest: Path = config.DATA_RAW) -> Path:
    """Download the Kaggle dataset into `dest` (if missing) with kagglehub.

    The dataset is public: kagglehub downloads it without an API key.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "dataset.csv").exists():
        return dest
    import kagglehub  # local import: only needed on the first run

    kagglehub.dataset_download(config.KAGGLE_DATASET, output_dir=str(dest))
    return dest


def load_raw_books(path: Path = config.RAW_DATASET,
                   usecols: Iterable[str] = config.RAW_USECOLS) -> pd.DataFrame:
    """Load dataset.csv, keeping only the useful columns.

    Everything is read as `str` to avoid costly or wrong type inference
    (ISBNs converted to float, etc.); numeric conversions are done
    explicitly afterwards.
    """
    return pd.read_csv(path, usecols=list(usecols), dtype=str)


def load_lookup(path: Path, key: str, value: str) -> dict[int, str]:
    """Load a lookup table (authors/categories/formats) as an id -> name dict."""
    table = pd.read_csv(path, dtype=str).dropna(subset=[key])
    table[value] = table[value].fillna("").str.strip()
    return dict(zip(table[key].astype(int), table[value]))


# ===========================================================================
# 2. Unit cleaning functions
# ===========================================================================

_ID_RE = re.compile(r"\d+")


def parse_id_list(raw: str | float) -> list[int]:
    """Convert a string such as "[214, 220, 237]" into a list of integers.

    A regex is used rather than `eval`/`ast.literal_eval`: it is faster
    on a million rows and never executes code.
    """
    if not isinstance(raw, str):
        return []
    return [int(x) for x in _ID_RE.findall(raw)]


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:\b(?:visit|see|at)\s+)?(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_description(text: str | float) -> str:
    """MINIMAL cleaning of a description.

    The natural text is kept (stopwords, punctuation, capitals) because
    contextual models such as SBERT use the whole context of the sentence.
    "Heavy" preprocessing (lemmatisation, stopwords) is reserved for the
    TF-IDF / GloVe baselines (person 3).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                       # &amp; -> &, &#39; -> '
    text = unicodedata.normalize("NFKC", text)       # unify unicode characters
    text = _TAG_RE.sub(" ", text)                    # remove leftover HTML tags
    text = _URL_RE.sub(" ", text)                    # remove URLs (noise for the embedding)
    text = text.replace(" ", " ")               # non-breaking spaces
    text = _MULTI_SPACE_RE.sub(" ", text)            # multiple spaces / line breaks
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)   # "word ." -> "word."
    return text.strip()


# Stock sentences of reprints / products without narrative content.
# Found during the EDA: tens of thousands of books share these texts.
# Applied SENTENCE BY SENTENCE (see strip_boilerplate): a sentence containing
# one of these patterns is removed, the rest of the description is kept.
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
    """True if the description CONTAINS generic text (EDA statistics)."""
    return bool(_BOILERPLATE_RE.search(text or ""))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“A-Z0-9])")


def strip_boilerplate(text: str) -> str:
    """Remove generic SENTENCES (reprints, "public domain"...) and keep the rest.

    We work sentence by sentence rather than throwing away the whole description:
    "The original book has been in the public domain in the US since 1956." is
    removed, but the blurb of *The Wizard of Oz* around it is kept.
    A book whose description was ONLY generic text then falls below the
    300-character threshold and is discarded.
    """
    if not text or not _BOILERPLATE_RE.search(text):
        return text
    kept = [s for s in _SENTENCE_SPLIT_RE.split(text) if not _BOILERPLATE_RE.search(s)]
    return " ".join(kept).strip()


# --- Marketing text ------------------------------------------------------------
# The logic (press quotes, "BESTSELLER" banners, attributions) lives in
# src/marketing.py; it is re-exported here for compatibility.
from src.marketing import MarketingStripper, norm_words, strip_marketing  # noqa: E402,F401


# --- Special editions, box sets, authors ----------------------------------------
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


# Spin-off / companion products: these are not reading experiences
# (pop-up books, film "making of" books, official guides, art books,
# sticker books or fill-in journals). The list is deliberately PRECISE:
# broad patterns ("guide to", "the art of", "companion") would remove
# The Art of War, The Hitchhiker's Guide to the Galaxy or "companion"
# novels such as Lying Out Loud: A Companion to the Duff.
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
# "Corporate" authors (studios, publishers): sign of a spin-off product
CORPORATE_AUTHOR_RE = re.compile(
    r"^(?:warner bros\.?|scholastic(?: inc\.?)?|insight editions|running press|lucasfilm(?: ltd)?|"
    r"disney(?: book group)?|dreamworks|pixar|nickelodeon|cartoon network|bbc|national geographic|"
    r"lonely planet|dk|universal studios|paramount)$",
    flags=re.IGNORECASE,
)


def is_companion_product(title: str, authors: list[str]) -> bool:
    """True for a spin-off product (pop-up, official guide, making of...)."""
    return bool(COMPANION_RE.search(title)) or any(CORPORATE_AUTHOR_RE.match(a) for a in authors)


def author_key(name: str) -> str:
    """Robust author key: "J.K. Rowling", "J. K. Rowling", "J K Rowling" -> "jkrowling"."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", n)


def display_title(title: str) -> str:
    """Display title stripped of edition mentions.

    "Dune : Deluxe Edition" -> "Dune"; "The Crucible : (penguin Orange Collection)" -> "The Crucible".
    If the whole title would be removed, the original is kept.
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
    """Title key used to detect multiple editions of the same work.

    "Pride & Prejudice (Penguin Classics)" and "Pride and Prejudice" give
    the same key "pride and prejudice"; "Cat's Eye" and "Cats Eye" -> "cats eye".
    """
    if not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = _TITLE_BRACKETS_RE.sub(" ", t)          # remove "(Penguin Classics)", "[Illustrated]"...
    t = _TITLE_SUFFIX_RE.sub("", t)            # remove ": A Novel", ": A Memoir"...
    t = t.lower().replace("&", " and ")
    # apostrophes are DELETED: "Cat's Eye" = "Cats Eye".
    # Hyphens become spaces ("Water-Babies" -> "water babies") so that the
    # word-by-word comparison of rule (b) works; rule (a) compares titles
    # WITHOUT spaces, so "Wall-paper" = "Wallpaper" is still covered.
    t = re.sub(r"['’`]", "", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return _MULTI_SPACE_RE.sub(" ", t).strip()


def clean_title(title: str) -> str:
    """Display title: only spaces and HTML entities are fixed."""
    if not isinstance(title, str):
        return ""
    return _MULTI_SPACE_RE.sub(" ", html.unescape(title)).strip()


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cleanly truncate a text at the last sentence end before max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    # if there is no sentence end in the second half, cut at the last space
    if last_end > max_chars // 2:
        return cut[: last_end + 1]
    return cut[: cut.rfind(" ")].rstrip(",;:") + "…"


# ===========================================================================
# 3. Classification of the Book Depository categories
# ===========================================================================
# The dataset has ~2,900 categories (BIC booksellers' nomenclature).
# For a "literary preferences" recommender, we keep:
#   * FICTION (novels, comics/manga, poetry, drama, children's books...);
#   * NARRATIVE NON-FICTION read for pleasure (biographies, history,
#     true crime, popular science, philosophy, self-help...).
# We exclude practical, school, technical or professional books
# (cookery, computing, textbooks, law, medicine, dictionaries...), whose
# description does not describe a "reading experience".

FICTION_RE = re.compile(
    r"fiction|novel|romance|thriller|crime|myster|horror|fantasy|sci[\s-]?fi|"
    r"science fiction|space opera|sagas?\b|graphic novel|manga|comic|cartoon|"
    r"poetry|poets|plays|playscript|drama|short stor|anthologies|classic books|"
    r"adventure|funny books|humou?r|fairy|folk tales|myths|legends|ghost|"
    r"supernatural|witches|story books|stories|espionage|erotic fiction|westerns\b|"
    r"dystopi|picture books|teenage|young adult|chick lit|superhero",
    flags=re.IGNORECASE,
)
# Categories that match FICTION_RE but are non-fiction or study material
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
    """Classify a category name: 'fiction', 'nonfiction', 'excluded' or 'neutral'.

    The order of the tests matters: exclusion is tested first so that
    "Teaching Resources & Education" never becomes non-fiction, then
    fiction (ruling out its false positives such as "True Crime"), then
    narrative non-fiction.
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
# 4. Target audience (Children / Young Adult / Adult)
# ===========================================================================
# for-ages is only filled in for ~7% of the books: it is complemented by the
# categories ("Children's Fiction", "Romance Books for Teenagers"...).

CHILDREN_CAT_RE = re.compile(
    r"children|for kids|kids’|kids'|picture books|baby books|early learning|"
    r"pre-school|story books|\(children's/ya\)",
    flags=re.IGNORECASE,
)
YA_CAT_RE = re.compile(r"teenage|teenagers|young adult", flags=re.IGNORECASE)
_AGES_RE = re.compile(r"(\d+)")


def parse_for_ages(raw: str | float) -> tuple[int | None, int | None]:
    """'9-12' -> (9, 12); '12+' -> (12, None); NaN -> (None, None)."""
    if not isinstance(raw, str):
        return None, None
    nums = [int(x) for x in _AGES_RE.findall(raw)]
    if not nums:
        return None, None
    return nums[0], (nums[1] if len(nums) > 1 else None)


def infer_audience(for_ages: str | float, category_names: list[str]) -> str:
    """Infer the target audience. for-ages (publisher info) first, then the categories."""
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
# 5. Full pipeline
# ===========================================================================

def _log(report: list[dict], step: str, df: pd.DataFrame, comment: str) -> None:
    """Add a row to the filtering report (step -> number of rows)."""
    report.append({"step": step, "rows": len(df), "comment": comment})
    print(f"{step:<45} {len(df):>10,}  {comment}")


def build_catalogue(raw: pd.DataFrame | None = None,
                    target_size: int = config.TARGET_CATALOGUE_SIZE,
                    min_rating_count: int = config.MIN_RATING_COUNT,
                    max_per_author: int = config.MAX_BOOKS_PER_AUTHOR,
                    detect_language: bool = True,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the clean catalogue from the raw CSV.

    Returns (catalogue, filtering_report, category_statistics).
    Every step is logged so that the choices can be justified in the report.
    """
    report: list[dict] = []
    if raw is None:
        raw = load_raw_books()
    df = raw.copy()
    _log(report, "0. Raw dataset", df, "all rows of dataset.csv")

    # --- 5.1 Language, title and description present -----------------------
    df = df[df["lang"] == config.KEEP_LANG]
    _log(report, "1. lang == 'en'", df, "English SBERT model + English UI")
    df = df[df["title"].notna() & df["description"].notna()]
    _log(report, "2. title & description not null", df, "nothing to embed otherwise")

    # --- 5.2 Physical format: paper books only ------------------------------
    formats = load_lookup(config.RAW_FORMATS, "format_id", "format_name")
    df["format"] = pd.to_numeric(df["format"], errors="coerce").map(formats)
    df = df[df["format"].isin(config.KEEP_FORMATS)]
    _log(report, "3. format in Paperback/Hardback", df, "drop audio CDs, calendars, toys...")

    # --- 5.3 Goodreads ratings (cheap numeric filter, done BEFORE the
    #         costly text cleaning) ---------------------------------------------
    df["rating_count"] = pd.to_numeric(df["rating-count"], errors="coerce")
    df["rating_avg"] = pd.to_numeric(df["rating-avg"], errors="coerce")
    df = df[df["rating_count"] >= min_rating_count]
    _log(report, f"4. rating_count >= {min_rating_count}", df,
         "known books with reliable ratings")

    # --- 5.4 Description quality ----------------------------------------------
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
    # Text used for NLP: without press quotes or commercial banners.
    # The "stripper" knows the author names and titles of the dataset so it
    # can tell where an attribution ends (see src/marketing.py).
    stripper = build_marketing_stripper(raw)
    df["description"] = df["description"].map(stripper)
    df = df[df["description"].str.len() >= config.MIN_DESC_CHARS]
    _log(report, f"7. >= {config.MIN_DESC_CHARS} chars without press quotes", df,
         "enough content once marketing is removed")

    # --- 5.5 Categories: names + domain ---------------------------------------
    cat_names = load_lookup(config.RAW_CATEGORIES, "category_id", "category_name")
    df["category_ids"] = df["categories"].map(parse_id_list)
    df["category_names"] = df["category_ids"].map(
        lambda ids: list(dict.fromkeys(cat_names.get(i, "") for i in ids if cat_names.get(i)))
    )
    # Classify each category name only once (≈ 2,900 names)
    domain_of = {name: classify_category(name) for name in set(cat_names.values())}
    domains = df["category_names"].map(lambda names: [domain_of[n] for n in names])
    df["n_fiction"] = domains.map(lambda d: d.count("fiction"))
    df["n_nonfiction"] = domains.map(lambda d: d.count("nonfiction"))
    df["n_excluded"] = domains.map(lambda d: d.count("excluded"))
    reader = df["n_fiction"] + df["n_nonfiction"]
    # A book is kept if it has at least one "leisure reading" category and
    # at least as many leisure categories as excluded ones.
    df = df[(reader >= 1) & (reader >= df["n_excluded"])]
    _log(report, "8. leisure-reading categories", df,
         "fiction + narrative non-fiction, no textbooks/cookery/IT...")
    df["is_fiction"] = df["n_fiction"] >= df["n_nonfiction"]
    df["domain"] = np.where(df["is_fiction"], "Fiction", "Non-fiction")

    # --- 5.6 Authors: canonical spelling ----------------------------------------
    author_names = load_lookup(config.RAW_AUTHORS, "author_id", "author_name")
    df["author_ids"] = df["authors"].map(parse_id_list)
    df["authors_list"] = df["author_ids"].map(
        lambda ids: [author_names[i] for i in ids if author_names.get(i)]
    )
    df = df[df["authors_list"].map(len) > 0]
    _log(report, "9. at least one named author", df, "needed for display & dedup")
    # The dataset has a different ID (and spelling) for "J.K. Rowling",
    # "J. K. Rowling" and "J K Rowling". We unify them: normalised key -> canonical
    # spelling (the most frequent one), so that "all the books by J.K. Rowling"
    # are found (favourite author, excluded authors of the questionnaire).
    canonical = canonical_author_names(df["authors_list"])
    df["authors_list"] = df["authors_list"].map(lambda lst: canonicalize_authors(lst, canonical))
    df["author_keys"] = df["authors_list"].map(lambda lst: [author_key(a) for a in lst])
    # Main author = the author of the list who is most frequent in the corpus.
    # Illustrators / translators / scholarly editors are sometimes listed
    # first ("Jim Kay, J.K. Rowling") but appear less often.
    freq = pd.Series([k for ks in df["author_keys"] for k in ks]).value_counts()
    df["main_author_key"] = df["author_keys"].map(lambda ks: max(ks, key=lambda k: freq.get(k, 0)))

    # --- 5.7 Box sets, bundles and spin-off products ----------------------------
    df["title"] = df["title"].map(clean_title)
    df = df[~df["title"].map(lambda t: bool(BOX_SET_RE.search(t)))]
    _log(report, "10. drop box sets / bundles", df, "a box set is not a single reading experience")
    df = df[~pd.Series([is_companion_product(t, a) for t, a in zip(df["title"], df["authors_list"])],
                       index=df.index)]
    _log(report, "11. drop companion products", df, "pop-up books, film tie-in guides, art books...")

    # --- 5.8 Deduplication of editions (union-find) -----------------------------
    df = dedup_editions(df)
    _log(report, "12. merge editions of the same work", df,
         "same title+author, or same Goodreads ratings + similar title/author")
    # Remaining identical descriptions = grouped editions, generic texts
    dup_counts = df["description"].map(df["description"].value_counts())
    df = df[dup_counts == 1]
    _log(report, "13. drop shared descriptions", df, "generic publisher texts")

    # --- 5.9 Diversity: cap on books per author ---------------------------------
    df = df.sort_values("rating_count", ascending=False)
    df["author_rank"] = df.groupby("main_author_key").cumcount()
    df = df[df["author_rank"] < max_per_author]
    _log(report, f"14. max {max_per_author} books per author", df,
         "avoid catalogue saturation by prolific authors / manga series")

    # --- 5.9 Checking the language of the description ---------------------------
    # lang describes the book, but some descriptions are in another
    # language. Only the best candidates are tested (cost ~5 ms/text).
    # df is already sorted by decreasing rating_count: we keep a 15% margin
    pool = df.head(int(target_size * 1.15)).copy()
    _log(report, f"15. top {len(pool):,} candidates by rating_count", pool,
         "well-known books: richer descriptions, recognisable in the demo")
    if detect_language:
        pool["desc_lang"] = detect_languages(pool["description"])
        pool = pool[pool["desc_lang"] == "en"]
        _log(report, "16. description really in English", pool,
             "langdetect (seed fixed)")

    # --- 5.10 Final selection ----------------------------------------------------
    cat_df = pool.head(target_size).copy()
    _log(report, f"17. final catalogue (top {target_size:,})", cat_df,
         "target size reached")

    cat_df = finalize_columns(cat_df)
    report_df = pd.DataFrame(report)
    stats = category_stats(cat_df, domain_of)
    return cat_df, report_df, stats


_TITLE_STOPWORDS = {
    "the", "a", "an", "and", "of", "or", "to", "in", "on", "s", "for", "with",
    # generic series / edition words: they do not prove that two titles are the same work
    "volume", "vol", "book", "books", "novel", "part", "series", "edition", "collection",
}


def build_marketing_stripper(raw: pd.DataFrame) -> MarketingStripper:
    """Build the MarketingStripper with knowledge from the dataset.

    * author names (authors.csv): to recognise "--Stephen King";
    * titles (dataset.csv, full title AND part before ':'): to know where
      "author of The Statistical Probability of Love at First Sight" ends.
    """
    authors = pd.read_csv(config.RAW_AUTHORS, dtype=str)["author_name"].dropna()
    titles = raw["title"].dropna()
    titles = pd.concat([titles, titles.str.split(r"\s*:\s*").str[0]]).drop_duplicates()
    return MarketingStripper(authors, titles)


def canonical_author_names(author_lists: Iterable[list[str]]) -> dict[str, str]:
    """Normalised key -> canonical spelling (the most frequent, then the longest).

    {"jkrowling": "J.K. Rowling"} if "J.K. Rowling" is the most frequent
    spelling among "J.K. Rowling" / "J. K. Rowling" / "J K Rowling".
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
    """Replace each author with its canonical spelling, without duplicates, order kept."""
    out: list[str] = []
    for a in authors:
        name = canonical.get(author_key(a), a)
        if name not in out:
            out.append(name)
    return out


def _title_tokens(key: str) -> set[str]:
    """Meaningful words of a normalised title (no stopwords or numbers)."""
    return {w for w in key.split() if w not in _TITLE_STOPWORDS and not w.isdigit()}


def dedup_editions(df: pd.DataFrame, rel_tol: float = 0.02, avg_tol: float = 0.05,
                   min_count_fuzzy: int = 1000, title_jaccard: float = 0.5,
                   return_groups: bool = False) -> pd.DataFrame:
    """Merge the editions of the same work and keep a single representative.

    Two rows are considered the same work if:
      (a) same normalised title (compared without spaces) AND at least one shared
          author (normalised keys);
      (b) OR nearly identical Goodreads ratings (number of ratings within ±2%, average
          within ±0.05, for works with ≥ 1,000 ratings: Goodreads aggregates ratings per
          work, and the dataset was scraped on different dates) AND similar titles
          (word Jaccard ≥ 0.5 or one title contained in the other).
          NB: a first version also accepted "same author" without a similar title;
          inspection showed wrong merges (different novels by the same prolific
          author), hence this stricter rule.
          E.g. "Harry Potter and the Philosopher's Stone - Gryffindor Edition"
          and "Harry Potter and the Sorcerer's Stone".
    The links are grouped with union-find (transitivity), then in each group we
    keep the most rated "standard" edition (without a special-edition word).
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

    # title key computed on the title WITHOUT edition mentions ("Dune : Deluxe Edition" -> "dune")
    title_keys = df["title"].map(lambda t: normalize_title(display_title(t))).tolist()
    author_sets = [set(k) for k in df["author_keys"]]  # used by rules (a) and (b)

    # (a) same normalised title + a shared author. The title is compared WITHOUT
    #     spaces: "Daddy-Long-Legs" = "Daddy Long Legs", "The Wasteland" = "The Waste Land"
    first_seen: dict[tuple[str, str], int] = {}
    for i, (tk, keys) in enumerate(zip(title_keys, author_sets)):
        tk = tk.replace(" ", "")
        for ak in keys:
            j = first_seen.setdefault((tk, ak), i)
            if j != i:
                union(i, j)

    # (b) same Goodreads ratings (sliding window over rows sorted by number of ratings)
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
            # one title contained in the other: the shorter one must have ≥ 3 meaningful
            # words, or 2 words AND a shared author. Otherwise "Out" would be
            # contained in "Knocked Out by my Nunga-Nungas", or "Civil War" (Lucan)
            # in "Spider-Man: Miles Morales: Civil War II"; but "Little Women"
            # is still merged with "Little Women: Or Meg, Jo, Beth and Amy".
            shortest = min(len(ti), len(tj))
            same_author = bool(author_sets[i] & author_sets[j])
            overlap = inter / shortest if shortest >= 3 or (shortest == 2 and same_author) else 0.0
            # The similar title is MANDATORY: a shared author is not enough
            # (two Lee Child novels can have ratings within ±2% of each other).
            if jac >= title_jaccard or overlap >= 0.8:
                union(i, j)

    df["work_id"] = [find(i) for i in range(n)]
    df["is_special_edition"] = df["title"].map(lambda t: bool(EDITION_WORDS_RE.search(t)))
    if "main_author_key" in df.columns:
        df = _harmonize_main_author(df)
    df["desc_len"] = df["description"].str.len()
    # Representative: standard edition first, then the most rated, then the longest description
    df = df.sort_values(["is_special_edition", "rating_count", "desc_len"],
                        ascending=[True, False, False])
    df["n_editions"] = df.groupby("work_id")["work_id"].transform("size")
    if return_groups:  # all rows with their work_id (to inspect the merges)
        return df
    return df.drop_duplicates("work_id", keep="first")


def _harmonize_main_author(df: pd.DataFrame) -> pd.DataFrame:
    """Main author of a work = the most frequent main author among its editions.

    Fixes typos in an isolated edition: the most rated edition of
    Pride and Prejudice is credited to "Jane Austin", but the vast majority of
    its 31 editions are credited to "Jane Austen".
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
        if good_k in keys:            # the right author is already listed (e.g. illustrator first)
            pass
        elif old_k in keys:           # typo: replace the misspelt author
            i = keys.index(old_k)
            lst[i], keys[i] = name_of[good_k], good_k
        df.at[idx, "authors_list"], df.at[idx, "author_keys"] = lst, keys
        df.at[idx, "main_author_key"] = good_k
    return df


def detect_languages(texts: pd.Series) -> list[str]:
    """Detect the language of each text with langdetect (fixed seed = deterministic)."""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = config.RANDOM_SEED
    out = []
    for t in texts:
        try:
            out.append(detect(t[:1000]))
        except LangDetectException:
            out.append("unknown")
    return out


# Order and names of the contract columns (team data contract, see README, section 7)
CLEAN_COLUMNS = [
    "book_id", "title", "authors", "authors_str", "main_author", "categories",
    "description", "description_raw", "pub_year", "rating_avg", "rating_count",
    "bestsellers_rank", "format", "for_ages", "audience", "domain", "is_fiction",
    "image_url", "url", "isbn13", "n_editions", "popularity", "n_words", "embed_text",
]


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived columns and rename them according to the team contract."""
    out = pd.DataFrame(index=df.index)
    out["book_id"] = df["id"].astype(str)
    out["title"] = df["title"].map(display_title)      # without "Deluxe Edition", "(Penguin Classics)"...
    out["authors"] = df["authors_list"]
    out["authors_str"] = df["authors_list"].map(", ".join)
    out["main_author"] = [
        lst[keys.index(k)] for lst, keys, k in zip(df["authors_list"], df["author_keys"], df["main_author_key"])
    ]
    out["categories"] = df["category_names"]
    out["description"] = df["description"]              # cleaned, without press quotes
    out["description_raw"] = df["description_raw"]      # minimal cleaning (traceability)
    year = pd.to_datetime(df["publication-date"], errors="coerce").dt.year
    # aberrant dates (e.g. 2099 = dummy "forthcoming" date) -> missing
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
    out["n_editions"] = df["n_editions"].astype(int)    # number of merged editions
    # Popularity ∈ [0,1]: average rating × normalised log(number of ratings).
    # The log prevents Harry Potter (7 M ratings) from crushing everything else.
    log_count = np.log1p(out["rating_count"])
    out["popularity"] = (out["rating_avg"].fillna(0) / 5.0) * (log_count / log_count.max())
    out["n_words"] = out["description"].str.split().map(len)
    # Text encoded by SBERT: the title adds context ("Dracula"...)
    out["embed_text"] = [
        truncate_at_sentence(f"{t}. {d}", config.EMBED_TEXT_MAX_CHARS)
        for t, d in zip(out["title"], out["description"])
    ]
    return out[CLEAN_COLUMNS].reset_index(drop=True)


def category_stats(catalogue: pd.DataFrame, domain_of: dict[str, str]) -> pd.DataFrame:
    """Frequency of each category in the final catalogue (useful for person 2)."""
    exploded = catalogue[["categories", "is_fiction"]].explode("categories").dropna()
    stats = (exploded.groupby("categories")
             .agg(n_books=("is_fiction", "size"), share_fiction=("is_fiction", "mean"))
             .sort_values("n_books", ascending=False)
             .reset_index().rename(columns={"categories": "category"}))
    stats["domain_rule"] = stats["category"].map(domain_of)
    stats["share_fiction"] = stats["share_fiction"].round(3)
    return stats


# Columns added later by person 2 (empty in the sample)
ENRICHMENT_COLUMNS = ["genres", "themes", "styles", "atmospheres", "keywords"]


def make_sample(catalogue: pd.DataFrame, n: int = config.SAMPLE_SIZE,
                seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Stratified sample (domain × audience) with the full contract schema.

    Person 2's enrichment columns are present but empty, so that everyone
    codes against the final schema right away.
    `label_scores` is stored as a JSON string (an empty dict cannot be
    serialised cleanly in parquet).
    """
    # Same fraction drawn in each stratum (domain × audience) -> proportions preserved.
    # (GroupBy.sample keeps all columns, unlike GroupBy.apply in pandas 3)
    frac = min(1.0, (n + 20) / len(catalogue))
    sample = catalogue.groupby(["domain", "audience"]).sample(frac=frac, random_state=seed)
    sample = sample.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    for col in ENRICHMENT_COLUMNS:
        sample[col] = [[] for _ in range(len(sample))]
    sample["label_scores"] = "{}"
    return sample


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")  # zstd: ≈ 30% more compact than snappy
    return path


# ===========================================================================
# 6. Utilities for the rest of the team
# ===========================================================================

def load_books(path: Path | None = None) -> pd.DataFrame:
    """Load the best available catalogue: enriched > clean > sample.

    List columns (numpy arrays after reading the parquet) are converted back
    to Python lists, and label_scores (JSON) to a dict.
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
    """Find the name (catalogue spelling) of an author from free-text input.

    1. EXACT match on the normalised key: "jk rowling", "J. K. Rowling",
       "J.K.Rowling" -> "J.K. Rowling";
    2. otherwise fuzzy matching (rapidfuzz) to tolerate typos: "stephen kng" -> "Stephen King".
    Returns None if nothing is close enough.
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
    """Mask: the book has at least one author whose normalised key is in `keys`."""
    return df["authors"].map(lambda lst: any(author_key(a) in keys for a in lst))


def get_books_by_author(df: pd.DataFrame, name: str, min_score: int = 85) -> pd.DataFrame:
    """All the catalogue books by an author (free-text input, fuzzy matching).

    The comparison uses the normalised key: all spelling variants of the
    same author are found.
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
    """Boolean mask of the books that satisfy the user's "hard" constraints.

    * audience: "Children" / "Young Adult" / "Adult" (or a list); None = all
    * formats: subset of ("Paperback", "Hardback"); empty/None = all
    * period: key of config.PUBLICATION_PERIODS (PUBLICATION year); None = all
    * excluded_authors: names (free-text input accepted, fuzzy matching)
    * fiction_only: True = fiction, False = non-fiction, None = both
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
