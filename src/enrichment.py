"""
enrichment.py: semantic enrichment of the catalogue (Person 2).

The Book Depository dataset contains NEITHER themes, NOR style, NOR keywords, and its
1,078 categories are noisy. This module DERIVES this information through semantic
analysis and produces ``books_enriched.parquet`` (contract columns, see README, section 7):

    genres        : list[str]  (top 3, Genre block)
    themes        : list[str]  (top 3, Themes block)
    styles        : list[str]  (top 2, Style block)
    atmospheres   : list[str]  (1 mood + 1 period ONLY if it is clear-cut, Atmosphere block)
    keywords      : list[str]  (top 5, KeyBERT-like method)
    label_scores  : JSON str   {block: {label: cosine}}  (load_books() converts it back to a dict)

Pipeline:
    1. blocks.json (taxonomy, 1 descriptive sentence per label)   -> load_blocks()
    2. Book Depository categories -> genres (rules + embeddings) -> build_category_map()
    3. genres of the books                                        -> assign_genres()
    4. zero-shot with embeddings (Themes / Style / Atmosphere)    -> assign_labels()
       Themes v2: each theme = average of 4 sentences (theme_variants.json), built in HERE,
       so `run_enrichment` always reproduces v2 (no more after-the-fact patch).
    5. keywords (noun phrases + TF-IDF + embedding rerank + MMR) -> extract_keywords()
    6. export                                                     -> save_enriched()

Any encoder exposing ``encode(list[str]) -> np.ndarray`` (L2-normalised) works:
``SbertEncoder`` below, or P3's ``EmbeddingEngine`` once it is ready.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
BLOCKS_PATH = ROOT / "data" / "reference" / "blocks.json"
CATEGORY_MAP_PATH = ROOT / "data" / "reference" / "category_genre_map.csv"
THEME_VARIANTS_PATH = ROOT / "data" / "reference" / "theme_variants.json"
EMB_DIR = ROOT / "data" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BLOCK_NAMES = ["Genre", "Themes", "Style", "Atmosphere"]

# Number of labels kept per book for each block (team data contract).
TOP_K = {"Genre": 3, "Themes": 3, "Style": 2, "Atmosphere": 2}

# Words without semantic value to describe a book (publisher vocabulary, Spanish/French function words...).
EXTRA_STOP = {"edition", "editions", "introduction", "includes", "include", "including", "novel", "novels",
              "book", "books", "series", "bestselling", "best", "selling", "author", "authors", "new", "york",
              "times", "reader", "readers", "read", "story", "stories", "life", "world", "first", "years",
              "year", "time", "isbn", "volume", "vol", "collection", "classic", "classics", "masterpiece",
              "translated", "translation", "penguin", "oxford", "press", "creates", "turns", "sets",
              "los", "las", "del", "della", "und", "der", "die", "das", "les", "des", "une", "est", "por",
              "para", "con", "una", "que", "el", "la", "le", "il", "di", "da"}

# z-score threshold (per label, over the catalogue) to keep a label. See threshold_report().
Z_MIN = 1.0

# Atmosphere = 1 period + 1 mood: the two families of labels are kept apart.
ERA_LABELS = {"Ancient world", "Medieval", "18th-19th century", "20th century",
              "Contemporary", "Future"}

# Period of the story: it is NOT forced. It is only kept if its z-score is high AND clearly
# higher than that of the 2nd best period. To be calibrated with scripts/calibrate_era.py.
ERA_Z_MIN = 2.0
ERA_MARGIN = 0.3

# "Young Adult" and "Children's" come ONLY from the `audience` column (P1):
# neither from the Book Depository categories, nor from the similarity with the description.
AUDIENCE_GENRES = {"Children's", "Young Adult"}
AUDIENCE_TO_GENRE = {"Children": "Children's", "Young Adult": "Young Adult"}
AUDIENCE_VOTE = 1.0

# Publisher / layout vocabulary: never a useful keyword.
NOISE_WORDS = {"font", "size", "paperback", "hardcover", "hardback", "edition", "editions", "reprint",
               "foreword", "afterword", "preface", "appendix", "bibliography", "index", "page", "pages",
               "chapter", "chapters", "illustration", "illustrations", "illustrated", "ebook", "kindle",
               "print", "format", "publisher", "copyright", "isbn", "deals", "issues", "issue", "part",
               "way", "thing", "things", "lot", "bit", "something", "everything", "nothing", "one", "two",
               "three", "man", "woman", "people", "day", "days", "place", "work", "works", "set", "sets"}

# For the "genre closest to the description" fallback, fiction / non-fiction is respected.
NONFICTION_GENRES = {"Biography & Memoir", "History", "Science & Nature",
                     "Self-help & Psychology", "Philosophy & Religion",
                     "Business & Economics", "Cooking & Lifestyle"}
NEUTRAL_GENRES = {"Poetry & Drama", "Graphic Novels & Comics", "Humour", "Children's", "Classics"}

# Minimum category -> genre similarity (embeddings) to accept the association.
CATEGORY_MIN_SIM = 0.30

# Weight of the "genre closest to the description" vote against the category votes (1 vote each).
EMB_VOTE = 1.0


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# 0. Minimal encoder (until P3's EmbeddingEngine is ready)
# --------------------------------------------------------------------------- #
class SbertEncoder:
    """Minimal SBERT encoder, lazy loading, L2-normalised output.

    Compatible with the ``encode`` interface of ``EmbeddingEngine`` (P3).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:  # the model is only loaded on the first call
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return emb.astype(np.float32)


def encode_cached(encoder: Encoder, texts: list[str], ids: list[str], cache_name: str,
                  batch_size: int = 64) -> np.ndarray:
    """Encode ``texts`` using a disk cache (.npy + .json of ids for alignment)."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    npy, ids_json = EMB_DIR / f"{cache_name}.npy", EMB_DIR / f"{cache_name}_ids.json"
    if npy.exists() and ids_json.exists() and json.loads(ids_json.read_text()) == list(ids):
        return np.load(npy)
    try:
        emb = encoder.encode(texts, batch_size=batch_size, show_progress=True)  # type: ignore[call-arg]
    except TypeError:  # encoder without these arguments (e.g. EmbeddingEngine)
        emb = encoder.encode(texts)
    np.save(npy, emb)
    ids_json.write_text(json.dumps(list(ids)))
    return emb


# --------------------------------------------------------------------------- #
# 1. Reference taxonomy
# --------------------------------------------------------------------------- #
def load_blocks(path: Path | str = BLOCKS_PATH) -> dict[str, dict[str, str]]:
    """Load and validate blocks.json: {block: {label: descriptive sentence}}."""
    blocks = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [b for b in BLOCK_NAMES if b not in blocks]
    if missing:
        raise ValueError(f"blocks.json : blocs manquants {missing}")
    for b, labels in blocks.items():
        for label, phrase in labels.items():
            if not isinstance(phrase, str) or len(phrase.split()) < 5:
                raise ValueError(f"blocks.json : phrase trop courte pour {b}/{label}")
    return blocks


def load_theme_variants(path: Path | str = THEME_VARIANTS_PATH, themes: Iterable[str] | None = None) -> dict:
    """Load theme_variants.json: {theme: [3 sentences]}. Checks that it covers exactly the themes of blocks.json."""
    variants = json.loads(Path(path).read_text(encoding="utf-8"))
    if themes is not None and set(variants) != set(themes):
        raise ValueError(f"theme_variants.json incohérent avec blocks.json : {sorted(set(variants) ^ set(themes))}")
    return variants


def encode_blocks(encoder: Encoder, blocks: dict[str, dict[str, str]], theme_variants="default"):
    """Encode the descriptive SENTENCE of each label (not the bare word).

    Themes v2: each theme is represented by the (renormalised) AVERAGE of the embedding of its sentence
    in blocks.json and of its variants in theme_variants.json. ``theme_variants="default"`` loads the
    file (error if it is missing: no silent fallback to v1); ``None`` disables it (v1, comparison).

    Returns {block: (labels, matrix (L, d))}.
    """
    if isinstance(theme_variants, str) and theme_variants == "default":
        theme_variants = load_theme_variants(themes=blocks["Themes"])
    out = {}
    for b in BLOCK_NAMES:
        labels = list(blocks[b])
        if b == "Themes" and theme_variants:
            mats = []
            for l in labels:
                v = np.asarray(encoder.encode([blocks[b][l]] + list(theme_variants[l])), dtype=np.float32)
                m = v.mean(0)
                mats.append(m / max(float(np.linalg.norm(m)), 1e-12))
            out[b] = (labels, np.stack(mats).astype(np.float32))
        else:
            out[b] = (labels, np.asarray(encoder.encode([blocks[b][l] for l in labels])))
    return out


# --------------------------------------------------------------------------- #
# 2. Book Depository categories -> Genres
# --------------------------------------------------------------------------- #
# (regex, genre, exclusion regex). A category name can trigger SEVERAL rules
# (e.g. "Crime, Thriller & Mystery"). These rules cover the obvious cases;
# the rest is handled by embedding similarity.
CATEGORY_RULES: list[tuple[str, str, str | None]] = [
    (r"fantasy|wizard|sword (and|&) sorcery|epic fantasy", "Fantasy", None),
    (r"sci[ -]?fi\b|science fiction|dystopi|space opera|cyberpunk", "Science Fiction", None),
    (r"detective|mystery|whodunn|crime fiction|crime, thriller", "Mystery & Detective", None),
    (r"thriller|suspense|espionage|spy\b|legal thriller", "Thriller & Suspense", None),
    (r"horror|ghost|supernatural|gothic|vampire|zombie|occult", "Horror", None),
    (r"romance(?!\s+languages?)|romantic", "Romance", None),
    (r"historical fiction|historical novel|historical romance|sagas?\b", "Historical Fiction", None),
    (r"literary fiction|general fiction|contemporary fiction|modern fiction|literary", "Literary Fiction", None),
    (r"\bclassics?\b", "Classics", None),
    (r"young adult|teen", "Young Adult", None),
    (r"children|picture book|baby|toddler|early learning|story books|nursery|bedtime", "Children's", None),
    (r"humou?r|comedy|satire|joke|funny", "Humour", None),
    (r"biograph|autobiograph|memoir|diaries|true story", "Biography & Memoir", None),
    (r"\bhistory\b|military|ancient|medieval|\bwars?\b", "History", r"fiction|children|young adult"),
    (r"science|physics|biology|chemistry|nature|astronomy|mathematic|ecolog|natural history|environment",
     "Science & Nature", r"fiction|children|young adult"),
    (r"self[- ]help|psycholog|personal development|self-esteem|mental health|mindful|motivat|happiness|well-?being",
     "Self-help & Psychology", None),
    (r"philosoph|religio|spiritual|christian|bible|buddh|islam|theolog|ethics|occult", "Philosophy & Religion", r"horror"),
    (r"poetry|poems|\bplays?\b|drama|theatre|shakespeare", "Poetry & Drama", None),
    (r"graphic novel|comic|manga|cartoon", "Graphic Novels & Comics", None),
    (r"business|econom|financ|management|marketing|invest|entrepreneur|leadership|accounting", "Business & Economics", None),
    (r"cook|food|recipe|baking|wine|diet\b|lifestyle|craft|gardening|travel|fitness", "Cooking & Lifestyle", None),
]
_COMPILED_RULES = [(re.compile(p, re.I), g, re.compile(x, re.I) if x else None)
                   for p, g, x in CATEGORY_RULES]


def rule_genres(category_name: str) -> list[str]:
    """Genres triggered by the keyword rules for a category name."""
    found: list[str] = []
    for pat, genre, excl in _COMPILED_RULES:
        if pat.search(category_name) and not (excl and excl.search(category_name)):
            if genre not in found:
                found.append(genre)
    return found


def category_counts(df: pd.DataFrame) -> Counter:
    """Frequency of each category in the catalogue (``categories`` column = list of names)."""
    c: Counter = Counter()
    for cats in df["categories"]:
        c.update(list(cats) if cats is not None else [])
    return c


def build_category_map(counts: Counter, encoder: Encoder, blocks: dict, existing: pd.DataFrame | None = None,
                       min_sim: float = CATEGORY_MIN_SIM) -> pd.DataFrame:
    """Mapping table category -> genre(s).

    Columns: category, n_books, rule_genres, emb_genre, emb_sim, genre_auto, genre_manual, genres_final.
    - ``genre_auto``   : rules if they trigger, otherwise the closest genre by embeddings
                         (if cosine >= min_sim, otherwise empty: generic categories "Fiction", "Books"...).
    - ``genre_manual`` : column TO BE FILLED by hand (review of the first 100 categories);
                         separator "|". It is kept if the file is regenerated (``existing``).
    - ``genres_final`` : genre_manual if it exists, otherwise genre_auto.
    """
    cats = [c for c, _ in counts.most_common()]
    genre_labels = list(blocks["Genre"])
    g_emb = np.asarray(encoder.encode([blocks["Genre"][g] for g in genre_labels]))
    c_emb = np.asarray(encoder.encode(cats))
    sims = c_emb @ g_emb.T
    best = sims.argmax(1)

    rows = []
    for i, cat in enumerate(cats):
        rg = rule_genres(cat)
        emb_g, emb_s = genre_labels[best[i]], float(sims[i, best[i]])
        if rg:
            auto = rg
        elif emb_s >= min_sim:
            auto = [emb_g]
        else:
            auto = []
        rows.append({"category": cat, "n_books": counts[cat], "rule_genres": "|".join(rg),
                     "emb_genre": emb_g, "emb_sim": round(emb_s, 3),
                     "genre_auto": "|".join(auto), "genre_manual": ""})
    out = pd.DataFrame(rows)
    if existing is not None and "genre_manual" in existing:  # the manual work already done is kept
        manual = existing.set_index("category")["genre_manual"].fillna("")
        out["genre_manual"] = out["category"].map(manual).fillna("")
    out["genres_final"] = np.where(out["genre_manual"].astype(str).str.strip() != "",
                                   out["genre_manual"], out["genre_auto"])
    return out


def load_category_map(path: Path | str = CATEGORY_MAP_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["genres_final"] = np.where(df["genre_manual"].astype(str).str.strip() != "",
                                  df["genre_manual"], df["genre_auto"])
    return df


def assign_genres(df: pd.DataFrame, book_emb: np.ndarray, genre_labels: list[str], genre_emb: np.ndarray,
                  cat_map: pd.DataFrame, top_k: int = TOP_K["Genre"]) -> tuple[list[list[str]], dict]:
    """Genres of each book.

    1. Union of the genres of its categories (the mapping table of build_category_map).
    2. + "Children's" / "Young Adult" ONLY from P1's ``audience`` column
       (never through categories or embeddings: they gave them to adult books).
    3. Ranking: number of categories supporting the genre, then description <-> genre cosine.
    4. Fallback (no usable category): genre closest to the description,
       restricted to the book's domain (fiction / non-fiction).
    """
    cat2genres = {r.category: [g for g in str(r.genres_final).split("|") if g]
                  for r in cat_map.itertuples()}
    g_index = {g: i for i, g in enumerate(genre_labels)}
    sims = book_emb @ genre_emb.T  # (N, G) cosine description <-> genre description

    result, n_fallback = [], 0
    for i, row in enumerate(df.itertuples(index=False)):
        votes: Counter = Counter()
        for cat in (list(row.categories) if row.categories is not None else []):
            votes.update(g for g in cat2genres.get(cat, []) if g not in AUDIENCE_GENRES)
        n_cat_votes = sum(votes.values())

        # Genre closest to the DESCRIPTION, consistent with the fiction / non-fiction domain.
        # It always votes (weight EMB_VOTE): fixes misleading commercial categories
        # (e.g. Harry Potter classified as "Sci Fi" by Book Depository).
        is_fic = bool(getattr(row, "is_fiction", True))
        allowed = [g for g in genre_labels if g not in AUDIENCE_GENRES
                   and ((g in NEUTRAL_GENRES) or ((g in NONFICTION_GENRES) != is_fic))]
        emb_best = max(allowed, key=lambda g: sims[i, g_index[g]])
        if n_cat_votes == 0:
            n_fallback += 1
        votes[emb_best] += EMB_VOTE
        # Target audience: the only source of "Children's" / "Young Adult" = P1's `audience` column.
        aud_genre = AUDIENCE_TO_GENRE.get(getattr(row, "audience", None))
        if aud_genre in g_index:
            votes[aud_genre] += AUDIENCE_VOTE
        ranked = sorted(votes, key=lambda g: (votes[g], sims[i, g_index[g]]), reverse=True)
        result.append(ranked[:top_k])
    stats = {"n_books": len(df), "n_fallback_embedding": n_fallback,
             "pct_fallback": round(100 * n_fallback / max(len(df), 1), 2)}
    return result, stats


# --------------------------------------------------------------------------- #
# 3. Zero-shot with embeddings (Themes / Style / Atmosphere)
# --------------------------------------------------------------------------- #
def score_block(book_emb: np.ndarray, label_emb: np.ndarray) -> np.ndarray:
    """Cosine (N, L) between the description of each book and the sentence of each label (normalised vectors)."""
    return book_emb @ label_emb.T


def zscore_columns(sims: np.ndarray) -> np.ndarray:
    """Normalise the scores per label: per-label centring + a scale SHARED by the whole block.

    Why centre: some labels have a high average cosine for ALL books (generic sentences).
    Subtracting the label's mean amounts to asking "is this book closer to this label than the average book?".
    Why a shared scale (and not one standard deviation per label): dividing by each label's standard deviation
    artificially inflated the less dispersed labels and produced aberrant labels (Gatsby -> "Freedom").
    """
    centered = sims - sims.mean(0, keepdims=True)
    return centered / max(float(centered.std()), 1e-9)


def select_labels(z: np.ndarray, labels: list[str], top_k: int, z_min: float = Z_MIN,
                  min_keep: int = 1, group_of: dict[str, str] | None = None,
                  group_max: int | None = None) -> list[list[str]]:
    """Keep, for each book, the ``top_k`` best labels whose z-score >= z_min.

    - ``min_keep`` : at least this number of labels is always kept (never a book without a label).
    - ``group_of`` / ``group_max`` : limits the number of labels per family (e.g. 1 period + 1 mood).
    """
    out = []
    for row in z:
        order = np.argsort(-row)
        chosen: list[int] = []
        per_group: Counter = Counter()
        for j in order:
            if len(chosen) >= top_k:
                break
            if group_of is not None and group_max is not None and per_group[group_of[labels[j]]] >= group_max:
                continue
            if row[j] < z_min and len(chosen) >= min_keep:
                break
            chosen.append(j)
            if group_of is not None:
                per_group[group_of[labels[j]]] += 1
        out.append([labels[j] for j in chosen])
    return out


def select_atmospheres(z: np.ndarray, labels: list[str], z_min: float = Z_MIN,
                       era_z_min: float = ERA_Z_MIN, era_margin: float = ERA_MARGIN) -> list[list[str]]:
    """Atmosphere = 1 mood + 1 period ONLY if it is clear-cut.

    - Mood   : the best one, kept if z >= z_min (at least one mood if no period is kept).
    - Period : never forced. Kept if z >= era_z_min AND higher by at least era_margin than the
               2nd best period (otherwise the book has no period, which is a valid result).
    Labels are returned from closest to least close.
    """
    era_idx = [j for j, l in enumerate(labels) if l in ERA_LABELS]
    mood_idx = [j for j, l in enumerate(labels) if l not in ERA_LABELS]
    out = []
    for row in z:
        picked: list[int] = []
        e = sorted(era_idx, key=lambda j: -row[j])
        if e and row[e[0]] >= era_z_min and (len(e) == 1 or row[e[0]] - row[e[1]] >= era_margin):
            picked.append(e[0])
        m = max(mood_idx, key=lambda j: row[j])
        if row[m] >= z_min or not picked:
            picked.append(m)
        picked.sort(key=lambda j: -row[j])
        out.append([labels[j] for j in picked])
    return out


def assign_labels(book_emb: np.ndarray, labels: list[str], label_emb: np.ndarray, block: str,
                  z_min: float = Z_MIN, top_k: int | None = None, era_z_min: float = ERA_Z_MIN):
    """Zero-shot with embeddings for one block. Returns (labels per book, raw cosine matrix)."""
    sims = score_block(book_emb, label_emb)
    z = zscore_columns(sims)
    if block == "Atmosphere":
        return select_atmospheres(z, labels, z_min, era_z_min), sims
    return select_labels(z, labels, top_k or TOP_K[block], z_min, 1), sims


def threshold_report(sims: np.ndarray, labels: list[str], block: str,
                     thresholds: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Helps CALIBRATE the z_min threshold: average number of labels / book and % of books without a label at the threshold."""
    z = zscore_columns(sims)
    rows = []
    for t in thresholds:
        chosen, _ = assign_labels_from_z(z, labels, block, t)
        n_pass = (z >= t).sum(1)
        rows.append({"z_min": t, "mean_labels_kept": round(float(np.mean([len(c) for c in chosen])), 2),
                     "pct_books_no_label_above_threshold": round(float((n_pass == 0).mean() * 100), 1)})
    return pd.DataFrame(rows)


def assign_labels_from_z(z, labels, block, z_min):
    if block == "Atmosphere":
        return select_atmospheres(z, labels, z_min), z
    return select_labels(z, labels, TOP_K[block], z_min, 1), z


def era_coverage_report(sims: np.ndarray, labels: list[str],
                        thresholds: Iterable[float] = (1.0, 1.5, 2.0, 2.5, 3.0), margin: float = ERA_MARGIN) -> pd.DataFrame:
    """Helps CALIBRATE ERA_Z_MIN: share of books that receive a period depending on the threshold."""
    z = zscore_columns(sims)
    rows = []
    for t in thresholds:
        out = select_atmospheres(z, labels, Z_MIN, t, margin)
        rows.append({"era_z_min": t, "pct_books_with_era": round(100 * float(np.mean(
            [any(l in ERA_LABELS for l in o) for o in out])), 1)})
    return pd.DataFrame(rows)


def build_label_scores(sims_by_block: dict[str, tuple[list[str], np.ndarray]], decimals: int = 3) -> list[str]:
    """label_scores per book = JSON string {block: {label: cosine}} (contract: dict converted back by load_books)."""
    n = next(iter(sims_by_block.values()))[1].shape[0]
    out = []
    for i in range(n):
        out.append(json.dumps({b: {l: round(float(s[i, j]), decimals) for j, l in enumerate(labels)}
                               for b, (labels, s) in sims_by_block.items()}))
    return out


# --------------------------------------------------------------------------- #
# 4. Keywords (KeyBERT-like method)
# --------------------------------------------------------------------------- #
def _words(phrase: str) -> set[str]:
    return set(phrase.split())


def mmr_select(cand_emb: np.ndarray, doc_emb: np.ndarray, phrases: list[str], top_n: int = 5,
               diversity: float = 0.5) -> list[str]:
    """Maximal Marginal Relevance: relevance to the document vs redundancy with the keywords already chosen.

    A candidate contained in an already chosen keyword (or the reverse) is also discarded:
    "wizard" vs "boy wizard".
    """
    if len(phrases) == 0:
        return []
    rel = cand_emb @ doc_emb
    chosen = [int(rel.argmax())]
    while len(chosen) < min(top_n, len(phrases)):
        red = (cand_emb @ cand_emb[chosen].T).max(1)
        score = (1 - diversity) * rel - diversity * red
        for j in chosen:
            score[j] = -np.inf
        for j in range(len(phrases)):  # lexical inclusion = redundancy
            if any(_words(phrases[j]) <= _words(phrases[k]) or _words(phrases[k]) <= _words(phrases[j])
                   for k in chosen):
                score[j] = -np.inf
        if np.isinf(score).all():
            break
        chosen.append(int(score.argmax()))
    return [phrases[j] for j in chosen]


# Parts of speech that can make up a noun phrase (adjectives + nouns). Verbs, determiners,
# prepositions, adverbs, numbers... break the phrase: "deals issues" (verb + noun) is therefore discarded.
NP_POS = {"ADJ", "NOUN", "PROPN"}
NP_HEAD_POS = {"NOUN", "PROPN"}


def noun_phrase_candidates(tagged: list[tuple[str, str]], stop: set[str], max_len: int = 2) -> list[str]:
    """Keyword candidates of ONE tagged text [(word, POS), ...]: noun phrases only.

    The text is split into maximal runs of ADJ/NOUN/PROPN words without stopwords; each run is trimmed to
    end with a noun (head of the phrase), then we keep the head alone and the "modifier + head" bigram.
    """
    cands: list[str] = []
    run: list[tuple[str, str]] = []

    def flush():
        while run and run[-1][1] not in NP_HEAD_POS:
            run.pop()
        if run:
            head = run[-1][0]
            cands.append(head)
            if len(run) >= 2 and max_len >= 2:
                cands.append(f"{run[-2][0]} {head}")
        run.clear()

    for text, pos in tagged:
        w = text.lower()
        ok = pos in NP_POS and w.isalpha() and len(w) >= 3 and w not in stop
        if ok:
            run.append((w, pos))
        else:
            flush()
    flush()
    return [c for c in cands if not any(len(t) < 3 for t in c.split())]


def spacy_tagger(model: str = "en_core_web_sm"):
    """spaCy part-of-speech tagger: texts -> [[(word, POS), ...], ...]. Raises ImportError/OSError if unavailable."""
    import spacy
    nlp = spacy.load(model, disable=["parser", "ner", "lemmatizer"])

    def tag(texts: list[str]) -> list[list[tuple[str, str]]]:
        return [[(t.text, t.pos_) for t in doc] for doc in nlp.pipe((x[:2000] for x in texts), batch_size=128)]
    return tag


def _identity(x):
    return x


def extract_keywords(df: pd.DataFrame, book_emb: np.ndarray, encoder: Encoder, top_n: int = 5,
                     n_candidates: int = 15, diversity: float = 0.5, ngram_range=(1, 2),
                     text_col: str = "description", pos: bool | str = "auto", tagger=None) -> list[list[str]]:
    """Keywords per book, following the KeyBERT principle (candidate <-> document cosine + MMR).

    Candidates = NOUN PHRASES only (spaCy part-of-speech tagging), ranked by TF-IDF, then reordered with
    embeddings + MMR. ``pos="auto"``: uses spaCy if installed, otherwise a fallback (n-grams filtered by a blacklist,
    noisier, with a warning); ``pos=True``: error if spaCy is missing; ``pos=False``: fallback forced.
    ``tagger`` allows injecting a tagger (tests).

    Why not KeyBERT directly on 12,000 books? KeyBERT encodes the WHOLE vocabulary of the collection:
    too slow on CPU. Here the ``n_candidates`` best candidates per book are pre-filtered with TF-IDF.
    """
    import warnings
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    docs = df[text_col].fillna("").astype(str).tolist()
    stop = set(ENGLISH_STOP_WORDS) | EXTRA_STOP | NOISE_WORDS

    if tagger is None and pos is not False:
        try:
            tagger = spacy_tagger()
        except (ImportError, OSError) as e:
            if pos is True:
                raise RuntimeError("spaCy + en_core_web_sm requis : pip install spacy && "
                                   "python -m spacy download en_core_web_sm") from e
            warnings.warn("spaCy indisponible : mots-clés par n-grammes filtrés (plus bruités).")

    if tagger is not None:
        tagged = tagger(docs)
        cand_docs = [noun_phrase_candidates(t, stop) for t in tagged]
        tfidf = TfidfVectorizer(analyzer=_identity, min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
        X = tfidf.fit_transform(cand_docs).tocsr()
    else:
        tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=sorted(stop), lowercase=True,
                                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
                                min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
        X = tfidf.fit_transform(docs).tocsr()
    vocab = np.array(tfidf.get_feature_names_out())

    authors = df["authors_str"].fillna("").astype(str) if "authors_str" in df else pd.Series([""] * len(df))
    author_words = [{w for w in re.findall(r"[a-z]{3,}", a.lower())} for a in authors]

    # 1. top-n_candidates TF-IDF per book
    cand_ids: list[np.ndarray] = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        idx, val = X.indices[start:end], X.data[start:end]
        top = idx[np.argsort(-val)]
        # candidates that repeat the name of one of the book's authors are removed ("scott fitzgerald")
        if author_words[i]:
            top = np.array([j for j in top if not (set(vocab[j].split()) & author_words[i])], dtype=int)
        cand_ids.append(top[:n_candidates])

    # 2. encoding of the deduplicated union of the candidates
    union = np.unique(np.concatenate(cand_ids)) if cand_ids else np.array([], dtype=int)
    pos_of = {int(j): k for k, j in enumerate(union)}
    union_emb = np.asarray(encoder.encode(vocab[union].tolist())) if len(union) else np.zeros((0, 1))

    # 3. embedding rerank + MMR
    out = []
    for i, ids in enumerate(cand_ids):
        if len(ids) == 0:
            out.append([])
            continue
        cand_emb = union_emb[[pos_of[int(j)] for j in ids]]
        out.append(mmr_select(cand_emb, book_emb[i], vocab[ids].tolist(), top_n, diversity))
    return out


def extract_keywords_keybert_lib(texts: list[str], model_name: str = DEFAULT_MODEL, top_n: int = 5):
    """Version using the KeyBERT library (to be used on a SAMPLE for comparison, e.g. 200 books)."""
    from keybert import KeyBERT
    kw = KeyBERT(model=model_name)
    res = kw.extract_keywords(texts, keyphrase_ngram_range=(1, 2), stop_words="english",
                              use_mmr=True, diversity=0.5, top_n=top_n)
    return [[k for k, _ in r] for r in res] if res and isinstance(res[0], list) else [[k for k, _ in res]]


# --------------------------------------------------------------------------- #
# 5. Full pipeline + export
# --------------------------------------------------------------------------- #
CONTRACT_NEW_COLS = ["genres", "themes", "styles", "atmospheres", "keywords", "label_scores"]


def enrich_catalogue(df: pd.DataFrame, encoder: Encoder, blocks: dict | None = None,
                     cat_map: pd.DataFrame | None = None, book_emb: np.ndarray | None = None,
                     z_min: float = Z_MIN, with_keywords: bool = True, verbose: bool = True,
                     theme_variants="default", era_z_min: float = ERA_Z_MIN, keyword_pos: bool | str = "auto"):
    """Enrich ``df`` (books_clean or books_sample). Returns (enriched df, report dict).

    FINAL version in a single call: v2 themes (``theme_variants``), period not forced (``era_z_min``),
    noun-phrase keywords (``keyword_pos``). P2 columns already present are overwritten.
    """
    blocks = blocks or load_blocks()
    df = df.drop(columns=[c for c in CONTRACT_NEW_COLS if c in df.columns]).reset_index(drop=True)
    log = print if verbose else (lambda *a, **k: None)

    log("[1/5] Encodage des livres et des phrases de labels")
    if book_emb is None:
        book_emb = np.asarray(encoder.encode(df["embed_text"].fillna("").tolist()))
    enc_blocks = encode_blocks(encoder, blocks, theme_variants)

    log("[2/5] Genres (catégories -> genres, repli embeddings, public via audience)")
    if cat_map is None:
        cat_map = build_category_map(category_counts(df), encoder, blocks)
    g_labels, g_emb = enc_blocks["Genre"]
    genres, g_stats = assign_genres(df, book_emb, g_labels, g_emb, cat_map)

    log("[3/5] Zero-shot Themes (v2) / Style / Atmosphere")
    sims_by_block = {"Genre": (g_labels, score_block(book_emb, g_emb))}
    assigned = {}
    for block in ["Themes", "Style", "Atmosphere"]:
        labels, emb = enc_blocks[block]
        assigned[block], sims = assign_labels(book_emb, labels, emb, block, z_min, era_z_min=era_z_min)
        sims_by_block[block] = (labels, sims)

    log("[4/5] Mots-clés (groupes nominaux)")
    keywords = (extract_keywords(df, book_emb, encoder, pos=keyword_pos) if with_keywords
                else [[] for _ in range(len(df))])

    log("[5/5] Assemblage")
    out = df.copy()
    out["genres"], out["themes"] = genres, assigned["Themes"]
    out["styles"], out["atmospheres"] = assigned["Style"], assigned["Atmosphere"]
    out["keywords"] = keywords
    out["label_scores"] = build_label_scores(sims_by_block)

    report = {"n_books": len(out), "z_min": z_min, "era_z_min": era_z_min,
              "themes_version": "v2" if theme_variants else "v1", "genres": g_stats,
              "pct_books_with_era": round(100 * float(np.mean(
                  [any(l in ERA_LABELS for l in a) for a in out["atmospheres"]])), 1),
              "mean_labels": {c: round(float(out[c].map(len).mean()), 2)
                              for c in ["genres", "themes", "styles", "atmospheres", "keywords"]}}
    return out, report


def subset_enriched(enriched: pd.DataFrame, book_ids: Iterable) -> pd.DataFrame:
    """EXACT subset of the full enrichment (same labels, same scores), in the order of ``book_ids``.

    This is how books_enriched_sample.parquet is built: never by re-enriching 500 isolated books
    (the per-label normalisation depends on the whole catalogue, so the labels differed from those of the full file).
    """
    ids = [str(b) for b in book_ids]
    keyed = enriched.assign(book_id=enriched["book_id"].astype(str)).set_index("book_id", drop=False)
    missing = [b for b in ids if b not in keyed.index]
    if missing:
        raise ValueError(f"{len(missing)} book_id absents du fichier enrichi, ex. {missing[:3]}")
    return keyed.loc[ids].reset_index(drop=True)


def save_enriched(df: pd.DataFrame, path: Path | str) -> None:
    """Write the parquet, normalising the types (Python lists, label_scores as a JSON string)."""
    df = df.copy()
    for c in ["authors", "categories", "genres", "themes", "styles", "atmospheres", "keywords"]:
        if c in df.columns:
            df[c] = df[c].map(lambda v: [] if v is None else [str(x) for x in list(v)])
    if "label_scores" in df.columns:
        df["label_scores"] = df["label_scores"].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# 6. Validation on the hand-annotated books
# --------------------------------------------------------------------------- #
def make_annotation_sheet(df: pd.DataFrame, n: int = 60, seed: int = 42, n_nonfiction: int = 15,
                          exclude_ids: Iterable = ()) -> pd.DataFrame:
    """Annotation sheet (WITHOUT the predictions, so as not to bias the annotator).

    ``exclude_ids``: books to leave out (e.g. the 60 already annotated and used for tuning), to get
    a never-seen TEST set. Fill in gold_genres (1-3 labels) and gold_themes (2-3 labels) with the labels
    of blocks.json, separated by "|".
    """
    excl = {str(x) for x in exclude_ids}
    df = df[~df["book_id"].astype(str).isin(excl)]
    fic = df[df["is_fiction"]].sample(n - n_nonfiction, random_state=seed)
    non = df[~df["is_fiction"]].sample(min(n_nonfiction, int((~df["is_fiction"]).sum())), random_state=seed)
    sheet = pd.concat([fic, non]).sample(frac=1, random_state=seed)
    sheet = sheet[["book_id", "title", "authors_str", "description"]].copy()
    sheet["description"] = sheet["description"].str.slice(0, 500)
    sheet["gold_genres"], sheet["gold_themes"], sheet["annotator"] = "", "", ""
    return sheet


def evaluate_against_gold(enriched: pd.DataFrame, gold_path: Path | str) -> dict:
    """Compare the predicted labels with the manual annotation (gold_genres / gold_themes columns)."""
    gold = pd.read_csv(gold_path, keep_default_na=False, dtype={"book_id": str})
    gold = gold[(gold["gold_genres"].str.strip() != "") | (gold["gold_themes"].str.strip() != "")]
    enriched = enriched.assign(book_id=enriched["book_id"].astype(str))
    merged = gold.merge(enriched[["book_id", "genres", "themes"]], on="book_id", how="inner")

    def metrics(pred_col: str, gold_col: str) -> dict:
        prec, rec, hit1, hit_any = [], [], [], []
        for _, r in merged.iterrows():
            g = {x.strip() for x in str(r[gold_col]).split("|") if x.strip()}
            p = list(r[pred_col])
            if not g or not p:
                continue
            inter = len(g & set(p))
            prec.append(inter / len(p))
            rec.append(inter / len(g))
            hit1.append(p[0] in g)
            hit_any.append(inter > 0)
        n = max(len(prec), 1)
        return {"n": len(prec), "precision@k": round(sum(prec) / n, 3), "recall": round(sum(rec) / n, 3),
                "top1_in_gold": round(sum(hit1) / n, 3), "at_least_one_match": round(sum(hit_any) / n, 3)}

    return {"genres": metrics("genres", "gold_genres"), "themes": metrics("themes", "gold_themes")}
