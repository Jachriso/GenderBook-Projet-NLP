"""
Construction of the reader's SEMANTIC PROFILE from the questionnaire answers (P4).

    answers.json (P5)  ──►  build_profile(answers, engine)  ──►  UserProfile

Guiding idea: EVERYTHING the user answers becomes TEXT and then an
EMBEDDING, including the numeric answers. A Likert "complexity = 5" is not
used as a number but converted into the sentence "a complex, dense,
literary and demanding narrative", encoded by SBERT and compared with the books.
This is what keeps the analysis semantic even for closed questions
(the assignment's "not purely numerical" requirement).

What build_profile produces:
1. q_global: vector of the overall request = average of the embeddings of the free-text
   answers (ideal_book, loved_book, atmosphere), mixed with the centroid of the
   favourite author's books if they are in the catalogue (0.7 / 0.3).
2. P_b(u): for each block (Genre, Themes, Style, Atmosphere), the list of the
   user's preference sentences and their embeddings.
3. W_b: block weights (importance Likert), λ: popularity bonus.
4. q_avoid: embedding of the "to avoid" text (negations removed).
5. block_affinities (profile radar) and summary (template-based summary).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.fallback_engine import l2_normalize
from src.schemas import BLOCKS, UserProfile

# ---------------------------------------------------------------------------
# 1. Likert -> sentences conversion (core of "semantic, not numerical")
# ---------------------------------------------------------------------------
# One sentence per level. Level 3 (neutral) = no preference => no sentence,
# so as not to dilute the block with a text that says nothing.
LIKERT_SENTENCES: dict[str, dict[int, str]] = {
    # Narrative complexity (Style block)
    "complexity": {
        1: "a simple, easy and accessible story with clear, straightforward writing",
        2: "an easy-to-read, light narrative that is not too demanding",
        4: "a rich and layered narrative with some depth and complexity",
        5: "a complex, dense, literary and demanding narrative with sophisticated prose",
    },
    # Pace (Style block)
    "pace": {
        1: "a slow, contemplative and quiet story that takes its time",
        2: "a gently paced, reflective story",
        4: "a well-paced story with momentum and suspense",
        5: "a fast-paced, gripping page-turner that is hard to put down",
    },
    # Light <-> dark tone (Atmosphere block)
    "darkness": {
        1: "a light, warm, cozy and feel-good story that makes you smile",
        2: "a mostly uplifting and gentle story",
        4: "a serious, somber story with darker moments",
        5: "a dark, disturbing and intense atmosphere",
    },
}

# Guided question "character-driven or plot-driven?" (Style block)
FOCUS_SENTENCES: dict[str, str] = {
    "Characters": "a character-driven story with deep psychological portraits and inner lives",
    "Plot": "a plot-driven story full of twists, events and narrative surprises",
    "Both": "a story that balances rich characters and a compelling plot",
}

# Q7 "Period of the story" -> Atmosphere block label of blocks.json (P2).
# Several spellings accepted, to be robust to the app's wording.
PERIOD_TO_LABEL: dict[str, str] = {
    "ancient": "Ancient world", "ancient world": "Ancient world", "antiquity": "Ancient world",
    "medieval": "Medieval", "middle ages": "Medieval",
    "18-19th c.": "18th-19th century", "18–19th c.": "18th-19th century",
    "18th-19th century": "18th-19th century", "19th century": "18th-19th century",
    "victorian": "18th-19th century",
    "20th c.": "20th century", "20th century": "20th century",
    "contemporary": "Contemporary", "present day": "Contemporary",
    "future": "Future", "futuristic": "Future",
}
NO_PREFERENCE = {"", "no preference", "any", "none"}

# Negation words removed from the "to avoid" text: SBERT handles negation poorly,
# "No romance" would be encoded almost like "romance"… which is precisely what
# we want to penalise. The negation is therefore removed to keep the SUBJECT to avoid.
_NEGATION_RE = re.compile(
    r"\b(no|not|nothing|never|avoid(ing)?|without|don'?t|do not|i don'?t want|"
    r"i do not want|please|anything|any|too much|i hate|i dislike)\b",
    flags=re.IGNORECASE,
)


def likert_to_sentence(key: str, value: int | float | None) -> str | None:
    """Convert a Likert answer (1-5) into a descriptive sentence, or None if neutral/missing."""
    if value is None:
        return None
    v = int(round(float(value)))
    v = min(max(v, 1), 5)
    if v == config.LIKERT_NEUTRAL:
        return None
    return LIKERT_SENTENCES.get(key, {}).get(v)


def clean_avoid_text(text: str) -> str:
    """Example: 'No romance-centered plots, no gore' -> 'romance-centered plots, gore'."""
    t = _NEGATION_RE.sub(" ", str(text or ""))
    t = re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s+", " ", t))
    return t.strip(" ,.;")


def lambda_from_likert(popularity: int | float | None) -> float:
    """Popularity Likert (1 = hidden gems ... 5 = best-sellers) -> λ ∈ [0, LAMBDA_POP_MAX]."""
    if popularity is None:
        popularity = 3
    p = min(max(float(popularity), 1.0), 5.0)
    return round(config.LAMBDA_POP_MAX * (p - 1.0) / 4.0, 4)


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace("&", " and ")).strip()


def resolve_label(blocks: dict[str, dict[str, str]], block: str, label: str) -> str | None:
    """Find the exact blocks.json label from the app's wording.

    Tolerates case and short variants: "Identity" -> "Identity & self-discovery",
    "Morality & Guilt" -> "Morality & guilt". Returns None if nothing matches.
    """
    labels = list(blocks.get(block, {}))
    if label in labels:
        return label
    key = _norm_label(label)
    for lab in labels:                       # 1. equality ignoring case and punctuation
        if _norm_label(lab) == key:
            return lab
    for lab in labels:                       # 2. the wording is the start of the label
        if key and _norm_label(lab).startswith(key):
            return lab
    return None


def label_phrase(blocks: dict[str, dict[str, str]], block: str, label: str) -> str:
    """Descriptive sentence of a label (blocks.json); otherwise the label itself (fallback)."""
    lab = resolve_label(blocks, block, label)
    return blocks[block][lab] if lab is not None else str(label)


# ---------------------------------------------------------------------------
# 2. Preference sentences per block: P_b(u)
# ---------------------------------------------------------------------------
def block_preference_phrases(answers: dict, blocks: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Build P_b(u) for each block from the answers.

    - Genre      : sentences of the ticked genres (Q5)
    - Themes     : sentences of the ticked themes (Q6) + "loved book and why" (Q2)
    - Style      : complexity (Q10) and pace (Q11) Likerts as sentences + focus (Q16)
    - Atmosphere : period (Q7) + tone Likert (Q12) as a sentence + free-text mood (Q3)
    A block without any sentence takes the ideal description (Q1): it thus
    remains measurable instead of being empty.
    """
    free = answers.get("free_text", {}) or {}
    likert = answers.get("likert", {}) or {}
    guided = answers.get("guided", {}) or {}
    P: dict[str, list[str]] = {b: [] for b in BLOCKS}

    # Genre: ticked labels -> their descriptive sentences
    P["Genre"] += [label_phrase(blocks, "Genre", g) for g in answers.get("genres", []) or []]

    # Themes: ticked labels + "loved book" free text (rich in themes)
    P["Themes"] += [label_phrase(blocks, "Themes", t) for t in answers.get("themes", []) or []]
    if (free.get("loved_book") or "").strip():
        P["Themes"].append(free["loved_book"].strip())

    # Style: Likerts converted into sentences + character/plot guided question
    for key in ("complexity", "pace"):
        s = likert_to_sentence(key, likert.get(key))
        if s:
            P["Style"].append(s)
    focus = guided.get("focus")
    if focus in FOCUS_SENTENCES:
        P["Style"].append(FOCUS_SENTENCES[focus])

    # Atmosphere: chosen period (P2 label) + tone + free text
    period = str(answers.get("period") or "").strip()
    if period.lower() not in NO_PREFERENCE:
        lab = PERIOD_TO_LABEL.get(period.lower(), period)
        P["Atmosphere"].append(label_phrase(blocks, "Atmosphere", lab))
    s = likert_to_sentence("darkness", likert.get("darkness"))
    if s:
        P["Atmosphere"].append(s)
    if (free.get("atmosphere") or "").strip():
        P["Atmosphere"].append(free["atmosphere"].strip())

    # Fallback: an empty block takes the ideal description
    ideal = (free.get("ideal_book") or "").strip()
    for b in BLOCKS:
        if not P[b] and ideal:
            P[b] = [ideal]
    return P


def block_weights(answers: dict) -> dict[str, float]:
    """W_b = importance Likert (1-5) of each block; 3 by default."""
    likert = answers.get("likert", {}) or {}
    keys = {"Genre": "w_genre", "Themes": "w_themes", "Style": "w_style", "Atmosphere": "w_atmosphere"}
    out = {}
    for b, k in keys.items():
        v = likert.get(k, config.DEFAULT_BLOCK_WEIGHT)
        out[b] = float(min(max(float(v if v is not None else config.DEFAULT_BLOCK_WEIGHT), 1), 5))
    return out


# ---------------------------------------------------------------------------
# 3. Global query q_global
# ---------------------------------------------------------------------------
def global_query_texts(answers: dict) -> list[str]:
    """Free-text answers describing the book being looked for (Q1, Q2, Q3)."""
    free = answers.get("free_text", {}) or {}
    return [free[k].strip() for k in ("ideal_book", "loved_book", "atmosphere")
            if (free.get(k) or "").strip()]


def encode_global_query(texts: list[str], engine, how: str = config.QUERY_COMBINATION) -> np.ndarray | None:
    """Encode the free-text answers into ONE vector.

    - "mean"   : average of the embeddings of each answer (each answer counts equally,
                 and none is truncated by MiniLM's 256-token limit);
    - "concat" : a single concatenated text (the model sees the full context,
                 but a long answer crushes the others and the whole may be truncated).
    Notebook 04 compares the two.
    """
    if not texts:
        return None
    if how == "concat":
        return engine.encode([" ".join(texts)])[0]
    return l2_normalize(engine.encode(texts).mean(axis=0))


def author_centroid(books: pd.DataFrame | None, engine, name: str | None) -> tuple[np.ndarray | None, str | None]:
    """Centroid of the embeddings of the favourite author's books (P1's fuzzy matching)."""
    if books is None or not (name or "").strip():
        return None, None
    from src.data_loader import find_author, get_books_by_author

    found = get_books_by_author(books, name)
    if found.empty:
        return None, None
    # books is indexed 0..N-1 in the same order as engine.book_matrix()
    vec = l2_normalize(engine.book_matrix()[found.index.to_numpy()].mean(axis=0))
    return vec, find_author(books, name)


# ---------------------------------------------------------------------------
# 4. Profile summary (template)
# ---------------------------------------------------------------------------
def _top_labels(aff: dict[str, float], n: int) -> list[str]:
    return [k for k, _ in sorted(aff.items(), key=lambda kv: -kv[1])[:n]]


def profile_summary(answers: dict, affinities: dict[str, dict[str, float]],
                    weights: dict[str, float], lambda_pop: float, author: str | None) -> str:
    """Readable summary of the reader profile, generated from a template (no LLM)."""
    genres = answers.get("genres") or _top_labels(affinities.get("Genre", {}), 2)
    themes = answers.get("themes") or _top_labels(affinities.get("Themes", {}), 2)
    style = _top_labels(affinities.get("Style", {}), 1)
    atmo = _top_labels(affinities.get("Atmosphere", {}), 2)
    most = max(weights, key=weights.get)
    parts = [
        f"You are looking for {', '.join(genres[:3]) or 'any genre'} books",
        f"exploring {', '.join(themes[:3]).lower() or 'varied themes'}",
    ]
    s = " ".join(parts) + "."
    if style:
        s += f" Your ideal style looks like: {style[0].lower()}."
    if atmo:
        s += f" Preferred atmosphere: {' and '.join(a.lower() for a in atmo)}."
    s += f" What matters most to you: {most.lower()}."
    if lambda_pop >= 0.15:
        s += " You enjoy popular, well-rated books."
    elif lambda_pop <= 0.05:
        s += " You like discovering lesser-known gems."
    if author:
        s += f" Favourite author: {author}."
    aud = answers.get("audience")
    if aud and aud != "Any":
        s += f" Audience: {aud}."
    return s


# ---------------------------------------------------------------------------
# Review fixes: automatic fiction_only and "requested" labels
# ---------------------------------------------------------------------------
# Style / atmosphere labels that each Likert answer REQUESTS or CONTRADICTS.
# E.g. pace = 1 or 2 requests "Slow and contemplative" and contradicts "Fast-paced page-turner".
LIKERT_LABELS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    #               (requested labels,                    contradicted labels)
    "pace": {
        "low":  (("Slow and contemplative",),          ("Fast-paced page-turner",)),
        "high": (("Fast-paced page-turner",),          ("Slow and contemplative",)),
    },
    "complexity": {
        "low":  (("Simple & accessible",),             ("Dense literary prose",)),
        "high": (("Dense literary prose",),            ("Simple & accessible",)),
    },
    "darkness": {
        "low":  (("Cozy & heartwarming",),             ("Dark & unsettling", "Dark & gritty voice", "Urban & gritty")),
        "high": (("Dark & unsettling", "Melancholic"), ("Cozy & heartwarming", "Whimsical & magical")),
    },
}
FOCUS_LABELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Characters": (("Character-driven",),     ("Plot-driven & twisty",)),
    "Plot":       (("Plot-driven & twisty",), ("Character-driven",)),
}


def auto_fiction_only(answers: dict, blocks: dict[str, dict[str, str]] | None = None) -> bool | None:
    """Effective fiction_only: the explicit answer, otherwise True if ONLY fiction genres are ticked.

    - explicit fiction_only (True / False): it is respected.
    - at least one genre of config.FICTION_GENRES and none of config.NONFICTION_GENRES: True.
    - otherwise: None (no filter).
    """
    explicit = answers.get("fiction_only")
    if explicit is not None:
        return explicit
    genres = [resolve_label(blocks or {}, "Genre", g) or g for g in answers.get("genres") or []]
    if any(g in config.NONFICTION_GENRES for g in genres):
        return None
    if any(g in config.FICTION_GENRES for g in genres):
        return True
    return None


def requested_labels(answers: dict, blocks: dict[str, dict[str, str]],
                     affinities: dict[str, dict[str, float]],
                     n_top: int = config.N_TOP_AFFINITIES) -> dict[str, list[str]]:
    """Labels the explanation is allowed to cite, per block.

    = ticked labels (genres, themes, period) + labels inferred from the Likerts and the focus
      + top n_top affinities of the block,
      MINUS the labels that contradict an answer (e.g. "Fast-paced" if pace = 2).
    """
    likert = answers.get("likert", {}) or {}
    guided = answers.get("guided", {}) or {}
    asked: dict[str, list[str]] = {b: [] for b in BLOCKS}
    banned: set[str] = set()

    asked["Genre"] += [resolve_label(blocks, "Genre", g) or g for g in answers.get("genres") or []]
    asked["Themes"] += [resolve_label(blocks, "Themes", t) or t for t in answers.get("themes") or []]
    period = str(answers.get("period") or "").strip()
    if period and period.lower() not in NO_PREFERENCE:
        asked["Atmosphere"].append(PERIOD_TO_LABEL.get(period.lower(), period))

    def _add(block: str, pair: tuple[tuple[str, ...], tuple[str, ...]]) -> None:
        asked[block] += list(pair[0])
        banned.update(pair[1])

    for key, block in (("pace", "Style"), ("complexity", "Style"), ("darkness", "Atmosphere")):
        v = likert.get(key)
        if v is None or v == config.LIKERT_NEUTRAL:
            continue
        _add(block, LIKERT_LABELS[key]["low" if v < config.LIKERT_NEUTRAL else "high"])
    if guided.get("focus") in FOCUS_LABELS:
        _add("Style", FOCUS_LABELS[guided["focus"]])

    out: dict[str, list[str]] = {}
    for b in BLOCKS:
        aff = affinities.get(b, {})
        top = sorted(aff, key=aff.get, reverse=True)[:n_top]
        labels = list(dict.fromkeys(asked[b] + top))      # order kept, no duplicates
        out[b] = [lab for lab in labels if lab not in banned]
    return out


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
def build_profile(answers: dict, engine, books: pd.DataFrame | None = None,
                  blocks: dict[str, dict[str, str]] | None = None,
                  how: str = config.QUERY_COMBINATION) -> UserProfile:
    """Turn answers.json into a UserProfile (see the module docstring).

    Args:
        answers: answers in the answers.json format (README, section 2).
        engine: object following the EmbeddingEngine contract (P3) or LocalEngine.
        books: catalogue indexed 0..N-1, aligned with engine.book_matrix() (for the favourite author).
        blocks: blocks.json taxonomy (P2). Empty = the labels themselves are encoded.
        how: "mean" or "concat" to combine the free-text answers.
    """
    blocks = blocks or {}
    free = answers.get("free_text", {}) or {}
    guided = answers.get("guided", {}) or {}
    likert = answers.get("likert", {}) or {}

    # Step 1: preference sentences per block and their embeddings
    phrases = block_preference_phrases(answers, blocks)
    block_vecs = {b: engine.encode(phrases[b]) for b in BLOCKS}

    # Step 2: global query = free-text answers (+ favourite author)
    texts = global_query_texts(answers)
    q_text = encode_global_query(texts, engine, how)
    if q_text is None:  # no free-text answer: fall back on all the block sentences
        allv = np.vstack([v for v in block_vecs.values() if len(v)]) if any(len(v) for v in block_vecs.values()) else None
        q_text = l2_normalize(allv.mean(0)) if allv is not None else np.zeros(engine.encode(["book"]).shape[1], np.float32)
    centroid, author = author_centroid(books, engine, answers.get("favorite_author"))
    if centroid is not None:
        q_global = l2_normalize(config.W_TEXT_QUERY * q_text + config.W_AUTHOR_CENTROID * centroid)
    else:
        q_global = q_text

    # Step 3: "to avoid" text (negations removed)
    avoid_raw = (free.get("avoid") or "").strip()
    avoid_txt = clean_avoid_text(avoid_raw)
    q_avoid = engine.encode([avoid_txt])[0] if avoid_txt else None

    # Step 4: block weights and popularity bonus
    weights = block_weights(answers)
    lam = lambda_from_likert(likert.get("popularity"))

    # Step 5: user ↔ label affinities (profile radar)
    # affinity(label) = max over the P_b sentences and q_global of the cosine with the label's sentence
    affinities: dict[str, dict[str, float]] = {}
    for b in BLOCKS:
        labels, L = engine.label_matrix(b)
        if not labels:
            affinities[b] = {}
            continue
        Q = np.vstack([block_vecs[b], q_global[None, :]]) if len(block_vecs[b]) else q_global[None, :]
        sims = (Q @ L.T).max(axis=0)
        affinities[b] = {lab: float(np.clip(s, 0.0, 1.0)) for lab, s in zip(labels, sims)}

    # Step 6: hard constraints (applied by data_loader.apply_constraints)
    excluded = list(answers.get("excluded_authors") or [])
    if guided.get("exclude_favorite_author") and answers.get("favorite_author"):
        excluded.append(answers["favorite_author"])
    constraints = {
        "audience": answers.get("audience") or None,
        "formats": list(answers.get("formats") or []) or None,
        "excluded_authors": excluded or None,
        "period": answers.get("publication_period") or None,   # PUBLICATION year (optional)
        # True / False / None; automatically True if only fiction genres are ticked
        "fiction_only": auto_fiction_only(answers, blocks),
        "fiction_only_auto": answers.get("fiction_only") is None and auto_fiction_only(answers, blocks) is True,
        # No series information in the dataset: preference recorded but not filterable
        "standalone_only": bool(guided.get("standalone_only", False)),
        "relaxed": [],                                           # filters relaxed if 0 results
    }

    # Step 7: profile texts and vectors (UserProfile contract)
    query_texts = {"global": " ".join(texts)}
    query_texts.update({b: " | ".join(phrases[b]) for b in BLOCKS})
    query_texts["avoid"] = avoid_txt
    query_vecs: dict[str, np.ndarray] = {"global": q_global}
    query_vecs.update(block_vecs)
    if q_avoid is not None:
        query_vecs["avoid"] = q_avoid

    summary = profile_summary(answers, affinities, weights, lam, author)
    return UserProfile(
        query_texts=query_texts, query_vecs=query_vecs, weights=weights,
        block_affinities=affinities, constraints=constraints, summary=summary,
        block_phrases=phrases, lambda_pop=lam, favorite_author=author,
        requested_labels=requested_labels(answers, blocks, affinities),
    )


def example_answers() -> dict[str, Any]:
    """Example of answers following the answers.json contract (README, section 2)."""
    return {
        "free_text": {
            "ideal_book": "I enjoy psychological novels exploring identity and moral dilemmas, "
                          "with complex characters who hide dark secrets.",
            "loved_book": "I loved The Secret History because of the dark academic atmosphere "
                          "and the group of students drawn into a crime.",
            "atmosphere": "Something melancholic, rainy, set in an old university",
            "avoid": "No romance-centered plots, no gore",
        },
        "genres": ["Literary Fiction", "Thriller & Suspense"],
        "themes": ["Identity & self-discovery", "Morality & guilt", "Friendship"],
        "period": "Contemporary",
        "audience": "Adult",
        "formats": ["Paperback", "Hardback"],
        "favorite_author": "Donna Tartt",
        "guided": {"focus": "Characters", "standalone_only": False, "exclude_favorite_author": True},
        "likert": {"complexity": 4, "pace": 2, "darkness": 4, "popularity": 3,
                   "w_genre": 4, "w_themes": 5, "w_style": 3, "w_atmosphere": 4},
    }
