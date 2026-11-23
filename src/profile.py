"""
Construction du PROFIL SÉMANTIQUE du lecteur à partir des réponses au questionnaire (P4).

    answers.json (P5)  ->  build_profile(answers, engine)  ->  UserProfile

Idée directrice : TOUT ce que l'utilisateur répond devient du TEXTE puis un
EMBEDDING, y compris les réponses numériques. Un Likert « complexité = 5 » n'est
pas utilisé comme un nombre mais converti en la phrase « a complex, dense,
literary and demanding narrative », encodée par SBERT et comparée aux livres.
C'est ce qui garde l'analyse sémantique même pour les questions fermées
(exigence « not purely numerical » du sujet).

Ce que produit build_profile :
1. q_global : vecteur de la demande globale = moyenne des embeddings des réponses
   libres (ideal_book, loved_book, atmosphere), mélangée avec le centroïde des
   livres de l'auteur préféré s'il est dans le catalogue (0.7 / 0.3).
2. P_b(u) : pour chaque bloc (Genre, Themes, Style, Atmosphere), la liste des
   phrases de préférence de l'utilisateur et leurs embeddings.
3. W_b : poids des blocs (Likert d'importance), λ : bonus popularité.
4. q_avoid : embedding du texte « à éviter » (négations retirées).
5. block_affinities (radar du profil) et summary (résumé par template).
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
# 1. Conversion Likert -> phrases (cœur du « semantic, not numerical »)
# ---------------------------------------------------------------------------
# Une phrase par niveau. Le niveau 3 (neutre) = pas de préférence => pas de phrase,
# pour ne pas diluer le bloc avec un texte qui ne dit rien.
LIKERT_SENTENCES: dict[str, dict[int, str]] = {
    # Complexité narrative (bloc Style)
    "complexity": {
        1: "a simple, easy and accessible story with clear, straightforward writing",
        2: "an easy-to-read, light narrative that is not too demanding",
        4: "a rich and layered narrative with some depth and complexity",
        5: "a complex, dense, literary and demanding narrative with sophisticated prose",
    },
    # Rythme (bloc Style)
    "pace": {
        1: "a slow, contemplative and quiet story that takes its time",
        2: "a gently paced, reflective story",
        4: "a well-paced story with momentum and suspense",
        5: "a fast-paced, gripping page-turner that is hard to put down",
    },
    # Tonalité légère <-> sombre (bloc Atmosphere)
    "darkness": {
        1: "a light, warm, cozy and feel-good story that makes you smile",
        2: "a mostly uplifting and gentle story",
        4: "a serious, somber story with darker moments",
        5: "a dark, disturbing and intense atmosphere",
    },
}

# Question guidée « character-driven ou plot-driven ? » (bloc Style)
FOCUS_SENTENCES: dict[str, str] = {
    "Characters": "a character-driven story with deep psychological portraits and inner lives",
    "Plot": "a plot-driven story full of twists, events and narrative surprises",
    "Both": "a story that balances rich characters and a compelling plot",
}

# Q7 « Époque du récit » -> label du bloc Atmosphere de blocks.json (P2).
# Plusieurs orthographes acceptées pour être robuste aux libellés de l'app.
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

# Mots de négation retirés du texte « à éviter » : SBERT gère mal la négation,
# « No romance » serait encodé presque comme « romance »… ce qui est justement ce
# qu'on veut pénaliser. On retire donc la négation pour garder le SUJET à éviter.
_NEGATION_RE = re.compile(
    r"\b(no|not|nothing|never|avoid(ing)?|without|don'?t|do not|i don'?t want|"
    r"i do not want|please|anything|any|too much|i hate|i dislike)\b",
    flags=re.IGNORECASE,
)


def likert_to_sentence(key: str, value: int | float | None) -> str | None:
    """Convertit une réponse Likert (1-5) en phrase descriptive, ou None si neutre/absente."""
    if value is None:
        return None
    v = int(round(float(value)))
    v = min(max(v, 1), 5)
    if v == config.LIKERT_NEUTRAL:
        return None
    return LIKERT_SENTENCES.get(key, {}).get(v)


def clean_avoid_text(text: str) -> str:
    """« No romance-centered plots, no gore » -> « romance-centered plots, gore »."""
    t = _NEGATION_RE.sub(" ", str(text or ""))
    t = re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s+", " ", t))
    return t.strip(" ,.;")


def lambda_from_likert(popularity: int | float | None) -> float:
    """Likert popularity (1 = pépites méconnues ... 5 = best-sellers) -> λ ∈ [0, LAMBDA_POP_MAX]."""
    if popularity is None:
        popularity = 3
    p = min(max(float(popularity), 1.0), 5.0)
    return round(config.LAMBDA_POP_MAX * (p - 1.0) / 4.0, 4)


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace("&", " and ")).strip()


def resolve_label(blocks: dict[str, dict[str, str]], block: str, label: str) -> str | None:
    """Retrouve le label exact de blocks.json à partir du libellé de l'app.

    Tolère la casse et les variantes courtes : « Identity » -> « Identity & self-discovery »,
    « Morality & Guilt » -> « Morality & guilt ». Retourne None si rien ne correspond.
    """
    labels = list(blocks.get(block, {}))
    if label in labels:
        return label
    key = _norm_label(label)
    for lab in labels:                       # 1. égalité sans casse ni ponctuation
        if _norm_label(lab) == key:
            return lab
    for lab in labels:                       # 2. le libellé est le début du label
        if key and _norm_label(lab).startswith(key):
            return lab
    return None


def label_phrase(blocks: dict[str, dict[str, str]], block: str, label: str) -> str:
    """Phrase descriptive d'un label (blocks.json) ; à défaut, le label lui-même (fallback)."""
    lab = resolve_label(blocks, block, label)
    return blocks[block][lab] if lab is not None else str(label)


# ---------------------------------------------------------------------------
# 2. Phrases de préférence par bloc : P_b(u)
# ---------------------------------------------------------------------------
def block_preference_phrases(answers: dict, blocks: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Construit P_b(u) pour chaque bloc à partir des réponses.

    - Genre      : phrases des genres cochés (Q5)
    - Themes     : phrases des thèmes cochés (Q6) + « livre adoré et pourquoi » (Q2)
    - Style      : Likert complexité (Q10) et rythme (Q11) en phrases + focus (Q16)
    - Atmosphere : époque (Q7) + Likert tonalité (Q12) en phrase + ambiance libre (Q3)
    Un bloc sans aucune phrase reprend la description idéale (Q1) : il reste
    ainsi mesurable au lieu d'être vide.
    """
    free = answers.get("free_text", {}) or {}
    likert = answers.get("likert", {}) or {}
    guided = answers.get("guided", {}) or {}
    P: dict[str, list[str]] = {b: [] for b in BLOCKS}

    # Genre : labels cochés -> leurs phrases descriptives
    P["Genre"] += [label_phrase(blocks, "Genre", g) for g in answers.get("genres", []) or []]

    # Thèmes : labels cochés + texte libre « livre adoré » (riche en thèmes)
    P["Themes"] += [label_phrase(blocks, "Themes", t) for t in answers.get("themes", []) or []]
    if (free.get("loved_book") or "").strip():
        P["Themes"].append(free["loved_book"].strip())

    # Style : Likert convertis en phrases + question guidée character/plot
    for key in ("complexity", "pace"):
        s = likert_to_sentence(key, likert.get(key))
        if s:
            P["Style"].append(s)
    focus = guided.get("focus")
    if focus in FOCUS_SENTENCES:
        P["Style"].append(FOCUS_SENTENCES[focus])

    # Atmosphère : époque choisie (label P2) + tonalité + texte libre
    period = str(answers.get("period") or "").strip()
    if period.lower() not in NO_PREFERENCE:
        lab = PERIOD_TO_LABEL.get(period.lower(), period)
        P["Atmosphere"].append(label_phrase(blocks, "Atmosphere", lab))
    s = likert_to_sentence("darkness", likert.get("darkness"))
    if s:
        P["Atmosphere"].append(s)
    if (free.get("atmosphere") or "").strip():
        P["Atmosphere"].append(free["atmosphere"].strip())

    # Repli : un bloc vide reprend la description idéale
    ideal = (free.get("ideal_book") or "").strip()
    for b in BLOCKS:
        if not P[b] and ideal:
            P[b] = [ideal]
    return P


def block_weights(answers: dict) -> dict[str, float]:
    """W_b = Likert d'importance (1-5) de chaque bloc ; 3 par défaut."""
    likert = answers.get("likert", {}) or {}
    keys = {"Genre": "w_genre", "Themes": "w_themes", "Style": "w_style", "Atmosphere": "w_atmosphere"}
    out = {}
    for b, k in keys.items():
        v = likert.get(k, config.DEFAULT_BLOCK_WEIGHT)
        out[b] = float(min(max(float(v if v is not None else config.DEFAULT_BLOCK_WEIGHT), 1), 5))
    return out


# ---------------------------------------------------------------------------
# 3. Requête globale q_global
# ---------------------------------------------------------------------------
def global_query_texts(answers: dict) -> list[str]:
    """Réponses libres qui décrivent le livre recherché (Q1, Q2, Q3)."""
    free = answers.get("free_text", {}) or {}
    return [free[k].strip() for k in ("ideal_book", "loved_book", "atmosphere")
            if (free.get(k) or "").strip()]


def encode_global_query(texts: list[str], engine, how: str = config.QUERY_COMBINATION) -> np.ndarray | None:
    """Encode les réponses libres en UN vecteur.

    - "mean"   : moyenne des embeddings de chaque réponse (chaque réponse compte autant,
                 et aucune n'est tronquée par la limite de 256 tokens de MiniLM) ;
    - "concat" : un seul texte concaténé (le modèle voit le contexte complet,
                 mais une réponse longue écrase les autres et le tout peut être tronqué).
    Le notebook 04 compare les deux.
    """
    if not texts:
        return None
    if how == "concat":
        return engine.encode([" ".join(texts)])[0]
    return l2_normalize(engine.encode(texts).mean(axis=0))


def author_centroid(books: pd.DataFrame | None, engine, name: str | None) -> tuple[np.ndarray | None, str | None]:
    """Centroïde des embeddings des livres de l'auteur préféré (matching flou de P1)."""
    if books is None or not (name or "").strip():
        return None, None
    from src.data_loader import find_author, get_books_by_author

    found = get_books_by_author(books, name)
    if found.empty:
        return None, None
    # books est indexé 0..N-1 dans le même ordre que engine.book_matrix()
    vec = l2_normalize(engine.book_matrix()[found.index.to_numpy()].mean(axis=0))
    return vec, find_author(books, name)


# ---------------------------------------------------------------------------
# 4. Résumé du profil (template)
# ---------------------------------------------------------------------------
def _top_labels(aff: dict[str, float], n: int) -> list[str]:
    return [k for k, _ in sorted(aff.items(), key=lambda kv: -kv[1])[:n]]


def profile_summary(answers: dict, affinities: dict[str, dict[str, float]],
                    weights: dict[str, float], lambda_pop: float, author: str | None) -> str:
    """Résumé lisible du profil lecteur, généré par template (sans LLM)."""
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
# 5. Point d'entrée
# ---------------------------------------------------------------------------
def build_profile(answers: dict, engine, books: pd.DataFrame | None = None,
                  blocks: dict[str, dict[str, str]] | None = None,
                  how: str = config.QUERY_COMBINATION) -> UserProfile:
    """Transforme answers.json en UserProfile (voir le docstring du module).

    Args:
        answers: réponses au format answers.json (PLAN_PROJET.md §4.3).
        engine: objet respectant le contrat EmbeddingEngine (P3) ou LocalEngine.
        books: catalogue indexé 0..N-1, aligné sur engine.book_matrix() (pour l'auteur préféré).
        blocks: référentiel blocks.json (P2). Vide = les labels eux-mêmes sont encodés.
        how: "mean" ou "concat" pour combiner les réponses libres.
    """
    blocks = blocks or {}
    free = answers.get("free_text", {}) or {}
    guided = answers.get("guided", {}) or {}
    likert = answers.get("likert", {}) or {}

    # Étape 1 : phrases de préférence par bloc et leurs embeddings
    phrases = block_preference_phrases(answers, blocks)
    block_vecs = {b: engine.encode(phrases[b]) for b in BLOCKS}

    # Étape 2 : requête globale = réponses libres (+ auteur préféré)
    texts = global_query_texts(answers)
    q_text = encode_global_query(texts, engine, how)
    if q_text is None:  # aucune réponse libre : on se rabat sur toutes les phrases de blocs
        allv = np.vstack([v for v in block_vecs.values() if len(v)]) if any(len(v) for v in block_vecs.values()) else None
        q_text = l2_normalize(allv.mean(0)) if allv is not None else np.zeros(engine.encode(["book"]).shape[1], np.float32)
    centroid, author = author_centroid(books, engine, answers.get("favorite_author"))
    if centroid is not None:
        q_global = l2_normalize(config.W_TEXT_QUERY * q_text + config.W_AUTHOR_CENTROID * centroid)
    else:
        q_global = q_text

    # Étape 3 : texte « à éviter » (négations retirées)
    avoid_raw = (free.get("avoid") or "").strip()
    avoid_txt = clean_avoid_text(avoid_raw)
    q_avoid = engine.encode([avoid_txt])[0] if avoid_txt else None

    # Étape 4 : poids des blocs et du bonus popularité
    weights = block_weights(answers)
    lam = lambda_from_likert(likert.get("popularity"))

    # Étape 5 : affinités utilisateur ↔ labels (radar du profil)
    # affinité(label) = max sur les phrases P_b et q_global du cosinus avec la phrase du label
    affinities: dict[str, dict[str, float]] = {}
    for b in BLOCKS:
        labels, L = engine.label_matrix(b)
        if not labels:
            affinities[b] = {}
            continue
        Q = np.vstack([block_vecs[b], q_global[None, :]]) if len(block_vecs[b]) else q_global[None, :]
        sims = (Q @ L.T).max(axis=0)
        affinities[b] = {lab: float(np.clip(s, 0.0, 1.0)) for lab, s in zip(labels, sims)}

    # Étape 6 : contraintes dures (appliquées par data_loader.apply_constraints)
    excluded = list(answers.get("excluded_authors") or [])
    if guided.get("exclude_favorite_author") and answers.get("favorite_author"):
        excluded.append(answers["favorite_author"])
    constraints = {
        "audience": answers.get("audience") or None,
        "formats": list(answers.get("formats") or []) or None,
        "excluded_authors": excluded or None,
        "period": answers.get("publication_period") or None,   # année de PUBLICATION (optionnel)
        "fiction_only": answers.get("fiction_only"),            # True / False / None
        # Pas d'information de série dans le dataset : préférence notée mais non filtrable
        "standalone_only": bool(guided.get("standalone_only", False)),
        "relaxed": [],                                           # filtres relâchés si 0 résultat
    }

    # Étape 7 : textes et vecteurs du profil (contrat UserProfile)
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
    )


def example_answers() -> dict[str, Any]:
    """Exemple de réponses conforme au contrat answers.json (PLAN_PROJET.md §4.3)."""
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
