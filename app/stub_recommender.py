"""
stub_recommender.py : remplaçant PROVISOIRE de src/recommender.recommend (Personne 5).

⚠️ Ce n'est PAS le moteur de recommandation du projet (c'est le travail de P4).
Il sert uniquement à développer et tester l'application tant que src/recommender.py
n'existe pas. Dès que P4 livre `recommend(answers, top_k)`, l'app l'utilise
automatiquement (voir get_recommend_fn dans streamlit_app.py) et ce fichier ne sert plus.

Il respecte EXACTEMENT le contrat de sortie (PLAN §4.4) pour que l'interface et les
visualisations soient déjà branchées sur les bons champs. Pour que la démo soit crédible,
il utilise de vraies données quand elles existent :
    - avec un EmbeddingEngine (P3) : retrieval par cosinus requête ↔ résumé + best_sentence ;
    - toujours : les label_scores de P2 (books_enriched.parquet) pour les scores par bloc ;
    - toujours : apply_constraints de P1 pour les filtres durs.
La formule est une version simplifiée de PLAN §5 (pas de phrases issues du Likert, calibration
grossière min/max des candidats) : les chiffres affichés ne doivent PAS être cités dans le rapport.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# 1. Dataclasses du contrat : on prend celles de P4 si elles existent déjà
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - dépend de l'avancement de P4
    from src.schemas import Recommendation, UserProfile  # type: ignore
except ImportError:  # copie conforme du contrat PLAN §4.4

    @dataclass
    class UserProfile:  # type: ignore[no-redef]
        query_texts: dict[str, str]
        query_vecs: dict[str, np.ndarray]
        weights: dict[str, float]
        block_affinities: dict[str, dict[str, float]]
        constraints: dict
        summary: str

    @dataclass
    class Recommendation:  # type: ignore[no-redef]
        rank: int
        book_id: str
        title: str
        authors: str
        image_url: str
        final_score: float
        match_label: str
        block_scores: dict[str, float]
        contributions: dict[str, float]
        coverage: float
        matched_labels: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
        best_sentence: str = ""
        best_sentence_score: float = 0.0
        explanation: str = ""


BLOCKS = tuple(config.BLOCKS)
LABEL_COLUMNS = {"Genre": "genres", "Themes": "themes", "Style": "styles", "Atmosphere": "atmospheres"}


# --------------------------------------------------------------------------- #
# 2. Préférences par bloc (version simplifiée de profile.py)
# --------------------------------------------------------------------------- #

def preferred_labels(answers: Mapping[str, Any]) -> dict[str, list[str]]:
    """Labels que l'utilisateur « demande » dans chaque bloc.

    Genre / Themes : cases cochées. Style : déduit du Likert et de la question guidée.
    Atmosphere : époque choisie + tonalité. (P4 fera mieux : phrases encodées, pas des labels.)
    """
    lik = answers.get("likert", {})
    style: list[str] = []
    if lik.get("complexity", 3) >= 4:
        style.append("Dense literary prose")
    elif lik.get("complexity", 3) <= 2:
        style.append("Simple & accessible")
    if lik.get("pace", 3) >= 4:
        style.append("Fast-paced page-turner")
    elif lik.get("pace", 3) <= 2:
        style.append("Slow and contemplative")
    focus = answers.get("guided", {}).get("focus")
    if focus == "Characters":
        style.append("Character-driven")
    elif focus == "Plot":
        style.append("Plot-driven & twisty")

    atmo: list[str] = []
    if answers.get("period") and answers.get("period") != "No preference":
        atmo.append(answers["period"])
    if lik.get("darkness", 3) >= 4:
        atmo.append("Dark & unsettling")
    elif lik.get("darkness", 3) <= 2:
        atmo.append("Cozy & heartwarming")
    return {"Genre": list(answers.get("genres", [])), "Themes": list(answers.get("themes", [])),
            "Style": style, "Atmosphere": atmo}


def _label_score_matrix(books: pd.DataFrame, block: str, labels: list[str]) -> np.ndarray:
    """(N, len(labels)) : cosinus livre ↔ label tirés de label_scores (P2)."""
    out = np.zeros((len(books), len(labels)), dtype=np.float32)
    for i, ls in enumerate(books["label_scores"].tolist()):
        if isinstance(ls, str):
            ls = json.loads(ls)
        d = (ls or {}).get(block, {})
        out[i] = [float(d.get(l, 0.0)) for l in labels]
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    """Calibration grossière sur [0, 1] (P4 utilise les percentiles p5/p95 du catalogue)."""
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(x, 0.5)


def _match_label(score: float) -> str:
    """Mêmes libellés que P4 : "Excellent" / "Good" / "Partial"."""
    if score >= config.MATCH_THRESHOLDS["Excellent"]:
        return "Excellent"
    if score >= config.MATCH_THRESHOLDS["Good"]:
        return "Good"
    return "Partial"


# --------------------------------------------------------------------------- #
# 3. recommend() : même signature que P4 (+ arguments optionnels pour l'app)
# --------------------------------------------------------------------------- #

def recommend(answers: dict, top_k: int = 3, engine: Any = None,
              books: pd.DataFrame | None = None) -> tuple[UserProfile, list[Recommendation]]:
    """Recommandations PROVISOIRES conformes au contrat. Voir l'avertissement en tête de fichier."""
    from src.data_loader import apply_constraints, find_author, load_books

    # Étape 0 : catalogue (celui du moteur s'il existe, pour garder l'alignement des indices)
    if books is None:
        books = engine.books if engine is not None else load_books()
    books = books.reset_index(drop=True)
    n = len(books)
    ft = answers.get("free_text", {})
    lik = answers.get("likert", {})
    guided = answers.get("guided", {})

    # Étape 1 : requête globale (Q1 + Q2 + Q3) et requête « à éviter »
    global_text = " ".join(t for t in (ft.get("ideal_book"), ft.get("loved_book"), ft.get("atmosphere")) if t)
    avoid_text = (ft.get("avoid") or "").strip()
    q_vecs: dict[str, np.ndarray] = {}
    s_desc = np.zeros(n, dtype=np.float32)
    s_avoid = np.zeros(n, dtype=np.float32)
    if engine is not None and global_text:
        q_vecs["global"] = engine.encode([global_text])[0]
        s_desc = engine.book_matrix() @ q_vecs["global"]
        if avoid_text:
            q_vecs["avoid"] = engine.encode([avoid_text])[0]
            s_avoid = engine.book_matrix() @ q_vecs["avoid"]

    # Étape 2 : filtres durs (P1)
    favourite = answers.get("favorite_author") or ""
    excluded = [favourite] if favourite and guided.get("exclude_favorite_author") else None
    mask = apply_constraints(books, audience=answers.get("audience"), formats=answers.get("formats"),
                             excluded_authors=excluded).to_numpy()
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        return _profile(answers, global_text, avoid_text, q_vecs, {}, engine, mask), []
    if engine is not None:                       # retrieval top-300 par similarité au résumé
        cand = cand[np.argsort(-s_desc[cand])][: config.RETRIEVAL_TOP_K]

    # Étape 3 : score par bloc = moyenne des cosinus (label_scores) sur les labels demandés
    prefs = preferred_labels(answers)
    weights = {"Genre": lik.get("w_genre", 3), "Themes": lik.get("w_themes", 3),
               "Style": lik.get("w_style", 3), "Atmosphere": lik.get("w_atmosphere", 3)}
    raw_block: dict[str, np.ndarray] = {}
    sub = books.iloc[cand]
    for b in BLOCKS:
        if prefs[b]:
            raw_block[b] = _label_score_matrix(sub, b, prefs[b]).mean(axis=1)
    cal_block = {b: _minmax(v) for b, v in raw_block.items()}
    active = [b for b in BLOCKS if b in cal_block]          # bloc sans préférence = ignoré
    w_sum = sum(weights[b] for b in active) or 1.0
    coverage = sum(weights[b] * cal_block[b] for b in active) / w_sum if active else np.zeros(len(cand))

    # Étape 4 : score final (forme de PLAN §5)
    desc_cal = _minmax(s_desc[cand]) if engine is not None else np.zeros(len(cand))
    alpha = config.ALPHA_DESC if engine is not None else 0.0
    beta = config.BETA_COVERAGE if engine is not None else config.ALPHA_DESC + config.BETA_COVERAGE
    lam = config.LAMBDA_POP_MAX * (lik.get("popularity", 3) - 1) / 4      # même sens que P4 : 5 = best-sellers
    pop = books["popularity"].to_numpy(dtype=float)[cand]
    pen = config.MU_AVOID * np.maximum(0, s_avoid[cand] - config.TAU_AVOID) if avoid_text else np.zeros(len(cand))
    denom = (alpha + beta + lam) or 1.0                                   # normalisation de P4
    final = np.clip((alpha * desc_cal + beta * coverage + lam * pop) / denom - pen, 0, 1)

    # Étape 5 : top-k avec diversité (1 livre max par auteur principal)
    recs: list[Recommendation] = []
    seen_authors: set[str] = set()
    for j in np.argsort(-final):
        row = books.iloc[cand[j]]
        author = str(row.get("main_author") or row.get("authors_str"))
        if author in seen_authors:
            continue
        seen_authors.add(author)
        block_scores = {"S_desc": float(desc_cal[j])}
        block_scores.update({f"S_{b}": float(cal_block[b][j]) for b in active})
        contributions = {"Description": float(alpha * desc_cal[j] / denom)}
        contributions.update({b: float(beta * weights[b] * cal_block[b][j] / w_sum / denom) for b in active})
        contributions["Popularity"] = float(lam * pop[j] / denom)
        contributions["Avoid penalty"] = float(-pen[j])
        matched = _matched_labels(row, prefs)
        best, best_score = ("", 0.0)
        if engine is not None and "global" in q_vecs:
            best, best_score = engine.best_sentence(q_vecs["global"], row["description"])
        rec = Recommendation(
            rank=len(recs) + 1, book_id=str(row["book_id"]), title=str(row["title"]),
            authors=str(row["authors_str"]), image_url=str(row.get("image_url") or ""),
            final_score=float(final[j]), match_label=_match_label(float(final[j])),
            block_scores=block_scores, contributions=contributions,
            coverage=float(coverage[j]) if active else 0.0, matched_labels=matched,
            best_sentence=best, best_sentence_score=float(best_score), explanation="",
        )
        rec.explanation = _explain(rec, answers)
        recs.append(rec)
        if len(recs) >= top_k:
            break
    return _profile(answers, global_text, avoid_text, q_vecs, weights, engine, mask), recs


def _matched_labels(row: pd.Series, prefs: dict[str, list[str]]) -> dict[str, list[tuple[str, float]]]:
    """Pour chaque bloc : les 2 labels (demandés par l'utilisateur, sinon ceux du livre) les plus forts."""
    ls = row["label_scores"]
    if isinstance(ls, str):
        ls = json.loads(ls)
    out: dict[str, list[tuple[str, float]]] = {}
    for b in BLOCKS:
        scores = (ls or {}).get(b, {})
        own = list(row.get(LABEL_COLUMNS[b]) or [])
        wanted = [l for l in prefs.get(b, []) if l in own] or own
        pairs = sorted(((l, float(scores.get(l, 0.0))) for l in wanted), key=lambda p: -p[1])[:2]
        if pairs:
            out[b] = pairs
    return out


def _explain(rec: Recommendation, answers: Mapping[str, Any]) -> str:
    """Explication par template (version simple ; P4 écrira la version finale dans explain.py)."""
    parts = []
    if rec.best_sentence:
        parts.append(f"Its synopsis is semantically close to what you described "
                     f"(closest sentence, similarity {rec.best_sentence_score:.2f}).")
    prefs = preferred_labels(answers)
    for b, word in (("Themes", "themes"), ("Genre", "genre"), ("Atmosphere", "atmosphere"), ("Style", "style")):
        if rec.matched_labels.get(b):
            labs = ", ".join(f"{l} ({s:.2f})" for l, s in rec.matched_labels[b])
            if any(l in prefs[b] for l, _ in rec.matched_labels[b]):
                parts.append(f"It matches your {word}: {labs}.")
            else:
                parts.append(f"Its {word}: {labs}.")
    parts.append(f"Overall preference coverage: {rec.coverage:.0%}.")
    return " ".join(parts)


def _profile(answers, global_text, avoid_text, q_vecs, weights, engine, mask) -> UserProfile:
    """Profil lecteur : affinités par label (radar) + résumé texte."""
    from src.enrichment import load_blocks

    blocks = load_blocks()
    prefs = preferred_labels(answers)
    affinities: dict[str, dict[str, float]] = {}
    for b in BLOCKS:
        labels = list(blocks[b])
        if engine is not None and "global" in q_vecs:
            names, mat = engine.label_matrix(b)
            sims = dict(zip(names, (mat @ q_vecs["global"]).tolist()))
        else:
            sims = {l: 0.0 for l in labels}
        # Un label explicitement choisi par l'utilisateur reçoit un bonus (il l'a demandé)
        affinities[b] = {l: float(sims.get(l, 0.0)) + (0.3 if l in prefs[b] else 0.0) for l in labels}
    top = {b: [l for l, _ in sorted(affinities[b].items(), key=lambda kv: -kv[1])[:2]] for b in BLOCKS}
    summary = (f"You are looking for {', '.join(top['Genre']).lower()} books exploring "
               f"{' and '.join(top['Themes']).lower()}, with a {top['Style'][0].lower()} style and a "
               f"{top['Atmosphere'][0].lower()} atmosphere.")
    return UserProfile(
        query_texts={"global": global_text, "avoid": avoid_text,
                     **{b: "; ".join(prefs[b]) for b in BLOCKS}},
        query_vecs=q_vecs,
        weights={b: float(v) for b, v in (weights or {}).items()} or
                {"Genre": answers["likert"]["w_genre"], "Themes": answers["likert"]["w_themes"],
                 "Style": answers["likert"]["w_style"], "Atmosphere": answers["likert"]["w_atmosphere"]},
        block_affinities=affinities,
        constraints={"audience": answers.get("audience"), "formats": answers.get("formats"),
                     "excluded_authors": [answers["favorite_author"]]
                     if answers.get("guided", {}).get("exclude_favorite_author") else [],
                     "n_candidates": int(np.asarray(mask).sum())},
        summary=summary,
    )
