"""
stub_recommender.py: TEMPORARY stand-in for src/recommender.recommend (Person 5).

This is NOT the project's recommendation engine (that is P4's work).
It is only used to develop and test the app while src/recommender.py does not exist.
As soon as P4's `recommend(answers, top_k)` is available, the app uses it
automatically (see get_recommend_fn in streamlit_app.py) and this file is no longer used.

It follows the output contract EXACTLY (PLAN §4.4) so that the interface and the
visualisations are already wired to the right fields. To keep the demo credible,
it uses real data whenever it exists:
    - with an EmbeddingEngine (P3): cosine retrieval query ↔ synopsis + best_sentence;
    - always: P2's label_scores (books_enriched.parquet) for the per-block scores;
    - always: P1's apply_constraints for the hard filters.
The formula is a simplified version of PLAN §5 (no sentences derived from the Likert answers,
rough min/max calibration on the candidates): the numbers it shows must NOT be quoted in the report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# 1. Contract dataclasses: use P4's ones if they already exist
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on P4's progress
    from src.schemas import Recommendation, UserProfile  # type: ignore
except ImportError:  # exact copy of the contract in PLAN §4.4

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
# 2. Preferences per block (simplified version of profile.py)
# --------------------------------------------------------------------------- #

def preferred_labels(answers: Mapping[str, Any]) -> dict[str, list[str]]:
    """Labels the user "asks for" in each block.

    Genre / Themes: ticked boxes. Style: derived from the Likert scales and the guided question.
    Atmosphere: chosen era + tone. (P4 does better: encoded sentences rather than labels.)
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
    """(N, len(labels)): book ↔ label cosines taken from label_scores (P2)."""
    out = np.zeros((len(books), len(labels)), dtype=np.float32)
    for i, ls in enumerate(books["label_scores"].tolist()):
        if isinstance(ls, str):
            ls = json.loads(ls)
        d = (ls or {}).get(block, {})
        out[i] = [float(d.get(l, 0.0)) for l in labels]
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    """Rough calibration to [0, 1] (P4 uses the catalogue's p5/p95 percentiles)."""
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(x, 0.5)


def _match_label(score: float) -> str:
    """Same labels as P4: "Excellent" / "Good" / "Partial"."""
    if score >= config.MATCH_THRESHOLDS["Excellent"]:
        return "Excellent"
    if score >= config.MATCH_THRESHOLDS["Good"]:
        return "Good"
    return "Partial"


# --------------------------------------------------------------------------- #
# 3. recommend(): same signature as P4 (+ optional arguments for the app)
# --------------------------------------------------------------------------- #

def recommend(answers: dict, top_k: int = 3, engine: Any = None,
              books: pd.DataFrame | None = None) -> tuple[UserProfile, list[Recommendation]]:
    """TEMPORARY recommendations that follow the contract. See the warning at the top of this file."""
    from src.data_loader import apply_constraints, find_author, load_books

    # Step 0: catalogue (the engine's one if available, to keep the indices aligned)
    if books is None:
        books = engine.books if engine is not None else load_books()
    books = books.reset_index(drop=True)
    n = len(books)
    ft = answers.get("free_text", {})
    lik = answers.get("likert", {})
    guided = answers.get("guided", {})

    # Step 1: global query (Q1 + Q2 + Q3) and "avoid" query
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

    # Step 2: hard filters (P1)
    favourite = answers.get("favorite_author") or ""
    excluded = [favourite] if favourite and guided.get("exclude_favorite_author") else None
    mask = apply_constraints(books, audience=answers.get("audience"), formats=answers.get("formats"),
                             excluded_authors=excluded).to_numpy()
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        return _profile(answers, global_text, avoid_text, q_vecs, {}, engine, mask), []
    if engine is not None:                       # top-300 retrieval by synopsis similarity
        cand = cand[np.argsort(-s_desc[cand])][: config.RETRIEVAL_TOP_K]

    # Step 3: block score = mean cosine (label_scores) over the requested labels
    prefs = preferred_labels(answers)
    weights = {"Genre": lik.get("w_genre", 3), "Themes": lik.get("w_themes", 3),
               "Style": lik.get("w_style", 3), "Atmosphere": lik.get("w_atmosphere", 3)}
    raw_block: dict[str, np.ndarray] = {}
    sub = books.iloc[cand]
    for b in BLOCKS:
        if prefs[b]:
            raw_block[b] = _label_score_matrix(sub, b, prefs[b]).mean(axis=1)
    cal_block = {b: _minmax(v) for b, v in raw_block.items()}
    active = [b for b in BLOCKS if b in cal_block]          # block without a preference = ignored
    w_sum = sum(weights[b] for b in active) or 1.0
    coverage = sum(weights[b] * cal_block[b] for b in active) / w_sum if active else np.zeros(len(cand))

    # Step 4: final score (shape of PLAN §5)
    desc_cal = _minmax(s_desc[cand]) if engine is not None else np.zeros(len(cand))
    alpha = config.ALPHA_DESC if engine is not None else 0.0
    beta = config.BETA_COVERAGE if engine is not None else config.ALPHA_DESC + config.BETA_COVERAGE
    lam = config.LAMBDA_POP_MAX * (lik.get("popularity", 3) - 1) / 4      # same direction as P4: 5 = best-sellers
    pop = books["popularity"].to_numpy(dtype=float)[cand]
    pen = config.MU_AVOID * np.maximum(0, s_avoid[cand] - config.TAU_AVOID) if avoid_text else np.zeros(len(cand))
    denom = (alpha + beta + lam) or 1.0                                   # P4's normalisation
    final = np.clip((alpha * desc_cal + beta * coverage + lam * pop) / denom - pen, 0, 1)

    # Step 5: top-k with diversity (at most 1 book per main author)
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
    """For each block: the 2 strongest labels (those requested by the user, otherwise the book's own)."""
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
    """Template-based explanation (simple version; P4 writes the final one in explain.py)."""
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
    """Reader profile: affinity per label (radar) + text summary."""
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
        # A label explicitly chosen by the user gets a bonus (they asked for it)
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
