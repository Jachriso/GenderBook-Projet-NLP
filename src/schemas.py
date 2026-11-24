"""
Data structures shared between the recommendation engine (P4) and the app (P5).

This file is the Python CONTRACT of section 4.4 of PLAN_PROJET.md:
- ``UserProfile``     : what the system understood about the reader (output of profile.build_profile)
- ``Recommendation``  : a recommended book with its score, its criteria and its explanation

All the contract fields are present, in the order of the plan. A few EXTRA
fields were added at the end, with a default value: they break nothing on the
app side (they can be ignored) but make the score fully traceable
("How was this score computed?").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# The 4 blocks of the taxonomy (same order everywhere: radar, bars, report)
BLOCKS: tuple[str, ...] = ("Genre", "Themes", "Style", "Atmosphere")


@dataclass
class UserProfile:
    """Semantic profile of the reader, built from answers.json.

    Contract attributes:
        query_texts: encoded texts, keys "global", "Genre", "Themes", "Style",
            "Atmosphere", "avoid". For a block, the preference sentences are
            joined by " | " (the detailed list is in ``block_phrases``).
        query_vecs: L2-normalised vectors. "global" and "avoid": shape (d,);
            a block: shape (m, d), one row per preference sentence.
        weights: importance W_b of each block (Likert 1-5).
        block_affinities: block -> label -> user affinity ∈ [0, 1]
            (used by the "reader profile" radar).
        constraints: hard filters (audience, formats, excluded authors...).
        summary: English summary of the profile, generated from a template.

    Added fields (optional):
        block_phrases: block -> list of preference sentences P_b(u).
        lambda_pop: weight λ of the popularity bonus, inferred from the "popularity" Likert.
        favorite_author: name of the favourite author found in the catalogue (or None).
        requested_labels: block -> labels the user asked for (ticked, inferred
            from the Likerts or top 3 affinities); only those are cited in the explanation.
    """

    query_texts: dict[str, str]
    query_vecs: dict[str, np.ndarray]
    weights: dict[str, float]
    block_affinities: dict[str, dict[str, float]]
    constraints: dict[str, Any]
    summary: str
    # --- fields added by P4 (default values => backward compatible) ---
    block_phrases: dict[str, list[str]] = field(default_factory=dict)
    lambda_pop: float = 0.0
    favorite_author: str | None = None
    requested_labels: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable version (the vectors are left out, too heavy)."""
        d = asdict(self)
        d["query_vecs"] = {k: list(np.shape(v)) for k, v in self.query_vecs.items()}
        return d


@dataclass
class Recommendation:
    """A recommended book, with everything needed to display and explain it.

    Contract attributes:
        rank: position in the top (1 = best).
        book_id, title, authors, image_url: display metadata.
        final_score: final score ∈ [0, 1].
        match_label: "Excellent" / "Good" / "Partial" (thresholds of config.MATCH_THRESHOLDS).
        block_scores: CALIBRATED scores ∈ [0, 1] of each component:
            "S_desc", "Genre", "Themes", "Style", "Atmosphere".
        contributions: share of each component in the final score
            ("Description", "Genre", ..., "Popularity", negative "Avoid penalty").
            The sum of the contributions = final_score (stacked bars).
        coverage: coverage score Σ W_b·S̃_b / Σ W_b ∈ [0, 1].
        matched_labels: block -> [(book label, similarity with the request)].
        best_sentence / best_sentence_score: sentence of the blurb closest
            to the user's request, and its cosine.
        explanation: English explanation generated from a template (explain.py).

    Added fields (optional):
        raw_scores: RAW cosines before calibration (transparency).
        popularity: popularity of the book ∈ [0, 1] (P1 column).
        avoid_similarity: cos(q_avoid, book), 0 if there is no "to avoid" text.
        main_author: main author (useful for the diversity rule).
    """

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
    matched_labels: dict[str, list[tuple[str, float]]]
    best_sentence: str
    best_sentence_score: float
    explanation: str
    # --- fields added by P4 ---
    raw_scores: dict[str, float] = field(default_factory=dict)
    popularity: float = 0.0
    avoid_similarity: float = 0.0
    main_author: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable version (to save the results)."""
        return asdict(self)
