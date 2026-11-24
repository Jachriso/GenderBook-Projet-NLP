"""
SINGLE entry point of the recommendation engine, used by the Streamlit app (P5).

    from src.recommender import recommend
    profile, recs = recommend(answers, top_k=3)

Pipeline (report, section 5):
    answers ─► build_profile ─► (1) hard filters ─► (2) top-300 retrieval by S_desc
            ─► (3) re-ranking with the full formula ─► (4) diversity (1 book / author)
            ─► (5) Recommendation (scores, contributions, labels, sentence, explanation)

Everything heavy (catalogue, SBERT model, embeddings, calibration) is
loaded ONCE by ``get_recommender()`` (singleton): later calls only
do matrix products (a few tens of ms).

Fallbacks (if the other parts are not delivered yet):
- no ``src/embeddings.py`` (P3)         -> ``fallback_engine.LocalEngine`` (on-the-fly SBERT + cache);
- no ``calibration.json`` (P3)          -> computed here from reference queries;
- no ``books_enriched`` (P2)            -> ``load_books()`` takes books_clean / books_sample,
                                            block scores are computed on the description;
- no ``blocks.json`` (P2)               -> the ticked labels are encoded as they are.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data_loader import apply_constraints, author_key, load_books
from src.explain import explain_recommendation
from src.profile import build_profile
from src.schemas import BLOCKS, Recommendation, UserProfile
from src.scoring import (compute_calibration, load_calibration, matched_labels_for_book,
                         score_books)

logger = logging.getLogger(__name__)

# Order in which the filters are relaxed if no book passes
# (excluded authors are never relaxed: it is an explicit request).
RELAX_ORDER = ("formats", "period", "fiction_only", "audience")


def load_blocks_safe() -> dict[str, dict[str, str]]:
    """P2's blocks.json, or an empty taxonomy if the file does not exist yet."""
    if config.BLOCKS_JSON.exists():
        return json.loads(config.BLOCKS_JSON.read_text(encoding="utf-8"))
    return {b: {} for b in BLOCKS}


def get_engine(books: pd.DataFrame, blocks: dict[str, dict[str, str]], encoder=None):
    """P3's EmbeddingEngine if it exists and is aligned, otherwise the fallback engine."""
    if encoder is None:
        try:
            from src.embeddings import EmbeddingEngine  # type: ignore  # delivered by P3

            engine = EmbeddingEngine()
            n_rows = engine.book_matrix().shape[0]
            if n_rows == len(books):
                return engine
            reason = f"book_matrix() a {n_rows} lignes alors que le catalogue a {len(books)} livres"
        except Exception as exc:  # module missing, model not downloaded, no Wi-Fi…
            reason = f"{type(exc).__name__}: {exc}"
        # No more silent switch: otherwise a failure during the demo would go unnoticed
        msg = (f"[GenderBook] EmbeddingEngine (P3) indisponible ({reason}). "
               "Bascule sur le moteur de secours fallback_engine.LocalEngine.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
    from src.fallback_engine import LocalEngine

    return LocalEngine(books, blocks, encoder=encoder)


class Recommender:
    """Full engine: keeps the catalogue, the embeddings and the calibration in memory.

    Args:
        books: catalogue; by default ``load_books()`` (enriched > clean > sample).
        engine: embedding engine (P3 contract); by default ``get_engine``.
        blocks: blocks.json taxonomy.
        calibration: {component: (p5, p95)}; by default calibration.json or computed.
        encoder: "sbert" / "lsa" / object with encode() for the fallback engine.
    """

    def __init__(self, books: pd.DataFrame | None = None, engine=None,
                 blocks: dict[str, dict[str, str]] | None = None,
                 calibration: dict[str, tuple[float, float]] | None = None, encoder=None):
        # Step 1: catalogue indexed 0..N-1 (the index is the position in the matrices)
        self.books = (books if books is not None else load_books()).reset_index(drop=True)
        self.non_narrative = self.non_narrative_mask(self.books)
        self.blocks = blocks if blocks is not None else load_blocks_safe()
        # Step 2: embedding engine + book matrix (disk cache)
        self.engine = engine if engine is not None else get_engine(self.books, self.blocks, encoder)
        self.book_emb = self.engine.book_matrix()
        # Step 3: embeddings of the label sentences (4 blocks)
        self.label_mats = {b: self.engine.label_matrix(b) for b in BLOCKS}
        # Step 4: calibration (P3's file, otherwise computed and saved)
        self.calibration = calibration or self._load_or_compute_calibration()

    def _load_or_compute_calibration(self) -> dict[str, tuple[float, float]]:
        """P3's calibration.json (if chosen) > P4 cache of the same model > computation (≈ 1 s)."""
        model = str(getattr(self.engine, "model_name", ""))
        if config.CALIBRATION_SOURCE == "p3":
            cal = load_calibration(config.CALIBRATION_JSON_P3)
            if cal:
                return cal
        cal = load_calibration(config.CALIBRATION_JSON, model=model)
        if cal and json.loads(config.CALIBRATION_JSON.read_text(encoding="utf-8")).get("n_books") == len(self.books):
            return cal
        return compute_calibration(self.engine, self.books, self.blocks)

    # ------------------------------------------------------------------ filters
    def _constraint_mask(self, c: dict) -> np.ndarray:
        mask = apply_constraints(
            self.books, audience=c.get("audience"), formats=c.get("formats"),
            period=c.get("period"), excluded_authors=c.get("excluded_authors"),
            fiction_only=c.get("fiction_only"),
        ).to_numpy()
        if c.get("fiction_only") is True:
            mask = mask & ~self.non_narrative
        return mask

    @staticmethod
    def non_narrative_mask(books: pd.DataFrame) -> np.ndarray:
        """Books classified as fiction that are not narratives (collections of confessions, jokes…).

        data_loader classifies "Humour Collections & Anthologies" as fiction: *My Secret:
        A PostSecret Book* therefore passed the fiction_only filter. We discard the books whose
        "fiction" categories are ALL of this kind (config.NON_NARRATIVE_CATEGORY_RE).
        """
        if "categories" not in books.columns:
            return np.zeros(len(books), dtype=bool)
        from src.data_loader import classify_category

        rx = re.compile(config.NON_NARRATIVE_CATEGORY_RE, re.IGNORECASE)

        def _only_non_narrative(cats) -> bool:
            fic = [c for c in (list(cats) if cats is not None else []) if classify_category(c) == "fiction"]
            return bool(fic) and all(rx.search(c) for c in fic)

        return books["categories"].apply(_only_non_narrative).to_numpy(dtype=bool)

    def filter_mask(self, profile: UserProfile, min_books: int) -> np.ndarray:
        """Hard filters; if too few books pass, they are relaxed step by step."""
        c = dict(profile.constraints)
        mask = self._constraint_mask(c)
        for key in RELAX_ORDER:
            if mask.sum() >= min_books:
                break
            if c.get(key) not in (None, [], False):
                c[key] = None
                profile.constraints["relaxed"].append(key)
                mask = self._constraint_mask(c)
        return mask

    # --------------------------------------------------------------- diversity
    @staticmethod
    def diversify(table: pd.DataFrame, books: pd.DataFrame, top_k: int,
                  max_per_author: int = config.MAX_PER_AUTHOR) -> list[int]:
        """Go through the sorted candidates and keep at most N books per main author."""
        counts: dict[str, int] = {}
        keep: list[int] = []
        for idx in table.index:
            a = author_key(str(books.at[idx, "main_author"] or books.at[idx, "authors_str"]))
            if counts.get(a, 0) >= max_per_author:
                continue
            counts[a] = counts.get(a, 0) + 1
            keep.append(idx)
            if len(keep) == top_k:
                break
        return keep

    # --------------------------------------------------------------- pipeline
    def score_table(self, profile: UserProfile, retrieval_k: int = config.RETRIEVAL_TOP_K,
                    **formula_kwargs) -> pd.DataFrame:
        """Steps 1 to 3: filters, retrieval, re-ranking. Returns the sorted table of candidates."""
        # (1) hard filters
        mask = self.filter_mask(profile, min_books=1)
        allowed = np.flatnonzero(mask)
        if len(allowed) == 0:
            return pd.DataFrame()
        # (2) retrieval: the retrieval_k allowed books closest to q_global
        sims = self.book_emb[allowed] @ profile.query_vecs["global"]
        k = min(retrieval_k, len(allowed))
        cand = allowed[np.argsort(-sims, kind="stable")[:k]]
        # (3) re-ranking with the full (vectorised) formula
        table = score_books(profile, self.book_emb[cand], self.books.loc[cand],
                            self.label_mats, self.calibration, **formula_kwargs)
        # stable sort: for equal scores the retrieval order is kept => deterministic
        return table.sort_values("final", ascending=False, kind="stable")

    def build_recommendation(self, rank: int, idx: int, row: pd.Series,
                             profile: UserProfile) -> Recommendation:
        """Step 5: build the Recommendation object of a book."""
        book = self.books.loc[idx]
        best_sent, best_score = self.engine.best_sentence(profile.query_vecs["global"],
                                                          book.get("description") or book["embed_text"])
        rec = Recommendation(
            rank=rank,
            book_id=str(book["book_id"]),
            title=str(book["title"]),
            authors=str(book.get("authors_str") or book.get("main_author") or ""),
            image_url=str(book.get("image_url") or ""),
            final_score=round(float(row["final"]), 4),
            match_label=str(row["match_label"]),
            block_scores={c: round(float(row[c]), 4) for c in ("S_desc", *BLOCKS)},
            contributions={c.replace("contrib_", ""): round(float(row[c]), 4)
                           for c in row.index if c.startswith("contrib_")},
            coverage=round(float(row["coverage"]), 4),
            matched_labels=matched_labels_for_book(profile, book, self.label_mats),
            best_sentence=best_sent,
            best_sentence_score=round(float(best_score), 4),
            explanation="",
            raw_scores={c: round(float(row[f"{c}_raw"]), 4) for c in ("S_desc", *BLOCKS)},
            popularity=round(float(row["popularity"]), 4),
            avoid_similarity=round(float(row["avoid_sim"]), 4),
            main_author=str(book.get("main_author") or ""),
        )
        rec.explanation = explain_recommendation(rec, profile)
        return rec

    def recommend(self, answers: dict, top_k: int = 3, return_table: bool = False,
                  **formula_kwargs):
        """Recommend top_k books. ``return_table=True`` also returns the table of candidates."""
        profile = build_profile(answers, self.engine, self.books, self.blocks)
        table = self.score_table(profile, **formula_kwargs)
        if table.empty:
            return (profile, [], table) if return_table else (profile, [])
        # (4) diversity
        chosen = self.diversify(table, self.books, top_k)
        # (5) Recommendation objects
        recs = [self.build_recommendation(r, idx, table.loc[idx], profile)
                for r, idx in enumerate(chosen, start=1)]
        return (profile, recs, table) if return_table else (profile, recs)


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    """Singleton: the catalogue and the model are only loaded once per process.

    In Streamlit, P5 can wrap it in ``st.cache_resource``.
    """
    return Recommender()


def recommend(answers: dict, top_k: int = 3) -> tuple[UserProfile, list[Recommendation]]:
    """App contract: questionnaire answers -> (profile, top_k recommendations)."""
    return get_recommender().recommend(answers, top_k=top_k)
