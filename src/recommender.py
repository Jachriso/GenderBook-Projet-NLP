"""
Point d'entrée UNIQUE du moteur de recommandation, utilisé par l'app Streamlit (P5).

    from src.recommender import recommend
    profile, recs = recommend(answers, top_k=3)

Pipeline (PLAN_PROJET.md §3) :
    answers -> build_profile -> (1) filtres durs -> (2) retrieval top-300 par S_desc
            -> (3) re-ranking par la formule complète -> (4) diversité (1 livre / auteur)
            -> (5) Recommendation (scores, contributions, labels, phrase, explication)

Tout ce qui est lourd (catalogue, modèle SBERT, embeddings, calibration) est
chargé UNE fois par ``get_recommender()`` (singleton) : les appels suivants ne
font que des produits matriciels (quelques dizaines de ms).

Fallbacks (si les autres parties ne sont pas encore livrées) :
- pas de ``src/embeddings.py`` (P3)     -> ``fallback_engine.LocalEngine`` (SBERT à la volée + cache) ;
- pas de ``calibration.json`` (P3)      -> calculé ici à partir de requêtes de référence ;
- pas de ``books_enriched`` (P2)        -> ``load_books()`` prend books_clean / books_sample,
                                            les scores de blocs se font sur la description ;
- pas de ``blocks.json`` (P2)           -> les labels cochés sont encodés tels quels.
"""
from __future__ import annotations

import json
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

# Ordre dans lequel on relâche les filtres si aucun livre ne passe
# (les auteurs exclus ne sont jamais relâchés : c'est une demande explicite).
RELAX_ORDER = ("formats", "period", "fiction_only", "audience")


def load_blocks_safe() -> dict[str, dict[str, str]]:
    """blocks.json de P2, ou un référentiel vide si le fichier n'existe pas encore."""
    if config.BLOCKS_JSON.exists():
        return json.loads(config.BLOCKS_JSON.read_text(encoding="utf-8"))
    return {b: {} for b in BLOCKS}


def get_engine(books: pd.DataFrame, blocks: dict[str, dict[str, str]], encoder=None):
    """EmbeddingEngine de P3 s'il existe et qu'il est aligné, sinon le moteur de secours."""
    if encoder is None:
        try:
            from src.embeddings import EmbeddingEngine  # type: ignore  # livré par P3

            engine = EmbeddingEngine()
            if engine.book_matrix().shape[0] == len(books):
                return engine
        except Exception:  # module absent, signature différente, cache non aligné…
            pass
    from src.fallback_engine import LocalEngine

    return LocalEngine(books, blocks, encoder=encoder)


class Recommender:
    """Moteur complet : garde en mémoire le catalogue, les embeddings et la calibration.

    Args:
        books: catalogue ; par défaut ``load_books()`` (enrichi > propre > échantillon).
        engine: moteur d'embeddings (contrat P3) ; par défaut ``get_engine``.
        blocks: référentiel blocks.json.
        calibration: {composante: (p5, p95)} ; par défaut calibration.json ou calcul.
        encoder: "sbert" / "lsa" / objet encode() pour le moteur de secours.
    """

    def __init__(self, books: pd.DataFrame | None = None, engine=None,
                 blocks: dict[str, dict[str, str]] | None = None,
                 calibration: dict[str, tuple[float, float]] | None = None, encoder=None):
        # Étape 1 : catalogue indexé 0..N-1 (l'index sert de position dans les matrices)
        self.books = (books if books is not None else load_books()).reset_index(drop=True)
        self.blocks = blocks if blocks is not None else load_blocks_safe()
        # Étape 2 : moteur d'embeddings + matrice des livres (cache disque)
        self.engine = engine if engine is not None else get_engine(self.books, self.blocks, encoder)
        self.book_emb = self.engine.book_matrix()
        # Étape 3 : embeddings des phrases de labels (4 blocs)
        self.label_mats = {b: self.engine.label_matrix(b) for b in BLOCKS}
        # Étape 4 : calibration (fichier de P3, sinon calculée et sauvegardée)
        self.calibration = calibration or self._load_or_compute_calibration()

    def _load_or_compute_calibration(self) -> dict[str, tuple[float, float]]:
        """calibration.json de P3 (si choisi) > cache P4 du même modèle > calcul (≈ 1 s)."""
        model = str(getattr(self.engine, "model_name", ""))
        if config.CALIBRATION_SOURCE == "p3":
            cal = load_calibration(config.CALIBRATION_JSON_P3)
            if cal:
                return cal
        cal = load_calibration(config.CALIBRATION_JSON, model=model)
        if cal and json.loads(config.CALIBRATION_JSON.read_text(encoding="utf-8")).get("n_books") == len(self.books):
            return cal
        return compute_calibration(self.engine, self.books, self.blocks)

    # ------------------------------------------------------------------ filtres
    def _constraint_mask(self, c: dict) -> np.ndarray:
        return apply_constraints(
            self.books, audience=c.get("audience"), formats=c.get("formats"),
            period=c.get("period"), excluded_authors=c.get("excluded_authors"),
            fiction_only=c.get("fiction_only"),
        ).to_numpy()

    def filter_mask(self, profile: UserProfile, min_books: int) -> np.ndarray:
        """Filtres durs ; si trop peu de livres passent, on relâche progressivement."""
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

    # --------------------------------------------------------------- diversité
    @staticmethod
    def diversify(table: pd.DataFrame, books: pd.DataFrame, top_k: int,
                  max_per_author: int = config.MAX_PER_AUTHOR) -> list[int]:
        """Parcourt les candidats triés et garde au plus N livres par auteur principal."""
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
        """Étapes 1 à 3 : filtres, retrieval, re-ranking. Renvoie la table triée des candidats."""
        # (1) filtres durs
        mask = self.filter_mask(profile, min_books=1)
        allowed = np.flatnonzero(mask)
        if len(allowed) == 0:
            return pd.DataFrame()
        # (2) retrieval : les retrieval_k livres autorisés les plus proches de q_global
        sims = self.book_emb[allowed] @ profile.query_vecs["global"]
        k = min(retrieval_k, len(allowed))
        cand = allowed[np.argsort(-sims, kind="stable")[:k]]
        # (3) re-ranking par la formule complète (vectorisée)
        table = score_books(profile, self.book_emb[cand], self.books.loc[cand],
                            self.label_mats, self.calibration, **formula_kwargs)
        # tri stable : à score égal, on garde l'ordre du retrieval => déterministe
        return table.sort_values("final", ascending=False, kind="stable")

    def build_recommendation(self, rank: int, idx: int, row: pd.Series,
                             profile: UserProfile) -> Recommendation:
        """Étape 5 : assemble l'objet Recommendation d'un livre."""
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
        """Recommande top_k livres. ``return_table=True`` renvoie aussi la table des candidats."""
        profile = build_profile(answers, self.engine, self.books, self.blocks)
        table = self.score_table(profile, **formula_kwargs)
        if table.empty:
            return (profile, [], table) if return_table else (profile, [])
        # (4) diversité
        chosen = self.diversify(table, self.books, top_k)
        # (5) objets Recommendation
        recs = [self.build_recommendation(r, idx, table.loc[idx], profile)
                for r, idx in enumerate(chosen, start=1)]
        return (profile, recs, table) if return_table else (profile, recs)


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    """Singleton : le catalogue et le modèle ne sont chargés qu'une fois par processus.
    Dans Streamlit, P5 peut l'envelopper dans ``st.cache_resource``.
    """
    return Recommender()


def recommend(answers: dict, top_k: int = 3) -> tuple[UserProfile, list[Recommendation]]:
    """Contrat de l'app : réponses du questionnaire -> (profil, top_k recommandations)."""
    return get_recommender().recommend(answers, top_k=top_k)
