"""Tests unitaires de la partie P4 (profil, scoring, recommandation, explications).

Lancer : python -m pytest tests/test_scoring.py -q

Ces tests n'ont besoin NI d'Internet NI du modèle SBERT : ils utilisent un
encodeur « sac de mots haché » déterministe et un mini-catalogue synthétique.
On teste la LOGIQUE (formule, filtres, top_k, diversité, déterminisme), pas la
qualité des embeddings (c'est le rôle du notebook 04 et de l'évaluation de P6).
"""
import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from src import config
from src import profile as pr
from src import scoring as sc
from src.fallback_engine import LocalEngine, l2_normalize, split_sentences
from src.recommender import Recommender
from src.schemas import BLOCKS, Recommendation, UserProfile


# ---------------------------------------------------------------------------
# Données de test
# ---------------------------------------------------------------------------
class HashEncoder:
    """Encodeur déterministe : chaque mot est haché dans un vecteur de 512 dimensions."""

    model_name = "hash512"

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", str(t).lower()):
                # hachage stable (hash() de Python change d'une exécution à l'autre)
                out[i, sum(ord(c) * 31 ** k for k, c in enumerate(w)) % 512] += 1.0
        out[:, 0] += 1e-3  # évite les vecteurs nuls
        return l2_normalize(out)


BLOCKS_REF = {
    "Genre": {"Fantasy": "fantasy magic wizards dragons quest",
              "Mystery & Detective": "mystery detective murder investigation crime",
              "Romance": "romance love passion relationship"},
    "Themes": {"Friendship": "friendship loyalty friends",
               "Crime & justice": "crime justice police murder"},
    "Style": {"Fast-paced page-turner": "fast paced gripping page turner suspense",
              "Slow and contemplative": "slow contemplative quiet reflective"},
    "Atmosphere": {"Dark & unsettling": "dark disturbing intense eerie",
                   "Cozy & heartwarming": "cozy warm light feel good smile"},
}


def _book(i, title, author, desc, audience="Adult", fmt="Paperback", genres=(), themes=(),
          styles=(), atmos=(), pop=0.5, fiction=True):
    return dict(book_id=f"b{i}", title=title, authors=[author], authors_str=author,
                main_author=author, description=desc, embed_text=f"{title}. {desc}",
                audience=audience, format=fmt, pub_year=2010, is_fiction=fiction,
                popularity=pop, image_url="", genres=list(genres), themes=list(themes),
                styles=list(styles), atmospheres=list(atmos))


@pytest.fixture(scope="module")
def books():
    rows = [
        _book(0, "The Dragon Quest", "Ann Wise", "A young wizard and his friends go on a magic quest to fight dragons. Friendship saves them.",
              audience="Young Adult", genres=["Fantasy"], themes=["Friendship"], styles=["Fast-paced page-turner"]),
        _book(1, "Dragon School", "Ann Wise", "Magic school for wizards where dragons hatch. A gripping fantasy quest.",
              audience="Young Adult", genres=["Fantasy"], themes=["Friendship"]),
        _book(2, "Wizard of Storms", "Bob Stone", "An old wizard fights a dark lord with magic and dragons.",
              audience="Adult", fmt="Hardback", genres=["Fantasy"], atmos=["Dark & unsettling"]),
        _book(3, "Cold Murder", "Cara Night", "A detective investigates a brutal murder in a dark, eerie town. Crime and justice.",
              genres=["Mystery & Detective"], themes=["Crime & justice"], atmos=["Dark & unsettling"]),
        _book(4, "The Last Clue", "Dan Gray", "A police detective hunts a serial killer. Gripping, fast paced suspense.",
              genres=["Mystery & Detective"], themes=["Crime & justice"], styles=["Fast-paced page-turner"]),
        _book(5, "Summer Love", "Eva Rose", "A warm, cozy romance about love and passion in a small seaside town.",
              genres=["Romance"], atmos=["Cozy & heartwarming"], pop=0.9),
        _book(6, "Quiet Hearts", "Eva Rose", "A slow, contemplative love story full of reflective moments.",
              genres=["Romance"], styles=["Slow and contemplative"]),
        _book(7, "Little Fox", "Finn Small", "A cozy picture book about a fox and his friends.",
              audience="Children", genres=[], atmos=["Cozy & heartwarming"]),
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def reco(books):
    engine = LocalEngine(books, BLOCKS_REF, encoder=HashEncoder(), cache=False)
    cal = {c: (0.0, 1.0) for c in sc.COMPONENTS}  # calibration neutre : scores = cosinus
    return Recommender(books=books, engine=engine, blocks=BLOCKS_REF, calibration=cal)


def fantasy_answers(**over):
    a = {
        "free_text": {"ideal_book": "A magic quest with wizards and dragons and loyal friends",
                      "loved_book": "", "atmosphere": "", "avoid": ""},
        "genres": ["Fantasy"], "themes": ["Friendship"], "period": "No preference",
        "audience": None, "formats": [], "favorite_author": "",
        "guided": {"focus": "Plot", "standalone_only": False, "exclude_favorite_author": False},
        "likert": {"complexity": 3, "pace": 5, "darkness": 3, "popularity": 3,
                   "w_genre": 5, "w_themes": 3, "w_style": 2, "w_atmosphere": 1},
    }
    a.update(over)
    return a


# ---------------------------------------------------------------------------
# 1. Contrat des dataclasses
# ---------------------------------------------------------------------------
def test_schemas_follow_contract():
    up = [f.name for f in dataclasses.fields(UserProfile)]
    assert up[:6] == ["query_texts", "query_vecs", "weights", "block_affinities", "constraints", "summary"]
    rec = [f.name for f in dataclasses.fields(Recommendation)]
    assert rec[:14] == ["rank", "book_id", "title", "authors", "image_url", "final_score", "match_label",
                        "block_scores", "contributions", "coverage", "matched_labels", "best_sentence",
                        "best_sentence_score", "explanation"]


# ---------------------------------------------------------------------------
# 2. Profil : Likert -> phrases, poids, λ, texte à éviter
# ---------------------------------------------------------------------------
def test_likert_converted_to_sentences():
    assert "complex" in pr.likert_to_sentence("complexity", 5)
    assert "slow" in pr.likert_to_sentence("pace", 1)
    assert pr.likert_to_sentence("darkness", 3) is None   # neutre = pas de phrase
    assert pr.likert_to_sentence("pace", None) is None


def test_lambda_from_popularity_likert():
    assert pr.lambda_from_likert(1) == 0.0
    assert pr.lambda_from_likert(5) == pytest.approx(config.LAMBDA_POP_MAX)
    assert 0 < pr.lambda_from_likert(3) < config.LAMBDA_POP_MAX


def test_avoid_text_negations_removed():
    assert pr.clean_avoid_text("No romance-centered plots, no gore") == "romance-centered plots, gore"


def test_resolve_label_tolerates_short_names():
    blocks = {"Themes": {"Identity & self-discovery": "x", "Morality & guilt": "y"}}
    assert pr.resolve_label(blocks, "Themes", "Identity") == "Identity & self-discovery"
    assert pr.resolve_label(blocks, "Themes", "Morality & Guilt") == "Morality & guilt"
    assert pr.resolve_label(blocks, "Themes", "Zombies") is None


def test_build_profile_contents(reco):
    p = pr.build_profile(fantasy_answers(), reco.engine, reco.books, BLOCKS_REF)
    assert set(p.weights) == set(BLOCKS) and p.weights["Genre"] == 5
    assert p.query_vecs["global"].shape == (512,)
    assert np.isclose(np.linalg.norm(p.query_vecs["global"]), 1.0, atol=1e-5)
    assert BLOCKS_REF["Genre"]["Fantasy"] in p.block_phrases["Genre"]
    assert any("fast-paced" in s for s in p.block_phrases["Style"])        # Likert pace = 5
    assert all(0.0 <= v <= 1.0 for d in p.block_affinities.values() for v in d.values())
    assert isinstance(p.summary, str) and "Fantasy" in p.summary


# ---------------------------------------------------------------------------
# 3. Formule
# ---------------------------------------------------------------------------
def test_block_similarity_max_then_mean():
    # 2 phrases utilisateur, 2 labels, 2 livres, d = 2
    P = np.array([[1, 0], [0, 1]], dtype=np.float32)
    L = np.array([[1, 0], [0, 1]], dtype=np.float32)
    E = l2_normalize(np.array([[1, 1], [1, 0]], dtype=np.float32))
    mask = np.array([[True, False], [False, False]])  # livre 0 porte le label 0 ; livre 1 aucun
    s, best = sc.block_similarity(P, L, mask, E)
    # livre 0 : phrase 0 -> max(label0=1, desc=0.707)=1 ; phrase 1 -> max(label0=0, desc=0.707)=0.707
    assert s[0] == pytest.approx((1 + 0.70710677) / 2, abs=1e-5)
    # livre 1 : pas de label -> description seule : (1 + 0) / 2
    assert s[1] == pytest.approx(0.5, abs=1e-5)


def test_calibration_bounds():
    out = sc.calibrate(np.array([-1.0, 0.2, 0.5, 2.0]), 0.2, 0.6)
    assert out.min() >= 0 and out.max() <= 1
    assert out[1] == 0 and out[2] == pytest.approx(0.75)


def test_coverage_is_weighted_mean():
    s = {"Genre": np.array([1.0]), "Themes": np.array([0.5]), "Style": np.array([0.0]), "Atmosphere": np.array([0.5])}
    w = {"Genre": 2, "Themes": 1, "Style": 1, "Atmosphere": 0}
    assert sc.coverage_score(s, w)[0] == pytest.approx((2 * 1 + 1 * 0.5) / 4)


def test_match_labels_thresholds():
    assert sc.match_label(0.80) == "Excellent"
    assert sc.match_label(0.60) == "Good"
    assert sc.match_label(0.30) == "Partial"


# ---------------------------------------------------------------------------
# 4. Recommandation de bout en bout
# ---------------------------------------------------------------------------
def test_scores_in_unit_interval_and_contributions_sum(reco):
    _, recs, table = reco.recommend(fantasy_answers(), top_k=3, return_table=True)
    assert ((table["final"] >= 0) & (table["final"] <= 1)).all()
    for comp in ("S_desc", *BLOCKS, "coverage"):
        assert ((table[comp] >= 0) & (table[comp] <= 1)).all()
    for r in recs:
        assert sum(r.contributions.values()) == pytest.approx(r.final_score, abs=1e-3)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_respected(reco, k):
    _, recs = reco.recommend(fantasy_answers(), top_k=k)
    assert len(recs) == k
    assert [r.rank for r in recs] == list(range(1, k + 1))
    assert all(recs[i].final_score >= recs[i + 1].final_score for i in range(k - 1))


def test_semantic_ranking_makes_sense(reco):
    _, recs = reco.recommend(fantasy_answers(), top_k=2)
    assert recs[0].title in {"The Dragon Quest", "Dragon School", "Wizard of Storms"}


def test_hard_filters_respected(reco):
    _, recs = reco.recommend(fantasy_answers(audience="Adult", formats=["Paperback"]), top_k=3)
    ids = {r.book_id for r in recs}
    allowed = set(reco.books[(reco.books.audience == "Adult") & (reco.books.format == "Paperback")].book_id)
    assert ids <= allowed


def test_excluded_favourite_author(reco):
    a = fantasy_answers(favorite_author="Ann Wise",
                        guided={"focus": "Plot", "standalone_only": False, "exclude_favorite_author": True})
    _, recs = reco.recommend(a, top_k=3)
    assert all(r.main_author != "Ann Wise" for r in recs)


def test_diversity_one_book_per_author(reco):
    _, recs = reco.recommend(fantasy_answers(), top_k=5)
    authors = [r.main_author for r in recs]
    assert len(authors) == len(set(authors))


def test_filters_relaxed_when_nothing_matches(reco):
    profile, recs = reco.recommend(fantasy_answers(audience="Children", formats=["Hardback"]), top_k=1)
    assert len(recs) == 1
    assert "formats" in profile.constraints["relaxed"]


def test_deterministic(reco):
    _, a = reco.recommend(fantasy_answers(), top_k=3)
    _, b = reco.recommend(fantasy_answers(), top_k=3)
    assert [r.book_id for r in a] == [r.book_id for r in b]
    assert [r.final_score for r in a] == [r.final_score for r in b]


def test_avoid_penalty_lowers_score(reco):
    base = fantasy_answers()
    avoid = fantasy_answers(free_text={**base["free_text"], "avoid": "no dragons, no wizards, no magic"})
    _, t0 = reco.recommend(base, top_k=3, return_table=True)[::2]
    _, t1 = reco.recommend(avoid, top_k=3, return_table=True)[::2]
    i = int(reco.books.index[reco.books.title == "Dragon School"][0])
    assert t1.loc[i, "final"] < t0.loc[i, "final"]


def test_recommendation_fields_and_explanation(reco):
    _, recs = reco.recommend(fantasy_answers(), top_k=1)
    r = recs[0]
    assert r.match_label in {"Excellent", "Good", "Partial"}
    assert set(r.block_scores) == {"S_desc", *BLOCKS}
    assert r.best_sentence and r.best_sentence in reco.books.loc[reco.books.book_id == r.book_id, "description"].iloc[0]
    assert "coverage" in r.explanation.lower()
    assert isinstance(r.to_dict(), dict)


def test_works_without_p2_columns(books):
    """Fallback : catalogue sans genres/themes/... (books_sample de P1) et sans blocks.json."""
    bare = books.drop(columns=["genres", "themes", "styles", "atmospheres"])
    empty_blocks = {b: {} for b in BLOCKS}
    engine = LocalEngine(bare, empty_blocks, encoder=HashEncoder(), cache=False)
    r = Recommender(books=bare, engine=engine, blocks=empty_blocks,
                    calibration={c: (0.0, 1.0) for c in sc.COMPONENTS})
    _, recs = r.recommend(fantasy_answers(), top_k=2)
    assert len(recs) == 2 and all(0 <= x.final_score <= 1 for x in recs)


def test_split_sentences():
    assert split_sentences("First sentence here. Second one is longer. Ok.") == \
        ["First sentence here.", "Second one is longer. Ok."]
