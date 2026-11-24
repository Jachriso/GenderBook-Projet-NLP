"""Tests of src/evaluation.py (Person 6): metrics on hand-made data, no model needed."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import (build_judgment_pool, first_relevant_rank, forbidden_ids, jaccard,
                            judged_precision, load_eval_personas, loved_book_ids, n_unjudged, persona_metrics,
                            relevant_works, summarize, with_ideal_book)


@pytest.fixture
def books():
    """Tiny catalogue: 5 books, index 0..4 like the Recommender's catalogue."""
    return pd.DataFrame({
        "book_id": ["b0", "b1", "b2", "b3", "b4"],
        "title": ["Zero", "One", "Two", "Three", "Four"],
        "authors": [["Ann Lee"], ["Stephen King"], ["Bob Ray"], ["Cy Doe"], ["Di Fox"]],
        "authors_str": ["Ann Lee", "Stephen King", "Bob Ray", "Cy Doe", "Di Fox"],
        "description": ["blurb zero", "blurb one", "blurb two", "blurb three", "blurb four"],
        "genres": [["Horror"], ["Horror"], ["Romance"], ["Horror", "Classics"], ["Fantasy"]],
        "themes": [["Madness & the mind"], ["Revenge"], ["Love & heartbreak"],
                   ["Grief & loss"], ["Friendship"]],
    })


@pytest.fixture
def persona():
    return {
        "id": "T01",
        "answers": {"free_text": {"ideal_book": "original", "atmosphere": "dark", "avoid": ""}},
        "paraphrases": [],
        "expected": {
            "loved_book_work": {"title": "Zero", "book_ids": ["b0"]},
            "genres": ["Horror"],
            "themes": ["Madness & the mind", "Grief & loss"],
            "excluded_authors": ["Stephen King"],
            "relevant": [{"title": "Three", "book_ids": ["b3"]},
                         {"title": "Four (2 editions)", "book_ids": ["b4", "bX"]}],
            "must_not_appear": [{"title": "Two", "book_ids": ["b2"]}],
        },
    }


def make_run(books, order, top):
    """Fake pipeline output: candidates ranked in `order`, `top` = chosen indices."""
    return {"table": pd.DataFrame(index=order), "top": top,
            "top_ids": [books.at[i, "book_id"] for i in top], "seconds": 0.1}


def test_jaccard():
    assert jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert jaccard(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(0.5)


def test_relevant_works_and_forbidden(persona):
    assert relevant_works(persona) == [{"b3"}, {"b4", "bX"}]
    assert forbidden_ids(persona) == {"b2"}


def test_first_relevant_rank_any_edition_counts(books, persona):
    table = pd.DataFrame(index=[0, 1, 4, 3])          # b4 (an edition of work 2) ranked 3rd
    assert first_relevant_rank(table, books, relevant_works(persona)) == 3
    assert first_relevant_rank(pd.DataFrame(index=[0, 1]), books, relevant_works(persona)) is None


def test_persona_metrics_values(books, persona):
    m = persona_metrics(persona, make_run(books, order=[0, 3, 4, 1], top=[0, 3, 4]), books)
    assert m["genre_precision@3"] == pytest.approx(2 / 3)       # Zero and Three are Horror
    assert m["theme_overlap@3"] == pytest.approx(1.0)           # both expected themes covered
    assert m["hit@3"] == 1.0
    assert m["first_relevant_rank"] == 2 and m["reciprocal_rank"] == pytest.approx(0.5)
    assert m["constraints_ok"] is True


def test_constraints_detect_excluded_author_and_forbidden_book(books, persona):
    m = persona_metrics(persona, make_run(books, order=[1, 0, 2], top=[1, 0, 2]), books)
    assert m["constraints_ok"] is False                         # Stephen King + forbidden b2
    assert m["hit@3"] == 0.0 and m["reciprocal_rank"] == 0.0


def test_summarize_mrr_is_mean_of_reciprocal_ranks():
    df = pd.DataFrame({"genre_precision@3": [1, 0], "theme_overlap@3": [0.5, 0.5], "hit@3": [1, 0],
                       "reciprocal_rank": [1.0, 0.5], "constraints_ok": [True, False], "seconds": [0.1, 0.3]})
    s = summarize(df)
    assert s["MRR"] == 0.75 and s["constraints_ok"] == "1/2"


def test_loved_book_is_reported_but_not_relevant(books, persona):
    assert loved_book_ids(persona) == {"b0"}
    m = persona_metrics(persona, make_run(books, order=[0, 4], top=[0, 4]), books)
    assert m["loved_book_in_top3"] is True
    assert m["first_relevant_rank"] == 2          # b0 (already read) does not count, b4 does


def test_judgment_pool_is_blind_extends_and_keeps_existing_labels(books, persona, tmp_path):
    path = tmp_path / "judgments.csv"
    pool = build_judgment_pool({"A": {"T01": ["b0", "b3"]}, "B": {"T01": ["b3", "b4"]}},
                               [persona], books, path=path)
    assert sorted(pool["book_id"]) == ["b0", "b3", "b4"]           # b3 pooled once
    assert "system" not in pool.columns and "encoder" not in pool.columns
    pool.loc[pool["book_id"] == "b3", "relevant"] = 1
    pool.loc[pool["book_id"] == "b0", "relevant"] = 0
    pool.to_csv(path, index=False)
    assert n_unjudged(pd.read_csv(path)) == 1                       # b4 still empty
    order_before = pd.read_csv(path, dtype={"book_id": str})["book_id"].tolist()
    pool2 = build_judgment_pool({"C": {"T01": ["b3", "b1"]}}, [persona], books, path=path)
    assert len(pool2) == 4                                          # only b1 added
    assert pool2["book_id"].tolist()[:3] == order_before            # existing rows never move
    assert pool2["book_id"].tolist()[3] == "b1"
    assert pool2.loc[pool2["book_id"] == "b3", "relevant"].astype(float).item() == 1.0
    prec = judged_precision({"A": {"T01": ["b0", "b3"]}, "B": {"T01": ["b3", "b4"]}}, path=path)
    assert prec.loc["A", "judged_precision@3"] == 0.5 and prec.loc["A", "n_unjudged"] == 0
    assert prec.loc["B", "judged_precision@3"] == 1.0 and prec.loc["B", "n_unjudged"] == 1


def test_with_ideal_book_does_not_modify_the_original(persona):
    new = with_ideal_book(persona["answers"], "rephrased")
    assert new["free_text"]["ideal_book"] == "rephrased"
    assert persona["answers"]["free_text"]["ideal_book"] == "original"


def test_eval_personas_file_is_valid():
    """The real personas: 10, 3 rephrasings each, labels from blocks.json, known book_ids."""
    personas = load_eval_personas()
    blocks = json.loads(config.BLOCKS_JSON.read_text(encoding="utf-8"))
    assert len(personas) == 10 and len({p["id"] for p in personas}) == 10
    ids = None
    if config.BOOKS_ENRICHED.exists():
        ids = set(pd.read_parquet(config.BOOKS_ENRICHED, columns=["book_id"])["book_id"].astype(str))
    for p in personas:
        assert len(p["paraphrases"]) == 3
        for key, block in (("genres", "Genre"), ("themes", "Themes")):
            assert set(p["expected"][key]) <= set(blocks[block]), p["id"]
            assert set(p["answers"][key]) <= set(blocks[block]), p["id"]
        if ids is not None:
            works = p["expected"]["relevant"] + p["expected"].get("must_not_appear", [])
            works += [p["expected"]["loved_book_work"]] if "loved_book_work" in p["expected"] else []
            for w in works:
                assert set(w["book_ids"]) <= ids, (p["id"], w["title"])
        # the book the user already read is never counted as relevant
        assert not (loved_book_ids(p) & set().union(*relevant_works(p))), p["id"]
