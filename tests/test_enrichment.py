"""Unit tests of src/enrichment.py (without downloading a model: deterministic fake encoder)."""
import json
import zlib

import numpy as np
import pandas as pd

from src import enrichment as en


class FakeEncoder:
    """Hashed bag of words + projection: close text => high cosine. L2-normalised output."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def _toy_catalogue(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    topics = {
        "wizard": ("A young wizard attends a magic school with dragons and spells in a haunted castle", ["Fantasy Books"]),
        "detective": ("A detective investigates a murder and a crime in a foggy city full of secrets", ["Crime, Thriller & Mystery"]),
        "cooking": ("Delicious recipes and cooking tips for food lovers and home bakers every day", ["Cookery"]),
        "unknown": ("A quiet novel about a family living in a small town during one long summer", ["Fiction"]),
    }
    rows = []
    for i in range(n):
        k = list(topics)[i % 4]
        desc, cats = topics[k]
        rows.append({"book_id": str(i), "title": f"Book {i}", "authors_str": "A", "main_author": "A",
                     "categories": np.array(cats), "description": f"{desc} number {rng.integers(1000)} {k}",
                     "audience": "Adult", "is_fiction": k != "cooking", "domain": "Fiction"})
    df = pd.DataFrame(rows)
    df["embed_text"] = df["title"] + ". " + df["description"]
    return df


def test_blocks_are_valid():
    blocks = en.load_blocks()
    assert set(en.BLOCK_NAMES) <= set(blocks)
    assert en.ERA_LABELS <= set(blocks["Atmosphere"])
    assert len(blocks["Genre"]) >= 15 and len(blocks["Themes"]) >= 20


def test_rule_genres():
    assert "Fantasy" in en.rule_genres("Fantasy Books")
    assert {"Mystery & Detective", "Thriller & Suspense"} <= set(en.rule_genres("Crime, Thriller & Mystery"))
    assert "Romance" not in en.rule_genres("Romance Languages")
    assert "History" not in en.rule_genres("War & military fiction")


def test_select_labels_groups_and_min_keep():
    labels = ["Ancient world", "Medieval", "Melancholic", "Romantic"]
    z = np.array([[3.0, 2.5, 2.0, -1.0], [-5.0, -6.0, -7.0, -8.0]])
    group_of = {l: ("era" if l in en.ERA_LABELS else "mood") for l in labels}
    out = en.select_labels(z, labels, 2, 1.0, 1, group_of, 1)
    assert out[0] == ["Ancient world", "Melancholic"]  # 1 period + 1 mood
    assert out[1] == ["Ancient world"]                 # min_keep: never empty


def test_mmr_no_redundant_keywords():
    enc = FakeEncoder()
    phrases = ["boy wizard", "wizard", "magic school", "dragons", "castle"]
    emb = enc.encode(phrases)
    kws = en.mmr_select(emb, enc.encode(["a boy wizard at a magic school"])[0], phrases, top_n=4)
    assert len(kws) == len(set(kws)) and not ("boy wizard" in kws and "wizard" in kws)


def test_enrich_catalogue_contract(tmp_path):
    df = _toy_catalogue()
    enc = FakeEncoder()
    out, report = en.enrich_catalogue(df, enc, verbose=False)
    for col in en.CONTRACT_NEW_COLS:
        assert col in out.columns
    assert out["genres"].map(len).between(1, 3).all()
    assert out["themes"].map(len).between(1, 3).all()
    assert out["styles"].map(len).between(1, 2).all()
    assert out["atmospheres"].map(len).between(1, 2).all()
    assert out["keywords"].map(len).max() <= 5
    ls = json.loads(out["label_scores"].iloc[0])
    assert set(ls) == set(en.BLOCK_NAMES)
    assert "Fantasy" in out.loc[0, "genres"] and "Mystery & Detective" in out.loc[1, "genres"]
    assert report["n_books"] == len(df)


def test_annotation_sheet_has_no_predictions():
    df = _toy_catalogue(60)
    sheet = en.make_annotation_sheet(df, n=20, n_nonfiction=5)
    assert len(sheet) == 20 and "genres" not in sheet.columns and "gold_genres" in sheet.columns


def test_evaluate_against_gold(tmp_path):
    enriched = pd.DataFrame({"book_id": ["1", "2"], "genres": [["Fantasy"], ["Romance", "Horror"]],
                             "themes": [["Friendship"], ["Revenge"]]})
    gold = pd.DataFrame({"book_id": ["1", "2"], "gold_genres": ["Fantasy|Young Adult", "Horror"],
                         "gold_themes": ["Friendship", "Grief & loss"]})
    p = tmp_path / "gold.csv"
    gold.to_csv(p, index=False)
    res = en.evaluate_against_gold(enriched, p)
    assert res["genres"]["at_least_one_match"] == 1.0 and res["themes"]["at_least_one_match"] == 0.5
