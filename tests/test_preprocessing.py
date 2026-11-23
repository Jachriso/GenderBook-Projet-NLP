"""Tests unitaires de src/preprocessing.py (P3). Lancer : python -m pytest tests -q"""

import math

from src.preprocessing import (
    preprocess_for_classic,
    preprocess_for_transformer,
    preprocess_many_for_classic,
    split_sentences,
)


# ---------- preprocess_for_transformer : nettoyage minimal ----------

def test_transformer_handles_missing_values():
    # None et NaN (valeur manquante pandas) doivent donner une chaîne vide, pas une erreur
    assert preprocess_for_transformer(None) == ""
    assert preprocess_for_transformer(math.nan) == ""


def test_transformer_removes_html_and_extra_spaces():
    assert preprocess_for_transformer("  A <b>dark</b>   tale&amp;more!!!  ") == "A dark tale&more!"


def test_transformer_keeps_stopwords_and_negation():
    # La négation doit survivre : c'est tout l'intérêt du nettoyage minimal
    out = preprocess_for_transformer("A story with no romance, not too dark.")
    assert "no romance" in out and "not" in out


def test_transformer_normalizes_quotes():
    assert preprocess_for_transformer("Harry’s “magic”") == "Harry's \"magic\""


# ---------- preprocess_for_classic : sac de mots ----------

def test_classic_lowercases_lemmatizes_and_removes_stopwords():
    out = preprocess_for_classic("The young wizards were running to the magic schools!")
    assert out == "young wizard run magic school"


def test_classic_tokens_mode_returns_list():
    assert preprocess_for_classic("Two dragons and 3 knights.", as_tokens=True) == ["dragon", "knight"]


def test_classic_batch_equals_single():
    texts = ["The wizards ran.", None, "A dark and stormy night."]
    assert preprocess_many_for_classic(texts) == [preprocess_for_classic(t) for t in texts]


# ---------- split_sentences : explicabilité ----------

def test_split_sentences_basic():
    text = "Harry Potter is an orphan boy. One day he learns that he is a wizard."
    assert split_sentences(text) == [
        "Harry Potter is an orphan boy.",
        "One day he learns that he is a wizard.",
    ]


def test_split_sentences_merges_short_fragments():
    # "Ok." est trop court : il est collé à la phrase précédente au lieu d'être perdu
    sents = split_sentences("Harry Potter is an orphan boy. Ok.")
    assert sents == ["Harry Potter is an orphan boy. Ok."]


def test_split_sentences_empty():
    assert split_sentences("") == [] and split_sentences(None) == []
