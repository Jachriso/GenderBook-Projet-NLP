"""Tests unitaires de src/data_loader.py (lancer : python -m pytest tests -q)."""
import pandas as pd
import pytest

from src import config
from src import data_loader as dl


# --- Nettoyage du texte -------------------------------------------------------

def test_parse_id_list():
    assert dl.parse_id_list("[214, 220, 237]") == [214, 220, 237]
    assert dl.parse_id_list("[]") == []
    assert dl.parse_id_list(float("nan")) == []


def test_clean_description_html_and_urls():
    out = dl.clean_description("Love &amp; war<br/>in   Paris . Visit www.example.com")
    assert out == "Love & war in Paris."


def test_boilerplate_detection():
    assert dl.is_boilerplate("This book is a facsimile reprint and may contain imperfections")
    assert dl.is_boilerplate("This is a Print on Demand title.")
    assert not dl.is_boilerplate("A young wizard discovers his destiny at a school of magic.")


def test_strip_marketing_removes_press_quotes_and_banners():
    text = ('THE INSTANT #1 NEW YORK TIMES BESTSELLER! "A chilling read." --Booklist (starred review) '
            'Jack and Grace seem like the perfect couple.')
    assert dl.strip_marketing(text) == "Jack and Grace seem like the perfect couple."


def test_strip_marketing_keeps_story_content():
    text = 'He whispered "run" and they fled. THE END OF THE WORLD came quietly.'
    assert dl.strip_marketing(text) == text


# --- Titres, auteurs, éditions -----------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Dune : Deluxe Edition", "Dune"),
    ("The Crucible : (penguin Orange Collection)", "The Crucible"),
    ("Pride and Prejudice [Large Print Unabridged Edition]", "Pride and Prejudice"),
    ("Chapterhouse: Dune", "Chapterhouse : Dune"),
])
def test_display_title(raw, expected):
    assert dl.display_title(raw) == expected


def test_author_key_unifies_initials():
    assert dl.author_key("J.K. Rowling") == dl.author_key("J K Rowling") == "jkrowling"


def test_box_set_regex():
    assert dl.BOX_SET_RE.search("The Hunger Games Trilogy (Box Set)")
    assert dl.BOX_SET_RE.search("Anne of Green Gables Complete 8 Book Box Set")
    assert not dl.BOX_SET_RE.search("The Collected Stories")


def _editions_frame(rows):
    df = pd.DataFrame(rows, columns=["title", "authors_list", "rating_count", "rating_avg"])
    df["author_keys"] = df["authors_list"].map(lambda l: [dl.author_key(a) for a in l])
    df["description"] = "x" * 400
    return df


def test_dedup_merges_editions_of_same_work():
    df = _editions_frame([
        ("Harry Potter and the Philosopher's Stone - Gryffindor Edition", ["J.K. Rowling"], 7363346, 4.47),
        ("Harry Potter and the Sorcerer's Stone", ["J. K. Rowling"], 7363346, 4.47),
        ("Harry Potter and the Philosopher's Stone", ["Jim Kay", "J K Rowling"], 7248094, 4.47),
    ])
    out = dl.dedup_editions(df)
    assert len(out) == 1
    assert "Edition" not in out.iloc[0]["title"]          # on garde l'édition standard


def test_dedup_keeps_different_books_of_same_author():
    # Même auteur, notes à ±1 % : ce sont pourtant deux romans différents
    df = _editions_frame([
        ("Make Me", ["Lee Child"], 61532, 3.99),
        ("Personal", ["Lee Child"], 61516, 3.97),
    ])
    assert len(dl.dedup_editions(df)) == 2


# --- Catégories et public -------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("Contemporary Fiction", "fiction"),
    ("Fantasy Books for Kids", "fiction"),
    ("True Crime Books", "nonfiction"),
    ("History Of Western Philosophy", "nonfiction"),
    ("Teaching Resources & Education", "excluded"),
    ("General Cookery", "excluded"),
    ("Teenagers: Advice For Parents", "neutral"),
])
def test_classify_category(name, expected):
    assert dl.classify_category(name) == expected


@pytest.mark.parametrize("ages, cats, expected", [
    ("0-5", [], "Children"),
    ("9-12", [], "Children"),
    ("12-17", [], "Young Adult"),
    (None, ["Romance Books for Teenagers"], "Young Adult"),
    (None, ["Picture Books"], "Children"),
    (None, ["Crime Fiction"], "Adult"),
])
def test_infer_audience(ages, cats, expected):
    assert dl.infer_audience(ages, cats) == expected


def test_truncate_at_sentence():
    text = "First sentence. " * 50
    out = dl.truncate_at_sentence(text, 100)
    assert len(out) <= 100 and out.endswith(".")


# --- Catalogue produit (si déjà généré) -----------------------------------------

catalogue_missing = not config.BOOKS_CLEAN.exists()


@pytest.mark.skipif(catalogue_missing, reason="lancer d'abord scripts/build_dataset.py")
def test_catalogue_contract():
    df = dl.load_books(config.BOOKS_CLEAN)
    assert list(df.columns) == dl.CLEAN_COLUMNS
    assert df["book_id"].is_unique
    assert (df["description"].str.len() >= config.MIN_DESC_CHARS).all()
    assert df["audience"].isin(config.AUDIENCES).all()
    assert df["popularity"].between(0, 1).all()
    assert df["main_author"].value_counts().max() <= config.MAX_BOOKS_PER_AUTHOR
    assert isinstance(df.loc[0, "authors"], list)


@pytest.mark.skipif(not config.BOOKS_SAMPLE.exists(), reason="échantillon absent")
def test_sample_has_full_schema():
    s = dl.load_books(config.BOOKS_SAMPLE)
    assert len(s) == config.SAMPLE_SIZE
    for col in [*dl.CLEAN_COLUMNS, *dl.ENRICHMENT_COLUMNS, "label_scores"]:
        assert col in s.columns
    assert s.loc[0, "label_scores"] == {}


@pytest.mark.skipif(catalogue_missing, reason="catalogue absent")
def test_utilities():
    df = dl.load_books(config.BOOKS_CLEAN)
    assert dl.find_author(df, "stephen kng") == "Stephen King"
    assert dl.find_author(df, "zzzz qqqq") is None
    mask = dl.apply_constraints(df, audience="Children", fiction_only=True)
    assert mask.sum() > 0 and (df.loc[mask, "audience"] == "Children").all()
