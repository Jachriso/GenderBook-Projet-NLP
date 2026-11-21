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


# =============================================================================
# Tests ajoutés suite à la revue de la PR #1
# =============================================================================

# --- 🔴 Variantes d'orthographe des auteurs ----------------------------------

def test_canonical_author_names_unifies_variants():
    lists = [["J.K. Rowling"], ["J.K. Rowling"], ["J. K. Rowling"], ["Jim Kay", "J K Rowling"]]
    canonical = dl.canonical_author_names(lists)
    assert canonical["jkrowling"] == "J.K. Rowling"          # orthographe la plus fréquente
    assert dl.canonicalize_authors(["Jim Kay", "J K Rowling"], canonical) == ["Jim Kay", "J.K. Rowling"]


def _toy_books():
    return pd.DataFrame({
        "title": ["HP 1", "HP 2", "HP 3", "Other"],
        "authors": [["J.K. Rowling"], ["J. K. Rowling"], ["J K Rowling", "Jim Kay"], ["Stephen King"]],
    })


def test_get_books_by_author_matches_all_spellings():
    books = _toy_books()
    assert len(dl.get_books_by_author(books, "jk rowling")) == 3
    assert len(dl.get_books_by_author(books, "J.K.Rowling")) == 3


def test_excluded_author_removes_all_spellings():
    books = _toy_books()
    mask = dl.apply_constraints(books, excluded_authors=["J. K. Rowling"])
    assert books.loc[mask, "title"].tolist() == ["Other"]


# --- 🟠 strip_marketing ne doit pas couper le contenu -------------------------

def test_publication_attribution_stops_before_story():
    # "--People" est un magazine : "Meghan" est déjà le début de l'intrigue
    text = '"A premium delivery." --People Meghan doesn\'t know Agatha, but Agatha knows Meghan.'
    assert dl.strip_marketing(text) == "Meghan doesn't know Agatha, but Agatha knows Meghan."


def test_author_of_title_is_fully_removed_with_dataset_knowledge():
    stripper = dl.MarketingStripper(author_names=["Jennifer E. Smith"],
                                    titles=["The Statistical Probability of Love at First Sight"])
    text = ('A New York Times Bestseller "Here are three things about this book." --Jennifer E. Smith, '
            'author of The Statistical Probability of Love at First Sight With the perfect mix of comedy '
            'and tragedy, the characters come to feel like old friends.')
    assert stripper(text).startswith("With the perfect mix of comedy")


def test_in_story_quote_is_kept():
    text = 'He was left with a Turkish inventor known as Hoja--"master"--a man who is his exact double.'
    assert dl.strip_marketing(text) == text


def test_sales_sentence_is_kept_when_removal_would_cut_a_phrase():
    # retirer "The #1 New York Times bestselling" laisserait "debut novel that introduced..."
    text = "The #1 New York Times bestselling debut novel that introduced Khaled Hosseini to millions."
    assert dl.strip_marketing(text) == text
    # mais un bandeau autonome est retiré
    assert dl.strip_marketing("#1 NEW YORK TIMES BESTSELLER - A pair of sisters meet.") == "A pair of sisters meet."


def test_praise_section_is_removed():
    text = ('Two boys grow up in Kabul. Praise for The Kite Runner "A powerful book." --The Washington Post '
            '"Haunting." --Booklist')
    assert dl.strip_marketing(text) == "Two boys grow up in Kabul."


# --- Mineur : boilerplate phrase par phrase --------------------------------------

def test_strip_boilerplate_keeps_real_content():
    text = ("Dorothy is swept away to the magical land of Oz. The original book has been in the public "
            "domain in the US since 1956. She meets a scarecrow, a tin man and a lion.")
    out = dl.strip_boilerplate(text)
    assert "public domain" not in out
    assert out.startswith("Dorothy") and out.endswith("lion.")


# --- Mineur : doublons dus à la ponctuation ----------------------------------------

def test_normalize_title_punctuation():
    assert dl.normalize_title("Cat's Eye") == dl.normalize_title("Cats Eye")
    assert dl.normalize_title("The Water-Babies") == dl.normalize_title("The Water Babies")


@pytest.mark.parametrize("a, b", [
    ("The Yellow Wall-paper", "The Yellow Wallpaper"),
    ("Daddy-Long-Legs", "Daddy Long Legs"),
    ("The Wasteland", "The Waste Land"),
])
def test_dedup_merges_punctuation_and_space_variants(a, b):
    df = _editions_frame([(a, ["Author Name"], 5000, 4.0), (b, ["Author Name"], 4000, 3.9)])
    assert len(dl.dedup_editions(df)) == 1


def test_dedup_does_not_merge_short_title_into_unrelated_book():
    # mêmes notes par coïncidence, "Civil War" inclus dans un titre plus long, auteurs différents
    df = _editions_frame([
        ("Civil War", ["Lucan"], 3000, 3.9),
        ("Spider-Man: Miles Morales Vol. 2: Civil War II", ["Brian Michael Bendis"], 3010, 3.9),
    ])
    assert len(dl.dedup_editions(df)) == 2


# --- Mineur : produits dérivés --------------------------------------------------

@pytest.mark.parametrize("title, authors, expected", [
    ("Harry Potter : A Pop-Up Book", ["Andrew Williamson"], True),
    ("Harry Potter : Page to Screen", ["Bob McCabe"], True),
    ("Harry Potter Film Wizardry", ["Warner Bros"], True),
    ("The World of the Hunger Games", ["Scholastic"], True),
    ("The Art of War", ["Sun Tzu"], False),
    ("The Hitchhiker's Guide to the Galaxy", ["Douglas Adams"], False),
    ("Lying Out Loud : A Companion to the Duff", ["Kody Keplinger"], False),
])
def test_companion_products(title, authors, expected):
    assert dl.is_companion_product(title, authors) is expected


@pytest.mark.skipif(catalogue_missing, reason="catalogue absent")
def test_catalogue_review_fixes():
    df = dl.load_books(config.BOOKS_CLEAN)
    # une seule orthographe par auteur
    ex = df[["authors"]].explode("authors").dropna()
    assert (ex.groupby(ex["authors"].map(dl.author_key))["authors"].nunique() == 1).all()
    # tous les Harry Potter de Rowling sont retrouvés, et tous exclus
    rowling = dl.get_books_by_author(df, "jk rowling")
    assert rowling["title"].str.startswith("Harry Potter and the").sum() == 7
    mask = dl.apply_constraints(df, excluded_authors=["J. K. Rowling"])
    assert not df.loc[mask, "title"].str.startswith("Harry Potter and the").any()
    # pas de doublon titre + auteur (ponctuation et espaces ignorés)
    key = df["title"].map(lambda t: dl.normalize_title(dl.display_title(t)).replace(" ", "")) \
        + "|" + df["main_author"].map(dl.author_key)
    assert not key.duplicated().any()
    # pas de produit dérivé
    assert not df["title"].str.contains("Pop-Up|Page to Screen|Film Wizardry", case=False).any()
