"""Tests des corrections de la review de la PR P2 (encodeur factice, sans téléchargement de modèle)."""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import enrichment as en

ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    """Sac de mots hashé : textes proches => cosinus élevé. Sortie L2-normalisée."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def toy_catalogue(n: int = 80, audience: str = "Adult") -> pd.DataFrame:
    topics = {
        "wizard": ("A young wizard attends a magic school with dragons and spells in a haunted castle", ["Fantasy Books"]),
        "detective": ("A detective investigates a murder and a crime in a foggy city full of secrets", ["Crime, Thriller & Mystery"]),
        "cooking": ("Delicious recipes and cooking tips for food lovers and home bakers every day", ["Cookery"]),
        "teen": ("A teenager discovers first love and friendship at school while growing up fast", ["Young Adult Fiction", "Teen"]),
    }
    rows = []
    for i in range(n):
        k = list(topics)[i % 4]
        desc, cats = topics[k]
        rows.append({"book_id": str(1000 + i), "title": f"Book {i}", "authors_str": "Jane Austen",
                     "categories": np.array(cats), "description": f"{desc} number {i} {k}",
                     "audience": audience, "is_fiction": k != "cooking"})
    df = pd.DataFrame(rows)
    df["embed_text"] = df["title"] + ". " + df["description"]
    return df


# --------------------------------------------------------------------------- #
# Point 2 : thèmes v2 intégrés au pipeline
# --------------------------------------------------------------------------- #
def test_themes_v2_are_mean_of_phrase_and_variants():
    enc, blocks = FakeEncoder(), en.load_blocks()
    variants = en.load_theme_variants(themes=blocks["Themes"])
    labels, mat = en.encode_blocks(enc, blocks)["Themes"]
    l = labels[0]
    expected = enc.encode([blocks["Themes"][l]] + variants[l]).mean(0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(mat[0], expected, atol=1e-5)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_enrich_catalogue_applies_v2_by_itself():
    df, enc = toy_catalogue(), FakeEncoder()
    v2, rep2 = en.enrich_catalogue(df, enc, verbose=False, with_keywords=False)
    v1, rep1 = en.enrich_catalogue(df, enc, verbose=False, with_keywords=False, theme_variants=None)
    assert rep2["themes_version"] == "v2" and rep1["themes_version"] == "v1"
    s2 = json.loads(v2["label_scores"].iloc[0])["Themes"]
    s1 = json.loads(v1["label_scores"].iloc[0])["Themes"]
    assert s1 != s2                                   # les scores de thèmes sont bien ceux de la v2
    again, _ = en.enrich_catalogue(df, enc, verbose=False, with_keywords=False)
    assert again["themes"].tolist() == v2["themes"].tolist()   # reproductible : même appel, même résultat


def test_missing_variants_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        en.load_theme_variants(tmp_path / "absent.json")


def test_variants_must_match_blocks(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"Friendship": ["a b c d e"]}))
    with pytest.raises(ValueError):
        en.load_theme_variants(p, themes=en.load_blocks()["Themes"])


# --------------------------------------------------------------------------- #
# Point 4 : l'époque n'est plus forcée
# --------------------------------------------------------------------------- #
LABELS = ["Ancient world", "Medieval", "Contemporary", "Melancholic", "Romantic"]


def test_no_era_when_scores_are_weak():
    z = np.array([[0.5, 0.2, 0.1, 1.5, 0.3]])
    out = en.select_atmospheres(z, LABELS, z_min=1.0, era_z_min=2.0)
    assert out == [["Melancholic"]]                      # pas d'époque forcée


def test_era_kept_when_strong_and_clear():
    z = np.array([[3.0, 1.0, 0.1, 1.5, 0.3]])
    assert en.select_atmospheres(z, LABELS, 1.0, 2.0, 0.3) == [["Ancient world", "Melancholic"]]


def test_era_rejected_when_ambiguous():
    z = np.array([[2.4, 2.3, 0.1, 1.5, 0.3]])            # deux époques quasi à égalité
    assert en.select_atmospheres(z, LABELS, 1.0, 2.0, 0.3) == [["Melancholic"]]


def test_always_at_least_one_atmosphere():
    z = np.array([[0.1, 0.1, 0.1, 0.2, 0.3]])
    out = en.select_atmospheres(z, LABELS, 1.0, 2.0)
    assert len(out[0]) == 1 and out[0][0] == "Romantic"


def test_era_coverage_decreases_with_threshold():
    rng = np.random.default_rng(0)
    sims = rng.normal(size=(300, len(LABELS)))
    cov = en.era_coverage_report(sims, LABELS)["pct_books_with_era"].tolist()
    assert cov == sorted(cov, reverse=True)


# --------------------------------------------------------------------------- #
# Point 5 : « Young Adult » / « Children's » seulement via audience
# --------------------------------------------------------------------------- #
def test_adult_books_never_get_audience_genres():
    df = toy_catalogue(80, audience="Adult")             # inclut des livres de catégorie « Young Adult Fiction »
    out, _ = en.enrich_catalogue(df, FakeEncoder(), verbose=False, with_keywords=False)
    assert not any(g in en.AUDIENCE_GENRES for gs in out["genres"] for g in gs)


@pytest.mark.parametrize("audience,label", [("Children", "Children's"), ("Young Adult", "Young Adult")])
def test_audience_column_gives_the_label(audience, label):
    df = toy_catalogue(40, audience=audience)
    out, _ = en.enrich_catalogue(df, FakeEncoder(), verbose=False, with_keywords=False)
    assert all(label in gs for gs in out["genres"])


# --------------------------------------------------------------------------- #
# Point 6 : mots-clés = groupes nominaux
# --------------------------------------------------------------------------- #
STOP = en.EXTRA_STOP | en.NOISE_WORDS | {"with", "the", "and"}


def test_noun_phrases_drop_verb_noun_pairs_and_noise():
    tagged = [("She", "PRON"), ("deals", "VERB"), ("with", "ADP"), ("issues", "NOUN"), ("The", "DET"),
              ("young", "ADJ"), ("wizard", "NOUN"), ("school", "NOUN"), ("font", "NOUN"), ("size", "NOUN")]
    c = en.noun_phrase_candidates(tagged, STOP)
    assert "deals issues" not in c and "deals" not in c      # verbe : coupe le groupe
    assert "font size" not in c and "size" not in c          # vocabulaire de mise en page
    assert "wizard school" in c and "school" in c            # groupe nominal conservé
    assert all(len(t) >= 3 for k in c for t in k.split())


def test_noun_phrase_must_end_with_a_noun():
    c = en.noun_phrase_candidates([("dark", "ADJ"), ("secret", "NOUN"), ("beautiful", "ADJ")], STOP)
    assert c == ["secret", "dark secret"]


def fake_tagger(texts):
    verbs, adps = {"investigates", "attends", "deals", "discovers"}, {"with", "in", "of", "and"}
    dets = {"a", "the"}
    out = []
    for t in texts:
        out.append([(w, "VERB" if w.lower() in verbs else "ADP" if w.lower() in adps else "DET" if w.lower() in dets
                     else "NOUN") for w in t.split()])
    return out


def test_keywords_use_only_noun_phrase_candidates_and_skip_authors():
    df = toy_catalogue(60)
    df["description"] = df["description"] + " Austen font size deals"
    book_emb = FakeEncoder().encode(df["embed_text"].tolist())
    kws = en.extract_keywords(df, book_emb, FakeEncoder(), tagger=fake_tagger)
    flat = {k for ks in kws for k in ks}
    assert flat and all(len(ks) <= 5 for ks in kws)
    assert not any("austen" in k or "font" in k or "size" in k or "deals" in k for k in flat)
    assert not any(k.split()[0] in {"investigates", "attends", "discovers"} for k in flat)


def test_pos_true_without_spacy_raises(monkeypatch):
    monkeypatch.setattr(en, "spacy_tagger", lambda *a, **k: (_ for _ in ()).throw(ImportError("no spacy")))
    df = toy_catalogue(10)
    with pytest.raises(RuntimeError):
        en.extract_keywords(df, FakeEncoder().encode(df["embed_text"].tolist()), FakeEncoder(), pos=True)


# --------------------------------------------------------------------------- #
# Point 3 : sample = sous-ensemble exact du fichier complet
# --------------------------------------------------------------------------- #
def test_subset_is_exact_and_ordered():
    full, _ = en.enrich_catalogue(toy_catalogue(60), FakeEncoder(), verbose=False, with_keywords=False)
    ids = ["1010", "1003", "1050"]
    sub = en.subset_enriched(full, ids)
    assert sub["book_id"].tolist() == ids
    ref = full.set_index("book_id").loc[ids]
    for col in ["genres", "themes", "styles", "atmospheres", "label_scores"]:
        assert [list(x) if not isinstance(x, str) else x for x in sub[col]] == \
               [list(x) if not isinstance(x, str) else x for x in ref[col]]


def test_subset_missing_id_raises():
    full, _ = en.enrich_catalogue(toy_catalogue(20), FakeEncoder(), verbose=False, with_keywords=False)
    with pytest.raises(ValueError):
        en.subset_enriched(full, ["1000", "nope"])


# --------------------------------------------------------------------------- #
# Point 1 : jeu de test disjoint du jeu de réglage
# --------------------------------------------------------------------------- #
def test_holdout_sheet_excludes_tuning_books():
    df = toy_catalogue(200)
    tuned = df["book_id"].iloc[:60].tolist()
    sheet = en.make_annotation_sheet(df, n=40, n_nonfiction=10, exclude_ids=tuned)
    assert len(sheet) == 40 and not set(sheet["book_id"]) & set(tuned)
    assert "genres" not in sheet.columns


# --------------------------------------------------------------------------- #
# Point 7 : plus de [x] dans le rapport
# --------------------------------------------------------------------------- #
def test_report_has_no_placeholders():
    report = ROOT / "docs" / "report" / "02_enrichment.md"
    if report.exists():
        txt = report.read_text(encoding="utf-8")
        assert "[x" not in txt and "[X" not in txt and "{{" not in txt


# --------------------------------------------------------------------------- #
# Outil d'annotation (point 1)
# --------------------------------------------------------------------------- #
def _gold_tools():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gold_tools", ROOT / "scripts" / "gold_tools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_reviewed():
    gt = _gold_tools()
    assert not gt.is_reviewed("LLM (Claude), à relire") and not gt.is_reviewed("") and gt.is_reviewed("Hajar")


def test_parse_choice_numbers_and_names():
    gt = _gold_tools()
    labels = ["Fantasy", "Romance", "Horror", "History"]
    assert gt.parse_choice("1,3", labels, 3) == ["Fantasy", "Horror"]
    assert gt.parse_choice("romance | fantasy", labels, 3) == ["Romance", "Fantasy"]
    assert gt.parse_choice("9", labels, 3) is None and gt.parse_choice("Nope", labels, 3) is None
    assert gt.parse_choice("1,2,3,4", labels, 3) is None and gt.parse_choice("", labels, 3) is None


def test_review_loop_keeps_edits_and_saves():
    gt = _gold_tools()
    blocks = en.load_blocks()
    df = pd.DataFrame({"book_id": ["1", "2", "3"], "title": ["A", "B", "C"], "authors_str": ["x"] * 3,
                       "description": ["d"] * 3, "gold_genres": ["Fantasy | Romance", "Horror", ""],
                       "gold_themes": ["Friendship | Revenge", "Survival", ""],
                       "annotator": ["LLM (Claude), à relire", "Marie", ""]})
    answers = iter(["", "", "1", "2,3", "q"])      # livre 1 : garder ; livre 3 : Fantasy + thèmes 2,3 ; puis quitter
    saves = []
    out = gt.review_rows(df, blocks, "Hajar", lambda d: saves.append(d.copy()), input_fn=lambda p: next(answers),
                         print_fn=lambda *a, **k: None)
    assert out.at[0, "annotator"] == "Hajar" and out.at[0, "gold_genres"] == "Fantasy | Romance"   # gardé
    assert out.at[1, "annotator"] == "Marie"                                                       # déjà humain : non touché
    assert out.at[2, "gold_genres"] == "Fantasy" and out.at[2, "annotator"] == "Hajar"
    assert len(saves) == 2                                                                         # sauvegarde à chaque livre


# --------------------------------------------------------------------------- #
# Évaluation réglage / test
# --------------------------------------------------------------------------- #
def _evaluate_module():
    import importlib.util, sys
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("evaluate_enrichment", ROOT / "scripts" / "evaluate_enrichment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _enr_and_gold(annotator="Hajar"):
    enr = pd.DataFrame({"book_id": ["1", "2", "3", "4"],
                        "genres": [["Fantasy"], ["Romance", "Horror"], ["History"], ["Horror"]],
                        "themes": [["Friendship"], ["Revenge"], ["War & conflict"], ["Survival", "Grief & loss"]]})
    gold = pd.DataFrame({"book_id": ["1", "2", "3", "4"], "title": list("ABCD"),
                         "gold_genres": ["Fantasy | Young Adult", "Horror", "Fantasy", "Horror"],
                         "gold_themes": ["Friendship", "Grief & loss", "War & conflict", "Survival"],
                         "annotator": [annotator] * 4})
    return enr, gold


def test_evaluate_set_metrics_and_ci():
    ev = _evaluate_module()
    enr, gold = _enr_and_gold()
    res = ev.evaluate_set(enr, gold)
    g = res["metrics"]["genres"]
    assert res["n"] == 4 and g["n"] == 4 and res["reviewed"]
    assert g["hit"][0] == 0.75                                   # 3 livres sur 4 ont >= 1 bon genre
    assert g["precision"][0] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    lo, hi = g["hit"][1], g["hit"][2]
    assert lo <= 0.75 <= hi


def test_unreviewed_annotation_is_flagged():
    ev = _evaluate_module()
    enr, gold = _enr_and_gold(annotator="LLM (Claude), à relire")
    res = ev.evaluate_set(enr, gold)
    assert not res["reviewed"] and res["n_unreviewed"] == 4


def test_inject_block_replaces_only_marked_zone():
    ev = _evaluate_module()
    enr, gold = _enr_and_gold()
    res = {"dev": ev.evaluate_set(enr, gold), "test": ev.evaluate_set(enr, gold)}
    block = ev.render_block(res)
    text = f"avant\n{ev.START}\nancien\n{ev.END}\naprès"
    out = ev.inject_block(text, block)
    assert out.startswith("avant\n") and out.endswith("\naprès") and "ancien" not in out and "jamais vus" in out
    assert "[x" not in out
    with pytest.raises(ValueError):
        ev.inject_block("pas de marqueurs", block)
