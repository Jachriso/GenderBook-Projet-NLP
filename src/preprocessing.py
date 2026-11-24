"""
preprocessing.py: cleaning and preparation of text for the semantic engine (P3).

Why TWO different preparations?
-------------------------------
1. For CONTEXTUAL models (SBERT) -> `preprocess_for_transformer`
   A transformer reads the whole sentence and interprets each word in its context.
   "Function" words (not, no, without...) and punctuation CARRY meaning:
   "a story with no romance" and "a story with romance" must stay different.
   We therefore do a MINIMAL cleaning (spaces, stray characters), nothing more.
   The model's tokenizer (WordPiece) handles case and unknown words itself.

2. For CLASSIC models (TF-IDF, GloVe) -> `preprocess_for_classic`
   These models treat the text as a "bag of words" without order or context.
   Very frequent words (the, a, of...) add nothing and drown the signal;
   variants of the same word (wizard / wizards) must be grouped together.
   We therefore apply: lowercasing, tokenisation, removal of stopwords and
   punctuation, lemmatisation (spaCy).

3. `split_sentences` splits a blurb into sentences: used for explainability
   (finding THE sentence of the blurb closest to the user's request).
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Iterable

import spacy

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# English spaCy model (the catalogue is in English). Small model = fast on CPU.
SPACY_MODEL = "en_core_web_sm"

# Minimum length (in characters) of a sentence kept by split_sentences:
# below that, they are often fragments ("Vol. 1.", "Book 2") without useful meaning.
MIN_SENTENCE_CHARS = 20

# Regular expressions compiled only once (faster in a loop)
_RE_HTML_TAG = re.compile(r"<[^>]+>")            # leftover HTML tags
_RE_URL = re.compile(r"https?://\S+|www\.\S+")   # web links
_RE_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # invisible control characters
_RE_SPACES = re.compile(r"\s+")                  # runs of spaces / line breaks
_RE_REPEAT_PUNCT = re.compile(r"([!?.,;:\-*_~=])\1{2,}")  # "!!!!" -> "!", "-----" -> "-"


# --------------------------------------------------------------------------- #
# 1. Minimal cleaning for SBERT
# --------------------------------------------------------------------------- #

def preprocess_for_transformer(text: str | None) -> str:
    """MINIMAL cleaning of a text before encoding by a transformer (SBERT).

    Case, stopwords and punctuation are kept: the model needs them
    to understand the context (negations, sentence structure...).

    Steps:
        1. handle missing values (None / NaN -> empty string);
        2. decode HTML entities (&amp; -> &, &quot; -> ");
        3. remove leftover HTML tags and URLs;
        4. normalise Unicode (NFKC: ligatures, non-breaking spaces...);
        5. standardise typographic quotes / apostrophes;
        6. reduce repeated punctuation and multiple spaces.

    >>> preprocess_for_transformer("  A <b>dark</b>   tale&amp;more!!!  ")
    'A dark tale&more!'
    """
    # Step 1: missing values (pandas returns NaN, which is a float)
    if text is None or not isinstance(text, str):
        return ""

    # Step 2: HTML entities -> normal characters
    text = html.unescape(text)

    # Step 3: HTML tags and links (meaningless for semantic comparison)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)

    # Step 4: Unicode normalisation (e.g. "ﬁ" -> "fi", non-breaking space -> space)
    text = unicodedata.normalize("NFKC", text)
    text = _RE_CTRL.sub(" ", text)

    # Step 5: typographic quotes and apostrophes -> simple versions
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))

    # Step 6: "!!!" -> "!" and multiple spaces -> a single space
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. Full preparation for TF-IDF / GloVe
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_nlp() -> "spacy.language.Language":
    """Load the spaCy model only ONCE (lru_cache): loading takes ~1 s.

    The parser and named-entity recognition (NER) are disabled, as they are useless here:
    we only need tokenisation, stopwords and lemmatisation.
    This divides the processing time by ~3.
    """
    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError as err:  # model not downloaded
        raise OSError(
            f"Le modèle spaCy '{SPACY_MODEL}' n'est pas installé. "
            f"Lance : python -m spacy download {SPACY_MODEL}"
        ) from err


def _tokens_from_doc(doc) -> list[str]:
    """Keep the useful lemmas of an already analysed spaCy document.

    A token is kept if it is NEITHER a stopword, NOR punctuation,
    NOR a space, NOR a number, and if it contains at least 2 letters.
    """
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space or tok.like_num:
            continue
        lemma = tok.lemma_.lower().strip()           # canonical form: wizards -> wizard
        if len(lemma) < 2 or not any(c.isalpha() for c in lemma):
            continue
        if lemma in _get_nlp().Defaults.stop_words:  # a lemma can turn into a stopword
            continue
        tokens.append(lemma)
    return tokens


def preprocess_for_classic(text: str | None, as_tokens: bool = False) -> str | list[str]:
    """"Bag of words" preparation for TF-IDF and GloVe.

    Steps: minimal cleaning -> lowercase -> spaCy tokenisation ->
    removal of stopwords / punctuation / numbers -> lemmatisation.

    Args:
        text: raw text.
        as_tokens: True -> returns the list of tokens (handy for GloVe),
                   False -> returns a string "tok1 tok2 ..." (handy for TF-IDF).

    >>> preprocess_for_classic("The young wizards were running to the magic schools!")
    'young wizard run magic school'
    """
    # Start from the minimal cleaning, then lowercase
    cleaned = preprocess_for_transformer(text).lower()
    tokens = _tokens_from_doc(_get_nlp()(cleaned))
    return tokens if as_tokens else " ".join(tokens)


def preprocess_many_for_classic(
    texts: Iterable[str | None], as_tokens: bool = False, batch_size: int = 256
) -> list[str] | list[list[str]]:
    """FAST version for a whole catalogue (12,000 books).

    `nlp.pipe` processes the texts in batches instead of one call per text:
    about 5 to 10 times faster than a loop over preprocess_for_classic.
    """
    cleaned = (preprocess_for_transformer(t).lower() for t in texts)
    results = []
    for doc in _get_nlp().pipe(cleaned, batch_size=batch_size):
        tokens = _tokens_from_doc(doc)
        results.append(tokens if as_tokens else " ".join(tokens))
    return results


# --------------------------------------------------------------------------- #
# 3. Sentence splitting (explainability)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_sentencizer() -> "spacy.language.Language":
    """RULE-based sentence splitter (spacy.blank + sentencizer).

    Faster than the statistical parser and good enough for book blurbs.
    It handles common abbreviations (Mr., Dr., U.S.) better than a simple split('.').
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str | None, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Split a blurb into clean sentences.

    Used by EmbeddingEngine.best_sentence: each sentence is encoded and the one
    closest to the user's request is returned.

    Sentences that are too short (< min_chars) are merged with the next one
    rather than removed, so that no information is lost.

    >>> split_sentences("Harry Potter is an orphan boy. He learns he is a wizard. Ok.")
    ['Harry Potter is an orphan boy.', 'He learns he is a wizard. Ok.']
    """
    cleaned = preprocess_for_transformer(text)
    if not cleaned:
        return []

    # Raw split by the sentencizer
    raw = [s.text.strip() for s in _get_sentencizer()(cleaned).sents if s.text.strip()]

    # Merge fragments that are too short with the previous (or next) sentence
    sentences: list[str] = []
    buffer = ""
    for sent in raw:
        buffer = f"{buffer} {sent}".strip() if buffer else sent
        if len(buffer) >= min_chars:
            sentences.append(buffer)
            buffer = ""
    if buffer:  # a short fragment is left at the end: attach it to the last sentence
        if sentences:
            sentences[-1] = f"{sentences[-1]} {buffer}"
        else:
            sentences.append(buffer)
    return sentences
