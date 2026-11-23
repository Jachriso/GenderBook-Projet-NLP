"""
preprocessing.py : nettoyage et préparation du texte pour le moteur sémantique (P3).

Pourquoi DEUX préparations différentes ?
-----------------------------------------
1. Pour les modèles CONTEXTUELS (SBERT) -> `preprocess_for_transformer`
   Un transformer lit la phrase entière et interprète chaque mot selon son contexte.
   Les mots « vides » (not, no, without...) et la ponctuation PORTENT du sens :
   "a story with no romance" et "a story with romance" doivent rester différents.
   On fait donc un nettoyage MINIMAL (espaces, caractères parasites), rien de plus.
   Le tokenizer du modèle (WordPiece) gère lui-même la casse et les mots inconnus.

2. Pour les modèles CLASSIQUES (TF-IDF, GloVe) -> `preprocess_for_classic`
   Ces modèles traitent le texte comme un « sac de mots » sans ordre ni contexte.
   Les mots très fréquents (the, a, of...) n'apportent rien et noient le signal ;
   les variantes d'un même mot (wizard / wizards) doivent être regroupées.
   On applique donc : minuscules, tokenisation, suppression des stopwords et de la
   ponctuation, lemmatisation (spaCy).

3. `split_sentences` découpe un résumé en phrases : sert à l'explicabilité
   (trouver LA phrase du résumé la plus proche de la demande utilisateur).
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Iterable

import spacy

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

# Modèle spaCy anglais (le catalogue est en anglais). Petit modèle = rapide sur CPU.
SPACY_MODEL = "en_core_web_sm"

# Longueur minimale (en caractères) d'une phrase gardée par split_sentences :
# en dessous, ce sont souvent des fragments ("Vol. 1.", "Book 2") sans sens utile.
MIN_SENTENCE_CHARS = 20

# Expressions régulières compilées une seule fois (plus rapide en boucle)
_RE_HTML_TAG = re.compile(r"<[^>]+>")            # balises HTML résiduelles
_RE_URL = re.compile(r"https?://\S+|www\.\S+")   # liens web
_RE_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # caractères de contrôle invisibles
_RE_SPACES = re.compile(r"\s+")                  # suites d'espaces / retours ligne
_RE_REPEAT_PUNCT = re.compile(r"([!?.,;:\-*_~=])\1{2,}")  # "!!!!" -> "!", "-----" -> "-"


# --------------------------------------------------------------------------- #
# 1. Nettoyage minimal pour SBERT
# --------------------------------------------------------------------------- #

def preprocess_for_transformer(text: str | None) -> str:
    """Nettoyage MINIMAL d'un texte avant encodage par un transformer (SBERT).

    On garde la casse, les stopwords et la ponctuation : le modèle en a besoin
    pour comprendre le contexte (négations, structure de la phrase...).

    Étapes :
        1. gérer les valeurs manquantes (None / NaN -> chaîne vide) ;
        2. décoder les entités HTML (&amp; -> &, &quot; -> ") ;
        3. retirer les balises HTML et les URL résiduelles ;
        4. normaliser l'Unicode (NFKC : ligatures, espaces insécables...) ;
        5. uniformiser les guillemets / apostrophes typographiques ;
        6. réduire la ponctuation répétée et les espaces multiples.

    >>> preprocess_for_transformer("  A <b>dark</b>   tale&amp;more!!!  ")
    'A dark tale&more!'
    """
    # Étape 1 : valeurs manquantes (pandas renvoie NaN, qui est un float)
    if text is None or not isinstance(text, str):
        return ""

    # Étape 2 : entités HTML -> caractères normaux
    text = html.unescape(text)

    # Étape 3 : balises HTML et liens (aucun sens pour la comparaison sémantique)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)

    # Étape 4 : normalisation Unicode (ex. "ﬁ" -> "fi", espace insécable -> espace)
    text = unicodedata.normalize("NFKC", text)
    text = _RE_CTRL.sub(" ", text)

    # Étape 5 : guillemets et apostrophes typographiques -> versions simples
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))

    # Étape 6 : "!!!" -> "!" et espaces multiples -> un seul espace
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. Préparation complète pour TF-IDF / GloVe
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_nlp() -> "spacy.language.Language":
    """Charge le modèle spaCy UNE seule fois (lru_cache) : le chargement prend ~1 s.

    On désactive le parser et la reconnaissance d'entités (NER), inutiles ici :
    on n'a besoin que de la tokenisation, des stopwords et de la lemmatisation.
    Cela divise le temps de traitement par ~3.
    """
    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError as err:  # modèle non téléchargé
        raise OSError(
            f"Le modèle spaCy '{SPACY_MODEL}' n'est pas installé. "
            f"Lance : python -m spacy download {SPACY_MODEL}"
        ) from err


def _tokens_from_doc(doc) -> list[str]:
    """Garde les lemmes utiles d'un document spaCy déjà analysé.

    Un token est gardé s'il n'est NI un stopword, NI de la ponctuation,
    NI un espace, NI un nombre, et s'il contient au moins 2 lettres.
    """
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space or tok.like_num:
            continue
        lemma = tok.lemma_.lower().strip()           # forme canonique : wizards -> wizard
        if len(lemma) < 2 or not any(c.isalpha() for c in lemma):
            continue
        if lemma in _get_nlp().Defaults.stop_words:  # un lemme peut devenir un stopword
            continue
        tokens.append(lemma)
    return tokens


def preprocess_for_classic(text: str | None, as_tokens: bool = False) -> str | list[str]:
    """Préparation « sac de mots » pour TF-IDF et GloVe.

    Étapes : nettoyage minimal -> minuscules -> tokenisation spaCy ->
    suppression stopwords / ponctuation / nombres -> lemmatisation.

    Args:
        text: texte brut.
        as_tokens: True -> renvoie la liste de tokens (pratique pour GloVe),
                   False -> renvoie une chaîne "tok1 tok2 ..." (pratique pour TF-IDF).

    >>> preprocess_for_classic("The young wizards were running to the magic schools!")
    'young wizard run magic school'
    """
    # On repart du nettoyage minimal, puis on passe en minuscules
    cleaned = preprocess_for_transformer(text).lower()
    tokens = _tokens_from_doc(_get_nlp()(cleaned))
    return tokens if as_tokens else " ".join(tokens)


def preprocess_many_for_classic(
    texts: Iterable[str | None], as_tokens: bool = False, batch_size: int = 256
) -> list[str] | list[list[str]]:
    """Version RAPIDE pour tout un catalogue (12 000 livres).

    `nlp.pipe` traite les textes par lots au lieu d'un appel par texte :
    environ 5 à 10 fois plus rapide qu'une boucle sur preprocess_for_classic.
    """
    cleaned = (preprocess_for_transformer(t).lower() for t in texts)
    results = []
    for doc in _get_nlp().pipe(cleaned, batch_size=batch_size):
        tokens = _tokens_from_doc(doc)
        results.append(tokens if as_tokens else " ".join(tokens))
    return results


# --------------------------------------------------------------------------- #
# 3. Découpage en phrases (explicabilité)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_sentencizer() -> "spacy.language.Language":
    """Découpeur de phrases à base de RÈGLES (spacy.blank + sentencizer).

    Plus rapide que le parser statistique et suffisant pour des résumés de livres.
    Il gère les abréviations courantes (Mr., Dr., U.S.) mieux qu'un simple split('.').
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str | None, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Découpe un résumé en phrases propres.

    Utilisé par EmbeddingEngine.best_sentence : on encode chaque phrase et on
    renvoie celle qui est la plus proche de la demande de l'utilisateur.

    Les phrases trop courtes (< min_chars) sont fusionnées avec la suivante
    plutôt que supprimées, pour ne perdre aucune information.

    >>> split_sentences("Harry Potter is an orphan boy. He learns he is a wizard. Ok.")
    ['Harry Potter is an orphan boy.', 'He learns he is a wizard. Ok.']
    """
    cleaned = preprocess_for_transformer(text)
    if not cleaned:
        return []

    # Découpage brut par le sentencizer
    raw = [s.text.strip() for s in _get_sentencizer()(cleaned).sents if s.text.strip()]

    # Fusion des fragments trop courts avec la phrase précédente (ou suivante)
    sentences: list[str] = []
    buffer = ""
    for sent in raw:
        buffer = f"{buffer} {sent}".strip() if buffer else sent
        if len(buffer) >= min_chars:
            sentences.append(buffer)
            buffer = ""
    if buffer:  # reste un fragment court à la fin : on le colle à la dernière phrase
        if sentences:
            sentences[-1] = f"{sentences[-1]} {buffer}"
        else:
            sentences.append(buffer)
    return sentences
