"""
encoders.py : les 4 façons de transformer un texte en vecteur, comparées par P3 et P6.

    | nom           | représentation                         | type            | comprend les synonymes ? | comprend le contexte ? |
    |---------------|----------------------------------------|-----------------|--------------------------|------------------------|
    | "tfidf"       | sac de mots pondéré (lexical)          | creux, ~50 000 d | non                      | non                    |
    | "glove"       | moyenne des vecteurs de mots (statique)| dense, 100 d     | oui                      | non                    |
    | "glove-tfidf" | moyenne pondérée par l'IDF             | dense, 100 d     | oui                      | non                    |
    | "minilm"      | SBERT all-MiniLM-L6-v2 (contextuel)    | dense, 384 d     | oui                      | oui                    |
    | "mpnet"       | SBERT all-mpnet-base-v2 (contextuel)   | dense, 768 d     | oui                      | oui                    |

Tous les encodeurs ont la MÊME interface, pour pouvoir les comparer avec le même code :
    enc = get_encoder("glove", corpus=textes_des_livres)
    enc.encode(["a boy discovers he is a wizard"])   # -> vecteurs L2-normalisés
    cosine_scores(q, B)                              # -> similarités (marche aussi pour TF-IDF creux)

Et `compare_encoders` produit le tableau « sémantique vs mots-clés » du notebook 03 :
à quel RANG chaque méthode place-t-elle le livre attendu pour une requête paraphrasée ?
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.embeddings import EmbeddingEngine, l2_normalize, resolve_model_name
from src.preprocessing import preprocess_many_for_classic

GLOVE_MODEL = "glove-wiki-gigaword-100"   # GloVe entraîné sur Wikipédia + Gigaword (400 000 mots, 100 dim)
ENCODER_NAMES = ("tfidf", "glove", "glove-tfidf", "minilm", "mpnet")


# --------------------------------------------------------------------------- #
# Similarité commune (dense ou creux)
# --------------------------------------------------------------------------- #

def cosine_scores(queries: Any, docs: Any) -> np.ndarray:
    """Cosinus entre chaque requête et chaque document -> tableau dense (n_requêtes, n_docs).

    Les encodeurs renvoient des vecteurs DÉJÀ normalisés, donc cosinus = produit scalaire.
    Fonctionne pour des matrices numpy (GloVe, SBERT) comme pour des matrices creuses
    scipy (TF-IDF : 12 000 x 50 000 en dense prendrait ~2,4 Go, en creux quelques Mo).
    """
    scores = queries @ docs.T
    if sparse.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. TF-IDF : baseline LEXICALE
# --------------------------------------------------------------------------- #

class TfidfEncoder:
    """Sac de mots pondéré TF-IDF.

    Poids d'un mot dans un texte = TF (fréquence dans le texte) x IDF (rareté dans le catalogue).
    Deux textes sont proches s'ils PARTAGENT des mots rares. Limite assumée : « wizard » et
    « sorcerer » sont deux dimensions différentes -> similarité nulle sans mot commun.
    C'est précisément ce que la démo « sémantique vs mots-clés » doit montrer.
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2,
                 max_df: float = 0.5, max_features: int = 50_000) -> None:
        # ngram_range=(1,2) : mots seuls + paires de mots (« magic school »)
        # min_df=2 : un mot vu dans un seul livre n'aide pas à comparer -> ignoré
        # max_df=0.5 : un mot présent dans plus de 50 % des livres n'est pas discriminant
        # sublinear_tf : 1 + log(tf), pour qu'un mot répété 10 fois ne pèse pas 10 fois plus
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,  # texte déjà prétraité
            ngram_range=ngram_range, min_df=min_df, max_df=max_df,
            max_features=max_features, sublinear_tf=True, dtype=np.float32,
        )
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "TfidfEncoder":
        """Apprend le vocabulaire et les IDF sur le catalogue (prétraitement « classique » spaCy)."""
        self.vectorizer.fit(preprocess_many_for_classic(list(corpus)))
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> sparse.csr_matrix:
        """Textes -> matrice CREUSE (n, taille du vocabulaire), lignes de norme 1."""
        if not self._fitted:
            raise RuntimeError("TfidfEncoder doit d'abord être entraîné : .fit(corpus)")
        return self.vectorizer.transform(preprocess_many_for_classic(list(texts))).tocsr()


# --------------------------------------------------------------------------- #
# 2. GloVe : embeddings de mots STATIQUES
# --------------------------------------------------------------------------- #

class GloveEncoder:
    """Vecteur d'un texte = moyenne des vecteurs GloVe de ses mots.

    GloVe apprend un vecteur par mot à partir des co-occurrences sur Wikipédia : « wizard » et
    « sorcerer » ont des vecteurs proches, donc les synonymes sont gérés. Mais chaque mot a UN
    seul vecteur quel que soit le contexte (« bank » rivière = « bank » banque) et la moyenne
    perd l'ordre des mots et les négations.

    weighting="mean"  : moyenne simple ;
    weighting="tfidf" : moyenne pondérée par l'IDF (les mots rares, plus informatifs, pèsent plus).
    """

    def __init__(self, model_name: str = GLOVE_MODEL, weighting: str = "mean",
                 keyed_vectors: Any = None) -> None:
        if weighting not in ("mean", "tfidf"):
            raise ValueError("weighting doit valoir 'mean' ou 'tfidf'")
        self.model_name = model_name
        self.weighting = weighting
        self.name = "glove" if weighting == "mean" else "glove-tfidf"
        self._kv = keyed_vectors        # injectable pour les tests
        self._idf: dict[str, float] = {}

    @property
    def kv(self):
        """Vecteurs GloVe chargés au premier usage (téléchargement ~128 Mo la 1re fois)."""
        if self._kv is None:
            import gensim.downloader as api
            self._kv = api.load(self.model_name)
        return self._kv

    def fit(self, corpus: Iterable[str]) -> "GloveEncoder":
        """Calcule les IDF sur le catalogue (utile seulement pour weighting='tfidf')."""
        if self.weighting == "tfidf":
            vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
            vec.fit(preprocess_many_for_classic(list(corpus)))
            self._idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Textes -> matrice dense (n, 100), lignes de norme 1 (vecteur nul si aucun mot connu)."""
        kv = self.kv
        default_idf = max(self._idf.values()) if self._idf else 1.0  # mot jamais vu = très rare
        token_lists = preprocess_many_for_classic(list(texts), as_tokens=True)
        out = np.zeros((len(token_lists), kv.vector_size), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            known = [t for t in tokens if t in kv]          # on ignore les mots hors vocabulaire
            if not known:
                continue
            vecs = np.stack([kv[t] for t in known])
            if self.weighting == "tfidf":
                w = np.array([self._idf.get(t, default_idf) for t in known], dtype=np.float32)
                out[i] = (vecs * w[:, None]).sum(0) / w.sum()
            else:
                out[i] = vecs.mean(0)
        return l2_normalize(out)


# --------------------------------------------------------------------------- #
# 3. SBERT : embeddings CONTEXTUELS (adaptateur vers EmbeddingEngine)
# --------------------------------------------------------------------------- #

class SbertEncoder:
    """Adaptateur : donne à EmbeddingEngine la même interface (name, fit, encode) que les baselines."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, engine: EmbeddingEngine | None = None,
                 long_text: str = "truncate") -> None:
        self.engine = engine or EmbeddingEngine(model_name, long_text=long_text)
        self.name = self.engine.model_name.split("/")[-1]

    def fit(self, corpus: Iterable[str]) -> "SbertEncoder":
        return self  # modèle pré-entraîné : rien à apprendre sur le catalogue

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.engine.encode(list(texts))


# --------------------------------------------------------------------------- #
# 4. Fabrique commune (pour P6)
# --------------------------------------------------------------------------- #

def get_encoder(name: str, corpus: Iterable[str] | None = None, **kwargs):
    """Renvoie un encodeur prêt à l'emploi : "tfidf", "glove", "glove-tfidf", "minilm", "mpnet"
    (ou tout nom de modèle sentence-transformers).

    `corpus` (textes des livres) est OBLIGATOIRE pour "tfidf" et "glove-tfidf", qui apprennent
    des statistiques sur le catalogue. Il est ignoré par les autres.

    >>> enc = get_encoder("tfidf", corpus=books["embed_text"])     # doctest: +SKIP
    >>> enc.encode(["a wizard school"]).shape                       # doctest: +SKIP
    """
    key = name.lower()
    if key == "tfidf":
        enc = TfidfEncoder(**kwargs)
    elif key in ("glove", "glove-tfidf"):
        enc = GloveEncoder(weighting="mean" if key == "glove" else "tfidf", **kwargs)
    else:
        return SbertEncoder(resolve_model_name(name), **kwargs)
    if key in ("tfidf", "glove-tfidf"):
        if corpus is None:
            raise ValueError(f"get_encoder({name!r}) a besoin de `corpus` (textes des livres)")
        enc.fit(corpus)
    return enc


# --------------------------------------------------------------------------- #
# 5. Comparaison « sémantique vs mots-clés »
# --------------------------------------------------------------------------- #

def compare_encoders(encoders: dict[str, Any], books: pd.DataFrame,
                     queries: list[tuple[str, str]], text_col: str = "embed_text") -> pd.DataFrame:
    """Pour chaque (requête, book_id attendu) et chaque encodeur : RANG du livre attendu.

    Rang 1 = le livre attendu est le plus proche de la requête parmi tout le catalogue.
    Rang = 1 + nombre de livres qui ont un score STRICTEMENT supérieur ; si le score est nul
    (aucun mot commun), rang moyen parmi tous les livres à 0.

    Retourne un DataFrame long : query, target, encoder, rank, score, top1_title, seconds.
    """
    ids = books["book_id"].astype(str).tolist()
    pos = {b: i for i, b in enumerate(ids)}
    texts = books[text_col].fillna("").astype(str).tolist()
    rows = []
    for enc_name, enc in encoders.items():
        t0 = time.perf_counter()
        doc_emb = enc.encode(texts)                       # tout le catalogue, une seule fois
        q_emb = enc.encode([q for q, _ in queries])
        scores = cosine_scores(q_emb, doc_emb)            # (n_requêtes, n_livres)
        elapsed = time.perf_counter() - t0
        for k, (query, target) in enumerate(queries):
            if str(target) not in pos:
                raise KeyError(f"book_id {target!r} absent du catalogue")
            s = scores[k]
            t = pos[str(target)]
            n_better = int((s > s[t]).sum())               # livres strictement mieux classés
            if s[t] == 0:
                # Score nul = aucun mot commun (cas typique de TF-IDF). Des milliers de livres sont
                # alors ex aequo à 0 : on donne le rang MOYEN du groupe (rang « au hasard »).
                rank = n_better + (int((s == 0).sum()) + 1) / 2
            else:
                rank = n_better + 1
            rows.append({
                "query": query, "target": books.iloc[t]["title"], "encoder": enc_name,
                "rank": rank, "score": round(float(s[t]), 3),
                "top1_title": books.iloc[int(np.argmax(s))]["title"], "seconds": round(elapsed, 2),
            })
    return pd.DataFrame(rows)
