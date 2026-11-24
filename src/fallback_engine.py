"""
Moteur d'embeddings DE SECOURS pour P4, en attendant ``src/embeddings.py`` (P3).

Il respecte exactement le contrat ``EmbeddingEngine`` de PLAN_PROJET.md §4.4 :
    encode(texts) -> np.ndarray L2-normalisé
    book_matrix() -> (N, d) aligné sur le DataFrame des livres
    label_matrix(block) -> (labels, matrice)
    search(query_vec, k) -> (indices, scores)
    best_sentence(query_vec, text) -> (phrase, score)

Dès que P3 livre ``EmbeddingEngine``, ``recommender.get_engine()`` l'utilise
automatiquement et ce fichier ne sert plus (il reste utile pour les tests).

Deux encodeurs sont disponibles :
- ``"sbert"`` (défaut) : all-MiniLM-L6-v2, comme le reste du projet ;
- ``"lsa"`` : TF-IDF + SVD tronquée entraînée sur le catalogue. Mode HORS-LIGNE
  uniquement pour tester le code sans télécharger de modèle (ce n'est PAS un
  embedding contextuel : ne pas l'utiliser pour la démo ni pour le rapport).
  Activable avec la variable d'environnement ``GENDERBOOK_ENCODER=lsa``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Découpage en phrases : après . ! ? suivi d'un espace et d'une majuscule/guillemet/chiffre
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“'A-Z0-9])")


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    """Découpe un résumé en phrases (les fragments très courts sont fusionnés au précédent)."""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(text or "")) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(p) < min_chars:
            out[-1] = f"{out[-1]} {p}"   # « Yes. » seul n'a pas de sens : on le recolle
        else:
            out.append(p)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne à la norme 1 (cosinus = produit scalaire)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class LsaEncoder:
    """Encodeur hors-ligne TF-IDF + SVD (LSA). Uniquement pour les tests sans Internet."""

    def __init__(self, corpus: list[str], dim: int = 256, seed: int = config.RANDOM_SEED):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = f"lsa{dim}"
        # Étape 1 : vocabulaire TF-IDF appris sur les résumés du catalogue
        self.tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                                     sublinear_tf=True, ngram_range=(1, 2), max_features=60_000)
        x = self.tfidf.fit_transform(corpus)
        # Étape 2 : réduction de dimension (capte des co-occurrences, un peu de « sémantique »)
        self.svd = TruncatedSVD(n_components=min(dim, x.shape[1] - 1), random_state=seed)
        self.svd.fit(x)

    def encode(self, texts: list[str], **_) -> np.ndarray:
        return l2_normalize(self.svd.transform(self.tfidf.transform(list(texts))))


class LocalEngine:
    """Implémentation minimale du contrat EmbeddingEngine (P3) pour débloquer P4.

    Args:
        books: catalogue (DataFrame de ``data_loader.load_books()``), dans l'ordre
            qui servira d'index partout.
        blocks: référentiel ``{bloc: {label: phrase}}`` (blocks.json).
        encoder: "sbert", "lsa" ou un objet possédant ``encode(texts)``.
        model_name: modèle SBERT à utiliser.
        cache: True = sauvegarde la matrice des livres dans data/embeddings/.
    """

    def __init__(self, books: pd.DataFrame, blocks: dict[str, dict[str, str]] | None = None,
                 encoder: str | object | None = None, model_name: str = config.EMBEDDING_MODEL,
                 cache: bool = True):
        self.books = books.reset_index(drop=True)
        self.blocks = blocks or {}
        self.cache = cache
        encoder = encoder or os.environ.get("GENDERBOOK_ENCODER", "sbert")
        if encoder == "sbert":
            from src.enrichment import SbertEncoder  # encodeur SBERT de P2 (chargement paresseux)
            self._encoder = SbertEncoder(model_name)
            self.model_name = model_name.split("/")[-1]
        elif encoder == "lsa":
            self._encoder = LsaEncoder(self.books["embed_text"].fillna("").tolist())
            self.model_name = self._encoder.model_name
        else:
            self._encoder = encoder
            self.model_name = getattr(encoder, "model_name", type(encoder).__name__)
        self._book_matrix: np.ndarray | None = None
        self._label_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    # ------------------------------------------------------------------ encode
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode une liste de textes -> matrice (n, d) L2-normalisée."""
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            emb = self._encoder.encode(texts, batch_size=64, show_progress=len(texts) > 500)
        except TypeError:  # encodeur qui n'accepte pas ces arguments
            emb = self._encoder.encode(texts)
        return l2_normalize(emb)

    @property
    def dim(self) -> int:
        return int(self.book_matrix().shape[1])

    # ------------------------------------------------------------- book_matrix
    def _cache_paths(self) -> tuple[Path, Path]:
        # Le nom dépend du modèle ET des book_id : un autre catalogue => un autre fichier
        ids = self.books["book_id"].astype(str).tolist()
        h = hashlib.md5("|".join(ids).encode()).hexdigest()[:8]
        stem = f"p4_{self.model_name}_{len(ids)}_{h}_books"
        return config.DATA_EMBEDDINGS / f"{stem}.npy", config.DATA_EMBEDDINGS / f"{stem}_ids.json"

    def book_matrix(self) -> np.ndarray:
        """Embeddings de ``embed_text`` pour tous les livres, (N, d), alignés sur ``books``."""
        if self._book_matrix is not None:
            return self._book_matrix
        npy, ids_json = self._cache_paths()
        ids = self.books["book_id"].astype(str).tolist()
        # Étape 1 : cache disque (le 1er encodage du catalogue complet prend quelques minutes sur CPU)
        if self.cache and npy.exists() and ids_json.exists() \
                and json.loads(ids_json.read_text(encoding="utf-8")) == ids:
            self._book_matrix = np.load(npy)
            return self._book_matrix
        # Étape 2 : encodage à la volée
        mat = self.encode(self.books["embed_text"].fillna("").tolist())
        if self.cache:
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, mat)
            ids_json.write_text(json.dumps(ids), encoding="utf-8")
        self._book_matrix = mat
        return mat

    # ------------------------------------------------------------ label_matrix
    def label_matrix(self, block: str) -> tuple[list[str], np.ndarray]:
        """Embeddings des PHRASES descriptives des labels d'un bloc (pas des mots seuls).

        Pour les thèmes, on reprend la méthode de P2 (moyenne phrase + variantes
        de theme_variants.json) pour que nos labels soient comparables aux siens.
        """
        if block in self._label_cache:
            return self._label_cache[block]
        labels = list(self.blocks.get(block, {}))
        if not labels:
            out: tuple[list[str], np.ndarray] = ([], np.zeros((0, self.dim), dtype=np.float32))
        else:
            mat = None
            if block == "Themes":
                try:
                    from src.enrichment import load_theme_variants
                    variants = load_theme_variants(themes=self.blocks["Themes"])
                    rows = []
                    for lab in labels:
                        v = self.encode([self.blocks[block][lab], *variants.get(lab, [])])
                        rows.append(v.mean(0))
                    mat = l2_normalize(np.stack(rows))
                except Exception:  # fichier de variantes absent : phrase seule
                    mat = None
            if mat is None:
                mat = self.encode([self.blocks[block][lab] for lab in labels])
            out = (labels, mat)
        self._label_cache[block] = out
        return out

    # ------------------------------------------------------------------ search
    def search(self, query_vec: np.ndarray, k: int = config.RETRIEVAL_TOP_K
               ) -> tuple[np.ndarray, np.ndarray]:
        """k plus proches livres par cosinus (= produit scalaire, vecteurs normalisés)."""
        scores = self.book_matrix() @ np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]          # sélection partielle O(N)
        idx = idx[np.argsort(-scores[idx], kind="stable")]  # puis tri des k retenus
        return idx, scores[idx]

    # ----------------------------------------------------------- best_sentence
    def best_sentence(self, query_vec: np.ndarray, text: str) -> tuple[str, float]:
        """Phrase du résumé la plus proche de la requête (explicabilité)."""
        sents = split_sentences(text)
        if not sents:
            return "", 0.0
        sims = self.encode(sents) @ np.asarray(query_vec, dtype=np.float32)
        i = int(np.argmax(sims))
        return sents[i], float(sims[i])
