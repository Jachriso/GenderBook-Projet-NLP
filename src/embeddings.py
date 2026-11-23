"""
embeddings.py : moteur sémantique du projet (Personne 3).

Rôle : transformer n'importe quel texte en VECTEUR (embedding) et mesurer la proximité
de sens entre deux textes par similarité COSINUS. Toute l'équipe passe par ici :
    - P2 (enrichissement)   : engine.encode(...)            (compatible avec son SbertEncoder)
    - P4 (scoring)          : engine.search, engine.label_matrix, engine.best_sentence
    - P5 (app)              : engine chargé une fois (st.cache_resource)
    - P6 (évaluation)       : comparaison des encodeurs

Choix techniques (à justifier dans le rapport) :
    * Modèle par défaut : sentence-transformers/all-MiniLM-L6-v2 (config.EMBEDDING_MODEL).
      SBERT = BERT affiné pour que deux phrases de même sens aient des vecteurs proches
      (entraîné sur plus d'un milliard de paires de phrases). 384 dimensions, rapide sur CPU.
    * Vecteurs NORMALISÉS (norme L2 = 1). Conséquence : cosinus(a, b) = a · b.
      Le score d'un livre pour une requête est donc un simple produit matrice-vecteur :
      12 000 livres comparés en une seule opération numpy (quelques millisecondes).
    * Cache disque : les 12 000 livres sont encodés UNE fois puis rechargés depuis un .npy.
      Un fichier d'identifiants (book_ids.json) garantit que la ligne i du .npy correspond
      bien au livre i du catalogue (sinon on ré-encode automatiquement).
    * Textes longs : MiniLM ne lit que 256 tokens (le reste est ignoré sans avertissement).
      Deux stratégies : "truncate" (par défaut) ou "chunks" (découpage en morceaux de
      phrases complètes, encodage de chaque morceau, moyenne des vecteurs).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import preprocess_for_transformer, split_sentences

# Raccourcis de noms de modèles (pratique dans les notebooks et pour P6)
MODEL_ALIASES = {
    "minilm": config.EMBEDDING_MODEL,       # 384 dim, rapide
    "mpnet": config.EMBEDDING_MODEL_ALT,    # 768 dim, plus précis, ~5x plus lent
}
LONG_TEXT_MODES = ("truncate", "chunks")


# --------------------------------------------------------------------------- #
# Fonctions utilitaires
# --------------------------------------------------------------------------- #

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Ramène chaque ligne à une norme de 1 (pour que produit scalaire = cosinus).

    Les lignes nulles (texte vide) restent nulles au lieu de produire une division par zéro.
    """
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre les lignes de a (n, d) et de b (m, d) -> matrice (n, m).

    cos(a, b) = (a · b) / (||a|| · ||b||). On normalise d'abord, puis un seul produit
    matriciel. Valeurs dans [-1, 1] : 1 = même direction (même sens), 0 = sans rapport.
    """
    a = l2_normalize(np.atleast_2d(a))
    b = l2_normalize(np.atleast_2d(b))
    return a @ b.T


def model_slug(model_name: str) -> str:
    """Nom de fichier sûr à partir d'un nom de modèle.

    >>> model_slug("sentence-transformers/all-MiniLM-L6-v2")
    'all-MiniLM-L6-v2'
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.split("/")[-1])


def resolve_model_name(name: str) -> str:
    """Accepte un alias ('minilm', 'mpnet') ou un nom Hugging Face complet."""
    return MODEL_ALIASES.get(name.lower(), name)


# --------------------------------------------------------------------------- #
# Le moteur
# --------------------------------------------------------------------------- #

class EmbeddingEngine:
    """Moteur sémantique : encodage SBERT, cache des livres, recherche par cosinus.

    Contrat d'interface (PLAN §4.4) :
        encode(texts)              -> np.ndarray (n, d), L2-normalisé
        book_matrix()              -> np.ndarray (N, d), aligné sur le catalogue
        label_matrix(block)        -> (labels, np.ndarray (L, d))
        search(query_vec, k=300)   -> (indices, scores) triés par score décroissant
        best_sentence(query_vec, text) -> (phrase, score)

    Exemple :
        engine = EmbeddingEngine()                       # rien n'est chargé à ce stade
        q = engine.encode(["a boy discovers he is a wizard"])[0]
        idx, scores = engine.search(q, k=3)
        engine.books.iloc[idx][["title", "authors_str"]]
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        books: pd.DataFrame | None = None,
        long_text: str = "truncate",
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: Path | str = config.DATA_EMBEDDINGS,
        model: Any = None,
    ) -> None:
        """
        Args:
            model_name: nom Hugging Face ou alias ('minilm', 'mpnet').
            books: catalogue à indexer. None -> data_loader.load_books() au premier besoin
                   (books_enriched si disponible, sinon books_clean, sinon l'échantillon).
            long_text: "truncate" (le modèle coupe à max_seq_length) ou "chunks".
            device: "cuda", "cpu" ou None (= GPU s'il existe, sinon CPU).
            batch_size: nombre de textes encodés en parallèle (64 convient au CPU ;
                        monter à 128-256 sur GPU).
            cache_dir: dossier des .npy (data/embeddings/).
            model: modèle déjà chargé (sert aux tests avec un faux modèle, ou à partager
                   un modèle entre deux moteurs).
        """
        if long_text not in LONG_TEXT_MODES:
            raise ValueError(f"long_text doit valoir {LONG_TEXT_MODES}, reçu {long_text!r}")
        self.model_name = resolve_model_name(model_name)
        self.long_text = long_text
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self._model = model            # chargé paresseusement (voir propriété `model`)
        self._books = books
        self._book_matrix: np.ndarray | None = None
        self._labels: dict[str, tuple[list[str], np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # Chargement paresseux du modèle et du catalogue
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        """Le modèle SBERT n'est chargé qu'au PREMIER encodage (~2 s, ~90 Mo pour MiniLM).

        Importer ce module ou créer le moteur reste donc instantané (utile pour l'app).
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # import lourd : seulement ici
            device = self.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    @property
    def max_tokens(self) -> int:
        """Nombre maximal de tokens lus par le modèle (256 pour MiniLM, 384 pour MPNet)."""
        return int(self.model.max_seq_length)

    @property
    def dim(self) -> int:
        """Dimension des vecteurs (384 pour MiniLM, 768 pour MPNet)."""
        return int(self.encode(["dimension probe"]).shape[1])

    @property
    def books(self) -> pd.DataFrame:
        """Catalogue indexé (chargé au premier accès via data_loader de P1)."""
        if self._books is None:
            from src.data_loader import load_books
            self._books = load_books()
        return self._books

    @property
    def book_ids(self) -> list[str]:
        return self.books["book_id"].astype(str).tolist()

    # ------------------------------------------------------------------ #
    # 1. Encodage
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode une liste de textes -> matrice (n, d) float32, chaque ligne de norme 1.

        Étapes :
            1. nettoyage minimal (preprocess_for_transformer : on garde le contexte) ;
            2. encodage SBERT (troncature) ou par morceaux (mode "chunks") ;
            3. normalisation L2 (produit scalaire = cosinus).

        Signature compatible avec SbertEncoder de P2 (batch_size, show_progress).
        """
        if isinstance(texts, str):          # tolérance : un seul texte -> liste d'un élément
            texts = [texts]
        cleaned = [preprocess_for_transformer(t) for t in texts]
        if len(cleaned) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self.long_text == "chunks":
            return self._encode_chunked(cleaned, batch_size, show_progress)
        return self._encode_raw(cleaned, batch_size, show_progress)

    def _encode_raw(self, texts: list[str], batch_size: int | None, show_progress: bool) -> np.ndarray:
        """Appel direct au modèle (le modèle tronque lui-même à max_seq_length tokens)."""
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # norme 1 -> cosinus = produit scalaire
        )
        return np.asarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 2. Textes longs : comptage de tokens et découpage en morceaux
    # ------------------------------------------------------------------ #
    def count_tokens(self, texts: Iterable[str]) -> np.ndarray:
        """Nombre de tokens (WordPiece) de chaque texte, SANS les tokens spéciaux [CLS]/[SEP].

        Sert à mesurer combien de résumés dépassent la limite du modèle (notebook 03).
        """
        texts = [preprocess_for_transformer(t) for t in texts]
        ids = self.model.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return np.array([len(x) for x in ids], dtype=int)

    def chunk_text(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Découpe un texte en morceaux de PHRASES COMPLÈTES tenant dans la limite du modèle.

        On remplit chaque morceau phrase par phrase tant que le total reste <= max_tokens
        (limite du modèle moins 2 places pour [CLS] et [SEP]). Une phrase plus longue que
        la limite forme un morceau à elle seule (et sera tronquée par le modèle).
        """
        limit = (max_tokens or self.max_tokens) - 2
        sentences = split_sentences(text) or [text]
        lengths = self.count_tokens(sentences)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sent, n_tok in zip(sentences, lengths):
            if current and current_len + n_tok > limit:   # le morceau est plein : on le ferme
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += int(n_tok)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _encode_chunked(self, texts: list[str], batch_size: int | None, show_progress: bool) -> np.ndarray:
        """Mode "chunks" : vecteur du texte = moyenne (renormalisée) des vecteurs de ses morceaux.

        Tous les morceaux de tous les textes sont encodés en UN seul appel (plus rapide),
        puis regroupés par texte d'origine grâce à `owner`.
        """
        all_chunks: list[str] = []
        owner: list[int] = []                       # owner[k] = indice du texte du morceau k
        for i, text in enumerate(texts):
            parts = self.chunk_text(text) if text else [""]
            all_chunks.extend(parts)
            owner.extend([i] * len(parts))
        chunk_emb = self._encode_raw(all_chunks, batch_size, show_progress)
        owner_arr = np.asarray(owner)
        out = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
        np.add.at(out, owner_arr, chunk_emb)       # somme des morceaux de chaque texte
        return l2_normalize(out)                   # moyenne puis norme 1 (même direction)

    # ------------------------------------------------------------------ #
    # 3. Matrice des livres (avec cache disque)
    # ------------------------------------------------------------------ #
    def _cache_paths(self) -> tuple[Path, Path, Path]:
        """data/embeddings/{modèle}[_chunks]_books.npy + _book_ids.json + _meta.json"""
        suffix = "" if self.long_text == "truncate" else f"_{self.long_text}"
        stem = f"{model_slug(self.model_name)}{suffix}"
        return (self.cache_dir / f"{stem}_books.npy",
                self.cache_dir / f"{stem}_book_ids.json",
                self.cache_dir / f"{stem}_meta.json")

    def book_matrix(self, force: bool = False, show_progress: bool = True) -> np.ndarray:
        """Matrice (N, d) des embeddings des livres, ligne i = livre i de `self.books`.

        1. si elle est déjà en mémoire : on la renvoie ;
        2. sinon, si un cache disque existe ET que ses identifiants correspondent exactement
           au catalogue courant (même livres, même ordre) : on le charge ;
        3. sinon : on encode la colonne `embed_text` (préparée par P1) et on sauvegarde.
        `force=True` ré-encode dans tous les cas.
        """
        if self._book_matrix is not None and not force:
            return self._book_matrix

        npy_path, ids_path, meta_path = self._cache_paths()
        ids = self.book_ids

        # Étape 2 : cache valide ?
        if not force and npy_path.exists() and ids_path.exists():
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if cached_ids == ids:
                self._book_matrix = np.load(npy_path)
                return self._book_matrix

        # Étape 3 : encodage complet du catalogue
        texts = self.books["embed_text"].fillna("").astype(str).tolist()
        start = time.perf_counter()
        matrix = self.encode(texts, show_progress=show_progress)
        elapsed = time.perf_counter() - start

        # Sauvegarde : matrice + identifiants (alignement) + métadonnées (traçabilité)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, matrix)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        meta_path.write_text(json.dumps({
            "model": self.model_name,
            "long_text": self.long_text,
            "n_books": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "encode_seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")
        self._book_matrix = matrix
        return matrix

    # ------------------------------------------------------------------ #
    # 4. Matrices des labels (blocs Genre / Themes / Style / Atmosphere)
    # ------------------------------------------------------------------ #
    def label_matrix(self, block: str) -> tuple[list[str], np.ndarray]:
        """(labels, matrice (L, d)) des phrases descriptives d'un bloc de blocks.json.

        On RÉUTILISE enrichment.encode_blocks (P2) pour obtenir exactement les mêmes vecteurs
        que ceux qui ont servi à enrichir le catalogue. En particulier, pour Themes (v2),
        chaque thème = moyenne renormalisée de 4 phrases (blocks.json + theme_variants.json).
        Les 4 blocs sont encodés au premier appel puis gardés en mémoire (~66 labels : instantané).
        """
        if block not in config.BLOCKS:
            raise KeyError(f"Bloc inconnu {block!r}. Blocs disponibles : {config.BLOCKS}")
        if not self._labels:
            from src.enrichment import encode_blocks, load_blocks
            encoded = encode_blocks(self, load_blocks())
            self._labels = {b: (list(labels), np.asarray(m, dtype=np.float32))
                            for b, (labels, m) in encoded.items()}
        return self._labels[block]

    # ------------------------------------------------------------------ #
    # 5. Recherche des livres les plus proches
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.RETRIEVAL_TOP_K,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Les k livres les plus proches d'une requête -> (indices, scores), du meilleur au moins bon.

        Args:
            query_vec: vecteur (d,) de la requête, issu de encode() (donc normalisé).
            k: nombre de résultats (300 par défaut = étape 1 « retrieval » de P4).
            mask: tableau booléen (N,) optionnel ; False = livre exclu (filtres durs).

        Calcul : scores = M @ q (cosinus de chaque livre, car tout est normalisé), puis
        np.argpartition pour isoler les k meilleurs en O(N) sans trier les 12 000 scores.
        Pour 12 000 livres, une requête prend environ 1 ms : un index FAISS n'est pas nécessaire.
        """
        matrix = self.book_matrix()
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        scores = matrix @ q                                   # (N,) cosinus
        if mask is not None:
            scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
            k = min(k, int(np.asarray(mask, dtype=bool).sum()))
        k = max(0, min(k, scores.shape[0]))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        top = np.argpartition(-scores, k - 1)[:k]             # les k meilleurs, non triés
        top = top[np.argsort(-scores[top])]                   # tri des k seulement
        return top, scores[top].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 6. Explicabilité : phrase du résumé la plus proche de la requête
    # ------------------------------------------------------------------ #
    def best_sentence(self, query_vec: np.ndarray, text: str) -> tuple[str, float]:
        """Phrase d'un résumé la plus proche (cosinus) de la requête, et son score.

        Utilisée par P4/P5 pour montrer d'OÙ vient la recommandation :
        « Recommandé notamment pour ce passage : <phrase> (similarité 0.62) ».
        """
        sentences = split_sentences(text)
        if not sentences:
            return "", 0.0
        sent_emb = self.encode(sentences)
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        sims = sent_emb @ q
        best = int(np.argmax(sims))
        return sentences[best], float(sims[best])

    def __repr__(self) -> str:  # affichage lisible dans les notebooks
        loaded = "chargé" if self._model is not None else "non chargé"
        n = None if self._book_matrix is None else self._book_matrix.shape[0]
        return (f"EmbeddingEngine(model={self.model_name!r}, long_text={self.long_text!r}, "
                f"modèle {loaded}, livres encodés={n})")
