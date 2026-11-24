"""
semantic_map.py : analyses de l'espace des embeddings (Personne 3).

1. Distribution des cosinus : pourquoi les scores SBERT bruts sont « compressés » et pourquoi
   la calibration doit se faire sur les CANDIDATS (choix de P4) et non sur tout le catalogue.
2. Carte 2D (UMAP) : projection des 12 000 vecteurs (384 dimensions) sur un plan, pour la
   « carte de similarité » de l'app (P5). Deux livres proches sur la carte ont des résumés
   de sens proche.
3. Projection d'une requête sur la carte (pour placer le point « vous » dans l'app).
4. Score silhouette par genre : les genres forment-ils des régions distinctes de l'espace ?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src import config
from src.embeddings import l2_normalize

UMAP_PATH = config.DATA_EMBEDDINGS / "umap_2d.npy"
UMAP_IDS_PATH = config.DATA_EMBEDDINGS / "umap_book_ids.json"


# --------------------------------------------------------------------------- #
# 1. Distribution des cosinus
# --------------------------------------------------------------------------- #

def cosine_distribution(book_matrix: np.ndarray, query_vecs: np.ndarray,
                        top_k: int = config.RETRIEVAL_TOP_K) -> dict:
    """Compare la distribution des cosinus sur TOUT le catalogue et sur les top_k candidats.

    Pour chaque requête : scores = M @ q (cosinus avec tous les livres), puis on isole les
    top_k meilleurs (= les candidats que garde le retrieval de P4).

    Retourne les percentiles des deux distributions, la part des candidats situés au-dessus
    du p95 du catalogue, et les scores bruts (pour tracer les histogrammes).
    """
    q = l2_normalize(np.atleast_2d(query_vecs))
    scores = book_matrix @ q.T                                # (N, n_requêtes)
    k = min(top_k, scores.shape[0])
    top = -np.sort(-scores, axis=0)[:k]                       # (k, n_requêtes) meilleurs scores
    all_s, top_s = scores.ravel(), top.ravel()
    p = lambda a, x: round(float(np.percentile(a, x)), 4)     # noqa: E731
    # Seuil p95 calculé requête par requête (chaque requête a sa propre distribution)
    p95_per_query = np.percentile(scores, 95, axis=0)
    above = float((top > p95_per_query[None, :]).mean())
    return {
        "n_books": int(scores.shape[0]), "n_queries": int(scores.shape[1]), "top_k": int(k),
        "catalogue": {"p5": p(all_s, 5), "p50": p(all_s, 50), "p95": p(all_s, 95),
                      "max": round(float(all_s.max()), 4)},
        "candidates": {"p5": p(top_s, 5), "p50": p(top_s, 50), "p95": p(top_s, 95),
                       "min": round(float(top_s.min()), 4)},
        "share_candidates_above_catalogue_p95": round(above, 4),
        "_all_scores": all_s, "_top_scores": top_s,
    }


# --------------------------------------------------------------------------- #
# 2. Carte 2D UMAP
# --------------------------------------------------------------------------- #

def compute_umap(book_matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                 seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Projette les vecteurs (N, d) en 2D avec UMAP -> (N, 2) float32.

    UMAP construit le graphe des n_neighbors plus proches voisins de chaque livre (distance
    cosinus, cohérente avec le reste du moteur) puis cherche une disposition 2D qui préserve
    ce voisinage. min_dist règle l'écartement des points dans les groupes.
    random_state fixe le résultat (même carte à chaque exécution, au prix du parallélisme).
    """
    import umap  # import lourd (numba) : seulement ici
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                        n_components=2, random_state=seed)
    return reducer.fit_transform(book_matrix).astype(np.float32)


def save_umap(coords: np.ndarray, book_ids: Sequence[str],
              path: Path = UMAP_PATH, ids_path: Path = UMAP_IDS_PATH) -> None:
    """Sauvegarde les coordonnées + les identifiants (même principe d'alignement que book_matrix)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)
    Path(ids_path).write_text(json.dumps([str(b) for b in book_ids]), encoding="utf-8")


def load_umap(book_ids: Sequence[str] | None = None,
              path: Path = UMAP_PATH, ids_path: Path = UMAP_IDS_PATH) -> np.ndarray | None:
    """Coordonnées 2D, ou None si absentes ou NON ALIGNÉES sur `book_ids` (catalogue changé)."""
    if not Path(path).exists() or not Path(ids_path).exists():
        return None
    if book_ids is not None:
        saved = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if saved != [str(b) for b in book_ids]:
            return None
    return np.load(path)


def project_query(query_vec: np.ndarray, book_matrix: np.ndarray, coords: np.ndarray,
                  k: int = 10) -> np.ndarray:
    """Place une requête sur la carte 2D : moyenne des positions de ses k plus proches livres,
    pondérée par leur similarité.

    Pourquoi pas umap.transform ? Il faudrait sauvegarder le modèle UMAP entier (lourd) et le
    recharger dans l'app. La moyenne des voisins est simple, rapide et lisible : le point
    « vous » apparaît au milieu des livres qui vous ressemblent le plus.
    """
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
    sims = book_matrix @ q
    k = min(k, len(sims))
    nn = np.argpartition(-sims, k - 1)[:k]
    w = np.clip(sims[nn], 1e-6, None)                       # poids positifs
    return (coords[nn] * w[:, None]).sum(0) / w.sum()


# --------------------------------------------------------------------------- #
# 3. Les genres sont-ils séparés dans l'espace ?
# --------------------------------------------------------------------------- #

def genre_silhouette(book_matrix: np.ndarray, labels: Sequence[str], sample: int = 3000,
                     min_count: int = 30, seed: int = config.RANDOM_SEED) -> float:
    """Score silhouette (distance cosinus) des livres groupés par genre principal.

    Entre -1 et 1 : proche de 1 = genres bien séparés ; proche de 0 = genres qui se
    chevauchent ; négatif = livres plus proches d'un autre genre que du leur.
    Les genres trop rares (< min_count livres) sont ignorés ; échantillon pour la rapidité.
    """
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    keep = np.isin(labels, values[counts >= min_count])
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)
    if len(np.unique(labels[idx])) < 2:
        return float("nan")
    return float(silhouette_score(book_matrix[idx], labels[idx], metric="cosine"))
