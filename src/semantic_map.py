"""
semantic_map.py: analyses of the embedding space (Person 3).

1. Cosine distribution: why raw SBERT scores are "compressed" and why the calibration
   must be done on the CANDIDATES (P4's choice) and not on the whole catalogue.
2. 2D map (UMAP): projection of the 12,000 vectors (384 dimensions) onto a plane, for the
   app's "similarity map" (P5). Two books close on the map have blurbs with a close
   meaning.
3. Projection of a query onto the map (to place the "you" point in the app).
4. Silhouette score by genre: do genres form distinct regions of the space?
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
# 1. Cosine distribution
# --------------------------------------------------------------------------- #

def cosine_distribution(book_matrix: np.ndarray, query_vecs: np.ndarray,
                        top_k: int = config.RETRIEVAL_TOP_K) -> dict:
    """Compare the cosine distribution over the WHOLE catalogue and over the top_k candidates.

    For each query: scores = M @ q (cosine with all books), then the top_k best are isolated
    (= the candidates kept by P4's retrieval).

    Returns the percentiles of both distributions, the share of candidates above the
    catalogue's p95, and the raw scores (to plot the histograms).
    """
    q = l2_normalize(np.atleast_2d(query_vecs))
    scores = book_matrix @ q.T                                # (N, n_queries)
    k = min(top_k, scores.shape[0])
    top = -np.sort(-scores, axis=0)[:k]                       # (k, n_queries) best scores
    all_s, top_s = scores.ravel(), top.ravel()
    p = lambda a, x: round(float(np.percentile(a, x)), 4)     # noqa: E731
    # p95 threshold computed query by query (each query has its own distribution)
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
# 2. 2D UMAP map
# --------------------------------------------------------------------------- #

def compute_umap(book_matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                 seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Project the (N, d) vectors into 2D with UMAP -> (N, 2) float32.

    UMAP builds the graph of the n_neighbors nearest neighbours of each book (cosine
    distance, consistent with the rest of the engine), then looks for a 2D layout that preserves
    this neighbourhood. min_dist sets how spread out the points are within groups.
    random_state fixes the result (same map on every run, at the cost of parallelism).
    """
    import inspect

    import umap  # heavy import (numba): only here
    params = dict(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                  n_components=2, random_state=seed)
    # umap-learn >= 0.6 announces a change of the default algorithm in a future version.
    # We FREEZE the current algorithm so that the map stays identical whatever the version
    # installed by a team member (reproducibility). Older versions do not have this
    # parameter and already use this algorithm.
    if "compatibility_layout" in inspect.signature(umap.UMAP.__init__).parameters:
        params["compatibility_layout"] = True
    reducer = umap.UMAP(**params)
    return reducer.fit_transform(book_matrix).astype(np.float32)


def save_umap(coords: np.ndarray, book_ids: Sequence[str],
              path: Path = UMAP_PATH, ids_path: Path = UMAP_IDS_PATH) -> None:
    """Save the coordinates + the identifiers (same alignment principle as book_matrix)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)
    Path(ids_path).write_text(json.dumps([str(b) for b in book_ids]), encoding="utf-8")


def load_umap(book_ids: Sequence[str] | None = None,
              path: Path = UMAP_PATH, ids_path: Path = UMAP_IDS_PATH) -> np.ndarray | None:
    """2D coordinates, or None if missing or NOT ALIGNED with `book_ids` (catalogue changed)."""
    if not Path(path).exists() or not Path(ids_path).exists():
        return None
    if book_ids is not None:
        saved = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if saved != [str(b) for b in book_ids]:
            return None
    return np.load(path)


def project_query(query_vec: np.ndarray, book_matrix: np.ndarray, coords: np.ndarray,
                  k: int = 10) -> np.ndarray:
    """Place a query on the 2D map: average position of its k closest books,
    weighted by their similarity.

    Why not umap.transform? The whole UMAP model (heavy) would have to be saved and
    reloaded in the app. The average of the neighbours is simple, fast and readable: the
    "you" point appears among the books that resemble you the most.
    """
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
    sims = book_matrix @ q
    k = min(k, len(sims))
    nn = np.argpartition(-sims, k - 1)[:k]
    w = np.clip(sims[nn], 1e-6, None)                       # positive weights
    return (coords[nn] * w[:, None]).sum(0) / w.sum()


# --------------------------------------------------------------------------- #
# 3. Are the genres separated in the space?
# --------------------------------------------------------------------------- #

def genre_silhouette(book_matrix: np.ndarray, labels: Sequence[str], sample: int = 3000,
                     min_count: int = 30, seed: int = config.RANDOM_SEED) -> float:
    """Silhouette score (cosine distance) of the books grouped by main genre.

    Between -1 and 1: close to 1 = well-separated genres; close to 0 = overlapping
    genres; negative = books closer to another genre than to their own.
    Genres that are too rare (< min_count books) are ignored; sampled for speed.
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


def genre_neighbor_agreement(book_matrix: np.ndarray, genres: Sequence[Sequence[str]], k: int = 10,
                             sample: int = 2000, seed: int = config.RANDOM_SEED) -> dict:
    """Share of a book's k nearest neighbours that share AT LEAST ONE of its genres,
    compared with k randomly drawn books.

    Better suited than the silhouette to MULTIPLE, OVERLAPPING genres (a book can be
    both "Thriller" and "Historical Fiction"): we do not require clear-cut boundaries,
    only that semantic neighbours share a genre more often than randomly picked
    books.

    Returns {"knn": share among neighbours, "random": random share, "lift": knn / random,
             "per_genre": {genre: share among the neighbours of the books of this genre}}.
    """
    rng = np.random.default_rng(seed)
    sets = [set(g) if g is not None else set() for g in genres]
    has_genre = np.flatnonzero([len(s) > 0 for s in sets])
    idx = rng.choice(has_genre, min(sample, len(has_genre)), replace=False)

    sims = book_matrix[idx] @ book_matrix.T                   # (sample, N)
    sims[np.arange(len(idx)), idx] = -np.inf                  # a book is not its own neighbour
    nn = np.argpartition(-sims, k, axis=1)[:, :k]             # k nearest neighbours

    knn_share, rand_share = [], []
    per_genre: dict[str, list[float]] = {}
    for row, i in enumerate(idx):
        mine = sets[i]
        share = np.mean([len(mine & sets[j]) > 0 for j in nn[row]])
        rand = rng.choice(len(sets), k, replace=False)
        knn_share.append(share)
        rand_share.append(np.mean([len(mine & sets[j]) > 0 for j in rand]))
        for g in mine:
            per_genre.setdefault(g, []).append(share)
    knn, rnd = float(np.mean(knn_share)), float(np.mean(rand_share))
    return {"k": k, "n_books": int(len(idx)), "knn": round(knn, 3), "random": round(rnd, 3),
            "lift": round(knn / max(rnd, 1e-9), 2),
            "per_genre": {g: round(float(np.mean(v)), 3)
                          for g, v in sorted(per_genre.items(), key=lambda kv: -np.mean(kv[1]))
                          if len(v) >= 30}}
