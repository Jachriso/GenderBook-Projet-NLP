"""
Analyses de l'espace sémantique sur le catalogue complet (MiniLM, cache des 12 000 livres).

    A. Distribution des cosinus : catalogue entier vs 300 candidats (justifie la calibration de P4).
    B. Carte UMAP 2D, colorée par genre (pour la « carte de similarité » de P5).
    C. Score silhouette par genre (les genres forment-ils des régions séparées ?).
    D. Voisins qui partagent un genre (mesure adaptée à des genres multiples qui se chevauchent).

Lancer depuis la racine du repo (environ 1 à 2 minutes ; le cache MiniLM doit exister,
sinon il est recalculé, ~7 min) :
    python -X utf8 -m scripts.p3_semantic_map

Sorties :
    data/embeddings/umap_2d.npy + umap_book_ids.json   (réutilisés par l'app de P5)
    docs/figures/p3_cosine_distribution.png + .json
    docs/figures/p3_umap_genres.png
    docs/figures/p3_genre_neighbors.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from src import config
from src.embeddings import EmbeddingEngine
from src.semantic_map import (compute_umap, cosine_distribution, genre_neighbor_agreement,
                              genre_silhouette, load_umap, project_query, save_umap)
from scripts.p3_build_embeddings import FAMOUS_QUERIES
from scripts.p3_compare_encoders import QUERIES

N_GENRE_COLORS = 10   # les 10 genres les plus fréquents en couleur, le reste en gris


def main(engine: EmbeddingEngine | None = None, force_umap: bool = False) -> dict:
    import matplotlib
    matplotlib.use("Agg")                          # pas de fenêtre : on enregistre des PNG
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    engine = engine or EmbeddingEngine()
    books = engine.books
    M = engine.book_matrix()
    print(f"Catalogue : {M.shape[0]} livres, vecteurs de dimension {M.shape[1]} ({engine.model_name})")

    # ------------------------------------------------------------------ #
    # A. Distribution des cosinus
    # ------------------------------------------------------------------ #
    queries = [q for q, _ in FAMOUS_QUERIES] + [q for q, _ in QUERIES]
    dist = cosine_distribution(M, engine.encode(queries))
    cat, cand = dist["catalogue"], dist["candidates"]
    print(f"\nA. DISTRIBUTION DES COSINUS ({dist['n_queries']} requêtes x {dist['n_books']} livres)")
    print(f"   Catalogue entier : p5 = {cat['p5']:.3f} | médiane = {cat['p50']:.3f} | "
          f"p95 = {cat['p95']:.3f} | max = {cat['max']:.3f}")
    print(f"   {dist['top_k']} candidats    : min = {cand['min']:.3f} | p5 = {cand['p5']:.3f} | "
          f"médiane = {cand['p50']:.3f} | p95 = {cand['p95']:.3f}")
    print(f"   Part des candidats au-dessus du p95 du catalogue : "
          f"{100 * dist['share_candidates_above_catalogue_p95']:.1f} %")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-0.2, 0.8, 101)
    ax.hist(dist["_all_scores"], bins=bins, density=True, alpha=0.55, label="whole catalogue")
    ax.hist(dist["_top_scores"], bins=bins, density=True, alpha=0.55,
            label=f"{dist['top_k']} candidates (retrieval)")
    for name, v, style in [("catalogue p5", cat["p5"], ":"), ("catalogue p95", cat["p95"], "--")]:
        ax.axvline(v, color="black", linestyle=style, linewidth=1)
        ax.text(v, ax.get_ylim()[1] * 0.92, f" {name}\n {v:.2f}", fontsize=8)
    ax.set_xlabel(f"cosine similarity query ↔ blurb ({engine.model_name.split('/')[-1]})")
    ax.set_ylabel("density")
    ax.set_title(f"Cosine distribution: whole catalogue vs {dist['top_k']} candidates "
                 f"({dist['n_queries']} queries)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "p3_cosine_distribution.png", dpi=150)
    plt.close(fig)
    stats = {k: v for k, v in dist.items() if not k.startswith("_")}
    stats["model"] = engine.model_name
    (config.FIGURES_DIR / "p3_cosine_distribution.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # B. Carte UMAP 2D
    # ------------------------------------------------------------------ #
    ids = engine.book_ids
    coords = None if force_umap else load_umap(ids)
    if coords is None:
        print("\nB. CARTE UMAP 2D : calcul en cours (≈ 1 min, plus la compilation numba au 1er lancement)...")
        t0 = time.perf_counter()
        coords = compute_umap(M)
        save_umap(coords, ids)
        print(f"   calculée en {time.perf_counter() - t0:.0f} s et sauvegardée -> data/embeddings/umap_2d.npy")
    else:
        print("\nB. CARTE UMAP 2D : rechargée depuis data/embeddings/umap_2d.npy")

    # Genre principal de chaque livre (1er genre attribué par P2)
    genres = books["genres"] if "genres" in books else None
    main_genre = np.array([(list(g)[0] if g is not None and len(g) else "Unknown")
                           for g in (genres if genres is not None else [None] * len(books))])
    values, counts = np.unique(main_genre, return_counts=True)
    top_genres = list(values[np.argsort(-counts)][:N_GENRE_COLORS])

    fig, ax = plt.subplots(figsize=(13, 8))
    other = ~np.isin(main_genre, top_genres)
    ax.scatter(coords[other, 0], coords[other, 1], s=2, c="lightgrey", label="other genres")
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(top_genres):
        m = main_genre == g
        ax.scatter(coords[m, 0], coords[m, 1], s=2, color=cmap(i), label=f"{g} ({m.sum()})")
    # Démonstration : où tombe une requête utilisateur sur la carte ?
    demo_q, _ = FAMOUS_QUERIES[0]
    xy = project_query(engine.encode([demo_q])[0], M, coords)
    ax.scatter([xy[0]], [xy[1]], marker="*", s=400, c="black", edgecolors="white", zorder=5,
               label='query: "a boy discovers he is a wizard…"')
    ax.set_title(f"UMAP semantic map of the {len(books)} books (colour = main genre)")
    ax.set_xticks([]); ax.set_yticks([])
    # Légende à droite du graphique, avec des marqueurs de taille lisible (y compris l'étoile)
    leg = ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_sizes([40])
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "p3_umap_genres.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # C. Silhouette par genre
    # ------------------------------------------------------------------ #
    sil = genre_silhouette(M, main_genre)
    print(f"\nC. SILHOUETTE PAR GENRE (cosinus, 3 000 livres) : {sil:.3f}")
    print("   (1 = genres parfaitement séparés, 0 = genres qui se chevauchent)")


    # Mesure adaptée à des genres multiples qui se chevauchent : les voisins partagent-ils un genre ?
    all_genres = [list(g) if g is not None else [] for g in
                  (genres if genres is not None else [None] * len(books))]
    agree = genre_neighbor_agreement(M, all_genres)
    print(f"\nD. VOISINS QUI PARTAGENT UN GENRE ({agree['n_books']} livres, k = {agree['k']}, tous les genres de P2)")
    print(f"   10 plus proches voisins : {100 * agree['knn']:.1f} % partagent au moins un genre")
    print(f"   10 livres au hasard     : {100 * agree['random']:.1f} %  -> {agree['lift']:.1f} fois plus que le hasard")
    print("   Par genre (part des voisins qui partagent un genre) :")
    for g, v in list(agree["per_genre"].items()):
        print(f"     {g:<28} {100 * v:5.1f} %")
    (config.FIGURES_DIR / "p3_genre_neighbors.json").write_text(
        json.dumps({**agree, "silhouette": round(sil, 4), "model": engine.model_name}, indent=2),
        encoding="utf-8")

    print("\nFigures : docs/figures/p3_cosine_distribution.png, docs/figures/p3_umap_genres.png")
    return {"distribution": stats, "silhouette": sil, "top_genres": top_genres, "neighbors": agree}


if __name__ == "__main__":
    main()
