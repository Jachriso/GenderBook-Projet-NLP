"""
prepare_app.py : pré-calcule tout ce dont l'application a besoin AVANT la démo (Personne 5).

    python -m scripts.prepare_app            # depuis la racine du repo

1. télécharge le modèle SBERT (si besoin) et encode les 12 000 livres -> data/embeddings/*_books.npy
2. calcule la projection 2D du catalogue -> data/embeddings/umap_2d.npy (+ book_ids pour l'alignement)
3. lance une recommandation complète sur les personas de démo (vérifie que tout est branché)

Ainsi, le jour de la soutenance, l'app démarre en quelques secondes au lieu de plusieurs minutes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import questions as Q  # noqa: E402
from src import viz  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    try:                                       # moteur final (P4) : il charge lui-même P3
        from src.recommender import get_recommender, recommend  # type: ignore
        rec = get_recommender()
        engine, books = rec.engine, rec.books
        mode, run = "P4", (lambda a: recommend(a, top_k=3))
    except ImportError:                        # P4 pas encore intégré : EmbeddingEngine + stub
        from app.stub_recommender import recommend as stub
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        books = engine.books
        mode, run = "stub", (lambda a: stub(a, top_k=3, engine=engine, books=books))
    matrix = engine.book_matrix()                                     # étape 1
    print(f"[1/3] embeddings des livres : {matrix.shape} ({time.perf_counter() - t0:.0f} s)")

    t1 = time.perf_counter()                                          # étape 2
    coords = viz.load_or_compute_map_coords(matrix, books["book_id"].astype(str).tolist())
    print(f"[2/3] carte 2D : {coords.shape} ({time.perf_counter() - t1:.0f} s)")

    blocks = Q.load_blocks()                                          # étape 3
    for name in Q.DEMO_PERSONAS:
        answers = Q.build_answers(Q.persona_state(name, blocks), blocks)
        _, recs = run(answers)
        print(f"[3/3] ({mode}) {name}: " + " | ".join(f"{r.title} ({r.final_score:.2f})" for r in recs))
    print(f"Prêt en {time.perf_counter() - t0:.0f} s. Lancer : streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
