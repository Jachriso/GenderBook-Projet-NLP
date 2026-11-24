"""
prepare_app.py: pre-computes everything the app needs BEFORE the demo (Person 5).

    python -m scripts.prepare_app            # from the root of the repo

1. downloads the SBERT model (if needed) and encodes the 12,000 books -> data/embeddings/*_books.npy
2. computes the 2D projection of the catalogue -> data/embeddings/umap_2d.npy (+ book_ids for alignment)
3. runs a full recommendation for the demo personas (checks that everything is wired)

This way, on the day of the presentation, the app starts in a few seconds instead of several minutes.
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
    try:                                       # final engine (P4): it loads P3 itself
        from src.recommender import get_recommender, recommend  # type: ignore
        rec = get_recommender()
        engine, books = rec.engine, rec.books
        mode, run = "P4", (lambda a: recommend(a, top_k=3))
    except ImportError:                        # P4 not integrated yet: EmbeddingEngine + stub
        from app.stub_recommender import recommend as stub
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        books = engine.books
        mode, run = "stub", (lambda a: stub(a, top_k=3, engine=engine, books=books))
    matrix = engine.book_matrix()                                     # step 1
    print(f"[1/3] book embeddings: {matrix.shape} ({time.perf_counter() - t0:.0f} s)")

    t1 = time.perf_counter()                                          # step 2
    coords = viz.load_or_compute_map_coords(matrix, books["book_id"].astype(str).tolist())
    print(f"[2/3] 2D map: {coords.shape} ({time.perf_counter() - t1:.0f} s)")

    blocks = Q.load_blocks()                                          # step 3
    for name in Q.DEMO_PERSONAS:
        answers = Q.build_answers(Q.persona_state(name, blocks), blocks)
        _, recs = run(answers)
        print(f"[3/3] ({mode}) {name}: " + " | ".join(f"{r.title} ({r.final_score:.2f})" for r in recs))
    print(f"Ready in {time.perf_counter() - t0:.0f} s. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
