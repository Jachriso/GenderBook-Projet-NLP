"""
prepare_app.py: pre-computes everything the app needs BEFORE the demo (Person 5).

    python -m scripts.prepare_app            # from the root of the repo

1. downloads the SBERT model (if needed) and encodes the 12,000 books -> data/embeddings/*_books.npy
2. loads P3's 2D map (data/embeddings/umap_2d.npy + umap_book_ids.json), or recomputes it if
   it is missing or no longer matches the catalogue (src/semantic_map.py)
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
from src.recommender import get_recommender, recommend  # noqa: E402
from src.semantic_map import compute_umap, load_umap, save_umap  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    rec = get_recommender()                                           # P4's engine (it loads P3)
    matrix = rec.engine.book_matrix()                                 # step 1
    print(f"[1/3] book embeddings: {matrix.shape} ({time.perf_counter() - t0:.0f} s)")

    t1 = time.perf_counter()                                          # step 2: P3's map
    book_ids = rec.books["book_id"].astype(str).tolist()
    coords = load_umap(book_ids)                                      # None if missing or not aligned
    if coords is None:
        coords = compute_umap(matrix)
        save_umap(coords, book_ids)
    print(f"[2/3] 2D map: {coords.shape} ({time.perf_counter() - t1:.0f} s)")

    blocks = Q.load_blocks()                                          # step 3
    for name in Q.DEMO_PERSONAS:
        answers = Q.build_answers(Q.persona_state(name, blocks), blocks)
        _, recs = recommend(answers, top_k=3)
        print(f"[3/3] {name}: " + " | ".join(f"{r.title} ({r.final_score:.2f})" for r in recs))
    print(f"Ready in {time.perf_counter() - t0:.0f} s. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
