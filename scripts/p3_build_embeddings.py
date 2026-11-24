"""
Encode the FULL CATALOGUE (12,000 books) and check semantic search at scale.

The file produced (data/embeddings/<model>_books.npy + _book_ids.json + _meta.json) is the
cache automatically reused by EmbeddingEngine(), and therefore by P4's recommender and P5's app.

Run from the repo root:
    python -X utf8 -m scripts.p3_build_embeddings                       # MiniLM, truncation (app default)
    python -X utf8 -m scripts.p3_build_embeddings --long-text chunks    # MiniLM, chunking
    python -X utf8 -m scripts.p3_build_embeddings --model mpnet         # MPNet (≈ 5x slower on CPU)
    python -X utf8 -m scripts.p3_build_embeddings --force               # re-encode even if the cache exists

What the script shows:
    1. the encoding time (or the instant reload if the cache already exists);
    2. how many blurbs exceed the model's token limit;
    3. paraphrased "famous" queries: at which rank out of 12,000 does the expected book come out?
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd

from src.embeddings import EmbeddingEngine

# (paraphrased query, regular expression recognising the expected book in the TITLE)
# The queries use neither the title, nor the characters' names, nor the author's name.
FAMOUS_QUERIES = [
    ("a boy discovers he is a wizard and goes to a school of magic",
     r"harry potter"),
    ("a teenage girl volunteers to take her sister's place in a televised fight to the death",
     r"hunger games"),
    ("a small hairy-footed creature leaves his comfortable home to help dwarves recover their "
     "treasure from a dragon", r"\bhobbit\b"),
    ("an obsessed sea captain hunts the white whale that took his leg",
     r"moby[- ]dick"),
    ("a witty young woman in Regency England clashes with a proud wealthy gentleman before "
     "falling in love", r"pride and prejudice"),
    ("a totalitarian state spies on its citizens and constantly rewrites history",
     r"nineteen eighty|\b1984\b"),
    ("a scientist builds a living creature from dead body parts and is horrified by it",
     r"frankenstein"),
]


def rank_of_expected(scores: np.ndarray, titles: pd.Series, pattern: str) -> tuple[int | None, str]:
    """Rank (1 = first) of the BEST book whose title matches `pattern`, and its title."""
    hits = np.flatnonzero(titles.str.contains(pattern, flags=re.IGNORECASE, regex=True).to_numpy())
    if len(hits) == 0:
        return None, ""
    best = hits[np.argmax(scores[hits])]                  # the best edition / the best volume
    return int((scores > scores[best]).sum()) + 1, str(titles.iloc[best])


def main(engine: EmbeddingEngine | None = None, top_n: int = 5) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="minilm", help="minilm (défaut), mpnet ou nom Hugging Face")
    parser.add_argument("--long-text", default="truncate", choices=["truncate", "chunks"])
    parser.add_argument("--force", action="store_true", help="ré-encoder même si le cache existe")
    args = parser.parse_args([] if engine is not None else None)

    # 1. Engine on the full catalogue (P1's load_books(): books_enriched if available)
    engine = engine or EmbeddingEngine(args.model, long_text=args.long_text)
    books = engine.books
    print(f"Catalogue : {len(books)} livres | modèle : {engine.model_name} | textes longs : {engine.long_text}")

    # 2. Encoding (or reloading the disk cache)
    t0 = time.perf_counter()
    matrix = engine.book_matrix(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"Matrice : {matrix.shape} en {elapsed:.1f} s "
          f"({'rechargée depuis le cache' if elapsed < 5 else f'{1000 * elapsed / len(books):.0f} ms / livre'})")

    # 3. Truncation: share of blurbs longer than the model's limit
    n_tok = engine.count_tokens(books["embed_text"].fillna("").tolist())
    limit = engine.max_tokens - 2                         # 2 slots for [CLS] and [SEP]
    print(f"Tokens par résumé : médiane {np.median(n_tok):.0f}, p95 {np.percentile(n_tok, 95):.0f}, "
          f"max {n_tok.max()} | {100 * (n_tok > limit).mean():.1f} % dépassent {limit} tokens")

    # 4. Famous queries: rank of the expected book in the WHOLE catalogue
    titles = books["title"].fillna("").astype(str)
    rows = []
    for query, pattern in FAMOUS_QUERIES:
        q = engine.encode([query])[0]
        scores = matrix @ q                               # cosine with the 12,000 books
        rank, found_title = rank_of_expected(scores, titles, pattern)
        top = np.argsort(-scores)[:top_n]
        print(f"\nREQUÊTE : {query}")
        if rank is None:
            print(f"  (livre attendu « {pattern} » absent du catalogue)")
        else:
            print(f"  -> livre attendu : {found_title[:60]} | RANG {rank} / {len(books)}")
        for r, i in enumerate(top, start=1):
            sent, s_sent = engine.best_sentence(q, books.iloc[i]["description"])
            print(f"  {r}. {titles.iloc[i][:55]:<55} cos = {scores[i]:.3f}")
            print(f"     « {sent[:110]} » ({s_sent:.2f})")
        rows.append({"query": query, "expected": (found_title or pattern)[:40], "rank": rank,
                     "top1": titles.iloc[top[0]][:45], "top1_cos": round(float(scores[top[0]]), 3)})

    # 5. Summary
    res = pd.DataFrame(rows)
    found = res["rank"].dropna()
    print("\nRÉCAPITULATIF")
    print(res[["expected", "rank", "top1"]].to_string(index=False))
    if len(found):
        print(f"\nTrouvé en 1er : {(found == 1).sum()}/{len(found)} | top 10 : {(found <= 10).sum()}/{len(found)} "
              f"| MRR : {(1 / found).mean():.2f}")
    return res


if __name__ == "__main__":
    main()
