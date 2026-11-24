"""
Quick check of the semantic engine (P3) with the REAL SBERT model, on 500 books.

Run from the repo root:
    python -X utf8 -m scripts.p3_quick_check

What the script shows:
    1. loading the model and encoding the 500 books (and how long it takes);
    2. how many blurbs exceed the model's token limit (truncation issue);
    3. natural-language queries -> the 3 closest books + the sentence that explains;
    4. the label matrices of the 4 blocks (reuse of P2's taxonomy).
"""
import time

import numpy as np
import pandas as pd

from src import config
from src.embeddings import EmbeddingEngine

# 1. Development catalogue: the 500 books enriched by P2 (otherwise P1's sample)
path = config.DATA_PROCESSED / "books_enriched_sample.parquet"
if not path.exists():
    path = config.BOOKS_SAMPLE
books = pd.read_parquet(path)
print(f"Catalogue : {path.name} -> {len(books)} livres")

# Separate cache for the sample: not mixed with the full catalogue's cache
engine = EmbeddingEngine(books=books, cache_dir=config.DATA_EMBEDDINGS / "sample")

# 2. Encoding the books (the model is downloaded on the first run: ~90 MB)
t0 = time.perf_counter()
matrix = engine.book_matrix()
print(f"\nMatrice des livres : {matrix.shape} en {time.perf_counter() - t0:.1f} s")
print(f"Modèle : {engine.model_name} | limite : {engine.max_tokens} tokens | dimension : {matrix.shape[1]}")

# 3. Truncation: how many blurbs are cut by the model?
n_tok = engine.count_tokens(books["embed_text"].tolist())
over = (n_tok > engine.max_tokens - 2).mean() * 100
print(f"Tokens par résumé : médiane {np.median(n_tok):.0f}, max {n_tok.max()} "
      f"-> {over:.1f} % des résumés dépassent la limite et sont tronqués")

# 4. Demo queries (none mentions a title or an author)
queries = [
    "a boy discovers he is a wizard and goes to a school of magic",
    "a detective investigates a murder in a small English village",
    "soldiers struggling to survive on the battlefield during the war",
    "a funny and heartwarming story about friendship",
]
for q in queries:
    q_vec = engine.encode([q])[0]
    idx, scores = engine.search(q_vec, k=3)
    print(f"\nREQUÊTE : {q}")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        row = books.iloc[i]
        sentence, s_sent = engine.best_sentence(q_vec, row["description"])
        print(f"  {rank}. {row['title'][:60]} ({row['authors_str'][:30]}) | cos = {s:.3f}")
        print(f"     phrase la plus proche ({s_sent:.2f}) : {sentence[:140]}")

# 5. Labels of the 4 blocks (vectors identical to P2's)
print()
for block in config.BLOCKS:
    labels, m = engine.label_matrix(block)
    print(f"Bloc {block:<10} : {len(labels):>2} labels -> matrice {m.shape}")

print("\nOK : le moteur fonctionne avec le vrai modèle.")
