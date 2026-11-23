"""
Vérification rapide du moteur sémantique (P3) avec le VRAI modèle SBERT, sur 500 livres.

Lancer depuis la racine du repo :
    python -X utf8 -m scripts.p3_quick_check

Ce que le script montre :
    1. le chargement du modèle et l'encodage des 500 livres (et le temps que ça prend) ;
    2. combien de résumés dépassent la limite de tokens du modèle (problème de troncature) ;
    3. des requêtes en langage naturel -> les 3 livres les plus proches + la phrase qui explique ;
    4. les matrices de labels des 4 blocs (réutilisation du référentiel de P2).
"""
import time

import numpy as np
import pandas as pd

from src import config
from src.embeddings import EmbeddingEngine

# 1. Catalogue de développement : les 500 livres enrichis par P2 (sinon l'échantillon de P1)
path = config.DATA_PROCESSED / "books_enriched_sample.parquet"
if not path.exists():
    path = config.BOOKS_SAMPLE
books = pd.read_parquet(path)
print(f"Catalogue : {path.name} -> {len(books)} livres")

# Cache séparé pour l'échantillon : on ne mélange pas avec le cache du catalogue complet
engine = EmbeddingEngine(books=books, cache_dir=config.DATA_EMBEDDINGS / "sample")

# 2. Encodage des livres (le modèle est téléchargé au premier lancement : ~90 Mo)
t0 = time.perf_counter()
matrix = engine.book_matrix()
print(f"\nMatrice des livres : {matrix.shape} en {time.perf_counter() - t0:.1f} s")
print(f"Modèle : {engine.model_name} | limite : {engine.max_tokens} tokens | dimension : {matrix.shape[1]}")

# 3. Troncature : combien de résumés sont coupés par le modèle ?
n_tok = engine.count_tokens(books["embed_text"].tolist())
over = (n_tok > engine.max_tokens - 2).mean() * 100
print(f"Tokens par résumé : médiane {np.median(n_tok):.0f}, max {n_tok.max()} "
      f"-> {over:.1f} % des résumés dépassent la limite et sont tronqués")

# 4. Requêtes de démonstration (aucune ne cite un titre ni un auteur)
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

# 5. Labels des 4 blocs (vecteurs identiques à ceux de P2)
print()
for block in config.BLOCKS:
    labels, m = engine.label_matrix(block)
    print(f"Bloc {block:<10} : {len(labels):>2} labels -> matrice {m.shape}")

print("\nOK : le moteur fonctionne avec le vrai modèle.")
