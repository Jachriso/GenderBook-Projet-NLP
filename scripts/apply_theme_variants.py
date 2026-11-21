"""v2 des thèmes : phrases enrichies (moyenne de 4 phrases par thème), z_min = 1.0.
Met à jour themes et label_scores['Themes'] dans books_enriched.parquet, sans ré-encoder les livres."""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en
from sentence_transformers import SentenceTransformer

Z_MIN_THEMES = 1.0
parquet = ROOT / "data/processed/books_enriched.parquet"
backup = ROOT / "data/embeddings/books_enriched_v1_backup.parquet"
if not backup.exists():
    shutil.copy(parquet, backup)
    print("sauvegarde ->", backup)

df = pd.read_parquet(parquet)
df["book_id"] = df["book_id"].astype(str)

blocks = en.load_blocks()["Themes"]
labels = list(blocks.keys())
variants = json.load(open(ROOT / "data/reference/theme_variants.json", encoding="utf-8"))
assert set(variants) == set(labels)

model = SentenceTransformer(getattr(en, "DEFAULT_MODEL", "all-MiniLM-L6-v2"), device="cpu")
enc = lambda t: model.encode(t, normalize_embeddings=True, show_progress_bar=False)
V = np.stack([enc([blocks[l]] + variants[l]) for l in labels])      # (K, 4, d)
mean_v = V.mean(1)
mean_v /= np.linalg.norm(mean_v, axis=1, keepdims=True)

emb_dir = ROOT / "data/embeddings"
E = np.load(emb_dir / "p2_all-MiniLM-L6-v2_clean_books.npy")
ids = [str(x) for x in json.load(open(emb_dir / "p2_all-MiniLM-L6-v2_clean_books_ids.json"))]
pos = {b: i for i, b in enumerate(ids)}
E = E[[pos[b] for b in df["book_id"]]]                               # aligné sur le parquet

sims = E @ mean_v.T
z = en.zscore_columns(sims)
df["themes"] = en.select_labels(z, labels, en.TOP_K["Themes"], Z_MIN_THEMES, 1)

new_scores = []
for i, raw in enumerate(df["label_scores"]):
    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    d["Themes"] = {l: round(float(sims[i, j]), 3) for j, l in enumerate(labels)}
    new_scores.append(json.dumps(d) if isinstance(raw, str) else d)
df["label_scores"] = new_scores

df.to_parquet(parquet, index=False)
print("OK ->", parquet, "| thèmes par livre :", round(df["themes"].map(len).mean(), 2))
