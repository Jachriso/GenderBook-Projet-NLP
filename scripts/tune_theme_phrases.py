"""Compare les phrases actuelles des thèmes à des variantes plus concrètes (moyenne ou max).
Réutilise les embeddings de livres déjà calculés. Ne modifie ni blocks.json ni le parquet.

IMPORTANT : ce script évalue sur le jeu de RÉGLAGE (gold_60_clean.csv) uniquement. Ne JAMAIS l'utiliser avec
le jeu de test (gold_holdout.csv) : il est réservé à la mesure finale (scripts/evaluate_enrichment.py).
Les variantes retenues sont déjà dans data/reference/theme_variants.json ; `--write` les réécrit depuis VARIANTS
(par défaut le fichier n'est plus modifié, pour éviter un écrasement accidentel)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en
from sentence_transformers import SentenceTransformer

VARIANTS = {
"Identity & self-discovery": ["A character struggles to understand who they really are and what they want from life.",
    "A journey of self-discovery in which someone searches for their true identity and place in the world.",
    "A protagonist questions their beliefs, origins and sense of self."],
"Family relationships": ["A story about the bonds, tensions and secrets between parents, children and siblings.",
    "A family is torn apart or brought together by conflict, loyalty and old wounds.",
    "A story about marriage, motherhood, fatherhood and the ties of family life."],
"Friendship": ["Close friends support each other through difficult times and adventures.",
    "A story about loyalty and the deep bond between best friends.",
    "A group of companions who stand by each other against all odds."],
"Love & heartbreak": ["Two people fall in love, fight their feelings and face obstacles to be together.",
    "A love story full of passion, longing, betrayal and heartbreak.",
    "A romantic relationship that blossoms, falters or ends in pain."],
"Coming of age": ["A teenager grows up, faces first love, school and the pains of adolescence.",
    "A young person leaves childhood behind and learns hard lessons about adult life.",
    "The awkward, funny and painful transition from childhood to adulthood."],
"Morality & guilt": ["A character is haunted by guilt after a terrible choice and must face the moral consequences.",
    "A story about right and wrong, conscience and difficult ethical dilemmas.",
    "Someone struggles with what they have done and whether they can live with it."],
"Power & politics": ["Rulers, politicians and rivals scheme and fight for power and control of a kingdom or nation.",
    "A story of political intrigue, corruption, dictatorship and government secrets.",
    "A struggle for the throne, for leadership or for control over a society."],
"War & conflict": ["Soldiers fight on the battlefield in a brutal war that changes the lives of everyone involved.",
    "A story set during a war, with battles, invasion, occupation and the cost of violence.",
    "The history of a military campaign, its commanders and its devastating human toll."],
"Survival": ["Someone must survive against hunger, danger, wilderness or a deadly threat.",
    "A fight to stay alive in extreme conditions after a disaster, shipwreck or attack.",
    "Characters struggle to endure and escape a life-threatening situation."],
"Grief & loss": ["A character mourns the death of a loved one and struggles to carry on.",
    "A story about bereavement, sorrow, absence and learning to live after a devastating loss.",
    "The aftermath of a death that changes a family or a person forever."],
"Social injustice": ["A story about racism, poverty, prejudice and unfair treatment of the powerless.",
    "People fight discrimination, inequality and an unjust society.",
    "The lives of the marginalised and oppressed, and their struggle for dignity and rights."],
"Revenge": ["A character plans and carries out revenge on those who wronged or betrayed them.",
    "A vendetta fuelled by hatred, with a hunger for payback and retribution.",
    "Someone seeks to destroy their enemy to avenge a past crime."],
"Redemption": ["A flawed person seeks forgiveness and a second chance to make up for past mistakes.",
    "A story of atonement, in which someone tries to become a better person.",
    "A character finds hope and a way to put right the wrongs of their past."],
"Freedom & oppression": ["People live under a tyrannical regime and dream of escape and freedom.",
    "A story of resistance against oppression, imprisonment, censorship or control.",
    "A fight to break free from confinement, slavery or an authoritarian system."],
"Technology & humanity": ["Advanced technology, artificial intelligence or science changes what it means to be human.",
    "A story exploring machines, robots, computers and the future of humankind.",
    "Scientific invention raises ethical questions about humans and technology."],
"Nature & environment": ["A story about wilderness, animals, climate and humanity's place in the natural world.",
    "The beauty and fragility of nature, the planet and its wildlife.",
    "People confront the forces of nature, the land and the environment."],
"Faith & spirituality": ["A story about God, faith, prayer and the search for spiritual meaning.",
    "Religious belief, doubt, devotion and the role of the church or religion in life.",
    "A spiritual quest for truth, divine purpose and inner peace."],
"Madness & the mind": ["A character descends into madness, paranoia, obsession or mental breakdown.",
    "A psychological story about the fragile mind, delusion, trauma and sanity.",
    "The workings of the human psyche, thoughts, memory and mental illness."],
"Adventure & exploration": ["A thrilling journey to unknown lands, with quests, treasure and daring expeditions.",
    "Heroes travel across the world and face dangers on a grand adventure.",
    "An explorer or traveller discovers new places, cultures and mysteries."],
"Crime & justice": ["A murder or crime is investigated by police, detectives or lawyers who seek the truth.",
    "A story about criminals, courts, trials, punishment and the pursuit of justice.",
    "A killer is hunted while the investigation uncovers dark secrets."],
}

model_name = getattr(en, "DEFAULT_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(model_name, device="cpu")
enc = lambda t: model.encode(t, normalize_embeddings=True, show_progress_bar=False)

blocks = en.load_blocks()["Themes"]
labels = list(blocks.keys())
assert set(VARIANTS) == set(labels), set(labels) ^ set(VARIANTS)

# embeddings livres, alignés sur books_clean (cache de P2)
emb_dir = ROOT / "data/embeddings"
E = np.load(emb_dir / "p2_all-MiniLM-L6-v2_clean_books.npy")
ids = [str(x) for x in json.load(open(emb_dir / "p2_all-MiniLM-L6-v2_clean_books_ids.json"))]
pos = {b: i for i, b in enumerate(ids)}

orig = enc([blocks[l] for l in labels])                                   # (K, d)
V = np.stack([enc([blocks[l]] + VARIANTS[l]) for l in labels])           # (K, 4, d)
mean_v = V.mean(1); mean_v /= np.linalg.norm(mean_v, axis=1, keepdims=True)

SIMS = {
    "actuel (1 phrase)": E @ orig.T,
    "enrichi, moyenne": E @ mean_v.T,
    "enrichi, max": np.max(np.einsum("nd,kmd->nkm", E, V), axis=2),
}

gold = pd.read_csv(ROOT / "data/reference/gold_60_clean.csv", dtype=str, keep_default_na=False)
gold = gold[gold["book_id"].isin(pos)]
truth = [{x.strip() for x in g.split("|") if x.strip()} for g in gold["gold_themes"]]
idx = [pos[b] for b in gold["book_id"]]

rows = []
for name, sims in SIMS.items():
    z = en.zscore_columns(sims)
    for z_min in (0.5, 1.0, 1.5):
        ch = en.select_labels(z[idx], labels, 3, z_min, 1)
        P = np.mean([len(set(c) & t) / len(c) for c, t in zip(ch, truth)])
        R = np.mean([len(set(c) & t) / len(t) for c, t in zip(ch, truth)])
        H = np.mean([len(set(c) & t) > 0 for c, t in zip(ch, truth)])
        rows.append(dict(methode=name, z_min=z_min, labels_par_livre=round(np.mean([len(c) for c in ch]), 2),
                         precision=round(P, 2), rappel=round(R, 2), F1=round(2*P*R/(P+R+1e-9), 2), au_moins_1=f"{H:.0%}"))
print(pd.DataFrame(rows).to_string(index=False))
if "--write" in sys.argv:
    json.dump(VARIANTS, open(ROOT / "data/reference/theme_variants.json", "w"), ensure_ascii=False, indent=1)
    print("theme_variants.json réécrit")
