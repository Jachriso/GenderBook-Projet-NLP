"""
Démo « sémantique vs mots-clés » : TF-IDF vs GloVe vs SBERT sur les 500 livres de l'échantillon.

Chaque requête PARAPHRASE le résumé d'un livre précis SANS reprendre ses mots importants
(ni titre, ni nom de personnage, ni mots-clés du résumé). Question posée à chaque méthode :
à quel RANG (sur 500) place-t-elle le livre attendu ?
    - rang 1   : la méthode a trouvé le bon livre en premier ;
    - rang ~250 : la méthode ne trouve rien (rang « au hasard »).

Lancer depuis la racine du repo :
    python -X utf8 -m scripts.p3_compare_encoders                 # tfidf, glove, glove-tfidf, minilm
    python -X utf8 -m scripts.p3_compare_encoders --mpnet         # + all-mpnet-base-v2 (plus lent)
    python -X utf8 -m scripts.p3_compare_encoders --only tfidf minilm
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.encoders import compare_encoders, get_encoder

# (requête paraphrasée, book_id du livre attendu). Mots ÉVITÉS volontairement entre crochets.
QUERIES = [
    # Magician (Feist)          [orphan, boy, magician, kingdom, alien, invaders, worlds]
    ("a parentless lad learns spells from an old wizard while creatures from another dimension "
     "attack his homeland", "9780586217832"),
    # Dust and Shadow (Faye)    [Sherlock, Holmes, Watson, Ripper, London, killer, street]
    ("the celebrated pipe-smoking sleuth tracks a serial murderer of prostitutes in Victorian times",
     "9781416583318"),
    # Skeletons on the Zahara   [shipwrecked, sailors, Sahara, nomads, captured, sold]
    ("mariners whose vessel sank are seized as captives by Bedouin tribes and marched across scorching dunes",
     "9780316159357"),
    # Anne of Green Gables      [girl, orphan, red hair, imagination, farm, brother]
    ("a chatty ginger-haired adopted child with a vivid fancy changes the lives of an elderly "
     "pair of siblings on a rural homestead", "9781442490017"),
    # Lysistrata (Aristophanes) [Athens, Attic, comedy, women, antiquity, politics]
    ("a Greek comic play from two thousand years ago showing what wives and mothers experienced",
     "9780198144960"),
    # Dracula (graphic novel)   [vampire, Count, Transylvania, horror, adaptation]
    ("an illustrated retelling where a bloodsucking aristocrat from the Carpathians stalks young ladies",
     "9780785149064"),
    # The Good Soldier Svejk    [war, soldier, satire, empire, novel]
    ("an antimilitarist comic tale mocking the Habsburg monarchy through a naive army private",
     "9781857151510"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpnet", action="store_true", help="ajoute all-mpnet-base-v2")
    parser.add_argument("--only", nargs="+", help="liste d'encodeurs à comparer (ex. tfidf minilm)")
    args = parser.parse_args()

    # 1. Catalogue de développement (500 livres)
    path = config.DATA_PROCESSED / "books_enriched_sample.parquet"
    books = pd.read_parquet(path if path.exists() else config.BOOKS_SAMPLE)
    corpus = books["embed_text"].fillna("").tolist()

    # 2. Encodeurs à comparer (TF-IDF et GloVe-TFIDF apprennent leurs IDF sur ce catalogue)
    names = args.only or ["tfidf", "glove", "glove-tfidf", "minilm"] + (["mpnet"] if args.mpnet else [])
    encoders = {}
    for name in names:
        print(f"Préparation de l'encodeur {name}...")
        encoders[name] = get_encoder(name, corpus=corpus)

    # 3. Rang du livre attendu pour chaque requête et chaque méthode
    res = compare_encoders(encoders, books, QUERIES)

    # 4. Tableau lisible : une ligne par livre attendu, une colonne par méthode
    table = res.pivot(index="target", columns="encoder", values="rank")[names]
    summary = pd.DataFrame({
        "rang moyen": table.mean(),
        "rang médian": table.median(),
        "top-1 (%)": (table == 1).mean() * 100,
        "top-10 (%)": (table <= 10).mean() * 100,
        "MRR": (1 / table).mean(),          # Mean Reciprocal Rank : 1 = toujours 1er, ~0 = jamais trouvé
    }).T.round(2)
    seconds = res.groupby("encoder")["seconds"].first()[names]

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 45)
    print(f"\nCatalogue : {len(books)} livres | rang au hasard ≈ {(len(books) + 1) / 2:.0f}\n")
    print("RANG DU LIVRE ATTENDU (1 = trouvé en premier)")
    print(table.to_string())
    print("\nRÉSUMÉ")
    print(summary.to_string())
    print("\nTEMPS (encodage des 500 livres + requêtes, en secondes)")
    print(seconds.to_string())

    # 5. Ce que chaque méthode a mis en 1re position (utile pour commenter les échecs)
    print("\nLIVRE CLASSÉ 1er PAR CHAQUE MÉTHODE")
    print(res.pivot(index="target", columns="encoder", values="top1_title")[names].to_string())

    # 6. Sauvegarde pour le notebook 03 et le rapport
    # Nom du fichier = encodeurs comparés, pour ne jamais écraser une comparaison précédente
    # (le lancement par défaut garde le nom historique p3_encoder_comparison_sample.csv).
    suffix = "" if not (args.only or args.mpnet) else "_" + "-".join(names)
    out = config.FIGURES_DIR / f"p3_encoder_comparison_sample{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nRésultats détaillés sauvegardés : {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
