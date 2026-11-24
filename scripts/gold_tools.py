"""
Tools for the reference ("gold") annotation.

  1) Create a TEST set of 40 books, disjoint from the 60 tuning books:
       python scripts/gold_tools.py make-holdout

  2) Review / fill in an annotation in the terminal (1 book at a time, can be resumed):
       python scripts/gold_tools.py review --gold data/reference/gold_60_clean.csv --annotator Hajar
       python scripts/gold_tools.py review --gold data/reference/gold_holdout.csv   --annotator Hajar

For each book the title + description are shown, then the NUMBERED list of the blocks.json labels.
Accepted answers: numbers ("3,7"), label names separated by "|", ENTER to keep the displayed labels
(review), "s" to skip, "q" to save and quit. Each validated book gets
`annotator` = your first name; the file is saved after each book.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import enrichment as en  # noqa: E402

SEP = " | "
GOLD_COLS = ["book_id", "title", "authors_str", "description", "gold_genres", "gold_themes", "annotator"]


def is_reviewed(annotator: str) -> bool:
    """A book is "reviewed by a human" if `annotator` is filled in and does not name an LLM."""
    a = str(annotator or "").strip().lower()
    return bool(a) and "llm" not in a and "claude" not in a and "relire" not in a


def split_labels(cell: str) -> list[str]:
    return [x.strip() for x in str(cell or "").split("|") if x.strip()]


def parse_choice(text: str, labels: list[str], max_labels: int) -> list[str] | None:
    """Numbers ("3,7") or names separated by "|" -> list of valid labels; None if invalid."""
    text = text.strip()
    if not text:
        return None
    chosen: list[str] = []
    if "|" in text or any(c.isalpha() for c in text):
        parts = [t.strip() for t in text.split("|") if t.strip()]
        lookup = {l.lower(): l for l in labels}
        for part in parts:
            if part.lower() not in lookup:
                return None
            chosen.append(lookup[part.lower()])
    else:
        for tok in text.replace(" ", ",").split(","):
            if not tok:
                continue
            if not tok.isdigit() or not (1 <= int(tok) <= len(labels)):
                return None
            chosen.append(labels[int(tok) - 1])
    chosen = list(dict.fromkeys(chosen))
    return chosen if 1 <= len(chosen) <= max_labels else None


def menu(labels: list[str]) -> str:
    cols = 3
    cells = [f"{i + 1:>2}. {l}" for i, l in enumerate(labels)]
    cells += [""] * (-len(cells) % cols)
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    return "\n".join("  ".join(c.ljust(34) for c in r).rstrip() for r in rows)


def ask(prompt: str, labels: list[str], current: list[str], max_labels: int, input_fn, print_fn):
    """Returns ('ok', labels) | ('skip', None) | ('quit', None)."""
    while True:
        ans = input_fn(prompt).strip()
        if ans.lower() == "q":
            return "quit", None
        if ans.lower() == "s":
            return "skip", None
        if ans == "" and current:
            return "ok", current
        parsed = parse_choice(ans, labels, max_labels)
        if parsed is not None:
            return "ok", parsed
        print_fn(f"  Réponse invalide : 1 à {max_labels} numéros entre 1 et {len(labels)}, ou noms exacts séparés par « | ».")


def review_rows(df: pd.DataFrame, blocks: dict, annotator: str, save_fn, input_fn=input, print_fn=print,
                redo: bool = False) -> pd.DataFrame:
    """Review loop. ``save_fn(df)`` is called after each validated book."""
    glabels, tlabels = list(blocks["Genre"]), list(blocks["Themes"])
    todo = [i for i in df.index if redo or not is_reviewed(df.at[i, "annotator"])]
    print_fn(f"{len(todo)} livre(s) à valider sur {len(df)}.")
    for n, i in enumerate(todo, 1):
        r = df.loc[i]
        print_fn("\n" + "=" * 100)
        print_fn(f"[{n}/{len(todo)}] {r['title']}  —  {r.get('authors_str', '')}")
        print_fn(str(r["description"])[:700])
        cur_g, cur_t = split_labels(r["gold_genres"]), split_labels(r["gold_themes"])
        print_fn("\nGENRES (1 à 3) :\n" + menu(glabels))
        if cur_g:
            print_fn(f"  Proposé : {SEP.join(cur_g)}   (ENTRÉE = garder)")
        st, g = ask("Genres > ", glabels, cur_g, 3, input_fn, print_fn)
        if st == "quit":
            break
        if st == "skip":
            continue
        print_fn("\nTHÈMES (2 à 3 conseillés, 1 à 3 acceptés) :\n" + menu(tlabels))
        if cur_t:
            print_fn(f"  Proposé : {SEP.join(cur_t)}   (ENTRÉE = garder)")
        st, t = ask("Thèmes > ", tlabels, cur_t, 3, input_fn, print_fn)
        if st == "quit":
            break
        if st == "skip":
            continue
        df.at[i, "gold_genres"], df.at[i, "gold_themes"], df.at[i, "annotator"] = SEP.join(g), SEP.join(t), annotator
        save_fn(df)
    return df


def cmd_make_holdout(a) -> None:
    cat = pd.read_parquet(a.catalogue)
    excl = pd.read_csv(a.exclude, dtype=str, keep_default_na=False)["book_id"] if Path(a.exclude).exists() else []
    sheet = en.make_annotation_sheet(cat, n=a.n, seed=a.seed, n_nonfiction=a.n_nonfiction, exclude_ids=excl)
    sheet["description"] = sheet["description"].str.slice(0, 700)
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} existe déjà (ton annotation en cours !). Utilise --force pour l'écraser.")
    sheet[GOLD_COLS].to_csv(out, index=False)
    print(f"{len(sheet)} livres écrits dans {out} (aucun des {len(set(excl))} livres du réglage). "
          f"Étape suivante : python scripts/gold_tools.py review --gold {out} --annotator TON_PRENOM")


def cmd_review(a) -> None:
    path = Path(a.gold)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in GOLD_COLS:
        if c not in df:
            df[c] = ""
    blocks = en.load_blocks()
    save = lambda d: d[[c for c in GOLD_COLS] + [c for c in d.columns if c not in GOLD_COLS]].to_csv(path, index=False)
    review_rows(df, blocks, a.annotator, save, redo=a.redo)
    save(df)
    done = sum(is_reviewed(x) for x in df["annotator"])
    print(f"\nSauvegardé : {path} — {done}/{len(df)} livres validés par un humain.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-holdout", help="crée la feuille du jeu de test (disjoint du réglage)")
    m.add_argument("--catalogue", default=str(ROOT / "data/processed/books_clean.parquet"))
    m.add_argument("--exclude", default=str(ROOT / "data/reference/gold_60_clean.csv"))
    m.add_argument("--out", default=str(ROOT / "data/reference/gold_holdout.csv"))
    m.add_argument("--n", type=int, default=40)
    m.add_argument("--n-nonfiction", type=int, default=10)
    m.add_argument("--seed", type=int, default=2026)
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_make_holdout)
    r = sub.add_parser("review", help="relire / remplir une annotation en terminal")
    r.add_argument("--gold", required=True)
    r.add_argument("--annotator", required=True, help="ton prénom (écrit dans la colonne annotator)")
    r.add_argument("--redo", action="store_true", help="repasser aussi sur les livres déjà validés")
    r.set_defaults(fn=cmd_review)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
