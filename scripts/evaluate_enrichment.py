"""
Validate the P2 enrichment against manual annotations, with a TUNING / TEST split.

  dev  = gold_60_clean.csv   : the 60 books used to choose the theme sentences and the z_min threshold
                               (optimistic scores: these books were "seen" during tuning)
  test = gold_holdout.csv    : 40 books NEVER used for tuning (created by gold_tools.py make-holdout)
                               => THIS is the score to quote as the system's performance.

Usage:
    python scripts/evaluate_enrichment.py                       # shows dev + test
    python scripts/evaluate_enrichment.py --write-report        # also writes the results block of the report

Each metric is given with a 95% confidence interval (bootstrap over the books): on ~40 books
it is wide, which is honest information to report.
The script REFUSES to write the report if an annotation has not been reviewed by a human (`annotator` column).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gold_tools import is_reviewed, split_labels  # noqa: E402

REPORT = ROOT / "docs" / "report" / "02_enrichment.md"
START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
KINDS = [("genres", "gold_genres"), ("themes", "gold_themes")]


def book_scores(pred: set, truth: set) -> tuple[float, float, int]:
    inter = pred & truth
    return (len(inter) / len(pred) if pred else 0.0, len(inter) / len(truth) if truth else 0.0, int(bool(inter)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% CI (percentiles) by resampling the books."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_set(enriched: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Metrics of an annotated set. Returns {'n', 'n_unannotated', 'reviewed', 'annotators', 'metrics', 'rows'}."""
    enr = enriched.assign(book_id=enriched["book_id"].astype(str)).set_index("book_id")
    gold = gold.assign(book_id=gold["book_id"].astype(str))
    rows, missing = [], []
    for _, g in gold.iterrows():
        if g["book_id"] not in enr.index:
            missing.append(g["book_id"])
            continue
        e = enr.loc[g["book_id"]]
        row = {"book_id": g["book_id"], "title": str(g["title"])[:40], "annotator": g["annotator"]}
        for col, gcol in KINDS:
            pred, truth = set(e[col]), set(split_labels(g[gcol]))
            row[f"{col}_pred"], row[f"{col}_gold"] = " | ".join(sorted(pred)), " | ".join(sorted(truth))
            row[f"{col}_annotated"] = bool(truth)
            row[f"{col}_precision"], row[f"{col}_recall"], row[f"{col}_hit"] = book_scores(pred, truth)
        rows.append(row)
    if missing:
        print(f"  ATTENTION : {len(missing)} book_id absents du catalogue enrichi, ex. {missing[:3]}")
    r = pd.DataFrame(rows)
    metrics = {}
    for col, _ in KINDS:
        sub = r[r[f"{col}_annotated"]]
        m = {"n": len(sub)}
        for key in ["precision", "recall", "hit"]:
            m[key] = bootstrap_ci(sub[f"{col}_{key}"].to_numpy()) if len(sub) else (float("nan"),) * 3
        p, rc = m["precision"][0], m["recall"][0]
        m["f1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
        metrics[col] = m
    names = sorted({a for a in r["annotator"] if is_reviewed(a)})
    return {"n": len(r), "n_unannotated": int((~r["genres_annotated"] | ~r["themes_annotated"]).sum()),
            "reviewed": bool(len(r)) and all(is_reviewed(a) for a in r["annotator"]),
            "n_unreviewed": int(sum(not is_reviewed(a) for a in r["annotator"])), "annotators": names,
            "metrics": metrics, "rows": r}


def fmt(t: tuple[float, float, float], pct: bool = False) -> str:
    m, lo, hi = t
    return f"{m:.0%} [{lo:.0%}-{hi:.0%}]" if pct else f"{m:.2f} [{lo:.2f}-{hi:.2f}]"


def render_block(results: dict[str, dict]) -> str:
    """Markdown block (table + sentence) to insert in the report."""
    labels = {"dev": "Réglage (60 livres, vus pendant le réglage)", "test": "**Test (livres jamais vus)**"}
    lines = ["| Jeu | Livres | Genres : précision / rappel / ≥1 bon label | Thèmes : précision / rappel / ≥1 bon label |",
             "|---|---|---|---|"]
    for name, res in results.items():
        g, t = res["metrics"]["genres"], res["metrics"]["themes"]
        lines.append(f"| {labels.get(name, name)} | {res['n']} | {fmt(g['precision'])} / {fmt(g['recall'])} / "
                     f"{fmt(g['hit'], True)} | {fmt(t['precision'])} / {fmt(t['recall'])} / {fmt(t['hit'], True)} |")
    lines.append("")
    lines.append("Les valeurs entre crochets sont des intervalles de confiance à 95 % (bootstrap sur les livres).")
    test = results.get("test")
    if test:
        t = test["metrics"]["themes"]
        lines.append(f"Les performances à retenir sont celles du jeu de test : sur {test['n']} livres jamais utilisés "
                     f"pour le réglage, les thèmes atteignent une F1 de {t['f1']:.2f}.")
    who = sorted({a for r in results.values() for a in r["annotators"]})
    lines.append(f"Annotations relues par : {', '.join(who)} (proposition initiale assistée par un LLM).")
    return "\n".join(lines)


def inject_block(text: str, block: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"marqueurs {START} / {END} absents du rapport")
    return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enriched", default=str(ROOT / "data/processed/books_enriched.parquet"))
    ap.add_argument("--gold", action="append", metavar="NOM=CHEMIN",
                    help="jeu annoté ; par défaut dev=gold_60_clean.csv et test=gold_holdout.csv")
    ap.add_argument("--write-report", action="store_true", help="insère les résultats dans docs/report/02_enrichment.md")
    a = ap.parse_args()

    specs = a.gold or [f"dev={ROOT / 'data/reference/gold_60_clean.csv'}", f"test={ROOT / 'data/reference/gold_holdout.csv'}"]
    sets = {}
    for sp in specs:
        name, path = sp.split("=", 1)
        if not Path(path).exists():
            print(f"[{name}] fichier absent : {path}")
            continue
        sets[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not sets:
        sys.exit("aucun fichier annoté trouvé")
    if "dev" in sets and "test" in sets:
        overlap = set(sets["dev"]["book_id"]) & set(sets["test"]["book_id"])
        if overlap:
            sys.exit(f"ERREUR : {len(overlap)} livres sont à la fois dans le réglage et dans le test : {sorted(overlap)[:5]}")

    enriched = pd.read_parquet(a.enriched)
    results = {}
    for name, gold in sets.items():
        res = results[name] = evaluate_set(enriched, gold)
        out = ROOT / "data/reference" / ("gold_eval_report.csv" if name == "dev" else f"gold_eval_report_{name}.csv")
        res["rows"].to_csv(out, index=False)
        print(f"\n=== {name} : {res['n']} livres (rapport détaillé : {out.name}) ===")
        for col, _ in KINDS:
            m = res["metrics"][col]
            print(f"{col:7s} n={m['n']:3d}  précision {fmt(m['precision'])}  rappel {fmt(m['recall'])}  "
                  f"F1 {m['f1']:.2f}  ≥1 bon label {fmt(m['hit'], True)}")
        if not res["reviewed"]:
            print(f"  ⚠ {res['n_unreviewed']} annotation(s) NON relue(s) par un humain : "
                  f"python scripts/gold_tools.py review --gold <fichier> --annotator TON_PRENOM")

    if a.write_report:
        if set(results) != {"dev", "test"}:
            sys.exit("--write-report exige les deux jeux (dev et test).")
        bad = [n for n, r in results.items() if not r["reviewed"]]
        if bad:
            sys.exit(f"Refus d'écrire le rapport : annotation non relue par un humain pour {bad}.")
        REPORT.write_text(inject_block(REPORT.read_text(encoding="utf-8"), render_block(results)), encoding="utf-8")
        print(f"\nRapport mis à jour : {REPORT}")


if __name__ == "__main__":
    main()
