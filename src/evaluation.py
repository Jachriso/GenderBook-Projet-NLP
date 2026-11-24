"""
evaluation.py: end-to-end evaluation of the recommender (Person 6).

Used by notebooks/06_evaluation.ipynb (report section 7).

The evaluation runs the REAL pipeline of P4 (filters -> retrieval -> re-ranking ->
diversity) on test personas whose expected answer was fixed by hand BEFORE running
the system (data/eval/personas.json). Nothing here re-implements the scoring: we only
call `Recommender.score_table` and `Recommender.diversify`, so what we measure is
exactly what the application shows.

Metrics (computed on the top 3 of each persona):
  * genre_precision@3  share of the 3 books with at least one expected genre
  * theme_overlap@3    share of the expected themes found among the themes of the 3 books
  * hit@3              1 if at least one hand-picked relevant book is in the top 3
  * reciprocal rank    1 / rank of the first relevant book in the ranked candidates
                       (0 if no relevant book is among the retrieved candidates); MRR = mean
  * constraints_ok     no excluded author and no "must not appear" book in the top 3
  * loved_book_in_top3 the system returns the book the user said they already loved
                       (reported, but never counted as relevant)
  * stability          Jaccard similarity of the top 3 obtained with rephrased requests
  * judged precision@3 share of the top 3 judged relevant by hand, in a blind pooled
                       judgment file (see section 7)
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src import config
from src.data_loader import author_key
from src.profile import build_profile

EVAL_DIR = config.DATA_DIR / "eval"
PERSONAS_PATH = EVAL_DIR / "personas.json"
RESULTS_DIR = EVAL_DIR / "results"


# --------------------------------------------------------------------------- #
# 1. Test personas
# --------------------------------------------------------------------------- #

def load_eval_personas(path: Path = PERSONAS_PATH) -> list[dict]:
    """Read the evaluation personas (list of {id, name, answers, paraphrases, expected})."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["personas"]


def relevant_works(persona: dict) -> list[set[str]]:
    """Hand-picked relevant works, each as a SET of book_ids.

    A work may still exist in several editions in the catalogue (Middlemarch x3):
    any of its book_ids counts as a hit.
    """
    return [set(map(str, w["book_ids"])) for w in persona["expected"]["relevant"]]


def forbidden_ids(persona: dict) -> set[str]:
    """book_ids that must NOT be recommended (e.g. relevant topic but excluded format)."""
    return {str(b) for w in persona["expected"].get("must_not_appear", []) for b in w["book_ids"]}


def loved_book_ids(persona: dict) -> set[str]:
    """book_ids of the work described in `loved_book`: the user has ALREADY read it.

    It is deliberately not counted as relevant (recommending a book the reader already
    knows is not useful); we only report how often the system returns it.
    """
    return {str(b) for b in persona["expected"].get("loved_book_work", {}).get("book_ids", [])}


# --------------------------------------------------------------------------- #
# 2. Running the pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(rec, answers: dict, top_k: int = 3, lambda_pop: float | None = None,
                 **formula_kwargs) -> dict[str, Any]:
    """Run steps 1-4 of P4's pipeline and return the ranked candidates and the top k.

    Same steps as `Recommender.recommend`, minus the construction of the Recommendation
    objects (explanations, best sentence), which the metrics do not need.

    Args:
        rec: a src.recommender.Recommender (holds the catalogue, engine and calibration).
        answers: questionnaire answers (answers.json format).
        lambda_pop: if given, overrides the popularity weight λ of the profile
            (ablation "without popularity" = 0). λ normally comes from the Likert answer.
        formula_kwargs: alpha, beta, mu, tau, use_calibration (see scoring.score_books).

    Returns:
        {"profile", "table" (candidates sorted by final score), "top" (row indices),
         "top_ids" (book_ids), "seconds"}.
    """
    t0 = time.perf_counter()
    profile = build_profile(answers, rec.engine, rec.books, rec.blocks)
    if lambda_pop is not None:
        profile.lambda_pop = float(lambda_pop)
    table = rec.score_table(profile, **formula_kwargs)            # (1) filters, (2) retrieval, (3) re-ranking
    top = rec.diversify(table, rec.books, top_k) if not table.empty else []   # (4) diversity
    return {
        "profile": profile,
        "table": table,
        "top": top,
        "top_ids": [str(rec.books.at[i, "book_id"]) for i in top],
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# 3. Metrics for one persona
# --------------------------------------------------------------------------- #

def first_relevant_rank(table: pd.DataFrame, books: pd.DataFrame, works: list[set[str]]) -> int | None:
    """1-based rank of the first relevant book in the ranked candidates (None if absent).

    The ranking is the re-ranked candidate list BEFORE diversity, i.e. the full ordering
    produced by the formula. Books outside the retrieved candidates have no rank.
    """
    wanted = set().union(*works) if works else set()
    ids = books.loc[table.index, "book_id"].astype(str).tolist()
    for rank, bid in enumerate(ids, start=1):
        if bid in wanted:
            return rank
    return None


def persona_metrics(persona: dict, run: dict, books: pd.DataFrame) -> dict[str, Any]:
    """All the metrics of one persona from one pipeline run."""
    exp = persona["expected"]
    top = run["top"]
    works = relevant_works(persona)

    # Genre-precision@3: does each recommended book carry at least one expected genre?
    exp_genres = set(exp["genres"])
    genre_hits = [bool(exp_genres & set(books.at[i, "genres"])) for i in top]

    # Theme-overlap@3: which expected themes appear among the themes of the 3 books?
    top_themes = set().union(*(set(books.at[i, "themes"]) for i in top)) if top else set()
    exp_themes = set(exp["themes"])

    # Hand-picked relevant books: in the top 3? at which rank among the candidates?
    top_ids = set(run["top_ids"])
    hit = any(top_ids & w for w in works)
    rank = first_relevant_rank(run["table"], books, works)

    # Explicit user choices: excluded authors and forbidden books never recommended
    excluded = {author_key(a) for a in exp.get("excluded_authors", [])}
    top_authors = {author_key(a) for i in top for a in books.at[i, "authors"]}
    constraints_ok = not (excluded & top_authors) and not (top_ids & forbidden_ids(persona))

    return {
        "persona": persona["id"],
        "genre_precision@3": float(np.mean(genre_hits)) if top else 0.0,
        "theme_overlap@3": len(exp_themes & top_themes) / len(exp_themes) if exp_themes else np.nan,
        "hit@3": float(hit),
        "first_relevant_rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "constraints_ok": bool(constraints_ok),
        "loved_book_in_top3": bool(top_ids & loved_book_ids(persona)),
        "n_candidates": int(len(run["table"])),
        "seconds": round(run["seconds"], 3),
        "top3": " | ".join(str(books.at[i, "title"])[:45] for i in top),
        "top_ids": list(run["top_ids"]),
    }


def tops_by_persona(df: pd.DataFrame) -> dict[str, list[str]]:
    """{persona id: top-3 book_ids} from the output of `evaluate` (for the judgment pool)."""
    return dict(zip(df["persona"], df["top_ids"]))


def load_tops(results_dir: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Reload the top 3 of every evaluated encoder from the saved results (main_<encoder>.csv).

    Lets the judgment cells run on their own, without re-running the whole notebook.
    """
    import ast
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    tops = {}
    for path in sorted(results_dir.glob("main_*.csv")):
        df = pd.read_csv(path)
        df["top_ids"] = df["top_ids"].map(ast.literal_eval)
        tops[path.stem.removeprefix("main_")] = tops_by_persona(df)
    return tops


def evaluate(rec, personas: list[dict], **run_kwargs) -> pd.DataFrame:
    """One row of metrics per persona (run_kwargs are passed to run_pipeline)."""
    rows = [persona_metrics(p, run_pipeline(rec, p["answers"], **run_kwargs), rec.books) for p in personas]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Mean of each metric over the personas (MRR = mean of the reciprocal ranks)."""
    return {
        "genre_precision@3": round(df["genre_precision@3"].mean(), 3),
        "theme_overlap@3": round(df["theme_overlap@3"].mean(), 3),
        "hit@3": round(df["hit@3"].mean(), 3),
        "MRR": round(df["reciprocal_rank"].mean(), 3),
        "constraints_ok": f'{int(df["constraints_ok"].sum())}/{len(df)}',
        "loved_book_returned": f'{int(df["loved_book_in_top3"].sum())}/{len(df)}' if "loved_book_in_top3" in df else "-",
        "median_seconds": round(df["seconds"].median(), 3),
    }


# --------------------------------------------------------------------------- #
# 4. Robustness to rephrasing
# --------------------------------------------------------------------------- #

def jaccard(a: Iterable, b: Iterable) -> float:
    """|A ∩ B| / |A ∪ B| (1 = same set, 0 = nothing in common)."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def with_ideal_book(answers: dict, text: str) -> dict:
    """Copy of the answers where only the main free-text request is replaced."""
    new = copy.deepcopy(answers)
    new["free_text"]["ideal_book"] = text
    return new


def paraphrase_stability(rec, personas: list[dict]) -> pd.DataFrame:
    """For each persona: Jaccard between the original top 3 and the top 3 of each rephrasing."""
    rows = []
    for p in personas:
        base = run_pipeline(rec, p["answers"])["top_ids"]
        scores = [jaccard(base, run_pipeline(rec, with_ideal_book(p["answers"], para))["top_ids"])
                  for para in p["paraphrases"]]
        rows.append({"persona": p["id"], "mean_jaccard": float(np.mean(scores)),
                     "identical_top3": sum(s == 1.0 for s in scores), "n_paraphrases": len(scores)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Ablation of the formula and sensitivity to the user's weights
# --------------------------------------------------------------------------- #

# Each variant removes ONE term of the formula of section 5 (or the calibration).
ABLATIONS: dict[str, dict[str, Any]] = {
    "Blurb only (S_desc)":         {"beta": 0.0, "mu": 0.0, "lambda_pop": 0.0},
    "Coverage only":               {"alpha": 0.0, "mu": 0.0, "lambda_pop": 0.0},
    "Without calibration":         {"use_calibration": False},
    "Without 'to avoid' penalty":  {"mu": 0.0},
    "Without popularity":          {"lambda_pop": 0.0},
    "Full formula":                {},
}


def ablation(rec, personas: list[dict], variants: dict[str, dict] = ABLATIONS) -> pd.DataFrame:
    """Summary metrics for each variant of the formula."""
    rows = []
    for name, kwargs in variants.items():
        rows.append({"variant": name, **summarize(evaluate(rec, personas, **kwargs))})
    return pd.DataFrame(rows).set_index("variant")


def weight_sensitivity(rec, personas: list[dict], block: str = "Themes",
                       values: Iterable[int] = (1, 2, 3, 4, 5), top_n: int = 10) -> pd.DataFrame:
    """Vary the importance Likert of one block and measure that block's score in the top n.

    If the system is consistent, a higher weight should push up the books that are
    close on that block, so the mean block score of the top n should increase.
    """
    key = f"w_{block.lower()}"
    rows = []
    for v in values:
        scores = []
        for p in personas:
            answers = copy.deepcopy(p["answers"])
            answers["likert"][key] = v
            table = run_pipeline(rec, answers)["table"]
            scores.append(table[block].head(top_n).mean())
        rows.append({f"W_{block}": v, f"mean {block} score of top {top_n}": round(float(np.nanmean(scores)), 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Building a recommender for a given encoder
# --------------------------------------------------------------------------- #

def make_recommender(encoder: str, books: pd.DataFrame | None = None, blocks: dict | None = None):
    """Recommender using one of P3's encoders: "minilm", "mpnet", "tfidf-lsa", "glove", "glove-tfidf".

    SBERT models use P3's EmbeddingEngine (the production engine). The baselines are
    plugged into P4's LocalEngine through get_encoder(). Each engine has a unique
    model_name, hence its own embedding cache and its own calibration.
    """
    from src.data_loader import load_books
    from src.recommender import Recommender, load_blocks_safe

    books = books if books is not None else load_books()
    blocks = blocks if blocks is not None else load_blocks_safe()
    if encoder in ("minilm", "mpnet"):
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine(encoder, books=books)
    else:
        from src.encoders import get_encoder
        from src.fallback_engine import LocalEngine
        enc = get_encoder(encoder, corpus=books["embed_text"].fillna("").tolist())
        engine = LocalEngine(books, blocks, encoder=enc)
    return Recommender(books=books, blocks=blocks, engine=engine)


# --------------------------------------------------------------------------- #
# 7. Blind manual relevance judgments
# --------------------------------------------------------------------------- #
# Hit@3 and MRR only know the 3 to 6 books we picked in advance: a relevant book we
# did not think of counts as a miss. The standard fix (pooling, as in TREC) is to judge
# by hand every book that ANY system put in its top 3. The file is BLIND: the judge
# sees the request and the book, never which encoder proposed it, and the rows are
# shuffled. Existing judgments are kept when new systems are added to the pool.

JUDGMENTS_PATH = EVAL_DIR / "judgments.csv"
JUDGMENT_COLUMNS = ["persona", "request", "book_id", "title", "authors", "genres", "blurb", "relevant"]


def build_judgment_pool(tops: dict[str, dict[str, list[str]]], personas: list[dict],
                        books: pd.DataFrame, path: Path = JUDGMENTS_PATH,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Create or extend the blind judgment file from the top 3 of each system.

    Args:
        tops: {system name: {persona id: [book_id, ...]}}.
    Returns the full pool; rows already judged keep their `relevant` value (1 / 0).
    """
    by_id = {p["id"]: p for p in personas}
    info = books.set_index(books["book_id"].astype(str))
    old = pd.read_csv(path, dtype={"book_id": str}) if Path(path).exists() else pd.DataFrame(columns=JUDGMENT_COLUMNS)
    seen = set(zip(old["persona"], old["book_id"]))
    new_rows = []
    for system_tops in tops.values():
        for pid, ids in system_tops.items():
            ft = by_id[pid]["answers"]["free_text"]
            request = f'{ft["ideal_book"]} Atmosphere: {ft["atmosphere"]} Avoid: {ft.get("avoid", "")}'
            for bid in ids:
                if (pid, bid) in seen:
                    continue
                seen.add((pid, bid))
                b = info.loc[bid]
                new_rows.append({"persona": pid, "request": request, "book_id": bid, "title": b["title"],
                                 "authors": b["authors_str"], "genres": ", ".join(b["genres"]),
                                 "blurb": str(b["description"])[:600], "relevant": ""})
    # Only the NEW rows are shuffled (so that their order never reveals the system), then
    # appended: rows already in the file keep their place, the judge never sees them move.
    new = pd.DataFrame(new_rows, columns=JUDGMENT_COLUMNS).sample(frac=1, random_state=seed)
    pool = pd.concat([old, new], ignore_index=True)
    pool = pool.sort_values("persona", kind="stable").reset_index(drop=True)   # stable: old order kept
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(path, index=False)
    return pool


def n_unjudged(pool: pd.DataFrame) -> int:
    """Number of rows whose `relevant` cell is still empty."""
    rel = pool["relevant"]
    return int((rel.isna() | rel.astype("string").str.strip().fillna("").eq("")).sum())


def judged_precision(tops: dict[str, dict[str, list[str]]], path: Path = JUDGMENTS_PATH) -> pd.DataFrame:
    """Precision@3 according to the manual judgments, for each system.

    precision@3 = share of the recommended books judged relevant (1), averaged over the
    personas. Books not judged yet are counted in `n_unjudged` and left out.
    """
    j = pd.read_csv(path, dtype={"book_id": str})
    j = j[j["relevant"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    label = {(r.persona, r.book_id): int(float(r.relevant)) for r in j.itertuples()}
    rows = []
    for system, system_tops in tops.items():
        per_persona, unjudged = [], 0
        for pid, ids in system_tops.items():
            vals = [label[(pid, b)] for b in ids if (pid, b) in label]
            unjudged += len(ids) - len(vals)
            if vals:
                per_persona.append(np.mean(vals))
        rows.append({"system": system,
                     "judged_precision@3": round(float(np.mean(per_persona)), 3) if per_persona else np.nan,
                     "n_unjudged": unjudged})
    return pd.DataFrame(rows).set_index("system")


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Save a results table as CSV in data/eval/results/ (used to fill the report)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path)
    return path
