r"""
Semantic scoring formula (P4), computed in a VECTORISED way (numpy, no Python
loop over the books).

Notation: u = user, j = book, e_j = embedding of embed_text, all the
vectors are L2-normalised so cos(a, b) = a · b.

1. Global similarity        S_desc(j) = cos(q_global, e_j)
2. Score per block b        S_b(j) = (1/|P_b|) Σ_{p ∈ P_b(u)} max_{l ∈ L_b(j) ∪ {e_j}} cos(p, l)
                            (L_b(j) = labels assigned to the book by P2, encoded by their sentence)
3. Calibration              s̃ = clip((s − p5) / (p95 − p5), 0, 1)
4. Coverage                 Coverage(j) = Σ_b W_b · S̃_b(j) / Σ_b W_b
5. Final score              Final(j) = [α·S̃_desc + β·Coverage + λ·Pop] / (α + β + λ)
                                       − μ · max(0, cos(q_avoid, e_j) − τ)
                            then clipped to [0, 1]

Dividing by (α + β + λ) guarantees Final ∈ [0, 1] whatever λ: a perfect
book is worth 1, and the Excellent / Good / Partial thresholds keep the same meaning
for a user who wants best-sellers (λ = 0.2) or hidden gems (λ = 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.schemas import BLOCKS, UserProfile

# books_enriched column (P2) holding the labels of each block
BLOCK_COLUMNS = {"Genre": "genres", "Themes": "themes", "Style": "styles", "Atmosphere": "atmospheres"}
COMPONENTS = ("S_desc", *BLOCKS)


# ---------------------------------------------------------------------------
# 1. Book label matrices
# ---------------------------------------------------------------------------
def book_label_mask(books: pd.DataFrame, block: str, labels: list[str]) -> np.ndarray:
    """Boolean matrix (N, L): mask[j, l] = True if book j carries label l.

    Fallback: if P2's column does not exist (books_sample / books_clean),
    the matrix is empty and S_b is computed on the description alone.
    """
    col = BLOCK_COLUMNS[block]
    mask = np.zeros((len(books), len(labels)), dtype=bool)
    if col not in books.columns or not labels:
        return mask
    pos = {lab: i for i, lab in enumerate(labels)}
    for j, labs in enumerate(books[col].tolist()):
        for lab in (labs if labs is not None else []):
            i = pos.get(lab)
            if i is not None:
                mask[j, i] = True
    return mask


# ---------------------------------------------------------------------------
# 2. Raw scores (cosine)
# ---------------------------------------------------------------------------
def desc_similarity(q: np.ndarray, book_emb: np.ndarray) -> np.ndarray:
    """S_desc = cos(q_global, e_j) for all books: one matrix-vector product."""
    return book_emb @ np.asarray(q, dtype=np.float32)


def block_similarity(P: np.ndarray, label_emb: np.ndarray, mask: np.ndarray,
                     book_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S_b(j) for all books j (the assignment's "max then mean" formula).

    Args:
        P: (m, d) user preference sentences for this block.
        label_emb: (L, d) sentences of the block's labels.
        mask: (N, L) labels assigned to each book.
        book_emb: (N, d) book embeddings.
    Returns:
        (S_b of shape (N,), best of shape (N, m) = max per user sentence).
    """
    N = book_emb.shape[0]
    if P is None or len(P) == 0:
        return np.zeros(N, dtype=np.float32), np.zeros((N, 0), dtype=np.float32)
    # Step a: cos(user sentence, book description) -> (m, N)
    sim_desc = P @ book_emb.T
    if label_emb is not None and len(label_emb) and mask.any():
        # Step b: cos(user sentence, sentence of each label) -> (m, L)
        sim_lab = P @ label_emb.T
        # Step c: only the labels CARRIED by the book are kept (the others at -inf)
        masked = np.where(mask[None, :, :], sim_lab[:, None, :], -np.inf)   # (m, N, L)
        best_lab = masked.max(axis=2)                                         # (m, N)
        # Step d: max between the best label and the description itself
        best = np.maximum(best_lab, sim_desc)
    else:
        best = sim_desc  # no labels (P2 missing): the description alone
    # Step e: average over the user's sentences
    return best.mean(axis=0).astype(np.float32), best.T.astype(np.float32)


def avoid_similarity(q_avoid: np.ndarray | None, book_emb: np.ndarray) -> np.ndarray:
    """cos(q_avoid, e_j), or 0 if the user has nothing to avoid."""
    if q_avoid is None:
        return np.zeros(book_emb.shape[0], dtype=np.float32)
    return book_emb @ np.asarray(q_avoid, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Percentile calibration
# ---------------------------------------------------------------------------
def calibrate(s: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """s̃ = clip((s − p5) / (p95 − p5), 0, 1): maps compressed cosines onto [0, 1]."""
    return np.clip((np.asarray(s, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def load_calibration(path: Path = config.CALIBRATION_JSON,
                     model: str | None = None) -> dict[str, tuple[float, float]] | None:
    """Read a calibration file: {"S_desc": {"p5": .., "p95": ..}, "Genre": {...}, ...}.

    If ``model`` is given and the file names another model, it is ignored
    (MiniLM percentiles make no sense for MPNet).
    """
    if not Path(path).exists():
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if model and raw.get("model") and str(raw["model"]) != str(model):
        return None
    out = {}
    for comp in COMPONENTS:
        v = raw.get(comp) or (raw.get("desc") if comp == "S_desc" else None)
        if isinstance(v, dict) and "p5" in v and "p95" in v:
            out[comp] = (float(v["p5"]), float(v["p95"]))
    if "S_desc" not in out:
        return None
    for comp in COMPONENTS:   # a partial file (e.g. P3: S_desc only) -> no calibration of the blocks
        out.setdefault(comp, (0.0, 1.0))
    return out


def load_personas(path: Path = config.PERSONAS_JSON) -> dict[str, dict]:
    """Reference personas (answers.json format); falls back on the contract example."""
    if Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    from src.profile import example_answers
    return {"example": example_answers()}


def compute_calibration(engine, books: pd.DataFrame, blocks: dict[str, dict[str, str]],
                        personas: dict[str, dict] | None = None,
                        percentiles: tuple[int, int] = config.CALIBRATION_PERCENTILES,
                        retrieval_k: int = config.RETRIEVAL_TOP_K,
                        save_to: Path | None = config.CALIBRATION_JSON) -> dict[str, tuple[float, float]]:
    """Compute p5 / p95 of each component over the CANDIDATES of reference personas.

    Why the candidates and not the whole catalogue? The retrieval keeps
    the 300 closest books out of 12,000 (top 2.5%): they are ALL above the
    "catalogue" p95, so a calibration on the whole catalogue would give
    S̃_desc = 1 to all candidates and this term would no longer separate anything.
    We therefore measure the score distribution where the scores are used: among the books
    retrieved for realistic profiles (data/reference/personas_p4.json).
    The result is fixed (not recomputed for each query), so a score stays
    comparable from one user to another.

    If P3 provides its own calibration.json, ``load_calibration`` reads it first.
    """
    from src.profile import build_profile

    personas = personas or load_personas()
    E = engine.book_matrix()
    label_mats = {b: engine.label_matrix(b) for b in BLOCKS}
    masks = {b: book_label_mask(books, b, label_mats[b][0]) for b in BLOCKS}
    pooled: dict[str, list[np.ndarray]] = {c: [] for c in COMPONENTS}
    for answers in personas.values():
        prof = build_profile(answers, engine, books, blocks)
        sims = desc_similarity(prof.query_vecs["global"], E)
        cand = np.argsort(-sims)[:retrieval_k]          # same candidates as the recommender
        pooled["S_desc"].append(sims[cand])
        for b in BLOCKS:
            L = label_mats[b][1]
            s_b, _ = block_similarity(prof.query_vecs.get(b), L, masks[b][cand], E[cand])
            pooled[b].append(s_b)
    lo_p, hi_p = percentiles
    out = {c: (float(np.percentile(np.concatenate(v), lo_p)), float(np.percentile(np.concatenate(v), hi_p)))
           for c, v in pooled.items()}
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(json.dumps(
            {**{k: {"p5": round(a, 4), "p95": round(c, 4)} for k, (a, c) in out.items()},
             "source": f"P4: candidates (top-{retrieval_k}) of {len(personas)} reference personas",
             "model": str(getattr(engine, "model_name", "unknown")), "n_books": int(len(books))},
            indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 4. Full formula
# ---------------------------------------------------------------------------
def match_label(score: float) -> str:
    """Readable label of the final score (assignment thresholds, config.MATCH_THRESHOLDS).

    Computed on the score ROUNDED to 2 decimals, as it is displayed: otherwise a
    0.7497 displayed as "0.75" would be labelled Good instead of Excellent.
    """
    score = round(float(score), 2)
    if score >= config.MATCH_THRESHOLDS["Excellent"]:
        return "Excellent"
    if score >= config.MATCH_THRESHOLDS["Good"]:
        return "Good"
    return "Partial"


def coverage_score(block_scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Coverage = Σ W_b · S̃_b / Σ W_b (formula required by the assignment)."""
    total_w = sum(weights[b] for b in BLOCKS)
    return sum(weights[b] * block_scores[b] for b in BLOCKS) / max(total_w, 1e-9)


def score_books(profile: UserProfile, book_emb: np.ndarray, books: pd.DataFrame,
                label_mats: dict[str, tuple[list[str], np.ndarray]],
                calibration: dict[str, tuple[float, float]],
                alpha: float = config.ALPHA_DESC, beta: float = config.BETA_COVERAGE,
                mu: float = config.MU_AVOID, tau: float = config.TAU_AVOID,
                use_calibration: bool = True) -> pd.DataFrame:
    """Apply the formula to a set of books (all computations vectorised).

    Args:
        profile: user profile (build_profile).
        book_emb: (n, d) embeddings of the candidate books.
        books: (n rows) metadata of the same books, same order.
        label_mats: {block: (labels, matrix)} = engine.label_matrix(block).
        calibration: {component: (p5, p95)}.
        alpha, beta, mu, tau: formula constants (config.py); adjustable for the ablation.
        use_calibration: False = raw cosines ("without calibration" ablation).

    Returns:
        DataFrame (one row per book, same order as ``books``) with all the
        raw and calibrated components, the coverage, the penalty, the final
        score and the contribution of each term.
    """
    lam = float(profile.lambda_pop)
    cal = (lambda s, comp: calibrate(s, *calibration[comp])) if use_calibration \
        else (lambda s, comp: np.clip(s, 0.0, 1.0))
    out = pd.DataFrame(index=books.index)

    # (1) global query ↔ blurb similarity
    raw_desc = desc_similarity(profile.query_vecs["global"], book_emb)
    out["S_desc_raw"] = raw_desc
    out["S_desc"] = cal(raw_desc, "S_desc")

    # (2) scores per block, then (3) calibration
    calibrated = {}
    for b in BLOCKS:
        labels, L = label_mats.get(b, ([], None))
        mask = book_label_mask(books, b, labels)
        raw_b, _ = block_similarity(profile.query_vecs.get(b), L, mask, book_emb)
        out[f"{b}_raw"] = raw_b
        calibrated[b] = cal(raw_b, b)
        out[b] = calibrated[b]

    # (4) coverage weighted by the importance Likerts
    out["coverage"] = coverage_score(calibrated, profile.weights)

    # (5) popularity and "to avoid" penalty
    pop = books["popularity"].fillna(0).to_numpy(dtype=np.float32) if "popularity" in books \
        else np.zeros(len(books), np.float32)
    out["popularity"] = pop
    avoid = avoid_similarity(profile.query_vecs.get("avoid"), book_emb)
    out["avoid_sim"] = avoid
    penalty = mu * np.maximum(0.0, avoid - tau)
    out["penalty"] = penalty

    # (6) final score, normalised by (α + β + λ) then bounded to [0, 1]
    denom = alpha + beta + lam
    out["contrib_Description"] = alpha * out["S_desc"] / denom
    total_w = sum(profile.weights[b] for b in BLOCKS)
    for b in BLOCKS:
        out[f"contrib_{b}"] = beta * (profile.weights[b] / total_w) * out[b] / denom
    out["contrib_Popularity"] = lam * pop / denom
    out["contrib_Avoid penalty"] = -penalty
    contrib_cols = [c for c in out.columns if c.startswith("contrib_")]
    out["final_unclipped"] = out[contrib_cols].sum(axis=1)
    out["final"] = out["final_unclipped"].clip(0.0, 1.0)
    out["match_label"] = out["final"].map(match_label)
    return out


def matched_labels_for_book(profile: UserProfile, book: pd.Series,
                            label_mats: dict[str, tuple[list[str], np.ndarray]],
                            n: int = config.N_MATCHED_LABELS,
                            min_sim: float = config.MATCHED_LABEL_MIN_SIM) -> dict[str, list[tuple[str, float]]]:
    """For each block: the n labels OF THE BOOK closest to the user's preferences.

    sim(label) = max over the P_b(u) sentences of cos(sentence, label). Used for the app's
    "chips" and for the explanation ("It shares the themes you care about: …").
    Only the labels the user asked for (profile.requested_labels: ticked,
    inferred from the Likerts or top 3 affinities) can be cited: we never show
    "Fast-paced page-turner" to someone who asked for a slow pace.
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for b in BLOCKS:
        labels, L = label_mats.get(b, ([], None))
        col = BLOCK_COLUMNS[b]
        book_labels = [l for l in (book.get(col) if col in book.index else []) or [] if l in labels]
        allowed = profile.requested_labels.get(b) if profile.requested_labels else None
        if allowed is not None:
            book_labels = [l for l in book_labels if l in allowed]
        P = profile.query_vecs.get(b)
        if not book_labels or P is None or len(P) == 0:
            out[b] = []
            continue
        idx = [labels.index(l) for l in book_labels]
        sims = (P @ L[idx].T).max(axis=0)
        order = [i for i in np.argsort(-sims)[:n] if sims[i] >= min_sim]
        out[b] = [(book_labels[i], round(float(sims[i]), 3)) for i in order]
    return out
