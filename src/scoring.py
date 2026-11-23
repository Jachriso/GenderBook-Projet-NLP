r"""
Formule de score sémantique (P4), calculée de façon VECTORISÉE (numpy, aucune
boucle Python sur les livres).

Notations : u = utilisateur, j = livre, e_j = embedding de embed_text, tous les
vecteurs sont L2-normalisés donc cos(a, b) = a · b.

1. Similarité globale       S_desc(j) = cos(q_global, e_j)
2. Score par bloc b         S_b(j) = (1/|P_b|) Σ_{p ∈ P_b(u)} max_{l ∈ L_b(j) ∪ {e_j}} cos(p, l)
                            (L_b(j) = labels attribués au livre par P2, encodés par leur phrase)
3. Calibration              s̃ = clip((s − p5) / (p95 − p5), 0, 1)
4. Couverture               Coverage(j) = Σ_b W_b · S̃_b(j) / Σ_b W_b
5. Score final              Final(j) = [α·S̃_desc + β·Coverage + λ·Pop] / (α + β + λ)
                                       − μ · max(0, cos(q_avoid, e_j) − τ)
                            puis clip dans [0, 1]

La division par (α + β + λ) garantit Final ∈ [0, 1] quel que soit λ : un livre
parfait vaut 1, et les seuils Excellent / Good / Partial gardent le même sens
pour un utilisateur qui veut des best-sellers (λ = 0.2) ou des pépites (λ = 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.schemas import BLOCKS, UserProfile

# Colonne de books_enriched (P2) qui contient les labels de chaque bloc
BLOCK_COLUMNS = {"Genre": "genres", "Themes": "themes", "Style": "styles", "Atmosphere": "atmospheres"}
COMPONENTS = ("S_desc", *BLOCKS)


# ---------------------------------------------------------------------------
# 1. Matrices de labels des livres
# ---------------------------------------------------------------------------
def book_label_mask(books: pd.DataFrame, block: str, labels: list[str]) -> np.ndarray:
    """Matrice booléenne (N, L) : mask[j, l] = True si le livre j porte le label l.

    Fallback : si la colonne de P2 n'existe pas (books_sample / books_clean),
    la matrice est vide et S_b se calcule sur la description seule.
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
# 2. Scores bruts (cosinus)
# ---------------------------------------------------------------------------
def desc_similarity(q: np.ndarray, book_emb: np.ndarray) -> np.ndarray:
    """S_desc = cos(q_global, e_j) pour tous les livres : un produit matrice-vecteur."""
    return book_emb @ np.asarray(q, dtype=np.float32)


def block_similarity(P: np.ndarray, label_emb: np.ndarray, mask: np.ndarray,
                     book_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S_b(j) pour tous les livres j (formule « max puis moyenne » du sujet).

    Args:
        P: (m, d) phrases de préférence de l'utilisateur pour ce bloc.
        label_emb: (L, d) phrases des labels du bloc.
        mask: (N, L) labels attribués à chaque livre.
        book_emb: (N, d) embeddings des livres.
    Returns:
        (S_b de forme (N,), best de forme (N, m) = max par phrase utilisateur).
    """
    N = book_emb.shape[0]
    if P is None or len(P) == 0:
        return np.zeros(N, dtype=np.float32), np.zeros((N, 0), dtype=np.float32)
    # Étape a : cos(phrase utilisateur, description du livre) -> (m, N)
    sim_desc = P @ book_emb.T
    if label_emb is not None and len(label_emb) and mask.any():
        # Étape b : cos(phrase utilisateur, phrase de chaque label) -> (m, L)
        sim_lab = P @ label_emb.T
        # Étape c : on ne garde que les labels PORTÉS par le livre (les autres à -inf)
        masked = np.where(mask[None, :, :], sim_lab[:, None, :], -np.inf)   # (m, N, L)
        best_lab = masked.max(axis=2)                                         # (m, N)
        # Étape d : max entre le meilleur label et la description elle-même
        best = np.maximum(best_lab, sim_desc)
    else:
        best = sim_desc  # pas de labels (P2 absent) : la description seule
    # Étape e : moyenne sur les phrases de l'utilisateur
    return best.mean(axis=0).astype(np.float32), best.T.astype(np.float32)


def avoid_similarity(q_avoid: np.ndarray | None, book_emb: np.ndarray) -> np.ndarray:
    """cos(q_avoid, e_j), ou 0 si l'utilisateur n'a rien à éviter."""
    if q_avoid is None:
        return np.zeros(book_emb.shape[0], dtype=np.float32)
    return book_emb @ np.asarray(q_avoid, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Calibration percentile
# ---------------------------------------------------------------------------
def calibrate(s: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """s̃ = clip((s − p5) / (p95 − p5), 0, 1) : ramène des cosinus compressés sur [0, 1]."""
    return np.clip((np.asarray(s, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def load_calibration(path: Path = config.CALIBRATION_JSON,
                     model: str | None = None) -> dict[str, tuple[float, float]] | None:
    """Lit un fichier de calibration : {"S_desc": {"p5": .., "p95": ..}, "Genre": {...}, ...}.

    Si ``model`` est donné et que le fichier indique un autre modèle, il est ignoré
    (des percentiles MiniLM n'ont pas de sens pour MPNet).
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
    for comp in COMPONENTS:   # un fichier partiel (ex. P3 : S_desc seul) -> pas de calibration des blocs
        out.setdefault(comp, (0.0, 1.0))
    return out


def load_personas(path: Path = config.PERSONAS_JSON) -> dict[str, dict]:
    """Personas de référence (format answers.json) ; repli sur l'exemple du contrat."""
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
    """Calcule p5 / p95 de chaque composante sur les CANDIDATS de personas de référence.

    Pourquoi sur les candidats et pas sur tout le catalogue ? Le retrieval garde
    les 300 livres les plus proches sur 12 000 (top 2,5 %) : ils sont TOUS au-dessus
    du p95 « catalogue », donc une calibration sur le catalogue entier donnerait
    S̃_desc = 1 à tous les candidats et ce terme ne départagerait plus rien.
    On mesure donc la distribution des scores là où ils servent : parmi les livres
    retrouvés pour des profils réalistes (data/reference/personas_p4.json).
    Le résultat est fixe (pas recalculé à chaque requête), donc un score reste
    comparable d'un utilisateur à l'autre.

    Si P3 fournit son propre calibration.json, ``load_calibration`` le lit en priorité.
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
        cand = np.argsort(-sims)[:retrieval_k]          # mêmes candidats que le recommender
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
# 4. Formule complète
# ---------------------------------------------------------------------------
def match_label(score: float) -> str:
    """Libellé lisible du score final (seuils du sujet, config.MATCH_THRESHOLDS)."""
    if score >= config.MATCH_THRESHOLDS["Excellent"]:
        return "Excellent"
    if score >= config.MATCH_THRESHOLDS["Good"]:
        return "Good"
    return "Partial"


def coverage_score(block_scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Coverage = Σ W_b · S̃_b / Σ W_b (formule exigée par le sujet)."""
    total_w = sum(weights[b] for b in BLOCKS)
    return sum(weights[b] * block_scores[b] for b in BLOCKS) / max(total_w, 1e-9)


def score_books(profile: UserProfile, book_emb: np.ndarray, books: pd.DataFrame,
                label_mats: dict[str, tuple[list[str], np.ndarray]],
                calibration: dict[str, tuple[float, float]],
                alpha: float = config.ALPHA_DESC, beta: float = config.BETA_COVERAGE,
                mu: float = config.MU_AVOID, tau: float = config.TAU_AVOID,
                use_calibration: bool = True) -> pd.DataFrame:
    """Applique la formule à un ensemble de livres (tous calculs vectorisés).

    Args:
        profile: profil utilisateur (build_profile).
        book_emb: (n, d) embeddings des livres candidats.
        books: (n lignes) métadonnées des mêmes livres, même ordre.
        label_mats: {bloc: (labels, matrice)} = engine.label_matrix(bloc).
        calibration: {composante: (p5, p95)}.
        alpha, beta, mu, tau: constantes de la formule (config.py) ; paramétrables pour l'ablation.
        use_calibration: False = cosinus bruts (ablation « sans calibration »).

    Returns:
        DataFrame (une ligne par livre, même ordre que ``books``) avec toutes les
        composantes brutes et calibrées, la couverture, la pénalité, le score
        final et les contributions de chaque terme.
    """
    lam = float(profile.lambda_pop)
    cal = (lambda s, comp: calibrate(s, *calibration[comp])) if use_calibration \
        else (lambda s, comp: np.clip(s, 0.0, 1.0))
    out = pd.DataFrame(index=books.index)

    # (1) similarité globale requête ↔ résumé
    raw_desc = desc_similarity(profile.query_vecs["global"], book_emb)
    out["S_desc_raw"] = raw_desc
    out["S_desc"] = cal(raw_desc, "S_desc")

    # (2) scores par bloc, puis (3) calibration
    calibrated = {}
    for b in BLOCKS:
        labels, L = label_mats.get(b, ([], None))
        mask = book_label_mask(books, b, labels)
        raw_b, _ = block_similarity(profile.query_vecs.get(b), L, mask, book_emb)
        out[f"{b}_raw"] = raw_b
        calibrated[b] = cal(raw_b, b)
        out[b] = calibrated[b]

    # (4) couverture pondérée par les Likert d'importance
    out["coverage"] = coverage_score(calibrated, profile.weights)

    # (5) popularité et pénalité « à éviter »
    pop = books["popularity"].fillna(0).to_numpy(dtype=np.float32) if "popularity" in books \
        else np.zeros(len(books), np.float32)
    out["popularity"] = pop
    avoid = avoid_similarity(profile.query_vecs.get("avoid"), book_emb)
    out["avoid_sim"] = avoid
    penalty = mu * np.maximum(0.0, avoid - tau)
    out["penalty"] = penalty

    # (6) score final, normalisé par (α + β + λ) puis borné dans [0, 1]
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
    """Pour chaque bloc : les n labels DU LIVRE les plus proches des préférences de l'utilisateur.

    sim(label) = max sur les phrases P_b(u) de cos(phrase, label). Sert aux « chips »
    de l'app et à l'explication (« It shares the themes you care about: … »).
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for b in BLOCKS:
        labels, L = label_mats.get(b, ([], None))
        col = BLOCK_COLUMNS[b]
        book_labels = [l for l in (book.get(col) if col in book.index else []) or [] if l in labels]
        P = profile.query_vecs.get(b)
        if not book_labels or P is None or len(P) == 0:
            out[b] = []
            continue
        idx = [labels.index(l) for l in book_labels]
        sims = (P @ L[idx].T).max(axis=0)
        order = [i for i in np.argsort(-sims)[:n] if sims[i] >= min_sim]
        out[b] = [(book_labels[i], round(float(sims[i]), 3)) for i in order]
    return out
