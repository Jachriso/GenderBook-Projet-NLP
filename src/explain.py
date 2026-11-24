"""
Explications TRANSPARENTES des recommandations, générées par template (P4).

Pas de LLM : chaque phrase est construite à partir des résultats NLP réellement
calculés (similarités, labels communs, phrase la plus proche, couverture). Le
texte est donc fidèle au score par construction. Le bonus LLM de P6
(llm_explain.py) pourra reformuler, mais l'app fonctionne sans.

Structure d'une explication (en anglais, comme l'interface) :
1. accroche selon le niveau de score (Excellent / Good / Partial) ;
2. similarité du résumé + phrase du résumé la plus proche de la demande ;
3. labels communs par bloc (thèmes, genre, style, atmosphère) ;
4. critère le plus contributif et critère le plus faible ;
5. couverture globale, popularité, et avertissement si le livre touche un sujet à éviter.
"""
from __future__ import annotations

from src.schemas import BLOCKS, Recommendation, UserProfile

# Plusieurs accroches par niveau : la variante dépend du rang (déterministe,
# mais les trois livres du top ne commencent pas tous par la même phrase).
OPENERS = {
    "Excellent": [
        "This book is an excellent match for what you described.",
        "A very strong fit for your reading wishes.",
        "This one ticks almost all of your boxes.",
    ],
    "Good": [
        "This book is a good match for your preferences.",
        "A solid fit for what you are looking for.",
        "This one lines up well with several of your wishes.",
    ],
    "Partial": [
        "This book matches part of what you asked for.",
        "A partial fit: worth a look, with some differences.",
        "This one shares some aspects of your wishes.",
    ],
}

BLOCK_PHRASES = {
    "Themes": "It shares the themes you care about: {labels}.",
    "Genre": "Genre match: {labels}.",
    "Style": "Its style ({labels}) fits the kind of writing you asked for.",
    "Atmosphere": "Atmosphere match: {labels}.",
}

COMPONENT_NAMES = {
    "Description": "how close its synopsis is to your description",
    "Genre": "the genre", "Themes": "the themes", "Style": "the writing style",
    "Atmosphere": "the atmosphere", "Popularity": "its popularity with readers",
}


def _fmt_labels(pairs: list[tuple[str, float]]) -> str:
    return ", ".join(f"{lab} ({sim:.2f})" for lab, sim in pairs)


def _truncate(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def explain_recommendation(rec: Recommendation, profile: UserProfile | None = None) -> str:
    """Construit l'explication textuelle d'une recommandation (template, sans LLM)."""
    parts: list[str] = []

    # 1. Accroche selon le niveau de score
    variants = OPENERS.get(rec.match_label, OPENERS["Partial"])
    parts.append(variants[(rec.rank - 1) % len(variants)])

    # 2. Similarité du résumé + phrase la plus proche (preuve sémantique)
    raw_desc = rec.raw_scores.get("S_desc", rec.block_scores.get("S_desc", 0.0))
    s = f"Its synopsis is semantically close to your description (similarity {raw_desc:.2f})"
    if rec.best_sentence:
        s += f", especially: «{_truncate(rec.best_sentence)}»"
    parts.append(s + ".")

    # 3. Labels communs par bloc, dans l'ordre d'importance donné par l'utilisateur
    order = list(BLOCKS)
    if profile is not None:
        order = sorted(BLOCKS, key=lambda b: -profile.weights.get(b, 0))
    for b in order:
        pairs = rec.matched_labels.get(b) or []
        if pairs:
            parts.append(BLOCK_PHRASES[b].format(labels=_fmt_labels(pairs)))

    # 4. Critère le plus fort / le plus faible (hors pénalité)
    contrib = {k: v for k, v in rec.contributions.items() if k != "Avoid penalty" and v > 0}
    if contrib:
        best = max(contrib, key=contrib.get)
        parts.append(f"What weighs most in its score: {COMPONENT_NAMES.get(best, best.lower())}.")
    blocks_only = {b: rec.block_scores.get(b, 0.0) for b in BLOCKS}
    weakest = min(blocks_only, key=blocks_only.get)
    if blocks_only[weakest] < 0.4:
        parts.append(f"It is a weaker match on {COMPONENT_NAMES[weakest]} "
                     f"({blocks_only[weakest]:.0%}).")

    # 5. Couverture, popularité, sujet à éviter
    parts.append(f"Overall preference coverage: {rec.coverage:.0%}.")
    if profile is not None and profile.lambda_pop >= 0.1 and rec.popularity >= 0.6:
        parts.append("It is also a popular, well-rated book.")
    if rec.contributions.get("Avoid penalty", 0.0) < 0:
        parts.append("Note: it is somewhat close to what you wanted to avoid, "
                     "so its score was lowered.")
    return " ".join(parts)


def explain_score_formula(rec: Recommendation, profile: UserProfile) -> str:
    """Détail chiffré du calcul (expander « How was this score computed? » de l'app)."""
    w = profile.weights
    tw = sum(w[b] for b in BLOCKS)
    lines = [
        f"S_desc (calibrated) = {rec.block_scores['S_desc']:.2f}  (raw cosine {rec.raw_scores.get('S_desc', 0):.2f})",
    ]
    for b in BLOCKS:
        lines.append(f"S_{b} = {rec.block_scores[b]:.2f}  (raw {rec.raw_scores.get(b, 0):.2f}, weight W={w[b]:.0f})")
    lines.append(f"Coverage = Σ W·S / Σ W = " +
                 " + ".join(f"{w[b]:.0f}×{rec.block_scores[b]:.2f}" for b in BLOCKS) +
                 f" / {tw:.0f} = {rec.coverage:.2f}")
    lines.append(f"Popularity = {rec.popularity:.2f} (λ = {profile.lambda_pop:.2f})")
    if rec.avoid_similarity:
        lines.append(f"Avoid similarity = {rec.avoid_similarity:.2f} → penalty "
                     f"{-rec.contributions.get('Avoid penalty', 0):.2f}")
    lines.append(f"Final = {rec.final_score:.2f} → {rec.match_label} match")
    return "\n".join(lines)
