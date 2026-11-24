"""
Structures de données partagées entre le moteur de recommandation (P4) et l'app (P5).

Ce fichier est le CONTRAT Python de la section 4.4 de PLAN_PROJET.md :
- ``UserProfile``     : ce que le système a compris du lecteur (sortie de profile.build_profile)
- ``Recommendation``  : un livre recommandé avec son score, ses critères et son explication

Les champs du contrat sont tous présents, dans l'ordre du plan. Quelques champs
SUPPLÉMENTAIRES ont été ajoutés à la fin, avec une valeur par défaut : ils ne
cassent rien côté app (on peut les ignorer) mais rendent le score totalement
traçable (« How was this score computed? »).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Les 4 blocs du référentiel (même ordre partout : radar, barres, rapport)
BLOCKS: tuple[str, ...] = ("Genre", "Themes", "Style", "Atmosphere")


@dataclass
class UserProfile:
    """Profil sémantique du lecteur, construit à partir de answers.json.

    Attributs du contrat :
        query_texts: textes encodés, clés "global", "Genre", "Themes", "Style",
            "Atmosphere", "avoid". Pour un bloc, les phrases de préférence sont
            jointes par " | " (la liste détaillée est dans ``block_phrases``).
        query_vecs: vecteurs L2-normalisés. "global" et "avoid" : forme (d,) ;
            un bloc : forme (m, d), une ligne par phrase de préférence.
        weights: importance W_b de chaque bloc (Likert 1-5).
        block_affinities: bloc -> label -> affinité de l'utilisateur ∈ [0, 1]
            (sert au radar « profil lecteur »).
        constraints: filtres durs (audience, formats, auteurs exclus...).
        summary: résumé en anglais du profil, généré par template.

    Champs ajoutés (optionnels) :
        block_phrases: bloc -> liste des phrases de préférence P_b(u).
        lambda_pop: poids λ du bonus popularité, déduit du Likert « popularity ».
        favorite_author: nom de l'auteur préféré retrouvé dans le catalogue (ou None).
        requested_labels: bloc -> labels que l'utilisateur a demandés (cochés, déduits
            des Likert ou top 3 des affinités) ; seuls ceux-là sont cités dans l'explication.
    """

    query_texts: dict[str, str]
    query_vecs: dict[str, np.ndarray]
    weights: dict[str, float]
    block_affinities: dict[str, dict[str, float]]
    constraints: dict[str, Any]
    summary: str
    # --- champs ajoutés par P4 (valeurs par défaut => rétro-compatible) ---
    block_phrases: dict[str, list[str]] = field(default_factory=dict)
    lambda_pop: float = 0.0
    favorite_author: str | None = None
    requested_labels: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Version sérialisable en JSON (les vecteurs sont omis, trop lourds)."""
        d = asdict(self)
        d["query_vecs"] = {k: list(np.shape(v)) for k, v in self.query_vecs.items()}
        return d


@dataclass
class Recommendation:
    """Un livre recommandé, avec tout ce qu'il faut pour l'afficher et l'expliquer.

    Attributs du contrat :
        rank: position dans le top (1 = meilleur).
        book_id, title, authors, image_url: métadonnées d'affichage.
        final_score: score final ∈ [0, 1].
        match_label: "Excellent" / "Good" / "Partial" (seuils de config.MATCH_THRESHOLDS).
        block_scores: scores CALIBRÉS ∈ [0, 1] de chaque composante :
            "S_desc", "Genre", "Themes", "Style", "Atmosphere".
        contributions: part de chaque composante dans le score final
            ("Description", "Genre", ..., "Popularity", "Avoid penalty" négatif).
            La somme des contributions = final_score (barres empilées).
        coverage: score de couverture Σ W_b·S̃_b / Σ W_b ∈ [0, 1].
        matched_labels: bloc -> [(label du livre, similarité avec la demande)].
        best_sentence / best_sentence_score: phrase du résumé la plus proche
            de la demande de l'utilisateur, et son cosinus.
        explanation: explication en anglais générée par template (explain.py).

    Champs ajoutés (optionnels) :
        raw_scores: cosinus BRUTS avant calibration (transparence).
        popularity: popularité du livre ∈ [0, 1] (colonne de P1).
        avoid_similarity: cos(q_avoid, livre), 0 si pas de texte « à éviter ».
        main_author: auteur principal (utile pour la règle de diversité).
    """

    rank: int
    book_id: str
    title: str
    authors: str
    image_url: str
    final_score: float
    match_label: str
    block_scores: dict[str, float]
    contributions: dict[str, float]
    coverage: float
    matched_labels: dict[str, list[tuple[str, float]]]
    best_sentence: str
    best_sentence_score: float
    explanation: str
    # --- champs ajoutés par P4 ---
    raw_scores: dict[str, float] = field(default_factory=dict)
    popularity: float = 0.0
    avoid_similarity: float = 0.0
    main_author: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Version sérialisable en JSON (pour sauvegarder les résultats)."""
        return asdict(self)
