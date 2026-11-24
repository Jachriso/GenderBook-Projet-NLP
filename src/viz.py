"""
viz.py : visualisations du projet (Personne 5).

Toutes les fonctions de figures sont PURES : elles reçoivent des données déjà calculées
(dicts, tableaux numpy) et renvoient une figure Plotly. Elles ne chargent ni modèle ni
catalogue, donc elles sont réutilisables telles quelles dans :
    - l'application Streamlit (st.plotly_chart(fig)) ;
    - les notebooks 04 / 05 / 06 ;
    - le rapport et les slides (export PNG avec save_figure, via kaleido).

Figures (PLAN §11.5) :
    radar_profile(block_affinities)          profil du lecteur : affinité pour chaque label, par bloc
    radar_compare(weights, books_scores)     « vous vs ce(s) livre(s) » sur les 4 blocs
    score_breakdown(recommendations)         décomposition du score final (barres empilées)
    similarity_map(coords, genres, ...)      carte 2D du catalogue (UMAP / PCA) + utilisateur + top 3
    similarity_heatmap(sim, rows, cols)      cosinus phrases utilisateur × labels/phrases du livre

Utilitaires de données (pour la carte) :
    compute_map_coords(matrix)               projection 2D (UMAP si installé, sinon PCA)
    load_or_compute_map_coords(...)          lit data/embeddings/umap_2d.npy (P3) ou le calcule + cache
    user_map_position(coords, sims)          position de l'utilisateur = barycentre de ses k plus proches livres

Choix de lisibilité : une couleur FIXE par bloc dans toutes les figures (Genre toujours
bleu, Thèmes toujours vert…), pour que le jury fasse le lien d'un graphique à l'autre.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config

# --------------------------------------------------------------------------- #
# 0. Charte graphique commune
# --------------------------------------------------------------------------- #

BLOCK_ORDER: tuple[str, ...] = tuple(config.BLOCKS)  # ("Genre", "Themes", "Style", "Atmosphere")

# Une couleur par composante du score, réutilisée PARTOUT (radars, barres, heatmap).
COMPONENT_COLORS: dict[str, str] = {
    "Description": "#5B6770",   # similarité requête ↔ résumé : gris ardoise (le « socle »)
    "Genre": "#2F6FDE",
    "Themes": "#1A9E77",
    "Style": "#E08A1E",
    "Atmosphere": "#9B4DCA",
    "Popularity": "#C9A227",
    "Avoid": "#D64545",         # pénalité « à éviter » (valeur négative)
}
USER_COLOR = "#111827"           # l'utilisateur est toujours en noir
BOOK_COLORS = ("#E4572E", "#17BEBB", "#FFC914")  # livres n°1, 2, 3 (contraste élevé)
MATCH_COLORS = {"Excellent": "#1A9E77", "Good": "#E08A1E", "Partial": "#8A8F98"}

FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13)

# Synonymes acceptés pour les clés de block_scores / contributions : le contrat (§4.4)
# écrit « S_desc, S_Genre… » mais P4 peut choisir « desc », « Genre », « pop »… On normalise
# tout vers les noms de COMPONENT_COLORS pour ne pas dépendre de la convention exacte.
_KEY_ALIASES = {
    "desc": "Description", "s_desc": "Description", "description": "Description",
    "synopsis": "Description", "global": "Description",
    "genre": "Genre", "themes": "Themes", "theme": "Themes", "style": "Style",
    "atmosphere": "Atmosphere", "coverage": "Coverage",
    "pop": "Popularity", "popularity": "Popularity",
    "avoid": "Avoid", "penalty": "Avoid", "avoid_penalty": "Avoid", "avoid penalty": "Avoid",
}


def normalize_key(key: str) -> str:
    """'S_Genre' -> 'Genre', 'S_desc' -> 'Description', 'pop' -> 'Popularity'…"""
    k = str(key).strip()
    low = k.lower()
    if low.startswith("s_"):
        low = low[2:]
    return _KEY_ALIASES.get(low, k[2:] if k.lower().startswith("s_") else k)


def normalize_scores(scores: Mapping[str, float] | None) -> dict[str, float]:
    """Applique normalize_key à toutes les clés (et convertit en float)."""
    return {normalize_key(k): float(v) for k, v in (scores or {}).items()}


def _base_layout(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    """Mise en forme commune : police, marges, fond transparent (s'adapte au thème Streamlit)."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)) if title else None,
        font=FONT,
        margin=dict(l=40, r=30, t=60 if title else 30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=height,
    )
    # automargin : Plotly agrandit la marge si les libellés d'axes sont longs (sinon ils sont coupés)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _short(text: str, n: int = 38) -> str:
    """Raccourcit un libellé trop long pour un axe (avec « … »)."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Lit un attribut de dataclass OU une clé de dict (les recos peuvent venir d'un JSON)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --------------------------------------------------------------------------- #
# 1. Radar « profil du lecteur »
# --------------------------------------------------------------------------- #

def radar_profile(
    block_affinities: Mapping[str, Mapping[str, float]],
    top_n: int = 6,
    title: str | None = "Your reader profile",
    height: int = 620,
) -> go.Figure:
    """Un radar par bloc (2 × 2) : affinité de l'utilisateur pour les labels les plus proches.

    Args:
        block_affinities: UserProfile.block_affinities = {bloc: {label: affinité}}.
            L'affinité vient d'un cosinus entre les réponses de l'utilisateur et la phrase
            descriptive du label (calculée par P4). Elle peut être brute (≈ 0,1-0,7) ou
            calibrée dans [0, 1] : on rééchelonne chaque bloc sur [0, 1] pour la lisibilité.
        top_n: nombre de labels montrés par bloc (les plus forts). 5 à 7 reste lisible.

    Pourquoi 4 petits radars plutôt qu'un grand : ~66 labels sur un seul radar seraient
    illisibles, et séparer les blocs montre directement la structure du référentiel de P2.
    """
    blocks = [b for b in BLOCK_ORDER if b in block_affinities] or list(block_affinities)
    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "polar"}] * cols for _ in range(rows)],
        subplot_titles=[b for b in blocks[: rows * cols]],
        horizontal_spacing=0.30, vertical_spacing=0.22,
    )
    for i, block in enumerate(blocks[: rows * cols]):
        aff = {k: float(v) for k, v in block_affinities[block].items()}
        if not aff:
            continue
        # Étape 1 : garder les top_n labels (ordre décroissant d'affinité)
        top = sorted(aff.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        labels = [_short(k, 18) for k, _ in top]
        vals = np.array([v for _, v in top], dtype=float)
        # Étape 2 : rééchelonner sur [0,1] DANS le bloc (min du bloc complet -> 0, max -> 1)
        all_vals = np.array(list(aff.values()), dtype=float)
        lo, hi = float(all_vals.min()), float(all_vals.max())
        scaled = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        # Étape 3 : fermer le polygone (répéter le premier point)
        color = COMPONENT_COLORS.get(block, "#444")
        fig.add_trace(
            go.Scatterpolar(
                r=list(scaled) + [scaled[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.25),
                line=dict(color=color, width=2),
                name=block,
                customdata=list(vals) + [vals[0]],
                hovertemplate="%{theta}<br>affinity %{customdata:.2f}<extra>" + block + "</extra>",
                showlegend=False,
            ),
            row=i // cols + 1, col=i % cols + 1,
        )
    fig.update_polars(
        radialaxis=dict(range=[0, 1], showticklabels=False, ticks="", gridcolor="#DDD"),
        angularaxis=dict(tickfont=dict(size=10), gridcolor="#DDD"),
        bgcolor="rgba(0,0,0,0)",
    )
    # Titres des sous-graphiques un peu remontés pour ne pas toucher les libellés du radar
    for ann in fig.layout.annotations:
        ann.update(yshift=18, font=dict(size=14))
    fig = _base_layout(fig, title, height)
    fig.update_layout(margin=dict(l=90, r=90, t=90 if title else 60, b=40))
    return fig


# --------------------------------------------------------------------------- #
# 2. Radar « vous vs ce(s) livre(s) »
# --------------------------------------------------------------------------- #

def radar_compare(
    weights: Mapping[str, float],
    books_block_scores: Mapping[str, Mapping[str, float]],
    title: str | None = "Your priorities vs the books",
    include_description: bool = True,
    max_weight: float = 5.0,
    height: int = 430,
) -> go.Figure:
    """Compare l'importance que l'utilisateur donne à chaque bloc et le score du livre sur ce bloc.

    Args:
        weights: UserProfile.weights = {bloc: W_b} (Likert 1-5, ou déjà normalisé).
        books_block_scores: {titre du livre: Recommendation.block_scores}. Un seul livre
            -> radar individuel ; trois livres -> comparaison des recommandations.
        include_description: ajoute l'axe « Description » (S̃_desc). Côté utilisateur, il
            vaut 1 : la similarité au résumé compte toujours (poids fixe 0,5 de la formule).
        max_weight: W_b maximal possible (5 pour un Likert) pour ramener les poids sur [0, 1].

    Lecture : là où le polygone du livre « couvre » celui de l'utilisateur, le livre satisfait
    les critères jugés importants ; un creux sur un axe important explique un score plus faible.
    """
    axes = (["Description"] if include_description else []) + list(BLOCK_ORDER)
    w = normalize_scores(weights)
    wmax = max(max_weight, max(w.values(), default=1.0))
    user_r = [1.0 if a == "Description" else w.get(a, 0.0) / wmax for a in axes]

    fig = go.Figure()
    # Polygone de l'utilisateur (pointillés noirs) : ce qu'il demande
    fig.add_trace(go.Scatterpolar(
        r=user_r + [user_r[0]], theta=axes + [axes[0]],
        name="Your priorities", line=dict(color=USER_COLOR, width=2, dash="dash"),
        fill="toself", fillcolor="rgba(17,24,39,0.06)",
        hovertemplate="%{theta}: importance %{r:.2f}<extra>You</extra>",
    ))
    # Un polygone par livre : ce que le livre offre (scores calibrés dans [0, 1])
    for i, (book, scores) in enumerate(books_block_scores.items()):
        s = normalize_scores(scores)
        r = [float(np.clip(s.get(a, 0.0), 0, 1)) for a in axes]
        color = BOOK_COLORS[i % len(BOOK_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]], theta=axes + [axes[0]], name=_short(book, 40),
            line=dict(color=color, width=2.5), fill="toself", fillcolor=_hex_to_rgba(color, 0.15),
            hovertemplate="%{theta}: score %{r:.2f}<extra>" + _short(book, 30) + "</extra>",
        ))
    fig.update_polars(
        radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1], tickfont=dict(size=9),
                        gridcolor="#DDD", angle=90),
        angularaxis=dict(gridcolor="#DDD"), bgcolor="rgba(0,0,0,0)",
    )
    return _base_layout(fig, title, height)


# --------------------------------------------------------------------------- #
# 3. Décomposition du score (barres horizontales empilées)
# --------------------------------------------------------------------------- #

def score_breakdown(
    recommendations: Sequence[Any],
    title: str | None = "Where does each score come from?",
    height: int | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Barres empilées : chaque segment = contribution d'une composante au score final.

    Args:
        recommendations: liste de Recommendation (ou de dicts) ; on lit `title`,
            `contributions` (poids × score de chaque composante, calculé par P4) et
            `final_score`. Une valeur négative (pénalité « à éviter ») part vers la gauche.

    La somme des segments = le score final (à la pénalité près) : c'est la preuve visuelle
    que la recommandation est TRAÇABLE, exigence « Demonstrate how results are derived ».
    """
    recs = list(recommendations)
    # Ordre d'affichage : n°1 en haut -> on inverse pour l'axe y de Plotly
    names = [f"#{_get(r, 'rank', i + 1)} {_short(_get(r, 'title', '?'), 34)}" for i, r in enumerate(recs)]
    contribs = [normalize_scores(_get(r, "contributions", {})) for r in recs]
    # Composantes présentes, dans un ordre stable (connues d'abord, puis les autres)
    known = [k for k in COMPONENT_COLORS if any(k in c for c in contribs)]
    others = sorted({k for c in contribs for k in c} - set(known))
    components = known + others

    fig = go.Figure()
    for comp in components:
        vals = [c.get(comp, 0.0) for c in contribs]
        fig.add_trace(go.Bar(
            y=names[::-1], x=vals[::-1], name=comp, orientation="h",
            marker=dict(color=COMPONENT_COLORS.get(comp, "#999")),
            hovertemplate=comp + ": %{x:+.3f}<extra></extra>",
            text=[f"{v:.2f}" if abs(v) >= 0.06 else "" for v in vals[::-1]],
            textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="white", size=11),
        ))
    # Marqueur du score final (losange noir) pour vérifier que la somme tombe juste
    finals = [float(_get(r, "final_score", np.nan)) for r in recs]
    fig.add_trace(go.Scatter(
        y=names[::-1], x=finals[::-1], mode="markers+text", name="Final score",
        marker=dict(symbol="diamond", size=12, color=USER_COLOR),
        text=[f"{f:.0%}" for f in finals[::-1]], textposition="middle right",
        hovertemplate="Final score %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_xaxes(title="contribution to the final score", zeroline=True, zerolinecolor="#888",
                     gridcolor="#EEE", range=[min(0, min((sum(min(v, 0) for v in c.values())
                                                          for c in contribs), default=0)) - 0.02,
                                              max(1.0, max(finals, default=1.0)) + 0.18])
    fig = _base_layout(fig, title, height or 160 + 70 * len(recs))
    # Légende AU-DESSUS du graphique : en dessous, elle chevauche les barres quand la figure est basse
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                  font=dict(size=11)),
                      margin=dict(t=95 if title and show_legend else 60), showlegend=show_legend)
    return fig


# --------------------------------------------------------------------------- #
# 4. Carte de similarité 2D du catalogue
# --------------------------------------------------------------------------- #

def similarity_map(
    coords: np.ndarray,
    genres: Sequence[str],
    titles: Sequence[str],
    highlight: Sequence[int] = (),
    highlight_labels: Sequence[str] | None = None,
    user_xy: tuple[float, float] | None = None,
    neighbours: Sequence[int] = (),
    max_points: int = 6000,
    top_genres: int = 10,
    title: str | None = "Semantic map of the catalogue",
    height: int = 600,
    seed: int = config.RANDOM_SEED,
) -> go.Figure:
    """Nuage 2D des livres (projection des embeddings) coloré par genre + l'utilisateur + le top 3.

    Args:
        coords: (N, 2) coordonnées 2D de TOUS les livres (même ordre que le catalogue).
        genres: genre principal de chaque livre (N,) ; les genres rares sont regroupés en « Other ».
        titles: titres (N,) pour le survol.
        highlight: indices des livres recommandés (étoiles numérotées).
        user_xy: position de l'utilisateur (voir user_map_position).
        neighbours: indices des plus proches voisins de la requête (cerclés).
        max_points: sous-échantillonnage pour garder l'app fluide (12 000 points restent ok,
            mais 6 000 suffisent à voir la structure ; les livres mis en avant sont toujours tracés).

    Interprétation : deux livres proches sur la carte ont des résumés de sens proche
    (pas forcément les mêmes mots). Les couleurs montrent que les genres forment des
    régions cohérentes, et l'étoile de l'utilisateur tombe dans la région qui lui correspond.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    genres = np.asarray([g if isinstance(g, str) and g else "Unknown" for g in genres], dtype=object)
    titles = np.asarray(titles, dtype=object)

    # Étape 1 : sous-échantillonnage reproductible (on garde toujours les points importants)
    rng = np.random.default_rng(seed)
    keep = np.arange(n) if n <= max_points else np.sort(rng.choice(n, max_points, replace=False))

    # Étape 2 : regrouper les genres rares pour une légende lisible (≤ top_genres + Other)
    uniq, counts = np.unique(genres[keep], return_counts=True)
    main = set(uniq[np.argsort(-counts)][:top_genres])
    shown_genre = np.where(np.isin(genres, list(main)), genres, "Other")

    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948",
               "#B07AA1", "#FF9DA7", "#9C755F", "#86BCB6", "#D37295", "#A0CBE8"]
    fig = go.Figure()
    order = sorted(main, key=lambda g: -int(counts[list(uniq).index(g)])) + ["Other"]
    for i, g in enumerate(order):
        idx = keep[shown_genre[keep] == g]
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=g,
            marker=dict(size=4, opacity=0.45 if g != "Other" else 0.2,
                        color=palette[i % len(palette)] if g != "Other" else "#BBBBBB"),
            text=titles[idx], hovertemplate="%{text}<extra>" + g + "</extra>",
        ))

    # Étape 3 : voisins de la requête (cercles), puis top 3 (étoiles), puis utilisateur
    nb = [i for i in neighbours if 0 <= i < n and i not in set(highlight)]
    if nb:
        fig.add_trace(go.Scatter(
            x=coords[nb, 0], y=coords[nb, 1], mode="markers", name="Closest books",
            marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=USER_COLOR, width=1.2)),
            text=titles[nb], hovertemplate="%{text}<extra>neighbour</extra>",
        ))
    for k, i in enumerate(highlight):
        if not 0 <= i < n:
            continue
        label = highlight_labels[k] if highlight_labels and k < len(highlight_labels) else titles[i]
        fig.add_trace(go.Scatter(
            x=[coords[i, 0]], y=[coords[i, 1]], mode="markers+text", name=f"#{k + 1} {_short(label, 30)}",
            marker=dict(symbol="star", size=20, color=BOOK_COLORS[k % len(BOOK_COLORS)],
                        line=dict(color="white", width=1.5)),
            text=[f"#{k + 1}"], textposition="top center", textfont=dict(size=13, color=USER_COLOR),
            hovertext=[str(label)], hovertemplate="%{hovertext}<extra>recommended</extra>",
        ))
    if user_xy is not None:
        fig.add_trace(go.Scatter(
            x=[user_xy[0]], y=[user_xy[1]], mode="markers+text", name="You",
            marker=dict(symbol="x", size=16, color=USER_COLOR, line=dict(width=2, color="white")),
            text=["YOU"], textposition="bottom center", textfont=dict(size=13, color=USER_COLOR),
            hovertemplate="Your position (weighted barycentre of your closest books)<extra></extra>",
        ))
    # Cadrage robuste : quelques livres isolés (descriptions atypiques) écraseraient le nuage
    # principal ; on cadre sur les percentiles 1-99 (+ marge), en gardant toujours le top 3 et l'utilisateur.
    pts = [coords[highlight[i]] for i in range(len(highlight)) if 0 <= highlight[i] < n]
    if user_xy is not None:
        pts.append(np.asarray(user_xy))
    ranges = []
    for d in (0, 1):
        lo, hi = np.percentile(coords[:, d], [1, 99])
        if pts:
            lo, hi = min(lo, min(p[d] for p in pts)), max(hi, max(p[d] for p in pts))
        pad = 0.06 * (hi - lo)
        ranges.append([lo - pad, hi + pad])
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, title=None, range=ranges[0])
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, title=None, range=ranges[1])
    fig = _base_layout(fig, title, height)
    fig.update_layout(legend=dict(orientation="v", y=1, x=1.02, yanchor="top", xanchor="left",
                                  font=dict(size=11), itemsizing="constant"), margin=dict(r=180))
    return fig


# --------------------------------------------------------------------------- #
# 5. Heatmap de similarité : phrases utilisateur × phrases/labels du livre
# --------------------------------------------------------------------------- #

def similarity_heatmap(
    sim: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str | None = "Semantic similarity: your words × the book",
    height: int | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Matrice de cosinus (lignes = textes de l'utilisateur, colonnes = labels ou phrases du livre).

    C'est la visualisation la plus directe de l'analyse SÉMANTIQUE : une case foncée relie
    une demande de l'utilisateur à un élément du livre qui exprime la même idée, même sans
    mot commun (ex. « old university » ↔ label « Academic & campus »).
    """
    sim = np.asarray(sim, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=sim,
        x=[_short(c, 30) for c in col_labels],
        y=[_short(r, 45) for r in row_labels],
        customdata=np.array([[f"{r}  ↔  {c}" for c in col_labels] for r in row_labels], dtype=object),
        colorscale="Blues", zmin=zmin, zmax=zmax,
        text=np.round(sim, 2), texttemplate="%{text:.2f}", textfont=dict(size=10),
        hovertemplate="%{customdata}<br>cosine %{z:.3f}<extra></extra>",
        colorbar=dict(title="cos", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return _base_layout(fig, title, height or 160 + 38 * len(row_labels))


# --------------------------------------------------------------------------- #
# 6. Petits graphiques d'appoint
# --------------------------------------------------------------------------- #

def weights_bar(weights: Mapping[str, float], title: str | None = "Importance you gave to each block",
                max_weight: float = 5.0, height: int = 220) -> go.Figure:
    """Barres horizontales des poids W_b choisis par l'utilisateur (Likert 1-5)."""
    w = normalize_scores(weights)
    blocks = [b for b in BLOCK_ORDER if b in w]
    fig = go.Figure(go.Bar(
        x=[w[b] for b in blocks], y=blocks, orientation="h",
        marker=dict(color=[COMPONENT_COLORS[b] for b in blocks]),
        text=[f"{w[b]:g}" for b in blocks], textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(max_weight, max(w.values(), default=0)) + 0.6], showgrid=False,
                     showticklabels=False)
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, title, height)


def score_gauge(score: float, label: str = "", height: int = 170) -> go.Figure:
    """Jauge compacte du score final (pour une carte de recommandation)."""
    color = MATCH_COLORS.get(label.split()[0] if label else "", "#2F6FDE")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(100 * float(score), 1),
        number=dict(suffix="%", font=dict(size=28)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[55, 75], ticktext=["55", "75"]),
                   bar=dict(color=color),
                   steps=[dict(range=[0, 55], color="#F1F1F1"), dict(range=[55, 75], color="#E6E6E6"),
                          dict(range=[75, 100], color="#D9D9D9")]),
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=0), height=height, font=FONT,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------- #
# 7. Données de la carte : projection 2D et position de l'utilisateur
# --------------------------------------------------------------------------- #

def compute_map_coords(matrix: np.ndarray, method: str = "auto", seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Projette les embeddings (N, d) en 2D.

    method = "umap" (préserve les voisinages : meilleure carte, ~30-60 s pour 12 000 livres),
    "pca" (instantané, linéaire : clusters moins nets) ou "auto" (UMAP si installé, sinon PCA).
    UMAP utilise la métrique cosinus, cohérente avec le reste du moteur.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if method in ("auto", "umap"):
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
            return reducer.fit_transform(matrix).astype(np.float32)
        except ImportError:
            if method == "umap":
                raise
    # PCA via SVD (numpy seul, aucune dépendance)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(np.float32)


def load_or_compute_map_coords(
    matrix: np.ndarray,
    book_ids: Sequence[str],
    path: Path | str = config.DATA_EMBEDDINGS / "umap_2d.npy",
    method: str = "auto",
) -> np.ndarray:
    """Coordonnées 2D alignées sur le catalogue : lues sur disque si valides, sinon calculées puis cachées.

    Le fichier umap_2d.npy est normalement produit par P3 (notebook 03). Pour garantir
    l'alignement ligne i ↔ livre i, on vérifie le fichier voisin umap_2d_book_ids.json
    (s'il existe) ou, à défaut, que le nombre de lignes correspond.
    """
    path = Path(path)
    ids_path = path.with_name(path.stem + "_book_ids.json")
    ids = [str(b) for b in book_ids]
    if path.exists():
        coords = np.load(path)
        aligned = coords.shape == (len(ids), 2)
        if aligned and ids_path.exists():
            aligned = json.loads(ids_path.read_text(encoding="utf-8")) == ids
        if aligned:
            return coords
    coords = compute_map_coords(matrix, method=method)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return coords


def user_map_position(coords: np.ndarray, sims: np.ndarray, k: int = 15) -> tuple[tuple[float, float], np.ndarray]:
    """Place l'utilisateur sur la carte = barycentre (pondéré par le cosinus) de ses k livres les plus proches.

    Pourquoi pas umap.transform : il faudrait garder l'objet UMAP entraîné (lourd, non
    sérialisé par P3) et le résultat serait instable. Le barycentre des plus proches voisins
    est simple, déterministe et fidèle à l'idée « vous êtes là où sont les livres qui vous ressemblent ».

    Returns:
        ((x, y), indices des k voisins)
    """
    sims = np.asarray(sims, dtype=float)
    k = int(min(k, len(sims)))
    nn = np.argsort(-sims)[:k]
    w = np.clip(sims[nn], 1e-6, None) ** 2      # on accentue les plus proches
    xy = (coords[nn] * w[:, None]).sum(axis=0) / w.sum()
    return (float(xy[0]), float(xy[1])), nn


def primary_genre(genres: Any) -> str:
    """Premier genre d'une liste (colonne `genres` de P2), 'Unknown' si vide."""
    if isinstance(genres, (list, tuple, np.ndarray)) and len(genres) > 0:
        return str(genres[0])
    if isinstance(genres, str) and genres:
        return genres
    return "Unknown"


# --------------------------------------------------------------------------- #
# 8. Export pour le rapport et les slides
# --------------------------------------------------------------------------- #

def save_figure(fig: go.Figure, name: str, folder: Path | str = config.FIGURES_DIR,
                width: int = 1100, height: int | None = None, scale: float = 2.0) -> Path | None:
    """Exporte une figure en PNG (docs/figures/05_<name>.png) via kaleido.

    Renvoie None (avec un message) si kaleido / Chrome n'est pas disponible, au lieu de planter
    le notebook : la figure reste visible en interactif.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / (name if name.endswith(".png") else f"{name}.png")
    try:
        fig.write_image(str(out), width=width, height=height or fig.layout.height or 600, scale=scale)
        return out
    except Exception as err:  # kaleido absent, Chrome introuvable (kaleido >= 1.0)…
        print(f"[save_figure] export PNG impossible ({type(err).__name__}: {err}). "
              "Installer kaleido (pip install kaleido) puis, si demandé, `plotly_get_chrome`.")
        return None


def export_all(figures: Mapping[str, go.Figure], **kwargs: Any) -> list[Path]:
    """Exporte plusieurs figures d'un coup : {"radar_profile": fig, ...} -> liste de chemins."""
    paths = [save_figure(f, n, **kwargs) for n, f in figures.items()]
    return [p for p in paths if p is not None]


__all__: Iterable[str] = [
    "COMPONENT_COLORS", "BOOK_COLORS", "normalize_key", "normalize_scores",
    "radar_profile", "radar_compare", "score_breakdown", "similarity_map", "similarity_heatmap",
    "weights_bar", "score_gauge", "compute_map_coords", "load_or_compute_map_coords",
    "user_map_position", "primary_genre", "save_figure", "export_all",
]
