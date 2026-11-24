"""
viz.py: project visualisations (Person 5).

All figure functions are PURE: they take data that has already been computed
(dicts, numpy arrays) and return a Plotly figure. They load neither a model nor the
catalogue, so they can be reused as they are in:
    - the Streamlit app (st.plotly_chart(fig));
    - notebooks 04 / 05 / 06;
    - the report and the slides (PNG export with save_figure, through kaleido).

Figures (PLAN §11.5):
    radar_profile(block_affinities)          reader profile: affinity for each label, per block
    radar_compare(weights, books_scores)     "you vs this/these book(s)" on the 4 blocks
    score_breakdown(recommendations)         breakdown of the final score (stacked bars)
    similarity_map(coords, genres, ...)      2D map of the catalogue (UMAP / PCA) + user + top 3
    similarity_heatmap(sim, rows, cols)      cosine between user sentences × the book's labels/sentences

Data helpers (for the map):
    compute_map_coords(matrix)               2D projection (UMAP if installed, otherwise PCA)
    load_or_compute_map_coords(...)          reads data/embeddings/umap_2d.npy (P3) or computes it + caches it
    user_map_position(coords, sims)          user position = barycentre of their k closest books

Readability choice: ONE FIXED colour per block in every figure (Genre is always
blue, Themes always green…), so that the reader can link one chart to the next.
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
# 0. Shared visual identity
# --------------------------------------------------------------------------- #

BLOCK_ORDER: tuple[str, ...] = tuple(config.BLOCKS)  # ("Genre", "Themes", "Style", "Atmosphere")

# One colour per score component, reused EVERYWHERE (radars, bars, heatmap).
# "Reading room" palette, shared with the app's stylesheet (app/streamlit_app.py):
# banker's-lamp green, brass, plum, dusk blue, oxblood, on paper.
COMPONENT_COLORS: dict[str, str] = {
    "Description": "#7A7064",   # query ↔ synopsis similarity: warm grey (the "base")
    "Genre": "#2F5D4A",         # reading-lamp green
    "Themes": "#B8892E",        # brass
    "Style": "#7A4E7E",         # plum
    "Atmosphere": "#3F6E8C",    # dusk blue
    "Popularity": "#C9A66B",    # light tan
    "Avoid": "#9E3B32",         # "avoid" penalty (negative value): oxblood
}
USER_COLOR = "#2B2620"           # the user is always drawn in ink colour
BOOK_COLORS = ("#9E3B32", "#3F6E8C", "#B8892E")  # books #1, 2, 3 (distinct, warm)
MATCH_COLORS = {"Excellent": "#2F5D4A", "Good": "#B8892E", "Partial": "#8A8178"}
GRID_COLOR = "#E6DDCB"           # warm light grid lines

FONT = dict(family="Nunito, 'Segoe UI', Helvetica, Arial, sans-serif", size=13, color="#2B2620")
TITLE_FONT = dict(family="Literata, Georgia, serif", size=17, color="#23493A")

# Accepted synonyms for the keys of block_scores / contributions: the contract (§4.4)
# says "S_desc, S_Genre…" but P4 may use "desc", "Genre", "pop"… Everything is mapped
# to the names of COMPONENT_COLORS so that we do not depend on the exact convention.
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
    """Apply normalize_key to every key (and convert the values to float)."""
    return {normalize_key(k): float(v) for k, v in (scores or {}).items()}


def _base_layout(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    """Shared layout: font, margins, transparent background (adapts to the Streamlit theme)."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=TITLE_FONT) if title else None,
        font=FONT,
        margin=dict(l=40, r=30, t=60 if title else 30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=height,
    )
    # automargin: Plotly widens the margin when axis labels are long (otherwise they are cut off)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _short(text: str, n: int = 38) -> str:
    """Shorten a label that is too long for an axis (adds "…")."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a dataclass attribute OR a dict key (recommendations may come from a JSON file)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --------------------------------------------------------------------------- #
# 1. "Reader profile" radar
# --------------------------------------------------------------------------- #

def radar_profile(
    block_affinities: Mapping[str, Mapping[str, float]],
    top_n: int = 6,
    title: str | None = "Your reader profile",
    height: int = 620,
) -> go.Figure:
    """One radar per block (2 × 2): the user's affinity for the closest labels.

    Args:
        block_affinities: UserProfile.block_affinities = {block: {label: affinity}}.
            The affinity is a cosine between the user's answers and the label's descriptive
            sentence (computed by P4). It can be raw (≈ 0.1-0.7) or calibrated to [0, 1]:
            each block is rescaled to [0, 1] for readability.
        top_n: number of labels shown per block (the strongest ones). 5 to 7 stays readable.

    Why 4 small radars rather than one big one: ~66 labels on a single radar would be
    unreadable, and splitting the blocks directly shows the structure of P2's label set.
    """
    blocks = [b for b in BLOCK_ORDER if b in block_affinities] or list(block_affinities)
    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "polar"}] * cols for _ in range(rows)],
        subplot_titles=[b for b in blocks[: rows * cols]],
        horizontal_spacing=0.36, vertical_spacing=0.24,
    )
    for i, block in enumerate(blocks[: rows * cols]):
        aff = {k: float(v) for k, v in block_affinities[block].items()}
        if not aff:
            continue
        # Step 1: keep the top_n labels (decreasing affinity)
        top = sorted(aff.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        labels = [_short(k, 16) for k, _ in top]
        vals = np.array([v for _, v in top], dtype=float)
        # Step 2: rescale to [0, 1] WITHIN the block (min of the whole block -> 0, max -> 1)
        all_vals = np.array(list(aff.values()), dtype=float)
        lo, hi = float(all_vals.min()), float(all_vals.max())
        scaled = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        # Step 3: close the polygon (repeat the first point)
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
        radialaxis=dict(range=[0, 1], showticklabels=False, ticks="", gridcolor=GRID_COLOR),
        angularaxis=dict(tickfont=dict(size=10), gridcolor=GRID_COLOR),
        bgcolor="rgba(0,0,0,0)",
    )
    # Subplot titles moved up a little so they do not touch the radar labels
    for ann in fig.layout.annotations:
        ann.update(yshift=18, font=dict(family=TITLE_FONT["family"], size=14, color=COMPONENT_COLORS.get(ann.text, "#2B2620")))
    fig = _base_layout(fig, title, height)
    fig.update_layout(margin=dict(l=110, r=110, t=90 if title else 60, b=40))
    return fig


# --------------------------------------------------------------------------- #
# 2. "You vs this/these book(s)" radar
# --------------------------------------------------------------------------- #

def radar_compare(
    weights: Mapping[str, float],
    books_block_scores: Mapping[str, Mapping[str, float]],
    title: str | None = "Your priorities vs the books",
    include_description: bool = True,
    max_weight: float = 5.0,
    height: int = 430,
) -> go.Figure:
    """Compare how important each block is to the user with the book's score on that block.

    Args:
        weights: UserProfile.weights = {block: W_b} (Likert 1-5, or already normalised).
        books_block_scores: {book title: Recommendation.block_scores}. One book
            -> individual radar; three books -> comparison of the recommendations.
        include_description: adds the "Description" axis (S̃_desc). On the user side it is
            1: the synopsis similarity always counts (fixed weight 0.5 in the formula).
        max_weight: largest possible W_b (5 for a Likert scale), used to map weights to [0, 1].

    How to read it: where the book's polygon "covers" the user's, the book meets the criteria
    the user cares about; a dip on an important axis explains a lower score.
    """
    axes = (["Description"] if include_description else []) + list(BLOCK_ORDER)
    w = normalize_scores(weights)
    wmax = max(max_weight, max(w.values(), default=1.0))
    user_r = [1.0 if a == "Description" else w.get(a, 0.0) / wmax for a in axes]

    fig = go.Figure()
    # User polygon (black dashes): what the user asks for
    fig.add_trace(go.Scatterpolar(
        r=user_r + [user_r[0]], theta=axes + [axes[0]],
        name="Your priorities", line=dict(color=USER_COLOR, width=2, dash="dash"),
        fill="toself", fillcolor="rgba(43,38,32,0.06)",
        hovertemplate="%{theta}: importance %{r:.2f}<extra>You</extra>",
    ))
    # One polygon per book: what the book offers (scores calibrated to [0, 1])
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
                        gridcolor=GRID_COLOR, angle=90),
        angularaxis=dict(gridcolor=GRID_COLOR), bgcolor="rgba(0,0,0,0)",
    )
    return _base_layout(fig, title, height)


# --------------------------------------------------------------------------- #
# 3. Score breakdown (horizontal stacked bars)
# --------------------------------------------------------------------------- #

def score_breakdown(
    recommendations: Sequence[Any],
    title: str | None = "Where does each score come from?",
    height: int | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Stacked bars: each segment = the contribution of one component to the final score.

    Args:
        recommendations: list of Recommendation objects (or dicts); reads `title`,
            `contributions` (weight × score of each component, computed by P4) and
            `final_score`. A negative value ("avoid" penalty) goes to the left.
        show_legend: hide the legend for small single-book charts.

    The segments add up to the final score: this is the visual proof that the
    recommendation is TRACEABLE (requirement "Demonstrate how results are derived").
    """
    recs = list(recommendations)
    # Display order: #1 at the top -> reversed for Plotly's y axis
    names = [f"#{_get(r, 'rank', i + 1)} {_short(_get(r, 'title', '?'), 34)}" for i, r in enumerate(recs)]
    contribs = [normalize_scores(_get(r, "contributions", {})) for r in recs]
    # Components present, in a stable order (known ones first, then the others)
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
    # Final score marker (black diamond), to check that the sum is right
    finals = [float(_get(r, "final_score", np.nan)) for r in recs]
    fig.add_trace(go.Scatter(
        y=names[::-1], x=finals[::-1], mode="markers+text", name="Final score",
        marker=dict(symbol="diamond", size=12, color=USER_COLOR),
        text=[f"{f:.0%}" for f in finals[::-1]], textposition="middle right",
        hovertemplate="Final score %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_xaxes(title="contribution to the final score", zeroline=True, zerolinecolor="#A89F8F",
                     gridcolor=GRID_COLOR, range=[min(0, min((sum(min(v, 0) for v in c.values())
                                                          for c in contribs), default=0)) - 0.02,
                                              max(1.0, max(finals, default=1.0)) + 0.18])
    fig = _base_layout(fig, title, height or 160 + 70 * len(recs))
    # Legend ABOVE the chart: below it, it overlaps the bars when the figure is short
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                  font=dict(size=11)),
                      margin=dict(t=95 if title and show_legend else 60), showlegend=show_legend)
    return fig


# --------------------------------------------------------------------------- #
# 4. 2D similarity map of the catalogue
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
    """2D scatter of the books (projected embeddings) coloured by genre + the user + the top 3.

    Args:
        coords: (N, 2) 2D coordinates of ALL the books (same order as the catalogue).
        genres: main genre of each book (N,); rare genres are grouped into "Other".
        titles: titles (N,) shown on hover.
        highlight: indices of the recommended books (numbered stars).
        user_xy: position of the user (see user_map_position).
        neighbours: indices of the query's nearest neighbours (circled).
        max_points: subsampling to keep the app responsive (12,000 points still work,
            but 6,000 are enough to see the structure; highlighted books are always drawn).

    Interpretation: two books that are close on the map have synopses with a similar
    meaning (not necessarily the same words). The colours show that genres form coherent
    regions, and the user's marker falls in the region that matches their request.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    genres = np.asarray([g if isinstance(g, str) and g else "Unknown" for g in genres], dtype=object)
    titles = np.asarray(titles, dtype=object)

    # Step 1: reproducible subsampling (important points are always kept)
    rng = np.random.default_rng(seed)
    keep = np.arange(n) if n <= max_points else np.sort(rng.choice(n, max_points, replace=False))

    # Step 2: group rare genres for a readable legend (≤ top_genres + Other)
    uniq, counts = np.unique(genres[keep], return_counts=True)
    main = set(uniq[np.argsort(-counts)][:top_genres])
    shown_genre = np.where(np.isin(genres, list(main)), genres, "Other")

    # Muted "book cloth" colours: distinct enough for 10 genres, but in the same warm family
    palette = ["#2F5D4A", "#B8892E", "#3F6E8C", "#9E3B32", "#7A4E7E", "#6F8F5B",
               "#C27C4E", "#4F7A87", "#A0685F", "#8C8A4B", "#5C5A8F", "#B5838D"]
    fig = go.Figure()
    order = sorted(main, key=lambda g: -int(counts[list(uniq).index(g)])) + ["Other"]
    for i, g in enumerate(order):
        idx = keep[shown_genre[keep] == g]
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=g,
            marker=dict(size=4, opacity=0.45 if g != "Other" else 0.2,
                        color=palette[i % len(palette)] if g != "Other" else "#C9C0B0"),
            text=titles[idx], hovertemplate="%{text}<extra>" + g + "</extra>",
        ))

    # Step 3: query neighbours (circles), then top 3 (stars), then the user
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
                        line=dict(color="#FBF8F1", width=1.5)),
            text=[f"#{k + 1}"], textposition="top center", textfont=dict(size=13, color=USER_COLOR),
            hovertext=[str(label)], hovertemplate="%{hovertext}<extra>recommended</extra>",
        ))
    if user_xy is not None:
        fig.add_trace(go.Scatter(
            x=[user_xy[0]], y=[user_xy[1]], mode="markers+text", name="You",
            marker=dict(symbol="x", size=16, color=USER_COLOR, line=dict(width=2, color="#FBF8F1")),
            text=["YOU"], textposition="bottom center", textfont=dict(size=13, color=USER_COLOR),
            hovertemplate="Your position (weighted barycentre of your closest books)<extra></extra>",
        ))
    # Robust framing: a few isolated books (unusual descriptions) would squash the main
    # cloud; we frame on the 1st-99th percentiles (+ margin), always keeping the top 3 and the user.
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
# 5. Similarity heatmap: user sentences × the book's sentences/labels
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
    """Cosine matrix (rows = the user's texts, columns = the book's labels or sentences).

    This is the most direct view of the SEMANTIC analysis: a dark cell links something the
    user asked for to an element of the book that expresses the same idea, even without any
    shared word (e.g. "old university" ↔ label "Academic & campus").
    """
    sim = np.asarray(sim, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=sim,
        x=[_short(c, 30) for c in col_labels],
        y=[_short(r, 45) for r in row_labels],
        customdata=np.array([[f"{r}  ↔  {c}" for c in col_labels] for r in row_labels], dtype=object),
        colorscale=[[0, "#FBF8F1"], [0.5, "#C9A66B"], [1, "#23493A"]], zmin=zmin, zmax=zmax,
        text=np.round(sim, 2), texttemplate="%{text:.2f}", textfont=dict(size=10),
        hovertemplate="%{customdata}<br>cosine %{z:.3f}<extra></extra>",
        colorbar=dict(title="cos", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return _base_layout(fig, title, height or 160 + 38 * len(row_labels))


# --------------------------------------------------------------------------- #
# 6. Small supporting charts
# --------------------------------------------------------------------------- #

def weights_bar(weights: Mapping[str, float], title: str | None = "Importance you gave to each block",
                max_weight: float = 5.0, height: int = 220) -> go.Figure:
    """Horizontal bars of the weights W_b chosen by the user (Likert 1-5)."""
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
    """Compact gauge of the final score (for a recommendation card)."""
    color = MATCH_COLORS.get(label.split()[0] if label else "", COMPONENT_COLORS["Genre"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(100 * float(score), 1),
        number=dict(suffix="%", font=dict(family=TITLE_FONT["family"], size=28, color="#23493A")),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[55, 75], ticktext=["55", "75"]),
                   bar=dict(color=color),
                   steps=[dict(range=[0, 55], color="#F3EEE4"), dict(range=[55, 75], color="#EAE2D2"),
                          dict(range=[75, 100], color="#DFD4BF")]),
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=0), height=height, font=FONT,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------- #
# 7. Map data: 2D projection and user position
# --------------------------------------------------------------------------- #

def compute_map_coords(matrix: np.ndarray, method: str = "auto", seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Project the (N, d) embeddings to 2D.

    method = "umap" (keeps neighbourhoods: best map, ~30-60 s for 12,000 books),
    "pca" (instant, linear: less distinct clusters) or "auto" (UMAP if installed, otherwise PCA).
    UMAP uses the cosine metric, consistent with the rest of the engine.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if method in ("auto", "umap"):
        try:   # P3's version (frozen UMAP layout: same map for every team member)
            from src.semantic_map import compute_umap  # type: ignore

            return compute_umap(matrix, seed=seed)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001 - e.g. spaCy blocked by Windows when src.embeddings is imported
            pass
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
            return reducer.fit_transform(matrix).astype(np.float32)
        except ImportError:
            if method == "umap":
                raise
    # PCA through SVD (numpy only, no extra dependency)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(np.float32)


def load_or_compute_map_coords(
    matrix: np.ndarray,
    book_ids: Sequence[str],
    path: Path | str = config.DATA_EMBEDDINGS / "umap_2d.npy",
    method: str = "auto",
) -> np.ndarray:
    """2D coordinates aligned with the catalogue: read from disk if valid, otherwise computed and cached.

    umap_2d.npy is normally produced by P3 (notebook 03, src/semantic_map.save_umap). To guarantee
    that row i ↔ book i, we check the companion file umap_book_ids.json written next to it
    (same convention as P3) or, if it does not exist, that the number of rows matches.
    """
    path = Path(path)
    ids_path = path.parent / "umap_book_ids.json"          # same file name as src/semantic_map.py (P3)
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
    """Place the user on the map = cosine-weighted barycentre of their k closest books.

    Why not umap.transform: we would have to keep the fitted UMAP object (heavy, not saved
    by P3) and the result would be unstable. The nearest-neighbour barycentre is simple,
    deterministic and faithful to the idea "you are where the books that resemble you are".

    Returns:
        ((x, y), indices of the k neighbours)
    """
    sims = np.asarray(sims, dtype=float)
    k = int(min(k, len(sims)))
    nn = np.argsort(-sims)[:k]
    w = np.clip(sims[nn], 1e-6, None) ** 2      # emphasise the closest ones
    xy = (coords[nn] * w[:, None]).sum(axis=0) / w.sum()
    return (float(xy[0]), float(xy[1])), nn


def primary_genre(genres: Any) -> str:
    """First genre of a list (P2's `genres` column), 'Unknown' if empty."""
    if isinstance(genres, (list, tuple, np.ndarray)) and len(genres) > 0:
        return str(genres[0])
    if isinstance(genres, str) and genres:
        return genres
    return "Unknown"


# --------------------------------------------------------------------------- #
# 8. Export for the report and the slides
# --------------------------------------------------------------------------- #

def save_figure(fig: go.Figure, name: str, folder: Path | str = config.FIGURES_DIR,
                width: int = 1100, height: int | None = None, scale: float = 2.0) -> Path | None:
    """Export a figure as PNG (docs/figures/05_<name>.png) through kaleido.

    Returns None (with a message) if kaleido / Chrome is not available, instead of crashing
    the notebook: the figure is still visible interactively.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / (name if name.endswith(".png") else f"{name}.png")
    try:
        fig.write_image(str(out), width=width, height=height or fig.layout.height or 600, scale=scale)
        return out
    except Exception as err:  # kaleido missing, Chrome not found (kaleido >= 1.0)…
        print(f"[save_figure] PNG export failed ({type(err).__name__}: {err}). "
              "Install kaleido (pip install kaleido) and, if asked, run `plotly_get_chrome`.")
        return None


def export_all(figures: Mapping[str, go.Figure], **kwargs: Any) -> list[Path]:
    """Export several figures at once: {"radar_profile": fig, ...} -> list of paths."""
    paths = [save_figure(f, n, **kwargs) for n, f in figures.items()]
    return [p for p in paths if p is not None]


__all__: Iterable[str] = [
    "COMPONENT_COLORS", "BOOK_COLORS", "normalize_key", "normalize_scores",
    "radar_profile", "radar_compare", "score_breakdown", "similarity_map", "similarity_heatmap",
    "weights_bar", "score_gauge", "compute_map_coords", "load_or_compute_map_coords",
    "user_map_position", "primary_genre", "save_figure", "export_all",
]
