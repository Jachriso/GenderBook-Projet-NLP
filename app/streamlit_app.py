"""
streamlit_app.py: the project's web application (Person 5).

How to run it (from the ROOT of the repo):
    streamlit run app/streamlit_app.py

User journey:
    1. 5-step questionnaire (17 questions, the 5 question types required by the brief);
    2. Answers saved as timestamped JSON (data/responses/) + download;
    3. Call to the engine: src.recommender.recommend(answers, top_k=3) (P4);
    4. Results page: reader profile, explained top 3, score breakdown,
       semantic map of the catalogue, similarity heatmap ("how the score is obtained").

File layout:
    0. imports and paths         3. sidebar (progress, demo mode)
    1. cached resources          4. the 5 questionnaire steps
    2. state and navigation      5. results page
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 0. Imports: the repo root is added to sys.path so that src/ and app/ can be imported
# --------------------------------------------------------------------------- #
import dataclasses
import html
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import questions as Q  # noqa: E402
from src import config, viz  # noqa: E402

# Streamlit version compatibility: `use_container_width` is replaced by
# `width="stretch"` in recent versions (>= 1.50) and will eventually be removed.
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH: dict[str, Any] = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}
# theme=None: keep the layout defined in src/viz.py (Streamlit's Plotly theme moves the
# legends and changes the colours, which would break the colour scheme shared by the figures).
CHART: dict[str, Any] = {**STRETCH, "theme": None}

# App name chosen by the team (the repository keeps its original name, GenderBook)
APP_NAME = "The Bibliophile's Quest"

# Logo (gold "Bibliophile" crest, background made transparent) and browser-tab icon
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS / "bibliophile_logo.png"
FAVICON_PATH = ASSETS / "favicon.png"


def _page_icon() -> Any:
    """The favicon image if it is there, otherwise a book emoji (never crash on a missing file)."""
    try:
        from PIL import Image
        return Image.open(FAVICON_PATH)
    except (ImportError, OSError):
        return "📚"


st.set_page_config(page_title=f"{APP_NAME}: find your next book", page_icon=_page_icon(),
                   layout="wide", initial_sidebar_state="expanded")

# "Reading room" stylesheet: banker's-lamp green, brass and paper, with a soft lamp glow at
# the top of the page. Literata (a serif designed for reading books) for headings, titles and
# quotes; Nunito (rounded, friendly) for the interface. Same palette as src/viz.py.
# If Google Fonts cannot be reached, the browser falls back to Georgia / Segoe UI.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;0,7..72,700;1,7..72,400&family=Nunito:wght@400;600;700&display=swap');
:root{--gb-paper:#FBF8F1;--gb-green:#23493A;--gb-glass:#E3EBE1;--gb-brass:#B8892E;
      --gb-ink:#2B2620;--gb-line:#E6DDCB;--gb-oxblood:#9E3B32}

/* Base: paper background lit by a reading lamp */
[data-testid="stAppViewContainer"]{
  background:radial-gradient(ellipse 70% 38% at 50% -6%, rgba(232,196,120,.30), transparent 70%), var(--gb-paper)}
[data-testid="stHeader"]{background:transparent}
html, body, [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label, [data-testid="stSidebar"] p, button, input, textarea{
  font-family:'Nunito','Segoe UI',sans-serif}
h1, h2, h3, h4{font-family:'Literata',Georgia,serif !important;color:var(--gb-green) !important;
  letter-spacing:-.01em}
h1{font-weight:700 !important}

/* Sidebar: green glass of a banker's lamp */
[data-testid="stSidebar"]{background:var(--gb-glass);border-right:1px solid #CFDCCB}

/* Tighter top spacing: Streamlit leaves ~6rem above the page and a tall header above the sidebar */
.block-container, [data-testid="stMainBlockContainer"]{padding-top:2.2rem !important}
[data-testid="stSidebarUserContent"]{padding-top:0 !important}
[data-testid="stSidebarHeader"]{height:2.5rem !important;min-height:0 !important;padding-top:.5rem !important;
  padding-bottom:0 !important}
[data-testid="stSidebar"] .stButton button{justify-content:flex-start}
[data-testid="stSidebar"] .stButton button div{text-align:left}

/* Text fields look like writing paper */
textarea, input[type="text"]{background:#FFFDF8 !important;border-color:var(--gb-line) !important}
textarea:focus, input[type="text"]:focus{border-color:var(--gb-brass) !important;
  box-shadow:0 0 0 2px rgba(184,137,46,.25) !important}

/* Cards (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--gb-line) !important;border-radius:10px;
  background:#FFFDF8}

/* Welcome headline */
.gb-hero{margin:0 0 1.6rem 0}
.gb-hero h1{font-size:2.7rem;line-height:1.12;margin:0 0 .6rem 0;padding-top:0}
.gb-hero p{font-size:1.08rem;line-height:1.6;color:var(--gb-ink);max-width:42rem}  /* readable line length */
.gb-step{color:var(--gb-brass);font-weight:700;font-size:.9rem;margin-bottom:-.6rem}

/* Match level as a library stamp: the one playful element of the page */
.gb-stamp{display:inline-block;font-family:'Literata',Georgia,serif;font-weight:700;font-size:.82rem;
  padding:1px 10px;border:2px solid currentColor;border-radius:4px;transform:rotate(-4deg);
  margin-left:.4rem;background:rgba(255,253,248,.7)}

/* Label chips: tinted, in the colour of their block */
.gb-chip{display:inline-block;padding:2px 10px;margin:3px 6px 3px 0;border-radius:12px;
  font-size:.8rem;font-weight:600;white-space:nowrap;border:1px solid}

/* Closest sentence, set like a quotation in a book */
.gb-quote{border-left:3px solid var(--gb-brass);padding:.5rem 1rem;margin:.5rem 0 .8rem 0;
  background:rgba(184,137,46,.08);border-radius:0 8px 8px 0;font-family:'Literata',Georgia,serif;
  font-style:italic;line-height:1.6}
.gb-quote small{font-family:'Nunito','Segoe UI',sans-serif;font-style:normal;color:#7A7064}

/* Covers sit like books on a shelf */
.gb-cover{width:100%;max-width:170px;aspect-ratio:2/3;object-fit:cover;border-radius:2px 6px 6px 2px;
  box-shadow:0 8px 18px rgba(60,40,20,.28);
  background:linear-gradient(90deg,#1C3A2E 0 8%,#23493A 8%)}  /* green cloth binding if the image fails */
.gb-nocover{width:100%;max-width:170px;aspect-ratio:2/3;border-radius:2px 6px 6px 2px;
  background:linear-gradient(90deg,#1C3A2E 0 8%,#23493A 8%);color:#E8D9B0;display:flex;
  align-items:center;justify-content:center;font-size:2.6rem;box-shadow:0 8px 18px rgba(60,40,20,.28)}
.gb-synopsis{font-family:'Literata',Georgia,serif;line-height:1.7}
.gb-synopsis mark{background:#F1DFA8;padding:0 2px;border-radius:3px}

@media (max-width:640px){.gb-hero h1{font-size:2rem}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 1. Heavy resources, loaded ONCE (st.cache_resource / st.cache_data)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_blocks() -> dict[str, dict[str, str]]:
    """The 4-block label set (P2's blocks.json): checkbox labels + tooltips."""
    return Q.load_blocks()


@st.cache_resource(show_spinner=False)
def get_books() -> pd.DataFrame:
    """Catalogue (books_enriched.parquet): used to look up the favourite author, the labels…"""
    from src.data_loader import load_books
    return load_books()


@st.cache_resource(show_spinner=False)
def get_engine_context() -> tuple[Any, pd.DataFrame | None, np.ndarray | None, str | None]:
    """(embedding engine, aligned catalogue, book matrix, error if any).

    Reuses the engine of P4's Recommender (singleton get_recommender): a single SBERT
    model in memory, and a catalogue whose rows are exactly aligned with the embedding matrix.
    First launch: the model is downloaded (~90 MB), then the 12,000 books are encoded
    (a few minutes on CPU); afterwards everything is read back from data/embeddings/.
    If something fails (no Internet, spaCy blocked…), the app keeps running in degraded mode.
    """
    try:
        from src.recommender import get_recommender
        rec = get_recommender()
        return rec.engine, rec.books, np.asarray(rec.engine.book_matrix(), dtype=np.float32), None
    except Exception as err:  # noqa: BLE001 - we want to show any error
        return None, None, None, f"{type(err).__name__}: {err}"


def get_engine() -> tuple[Any, str | None]:
    engine, _, _, err = get_engine_context()
    return engine, err


@st.cache_resource(show_spinner=False)
def get_map_coords() -> np.ndarray | None:
    """2D coordinates of the catalogue, from P3's src/semantic_map.py.

    load_umap returns None when umap_2d.npy is missing OR when its saved book_ids no longer
    match the catalogue; in both cases the map is recomputed and saved, so the dots can never
    silently point at the wrong books.
    """
    engine, books, matrix, _ = get_engine_context()
    if engine is None:
        return None
    from src.semantic_map import compute_umap, load_umap, save_umap
    book_ids = books["book_id"].astype(str).tolist()
    coords = load_umap(book_ids)
    if coords is None:
        coords = compute_umap(matrix)
        save_umap(coords, book_ids)
    return coords


def run_recommendation(answers: dict) -> dict[str, Any]:
    """Call P4's engine: {"profile", "recs", "error"}. An error is shown on the page, not raised."""
    try:
        from src.recommender import recommend
        profile, recs = recommend(answers, top_k=3)
        return {"profile": profile, "recs": list(recs), "error": None}
    except Exception as err:  # noqa: BLE001 - shown to the user instead of a Streamlit traceback
        return {"profile": None, "recs": [], "error": f"{type(err).__name__}: {err}"}


# --------------------------------------------------------------------------- #
# 2. Session state and navigation
# --------------------------------------------------------------------------- #
BLOCKS = get_blocks()
DEFAULTS = Q.default_state(BLOCKS)
RESULTS_STEP = len(Q.STEPS)        # steps 0..4 = questionnaire, 5 = results

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
    # Streamlit trick: re-assigning the value stops Streamlit from deleting the state of a
    # widget that is not shown on the current step (otherwise the answers would be lost).
    st.session_state[_k] = st.session_state[_k]
st.session_state.setdefault("step", 0)
st.session_state.setdefault("errors", [])

# Direct link to a demo persona: http://localhost:8501/?persona=Young%20fantasy%20fan
# (handy for the oral presentation and for scripts/capture_screens.py). Applied only once.
_qp = st.query_params.get("persona")
if _qp in Q.DEMO_PERSONAS and not st.session_state.get("_persona_from_url"):
    st.session_state._persona_from_url = True
    for _k, _v in Q.persona_state(_qp, BLOCKS).items():
        st.session_state[_k] = _v

# The demo tools are only shown with ?demo=1 in the link (e.g. http://localhost:8501/?demo=1).
# The flag is kept in the session so the menu stays visible while navigating between steps.
if str(st.query_params.get("demo", "")).lower() in ("1", "true", "yes"):
    st.session_state._demo_enabled = True
DEMO_ENABLED = bool(st.session_state.get("_demo_enabled", False))


def form_state() -> dict[str, Any]:
    return {k: st.session_state.get(k, v) for k, v in DEFAULTS.items()}


def go_next() -> None:
    errors = Q.validate_step(st.session_state.step, form_state())
    st.session_state.errors = errors
    if not errors:
        st.session_state.step = min(st.session_state.step + 1, RESULTS_STEP - 1)


def go_back() -> None:
    st.session_state.errors = []
    st.session_state.step = max(st.session_state.step - 1, 0)


def go_to(step: int) -> None:
    st.session_state.errors = []
    st.session_state.step = step


def submit() -> None:
    """Validate the whole questionnaire, build answers.json, save it, go to the results."""
    errors = Q.validate_all(form_state())
    st.session_state.errors = errors
    if errors:
        return
    answers = Q.build_answers(form_state(), BLOCKS)
    st.session_state.answers = answers
    try:
        st.session_state.answers_path = str(Q.save_answers(answers))
    except OSError as err:  # read-only folder (e.g. cloud deployment): not blocking
        st.session_state.answers_path = f"not saved ({err})"
    st.session_state.pop("result", None)
    st.session_state.step = RESULTS_STEP


def load_state(state: dict[str, Any], jump_to_results: bool = False) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state.errors = []
    if jump_to_results:
        submit()
    else:
        st.session_state.step = 0


def reset_all() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for k in ("answers", "answers_path", "result", "errors"):
        st.session_state.pop(k, None)
    st.session_state.step = 0


def relax_filters() -> None:
    """"Relax the filters" button when no book passes the hard constraints."""
    st.session_state.q8_audience = "Any"
    st.session_state.q9_formats = list(Q.FORMATS)
    st.session_state.q17_exclude = False
    submit()


# --------------------------------------------------------------------------- #
# 3. Sidebar: progress, demo mode, engine status
# --------------------------------------------------------------------------- #
with st.sidebar:
    if LOGO_PATH.exists():
        _, logo_col, _ = st.columns([1, 1.6, 1])          # small crest, centred
        logo_col.image(str(LOGO_PATH), **STRETCH)
    st.markdown(f"## {APP_NAME}")
    st.caption("A semantic book recommender, ECE ING5 NLP project")
    step = st.session_state.step
    st.progress(min(step, RESULTS_STEP) / RESULTS_STEP,
                text="Results" if step == RESULTS_STEP else f"Step {step + 1} / {RESULTS_STEP}")
    for i, name in enumerate(Q.STEPS):
        # Done steps get a plain tick, the current step is the highlighted (primary) button
        quiet = "tertiary" if hasattr(st, "badge") else "secondary"
        st.button(f"✓ {name}" if i < step else name, key=f"nav_{i}", on_click=go_to, args=(i,),
                  **STRETCH, type="primary" if i == step else quiet)

    # Demo tools, folded away at the bottom of the sidebar and only shown with ?demo=1:
    # a user goes straight to the questions, the presenter can still pre-fill a persona
    # or reload a saved answers.json.
    if DEMO_ENABLED:
        st.divider()
        with st.expander("Demo mode"):
            persona = st.selectbox("Persona", list(Q.DEMO_PERSONAS), key="demo_persona")
            c1, c2 = st.columns(2)
            c1.button("Fill form", on_click=lambda: load_state(Q.persona_state(persona, BLOCKS)),
                      **STRETCH, help="Pre-fill the questionnaire with this persona.")
            c2.button("Results ▶", on_click=lambda: load_state(Q.persona_state(persona, BLOCKS), True),
                      **STRETCH, help="Fill and go straight to the results.")

            uploaded = st.file_uploader("Load an answers.json", type="json")
            if uploaded is not None and st.button("Load these answers", **STRETCH):
                try:
                    load_state(Q.state_from_answers(json.load(uploaded), BLOCKS))
                    st.rerun()
                except (ValueError, KeyError, TypeError) as err:
                    st.error(f"Invalid file: {err}")


# --------------------------------------------------------------------------- #
# 4. The 5 questionnaire steps
# --------------------------------------------------------------------------- #

def show_errors() -> None:
    for e in st.session_state.errors:
        st.error(e)


def nav_buttons(last: bool = False) -> None:
    show_errors()
    c1, _, c3 = st.columns([1, 3, 1.4])
    if st.session_state.step > 0:
        c1.button("← Back", on_click=go_back, **STRETCH)
    if last:
        c3.button("See my recommendations", on_click=submit, type="primary", **STRETCH)
    else:
        c3.button("Next →", on_click=go_next, type="primary", **STRETCH)


def step_heading(i: int) -> None:
    """Small "Step i of 5" line in brass, then the step title in Literata."""
    st.markdown(f'<p class="gb-step">Step {i + 1} of {len(Q.STEPS)}</p>', unsafe_allow_html=True)
    st.subheader(Q.STEPS[i])


def likert(key: str, question: str, left: str, right: str) -> None:
    """Likert scale 1-5 with labels at both ends (required question type)."""
    st.markdown(f"**{question}**")
    c1, c2, c3 = st.columns([1.7, 3.6, 1.7], vertical_alignment="center")
    c1.caption(f"1: {left}")
    c2.slider(question, 1, 5, key=key, label_visibility="collapsed")
    c3.caption(f"5: {right}")


def step_free_text() -> None:
    step_heading(0)
    st.caption("Write freely, in English: your words are analysed **semantically** (by meaning, not by "
               "keywords), so a description like *“an old university in the rain”* can match a book "
               "that never uses those words.")
    for key, (_, question, placeholder, required) in Q.FREE_TEXT_QUESTIONS.items():
        st.text_area(question + (" *" if required else ""), key=key, placeholder=placeholder,
                     height=110 if required else 90)
        if required:
            n = len(st.session_state[key].strip())
            st.caption(f"{n} characters" + ("" if n >= Q.MIN_Q1_CHARS else f" · minimum {Q.MIN_Q1_CHARS}"))
    nav_buttons()


def checkbox_grid(labels: dict[str, str], key_fn: Callable[[str], str], n_cols: int) -> int:
    """Checkboxes in a grid; the label's descriptive sentence is shown in the tooltip."""
    cols = st.columns(n_cols)
    for i, (label, description) in enumerate(labels.items()):
        cols[i % n_cols].checkbox(label, key=key_fn(label), help=description)
    return sum(bool(st.session_state[key_fn(l)]) for l in labels)


def step_genres_themes() -> None:
    step_heading(1)
    st.caption("Tick as many as you like. Hover the ⓘ to see the sentence the semantic engine uses for "
               "each label.")
    st.markdown("**Q5. Which genres do you enjoy?**")
    n_g = checkbox_grid(BLOCKS["Genre"], Q.genre_key, 3)
    st.markdown("**Q6. Which themes do you like to read about?**")
    n_t = checkbox_grid(BLOCKS["Themes"], Q.theme_key, 4)
    st.caption(f"{n_g} genre(s) and {n_t} theme(s) selected.")
    nav_buttons()


def step_context() -> None:
    step_heading(2)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        era_options = [Q.NO_PREFERENCE] + [e for e in Q.ERA_LABELS if e in BLOCKS["Atmosphere"]]
        captions = ["Any era is fine"] + [BLOCKS["Atmosphere"][e].split(",")[0] for e in era_options[1:]]
        st.radio("Q7. In which era would you like the story to take place?", era_options,
                 captions=captions, key="q7_period")
    with c2:
        st.selectbox("Q8. Who is the book for?", Q.AUDIENCES, key="q8_audience",
                     help="Hard filter on the audience of the book (Any = no filter).")
        st.multiselect("Q9. Which formats are fine for you?", Q.FORMATS, key="q9_formats")
    nav_buttons()


def step_likert() -> None:
    step_heading(3)
    st.caption("Your answers are turned into sentences (e.g. 5/5 complexity → *“a complex, dense, literary "
               "narrative”*) and compared semantically with the books, not used as raw numbers.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### How do you like your books?")
        for key, (question, left, right) in Q.LIKERT_TASTE.items():
            likert(key, question, left, right)
    with c2:
        st.markdown("##### Q14. How important is each criterion to you?")
        for key, (question, left, right) in Q.LIKERT_WEIGHTS.items():
            likert(key, question, left, right)
        st.caption("These weights W_b enter the coverage score Σ W_b·S_b / Σ W_b.")
    nav_buttons()


def step_guided() -> None:
    step_heading(4)
    st.radio("Q15. Do you have a favourite author?", ("Yes", "No"), key="q15_has_author", horizontal=True)
    if st.session_state.q15_has_author == "Yes":
        st.text_input("Who is it?", key="q15_author", placeholder="e.g. Donna Tartt")
        name = st.session_state.q15_author.strip()
        if name:
            from src.data_loader import find_author, get_books_by_author
            found = find_author(get_books(), name)
            if found:
                n = len(get_books_by_author(get_books(), found))
                st.success(f"Found in the catalogue: **{found}** ({n} book{'s' * (n > 1)}).")
            else:
                st.info("This author is not in our catalogue: we will rely on your descriptions only.")
        st.checkbox("Q17. Exclude books by this author (I want to discover someone new)", key="q17_exclude")
    st.radio("Q16. Do you prefer character-driven or plot-driven stories?", Q.FOCUS_OPTIONS,
             key="q16_focus", horizontal=True,
             captions=("Deep psychological portraits", "Twists and action", "A balance of both"))
    st.checkbox("I prefer standalone books (no long series)", key="q_standalone")
    nav_buttons(last=True)


# --------------------------------------------------------------------------- #
# 5. Results page
# --------------------------------------------------------------------------- #

def chips(labels: dict[str, list[tuple[str, float]]]) -> str:
    """Coloured "label (score)" chips, one colour per block (same colour scheme as the charts)."""
    out = []
    for block in viz.BLOCK_ORDER:
        for label, score in labels.get(block, []) or []:
            color = viz.COMPONENT_COLORS.get(block, "#666")
            out.append(f'<span class="gb-chip" style="background:{color}1A;color:{color};'
                       f'border-color:{color}55" title="{block} label, similarity {float(score):.2f}">'
                       f"{html.escape(str(label))} {float(score):.2f}</span>")
    return "".join(out)


def badge(match_label: str) -> str:
    label = str(match_label)
    if "match" not in label.lower():               # P4: "Excellent" -> "Excellent match"
        label += " match"
    color = viz.MATCH_COLORS.get(label.split()[0], "#8A8178")
    return f'<span class="gb-stamp" style="color:{color}">{html.escape(label)}</span>'


def cover(url: str) -> str:
    if url and url.startswith("http"):
        return f'<img class="gb-cover" src="{html.escape(url)}" alt="" referrerpolicy="no-referrer">'
    return '<div class="gb-nocover"></div>'


def highlighted_synopsis(text: str, sentence: str) -> str:
    """Synopsis with the sentence closest to the request highlighted (<mark>)."""
    safe = html.escape(text or "")
    if sentence:
        s = html.escape(sentence)
        if s in safe:
            safe = safe.replace(s, f"<mark>{s}</mark>", 1)
    return f'<div class="gb-synopsis">{safe}</div>'


def score_table(rec: Any, weights: dict[str, float]) -> pd.DataFrame:
    """"How was this score computed?" table: score, weight and contribution of each component."""
    bs = viz.normalize_scores(rec.block_scores)
    contrib = viz.normalize_scores(rec.contributions)
    w = viz.normalize_scores(weights)
    w_sum = sum(w.get(b, 0) for b in viz.BLOCK_ORDER if b in bs) or 1.0
    rows = [{"Component": "Synopsis similarity S̃_desc", "Score": bs.get("Description"),
             "Weight": f"α = {config.ALPHA_DESC}", "Contribution": contrib.get("Description")}]
    for b in viz.BLOCK_ORDER:
        if b in bs or b in contrib:
            rows.append({"Component": f"{b} affinity S̃_{b}", "Score": bs.get(b),
                         "Weight": f"β·W/ΣW = {config.BETA_COVERAGE}×{w.get(b, 0):g}/{w_sum:g}",
                         "Contribution": contrib.get(b)})
    if "Popularity" in contrib:
        rows.append({"Component": "Popularity bonus", "Score": None, "Weight": "λ (Q13)",
                     "Contribution": contrib["Popularity"]})
    if "Avoid" in contrib:
        rows.append({"Component": "“Avoid” penalty", "Score": None,
                     "Weight": f"μ = {config.MU_AVOID}, τ = {config.TAU_AVOID}", "Contribution": contrib["Avoid"]})
    for k in sorted(set(contrib) - {"Description", "Popularity", "Avoid", *viz.BLOCK_ORDER}):
        rows.append({"Component": k, "Score": bs.get(k), "Weight": "", "Contribution": contrib[k]})
    rows.append({"Component": "**Final score**", "Score": None, "Weight": "",
                 "Contribution": float(rec.final_score)})
    return pd.DataFrame(rows)


FORMULA = (r"\text{Final}= \frac{%s\,\tilde S_{desc} + %s\,\overbrace{\frac{\sum_b W_b\,\tilde S_b}{\sum_b W_b}}"
           r"^{\text{coverage}} + \lambda\,\text{Pop}}{%s + %s + \lambda}"
           r" - %s\,\max(0,\ \cos(q_{avoid},e)-%s)"
           % (config.ALPHA_DESC, config.BETA_COVERAGE, config.ALPHA_DESC, config.BETA_COVERAGE,
              config.MU_AVOID, config.TAU_AVOID))


def render_recommendation(rec: Any, profile: Any, book_row: pd.Series | None) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 3.2, 1.3], gap="medium")
        c1.markdown(cover(rec.image_url), unsafe_allow_html=True)
        with c2:
            st.markdown(f"### {rec.rank}. {rec.title}")
            st.markdown(f"*{rec.authors}* &nbsp; {badge(rec.match_label)}", unsafe_allow_html=True)
            if rec.best_sentence:
                st.markdown(f'<div class="gb-quote">“{html.escape(rec.best_sentence)}”<br>'
                            f'<small>The sentence of the synopsis closest to your request '
                            f'(cosine {rec.best_sentence_score:.2f})</small></div>', unsafe_allow_html=True)
            st.write(rec.explanation)
            st.markdown(chips(rec.matched_labels), unsafe_allow_html=True)
        with c3:
            st.plotly_chart(viz.score_gauge(rec.final_score, rec.match_label), **CHART,
                            config={"displayModeBar": False}, key=f"gauge_{rec.rank}")
            st.metric("Preference coverage", f"{rec.coverage:.0%}")

        g1, g2 = st.columns(2)
        g1.plotly_chart(viz.radar_compare(profile.weights, {rec.title: rec.block_scores}, title="You vs this book",
                                          height=340), **CHART, key=f"radar_{rec.rank}")
        g2.plotly_chart(viz.score_breakdown([rec], title="Score decomposition", height=230, show_legend=False),
                        **CHART, key=f"bar_{rec.rank}")
        with st.expander("How was this score computed?"):
            st.latex(FORMULA)
            st.dataframe(score_table(rec, profile.weights), hide_index=True, **STRETCH,
                         column_config={"Score": st.column_config.NumberColumn(format="%.2f"),
                                        "Contribution": st.column_config.NumberColumn(format="%+.3f")})
            lam = getattr(profile, "lambda_pop", None)
            st.caption("S̃ = cosine similarity calibrated to [0, 1] with the catalogue percentiles p5/p95. "
                       "W_b = the importance you gave to each block (Q14). λ comes from Q13"
                       + (f" (here λ = {lam:.2f})" if lam is not None else "")
                       + ". Contributions already include the division by α + β + λ, so they add up "
                       "to the final score.")
        if book_row is not None:
            with st.expander("Synopsis (closest sentence highlighted)"):
                st.markdown(highlighted_synopsis(book_row.get("description", ""), rec.best_sentence),
                            unsafe_allow_html=True)
                meta = [f"{book_row.get('format', '')}", f"audience: {book_row.get('audience', '')}"]
                if pd.notna(book_row.get("pub_year")):
                    meta.append(f"published {int(book_row['pub_year'])}")
                if pd.notna(book_row.get("rating_avg")):
                    meta.append(f"rated {book_row['rating_avg']:.2f}/5 ({int(book_row['rating_count']):,} ratings)")
                st.caption(" · ".join(meta))
                if book_row.get("url"):
                    st.markdown(f"[See on Book Depository]({book_row['url']})")


def heatmap_data(engine: Any, profile: Any, answers: dict, rec: Any, book_row: pd.Series):
    """Rows = the user's texts, columns = the book's labels (+ closest sentence).

    Labels are represented by the SAME vectors as in the engine (P3's label_matrix, which
    reuses the sentences of blocks.json): the heatmap shows exactly what the system compares.
    """
    ft = answers.get("free_text", {})
    rows: list[tuple[str, str]] = [(f"Q{i}: {ft[k]}", ft[k]) for i, k in
                                   ((1, "ideal_book"), (2, "loved_book"), (3, "atmosphere")) if ft.get(k)]
    phrases = getattr(profile, "block_phrases", None) or {}
    for b in viz.BLOCK_ORDER:                      # preference sentences per block (P4's profile)
        items = list(phrases.get(b) or [])
        if not items:                              # otherwise: the block's text (sentences joined by " | " or ";")
            txt = (profile.query_texts or {}).get(b) or ""
            items = [t.strip() for t in txt.replace(" | ", ";").split(";") if t.strip()]
        for t in items[:3]:                        # at most 3 sentences per block: readable heatmap
            rows.append((f"{b}: {t}", t))
    cols_lbl: list[str] = []
    col_vecs: list[np.ndarray] = []
    for b, col in (("Genre", "genres"), ("Themes", "themes"), ("Style", "styles"), ("Atmosphere", "atmospheres")):
        names, mat = engine.label_matrix(b)
        for label in list(book_row.get(col) or []):
            if label in names:
                cols_lbl.append(f"{b} · {label}")
                col_vecs.append(mat[names.index(label)])
    if rec.best_sentence:
        cols_lbl.append("Closest sentence")
        col_vecs.append(engine.encode([rec.best_sentence])[0])
    if not rows or not col_vecs:
        return None
    row_vecs = engine.encode([t for _, t in rows])
    sim = row_vecs @ np.vstack(col_vecs).T
    return sim, [r for r, _ in rows], cols_lbl


def filters_summary(constraints: dict[str, Any]) -> str:
    """Hard filters in plain words (one bullet per filter) instead of raw JSON."""
    lines = []
    audience = constraints.get("audience")
    if audience and audience != "Any":
        lines.append(f"- Audience: {audience}")
    formats = constraints.get("formats")
    if formats:
        lines.append(f"- Formats: {', '.join(formats)}")
    excluded = constraints.get("excluded_authors") or []
    if excluded:
        lines.append(f"- Leaving out books by {', '.join(excluded)}")
    if constraints.get("fiction_only"):
        lines.append("- Fiction only")
    if constraints.get("standalone_only"):
        lines.append("- Standalone books preferred")
    return "\n".join(lines) or "No hard filter: every book in the catalogue was considered."


def results_to_json(result: dict) -> str:
    """JSON export of the results (without the numpy vectors) for download / P6."""
    def clean(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            obj = dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "query_vecs"}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    return json.dumps({"answers": st.session_state.get("answers"),
                       "profile": clean(result["profile"]), "recommendations": clean(result["recs"])},
                      indent=2, ensure_ascii=False)


def page_results() -> None:
    answers = st.session_state.get("answers")
    if not answers:
        st.info("Please fill in the questionnaire first.")
        st.button("Go to the questionnaire", on_click=go_to, args=(0,))
        return

    # Computation (only once per set of answers)
    key = json.dumps(answers, sort_keys=True)
    if st.session_state.get("result", {}).get("key") != key:
        with st.spinner("Loading the semantic engine and scoring 12,000 books… "
                        "(the very first launch can take a few minutes)"):
            get_engine()
            result = run_recommendation(answers)
        result["key"] = key
        st.session_state.result = result
    result = st.session_state.result
    profile, recs = result["profile"], result["recs"]
    engine, engine_err = get_engine()

    st.title("Three books picked for you")
    if result["error"]:
        st.error(f"The recommendation engine failed ({result['error']}). "
                 "Check that `python -m scripts.prepare_app` runs without errors, then try again.")
        st.button("Edit my answers", on_click=go_to, args=(0,))
        return
    if engine_err:
        st.info(f"Semantic engine unavailable ({engine_err}): map and similarity heatmap are disabled.")

    if not recs:
        st.error("No book satisfies all your hard filters (audience, formats, excluded author).")
        st.button("Relax the filters and try again", on_click=relax_filters, type="primary")
        st.button("Edit my answers", on_click=go_to, args=(2,))
        return

    books = get_books()
    by_id = {str(b): i for i, b in enumerate(books["book_id"].astype(str))}
    rows = {r.book_id: (books.iloc[by_id[str(r.book_id)]] if str(r.book_id) in by_id else None) for r in recs}

    tab_top, tab_profile, tab_compare, tab_map, tab_hood = st.tabs(
        ["Top 3", "Your profile", "Compare", "Semantic map", "Under the hood"])

    with tab_top:
        for rec in recs:
            render_recommendation(rec, profile, rows[rec.book_id])

    with tab_profile:
        st.markdown(f"> {profile.summary}")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(viz.radar_profile(profile.block_affinities), **CHART)
        with c2:
            st.plotly_chart(viz.weights_bar(profile.weights), **CHART)
            st.markdown("**Hard filters applied**")
            st.markdown(filters_summary(profile.constraints or {}))
            with st.expander("Raw filter values"):
                st.json(profile.constraints or {}, expanded=True)
        st.caption("Each radar shows the labels of one block that are semantically closest to your answers "
                   "(cosine between your text and the label's descriptive sentence, rescaled per block).")

    with tab_compare:
        st.plotly_chart(viz.score_breakdown(recs), **CHART)
        st.caption("Each bar segment is the contribution of one component (weight × calibrated similarity); "
                   "the diamond is the final score. Red segments are penalties.")
        st.plotly_chart(viz.radar_compare(profile.weights, {r.title: r.block_scores for r in recs}),
                        **CHART)

    with tab_map:
        if engine is None:
            st.info("The semantic map needs the embedding engine.")
        else:
            with st.spinner("Projecting the catalogue in 2D (first time only)…"):
                coords = get_map_coords()
            _, ebooks, matrix, _ = get_engine_context()
            q = (profile.query_vecs or {}).get("global")
            if q is None:
                q = engine.encode([profile.query_texts.get("global", "")])[0]
            from src.semantic_map import project_query
            q = np.asarray(q, dtype=np.float32).ravel()
            user_xy = tuple(project_query(q, matrix, coords, k=15))   # weighted mean of the 15 closest books
            nn = np.argsort(-(matrix @ q))[:15]                         # the same 15 books, circled on the map
            eng_idx = {b: i for i, b in enumerate(ebooks["book_id"].astype(str))}
            hl = [eng_idx[str(r.book_id)] for r in recs if str(r.book_id) in eng_idx]
            fig = viz.similarity_map(coords, [viz.primary_genre(g) for g in ebooks["genres"]],
                                     ebooks["title"].tolist(), highlight=hl,
                                     highlight_labels=[r.title for r in recs], user_xy=user_xy, neighbours=nn)
            st.plotly_chart(fig, **CHART)
            st.caption("Each dot is a book, placed by the meaning of its synopsis (SBERT embeddings projected in "
                       "2D). Nearby books tell similar stories. ✖ = you (barycentre of your 15 closest books), "
                       "★ = your top 3, circles = your closest books.")

    with tab_hood:
        if engine is None:
            st.info("The similarity heatmap needs the embedding engine.")
        else:
            titles = {f"#{r.rank} {r.title}": r for r in recs}
            choice = st.selectbox("Book", list(titles))
            rec = titles[choice]
            data = heatmap_data(engine, profile, answers, rec, rows[rec.book_id])
            if data is None:
                st.info("Not enough text to compare.")
            else:
                sim, rlab, clab = data
                st.plotly_chart(viz.similarity_heatmap(sim, rlab, clab), **CHART)
                st.caption("Cosine similarity between your own words (rows) and the labels the NLP enrichment "
                           "assigned to this book (columns). Dark cells link ideas that mean the same thing, "
                           "even without any word in common.")
        st.caption(f"Answers saved to `{st.session_state.get('answers_path', '?')}`")
        with st.expander("Raw answers.json"):
            st.json(answers)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Download my answers (JSON)", json.dumps(answers, indent=2, ensure_ascii=False),
                       file_name="answers.json", mime="application/json", **STRETCH)
    c2.download_button("Download the results (JSON)", results_to_json(result), file_name="recommendations.json",
                       mime="application/json", **STRETCH)
    c3.button("Edit my answers", on_click=go_to, args=(0,), **STRETCH)
    c4.button("Start again", on_click=reset_all, type="primary", **STRETCH)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
PAGES = [step_free_text, step_genres_themes, step_context, step_likert, step_guided]
if st.session_state.step >= RESULTS_STEP:
    page_results()
else:
    if st.session_state.step == 0 and not st.session_state.q1_ideal:
        st.markdown(
            '<div class="gb-hero"><h1>What would you like to read next?</h1>'
            "<p>Tell us about the books you love and the mood you are in. We read the meaning of your "
            "answers, compare it with 12,000 book summaries and suggest three books, with the reasons "
            "behind each choice. It takes about three minutes.</p></div>", unsafe_allow_html=True)
    PAGES[st.session_state.step]()
