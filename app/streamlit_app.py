"""
streamlit_app.py : application web du projet (Personne 5).

Lancement (depuis la RACINE du repo) :
    streamlit run app/streamlit_app.py

Parcours utilisateur :
    1. Questionnaire en 5 étapes (17 questions, les 5 types exigés par le sujet) ;
    2. Sauvegarde des réponses en JSON horodaté (data/responses/) + téléchargement ;
    3. Appel du moteur : src.recommender.recommend(answers, top_k=3) (P4).
       Tant que P4 n'a pas livré, on utilise app/stub_recommender.py (même contrat) ;
    4. Page résultats : profil lecteur, top 3 expliqué, décomposition des scores,
       carte sémantique du catalogue, heatmap de similarité (« comment le score est obtenu »).

Organisation du fichier :
    0. imports et chemins        3. barre latérale (progression, mode démo)
    1. ressources en cache       4. les 5 étapes du questionnaire
    2. état et navigation        5. page résultats
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 0. Imports : on ajoute la racine du repo au sys.path pour importer src/ et app/
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

# Compatibilité des versions de Streamlit : `use_container_width` est remplacé par
# `width="stretch"` dans les versions récentes (>= 1.50) et supprimé à terme.
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH: dict[str, Any] = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}
# theme=None : on garde la mise en page définie dans src/viz.py (le thème Plotly de Streamlit
# déplace les légendes et change les couleurs, ce qui casserait la charte commune aux figures).
CHART: dict[str, Any] = {**STRETCH, "theme": None}

st.set_page_config(page_title="GenderBook · semantic book recommender", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

# Petite feuille de style : badges, puces de labels, citation de la phrase la plus proche
st.markdown("""
<style>
.gb-chip{display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:12px;
         font-size:.78rem;color:#fff;white-space:nowrap}
.gb-badge{display:inline-block;padding:3px 12px;border-radius:6px;font-weight:600;color:#fff;font-size:.85rem}
.gb-quote{border-left:4px solid #2F6FDE;padding:.45rem .9rem;margin:.4rem 0 .6rem 0;
          background:rgba(47,111,222,.07);font-style:italic;border-radius:0 6px 6px 0}
.gb-cover{width:100%;max-width:170px;border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.18)}
.gb-nocover{width:100%;max-width:170px;aspect-ratio:2/3;border-radius:6px;background:#E9ECEF;
            display:flex;align-items:center;justify-content:center;font-size:3rem}
.gb-synopsis mark{background:#FFE58A;padding:0 2px;border-radius:3px}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 1. Ressources lourdes, chargées UNE fois (st.cache_resource / st.cache_data)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_blocks() -> dict[str, dict[str, str]]:
    """Référentiel des 4 blocs (blocks.json de P2) : labels des cases à cocher + bulles d'aide."""
    return Q.load_blocks()


@st.cache_resource(show_spinner=False)
def get_books() -> pd.DataFrame:
    """Catalogue (books_enriched.parquet) : utilisé pour retrouver l'auteur favori, les labels…"""
    from src.data_loader import load_books
    return load_books()


@st.cache_resource(show_spinner=False)
def get_engine_context() -> tuple[Any, pd.DataFrame | None, np.ndarray | None, str | None]:
    """(moteur d'embeddings, catalogue aligné, matrice des livres, erreur éventuelle).

    - Si P4 est intégré : on réutilise le moteur de son Recommender (singleton
      get_recommender) : un seul modèle SBERT en mémoire, et un catalogue dont les
      lignes sont exactement alignées sur la matrice d'embeddings.
    - Sinon : EmbeddingEngine de P3.
    Premier lancement : téléchargement du modèle (~90 Mo) puis encodage des 12 000 livres
    (quelques minutes sur CPU) ; ensuite tout est relu depuis data/embeddings/.
    En cas d'échec (pas d'Internet, module manquant…), l'app continue en mode dégradé.
    """
    try:
        try:
            from src.recommender import get_recommender  # type: ignore  # P4
            rec = get_recommender()
            engine, books = rec.engine, rec.books
        except ImportError:
            from src.embeddings import EmbeddingEngine  # P3
            books = get_books().reset_index(drop=True)
            engine = EmbeddingEngine(books=books)
        return engine, books, np.asarray(engine.book_matrix(), dtype=np.float32), None
    except Exception as err:  # noqa: BLE001 - on veut afficher n'importe quelle erreur
        return None, None, None, f"{type(err).__name__}: {err}"


def get_engine() -> tuple[Any, str | None]:
    engine, _, _, err = get_engine_context()
    return engine, err


@st.cache_resource(show_spinner=False)
def get_map_coords() -> np.ndarray | None:
    """Coordonnées 2D du catalogue (umap_2d.npy de P3 si présent, sinon calculées et mises en cache)."""
    engine, books, matrix, _ = get_engine_context()
    if engine is None:
        return None
    return viz.load_or_compute_map_coords(matrix, books["book_id"].astype(str).tolist())


def get_recommend_fn() -> tuple[Callable[..., Any], str]:
    """Le vrai moteur de P4 s'il existe, sinon le stub. Renvoie (fonction, nom du mode)."""
    try:
        from src.recommender import recommend  # type: ignore  # livré par P4
        return recommend, "P4 recommender"
    except ImportError:
        from app.stub_recommender import recommend
        return recommend, "stub"


def run_recommendation(answers: dict) -> dict[str, Any]:
    """Appelle le moteur et renvoie {"profile", "recs", "mode", "warning"}.

    Si le moteur de P4 lève une exception (intégration en cours), on bascule sur le stub
    avec un avertissement visible plutôt que de planter pendant la démo.
    """
    fn, mode = get_recommend_fn()
    engine, _ = get_engine()
    warning = None
    try:
        if mode == "stub":
            _, ebooks, _, _ = get_engine_context()
            profile, recs = fn(answers, top_k=3, engine=engine, books=ebooks if ebooks is not None else get_books())
        else:
            profile, recs = fn(answers, top_k=3)
    except Exception as err:  # noqa: BLE001
        from app.stub_recommender import recommend as stub
        warning = f"The recommender raised {type(err).__name__}: {err}. Showing the stub results instead."
        _, ebooks, _, _ = get_engine_context()
        profile, recs = stub(answers, top_k=3, engine=engine, books=ebooks if ebooks is not None else get_books())
        mode = "stub (fallback)"
    return {"profile": profile, "recs": list(recs), "mode": mode, "warning": warning}


# --------------------------------------------------------------------------- #
# 2. État de session et navigation
# --------------------------------------------------------------------------- #
BLOCKS = get_blocks()
DEFAULTS = Q.default_state(BLOCKS)
RESULTS_STEP = len(Q.STEPS)        # étapes 0..4 = questionnaire, 5 = résultats

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
    # Astuce Streamlit : réaffecter la valeur empêche la suppression de l'état d'un widget
    # qui n'est pas affiché à l'étape courante (sinon les réponses seraient perdues).
    st.session_state[_k] = st.session_state[_k]
st.session_state.setdefault("step", 0)
st.session_state.setdefault("errors", [])

# Lien direct vers un persona de démo : http://localhost:8501/?persona=Young%20fantasy%20fan
# (pratique pour la soutenance et pour scripts/capture_screens.py). Appliqué une seule fois.
_qp = st.query_params.get("persona")
if _qp in Q.DEMO_PERSONAS and not st.session_state.get("_persona_from_url"):
    st.session_state._persona_from_url = True
    for _k, _v in Q.persona_state(_qp, BLOCKS).items():
        st.session_state[_k] = _v


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
    """Valide tout le questionnaire, construit answers.json, le sauvegarde, passe aux résultats."""
    errors = Q.validate_all(form_state())
    st.session_state.errors = errors
    if errors:
        return
    answers = Q.build_answers(form_state(), BLOCKS)
    st.session_state.answers = answers
    try:
        st.session_state.answers_path = str(Q.save_answers(answers))
    except OSError as err:  # dossier en lecture seule (ex. déploiement cloud) : pas bloquant
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
    """Bouton « relâcher les filtres » quand aucun livre ne passe les contraintes dures."""
    st.session_state.q8_audience = "Any"
    st.session_state.q9_formats = list(Q.FORMATS)
    st.session_state.q17_exclude = False
    submit()


# --------------------------------------------------------------------------- #
# 3. Barre latérale : progression, mode démo, état du moteur
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📚 GenderBook")
    st.caption("Semantic book recommendation · NLP project ECE ING5")
    step = st.session_state.step
    st.progress(min(step, RESULTS_STEP) / RESULTS_STEP,
                text="Results" if step == RESULTS_STEP else f"Step {step + 1} / {RESULTS_STEP}")
    for i, name in enumerate(Q.STEPS):
        icon = "✅" if i < step else ("👉" if i == step else "▫️")
        st.button(f"{icon} {name}", key=f"nav_{i}", on_click=go_to, args=(i,),
                  **STRETCH, type="tertiary" if hasattr(st, "badge") else "secondary")

    st.divider()
    st.markdown("**Demo mode**")
    persona = st.selectbox("Persona", list(Q.DEMO_PERSONAS), key="demo_persona",
                           label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.button("Fill form", on_click=lambda: load_state(Q.persona_state(persona, BLOCKS)),
              **STRETCH, help="Pre-fill the questionnaire with this persona.")
    c2.button("Results ▶", on_click=lambda: load_state(Q.persona_state(persona, BLOCKS), True),
              **STRETCH, type="primary", help="Fill and go straight to the results.")

    uploaded = st.file_uploader("Load an answers.json", type="json")
    if uploaded is not None and st.button("Load these answers", **STRETCH):
        try:
            load_state(Q.state_from_answers(json.load(uploaded), BLOCKS))
            st.rerun()
        except (ValueError, KeyError, TypeError) as err:
            st.error(f"Invalid file: {err}")

    st.divider()
    _, mode_name = get_recommend_fn()
    st.caption(f"Recommender: **{mode_name}**"
               + (" ⚠️ (temporary, waiting for P4)" if mode_name == "stub" else ""))
    st.caption(f"Embedding model: `{config.EMBEDDING_MODEL.split('/')[-1]}`")


# --------------------------------------------------------------------------- #
# 4. Les 5 étapes du questionnaire
# --------------------------------------------------------------------------- #

def show_errors() -> None:
    for e in st.session_state.errors:
        st.error(e, icon="⚠️")


def nav_buttons(last: bool = False) -> None:
    show_errors()
    c1, _, c3 = st.columns([1, 3, 1.4])
    if st.session_state.step > 0:
        c1.button("← Back", on_click=go_back, **STRETCH)
    if last:
        c3.button("See my recommendations ✨", on_click=submit, type="primary", **STRETCH)
    else:
        c3.button("Next →", on_click=go_next, type="primary", **STRETCH)


def likert(key: str, question: str, left: str, right: str) -> None:
    """Échelle de Likert 1-5 avec les libellés des deux extrémités (type de question exigé)."""
    st.markdown(f"**{question}**")
    c1, c2, c3 = st.columns([1.7, 3.6, 1.7], vertical_alignment="center")
    c1.caption(f"1 · {left}")
    c2.slider(question, 1, 5, key=key, label_visibility="collapsed")
    c3.caption(f"5 · {right}")


def step_free_text() -> None:
    st.subheader("1 · Tell us about your reading")
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
    """Cases à cocher en grille ; la phrase descriptive du label est dans la bulle d'aide."""
    cols = st.columns(n_cols)
    for i, (label, description) in enumerate(labels.items()):
        cols[i % n_cols].checkbox(label, key=key_fn(label), help=description)
    return sum(bool(st.session_state[key_fn(l)]) for l in labels)


def step_genres_themes() -> None:
    st.subheader("2 · Genres & themes")
    st.caption("Tick as many as you like. Hover the ⓘ to see the sentence the semantic engine uses for "
               "each label.")
    st.markdown("**Q5. Which genres do you enjoy?**")
    n_g = checkbox_grid(BLOCKS["Genre"], Q.genre_key, 3)
    st.markdown("**Q6. Which themes do you like to read about?**")
    n_t = checkbox_grid(BLOCKS["Themes"], Q.theme_key, 4)
    st.caption(f"{n_g} genre(s) and {n_t} theme(s) selected.")
    nav_buttons()


def step_context() -> None:
    st.subheader("3 · Context")
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
    st.subheader("4 · Your taste")
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
    st.subheader("5 · A few guided questions")
    st.radio("Q15. Do you have a favourite author?", ("Yes", "No"), key="q15_has_author", horizontal=True)
    if st.session_state.q15_has_author == "Yes":
        st.text_input("Who is it?", key="q15_author", placeholder="e.g. Donna Tartt")
        name = st.session_state.q15_author.strip()
        if name:
            from src.data_loader import find_author, get_books_by_author
            found = find_author(get_books(), name)
            if found:
                n = len(get_books_by_author(get_books(), found))
                st.success(f"Found in the catalogue: **{found}** ({n} book{'s' * (n > 1)}).", icon="✅")
            else:
                st.info("This author is not in our catalogue: we will rely on your descriptions only.")
        st.checkbox("Q17. Exclude books by this author (I want to discover someone new)", key="q17_exclude")
    st.radio("Q16. Do you prefer character-driven or plot-driven stories?", Q.FOCUS_OPTIONS,
             key="q16_focus", horizontal=True,
             captions=("Deep psychological portraits", "Twists and action", "A balance of both"))
    st.checkbox("I prefer standalone books (no long series)", key="q_standalone")
    nav_buttons(last=True)


# --------------------------------------------------------------------------- #
# 5. Page résultats
# --------------------------------------------------------------------------- #

def chips(labels: dict[str, list[tuple[str, float]]]) -> str:
    """Puces colorées « label (score) », une couleur par bloc (même charte que les graphiques)."""
    out = []
    for block in viz.BLOCK_ORDER:
        for label, score in labels.get(block, []) or []:
            color = viz.COMPONENT_COLORS.get(block, "#666")
            out.append(f'<span class="gb-chip" style="background:{color}" title="{block}">'
                       f"{html.escape(str(label))} · {float(score):.2f}</span>")
    return "".join(out)


def badge(match_label: str) -> str:
    label = str(match_label)
    if "match" not in label.lower():               # P4 : "Excellent" -> "Excellent match"
        label += " match"
    color = viz.MATCH_COLORS.get(label.split()[0], "#8A8F98")
    return f'<span class="gb-badge" style="background:{color}">{html.escape(label)}</span>'


def cover(url: str) -> str:
    if url and url.startswith("http"):
        return f'<img class="gb-cover" src="{html.escape(url)}" alt="" referrerpolicy="no-referrer">'
    return '<div class="gb-nocover">📖</div>'


def highlighted_synopsis(text: str, sentence: str) -> str:
    """Résumé avec la phrase la plus proche de la demande surlignée (<mark>)."""
    safe = html.escape(text or "")
    if sentence:
        s = html.escape(sentence)
        if s in safe:
            safe = safe.replace(s, f"<mark>{s}</mark>", 1)
    return f'<div class="gb-synopsis">{safe}</div>'


def score_table(rec: Any, weights: dict[str, float]) -> pd.DataFrame:
    """Tableau « How was this score computed? » : score de chaque composante, poids, contribution."""
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
            st.markdown(f"### #{rec.rank} · {rec.title}")
            st.markdown(f"*{rec.authors}* &nbsp; {badge(rec.match_label)}", unsafe_allow_html=True)
            if rec.best_sentence:
                st.markdown(f'<div class="gb-quote">“{html.escape(rec.best_sentence)}”<br>'
                            f'<small>closest sentence to your request · cosine '
                            f'{rec.best_sentence_score:.2f}</small></div>', unsafe_allow_html=True)
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
        with st.expander("🔎 How was this score computed?"):
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
            with st.expander("📖 Synopsis (closest sentence highlighted)"):
                st.markdown(highlighted_synopsis(book_row.get("description", ""), rec.best_sentence),
                            unsafe_allow_html=True)
                meta = [f"{book_row.get('format', '')}", f"audience: {book_row.get('audience', '')}"]
                if pd.notna(book_row.get("pub_year")):
                    meta.append(f"published {int(book_row['pub_year'])}")
                if pd.notna(book_row.get("rating_avg")):
                    meta.append(f"⭐ {book_row['rating_avg']:.2f} ({int(book_row['rating_count']):,} ratings)")
                st.caption(" · ".join(meta))
                if book_row.get("url"):
                    st.markdown(f"[See on Book Depository]({book_row['url']})")


def heatmap_data(engine: Any, profile: Any, answers: dict, rec: Any, book_row: pd.Series):
    """Lignes = textes de l'utilisateur, colonnes = labels du livre (+ phrase la plus proche).

    Les labels sont représentés par les MÊMES vecteurs que ceux du moteur (label_matrix de P3,
    qui réutilise les phrases de blocks.json) : la heatmap montre exactement ce que compare le système.
    """
    ft = answers.get("free_text", {})
    rows: list[tuple[str, str]] = [(f"Q{i}: {ft[k]}", ft[k]) for i, k in
                                   ((1, "ideal_book"), (2, "loved_book"), (3, "atmosphere")) if ft.get(k)]
    phrases = getattr(profile, "block_phrases", None) or {}
    for b in viz.BLOCK_ORDER:                      # phrases de préférence par bloc (profil de P4)
        items = list(phrases.get(b) or [])
        if not items:                              # sinon : texte du bloc (phrases jointes par « | » ou « ; »)
            txt = (profile.query_texts or {}).get(b) or ""
            items = [t.strip() for t in txt.replace(" | ", ";").split(";") if t.strip()]
        for t in items[:3]:                        # 3 phrases max par bloc : heatmap lisible
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


def results_to_json(result: dict) -> str:
    """Export JSON des résultats (sans les vecteurs numpy) pour le téléchargement / P6."""
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
    return json.dumps({"answers": st.session_state.get("answers"), "mode": result["mode"],
                       "profile": clean(result["profile"]), "recommendations": clean(result["recs"])},
                      indent=2, ensure_ascii=False)


def page_results() -> None:
    answers = st.session_state.get("answers")
    if not answers:
        st.info("Please fill in the questionnaire first.")
        st.button("Go to the questionnaire", on_click=go_to, args=(0,))
        return

    # Calcul (une seule fois par jeu de réponses)
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

    st.title("Your recommendations")
    if result["mode"].startswith("stub"):
        st.warning("Temporary recommender (stub): the interface is final, but the scores will change once "
                   "the real scoring engine (P4) is plugged in.", icon="🚧")
    if result["warning"]:
        st.error(result["warning"])
    if engine_err:
        st.info(f"Semantic engine unavailable ({engine_err}): map and similarity heatmap are disabled.")
    st.caption(f"Answers saved to `{st.session_state.get('answers_path', '?')}`")

    if not recs:
        st.error("No book satisfies all your hard filters (audience, formats, excluded author).", icon="🙁")
        st.button("Relax the filters and try again", on_click=relax_filters, type="primary")
        st.button("Edit my answers", on_click=go_to, args=(2,))
        return

    books = get_books()
    by_id = {str(b): i for i, b in enumerate(books["book_id"].astype(str))}
    rows = {r.book_id: (books.iloc[by_id[str(r.book_id)]] if str(r.book_id) in by_id else None) for r in recs}

    tab_top, tab_profile, tab_compare, tab_map, tab_hood = st.tabs(
        ["📚 Top 3", "👤 Your profile", "📊 Compare", "🗺️ Semantic map", "🔬 Under the hood"])

    with tab_top:
        for rec in recs:
            render_recommendation(rec, profile, rows[rec.book_id])

    with tab_profile:
        st.markdown(f"> {profile.summary}")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(viz.radar_profile(profile.block_affinities), **CHART)
        with c2:
            st.plotly_chart(viz.weights_bar(profile.weights), **CHART)
            cons = {k: v for k, v in (profile.constraints or {}).items() if v not in (None, [], "")}
            st.markdown("**Hard filters applied**")
            st.json(cons, expanded=True)
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
            sims = matrix @ np.asarray(q, dtype=np.float32).ravel()
            user_xy, nn = viz.user_map_position(coords, sims, k=15)
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
        with st.expander("Raw answers.json"):
            st.json(answers)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ My answers (JSON)", json.dumps(answers, indent=2, ensure_ascii=False),
                       file_name="answers.json", mime="application/json", **STRETCH)
    c2.download_button("⬇️ Results (JSON)", results_to_json(result), file_name="recommendations.json",
                       mime="application/json", **STRETCH)
    c3.button("✏️ Edit my answers", on_click=go_to, args=(0,), **STRETCH)
    c4.button("🔄 Start again", on_click=reset_all, type="primary", **STRETCH)


# --------------------------------------------------------------------------- #
# Routage
# --------------------------------------------------------------------------- #
PAGES = [step_free_text, step_genres_themes, step_context, step_likert, step_guided]
if st.session_state.step >= RESULTS_STEP:
    page_results()
else:
    if st.session_state.step == 0 and not st.session_state.q1_ideal:
        st.title("Find your next book 📚")
        st.write("Answer a few questions (about 3 minutes). Our system analyses the **meaning** of your answers "
                 "with sentence embeddings (SBERT) and compares them with 12,000 book synopses to recommend "
                 "the 3 books that fit you best, with a transparent explanation of every score.")
    PAGES[st.session_state.step]()
