"""
questions.py : contenu du questionnaire (Personne 5).

Ce module ne dépend PAS de Streamlit : il définit les questions, les personas de démo,
la validation et la construction de answers.json (contrat PLAN §4.3). On peut donc le
tester avec pytest et le réutiliser dans les notebooks (P4, P6) pour fabriquer des réponses.

Correspondance avec le tableau des questions (PLAN §6) et les 5 types exigés par le sujet :
    Q1-Q4   texte libre (Q1 obligatoire, >= 30 caractères)    -> free_text
    Q5-Q6   cases à cocher (genres, thèmes)                   -> genres, themes
    Q7      choix multiple, radio (époque du récit)            -> period
    Q8      choix multiple, selectbox (public)                 -> audience
    Q9      multiselect (formats)                              -> formats
    Q10-Q14 échelles de Likert 1-5                             -> likert
    Q15-Q17 questions guidées (conditionnelles)                -> favorite_author, guided
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from src import config

# --------------------------------------------------------------------------- #
# 1. Référentiel des labels (lu dans blocks.json de P2 : une seule source de vérité)
# --------------------------------------------------------------------------- #


def load_blocks(path: Path | str = config.BLOCKS_JSON) -> dict[str, dict[str, str]]:
    """{bloc: {label: phrase descriptive}}. Les phrases servent de bulles d'aide (help=)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Labels d'ÉPOQUE du bloc Atmosphere (P2). Q7 renvoie exactement l'un de ces labels :
# P4 peut donc retrouver directement la phrase descriptive dans blocks.json.
ERA_LABELS = ("Ancient world", "Medieval", "18th-19th century", "20th century", "Contemporary", "Future")
NO_PREFERENCE = "No preference"

AUDIENCES = ("Adult", "Young Adult", "Children", "Any")   # "Any" = pas de filtre (géré par apply_constraints)
FORMATS = tuple(config.KEEP_FORMATS)                       # ("Paperback", "Hardback")
FOCUS_OPTIONS = ("Characters", "Plot", "Both")

MIN_Q1_CHARS = 30

# --------------------------------------------------------------------------- #
# 2. Textes des questions (interface en anglais, cohérente avec le dataset)
# --------------------------------------------------------------------------- #

STEPS = (
    "Tell us about your reading",
    "Genres & themes",
    "Context",
    "Your taste (1-5)",
    "A few guided questions",
)

FREE_TEXT_QUESTIONS = {
    # clé de session : (clé dans answers.json, question, placeholder, obligatoire)
    "q1_ideal": ("ideal_book", "Q1. Describe the kind of book you'd love to read right now.",
                 "e.g. A psychological novel about a group of students who commit a crime and must live "
                 "with the guilt, full of moral dilemmas…", True),
    "q2_loved": ("loved_book", "Q2. Tell us about a book you loved, and why.",
                 "e.g. I loved The Secret History because of its dark academic atmosphere and its "
                 "complex characters…", False),
    "q3_atmo": ("atmosphere", "Q3. What atmosphere or setting are you in the mood for?",
                "e.g. Something melancholic and rainy, set in an old university…", False),
    "q4_avoid": ("avoid", "Q4. Anything you want to avoid? (optional)",
                 "e.g. No romance-centred plots, no gore…", False),
}

# Échelles de Likert : clé -> (question, libellé à 1, libellé à 5)
LIKERT_TASTE = {
    "complexity": ("Q10. Narrative complexity", "Simple & easy to read", "Complex, dense & literary"),
    "pace": ("Q11. Pace", "Slow & contemplative", "Fast-paced page-turner"),
    "darkness": ("Q12. Tone", "Light & feel-good", "Dark & intense"),
    # Même sens que P4 (profile.lambda_from_likert) : 1 = pépites méconnues, 5 = best-sellers
    "popularity": ("Q13. Popularity", "Hidden gems", "Well-known best-sellers"),
}
LIKERT_WEIGHTS = {
    "w_genre": ("Genre", "Not important", "Essential"),
    "w_themes": ("Themes", "Not important", "Essential"),
    "w_style": ("Writing style", "Not important", "Essential"),
    "w_atmosphere": ("Atmosphere & era", "Not important", "Essential"),
}
LIKERT_KEYS = tuple(LIKERT_TASTE) + tuple(LIKERT_WEIGHTS)

# --------------------------------------------------------------------------- #
# 3. État par défaut (clés de st.session_state)
# --------------------------------------------------------------------------- #


def genre_key(label: str) -> str:
    return f"g::{label}"


def theme_key(label: str) -> str:
    return f"t::{label}"


def default_state(blocks: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Valeurs initiales de TOUS les widgets du questionnaire.

    On initialise st.session_state avec ce dict AVANT d'afficher les widgets et on ne passe
    jamais `value=` aux widgets : c'est ce qui permet au mode démo de pré-remplir le
    formulaire et aux réponses de survivre à la navigation entre étapes.
    """
    state: dict[str, Any] = {k: "" for k in FREE_TEXT_QUESTIONS}
    state.update({genre_key(g): False for g in blocks["Genre"]})
    state.update({theme_key(t): False for t in blocks["Themes"]})
    state.update({
        "q7_period": NO_PREFERENCE,
        "q8_audience": "Adult",
        "q9_formats": list(FORMATS),
        **{k: 3 for k in LIKERT_KEYS},
        "q15_has_author": "No",
        "q15_author": "",
        "q16_focus": "Both",
        "q17_exclude": False,
        "q_standalone": False,
    })
    return state


# --------------------------------------------------------------------------- #
# 4. Personas de démo (bouton « mode démo » pour la soutenance)
# --------------------------------------------------------------------------- #

DEMO_PERSONAS: dict[str, dict[str, Any]] = {
    "Dark academia reader": {
        "q1_ideal": "A psychological literary novel about a group of brilliant, isolated students who get "
                    "drawn into a crime and then have to live with the guilt. I want moral dilemmas, "
                    "obsession and complex friendships rather than action.",
        "q2_loved": "I loved The Secret History by Donna Tartt for its elitist university setting, the "
                    "slow unravelling of the characters and the feeling of inevitable tragedy.",
        "q3_atmo": "Melancholic and autumnal, an old college with libraries, rain and long conversations "
                   "late at night.",
        "q4_avoid": "No romance-centred plots and no gore.",
        "genres": ["Literary Fiction", "Mystery & Detective"],
        "themes": ["Morality & guilt", "Friendship", "Identity & self-discovery"],
        "q7_period": "20th century", "q8_audience": "Adult", "q9_formats": ["Paperback", "Hardback"],
        "complexity": 4, "pace": 2, "darkness": 4, "popularity": 3,
        "w_genre": 3, "w_themes": 5, "w_style": 4, "w_atmosphere": 4,
        "q15_has_author": "Yes", "q15_author": "Donna Tartt", "q16_focus": "Characters",
        "q17_exclude": True, "q_standalone": False,
    },
    "Young fantasy fan": {
        "q1_ideal": "An epic adventure in a magical world where a teenager discovers hidden powers, goes "
                    "to a school of magic and fights a dark lord with loyal friends.",
        "q2_loved": "Harry Potter, because of the magic school, the friendship between the three heroes "
                    "and the humour.",
        "q3_atmo": "Whimsical and enchanting: castles, forests full of magical creatures, a sense of wonder.",
        "q4_avoid": "Nothing too scary or violent.",
        "genres": ["Fantasy", "Young Adult"],
        "themes": ["Friendship", "Coming of age", "Adventure & exploration"],
        "q7_period": NO_PREFERENCE, "q8_audience": "Young Adult", "q9_formats": ["Paperback", "Hardback"],
        "complexity": 2, "pace": 4, "darkness": 2, "popularity": 4,
        "w_genre": 5, "w_themes": 4, "w_style": 2, "w_atmosphere": 3,
        "q15_has_author": "Yes", "q15_author": "J.K. Rowling", "q16_focus": "Plot",
        "q17_exclude": True, "q_standalone": False,
    },
    "Dystopia thinker": {
        "q1_ideal": "A dystopian novel about a society controlled by surveillance and technology, where one "
                    "ordinary person slowly starts to question the system and rebel.",
        "q2_loved": "Nineteen Eighty-Four, for its chilling picture of totalitarian power and the way it "
                    "makes you think about our own world.",
        "q3_atmo": "Cold, oppressive and grey: a future city where everyone is watched.",
        "q4_avoid": "Light romantic comedies.",
        "genres": ["Science Fiction", "Classics"],
        "themes": ["Freedom & oppression", "Technology & humanity", "Power & politics"],
        "q7_period": "Future", "q8_audience": "Adult", "q9_formats": ["Paperback", "Hardback"],
        "complexity": 4, "pace": 3, "darkness": 4, "popularity": 4,
        "w_genre": 4, "w_themes": 5, "w_style": 2, "w_atmosphere": 4,
        "q15_has_author": "No", "q15_author": "", "q16_focus": "Both",
        "q17_exclude": False, "q_standalone": False,
    },
}


def persona_state(name: str, blocks: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """État complet du questionnaire pour un persona (les cases non citées sont décochées)."""
    persona = DEMO_PERSONAS[name]
    state = default_state(blocks)
    for k, v in persona.items():
        if k == "genres":
            state.update({genre_key(g): g in v for g in blocks["Genre"]})
        elif k == "themes":
            state.update({theme_key(t): t in v for t in blocks["Themes"]})
        else:
            state[k] = v
    return state


# --------------------------------------------------------------------------- #
# 5. Validation et construction de answers.json
# --------------------------------------------------------------------------- #


def validate_step(step: int, state: Mapping[str, Any]) -> list[str]:
    """Erreurs bloquantes de l'étape `step` (0 à 4). Liste vide = on peut passer à la suite."""
    errors: list[str] = []
    if step == 0:
        q1 = str(state.get("q1_ideal", "")).strip()
        if len(q1) < MIN_Q1_CHARS:
            errors.append(f"Q1 is required: please write at least {MIN_Q1_CHARS} characters "
                          f"({len(q1)} so far). The more you describe, the better the semantic match.")
    elif step == 2:
        if not state.get("q9_formats"):
            errors.append("Q9: select at least one format.")
    elif step == 4:
        if state.get("q15_has_author") == "Yes" and not str(state.get("q15_author", "")).strip():
            errors.append("Q15: type the name of your favourite author, or answer « No ».")
    return errors


def validate_all(state: Mapping[str, Any]) -> list[str]:
    return [e for s in range(len(STEPS)) for e in validate_step(s, state)]


def build_answers(state: Mapping[str, Any], blocks: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Transforme l'état du formulaire en answers.json EXACTEMENT conforme au contrat (PLAN §4.3).

    Les textes sont nettoyés (espaces) ; le Likert est en entiers 1-5 ; les champs guidés
    non pertinents sont neutralisés (pas d'auteur -> favorite_author = "" et exclusion False).
    """
    has_author = state.get("q15_has_author") == "Yes"
    author = " ".join(str(state.get("q15_author", "")).split()) if has_author else ""
    return {
        "free_text": {
            ans_key: " ".join(str(state.get(sk, "")).split())
            for sk, (ans_key, *_rest) in FREE_TEXT_QUESTIONS.items()
        },
        "genres": [g for g in blocks["Genre"] if state.get(genre_key(g))],
        "themes": [t for t in blocks["Themes"] if state.get(theme_key(t))],
        "period": state.get("q7_period", NO_PREFERENCE),
        "audience": state.get("q8_audience", "Any"),
        "formats": list(state.get("q9_formats") or FORMATS),
        "favorite_author": author,
        "guided": {
            "focus": state.get("q16_focus", "Both"),
            "standalone_only": bool(state.get("q_standalone", False)),
            "exclude_favorite_author": bool(state.get("q17_exclude", False)) and bool(author),
        },
        "likert": {k: int(state.get(k, 3)) for k in LIKERT_KEYS},
    }


def save_answers(answers: Mapping[str, Any], folder: Path | str = config.DATA_RESPONSES) -> Path:
    """Sauvegarde horodatée data/responses/answers_YYYYmmdd_HHMMSS.json (consigne « structured format »)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = folder / f"answers_{stamp}.json"
    i = 1
    while path.exists():                     # deux envois dans la même seconde
        path = folder / f"answers_{stamp}_{i}.json"
        i += 1
    path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def state_from_answers(answers: Mapping[str, Any], blocks: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Inverse de build_answers : recharge un answers.json dans le formulaire (bouton « Load answers »)."""
    state = default_state(blocks)
    ft = answers.get("free_text", {})
    for sk, (ans_key, *_rest) in FREE_TEXT_QUESTIONS.items():
        state[sk] = ft.get(ans_key, "")
    state.update({genre_key(g): g in answers.get("genres", []) for g in blocks["Genre"]})
    state.update({theme_key(t): t in answers.get("themes", []) for t in blocks["Themes"]})
    if answers.get("period") in (NO_PREFERENCE, *ERA_LABELS):
        state["q7_period"] = answers["period"]
    if answers.get("audience") in AUDIENCES:
        state["q8_audience"] = answers["audience"]
    state["q9_formats"] = [f for f in answers.get("formats", FORMATS) if f in FORMATS] or list(FORMATS)
    for k in LIKERT_KEYS:
        state[k] = int(answers.get("likert", {}).get(k, 3))
    author = answers.get("favorite_author", "")
    state["q15_has_author"] = "Yes" if author else "No"
    state["q15_author"] = author
    guided = answers.get("guided", {})
    state["q16_focus"] = guided.get("focus", "Both") if guided.get("focus") in FOCUS_OPTIONS else "Both"
    state["q17_exclude"] = bool(guided.get("exclude_favorite_author", False))
    state["q_standalone"] = bool(guided.get("standalone_only", False))
    return state
