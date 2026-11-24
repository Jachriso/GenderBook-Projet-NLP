"""Tests for the questionnaire (app/questions.py, P5): answers.json follows the contract in PLAN §4.3.

Run: python -m pytest tests/test_app_questions.py -q
"""
import json

import pytest

from app import questions as Q

BLOCKS = Q.load_blocks()
CONTRACT_KEYS = {"free_text", "genres", "themes", "period", "audience", "formats",
                 "favorite_author", "guided", "likert"}


@pytest.mark.parametrize("persona", list(Q.DEMO_PERSONAS))
def test_personas_produce_valid_contract(persona):
    state = Q.persona_state(persona, BLOCKS)
    assert Q.validate_all(state) == []
    ans = Q.build_answers(state, BLOCKS)
    assert set(ans) == CONTRACT_KEYS
    assert set(ans["free_text"]) == {"ideal_book", "loved_book", "atmosphere", "avoid"}
    assert set(ans["guided"]) == {"focus", "standalone_only", "exclude_favorite_author"}
    assert set(ans["likert"]) == {"complexity", "pace", "darkness", "popularity",
                                  "w_genre", "w_themes", "w_style", "w_atmosphere"}
    assert all(1 <= v <= 5 for v in ans["likert"].values())
    assert set(ans["genres"]) <= set(BLOCKS["Genre"])
    assert set(ans["themes"]) <= set(BLOCKS["Themes"])
    assert ans["period"] in (Q.NO_PREFERENCE, *BLOCKS["Atmosphere"])
    json.dumps(ans)                                   # JSON-serialisable


def test_era_labels_exist_in_blocks():
    assert set(Q.ERA_LABELS) <= set(BLOCKS["Atmosphere"])


def test_q1_required():
    state = Q.default_state(BLOCKS)
    assert Q.validate_step(0, state)
    state["q1_ideal"] = "x" * Q.MIN_Q1_CHARS
    assert Q.validate_step(0, state) == []


def test_no_author_means_no_exclusion():
    state = Q.default_state(BLOCKS)
    state.update(q1_ideal="a" * 40, q15_has_author="No", q15_author="Someone", q17_exclude=True)
    ans = Q.build_answers(state, BLOCKS)
    assert ans["favorite_author"] == "" and ans["guided"]["exclude_favorite_author"] is False


def test_round_trip_answers_state():
    state = Q.persona_state("Dark academia reader", BLOCKS)
    ans = Q.build_answers(state, BLOCKS)
    assert Q.build_answers(Q.state_from_answers(ans, BLOCKS), BLOCKS) == ans


def test_save_answers(tmp_path):
    ans = Q.build_answers(Q.persona_state("Young fantasy fan", BLOCKS), BLOCKS)
    p1, p2 = Q.save_answers(ans, tmp_path), Q.save_answers(ans, tmp_path)
    assert p1 != p2 and json.loads(p1.read_text(encoding="utf-8")) == ans
