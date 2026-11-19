"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt Werte und ignoriert unbekannte Verteilungen."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import zsp_constants as C
import zsp_evaluation as ev
import zsp_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for preset in C.PRESETS.values():
        assert C.A_MIN <= preset["a"] <= C.A_MAX and C.B_MIN <= preset["b"] <= C.B_MAX
        assert C.Q_MIN <= preset["q"] <= C.Q_MAX and 0 <= preset["seed"] <= C.SEED_MAX and preset["dist"] in C.DISTS


def test_default_preset_is_the_first_and_equals_the_default_settings():
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_regler_maximum_a_plus_b_never_exceeds_the_shapley_limit():
    assert C.A_MAX + C.B_MAX <= C.SHAPLEY_MAX_PLAYERS


def test_permalink_clips_values_to_the_regler_bounds():
    at = _app(a="99", b="0", q="1", seed="99999999999")
    assert at.session_state["a_slider"] == C.A_MAX and at.session_state["b_slider"] == C.B_MIN
    assert at.session_state["q_slider"] == C.Q_MIN and at.session_state["seed_input"] == C.SEED_MAX


def test_permalink_ignores_unknown_distributions_and_garbage():
    at = _app(dist="gibt-es-nicht", a="abc")
    assert at.session_state["dist_select"] == C.DEFAULT_DIST and at.session_state["a_slider"] == C.DEFAULT_A


def test_permalink_loads_a_known_distribution_and_writes_the_settings_back():
    at = _app(dist="geo", a="2", b="5", q="40", seed="12")
    assert at.session_state["dist_select"] == "geo" and at.session_state["a_slider"] == 2 and at.session_state["b_slider"] == 5
    assert at.query_params["dist"] == "geo" and at.query_params["a"] == "2" and at.query_params["seed"] == "12"
