"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- ergänzt für das Zuordnungsspiel -------------------------------------------------------------------------------------------------------------------

import pytest  # noqa: E402

import zsp_constants as C  # noqa: E402

EXPERIMENT_BUTTONS = ["Messreihe 1 rechnen", "Messreihe 2 rechnen", "Messreihe 3 rechnen", "Messreihe 4 rechnen", "Messreihe 5 rechnen"]


def _texts(at):
    return " ".join(m.value for m in list(at.success) + list(at.info) + list(at.warning) + list(at.error))


def _kernfrage(at):
    return next(m for m in list(at.success) + list(at.warning) if m.value.startswith("Der Shapley-Wert") or m.value.startswith("Alle Paarwerte sind 0"))


@pytest.mark.parametrize("label", EXPERIMENT_BUTTONS)
def test_each_experiment_button_runs_and_shows_a_verdict(label):
    at = _fresh_app()
    at.button(key=f"btn_exp{label.split()[1]}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Rechenzeit" in " ".join(c.value for c in at.caption)
    assert f"exp{label.split()[1]}" in at.session_state


def test_experiment_results_stay_after_another_widget_changes():
    at = _fresh_app()
    at.button(key="btn_exp3").click().run(timeout=TIMEOUT)
    at.slider(key="t_slider").set_value(80).run(timeout=TIMEOUT)
    assert not at.exception and "exp3" in at.session_state
    assert any("ungleich großen Märkten" in m.value for m in at.info)


def test_every_distribution_runs_through_the_app():
    for dist in C.DISTS:
        at = _fresh_app()
        at.selectbox(key="dist_select").set_value(dist).run(timeout=TIMEOUT)
        assert not at.exception, (dist, [str(e) for e in at.exception])


def test_hand_examples_gray_out_seed_and_density_and_random_ones_do_not():
    at = _fresh_app()
    assert not at.sidebar.slider(key="q_slider").disabled and not at.sidebar.number_input(key="seed_input").disabled
    at.selectbox(key="dist_select").set_value("h_diag").run(timeout=TIMEOUT)
    assert at.sidebar.slider(key="q_slider").disabled and at.sidebar.number_input(key="seed_input").disabled
    assert next(b for b in at.sidebar.button if b.label.startswith("🎲")).proto.disabled


def test_seed_number_input_at_its_minimum_and_maximum():
    for value in (0, C.SEED_MAX):
        at = _fresh_app()
        at.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception


def test_all_minima_and_all_maxima_together():
    for a, b, q in ((C.A_MIN, C.B_MIN, C.Q_MIN), (C.A_MAX, C.B_MAX, C.Q_MAX)):
        for dist in ("int20", "unif", "pareto"):
            at = _fresh_app()
            at.slider(key="a_slider").set_value(a)
            at.slider(key="b_slider").set_value(b)
            at.slider(key="q_slider").set_value(q)
            at.selectbox(key="dist_select").set_value(dist)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (a, b, q, dist, [str(e) for e in at.exception])


def test_price_slider_extremes_and_other_choice_widgets():
    at = _fresh_app()
    for t in (0, 100):
        at.slider(key="t_slider").set_value(t).run(timeout=TIMEOUT)
        assert not at.exception
        assert any("liegt im Kern" in m.value and f"bei {t} %" in m.value for m in at.success)
    for rows in ("Ladungen", "Lkw"):
        at.radio(key="hung_rows").set_value(rows).run(timeout=TIMEOUT)
        assert not at.exception
        assert any(f"Mit {rows} als Zeilen" in m.value for m in at.success)
    for pick in (0, 999):
        at.number_input(key="pick_input").set_value(pick).run(timeout=TIMEOUT)
        assert not at.exception


def test_default_shows_kernfrage_outside_the_core_and_the_expected_verdicts():
    at = _fresh_app()
    assert "außerhalb des Kerns" in _kernfrage(at).value
    text = _texts(at)
    assert "Die Auktion endet nach" in text and "Join" in text and "Ungarische Methode" in text


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_its_kernfrage_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    inside = name in {"Entkoppelt 3 × 3 (Diagonale 10, 8, 6)", "Zwei getrennte Paare (10 und 10)"}
    assert ("liegt im Kern" in _kernfrage(at).value) == inside


def test_wertloser_markt_zeigt_den_hinweis_statt_eines_leeren_diagramms():
    at = _fresh_app()
    at.slider(key="a_slider").set_value(1)
    at.slider(key="b_slider").set_value(1)
    at.slider(key="q_slider").set_value(C.Q_MIN)
    at.selectbox(key="dist_select").set_value("int3")
    for seed in range(60):
        at.number_input(key="seed_input").set_value(seed).run(timeout=TIMEOUT)
        if _kernfrage(at).value.startswith("Alle Paarwerte sind 0"):
            assert any("Kein Paar hat einen Wert über 0" in m.value for m in at.info)
            return
    raise AssertionError("kein wertloser Markt gefunden")
