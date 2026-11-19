"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier; dazu laufen alle Presets und die Reglerränder durch."""

import pytest

import zsp_constants as C
import zsp_evaluation as ev


def test_de_and_pct_use_the_decimal_comma():
    assert ev.de(1234.5) == "1234,50" and ev.de(0.0833, 3) == "0,083" and ev.pct(0.3654, 1) == "36,5 %" and ev.pct(1 / 3, 0) == "33 %"


def test_shapley_core_verdict_without_value():
    kind, text = ev.verdict_shapley_core(2, 2, 0.0, True, 0, 0.0)
    assert kind == "warning" and "nichts zu verteilen" in text


def test_shapley_core_verdict_inside():
    kind, text = ev.verdict_shapley_core(3, 3, 24.0, True, 0, 0.0)
    assert kind == "success" and "3 Ladungen, 3 Lkw" in text and "liegt im Kern" in text


def test_shapley_core_verdict_outside_names_count_and_gap_and_singular():
    kind, text = ev.verdict_shapley_core(3, 3, 38.0, False, 3, 2.0833)
    assert kind == "warning" and "3 Paare" in text and "2,08 €" in text
    kind, text = ev.verdict_shapley_core(2, 1, 10.0, False, 1, 1.3333)
    assert "1 Paar " in text and "1 Paare" not in text and "1,33 €" in text


def test_core_span_verdict_branches():
    assert ev.verdict_core_span(0.0, 0.0, 0, 2)[0] == "warning"
    kind, text = ev.verdict_core_span(1.0, 0.0, 0, 3)
    assert kind == "info" and "einzigen Punkt" in text and "Alle 3 Spieler" in text
    kind, text = ev.verdict_core_span(38.0, 38.0, 6, 6)
    assert kind == "info" and "6 von 6 Spielern" in text and "38,00 €" in text


def test_slider_point_verdict():
    kind, text = ev.verdict_slider_point(True, 50, 0)
    assert kind == "success" and "bei 50 %" in text and "kein Paar blockiert" in text
    kind, text = ev.verdict_slider_point(False, 30, 2)
    assert kind == "error" and "NICHT im Kern" in text and "2 Paare" in text


def test_lattice_verdict():
    kind, text = ev.verdict_lattice(True, True, False, True)
    assert kind == "success" and "Join" in text and "liegt im Kern" in text and "Gegenprobe" in text and "liegt nicht im Kern" in text and "Was den Ladungen nützt" in text
    kind, text = ev.verdict_lattice(True, False, True, False)
    assert kind == "error" and "Meet liegt NICHT im Kern" in text and "liegt trotzdem im Kern" in text and "Was den Ladungen nützt" not in text


def test_auction_verdict_branches_and_rounding_note():
    kind, text = ev.verdict_auction(True, 13, True, False)
    assert kind == "success" and "nach 13 Runden" in text and "gerundet" not in text
    assert "ganze Euro gerundet" in ev.verdict_auction(True, 13, True, True)[1]
    kind, text = ev.verdict_auction(True, 4, False, False)
    assert kind == "error" and "nicht auf der Käuferecke" in text
    kind, text = ev.verdict_auction(False, C.AUCTION_ROUND_CAP, False, False)
    assert kind == "warning" and f"nach {C.AUCTION_ROUND_CAP} Runden" in text


def test_hungarian_verdict_branches():
    assert ev.verdict_hungarian("Ladungen", False, False, False)[0] == "warning"
    assert "einziger Punkt" in ev.verdict_hungarian("Lkw", True, True, True)[1]
    kind, text = ev.verdict_hungarian("Ladungen", True, True, False)
    assert kind == "success" and "Käuferecke" in text and "Ladungen als Zeilen" in text
    kind, text = ev.verdict_hungarian("Lkw", True, False, True)
    assert kind == "success" and "Verkäuferecke" in text and "Lkw als Zeilen" in text
    assert "an keiner der beiden Ecken" in ev.verdict_hungarian("Lkw", True, False, False)[1]


def _row(dist, k, n_in, n_total, n_zero=0):
    return {"dist": dist, "k": k, "share": n_in / n_total, "std": 0.0, "n_in": n_in, "n_total": n_total, "n_zero": n_zero, "blocking_mean": 0.0}


def test_share_verdict_names_range_zero_cells_and_the_exception():
    rows = [_row("a", 2, 10, 100), _row("b", 2, 70, 100), _row("a", 3, 1, 100), _row("b", 3, 40, 100),
            _row("a", 4, 0, 50), _row("b", 4, 5, 50), _row("a", 5, 0, 25), _row("b", 5, 0, 25)]
    kind, text = ev.verdict_share(rows)
    assert kind == "info"
    assert "Bei k = 2 liegt der Shapley-Wert in 10 % (a) bis 70 % (b)" in text and "bei k = 3 in 1 % (a) bis 40 % (b)" in text
    assert "Ab k = 4 sind es in 3 von 4 Zellen keine einzige Instanz" in text and "10,0 % bei b, k = 4" in text


def test_share_verdict_without_cells_from_k_four_has_no_zero_cell_sentence():
    rows = [_row("a", 2, 10, 100), _row("a", 3, 1, 100)]
    assert "Ab k = 4" not in ev.verdict_share(rows)[1]


def _mech(dist, k, n11, n10, n01, n00, eq_not_core=0, core_not_eq=0):
    return {"dist": dist, "k": k, "n11": n11, "n10": n10, "n01": n01, "n00": n00, "eq_not_core": eq_not_core, "core_not_eq": core_not_eq,
            "n_total": n11 + n10 + n01 + n00, "n_zero": 0}


def test_mechanism_verdict_counts():
    rows = [_mech("u", 2, 127, 0, 0, 273), _mech("i", 2, 161, 2, 0, 237, eq_not_core=2), _mech("u", 3, 27, 0, 24, 349)]
    kind, text = ev.verdict_mechanism(rows)
    assert kind == "info"
    assert "Von 1200 Märkten haben 317 den Shapley-Wert im Kern; 315 davon sind entkoppelt" in text
    assert "in 1 von 2 Verteilungen gleichbedeutend" in text and "24 entkoppelte Märkte" in text
    assert "0 Märkte; erfüllt und doch nicht im Kern: 2" in text


def test_density_verdict_trend_wording():
    def drow(q, n_in, n_total=100, n_zero=0):
        return {"q": q, "k": 4, "share": n_in / n_total, "std": 0.0, "n_in": n_in, "n_total": n_total, "n_zero": n_zero}
    strict = [drow(10, 60), drow(50, 20), drow(100, 1)]
    assert "sinkt in jedem Schritt" in ev.verdict_density(strict)[1]
    noisy = [drow(10, 60), drow(50, 0), drow(100, 1)]
    assert "Anstiege bis 1 Prozentpunkt sind bei dieser Stichprobe nicht von Rauschen zu unterscheiden" in ev.verdict_density(noisy)[1]
    up = [drow(10, 5), drow(50, 40), drow(100, 1)]
    assert "sinkt nicht durchgehend" in ev.verdict_density(up)[1]
    text = ev.verdict_density([drow(10, 61, 154, 46), drow(100, 1, 200, 0)])[1]
    assert "Dichte 10 % liegt der Shapley-Wert in 39,6 %" in text and "bei 100 % in 0,5 %" in text and "154 bzw. 200 Märkte" in text and "46 bzw. 0" in text


def test_unequal_verdict_counts():
    rows = [{"a": 1, "b": 2, "n_in": 0, "n_total": 200, "n_zero": 0, "with_zero_max": 200, "zero_paid": 190, "blocking_mean": 1.0},
            {"a": 2, "b": 3, "n_in": 3, "n_total": 200, "n_zero": 0, "with_zero_max": 180, "zero_paid": 100, "blocking_mean": 2.0}]
    text = ev.verdict_unequal(rows)[1]
    assert "In 3 von 400 ungleich großen Märkten" in text and "In 380 von 400 gibt es Spieler ohne Beitrag" in text and "in 290 davon zahlt" in text


def test_width_verdict_names_first_and_last_k_and_lp_deviation():
    rows = [{"dist": "unif", "k": 3, "width": 0.81, "unique": 0.0, "n": 12, "lp_dev": 1e-10}, {"dist": "unif", "k": 80, "width": 0.92, "unique": 0.0, "n": 12, "lp_dev": None},
            {"dist": "geo", "k": 3, "width": 0.73, "unique": 0.0, "n": 12, "lp_dev": 0.0}, {"dist": "geo", "k": 80, "width": 0.16, "unique": 0.0, "n": 12, "lp_dev": None}]
    text = ev.verdict_width(rows)[1]
    assert "unif: 0,81 (k = 3) → 0,92 (k = 80)" in text and "geo: 0,73 (k = 3) → 0,16 (k = 80)" in text and "1.0e-10 Hundertstel" in text


def _lat(**kw):
    base = {"dist": "int20", "a": 3, "b": 3, "n": 10, "pairs": 100, "join_ok": 100, "meet_ok": 100, "neg_in": 30, "auction_ok": 10, "rounds_mean": 3.0,
            "rounds_max": 9, "hung_core": 10, "hung_buyer": 10, "hung_seller": 10}
    base.update(kw)
    return base


def test_lattice_prices_verdict_success_and_warning():
    kind, text = ev.verdict_lattice_prices([_lat(), _lat()])
    assert kind == "success" and "Join im Kern: 200 von 200 Paaren, Meet: 200 von 200" in text and "nur in 60 von 200" in text
    assert "Die Auktion endet in 20 von 20 Märkten" in text
    assert ev.verdict_lattice_prices([_lat(auction_ok=9)])[0] == "warning"
    assert ev.verdict_lattice_prices([_lat(join_ok=99)])[0] == "warning"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_through_the_kernfrage_verdict_with_the_expected_outcome(name):
    """Ein Preset ohne Effekt meldet keinen Effekt: erwartet ist je Preset die Lage des Shapley-Werts (aus der Handrechnung bzw. der Messung)."""
    p = C.PRESETS[name]
    res = ev.analyse(ev.Settings(p["a"], p["b"], p["dist"], p["q"], p["seed"]))
    max_gap = max((g for _, _, g in res.blocking), default=0.0)
    kind, text = ev.verdict_shapley_core(res.a, res.b, res.v_n, res.in_core, len(res.blocking), max_gap)
    expected_inside = {"Entkoppelt 3 × 3 (Diagonale 10, 8, 6)", "Zwei getrennte Paare (10 und 10)"}
    if name in expected_inside:
        assert kind == "success" and res.in_core and res.blocking == []
    else:
        assert kind == "warning" and not res.in_core and len(res.blocking) >= 1 and "außerhalb des Kerns" in text


def test_preset_help_texts_name_the_measured_outcomes():
    """Die Zahlen in den Hilfetexten der Presets stimmen mit der Rechnung überein."""
    def run(name):
        p = C.PRESETS[name]
        return ev.analyse(ev.Settings(p["a"], p["b"], p["dist"], p["q"], p["seed"]))
    assert len(run("Zufallsmarkt 3 × 3 (0 bis 19)").blocking) == 3 and "drei Paare blockieren" in C.PRESET_HELP["Zufallsmarkt 3 × 3 (0 bis 19)"]
    assert len(run("Geografie 3 × 3").blocking) == 3 and "drei Paare blockieren" in C.PRESET_HELP["Geografie 3 × 3"]
    one = run("Eine Ladung, zwei Lkw")
    assert [str(f) for f in one.phi_frac] == ["2/3", "1/6", "1/6"] and float(one.width.sum()) == 0.0
    two = run("Zwei Ladungen, ein Lkw (10 und 8)")
    assert [str(f) for f in two.phi_frac] == ["7/3", "4/3", "19/3"]
    assert [round(float(x), 2) for x in two.buyer] == [2.0, 0.0, 8.0] and [round(float(x), 2) for x in two.seller] == [0.0, 0.0, 10.0]
    rank = run("Rang eins 3 × 3 (s = t = 3, 2, 1)")
    assert len(rank.blocking) == 1 and rank.blocking[0][2] == pytest.approx(0.9)
    assert [str(f) for f in run("Entkoppelt 3 × 3 (Diagonale 10, 8, 6)").phi_frac] == ["5", "4", "3", "5", "4", "3"]
    assert [str(f) for f in run("Zwei getrennte Paare (10 und 10)").phi_frac] == ["5"] * 4 and float(run("Zwei getrennte Paare (10 und 10)").width.sum()) == 40.0


@pytest.mark.parametrize("a,b", [(1, 1), (7, 7), (1, 7), (7, 1)])
@pytest.mark.parametrize("q", [C.Q_MIN, C.Q_MAX])
def test_kernfrage_verdict_runs_at_the_regler_edges(a, b, q):
    res = ev.analyse(ev.Settings(a, b, "int20", q, 4))
    max_gap = max((g for _, _, g in res.blocking), default=0.0)
    kind, text = ev.verdict_shapley_core(a, b, res.v_n, res.in_core, len(res.blocking), max_gap)
    assert kind in {"success", "warning"} and text
    assert ev.verdict_core_span(res.v_n, float(res.width.sum()), int((res.width > 1e-9).sum()), a + b)[1]
