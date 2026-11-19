"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): die Messreihen neu gerechnet gegen `claims_expected.py` (die Werte des Baus),
dazu die Handrechnungen und die Aussagen der Befunde-Tabelle.

Plattformregeln: Ganzzahl-Rechnung (Shapley-Wert, Kerntest, Auktion, Ungarische Methode) ist auf jeder Plattform gleich und wird exakt verglichen.
Bänder gelten nur dort, wo eine C-Bibliothek oder ein Optimierer im Spiel ist: die Pareto-Verteilung (Potenz), die Eckenwahl des LP (Gegenprobe im Experiment 5)
und die Paargleichungen bei Gleichstand der besten Zuordnung."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import claims_expected as X
import zsp_constants as C
import zsp_core as K
import zsp_evaluation as ev
import zsp_experiments as EX
import zsp_game as G
from zsp_rng import uniform_block


@pytest.fixture(scope="module")
def e1():
    return {f"{r['dist']}|{r['k']}": r for r in EX.shapley_share_table()}


@pytest.fixture(scope="module")
def e2():
    return {f"{r['dist']}|{r['k']}": r for r in EX.mechanism_table()}


@pytest.fixture(scope="module")
def e5():
    return {f"{r['dist']}|{r['a']}x{r['b']}": r for r in EX.lattice_price_table()}


def _close(got, want, band):
    return abs(got - want) <= band


# --- Handrechnungen (Befunde der README, Tabelle 1) --------------------------------------------------------------------------------------------------------

def _facts(a, b, dist, seed=0):
    return ev.analyse(ev.Settings(a, b, dist, 100, seed))


def test_hand_one_load_two_trucks_core_is_a_point_and_shapley_splits():
    r = _facts(1, 2, "h_eins")
    assert [str(f) for f in r.phi_frac] == ["2/3", "1/6", "1/6"] and not r.in_core
    assert list(r.buyer) == [1.0, 0.0, 0.0] and list(r.seller) == [1.0, 0.0, 0.0]


def test_hand_two_loads_one_truck_numbers_and_the_midpoint_myth():
    r = _facts(2, 1, "h_spalte")
    assert [str(f) for f in r.phi_frac] == ["7/3", "4/3", "19/3"]
    low = np.minimum(r.buyer, r.seller)
    high = np.maximum(r.buyer, r.seller)
    assert list(low) == [0.0, 0.0, 8.0] and list(high) == [2.0, 0.0, 10.0]        # Kern: Ladung 1 0 bis 2, Ladung 2 genau 0, Lkw 8 bis 10
    assert [(i, j) for i, j, _ in r.blocking] == [(0, 0), (1, 0)]
    assert r.blocking[0][2] == pytest.approx(4 / 3) and r.blocking[1][2] == pytest.approx(1 / 3)
    midpoint = (r.buyer + r.seller) / 2
    assert list(midpoint) == [1.0, 0.0, 9.0] and not np.allclose(midpoint, r.phi)   # Shapley ist nicht der Mittelpunkt der Ecken


def test_hand_decoupled_three_by_three_halves_each_pair_and_is_in_the_core():
    r = _facts(3, 3, "h_diag")
    assert [str(f) for f in r.phi_frac] == ["5", "4", "3", "5", "4", "3"] and r.in_core and r.decoupled is True


def test_hand_rank_one_three_by_three_has_one_blocking_pair_of_gap_ninety_cents():
    r = _facts(3, 3, "h_rang")
    assert [float(f) for f in r.phi_frac[:3]] == pytest.approx([4.05, 640 / 300, 245 / 300])
    assert len(r.blocking) == 1 and r.blocking[0][:2] == (0, 0) and r.blocking[0][2] == pytest.approx(0.9)


def test_hand_two_separate_pairs_have_shapley_five_in_the_core_and_room_to_move():
    r = _facts(2, 2, "h_getrennt")
    assert [str(f) for f in r.phi_frac] == ["5"] * 4 and r.in_core and float(r.width.sum()) == 40.0


def test_hand_two_by_two_core_condition_difference_at_most_four():
    """[[10, 6], [6, 10]]: Kern |u1 - u2| <= 4, u in [0, 10] (Handrechnung der README): Ecken und ein Punkt am Rand."""
    wi = np.array([[1000, 600], [600, 1000]])
    inside = np.array([800, 400, 200, 600])                    # u = (8, 4), y = (2, 6): |u1 - u2| = 4, Paare 10 und 10, Kreuzpaare 8 + 6 = 14 >= 6 und 4 + 2 = 6 >= 6
    outside = np.array([900, 300, 100, 700])                   # |u1 - u2| = 6 > 4: Kreuzpaar (Ladung 2, Lkw 1) bekommt 3 + 1 = 4 < 6
    assert G.in_core_pairs(wi, inside) and not G.in_core_pairs(wi, outside)


def test_smallest_integer_squares_with_shapley_outside_the_core():
    total, outside = X.SMALL
    got = EX.small_squares()
    assert (got["total"], got["outside"]) == (total, outside) == (255, 120)


# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------------------

def test_e1_every_cell_matches_the_pinned_counts(e1):
    assert set(e1) == set(X.E1)
    for key, (n_in, n_total, n_zero, share, std, blocks) in X.E1.items():
        row = e1[key]
        band = 2 if key.startswith("pareto") else 0                    # Pareto: Potenz, nur hier ein Band
        assert _close(row["n_in"], n_in, band), key
        assert row["n_total"] + row["n_zero"] == n_total + n_zero and row["n_zero"] == n_zero, key
        assert row["share"] == pytest.approx(share, abs=0.011 if band else 1e-6), key
        assert row["blocking_mean"] == pytest.approx(blocks, abs=0.3 if band else 1e-3), key


def test_e1_stetig_share_at_k2_is_a_third_by_hand_and_by_count(e1):
    """Handrechnung: bei 2 x 2 und stetigen Werten sind „im Kern“ und „entkoppelt“ gleichbedeutend, und die beiden Einträge einer Zuordnung sind mit
    Wahrscheinlichkeit 1/3 die zwei größten von vier (das größte Element hat drei mögliche Partner, einer davon liegt in anderer Zeile und Spalte)."""
    n_in, n = X.UNIF2_BIG
    sigma = math.sqrt(n * (1 / 3) * (2 / 3))
    assert abs(n_in - n / 3) < 3 * sigma and n == 1600
    rows = EX.shapley_share_table(dists=["unif"], ks=(2,), per_seed={2: 400})
    assert (rows[0]["n_in"], rows[0]["n_total"]) == (n_in, n)
    assert e1["unif|2"]["n_in"] == 129 and e1["unif|3"]["n_in"] == 18


def test_e1_fast_never_from_k3_zero_cells_and_the_01_exception(e1):
    zero_cells = [k for k, r in e1.items() if int(k.split("|")[1]) >= 4 and r["n_in"] == 0]
    assert len(zero_cells) == 19 and len([k for k in e1 if int(k.split("|")[1]) >= 4]) == 24
    nonzero = {k: r["n_in"] for k, r in e1.items() if int(k.split("|")[1]) >= 4 and r["n_in"] > 0}
    assert set(nonzero) == {"int100|4", "int3|4", "int3|5", "bin|4", "pareto|4"}
    assert e1["bin|4"]["n_in"] == 30 and e1["bin|4"]["n_total"] == 200 and e1["bin|5"]["n_in"] == 0
    assert all(e1[f"prod|{k}"]["n_in"] == 0 for k in (2, 3, 4, 5, 6))              # Rang eins: in keinem der 1 140 Märkte


def test_e1_rank_one_is_never_decoupled():
    """Rang eins ist nie entkoppelt (die beste Zuordnung ist sortiert, ein Fremdpaar ist größer): 400 Märkte 3 x 3."""
    count = 0
    for run in range(100):
        for s in range(4):
            wi = EX.instance(EX.TAG_E1, "prod", 3, 3, s, run)
            assert K.decoupled(wi) is False
            count += 1
    assert count == 400


# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------------------

def test_e2_every_cell_matches_the_pinned_counts(e2):
    assert set(e2) == set(X.E2)
    for key, (n11, n10, n01, n00, eq_not_core, core_not_eq, n_total) in X.E2.items():
        row = e2[key]
        band = 2 if key.startswith("pareto") else 0
        assert _close(row["n11"], n11, band) and _close(row["n10"], n10, band) and _close(row["n01"], n01, band) and _close(row["n00"], n00, band), key
        assert row["n_total"] == n_total and row["core_not_eq"] == core_not_eq == 0, key
        assert row["eq_not_core"] <= eq_not_core + 2, key


def test_e2_in_the_core_implies_decoupled_except_two_integer_cases_at_k2(e2):
    exceptions = {k: r["n10"] for k, r in e2.items() if r["n10"] > 0}
    assert exceptions == {"int20|2": 2}
    pinned = X.E2.values()                                                       # die Summen der README stehen auf den gemessenen Zeilen (Pareto-Zellen mit Band im Test oben)
    assert sum(v[0] + v[1] for v in pinned) == 567 and sum(v[0] for v in pinned) == 565 and sum(v[6] for v in pinned) == 4000


def test_e2_continuous_k2_in_core_iff_decoupled_in_1600_of_1600():
    rows = EX.mechanism_table(dists=("unif",), ks=(2,), per_seed={2: 400})
    r = rows[0]
    assert [r["n11"], r["n10"], r["n01"], r["n00"]] == X.UNIF2_MECH == [521, 0, 0, 1079]


def test_e2_from_k3_decoupled_is_necessary_not_sufficient(e2):
    assert sum(v[2] for k, v in X.E2.items() if int(k.split("|")[1]) >= 3) == 95                          # entkoppelt, aber Shapley außerhalb (gemessene Zeilen)
    assert all(r["n10"] == 0 for k, r in e2.items() if int(k.split("|")[1]) >= 3)           # im Kern ohne entkoppelt: in keinem Markt ab k = 3
    assert e2["unif|3"]["n01"] == 24 and e2["unif|4"]["n01"] == 5


def test_core_always_satisfies_the_pair_equations_of_the_matching():
    """Im Kern gilt für jedes Paar der gefundenen besten Zuordnung u + y = w (Kern ohne Inneres): beide Ecken und fünf LP-Kernpunkte, 80 Märkte."""
    checked = 0
    for dist in ("int20", "unif", "geo", "bin"):
        for run in range(20):
            wi = EX.instance(11, dist, 3, 3, 0, run)
            if wi.sum() == 0:
                continue
            matching = G.optimal_matching(wi)
            buyer, seller = K.extreme_points(wi)
            for point in (buyer, seller):
                assert all(int(point[i] + point[3 + j]) == int(wi[i, j]) for i, j in matching)
            checked += 1
    assert checked >= 70


@pytest.mark.parametrize("key", list(X.DIMENSION))
def test_core_dimension_counts(key):
    dist, k = key.split("|")
    ok = cnt = 0
    for s in range(4):
        for run in range(8):
            wi = EX.instance(9, dist, int(k), int(k), s, run)
            if wi.sum() == 0:
                continue
            cnt += 1
            ok += K.core_dimension(wi / 100.0) == int(k)
    assert [ok, cnt] == X.DIMENSION[key]
    if dist == "unif":
        assert ok == cnt                                                 # stetig: Dimension genau k in jedem Markt (32 von 32 je Zelle)


# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------------------

def test_e3_density_table_matches_the_pinned_counts():
    rows = {str(r["q"]): r for r in EX.density_table()}
    assert set(rows) == set(X.E3A)
    for key, (n_in, n_total, n_zero, share) in X.E3A.items():
        assert (rows[key]["n_in"], rows[key]["n_total"], rows[key]["n_zero"]) == (n_in, n_total, n_zero), key
        assert rows[key]["share"] == pytest.approx(share, abs=1e-6)
    assert rows["10"]["share"] > 0.5 > 0.1 > rows["50"]["share"] and rows["100"]["n_in"] == 1


def test_e3_unequal_markets_never_have_shapley_in_the_core():
    rows = {f"{r['a']}x{r['b']}": r for r in EX.unequal_market_table()}
    assert set(rows) == set(X.E3B)
    for key, (n_in, n_total, with_zero, paid, blocks) in X.E3B.items():
        r = rows[key]
        assert (r["n_in"], r["n_total"], r["with_zero_max"], r["zero_paid"]) == (n_in, n_total, with_zero, paid), key
        assert r["blocking_mean"] == pytest.approx(blocks, abs=1e-3), key
    assert sum(r["n_total"] for r in rows.values()) == 1000 and sum(r["n_in"] for r in rows.values()) == 0
    assert sum(r["with_zero_max"] for r in rows.values()) == 1000 == sum(r["zero_paid"] for r in rows.values())


# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------------------

def test_e4_width_table_matches_the_pinned_values_and_the_lp_agrees():
    rows = {f"{r['dist']}|{r['k']}": r for r in EX.width_table()}
    assert set(rows) == set(X.E4)
    for key, (width, unique, n) in X.E4.items():
        assert rows[key]["width"] == pytest.approx(width, rel=1e-5) and rows[key]["unique"] == pytest.approx(unique, abs=1e-6) and rows[key]["n"] == n, key
    assert all(r["lp_dev"] < 1e-6 for r in rows.values() if r["lp_dev"] is not None)                 # Ecken gegen LP (k <= 20)
    assert rows["geo|80"]["width"] < 0.2 < 0.9 < min(rows["unif|80"]["width"], rows["int20|80"]["width"])


# --- Experiment 5 -------------------------------------------------------------------------------------------------------------------------------------

def test_e5_lattice_auction_and_hungarian_counts(e5):
    assert set(e5) == set(X.E5)
    for key, (n, pairs, join_ok, meet_ok, neg_in, auction_ok, rounds_mean, rounds_max, h_core, h_buyer, h_seller) in X.E5.items():
        r = e5[key]
        assert (r["n"], r["pairs"], r["join_ok"], r["meet_ok"]) == (n, pairs, pairs, pairs), key                    # Join und Meet immer im Kern
        assert 0 < r["neg_in"] < pairs and abs(r["neg_in"] - neg_in) <= 8, key                                       # Gegenprobe: nicht leer, nicht immer
        assert r["auction_ok"] == auction_ok == n and r["hung_core"] == h_core == n and r["hung_buyer"] == h_buyer == n and r["hung_seller"] == h_seller == n, key
        assert r["rounds_mean"] == pytest.approx(rounds_mean, abs=1e-3) and r["rounds_max"] == rounds_max, key
    assert sum(r["n"] for r in e5.values()) == 384 and sum(r["pairs"] for r in e5.values()) == 3840
    assert sum(r["neg_in"] for r in e5.values()) in range(600, 800)


def test_e5_round_counts_range_from_few_to_over_a_hundred(e5):
    assert e5["int20|3x3"]["rounds_mean"] == pytest.approx(3.0) and e5["int100|6x4"]["rounds_mean"] == pytest.approx(139.0)
    assert e5["int3|4x6"]["rounds_max"] == 2


# --- Texte und Konstanten -----------------------------------------------------------------------------------------------------------------------------

def test_constants_named_in_the_texts():
    assert C.SHAPLEY_MAX_PLAYERS == 14 and 2 ** C.SHAPLEY_MAX_PLAYERS == 16384
    assert C.A_MAX == 7 and C.B_MAX == 7 and C.EXP_SEEDS == 4 and C.E5_PAIRS_PER_INSTANCE == 10
    assert C.PARETO_CAP == 300.0 and C.E4_KS[-1] == 80 and C.AUCTION_ROUND_CAP == 5000


def test_hand_calculation_table_has_six_orders_and_the_documented_mean():
    hand = ev.hand_calculation()
    assert len(hand["orders"]) == 6 and hand["shapley"] == [F(7, 3), F(4, 3), F(19, 3)]
    assert all(sum(o["contrib"]) == 10 for o in hand["orders"])                # jede Reihenfolge verteilt genau v(N) = 10 €


def test_opposite_order_of_loads_and_trucks_on_comparable_core_points():
    """Verband: sind zwei Kernpunkte bei den Ladungen vergleichbar (u >= u' bei jeder Ladung), dann haben die Lkw die umgekehrte Ordnung (y <= y' bei jedem Lkw)."""
    comparable = opposite = markets = 0
    for dist in ("int20", "unif"):
        for a, b in ((3, 3), (4, 3), (3, 4)):
            for run in range(8):
                wi = EX.instance(12, dist, a, b, 0, run)
                if wi.sum() == 0:
                    continue
                markets += 1
                pts = EX._core_points(wi / 100.0, 1000 + run, 5)
                for i in range(len(pts)):
                    for j in range(len(pts)):
                        if i != j and (pts[i][:a] >= pts[j][:a] - 1e-9).all():
                            comparable += 1
                            opposite += bool((pts[i][a:] <= pts[j][a:] + 1e-7).all())
    assert markets >= 40 and comparable > markets and opposite == comparable


S_SEED = 5000                                                                   # Preise aus SplitMix64 (kein numpy.random für Befunde)


def test_core_points_are_price_systems_and_random_prices_are_not():
    """Kernpunkt ⇔ Preissystem: In jedem Kernpunkt ist u_i = max(0, max_j (w_ij - y_j)); bei zufälligen Preisen ist die Summe u + y meist größer als v(N)."""
    core_points = equal_total = random_prices = 0
    for dist in ("int20", "unif", "geo"):
        for run in range(10):
            wi = EX.instance(13, dist, 4, 4, 0, run)
            if wi.sum() == 0:
                continue
            w = wi / 100.0
            v_n = G.assignment_value(wi) / 100.0
            for x in EX._core_points(w, 2000 + run, 4) + list(K.extreme_points(wi)):
                x = np.asarray(x, dtype=float) / (100.0 if x.dtype.kind == "i" else 1.0)
                y = x[4:]
                u = np.maximum(0.0, (w - y[None, :]).max(axis=1))
                core_points += 1
                assert np.allclose(u, x[:4], atol=1e-7)
            for pick in range(6):
                y = uniform_block(S_SEED + 17 * run + pick, 4) * w.max() * 0.7
                u = np.maximum(0.0, (w - y[None, :]).max(axis=1))
                total = float(u.sum() + y.sum())
                assert total >= v_n - 1e-7                                  # schwache Dualität
                random_prices += 1
                equal_total += abs(total - v_n) < 1e-6
    assert core_points >= 150 and equal_total < random_prices / 4


