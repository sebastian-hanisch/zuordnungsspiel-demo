"""Einheiten von `zsp_game`, `zsp_core`, `zsp_auction` an den von Hand gerechneten Mini-Instanzen (DEMO-PLAYBOOK §1b)."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import zsp_auction as AU
import zsp_core as K
import zsp_game as G
from conftest import arr


def shapley_fractions(wi):
    """Shapley-Wert als Brüche in Euro."""
    a, b = wi.shape
    n = a + b
    return [F(int(x), math.factorial(n) * 100) for x in G.shapley_scaled(n, G.value_table(wi))]


def test_value_table_one_load_two_trucks(one_load_two_trucks):
    v = G.value_table(one_load_two_trucks)
    assert list(v) == [0, 0, 0, 100, 0, 100, 0, 100]        # Bits: Ladung, Lkw 1, Lkw 2; nur Koalitionen mit der Ladung und mindestens einem Lkw haben den Wert 1 €


def test_shapley_one_load_two_trucks(one_load_two_trucks):
    assert shapley_fractions(one_load_two_trucks) == [F(2, 3), F(1, 6), F(1, 6)]


def test_kern_of_one_load_two_trucks_is_a_single_point(one_load_two_trucks):
    buyer, seller = K.extreme_points(one_load_two_trucks)
    assert list(buyer) == [100, 0, 0] and list(seller) == [100, 0, 0]
    wi = one_load_two_trucks
    assert not G.in_core_pairs(wi, G.shapley_scaled(3, G.value_table(wi)), math.factorial(3))


def test_shapley_two_loads_one_truck(two_loads_one_truck):
    assert shapley_fractions(two_loads_one_truck) == [F(7, 3), F(4, 3), F(19, 3)]


def test_corners_two_loads_one_truck(two_loads_one_truck):
    buyer, seller = K.extreme_points(two_loads_one_truck)
    assert list(buyer) == [200, 0, 800] and list(seller) == [0, 0, 1000]


def test_blocking_pairs_two_loads_one_truck(two_loads_one_truck):
    wi = two_loads_one_truck
    phi = G.shapley_scaled(3, G.value_table(wi))
    pairs = G.blocking_pairs(wi, phi, math.factorial(3))
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 0)]
    assert pairs[0][2] == pytest.approx(400 / 3) and pairs[1][2] == pytest.approx(100 / 3)       # Lücken in Hundertsteln: 4/3 € und 1/3 €


def test_shapley_square_2x2_is_half_the_pair_value_in_the_core(square_2x2):
    assert shapley_fractions(square_2x2) == [F(5)] * 4
    wi = square_2x2
    assert G.in_core_pairs(wi, G.shapley_scaled(4, G.value_table(wi)), math.factorial(4))
    buyer, seller = K.extreme_points(wi)
    assert list(buyer) == [1000, 1000, 0, 0] and list(seller) == [0, 0, 1000, 1000]


def test_diagonal_pairs_have_shapley_five(diagonal_2x2):
    assert shapley_fractions(diagonal_2x2) == [F(5)] * 4


def test_decoupled_market_has_shapley_in_the_core(decoupled_3x3):
    assert shapley_fractions(decoupled_3x3) == [F(5), F(4), F(3), F(5), F(4), F(3)]
    assert K.decoupled(decoupled_3x3) is True


def test_rank_one_market_has_shapley_outside_the_core(rank_one_3x3):
    wi = rank_one_3x3
    phi = shapley_fractions(wi)
    assert phi[0] == F(405, 100) and phi[1] == F(640, 300) and phi[2] == F(245, 300)
    pairs = G.blocking_pairs(wi, G.shapley_scaled(6, G.value_table(wi)), math.factorial(6))
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)] and pairs[0][2] == pytest.approx(90.0)
    assert K.decoupled(wi) is False


def test_value_table_covers_every_coalition_of_the_two_by_two(square_2x2):
    v = G.value_table(square_2x2)
    # Bits: 0, 1 = Ladungen, 2, 3 = Lkw. Beispiel: Ladung 1 + Lkw 1 = 10, Ladung 1 + Lkw 2 = 6, alle vier = 20
    assert v[0b0101] == 1000 and v[0b1001] == 600 and v[0b1111] == 2000 and v[0b0011] == 0 and v[0b1100] == 0


def test_in_core_pairs_accepts_corners_and_rejects_a_shifted_point(square_2x2):
    wi = square_2x2
    buyer, seller = K.extreme_points(wi)
    assert G.in_core_pairs(wi, buyer) and G.in_core_pairs(wi, seller)
    shifted = buyer.copy()
    shifted[0] -= 1
    shifted[1] += 1                                   # Summe bleibt, aber Ladung 1 bekommt weniger als nötig: Paar (1, 1) blockiert
    assert not G.in_core_pairs(wi, shifted)


def test_in_core_pairs_rejects_a_point_with_wrong_total(square_2x2):
    assert not G.in_core_pairs(square_2x2, np.array([1000, 1000, 0, 1]))


def test_pair_equations_hold_in_the_core_and_fail_outside(one_load_two_trucks, square_2x2):
    wi = square_2x2
    buyer, _ = K.extreme_points(wi)
    assert G.pair_equations_hold(wi, buyer)
    wi1 = one_load_two_trucks
    assert not G.pair_equations_hold(wi1, G.shapley_scaled(3, G.value_table(wi1)), math.factorial(3))


def test_join_and_meet_of_the_two_corners_are_the_corners(square_2x2):
    wi = square_2x2
    buyer, seller = K.extreme_points(wi)
    join, meet = K.join_meet(buyer.astype(float), seller.astype(float), 2)
    assert list(join) == list(buyer) and list(meet) == list(seller)


def test_negative_probe_is_the_componentwise_maximum(square_2x2):
    buyer, seller = K.extreme_points(square_2x2)
    assert list(K.negative_probe(buyer, seller, 2)) == [1000, 1000, 1000, 1000]          # überschreitet v(N) = 2000


def test_decoupled_needs_a_square_market(two_loads_one_truck):
    assert K.decoupled(two_loads_one_truck) is None


@pytest.mark.parametrize("rows", [[[0, 3], [3, 4]], [[18, 15], [15, 4]], [[13, 9], [14, 13]], [[3, 10, 9], [7, 2, 4], [5, 9, 0]]])
def test_decoupled_is_strict_with_ties_shapley_can_be_in_the_core_without_it(rows):
    """Gleichstände: Shapley im Kern, aber nicht „entkoppelt“ (Handrechnung zu [[0, 3], [3, 4]]: v(N) = 6 mit den Einträgen 3 und 3, Shapley (1; 2 | 1; 2)
    liegt im Kern: 1 + 2 = 3, 2 + 1 = 3, 2 + 2 = 4 reichen für alle Paare; aber die beiden Einträge 3 sind kein Zeilen- und Spaltenmaximum). Die Definition bleibt streng, die beste
    Zuordnung hat in jedem dieser Fälle zwei gleiche Einträge (README, Mechanismus)."""
    wi = arr(rows)
    n = 2 * len(rows)
    assert G.in_core_pairs(wi, G.shapley_scaled(n, G.value_table(wi)), math.factorial(n))
    assert K.decoupled(wi) is False
    k = len(rows)
    best = max(sum(rows[i][p[i]] for i in range(k)) for p in itertools.permutations(range(k)))
    entries = [sorted(rows[i][p[i]] for i in range(k)) for p in itertools.permutations(range(k)) if sum(rows[i][p[i]] for i in range(k)) == best]
    assert all(len(set(e)) < len(e) for e in entries)               # Gleichstand unter den Einträgen jeder besten Zuordnung


def test_decoupled_tie_example_hand_values():
    wi = arr([[0, 3], [3, 4]])
    assert [str(f) for f in shapley_fractions(wi)] == ["1", "2", "1", "2"]    # Summe 6 = v(N); u + y = 3, 3, 4 für die Paare (L0,T1), (L1,T0), (L1,T1), 2 für (L0,T0) >= 0


def test_ascending_auction_on_the_two_by_two(square_2x2):
    res = AU.ascending_auction(square_2x2 // 100)
    assert res["converged"] and list(res["price"]) == [0, 0] and res["rounds"] == 0        # niemand ist überfragt: jede Ladung hat einen eigenen besten Lkw


def test_ascending_auction_two_loads_one_truck(two_loads_one_truck):
    res = AU.ascending_auction(two_loads_one_truck // 100)
    # beide Ladungen wollen den Lkw: Preis steigt auf 8, dann fragt nur noch Ladung 1 nach
    assert res["converged"] and list(res["price"]) == [8] and res["rounds"] == 8
    assert [t["set"] for t in res["trace"]] == [[0]] * 8


def test_hungarian_potentials_dual_feasible_and_tight_on_the_matching():
    cost = np.array([[-10, -6], [-6, -10]])
    u, v = AU.hungarian_potentials(cost)
    assert (u[:, None] + v[None, :] <= cost).all()
    assert u.sum() + v.sum() == -20


def test_hungarian_prices_rows_swap_picks_the_other_corner(two_loads_one_truck):
    wi = two_loads_one_truck
    buyer, seller = K.extreme_points(wi)
    assert list(AU.hungarian_prices(wi, "Ladungen")) == list(buyer)
    assert list(AU.hungarian_prices(wi, "Lkw")) == list(seller)
    assert not np.array_equal(buyer, seller)


def test_core_dimension_hand_cases(one_load_two_trucks, square_2x2, diagonal_2x2):
    assert K.core_dimension(one_load_two_trucks / 100.0) == 0           # Kern ist ein Punkt
    assert K.core_dimension(square_2x2 / 100.0) == 2                    # k = 2 Paargleichungen, n - 2 = 2
    assert K.core_dimension(diagonal_2x2 / 100.0) == 2


def test_hand_values_in_arr_helper():
    assert arr([[1, 2]]).tolist() == [[100, 200]]
