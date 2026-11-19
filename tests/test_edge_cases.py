"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1 und n = 2, Gleichstände, v(N) = 0, a ≠ b, Reglergrenzen."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import zsp_auction as AU
import zsp_constants as C
import zsp_core as K
import zsp_evaluation as ev
import zsp_game as G
import zsp_scenario as S
from conftest import arr


def _shapley(wi):
    n = sum(wi.shape)
    return [F(int(x), math.factorial(n) * 100) for x in G.shapley_scaled(n, G.value_table(wi))]


def test_one_by_one_market_splits_the_pair_value_in_the_middle():
    """n = 2 (kleinster Markt): Shapley 5 und 5, Kern (a, 10 - a), Ecken (10, 0) und (0, 10), Dimension 1."""
    wi = arr([[10]])
    assert _shapley(wi) == [F(5), F(5)]
    buyer, seller = K.extreme_points(wi)
    assert list(buyer) == [1000, 0] and list(seller) == [0, 1000]
    assert G.in_core_pairs(wi, G.shapley_scaled(2, G.value_table(wi)), 2)
    assert K.core_dimension(wi / 100.0) == 1
    assert K.decoupled(wi) is True


def test_one_by_one_market_without_value():
    """Paarwert 0: nichts zu verteilen, alles ist stabil, kein Fehler in Ecken, Auktion, Ungarischer Methode."""
    wi = arr([[0]])
    assert _shapley(wi) == [F(0), F(0)]
    assert list(K.extreme_points(wi)[0]) == [0, 0]
    assert G.optimal_matching(wi) == []
    res = AU.ascending_auction(wi // 100)
    assert res["converged"] and res["rounds"] == 0
    assert list(AU.hungarian_prices(wi, "Ladungen")) == [0, 0] and list(AU.hungarian_prices(wi, "Lkw")) == [0, 0]
    assert G.in_core_pairs(wi, np.array([0, 0]))


def test_market_without_any_value_is_handled_in_every_unit():
    """v(N) = 0 (bei Dichte 5 % ist fast jeder 1 x 1 Markt wertlos): Kern ist der Nullpunkt, Shapley 0, nichts blockiert, kein Fehler."""
    zero_seeds = [seed for seed in range(40) if ev.analyse(ev.Settings(1, 1, "int3", 5, seed)).v_n == 0]
    assert len(zero_seeds) > 20                                     # der Pfad v(N) = 0 wird wirklich durchlaufen
    r = ev.analyse(ev.Settings(1, 1, "int3", 5, zero_seeds[0]))
    assert r.in_core and r.blocking == [] and float(r.width.sum()) == 0.0 and float(np.abs(r.phi).sum()) == 0.0
    assert r.matching == [] and r.decoupled is True
    kind, text = ev.verdict_shapley_core(1, 1, r.v_n, r.in_core, len(r.blocking), 0.0)
    assert kind == "warning" and "nichts zu verteilen" in text


def test_ties_two_optimal_matchings_give_the_same_corners_and_core_verdict():
    """Gleichstand: [[1, 1], [1, 1]] hat zwei optimale Zuordnungen. Ecken und Urteil hängen nicht an der gewählten."""
    wi = arr([[1, 1], [1, 1]])
    buyer, seller = K.extreme_points(wi)
    assert list(buyer) == [100, 100, 0, 0] and list(seller) == [0, 0, 100, 100]
    assert _shapley(wi) == [F(1, 2)] * 4
    assert G.in_core_pairs(wi, G.shapley_scaled(4, G.value_table(wi)), math.factorial(4))
    flipped = wi[::-1, :]
    assert list(K.extreme_points(flipped)[0]) == list(buyer)


def test_exactly_two_loads_and_two_trucks_with_one_empty_row():
    """n = 4, eine Ladung ohne jeden Wert: sie bekommt in Shapley und Kern 0."""
    wi = arr([[0, 0], [5, 3]])
    v = G.value_table(wi)
    assert _shapley(wi)[0] == F(0)
    assert K.extreme_points(wi)[0][0] == 0 and K.extreme_points(wi)[1][0] == 0
    assert int(v[-1]) == 500


@pytest.mark.parametrize("a,b", [(1, 2), (2, 1), (1, 7), (7, 1), (2, 5), (5, 2)])
def test_unequal_markets_both_directions_run_through_every_unit(a, b):
    for dist in C.DISTS:
        res = ev.analyse(ev.Settings(a, b, dist, 100, 3))
        assert res.buyer.shape == (a + b,) and res.seller.shape == (a + b,)
        assert np.allclose(res.phi.sum(), res.v_n)
        assert G.in_core_pairs(res.wi, np.rint(res.buyer * 100).astype(np.int64)) and G.in_core_pairs(res.wi, np.rint(res.seller * 100).astype(np.int64))
        assert AU.ascending_auction(np.rint(res.wi / 100).astype(np.int64))["converged"]
        for rows in ("Ladungen", "Lkw"):
            AU.hungarian_prices(res.wi, rows)


def test_regler_maximum_seven_by_seven_runs_with_every_distribution():
    """Regler-Maximum a = b = 7 (n = 14): exakter Shapley-Wert über 16 384 Koalitionen, Summe stimmt exakt."""
    for dist in C.DISTS:
        res = ev.analyse(ev.Settings(7, 7, dist, 100, 11))
        assert float(res.phi.sum()) == pytest.approx(res.v_n, rel=1e-9)
        assert sum(res.phi_frac) == F(int(round(res.v_n * 100)), 100)


def test_density_at_both_ends_and_seed_at_both_ends():
    for q in (C.Q_MIN, C.Q_MAX):
        for seed in (0, C.SEED_MAX):
            res = ev.analyse(ev.Settings(4, 4, "unif", q, seed))
            assert res.wi.shape == (4, 4)
    full = np.array(S.make_market(4, 4, "unif", C.Q_MAX, 5))
    thin = np.array(S.make_market(4, 4, "unif", C.Q_MIN, 5))
    assert (thin != 0).sum() < (full != 0).sum()                    # die Dichte wirkt: dünn hat weniger Paare mit Wert


def test_hand_examples_do_not_depend_on_seed_or_density():
    for dist in C.HAND_DISTS:
        assert S.make_market(4, 3, dist, 100, 0) == S.make_market(4, 3, dist, 5, 999999)


def test_unknown_distribution_is_an_error():
    with pytest.raises(ValueError):
        S.make_market(2, 2, "gibt-es-nicht", 100, 0)


def test_blocking_pairs_never_include_a_satisfied_pair():
    wi = arr([[10, 6], [6, 10]])
    assert G.blocking_pairs(wi, np.array([500, 500, 500, 500]), 1) == []
    assert [(i, j) for i, j, _ in G.blocking_pairs(wi, np.array([400, 500, 500, 500]), 1)] == [(0, 0)]
