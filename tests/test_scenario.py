"""Szenario: Zufall nur über SplitMix64, Wertebereiche je Verteilung, Dichte, Handbeispiele, Zellschlüssel (Zweig-Tests: jede Verteilung weicht von jeder anderen ab)."""

import numpy as np
import pytest

import zsp_constants as C
import zsp_scenario as S
from zsp_rng import uniform_block


def market(dist, a=4, b=4, q=100, seed=3):
    return np.array(S.make_market(a, b, dist, q, seed))


def test_same_input_gives_the_same_market_and_another_seed_another_market():
    for dist in C.RANDOM_DISTS:
        assert S.make_market(4, 4, dist, 100, 3) == S.make_market(4, 4, dist, 100, 3)
        assert S.make_market(4, 4, dist, 100, 3) != S.make_market(4, 4, dist, 100, 4)


def test_value_ranges_per_distribution():
    for seed in range(20):
        i20, i100, i3, bn = (market(d, seed=seed) for d in ("int20", "int100", "int3", "bin"))
        for m, top in ((i20, 19), (i100, 99), (i3, 3), (bn, 1)):
            assert (m % 100 == 0).all() and m.min() >= 0 and m.max() <= top * 100
        u = market("unif", seed=seed)
        assert u.min() >= 0 and u.max() <= 10000
        p = market("pareto", seed=seed)
        assert p.min() >= 1000 and p.max() <= 30000                      # ab 10 €, bei 300 € gekappt
        r = market("prod", seed=seed)
        assert r.min() >= 1250 and r.max() <= 11250                      # 50 * (0,5 bis 1,5)^2 = 12,50 bis 112,50 €
        g = market("geo", seed=seed)
        assert g.min() >= 0 and g.max() <= 10000


def test_integer_distributions_use_every_value_and_unif_is_not_integer_valued():
    values = np.concatenate([market("int20", 7, 7, seed=s).ravel() for s in range(20)]) // 100
    assert set(values.tolist()) == set(range(20))
    assert set((np.concatenate([market("int3", 7, 7, seed=s).ravel() for s in range(5)]) // 100).tolist()) == {0, 1, 2, 3}
    assert (market("unif", 7, 7, seed=1) % 100 != 0).any()


def test_rank_one_distribution_is_an_outer_product_up_to_rounding():
    m = market("prod", 5, 4, seed=9) / 100.0
    ratio = m / m[0:1, :]
    assert np.allclose(ratio, ratio[:, 0:1], rtol=2e-3)                  # Zeilen sind (bis auf Rundung auf Hundertstel) Vielfache der ersten


def test_every_distribution_differs_from_every_other_on_the_same_seed():
    names = list(C.RANDOM_DISTS)
    mats = {d: market(d, 5, 5, seed=2) for d in names}
    for i, d1 in enumerate(names):
        for d2 in names[i + 1:]:
            assert not np.array_equal(mats[d1], mats[d2]), (d1, d2)


def test_density_thins_the_market_and_keeps_the_full_values_where_kept():
    full, thin = market("unif", 6, 6, 100, 5), market("unif", 6, 6, 30, 5)
    assert (thin != 0).sum() < (full != 0).sum()
    assert ((thin == 0) | (thin == full)).all()                          # was bleibt, behält seinen Wert: nur die Maske ändert sich
    share = np.mean([(market("unif", 7, 7, 30, s) != 0).mean() for s in range(60)])
    assert abs(share - 0.30) < 0.04


def test_hand_examples_have_the_documented_values():
    assert market("h_eins", 1, 2).tolist() == [[100, 100]]
    assert market("h_spalte", 7, 2)[:, 0].tolist() == [1000, 800, 600, 400, 200, 0, 0]
    assert market("h_diag", 3, 3).tolist() == [[1000, 100, 100], [100, 800, 100], [100, 100, 600]]
    assert np.diag(market("h_diag", 7, 7)).tolist() == [1000, 800, 600, 400, 200, 200, 200]
    assert market("h_rang", 3, 3).tolist() == (np.outer([3, 2, 1], [3, 2, 1]) * 100).tolist()
    assert market("h_getrennt", 2, 3).tolist() == [[1000, 0, 0], [0, 1000, 0]]


def test_pareto_cap_probability_is_about_six_per_thousand():
    p_cap = (1 / 30) ** 1.5                                              # P(10 (1-u)^(-2/3) > 300)
    assert p_cap == pytest.approx(0.00609, abs=1e-5)
    u = uniform_block(5, 200_000)
    share = float((10.0 * (1.0 - u) ** (-1.0 / 1.5) > C.PARETO_CAP).mean())
    assert 0.004 < share < 0.009


def test_cell_seed_is_reproducible_and_distinguishes_every_key_part():
    assert S.cell_seed(1, 2, 3) == S.cell_seed(1, 2, 3)
    keys = {S.cell_seed(t, d, k, s, r) for t in (1, 2) for d in (0, 1) for k in (2, 3) for s in (0, 1) for r in (0, 1)}
    assert len(keys) == 32
    assert S.cell_seed(1, 2) != S.cell_seed(2, 1)


def test_scenario_uses_no_global_random_module():
    source = open(S.__file__, encoding="utf-8").read()
    assert "import random" not in source and "numpy.random" not in source and "np.random" not in source
