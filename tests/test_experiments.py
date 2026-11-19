"""Messreihen-Funktionen: Struktur, Zählbilanz, Determinismus, Unabhängigkeit der Zellen (die gemessenen Werte pinnt test_claims.py)."""

import numpy as np

import zsp_constants as C
import zsp_evaluation as ev
import zsp_experiments as EX


def test_share_table_bookkeeping_and_determinism():
    kw = dict(dists=["int20", "bin"], ks=(2, 3), per_seed={2: 20, 3: 10, 4: 1, 5: 1, 6: 1})
    rows = EX.shapley_share_table(**kw)
    assert rows == EX.shapley_share_table(**kw)
    assert [(r["dist"], r["k"]) for r in rows] == [("int20", 2), ("int20", 3), ("bin", 2), ("bin", 3)]
    for r in rows:
        assert r["n_total"] + r["n_zero"] == C.EXP_SEEDS * kw["per_seed"][r["k"]]
        assert 0 <= r["n_in"] <= r["n_total"] and r["std"] >= 0.0


def test_instance_cells_are_independent_of_the_order_of_calls():
    first = EX.instance(1, "unif", 3, 3, 2, 5)
    EX.instance(1, "unif", 3, 3, 2, 6)
    assert np.array_equal(first, EX.instance(1, "unif", 3, 3, 2, 5))
    assert not np.array_equal(first, EX.instance(1, "unif", 3, 3, 2, 6)) and not np.array_equal(first, EX.instance(2, "unif", 3, 3, 2, 5))


def test_shapley_facts_agree_with_the_analysis_of_the_same_market():
    wi = EX.instance(1, "int20", 3, 3, 0, 0)
    facts = EX.shapley_facts(wi)
    from zsp_scenario import make_market
    res = ev.analyse(ev.Settings(3, 3, "int20", 100, 0))
    other = EX.shapley_facts(np.array(make_market(3, 3, "int20", 100, 0)))
    assert other["in_core"] == res.in_core and other["pair_eq"] == res.pair_eq and len(other["blocking"]) == len(res.blocking)


def test_unequal_market_table_always_has_a_zero_contribution_player():
    rows = EX.unequal_market_table(shapes=((1, 3), (3, 1)), per_seed=5, seeds=2)
    assert all(r["with_zero_max"] == r["n_total"] for r in rows)
    assert all(r["n_in"] <= r["n_total"] for r in rows)


def test_width_table_rows_cover_every_cell_with_the_documented_check_range():
    rows = EX.width_table(dists=("unif",), ks=(3, 30), per_seed=1, seeds=2, lp_check_max_k=20)
    assert [(r["k"], r["lp_dev"] is None) for r in rows] == [(3, False), (30, True)]


def test_lattice_table_negative_probe_is_neither_empty_nor_always_in_the_core():
    rows = EX.lattice_price_table(dists=("int20",), shapes=((3, 3),), per_seed=3, seeds=2)
    r = rows[0]
    assert r["pairs"] > 0 and r["join_ok"] == r["meet_ok"] == r["pairs"] and 0 < r["neg_in"] < r["pairs"]


def test_density_table_orders_the_densities_as_given():
    rows = EX.density_table(densities=(10, 100), k=3, per_seed=5, seeds=2)
    assert [r["q"] for r in rows] == [10, 100]
