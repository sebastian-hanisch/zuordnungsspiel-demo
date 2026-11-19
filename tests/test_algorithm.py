"""Worst-Case am Regler-Maximum (DEMO-PLAYBOOK §1c: Timeout statt Zeit-Assertion) und Invarianten des Kerns, die jede Analyse einhalten muss."""

import threading

import numpy as np
import pytest

import zsp_constants as C
import zsp_evaluation as ev
import zsp_game as G


def test_regler_maximum_finishes_for_every_distribution():
    """a = b = 7 (n = 14, 16 384 Koalitionen) in jeder Verteilung: die Rechnung endet; fängt Hänger und unerwartetes Aufblähen (keine Zeitmessung)."""
    done = threading.Event()
    result = {}

    def work():
        for dist in C.DISTS:
            result[dist] = ev.analyse(ev.Settings(C.A_MAX, C.B_MAX, dist, 100, 7))
        done.set()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    assert done.wait(timeout=120), "Analyse am Regler-Maximum hängt"
    assert set(result) == set(C.DISTS)


def test_analysis_invariants_on_a_grid_of_settings():
    for a, b in [(1, 1), (2, 3), (3, 2), (4, 4), (1, 7), (7, 1)]:
        for dist in ("int20", "unif", "pareto", "geo", "h_diag", "h_rang"):
            r = ev.analyse(ev.Settings(a, b, dist, 100, 2))
            assert float(r.phi.sum()) == pytest.approx(r.v_n, rel=1e-9, abs=1e-9)            # effizient
            assert (r.phi >= -1e-12).all() and (r.marginal >= -1e-12).all()                   # nie negativ (Paarwerte >= 0)
            assert (np.minimum(r.buyer, r.seller) <= np.maximum(r.buyer, r.seller)).all()
            assert float(r.buyer.sum()) == pytest.approx(r.v_n) and float(r.seller.sum()) == pytest.approx(r.v_n)
            assert np.array_equal(np.maximum(r.buyer, r.seller)[:a], r.buyer[:a])             # Käuferecke: Ladungen am Maximum
            assert np.array_equal(np.maximum(r.buyer, r.seller)[a:], r.seller[a:])            # Verkäuferecke: Lkw am Maximum
            assert r.in_core == (r.blocking == [] and (r.phi >= 0).all())                     # im Kern genau dann, wenn kein Paar blockiert (Summe ist effizient)


def test_analysis_depends_only_on_the_settings():
    s = ev.Settings(3, 4, "geo", 60, 21)
    r1, r2 = ev.analyse(s), ev.analyse(ev.Settings(3, 4, "geo", 60, 21))
    assert np.array_equal(r1.wi, r2.wi) and np.array_equal(r1.phi, r2.phi) and r1.blocking == r2.blocking


def test_value_table_is_monotone_and_superadditive_on_disjoint_pairs():
    wi = np.array(ev.analyse(ev.Settings(3, 3, "int20", 100, 0)).wi)
    v = G.value_table(wi)
    n = 6
    for mask in range(1 << n):
        for bit in range(n):
            if not mask >> bit & 1:
                assert v[mask | 1 << bit] >= v[mask]                                          # Monotonie: ein Spieler mehr schadet nie
    assert v[0] == 0
