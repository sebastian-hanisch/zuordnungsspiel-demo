"""Orakel-Tests: die Kernroutinen gegen andere Rechenwege (ORACLE-PLAYBOOK §2: Spieltheorie, Matching, LP).

Jede Prüfung nennt ihren Umfang; der große Lauf (mehr als 200 Instanzen je Prüfung) lief im Scratch-Ordner, die Zahlen stehen in der README.
- v(S): DP gegen `linear_sum_assignment` (scipy) gegen Aufzählung aller Teilzuordnungen
- Shapley-Wert: Teilmengenformel gegen alle n! Reihenfolgen in exakten Brüchen
- Kern: Paarbedingungen gegen alle 2^n Koalitionen (Shapley-Wert, Ecken, gestörte Punkte)
- Ecken: geschlossene Form (Beiträge zur großen Koalition) gegen LP über Paare und gegen LP über alle Koalitionen
- Least-Core-Wert über alle Koalitionen ist 0: der Kern hat kein Inneres und ist nie leer
- aufsteigende Auktion gegen die LP-Ecke; Ungarische Methode: Dualzulässigkeit und Summe gleich v(N)
- 0/1-Zuordnungsspiel gegen ein Flussspiel (networkx)"""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog

import zsp_auction as AU
import zsp_constants as C
import zsp_core as K
import zsp_experiments as EX
import zsp_game as G
import zsp_scenario as S
from conftest import brute_value

DISTS = list(C.RANDOM_DISTS)


def markets(a, b, runs=3, dists=DISTS, tag=90):
    """Feste Zufallsmärkte (SplitMix64) in Hundertsteln; Märkte ohne Wert werden übersprungen."""
    for dist in dists:
        for run in range(runs):
            wi = EX.instance(tag, dist, a, b, 0, run)
            if wi.sum() > 0:
                yield dist, run, wi


def lsa_value(wi, mask):
    a, b = wi.shape
    loads = [i for i in range(a) if mask >> i & 1]
    trucks = [j for j in range(b) if mask >> (a + j) & 1]
    if not loads or not trucks:
        return 0
    sub = wi[np.ix_(loads, trucks)]
    r, c = linear_sum_assignment(-sub)
    return int(sub[r, c].sum())


def test_value_table_equals_scipy_and_brute_force_on_all_coalitions():
    checked = 0
    for a, b in [(2, 2), (3, 2), (2, 3), (3, 3), (1, 3)]:
        for dist, run, wi in markets(a, b, runs=2):
            v = G.value_table(wi)
            for mask in range(1 << (a + b)):
                assert v[mask] == lsa_value(wi, mask) == brute_value(wi, mask), (dist, a, b, run, mask)
                checked += 1
    assert checked > 2500


def test_shapley_equals_the_average_over_all_orders_in_exact_fractions():
    for a, b in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=2, dists=["int20", "unif", "geo", "int3"]):
            v = G.value_table(wi)
            total = [0] * n
            for order in itertools.permutations(range(n)):
                mask = 0
                for p in order:
                    total[p] += int(v[mask | (1 << p)] - v[mask])
                    mask |= 1 << p
            expected = [F(t, math.factorial(n)) for t in total]
            got = [F(int(x), math.factorial(n)) for x in G.shapley_scaled(n, v)]
            assert got == expected, (dist, a, b, run)


def test_shapley_is_efficient_and_symmetric_in_exact_arithmetic():
    for a, b in [(3, 3), (2, 4), (4, 3)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=2):
            v = G.value_table(wi)
            phi = G.shapley_scaled(n, v)
            assert int(phi.sum()) == int(v[-1]) * math.factorial(n)


def test_pair_conditions_agree_with_all_coalition_conditions():
    points = verdicts = 0
    for a, b in [(2, 2), (3, 3), (3, 2), (2, 4), (4, 4)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=3, dists=["int20", "unif", "int3", "geo", "bin"]):
            v = G.value_table(wi)
            denom = math.factorial(n)
            phi = G.shapley_scaled(n, v)
            buyer, seller = K.extreme_points(wi)
            candidates = [(phi, denom), (buyer, 1), (seller, 1), ((buyer + seller), 2)]
            for t in range(4):                                   # gestörte Punkte: Betrag t+1 Cent von einem Spieler zum anderen
                x = buyer.copy()
                x[t % n] += 7 * (t + 1)
                x[(t + 1) % n] -= 7 * (t + 1)
                candidates.append((x, 1))
            for x, den in candidates:
                assert G.in_core_pairs(wi, x, den) == G.in_core_full(n, v, x, den), (dist, a, b, run)
                points += 1
                verdicts += G.in_core_pairs(wi, x, den)
    assert points > 500 and 0 < verdicts < points            # beide Urteile kommen vor: der Test ist nicht leer


def test_corners_equal_the_lp_over_pairs_and_the_lp_over_all_coalitions():
    for a, b in [(2, 2), (3, 3), (2, 3), (3, 2), (4, 2)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=2, dists=["int20", "unif", "geo"]):
            w = wi / 100.0
            v = G.value_table(wi) / 100.0
            buyer, seller = K.extreme_points(wi)
            marg = G.marginal_contributions(n, G.value_table(wi))
            # Maximum je Spieler im Kern: LP über Paare, LP über alle Koalitionen, Beitrag zur großen Koalition
            rows, rhs = [], []
            for mask in range(1, (1 << n) - 1):
                rows.append([-(mask >> t & 1) for t in range(n)])
                rhs.append(-v[mask])
            for i in range(n):
                c = np.zeros(n)
                c[i] = -1.0
                lp_pairs = -c @ K.core_lp(w, c)
                res = linprog(c, A_ub=rows, b_ub=rhs, A_eq=[[1] * n], b_eq=[v[-1]], bounds=[(0, None)] * n, method="highs")
                assert -res.fun == pytest.approx(lp_pairs, abs=1e-7), (dist, run, i)
                assert lp_pairs == pytest.approx(marg[i] / 100.0, abs=1e-7), (dist, run, i)
            high = np.concatenate([buyer[:a], seller[a:]])
            assert np.array_equal(high, marg), (dist, a, b, run)         # beide Ecken erreichen alle Maxima gleichzeitig


def test_least_core_value_is_zero_so_the_core_has_no_interior():
    checked = 0
    for a, b in [(2, 2), (3, 3), (3, 2), (4, 4)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=2, dists=["int20", "unif", "geo", "int3"]):
            v = G.value_table(wi) / 100.0
            rows = [[-(mask >> t & 1) for t in range(n)] + [-1] for mask in range(1, (1 << n) - 1)]
            rhs = [-v[mask] for mask in range(1, (1 << n) - 1)]
            res = linprog([0] * n + [1], A_ub=rows, b_ub=rhs, A_eq=[[1] * n + [0]], b_eq=[v[-1]], bounds=[(None, None)] * (n + 1), method="highs")
            assert res.status == 0 and abs(res.x[-1]) < 1e-7, (dist, a, b, run, res.x[-1])
            checked += 1
    assert checked >= 30


def test_every_corner_is_in_the_core_over_all_coalitions():
    for a, b in [(3, 3), (2, 4), (4, 2)]:
        n = a + b
        for dist, run, wi in markets(a, b, runs=2):
            v = G.value_table(wi)
            for corner in K.extreme_points(wi):
                assert G.in_core_full(n, v, corner), (dist, a, b, run)


def test_ascending_auction_ends_on_the_lowest_core_prices():
    checked = 0
    for a, b in [(3, 3), (4, 3), (3, 4), (2, 5)]:
        for dist, run, wi in markets(a, b, runs=3, dists=["int20", "int3", "bin"]):
            ints = wi // 100
            res = AU.ascending_auction(ints)
            lp = K.core_lp(ints.astype(float), np.concatenate([-np.ones(a), np.zeros(b)]))        # Käuferecke: Summe der Ladungs-Überschüsse maximal
            assert res["converged"]
            assert np.allclose(res["price"], lp[a:], atol=1e-7), (dist, a, b, run, res["price"], lp[a:])
            checked += 1
    assert checked >= 30


def test_hungarian_potentials_are_dual_feasible_and_sum_to_the_optimum():
    for a, b in [(3, 3), (4, 4), (2, 2)]:
        for dist, run, wi in markets(a, b, runs=3, dists=["int20", "unif", "geo", "int100"]):
            u, v = AU.hungarian_potentials(-wi)
            assert (u[:, None] + v[None, :] <= -wi).all()
            assert int(u.sum() + v.sum()) == -int(G.assignment_value(wi))


def test_hungarian_prices_are_core_points_and_pick_the_corner_of_the_row_side():
    for a, b in [(3, 3), (4, 3), (3, 4), (2, 2)]:
        for dist, run, wi in markets(a, b, runs=3, dists=["int20", "unif", "geo"]):
            buyer, seller = K.extreme_points(wi)
            rows_loads = AU.hungarian_prices(wi, "Ladungen")
            rows_trucks = AU.hungarian_prices(wi, "Lkw")
            assert G.in_core_pairs(wi, rows_loads) and G.in_core_pairs(wi, rows_trucks)
            assert np.array_equal(rows_loads[a:], buyer[a:]) and np.array_equal(rows_trucks[a:], seller[a:]), (dist, a, b, run)


def test_zero_one_assignment_game_equals_a_flow_game():
    nx = pytest.importorskip("networkx")
    checked = 0
    for a, b in [(3, 3), (3, 4), (4, 3)]:
        for run in range(4):
            wi = EX.instance(91, "bin", a, b, 0, run)
            v = G.value_table(wi)
            for mask in range(1 << (a + b)):
                g = nx.DiGraph()
                g.add_nodes_from(["s", "t"])
                for i in range(a):
                    if mask >> i & 1:
                        g.add_edge("s", ("L", i), capacity=1)
                        for j in range(b):
                            if wi[i, j] > 0:
                                g.add_edge(("L", i), ("T", j), capacity=1)
                for j in range(b):
                    if mask >> (a + j) & 1:
                        g.add_edge(("T", j), "t", capacity=1)
                flow = nx.maximum_flow_value(g, "s", "t") if g.out_degree("s") > 0 and g.in_degree("t") > 0 else 0
                assert int(v[mask]) == 100 * flow, (a, b, run, mask)
                checked += 1
    assert checked > 1200


def test_core_dimension_is_k_for_continuous_squares():
    """Kern ohne Inneres: n - k freie Richtungen statt n - 1 (stetige Werte, quadratische Märkte, nur Instanzen mit v(N) > 0)."""
    for k in (2, 3):
        for run in range(6):
            wi = EX.instance(92, "unif", k, k, 0, run)
            assert K.core_dimension(wi / 100.0) == k
