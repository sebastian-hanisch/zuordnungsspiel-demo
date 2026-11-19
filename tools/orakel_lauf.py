"""Großer Orakel-Lauf (nicht Teil der Testsuite, rund 10 Sekunden): die Kernroutinen gegen andere Rechenwege auf mehr als 200 Instanzen je Prüfung.

Aufruf aus dem Ordner der Demo: python tools/orakel_lauf.py. Die Zahlen der README stehen in der Ausgabe; die letzte Zeile nennt Zahl der Prüfungen und Fehler."""

import itertools
import math
import sys
import time
from fractions import Fraction as F
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from conftest import brute_value  # noqa: E402
from scipy.optimize import linear_sum_assignment, linprog  # noqa: E402

import zsp_auction as AU  # noqa: E402
import zsp_constants as C  # noqa: E402
import zsp_core as K  # noqa: E402
import zsp_experiments as EX  # noqa: E402
import zsp_game as G  # noqa: E402

DISTS = list(C.RANDOM_DISTS)
checks = errors = 0


def ok(cond, msg):
    global checks, errors
    checks += 1
    if not cond:
        errors += 1
        print("FEHLER", msg)


def inst(tag, dist, a, b, s, r):
    return EX.instance(tag, dist, a, b, s, r)


t0 = time.time()

# O1: v(S) per DP gegen scipy und Aufzählung aller Teilzuordnungen
n_inst = n_vals = 0
for a, b in [(2, 2), (3, 2), (2, 3), (3, 3), (1, 3), (4, 2), (2, 4), (1, 1)]:
    for dist in DISTS:
        for r in range(4):
            wi = inst(200, dist, a, b, 0, r)
            v = G.value_table(wi)
            n_inst += 1
            for m in range(1 << (a + b)):
                loads = [i for i in range(a) if m >> i & 1]
                trucks = [j for j in range(b) if m >> (a + j) & 1]
                ref = 0
                if loads and trucks:
                    sub = wi[np.ix_(loads, trucks)]
                    rr, cc = linear_sum_assignment(-sub)
                    ref = int(sub[rr, cc].sum())
                n_vals += 1
                ok(v[m] == ref == brute_value(wi, m), ("v", dist, a, b, r, m))
print(f"O1 v(S): {n_inst} Instanzen, {n_vals} Koalitionswerte, DP = scipy = Aufzählung")

# O2: Shapley-Wert gegen alle Reihenfolgen in exakten Brüchen
n2 = 0
for a, b in [(2, 2), (3, 3), (2, 4), (4, 2), (3, 2)]:
    n = a + b
    for dist in DISTS:
        for r in range(6):
            wi = inst(201, dist, a, b, 0, r)
            v = G.value_table(wi)
            tot = [0] * n
            for order in itertools.permutations(range(n)):
                m = 0
                for p in order:
                    tot[p] += int(v[m | 1 << p] - v[m])
                    m |= 1 << p
            expected = [F(x, math.factorial(n)) for x in tot]
            got = [F(int(x), math.factorial(n)) for x in G.shapley_scaled(n, v)]
            ok(expected == got, ("shapley", dist, a, b, r))
            n2 += 1
print(f"O2 Shapley-Wert: {n2} Instanzen (n = 4 bis 6), Teilmengenformel = alle n! Reihenfolgen in exakten Brüchen")

# O3: Kern über Paare gegen Kern über alle Koalitionen
pts = inc = n3 = 0
for a, b in [(2, 2), (3, 3), (3, 2), (2, 4), (4, 4), (4, 3), (1, 3), (5, 2)]:
    n = a + b
    for dist in DISTS:
        for r in range(6):
            wi = inst(202, dist, a, b, 0, r)
            if wi.sum() == 0:
                continue
            n3 += 1
            v = G.value_table(wi)
            den = math.factorial(n)
            phi = G.shapley_scaled(n, v)
            bu, se = K.extreme_points(wi)
            cands = [(phi, den), (bu, 1), (se, 1), (bu + se, 2)]
            for t in range(6):
                x = bu.copy()
                x[t % n] += 7 * (t + 1)
                x[(t + 1) % n] -= 7 * (t + 1)
                cands.append((x, 1))
            for x, d in cands:
                p = G.in_core_pairs(wi, x, d)
                f = G.in_core_full(n, v, x, d)
                ok(p == f, ("kern", dist, a, b, r))
                pts += 1
                inc += p
print(f"O3 Kern über Paare = Kern über alle Koalitionen: {n3} Instanzen, {pts} Punkte ({inc} im Kern, {pts - inc} außerhalb)")

# O4: Least-Core-Wert über alle Koalitionen ist 0
n4 = 0
worst = 0.0
for a, b in [(2, 2), (3, 3), (3, 2), (4, 4), (2, 3), (4, 2)]:
    n = a + b
    for dist in DISTS:
        for r in range(8 if n < 8 else 3):
            wi = inst(203, dist, a, b, 0, r)
            if wi.sum() == 0:
                continue
            v = G.value_table(wi) / 100.0
            rows = [[-(m >> t & 1) for t in range(n)] + [-1] for m in range(1, (1 << n) - 1)]
            rhs = [-v[m] for m in range(1, (1 << n) - 1)]
            res = linprog([0] * n + [1], A_ub=rows, b_ub=rhs, A_eq=[[1] * n + [0]], b_eq=[v[-1]], bounds=[(None, None)] * (n + 1), method="highs")
            ok(res.status == 0 and abs(res.x[-1]) < 1e-7, ("eps", dist, a, b, r))
            worst = max(worst, abs(res.x[-1]))
            n4 += 1
print(f"O4 Least-Core-Wert über alle Koalitionen: {n4} Instanzen, größter Betrag {worst:.1e} (null bis auf die LP-Genauigkeit)")

# O5: Ecken gegen LP über Paare und über alle Koalitionen
n5 = 0
worst5 = 0.0
for a, b in [(2, 2), (3, 3), (2, 3), (3, 2), (4, 2), (2, 4), (4, 4), (1, 3)]:
    n = a + b
    for dist in DISTS:
        for r in range(4):
            wi = inst(204, dist, a, b, 0, r)
            if wi.sum() == 0:
                continue
            w = wi / 100.0
            vt = G.value_table(wi)
            v = vt / 100.0
            marg = G.marginal_contributions(n, vt)
            bu, se = K.extreme_points(wi)
            ok(np.array_equal(np.concatenate([bu[:a], se[a:]]), marg), ("ecke-marg", dist, a, b, r))
            for i in range(n):
                c = np.zeros(n)
                c[i] = -1
                lp = -c @ K.core_lp(w, c)
                worst5 = max(worst5, abs(lp - marg[i] / 100.0))
                ok(abs(lp - marg[i] / 100.0) < 1e-7, ("lp", dist, a, b, r, i))
            if n <= 6:
                rows = [[-(m >> t & 1) for t in range(n)] for m in range(1, (1 << n) - 1)]
                rhs = [-v[m] for m in range(1, (1 << n) - 1)]
                for i in range(n):
                    c = np.zeros(n)
                    c[i] = -1
                    res = linprog(c, A_ub=rows, b_ub=rhs, A_eq=[[1] * n], b_eq=[v[-1]], bounds=[(0, None)] * n, method="highs")
                    ok(abs(-res.fun - marg[i] / 100.0) < 1e-7, ("lpfull", dist, a, b, r, i))
            n5 += 1
print(f"O5 Ecken: {n5} Instanzen, Maximum je Spieler im Kern = Beitrag zur großen Koalition = LP (über Paare, bei n <= 6 auch über alle Koalitionen); größte Abweichung {worst5:.1e}")

# O6: aufsteigende Auktion gegen die LP-Ecke, Ungarische Methode gegen die Ecken
n6 = nh = 0
for a, b in [(3, 3), (4, 3), (3, 4), (2, 5), (5, 2), (5, 5), (4, 6), (6, 4), (1, 1), (1, 4)]:
    for dist in ["int20", "int100", "int3", "bin"]:
        for r in range(6):
            wi = inst(205, dist, a, b, 0, r)
            if wi.sum() == 0:
                continue
            ints = wi // 100
            res = AU.ascending_auction(ints)
            lp = K.core_lp(ints.astype(float), np.concatenate([-np.ones(a), np.zeros(b)]))
            ok(res["converged"] and np.allclose(res["price"], lp[a:], atol=1e-7), ("auktion", dist, a, b, r))
            n6 += 1
for a, b in [(3, 3), (4, 3), (3, 4), (2, 2), (5, 5), (2, 5), (5, 2)]:
    for dist in DISTS:
        for r in range(5):
            wi = inst(206, dist, a, b, 0, r)
            if wi.sum() == 0:
                continue
            bu, se = K.extreme_points(wi)
            hl, ht = AU.hungarian_prices(wi, "Ladungen"), AU.hungarian_prices(wi, "Lkw")
            ok(G.in_core_pairs(wi, hl) and G.in_core_pairs(wi, ht), ("hung-kern", dist, a, b, r))
            ok(np.array_equal(hl[a:], bu[a:]) and np.array_equal(ht[a:], se[a:]), ("hung-ecke", dist, a, b, r))
            nh += 1
print(f"O6 Auktion = LP-Ecke: {n6} Märkte; Ungarische Methode (beide Rollen) im Kern und an der Ecke der Zeilenseite: {nh} Märkte")

# O7: 0/1-Zuordnungsspiel gegen ein Flussspiel
tot = 0
for a, b in [(3, 3), (3, 4), (4, 3), (2, 5), (4, 4)]:
    for r in range(15):
        wi = inst(207, "bin", a, b, 0, r)
        v = G.value_table(wi)
        for m in range(1 << (a + b)):
            g = nx.DiGraph()
            g.add_nodes_from(["s", "t"])
            for i in range(a):
                if m >> i & 1:
                    g.add_edge("s", ("L", i), capacity=1)
                    for j in range(b):
                        if wi[i, j] > 0:
                            g.add_edge(("L", i), ("T", j), capacity=1)
            for j in range(b):
                if m >> (a + j) & 1:
                    g.add_edge(("T", j), "t", capacity=1)
            f = nx.maximum_flow_value(g, "s", "t") if g.out_degree("s") > 0 and g.in_degree("t") > 0 else 0
            tot += 1
            ok(int(v[m]) == 100 * f, ("fluss", a, b, r, m))
print(f"O7 0/1-Zuordnungsspiel = Flussspiel (networkx): {tot} Koalitionswerte")

print(f"GESAMT: {checks} Prüfungen, {errors} Fehler, {time.time() - t0:.0f} s")
