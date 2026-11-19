"""Messreihen der Experimente (reine Funktionen; die App ruft sie hinter einem Knopf, `tests/test_claims.py` pinnt jede Zahl der README).

Jede Zelle (Verteilung, Größe) bekommt EXP_SEEDS Zufalls-Seeds mit je mehreren Instanzen; Mittel und Standardabweichung laufen über die Seeds.
Instanzen mit v(N) = 0 (alle Paarwerte 0) werden ausgeschlossen und gezählt: bei ihnen gibt es nichts zu verteilen.
Instanz-Zufalls-Seeds kommen aus `zsp_scenario.cell_seed` (SplitMix64), nie aus `random` oder `numpy.random`."""

import itertools
import math

import numpy as np

import zsp_constants as C
import zsp_core as K
import zsp_game as G
import zsp_scenario as S
from zsp_auction import ascending_auction, hungarian_prices
from zsp_rng import uniform_block

DIST_INDEX = {d: i for i, d in enumerate(C.DISTS)}
TAG_E1, TAG_E2, TAG_E3A, TAG_E3B, TAG_E4, TAG_E5 = 1, 2, 3, 4, 5, 6


def instance(tag: int, dist: str, a: int, b: int, seed_no: int, run: int, q: int = 100) -> np.ndarray:
    """Eine Instanz einer Zelle als ganzzahlige Matrix in Hundertsteln."""
    return np.array(S.make_market(a, b, dist, q, S.cell_seed(tag, DIST_INDEX[dist], a * 100 + b, seed_no, run)), dtype=np.int64)


def shapley_facts(wi: np.ndarray) -> dict:
    """Exakter Shapley-Wert eines Marktes und alles, was die Experimente davon brauchen (v(N) > 0 vorausgesetzt)."""
    a, b = wi.shape
    n = a + b
    v = G.value_table(wi)
    phi = G.shapley_scaled(n, v)
    f = math.factorial(n)
    marg = G.marginal_contributions(n, v)
    blocking = G.blocking_pairs(wi, phi, f)
    return {
        "v_n": int(v[-1]), "phi": phi, "denom": f, "in_core": G.in_core_pairs(wi, phi, f),
        "pair_eq": G.pair_equations_hold(wi, phi, f), "blocking": blocking,
        "over_max": int((phi > marg * f).sum()),            # Spieler, deren Shapley-Wert ihr Kern-Maximum übersteigt
        "zero_max_paid": int(((marg == 0) & (phi > 0)).sum()),   # Spieler ohne Beitrag zur großen Koalition (Kern-Maximum 0), denen Shapley > 0 zahlt
        "zero_max": int((marg == 0).sum()),
    }


def _mean_std(per_seed: list[float]) -> tuple[float, float]:
    arr = np.array(per_seed, dtype=float)
    return float(arr.mean()), float(arr.std())


def shapley_share_table(dists=None, ks=C.E1_KS, per_seed=None, seeds=C.EXP_SEEDS) -> list[dict]:
    """E1: Anteil der quadratischen k x k Märkte, in denen der Shapley-Wert im Kern liegt, je Verteilung und k."""
    per_seed = per_seed or C.E1_PER_SEED
    rows = []
    for dist in dists or list(C.RANDOM_DISTS):
        for k in ks:
            shares, n_in, n_tot, n_zero, blocks = [], 0, 0, 0, []
            for s in range(seeds):
                inc = cnt = 0
                for run in range(per_seed[k]):
                    wi = instance(TAG_E1, dist, k, k, s, run)
                    if wi.sum() == 0:
                        n_zero += 1
                        continue
                    facts = shapley_facts(wi)
                    cnt += 1
                    inc += facts["in_core"]
                    if not facts["in_core"]:
                        blocks.append(len(facts["blocking"]))
                shares.append(inc / cnt if cnt else float("nan"))
                n_in += inc
                n_tot += cnt
            mean, std = _mean_std(shares)
            rows.append({"dist": dist, "k": k, "share": mean, "std": std, "n_in": n_in, "n_total": n_tot, "n_zero": n_zero,
                         "blocking_mean": float(np.mean(blocks)) if blocks else 0.0})
    return rows


def mechanism_table(dists=("unif", "int20", "geo", "pareto"), ks=C.E2_KS, per_seed=None, seeds=C.EXP_SEEDS) -> list[dict]:
    """E2: Kontingenz (Shapley im Kern) x (entkoppelt) und Übereinstimmung mit den Paargleichungen, quadratische Märkte."""
    per_seed = per_seed or C.E2_PER_SEED
    rows = []
    for dist in dists:
        for k in ks:
            n11 = n10 = n01 = n00 = eq_not_core = core_not_eq = n_zero = 0
            for s in range(seeds):
                for run in range(per_seed[k]):
                    wi = instance(TAG_E2, dist, k, k, s, run)
                    if wi.sum() == 0:
                        n_zero += 1
                        continue
                    facts = shapley_facts(wi)
                    dec = bool(K.decoupled(wi))
                    inc = facts["in_core"]
                    n11 += inc and dec
                    n10 += inc and not dec
                    n01 += (not inc) and dec
                    n00 += (not inc) and not dec
                    eq_not_core += facts["pair_eq"] and not inc
                    core_not_eq += inc and not facts["pair_eq"]
            rows.append({"dist": dist, "k": k, "n11": int(n11), "n10": int(n10), "n01": int(n01), "n00": int(n00),
                         "eq_not_core": int(eq_not_core), "core_not_eq": int(core_not_eq), "n_total": int(n11 + n10 + n01 + n00), "n_zero": n_zero})
    return rows


def density_table(densities=C.E3_DENSITIES, k=C.E3_K, per_seed=C.E3_PER_SEED, seeds=C.EXP_SEEDS, dist="unif") -> list[dict]:
    """E3a: Anteil Shapley im Kern bei Dichte q (Anteil der Paare mit Wert ungleich 0), quadratische Märkte."""
    rows = []
    for q in densities:
        shares, n_in, n_tot, n_zero = [], 0, 0, 0
        for s in range(seeds):
            inc = cnt = 0
            for run in range(per_seed):
                wi = instance(TAG_E3A, dist, k, k, s, run, q=q)
                if wi.sum() == 0:
                    n_zero += 1
                    continue
                cnt += 1
                inc += shapley_facts(wi)["in_core"]
            shares.append(inc / cnt if cnt else float("nan"))
            n_in += inc
            n_tot += cnt
        mean, std = _mean_std(shares)
        rows.append({"q": q, "k": k, "share": mean, "std": std, "n_in": n_in, "n_total": n_tot, "n_zero": n_zero})
    return rows


def unequal_market_table(shapes=C.E3_MARKETS, per_seed=C.E3_MARKET_PER_SEED, seeds=C.EXP_SEEDS, dist="unif") -> list[dict]:
    """E3b: ungleich große Märkte a x b (a != b): Shapley im Kern? Zahlt der Shapley-Wert Spielern, deren Kern-Maximum 0 ist?"""
    rows = []
    for a, b in shapes:
        n_in = n_tot = n_zero = with_zero_max = zero_paid = 0
        blocks = []
        for s in range(seeds):
            for run in range(per_seed):
                wi = instance(TAG_E3B, dist, a, b, s, run)
                if wi.sum() == 0:
                    n_zero += 1
                    continue
                facts = shapley_facts(wi)
                n_tot += 1
                n_in += facts["in_core"]
                blocks.append(len(facts["blocking"]))
                with_zero_max += facts["zero_max"] > 0
                zero_paid += facts["zero_max_paid"] > 0
        rows.append({"a": a, "b": b, "n_in": int(n_in), "n_total": int(n_tot), "n_zero": n_zero,
                     "with_zero_max": int(with_zero_max), "zero_paid": int(zero_paid), "blocking_mean": float(np.mean(blocks)) if blocks else 0.0})
    return rows


def width_table(dists=("unif", "int20", "geo"), ks=C.E4_KS, per_seed=C.E4_PER_SEED, seeds=C.EXP_SEEDS, lp_check_max_k=20) -> list[dict]:
    """E4: mittlere Kernbreite (Abstand der beiden Ecken je Spieler, geteilt durch v(N)/k) gegen k; Ecken aus Beiträgen zur großen Koalition,
    für k <= lp_check_max_k zusätzlich gegen das Linearprogramm geprüft (größte Abweichung der Eckensummen in Hundertsteln)."""
    rows = []
    for dist in dists:
        for k in ks:
            widths, unique_share, worst_lp = [], [], 0.0
            for s in range(seeds):
                for run in range(per_seed):
                    wi = instance(TAG_E4, dist, k, k, s, run)
                    v_n = int(G.assignment_value(wi))
                    if v_n == 0:
                        continue
                    buyer, seller = K.extreme_points(wi)
                    width = K.core_width(buyer, seller)
                    widths.append(float(width.mean()) / (v_n / k))
                    unique_share.append(float((width == 0).mean()))
                    if k <= lp_check_max_k:
                        w = wi.astype(float)
                        lb = K.core_lp(w, np.concatenate([-np.ones(k), np.zeros(k)]))
                        ls = K.core_lp(w, np.concatenate([np.zeros(k), -np.ones(k)]))
                        worst_lp = max(worst_lp, abs(float(lb[:k].sum()) - float(buyer[:k].sum())), abs(float(ls[k:].sum()) - float(seller[k:].sum())))
            rows.append({"dist": dist, "k": k, "width": float(np.mean(widths)), "unique": float(np.mean(unique_share)),
                         "n": len(widths), "lp_dev": worst_lp if k <= lp_check_max_k else None})
    return rows


def _core_points(w: np.ndarray, seed: int, m: int) -> list[np.ndarray]:
    """m Ecken des Kerns: LP-Lösungen zu Zielvektoren aus SplitMix64 (u - 1/2)."""
    n = sum(w.shape)
    u = uniform_block(seed, m * n).reshape(m, n) - 0.5
    return K.core_vertices(w, u)


def lattice_price_table(dists=("int20", "int100", "int3"), shapes=C.E5_SHAPES, per_seed=C.E5_PER_SEED, seeds=C.EXP_SEEDS, vertices=5) -> list[dict]:
    """E5: (1) Join und Meet zweier Kernpunkte im Kern, Negativprobe (größeres u UND größeres y) im Kern; (2) aufsteigende Auktion endet auf den
    niedrigsten Kernpreisen (Käuferecke); (3) Potenziale der Ungarischen Methode: Dualpunkt im Kern, Lage der Lkw-Preise (Zeilen = Ladungen: Käuferecke)."""
    rows = []
    for dist in dists:
        for a, b in shapes:
            pairs = join_ok = meet_ok = neg_in = auction_ok = n = hung_core = hung_buyer = hung_seller = rounds_total = 0
            max_rounds = 0
            for s in range(seeds):
                for run in range(per_seed):
                    wi = instance(TAG_E5, dist, a, b, s, run)
                    if wi.sum() == 0:
                        continue
                    n += 1
                    w = wi / float(C.SCALE)
                    pts = _core_points(w, S.cell_seed(TAG_E5, 99, s, run), vertices)
                    for p in range(len(pts)):
                        for r in range(p + 1, len(pts)):
                            join, meet = K.join_meet(pts[p], pts[r], a)
                            neg = K.negative_probe(pts[p], pts[r], a)
                            pairs += 1
                            join_ok += G.in_core_pairs(wi, join * C.SCALE, 1, 1e-4)
                            meet_ok += G.in_core_pairs(wi, meet * C.SCALE, 1, 1e-4)
                            neg_in += G.in_core_pairs(wi, neg * C.SCALE, 1, 1e-4)
                    buyer, seller = K.extreme_points(wi)
                    ints = wi // C.SCALE                        # ganze Euro: die Werte der ganzzahligen Verteilungen
                    res = ascending_auction(ints)
                    buyer_i, _ = K.extreme_points(ints)
                    auction_ok += bool(res["converged"]) and bool((res["price"] == buyer_i[a:]).all())
                    rounds_total += res["rounds"]
                    max_rounds = max(max_rounds, res["rounds"])
                    hr = hungarian_prices(wi, "Ladungen")
                    hl = hungarian_prices(wi, "Lkw")
                    hung_core += G.in_core_pairs(wi, hr, 1)
                    hung_buyer += bool((hr[a:] == buyer[a:]).all())
                    hung_seller += bool((hl[a:] == seller[a:]).all())
            rows.append({"dist": dist, "a": a, "b": b, "n": n, "pairs": pairs, "join_ok": int(join_ok), "meet_ok": int(meet_ok), "neg_in": int(neg_in),
                         "auction_ok": int(auction_ok), "rounds_mean": rounds_total / n if n else 0.0, "rounds_max": max_rounds,
                         "hung_core": int(hung_core), "hung_buyer": int(hung_buyer), "hung_seller": int(hung_seller)})
    return rows


def small_squares(max_value: int = 3) -> dict:
    """Alle 2 x 2 Märkte mit ganzzahligen Paarwerten 0 bis max_value (Aufzählung, kein Zufall): in wie vielen mit v(N) > 0 liegt der Shapley-Wert
    außerhalb des Kerns, und welche Matrix mit kleinster Wertsumme zeigt es zuerst?"""
    total = outside = 0
    smallest = None
    for entries in itertools.product(range(max_value + 1), repeat=4):
        wi = np.array(entries, dtype=np.int64).reshape(2, 2) * C.SCALE
        if wi.sum() == 0:
            continue
        total += 1
        if not shapley_facts(wi)["in_core"]:
            outside += 1
            if smallest is None or sum(entries) < sum(smallest):
                smallest = entries
    return {"total": total, "outside": outside, "smallest": smallest}
