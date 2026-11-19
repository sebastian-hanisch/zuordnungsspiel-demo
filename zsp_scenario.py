"""Szenario: Märkte (Ladungen x Lkw) mit Paarwerten aus Verteilungen oder Handbeispielen. Zufall nur über SplitMix64 (`zsp_rng`).

Ein Markt ist ein Tupel von Tupeln ganzer Zahlen in Hundertsteln (`wi`, hashbar für die Caches der App). Zufallsverteilungen ziehen aus EINEM
Block `uniform_block(seed, ...)`: zuerst die Paarwerte (a*b Zahlen), dann Hilfsgrößen (3a + 2b), danach die a*b Zahlen der Dichtemaske."""

import numpy as np

import zsp_constants as C
from zsp_rng import SplitMix64, uniform_block


def cell_seed(*key: int) -> int:
    """Instanz-Zufalls-Seed aus einem Zellschlüssel (Experiment, Verteilung, Größe, Lauf): ein SplitMix64-Schritt über den gemischten Schlüssel."""
    state = 0
    for part in key:
        state = SplitMix64((state * 1_000_003 + int(part) + 1) & ((1 << 64) - 1)).next()
    return state & ((1 << 62) - 1)


def _cents(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 1e-9).astype(np.int64)


def _hand(dist: str, a: int, b: int) -> np.ndarray:
    """Handbeispiele in ganzen Zahlen (Euro); sie hängen nur von a und b ab."""
    w = np.zeros((a, b), dtype=np.int64)
    if dist == "h_eins":
        w[:] = 1
    elif dist == "h_spalte":
        for i in range(a):
            w[i, :] = max(0, 10 - 2 * i)
    elif dist == "h_diag":
        w[:] = 1
        for i in range(min(a, b)):
            w[i, i] = max(2, 10 - 2 * i)
    elif dist == "h_rang":
        for i in range(a):
            for j in range(b):
                w[i, j] = (a - i) * (b - j)
    elif dist == "h_getrennt":
        for i in range(min(a, b)):
            w[i, i] = 10
    else:
        raise ValueError(f"unbekanntes Handbeispiel: {dist}")
    return w * C.SCALE


def _random_base(dist: str, a: int, b: int, u: np.ndarray) -> np.ndarray:
    """Paarwerte in Hundertsteln aus den ersten Gleichverteilten `u` (a*b Basiszahlen, danach Hilfsgrößen)."""
    base = u[: a * b].reshape(a, b)
    aux = u[a * b: a * b + 3 * a + 2 * b]
    s = C.SCALE
    if dist == "int20":
        return np.floor(base * 20).astype(np.int64) * s
    if dist == "int100":
        return np.floor(base * 100).astype(np.int64) * s
    if dist == "int3":
        return np.floor(base * 4).astype(np.int64) * s
    if dist == "bin":
        return (base < 0.5).astype(np.int64) * s
    if dist == "unif":
        return np.floor(base * (100 * s + 1)).astype(np.int64)
    if dist == "pareto":
        value = np.minimum(C.PARETO_CAP, 10.0 * (1.0 - base) ** (-1.0 / 1.5))
        return np.floor(value * s).astype(np.int64)
    if dist == "prod":
        sl, tl = 0.5 + aux[:a], 0.5 + aux[a: a + b]
        return np.floor(50.0 * s * np.outer(sl, tl)).astype(np.int64)
    if dist == "geo":
        revenue = 40.0 + 60.0 * aux[:a]
        loads = aux[a: 3 * a].reshape(a, 2)
        trucks = aux[3 * a: 3 * a + 2 * b].reshape(b, 2)
        dist_ij = np.sqrt(((loads[:, None, :] - trucks[None, :, :]) ** 2).sum(axis=2))
        return np.floor(np.maximum(0.0, revenue[:, None] - 60.0 * dist_ij) * s).astype(np.int64)
    raise ValueError(f"unbekannte Verteilung: {dist}")


def make_market(a: int, b: int, dist: str, q: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Markt in Hundertsteln. Zufallsverteilungen: Zufalls-Seed und Dichte q (Prozent der Paare, die ungleich 0 sein dürfen); Handbeispiele: weder noch."""
    if dist in C.HAND_DISTS:
        w = _hand(dist, a, b)
    elif dist in C.RANDOM_DISTS:
        n_base = a * b + 3 * a + 2 * b
        u = uniform_block(int(seed), n_base + a * b)
        w = _random_base(dist, a, b, u)
        keep = u[n_base: n_base + a * b].reshape(a, b) < q / 100.0
        w = w * keep
    else:
        raise ValueError(f"unbekannte Verteilung: {dist}")
    return tuple(tuple(int(x) for x in row) for row in w)


def as_array(market) -> np.ndarray:
    return np.array(market, dtype=np.int64).reshape(len(market), -1)
