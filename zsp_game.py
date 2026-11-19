"""Das Zuordnungsspiel: Koalitionswerte, Shapley-Wert und Kern-Tests (Einheiten nach DEMO-PLAYBOOK 1b).

Spieler 0 .. a-1 sind Ladungen (Käuferseite, Überschuss u), a .. a+b-1 sind Lkw (Verkäuferseite, Preis y). Ein Paar (i, j) schafft den
Paarwert w[i, j] >= 0; der Wert einer Koalition S ist das beste Matching innerhalb von S (nicht gepaart zu bleiben ist erlaubt und bringt 0).
Alle Werte werden ganzzahlig in Hundertsteln gerechnet (`wi`, Form (a, b), int64): Tabelle, Shapley-Wert und Kerntests sind dann exakt
und auf jeder Plattform gleich. Koalitionen sind Bitmasken: Bit i (i < a) ist Ladung i, Bit a + j ist Lkw j."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def popcount_table(n: int) -> np.ndarray:
    """Zahl der gesetzten Bits für alle Masken 0 .. 2^n - 1."""
    pc = np.zeros(1 << n, dtype=np.int64)
    for bit in range(n):
        pc += (np.arange(1 << n) >> bit) & 1
    return pc


def value_table(wi: np.ndarray) -> np.ndarray:
    """v(S) für alle 2^(a+b) Koalitionen per Teilmengen-DP (Ladungen aufsteigend, je Ladung alle Lkw-Mengen vektorisiert)."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    nb = 1 << b
    tmasks = np.arange(nb)
    has = [((tmasks >> j) & 1) == 1 for j in range(b)]
    f = np.zeros((1 << a, nb), dtype=np.int64)
    for m_load in range(1, 1 << a):
        i = (m_load & -m_load).bit_length() - 1
        rest = m_load & (m_load - 1)
        best = f[rest].copy()
        for j in range(b):
            cand = f[rest, tmasks[has[j]] ^ (1 << j)] + wi[i, j]
            best[has[j]] = np.maximum(best[has[j]], cand)
        f[m_load] = best
    return f.T.ravel().copy()          # Index = (Lkw-Maske << a) | Ladungsmaske


def assignment_value(wi: np.ndarray, drop_row: int | None = None, drop_col: int | None = None) -> int:
    """Wert des besten Matchings (Rechenweg 2: Ungarische Methode aus scipy), optional ohne eine Ladung oder einen Lkw."""
    wi = np.asarray(wi, dtype=np.int64)
    rows = [i for i in range(wi.shape[0]) if i != drop_row]
    cols = [j for j in range(wi.shape[1]) if j != drop_col]
    if not rows or not cols:
        return 0
    sub = wi[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(-sub)
    return int(sub[r, c].sum())


def optimal_matching(wi: np.ndarray) -> list[tuple[int, int]]:
    """Ein optimales Matching als Paare (Ladung, Lkw) mit Paarwert > 0. Bei Gleichstand gilt das, was scipy zuerst findet (deterministisch)."""
    wi = np.asarray(wi, dtype=np.int64)
    r, c = linear_sum_assignment(-wi)
    return [(int(i), int(j)) for i, j in zip(r, c) if wi[i, j] > 0]


def shapley_scaled(n: int, v: np.ndarray) -> np.ndarray:
    """n! mal der Shapley-Wert (exakt, ganzzahlig): sum_S s! (n-s-1)! (v(S ∪ i) - v(S)) über alle S ohne i."""
    fact = [math.factorial(i) for i in range(n + 1)]
    weight = np.array([fact[s] * fact[n - s - 1] for s in range(n)], dtype=np.int64)
    masks = np.arange(1 << n)
    pc = popcount_table(n)
    phi = np.zeros(n, dtype=np.int64)
    for i in range(n):
        without = masks[((masks >> i) & 1) == 0]
        phi[i] = (weight[pc[without]] * (v[without | (1 << i)] - v[without])).sum()
    return phi


def marginal_contributions(n: int, v: np.ndarray) -> np.ndarray:
    """Beitrag zur großen Koalition v(N) - v(N ohne i), ganzzahlig."""
    full = (1 << n) - 1
    return np.array([v[full] - v[full ^ (1 << i)] for i in range(n)], dtype=np.int64)


def _exact(x: np.ndarray, tol) -> tuple[bool, float | int]:
    """Ganzzahlige Punkte (Shapley-Wert als n! * phi) werden ohne Toleranz und ohne Umweg über float verglichen; Gleitkommapunkte mit `tol`."""
    if np.issubdtype(x.dtype, np.integer):
        return True, 0
    return False, tol


def in_core_pairs(wi: np.ndarray, x: np.ndarray, denom: int = 1, tol: float = 1e-7) -> bool:
    """Kern-Test über die Paarbedingungen: x >= 0, Summe = v(N), x_i + x_(a+j) >= w_ij für alle a*b Paare.
    `x` steht in Einheiten von 1/denom Hundertsteln (Shapley-Wert exakt: x = n! * phi als int64, denom = n!); `tol` gilt nur für Gleitkommapunkte
    (absolut, in Hundertsteln), ganzzahlige Punkte werden exakt verglichen."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    x = np.asarray(x)
    exact, tol = _exact(x, tol)
    v_n = assignment_value(wi) * denom
    total = int(x.sum()) if exact else float(x.sum())
    if (x < -tol).any() or abs(total - v_n) > tol:
        return False
    return bool((x[:a, None] + x[None, a:] >= wi * denom - tol).all())


def coalition_sums(x: np.ndarray) -> np.ndarray:
    """Summe von x über alle Koalitionen (Bitmasken)."""
    n = len(x)
    s = np.zeros(1 << n, dtype=x.dtype)
    for bit in range(n):
        s[1 << bit: 1 << (bit + 1)] = s[: 1 << bit] + x[bit]
    return s


def in_core_full(n: int, v: np.ndarray, x: np.ndarray, denom: int = 1, tol: float = 1e-7) -> bool:
    """Kern-Test über ALLE Koalitionen (Orakel zu `in_core_pairs`): Effizienz und sum_(i in S) x_i >= v(S) für jede Koalition S."""
    x = np.asarray(x)
    exact, tol = _exact(x, tol)
    s = coalition_sums(x)
    full = (1 << n) - 1
    if abs((int(s[full]) - int(v[full]) * denom) if exact else (float(s[full]) - float(v[full]) * denom)) > tol:
        return False
    return bool((s[1:full] >= v[1:full] * denom - tol).all())


def blocking_pairs(wi: np.ndarray, x: np.ndarray, denom: int = 1, tol: float = 1e-7) -> list[tuple[int, int, float]]:
    """Paare (Ladung, Lkw), die sich mit x besserstellen könnten: Lücke w_ij - (x_i + x_(a+j)) > 0, in Hundertsteln; absteigend nach Lücke."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    x = np.asarray(x)
    exact, tol = _exact(x, tol)
    short = (wi * denom - (x[:a, None] + x[None, a:])) > tol          # exakt in ganzen Zahlen bei ganzzahligem x
    found = [(i, j, float(wi[i, j] - (x[i] + x[a + j]) / denom)) for i in range(a) for j in range(b) if short[i, j]]
    found.sort(key=lambda t: (-t[2], t[0], t[1]))
    return found


def pair_equations_hold(wi: np.ndarray, x: np.ndarray, denom: int = 1, tol: float = 1e-7) -> bool:
    """Gilt x_i + x_(a+j) = w_ij für jedes Paar des optimalen Matchings? (Im Kern gilt das für jedes optimale Matching.)"""
    wi = np.asarray(wi, dtype=np.int64)
    a = wi.shape[0]
    x = np.asarray(x)
    exact, tol = _exact(x, tol)
    if exact:
        return all(int(x[i]) + int(x[a + j]) == int(wi[i, j]) * denom for i, j in optimal_matching(wi))
    return all(abs(float(x[i] + x[a + j]) - float(wi[i, j]) * denom) <= tol for i, j in optimal_matching(wi))
