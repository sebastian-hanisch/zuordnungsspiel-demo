"""Der Kern des Zuordnungsspiels: Ecken, Linearprogramm, Verband, Dimension, Entkopplung (Einheiten nach DEMO-PLAYBOOK 1b).

Der Kern ist {x >= 0 : Summe = v(N), x_i + x_(a+j) >= w_ij für alle Paare}. Er ist nie leer und hat kein Inneres: in jedem Kernpunkt gilt
x_i + x_(a+j) = w_ij für jedes Paar eines optimalen Matchings. Alle Größen in Hundertsteln, `wi` ganzzahlig (siehe `zsp_game`)."""

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

from zsp_game import assignment_value, optimal_matching


def extreme_points(wi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Käuferecke (Ladungen bekommen ihr Kern-Maximum) und Verkäuferecke (Lkw bekommen ihr Kern-Maximum), ganzzahlig in Hundertsteln.

    Rechenweg ohne Linearprogramm: Das Kern-Maximum jedes Spielers ist sein Beitrag zur großen Koalition v(N) - v(N ohne i); die Gegenseite
    bekommt in jedem optimalen Paar den Rest w_ij - x_i. Nicht gepaarte Spieler bekommen 0."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    v_n = assignment_value(wi)
    u_max = np.array([v_n - assignment_value(wi, drop_row=i) for i in range(a)], dtype=np.int64)
    y_max = np.array([v_n - assignment_value(wi, drop_col=j) for j in range(b)], dtype=np.int64)
    buyer, seller = np.zeros(a + b, dtype=np.int64), np.zeros(a + b, dtype=np.int64)
    buyer[:a], seller[a:] = u_max, y_max
    for i, j in optimal_matching(wi):
        buyer[a + j] = wi[i, j] - u_max[i]
        seller[i] = wi[i, j] - y_max[j]
    return buyer, seller


def core_lp(w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """min c·x über den Kern (Paarbeschreibung, HiGHS); `w` in beliebiger Einheit, Ergebnis in derselben. Orakel zu `extreme_points`."""
    w = np.asarray(w, dtype=float)
    a, b = w.shape
    n = a + b
    rows = np.zeros((a * b, n))
    for i in range(a):
        for j in range(b):
            rows[i * b + j, i] = -1.0
            rows[i * b + j, a + j] = -1.0
    r_idx, c_idx = linear_sum_assignment(-w)
    v_n = float(w[r_idx, c_idx].sum())
    res = linprog(c, A_ub=rows, b_ub=-w.reshape(-1), A_eq=np.ones((1, n)), b_eq=[v_n], bounds=[(0, None)] * n, method="highs")
    if res.status != 0:  # pragma: no cover - der Kern ist nie leer
        raise RuntimeError(f"Kern-LP ohne Lösung: {res.message}")
    return res.x


def l1_distance_to_core(w: np.ndarray, x: np.ndarray) -> float:
    """Kleinster L1-Abstand zwischen x und einem Kernpunkt (LP: min Summe |y - x| über y im Kern), in der Einheit von `w`."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    a, b = w.shape
    n = a + b
    r_idx, c_idx = linear_sum_assignment(-w)
    v_n = float(w[r_idx, c_idx].sum())
    rows, rhs = [], []
    for i in range(a):
        for j in range(b):
            r = np.zeros(2 * n)
            r[i] = r[a + j] = -1.0
            rows.append(r)
            rhs.append(-w[i, j])
    for t in range(n):
        r = np.zeros(2 * n)
        r[t], r[n + t] = 1.0, -1.0
        rows.append(r)
        rhs.append(x[t])
        r = np.zeros(2 * n)
        r[t], r[n + t] = -1.0, -1.0
        rows.append(r)
        rhs.append(-x[t])
    res = linprog(np.concatenate([np.zeros(n), np.ones(n)]), A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=[np.concatenate([np.ones(n), np.zeros(n)])], b_eq=[v_n], bounds=[(0, None)] * (2 * n), method="highs")
    if res.status != 0:  # pragma: no cover - der Kern ist nie leer
        raise RuntimeError(f"Abstands-LP ohne Lösung: {res.message}")
    return float(res.fun)


def join_meet(x: np.ndarray, z: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
    """Verband der Kernpunkte: Join = (größeres u, kleineres y), Meet = (kleineres u, größeres y), je Spieler."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    join = np.concatenate([np.maximum(x[:a], z[:a]), np.minimum(x[a:], z[a:])])
    meet = np.concatenate([np.minimum(x[:a], z[:a]), np.maximum(x[a:], z[a:])])
    return join, meet


def negative_probe(x: np.ndarray, z: np.ndarray, a: int) -> np.ndarray:
    """Gegenprobe zum Verband: (größeres u, größeres y). Diese Verbindung liegt meist NICHT im Kern."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    return np.maximum(x, z)


def core_vertices(w: np.ndarray, picks: np.ndarray) -> list[np.ndarray]:
    """Kernpunkte als LP-Lösungen zu den Zielvektoren (Zeilen von `picks`): Ecken des Kerns."""
    return [core_lp(w, c) for c in picks]


def core_dimension(w: np.ndarray) -> int:
    """Dimension des Kerns: n minus Rang der Gleichungen, die in jedem Kernpunkt gelten (Effizienz plus die Bedingungen, die der Kern nie
    strikt erfüllt; jede Bedingung wird per LP geprüft: ist ihr Maximum über den Kern gleich der Schranke, gilt sie überall mit Gleichheit)."""
    w = np.asarray(w, dtype=float)
    a, b = w.shape
    n = a + b
    equalities = [np.ones(n)]
    candidates = []
    for i in range(a):
        for j in range(b):
            r = np.zeros(n)
            r[i] = r[a + j] = 1.0
            candidates.append((r, w[i, j]))
    for t in range(n):
        r = np.zeros(n)
        r[t] = 1.0
        candidates.append((r, 0.0))
    for r, bound in candidates:
        x = core_lp(w, -r)                                  # Maximum von r·x über den Kern
        if abs(float(r @ x) - float(bound)) <= 1e-7 * max(1.0, abs(float(bound))):
            equalities.append(r)
    return n - int(np.linalg.matrix_rank(np.array(equalities), tol=1e-9))


def _augment(i: int, adj: list[list[int]], seen: set[int], match_col: dict[int, int]) -> bool:
    for j in adj[i]:
        if j in seen:
            continue
        seen.add(j)
        if j not in match_col or _augment(match_col[j], adj, seen, match_col):
            match_col[j] = i
            return True
    return False


def decoupled(wi: np.ndarray) -> bool | None:
    """Entkoppelt (nur quadratische Märkte): es gibt eine vollständige Zuordnung, in der jeder Paarwert zugleich das Maximum seiner Zeile und
    seiner Spalte ist (gegenseitig beste Partner). Gesucht per Matching im Graphen der Kanten, die beides erfüllen. Nicht quadratisch: None.
    Streng definiert: Bei Gleichständen kann der Shapley-Wert im Kern liegen, obwohl dies False liefert (Beispiel [[0, 3], [3, 4]]: die beste Zuordnung nutzt die
    beiden gleichen Einträge 3 und 3, der Eintrag 4 liegt nicht in ihr). Die Definition bleibt, damit die gepinnten Zählungen der Messreihe 2 gelten."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    if a != b:
        return None
    row_max, col_max = wi.max(axis=1), wi.max(axis=0)
    adj = [[j for j in range(b) if wi[i, j] == row_max[i] and wi[i, j] == col_max[j]] for i in range(a)]
    match_col: dict[int, int] = {}
    return all(_augment(i, adj, set(), match_col) for i in range(a))


def core_width(buyer: np.ndarray, seller: np.ndarray) -> np.ndarray:
    """Breite des Kerns je Spieler: Abstand der beiden Ecken (Maximum minus Minimum der Auszahlung im Kern)."""
    return np.abs(np.asarray(buyer) - np.asarray(seller))
