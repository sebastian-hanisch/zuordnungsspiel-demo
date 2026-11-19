"""Preise finden: aufsteigende Auktion und Ungarische Methode (Einheiten nach DEMO-PLAYBOOK 1b).

Preise y der Lkw und Überschüsse u der Ladungen bilden genau dann einen Kernpunkt, wenn u_i = max(0, max_j (w_ij - y_j)) gilt (jede Ladung
nimmt ihren besten Lkw) und u und y zusammen v(N) ergeben. Beide Verfahren hier liefern solche Preise. Werte ganzzahlig (Hundertstel oder Euro)."""

import numpy as np


def overdemanded_set(demand: list[int], b: int, subsets: np.ndarray, sizes: np.ndarray) -> int | None:
    """Kleinste überfragte Lkw-Menge als Bitmaske: eine Menge S von Lkw, die mehr Ladungen ausschließlich nachfragen (ihre Nachfragemenge liegt in S),
    als S Lkw hat. Die kleinste Mächtigkeit ist inklusionsminimal. `demand[i]` ist die Nachfragemaske von Ladung i (0 = fragt nichts nach)."""
    dem = np.array([d for d in demand if d != 0], dtype=np.int64)
    if dem.size == 0:
        return None
    inside = ((dem[None, :] & ~subsets[:, None]) == 0).sum(axis=1)      # Ladungen mit Nachfragemenge in S
    over = np.flatnonzero(inside > sizes)
    if over.size == 0:
        return None
    return int(subsets[over[np.argmin(sizes[over])]])


def ascending_auction(w: np.ndarray, round_cap: int = 5000) -> dict:
    """Aufsteigende Auktion (Demange, Gale, Sotomayor): Start bei Preis 0, in jeder Runde fragt jede Ladung ihre besten Lkw nach (Überschuss
    w_ij - Preis_j > 0), die kleinste überfragte Menge wird um 1 teurer. Ganzzahlige Werte; Ende, sobald keine Menge überfragt ist.

    Rückgabe: Preise, Runden und die Spur (je Runde: Preise vor der Erhöhung, überfragte Menge)."""
    w = np.asarray(w, dtype=np.int64)
    a, b = w.shape
    subsets = np.arange(1, 1 << b, dtype=np.int64)
    sizes = np.array([bin(int(s)).count("1") for s in subsets], dtype=np.int64)
    price = np.zeros(b, dtype=np.int64)
    trace = []
    for round_no in range(round_cap + 1):
        surplus = w - price[None, :]
        best = surplus.max(axis=1)
        demand = [0 if best[i] <= 0 else int(sum(1 << j for j in range(b) if surplus[i, j] == best[i])) for i in range(a)]
        found = overdemanded_set(demand, b, subsets, sizes)
        if found is None:
            return {"price": price, "rounds": round_no, "trace": trace, "converged": True}
        trace.append({"round": round_no + 1, "price": price.copy(), "set": [j for j in range(b) if found >> j & 1]})
        if round_no == round_cap:
            break
        for j in range(b):
            if found >> j & 1:
                price[j] += 1
    return {"price": price, "rounds": round_cap, "trace": trace, "converged": False}


def hungarian_potentials(cost: list[list[int]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Potenziale u (Zeilen) und v (Spalten) der Ungarischen Methode in der Fassung mit Zeilen als Neuankömmlingen (n Zeilen <= m Spalten),
    Kosten c: u_i + v_j <= c_ij mit Gleichheit auf der gefundenen Zuordnung. Ganzzahlige Kosten bleiben ganzzahlig (exakt)."""
    cost = [[int(x) for x in row] for row in np.asarray(cost)]
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError("die Fassung braucht höchstens so viele Zeilen wie Spalten")
    inf = float("inf")
    u, v = [0] * (n + 1), [0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return np.array(u[1:], dtype=np.int64), np.array(v[1:], dtype=np.int64)


def hungarian_prices(wi: np.ndarray, rows: str = "Ladungen") -> np.ndarray:
    """Auszahlungen (Ladungen | Lkw) aus den Potenzialen der Ungarischen Methode, Kosten = - Wert. `rows` ist die Seite, die als Zeilen
    geführt wird (Ladungen oder Lkw). Der Markt wird mit Nullwerten zu einem quadratischen aufgefüllt (Nichtpaarung bringt 0); die
    Auszahlungen der Auffüllspieler werden verworfen. Das Ergebnis ist ein Dualpunkt, nicht notwendig ein Kernpunkt (siehe Tests)."""
    wi = np.asarray(wi, dtype=np.int64)
    a, b = wi.shape
    k = max(a, b)
    padded = np.zeros((k, k), dtype=np.int64)
    padded[:a, :b] = wi
    if rows == "Ladungen":
        u, v = hungarian_potentials(-padded)
        return np.concatenate([-u[:a], -v[:b]])
    u, v = hungarian_potentials(-padded.T)           # Zeilen = Lkw
    return np.concatenate([-v[:a], -u[:b]])
