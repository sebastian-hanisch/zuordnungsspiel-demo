"""Auswertung: Einstellungen, Analyse eines Marktes, Verdikte als reine Funktionen `verdict_*(…) -> (Art, Text)`.

Alle Zahlen in den Texten kommen aus der Rechnung (f-Strings); Euro mit Dezimalkomma. Die App ruft `getattr(st, Art)(Text)`."""

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations

import numpy as np

import zsp_constants as C
import zsp_core as K
import zsp_game as G
import zsp_scenario as S


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(share: float, nd: int = 1) -> str:
    return de(100.0 * share, nd) + " %"


def player_names(a: int, b: int) -> list[str]:
    return [f"Ladung {i + 1}" for i in range(a)] + [f"Lkw {j + 1}" for j in range(b)]


@dataclass(frozen=True)
class Settings:
    a: int = C.DEFAULT_A
    b: int = C.DEFAULT_B
    dist: str = C.DEFAULT_DIST
    q: int = C.DEFAULT_Q
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    wi: np.ndarray            # Paarwerte in Hundertsteln
    v_n: float                # Wert der großen Koalition in Euro
    matching: list            # optimale Paare (Ladung, Lkw)
    phi: np.ndarray           # Shapley-Wert in Euro
    phi_frac: list            # Shapley-Wert als Brüche (Euro)
    in_core: bool
    pair_eq: bool
    blocking: list            # (Ladung, Lkw, Lücke in Euro), absteigend
    decoupled: bool | None    # nur quadratisch
    marginal: np.ndarray      # Beitrag zur großen Koalition in Euro (= Kern-Maximum)
    buyer: np.ndarray         # Käuferecke in Euro
    seller: np.ndarray        # Verkäuferecke in Euro

    @property
    def a(self) -> int:
        return self.wi.shape[0]

    @property
    def b(self) -> int:
        return self.wi.shape[1]

    @property
    def w(self) -> np.ndarray:
        return self.wi / float(C.SCALE)

    @property
    def width(self) -> np.ndarray:
        return np.abs(self.buyer - self.seller)


def analyse(settings: Settings) -> Analysis:
    wi = S.as_array(S.make_market(settings.a, settings.b, settings.dist, settings.q, settings.seed))
    a, b = wi.shape
    n = a + b
    v = G.value_table(wi)
    denom = math.factorial(n)
    phi_scaled = G.shapley_scaled(n, v)
    buyer, seller = K.extreme_points(wi)
    marg = G.marginal_contributions(n, v)
    scale = float(C.SCALE)
    return Analysis(
        settings=settings, wi=wi, v_n=int(v[-1]) / scale, matching=G.optimal_matching(wi),
        phi=phi_scaled / denom / scale, phi_frac=[Fraction(int(x), denom * C.SCALE) for x in phi_scaled],
        in_core=G.in_core_pairs(wi, phi_scaled, denom), pair_eq=G.pair_equations_hold(wi, phi_scaled, denom),
        blocking=[(i, j, gap / scale) for i, j, gap in G.blocking_pairs(wi, phi_scaled, denom)],
        decoupled=K.decoupled(wi), marginal=marg / scale, buyer=buyer / scale, seller=seller / scale,
    )


def hand_calculation() -> dict:
    """Handrechnung zwei Ladungen (Paarwerte 10 und 8) und ein Lkw: alle sechs Reihenfolgen, Beiträge und exakter Shapley-Wert (Brüche)."""
    wi = np.array([[1000], [800]], dtype=np.int64)
    v = G.value_table(wi)
    names = ["Ladung 1", "Ladung 2", "Lkw"]
    orders, total = [], [Fraction(0)] * 3
    for perm in permutations(range(3)):
        mask, contrib = 0, [0, 0, 0]
        for p in perm:
            contrib[p] = Fraction(int(v[mask | (1 << p)] - v[mask]), C.SCALE)
            mask |= 1 << p
        orders.append({"order": " → ".join(names[p] for p in perm), "contrib": contrib})
        total = [t + c for t, c in zip(total, contrib)]
    return {"names": names, "orders": orders, "shapley": [t / 6 for t in total]}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def verdict_shapley_core(a: int, b: int, v_n: float, in_core: bool, n_blocking: int, max_gap: float) -> tuple[str, str]:
    """Kernfrage: liegt die „faire“ Aufteilung nach Shapley im Kern?"""
    if v_n <= 0:
        return "warning", "Alle Paarwerte sind 0: Es gibt nichts zu verteilen, und jede Aufteilung ist stabil. Hier lässt sich der Unterschied nicht zeigen."
    if in_core:
        return "success", (f"Der Shapley-Wert liegt im Kern: Kein Paar aus Ladung und Lkw kann sich besserstellen. Das ist bei diesem Markt "
                           f"({a} Ladungen, {b} Lkw) so, aber nicht die Regel (siehe Experiment 1).")
    word = "Paar" if n_blocking == 1 else "Paare"
    return "warning", (f"Der Shapley-Wert liegt **außerhalb des Kerns**: {n_blocking} {word} aus Ladung und Lkw könnten sich zusammen besserstellen, "
                       f"die größte Lücke beträgt {de(max_gap)} €. Fair im Mittel über alle Reihenfolgen heißt hier nicht stabil.")


def verdict_core_span(v_n: float, total_width: float, n_free: int, n_players: int) -> tuple[str, str]:
    """Kern als Preisspanne: wie viel Spielraum lässt der Kern?"""
    if v_n <= 0:
        return "warning", "Alle Paarwerte sind 0: Der Kern ist der eine Punkt, an dem alle 0 bekommen."
    if total_width <= 1e-9:
        return "info", f"Der Kern besteht aus einem einzigen Punkt: Die Aufteilung ist durch die Paarwerte festgelegt. Alle {n_players} Spieler haben genau einen stabilen Betrag."
    return "info", (f"Der Kern lässt Spielraum: Die Beträge der beiden Ecken unterscheiden sich bei {n_free} von {n_players} Spielern, "
                    f"insgesamt um {de(total_width)} € (Summe der Breiten, ohne Gewichtung).")


def verdict_slider_point(in_core: bool, t_percent: int, n_blocking: int) -> tuple[str, str]:
    """Kontrolle des Preis-Schiebers: der Punkt zwischen den beiden Ecken ist ein Kernpunkt."""
    if in_core:
        return "success", f"Der Punkt bei {t_percent} % zwischen den Ecken liegt im Kern: kein Paar blockiert (alle Paarbedingungen gelten)."
    return "error", f"Der Punkt bei {t_percent} % liegt NICHT im Kern: {n_blocking} Paare blockieren. Das dürfte nicht vorkommen: Der Kern ist konvex."


def verdict_lattice(join_in: bool, meet_in: bool, neg_in: bool, comparable: bool) -> tuple[str, str]:
    """Verband zweier Kernpunkte: Join und Meet im Kern, die Verbindung beider Maxima meist nicht."""
    parts = [f"Join (die Ladungen nehmen je Spieler das Bessere, die Lkw das Schlechtere) {'liegt im Kern' if join_in else 'liegt NICHT im Kern'}",
             f"Meet {'liegt im Kern' if meet_in else 'liegt NICHT im Kern'}",
             f"die Gegenprobe (beide Seiten nehmen das Bessere) {'liegt trotzdem im Kern' if neg_in else 'liegt nicht im Kern'}"]
    kind = "success" if join_in and meet_in else "error"
    order = (" Was den Ladungen nützt, schadet den Lkw: Wo eine Ladung mehr bekommt, bekommen die Lkw weniger."
             if comparable else "")
    return kind, "; ".join(parts) + "." + order


def verdict_auction(converged: bool, rounds: int, same_as_corner: bool, rounded: bool) -> tuple[str, str]:
    """Aufsteigende Auktion gegen die Käuferecke des Kerns."""
    note = " Die Paarwerte wurden dafür auf ganze Euro gerundet." if rounded else ""
    if not converged:
        return "warning", f"Die Auktion bricht nach {rounds} Runden ab (Obergrenze {C.AUCTION_ROUND_CAP}).{note}"
    if same_as_corner:
        return "success", (f"Die Auktion endet nach {rounds} Runden auf den niedrigsten Preisen des Kerns, also auf den Preisen der Käuferecke "
                           f"(gleich dem Ergebnis der Beiträge zur großen Koalition).{note}")
    return "error", f"Die Auktion endet nach {rounds} Runden, aber nicht auf der Käuferecke. Das ist nach Messreihe 5 nicht zu erwarten.{note}"


def verdict_hungarian(rows: str, in_core: bool, at_buyer: bool, at_seller: bool) -> tuple[str, str]:
    """Welche Ecke wählt die Ungarische Methode, je nachdem, wer als Zeile geführt wird?"""
    if not in_core:
        return "warning", "Die Potenziale der Ungarischen Methode sind hier kein Kernpunkt (Dualpunkt, aber mit negativer Auszahlung oder anderem Fehler)."
    if at_buyer and at_seller:
        return "info", "Der Kern ist ein einziger Punkt: Beide Ecken fallen zusammen, die Wahl der Zeilenseite ändert nichts."
    if at_buyer:
        return "success", f"Mit {rows} als Zeilen liefert die Ungarische Methode die Käuferecke: Die Ladungen bekommen ihr Maximum, die Lkw ihr Minimum im Kern."
    if at_seller:
        return "success", f"Mit {rows} als Zeilen liefert die Ungarische Methode die Verkäuferecke: Die Lkw bekommen ihr Maximum, die Ladungen ihr Minimum im Kern."
    return "info", "Der Dualpunkt liegt im Kern, aber an keiner der beiden Ecken."


def _zero_cells(rows: list[dict], k_from: int) -> tuple[int, int]:
    cells = [r for r in rows if r["k"] >= k_from]
    return sum(1 for r in cells if r["n_in"] == 0), len(cells)


def verdict_share(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1: Anteil Shapley im Kern über k und Verteilung, mit der benannten Ausnahme."""
    ks = sorted({r["k"] for r in rows})
    k0, k1 = ks[0], (ks[1] if len(ks) > 1 else ks[0])

    def span(k):
        sel = [r for r in rows if r["k"] == k and r["n_total"] > 0]
        return min(sel, key=lambda r: r["share"]), max(sel, key=lambda r: r["share"])
    lo0, hi0 = span(k0)
    lo1, hi1 = span(k1)
    zero, cells = _zero_cells(rows, 4)
    exception = max((r for r in rows if r["k"] >= 4), key=lambda r: r["share"], default=None)
    text = (f"Bei k = {k0} liegt der Shapley-Wert in {pct(lo0['share'], 0)} ({lo0['dist']}) bis {pct(hi0['share'], 0)} ({hi0['dist']}) der Märkte im Kern, "
            f"bei k = {k1} in {pct(lo1['share'], 0)} ({lo1['dist']}) bis {pct(hi1['share'], 0)} ({hi1['dist']}).")
    if cells:
        text += f" Ab k = 4 sind es in {zero} von {cells} Zellen keine einzige Instanz."
        if exception and exception["n_in"] > 0:
            text += f" Größter Anteil dort: {pct(exception['share'], 1)} bei {exception['dist']}, k = {exception['k']}."
    return "info", text


def verdict_mechanism(rows: list[dict]) -> tuple[str, str]:
    """Experiment 2: Wie hängen „Shapley im Kern“, „entkoppelt“ und die Paargleichungen zusammen?"""
    n_total = sum(r["n_total"] for r in rows)
    in_core = sum(r["n11"] + r["n10"] for r in rows)
    not_decoupled_in_core = sum(r["n10"] for r in rows)
    k2 = [r for r in rows if r["k"] == 2]
    k2_equal = sum(1 for r in k2 if r["n01"] + r["n10"] == 0)
    decoupled_outside = sum(r["n01"] for r in rows if r["k"] >= 3)
    eq_not_core = sum(r["eq_not_core"] for r in rows)
    core_not_eq = sum(r["core_not_eq"] for r in rows)
    text = (f"Von {n_total} Märkten haben {in_core} den Shapley-Wert im Kern; {in_core - not_decoupled_in_core} davon sind entkoppelt. "
            f"Bei k = 2 sind „im Kern“ und „entkoppelt“ in {k2_equal} von {len(k2)} Verteilungen gleichbedeutend. Ab k = 3 ist entkoppelt nur notwendig: "
            f"{decoupled_outside} entkoppelte Märkte haben den Shapley-Wert trotzdem außerhalb. "
            f"Die Paargleichungen verletzt und doch im Kern: {core_not_eq} Märkte; erfüllt und doch nicht im Kern: {eq_not_core}.")
    return "info", text


def verdict_density(rows: list[dict]) -> tuple[str, str]:
    """Experiment 3a: je dünner der Markt, desto öfter liegt Shapley im Kern."""
    first, last = rows[0], rows[-1]
    steps = [rows[i + 1]["share"] - rows[i]["share"] for i in range(len(rows) - 1)]
    if all(d < 0 for d in steps):
        trend = "sinkt in jedem Schritt"
    elif all(d <= 0.01 for d in steps):
        trend = "sinkt mit der Dichte; Anstiege bis 1 Prozentpunkt sind bei dieser Stichprobe nicht von Rauschen zu unterscheiden"
    else:
        trend = "sinkt nicht durchgehend"
    text = (f"Bei Dichte {first['q']} % liegt der Shapley-Wert in {pct(first['share'], 1)} der Märkte im Kern, bei {last['q']} % in {pct(last['share'], 1)} "
            f"(k = {first['k']}, {first['n_total']} bzw. {last['n_total']} Märkte; Instanzen ohne Wert ausgeschlossen: {first['n_zero']} bzw. {last['n_zero']}). "
            f"Der Anteil {trend}.")
    return "info", text


def verdict_unequal(rows: list[dict]) -> tuple[str, str]:
    """Experiment 3b: ungleich große Märkte."""
    n_in = sum(r["n_in"] for r in rows)
    n_tot = sum(r["n_total"] for r in rows)
    paid = sum(r["zero_paid"] for r in rows)
    zero_max = sum(r["with_zero_max"] for r in rows)
    text = (f"In {n_in} von {n_tot} ungleich großen Märkten liegt der Shapley-Wert im Kern. In {zero_max} von {n_tot} gibt es Spieler ohne Beitrag zur großen Koalition "
            f"(ihr Kern-Maximum ist 0), und in {paid} davon zahlt der Shapley-Wert einem solchen Spieler mehr als 0.")
    return "info", text


def verdict_width(rows: list[dict]) -> tuple[str, str]:
    """Experiment 4: Kernbreite gegen k, Zufallsmärkte gegen Geografie."""
    ks = sorted({r["k"] for r in rows})
    k0, k1 = ks[0], ks[-1]

    def mean_width(dist, k):
        sel = [r["width"] for r in rows if r["dist"] == dist and r["k"] == k]
        return sel[0] if sel else float("nan")
    dists = list(dict.fromkeys(r["dist"] for r in rows))
    parts = [f"{d}: {de(mean_width(d, k0))} (k = {k0}) → {de(mean_width(d, k1))} (k = {k1})" for d in dists]
    lp_devs = [r["lp_dev"] for r in rows if r["lp_dev"] is not None]
    check = f" Ecken gegen das Linearprogramm (k ≤ 20): größte Abweichung {max(lp_devs):.1e} Hundertstel." if lp_devs else ""
    return "info", "Mittlere Kernbreite je Spieler (in Einheiten von v(N)/k): " + "; ".join(parts) + "." + check


def verdict_lattice_prices(rows: list[dict]) -> tuple[str, str]:
    """Experiment 5: Verband, Auktion und Ungarische Methode gegen die LP-Ecken."""
    pairs = sum(r["pairs"] for r in rows)
    join = sum(r["join_ok"] for r in rows)
    meet = sum(r["meet_ok"] for r in rows)
    neg = sum(r["neg_in"] for r in rows)
    n = sum(r["n"] for r in rows)
    auction = sum(r["auction_ok"] for r in rows)
    hb = sum(r["hung_buyer"] for r in rows)
    hs = sum(r["hung_seller"] for r in rows)
    kind = "success" if join == pairs and meet == pairs and auction == n and hb == n and hs == n else "warning"
    return kind, (f"Join im Kern: {join} von {pairs} Paaren, Meet: {meet} von {pairs}; die Gegenprobe (beide Maxima) liegt nur in {neg} von {pairs} im Kern. "
                  f"Die Auktion endet in {auction} von {n} Märkten auf der Käuferecke. Die Ungarische Methode liefert mit Ladungen als Zeilen in {hb} von {n} Märkten "
                  f"die Käuferecke, mit Lkw als Zeilen in {hs} von {n} die Verkäuferecke.")
