"""Wer fährt mit wem – das Zuordnungsspiel - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Teil der Linie „Koalitionen und Teilen“: Ladungen und Lkw bilden Paare mit einem Paarwert. Welche Aufteilung des Gesamtgewinns ist
stabil (Kern), wie viel Spielraum bleibt, und warum liegt die „faire“ Aufteilung nach Shapley bei größeren Märkten meist nicht darin?

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import zsp_auction as AU
import zsp_constants as C
import zsp_core as K
import zsp_evaluation as ev
import zsp_experiments as EX
import zsp_game as G
import zsp_scenario as S
import zsp_texts as T
import zsp_visualization as V
from zsp_layout import dunkles_design, kopf_css
from zsp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from zsp_rng import uniform_block

st.set_page_config(page_title="Wer fährt mit wem – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _dimension(market):
    return K.core_dimension(S.as_array(market) / float(C.SCALE))


@st.cache_data(show_spinner=False)
def _distance(market, phi):
    return K.l1_distance_to_core(S.as_array(market) / float(C.SCALE), np.array(phi))


@st.cache_data(show_spinner=False)
def _two_core_points(market, pick):
    """Zwei Ecken des Kerns zu zufälligen Zielvektoren (SplitMix64, Auswahl `pick`)."""
    w = S.as_array(market) / float(C.SCALE)
    n = sum(w.shape)
    u = uniform_block(S.cell_seed(7, pick), 2 * n).reshape(2, n) - 0.5
    pts = K.core_vertices(w, u)
    return pts[0], pts[1]


@st.cache_data(show_spinner=False)
def _auction(market):
    ints = np.rint(S.as_array(market) / float(C.SCALE)).astype(np.int64)
    return AU.ascending_auction(ints, C.AUCTION_ROUND_CAP), K.extreme_points(ints)[0]


def _experiment(key, label, func, spinner):
    """Knopf + Spinner + Zeitangabe; das Ergebnis bleibt in st.session_state (die Messreihen hängen nicht von den Reglern ab)."""
    if st.button(label, key=f"btn_{key}"):
        t0 = time.perf_counter()
        with st.spinner(spinner):
            rows = func()
        st.session_state[key] = (rows, time.perf_counter() - t0)
    return st.session_state.get(key)


def _show_verdict(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zsp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚚 Wer fährt mit wem – das Zuordnungsspiel")
st.markdown(
    """
Teil der Linie [Koalitionen und Teilen](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html). **Ladungen und Lkw** bilden Paare, jedes Paar schafft einen **Paarwert**; der **Kern** sagt, welche
Aufteilung des Gesamtgewinns stabil ist, und der **Shapley-Wert**, welche im Mittel über alle Reihenfolgen „fair“ ist. Die Demo setzt `austauschnetz-demo` fort
(dort: Matching auf Graphen mit ganzzahligen Kantenwerten): hier zweiseitig, mit stetigen Werten, mit dem **Kern als Preisspanne** und einem Mechanismus dafür, warum der
Shapley-Wert bei größeren Märkten meist nicht im Kern liegt. Eigennutz und Paarwerte sind Modellannahmen.

Weiter unten: vier Ansichten (Markt, Kern, Shapley, Preise), die **Kernfrage**, fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen,
Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/zuordnungsspiel-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ladungen und Lkw bilden Paare, und jedes Paar schafft einen **Paarwert**, gedacht als Erlös minus Fahrkosten. Die Demo fragt, wie sich der Gesamtwert der besten Zuordnung aufteilen lässt, ohne dass sich ein Paar von einem anderen Partner abwerben lässt. Alle stabilen Aufteilungen bilden den **Kern**, der hier als Preisspanne mit zwei äußersten Ecken erscheint. Der **Shapley-Wert** teilt dagegen nach dem durchschnittlichen Beitrag über alle Reihenfolgen und ist nicht immer stabil, wie Handrechnungen und Messreihen zeigen. Zwei Verfahren, eine aufsteigende Auktion und die Ungarische Methode, finden Preise im Kern. **Alle Märkte sind erfunden. Die Zählungen gelten für die gezogenen Zufallsmärkte und sind kein Beweis.**

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/zuordnungsspiel-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Zuordnungsspiel", expanded=True):
    st.markdown(
        """
1. **Paarwerte:** Jede Ladung und jeder Lkw haben einen Paarwert (Erlös minus Fahrkosten). Eine Koalition erreicht den Wert ihres besten Matchings.
2. **Aufteilung:** Der Gesamtwert v(N) wird verteilt. **Stabil** (im Kern) heißt: Kein Paar kann zusammen mehr erreichen, als beide bekommen.
3. **Preise:** Eine stabile Aufteilung ist dasselbe wie ein Preissystem: Jeder Lkw hat einen Preis, jede Ladung nimmt den Lkw mit dem größten Überschuss. Der Kern hat zwei äußerste Ecken (bei manchen Märkten fallen sie zusammen): in der einen sind
   die Preise am niedrigsten (gut für die Ladungen), in der anderen am höchsten.
4. **Shapley-Wert:** Der Mittelwert der Beiträge über alle Reihenfolgen. Er ist exakt gerechnet (Brüche) und liegt bei größeren Märkten meist nicht im Kern (Experiment 1).
        """
    )
    st.markdown(T.GLOSSARY_MARKDOWN)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Markt**")
    n_a = st.slider("Ladungen a (Käuferseite)", *bounds("a_slider"), key="a_slider", help="Wie viele Ladungen zur Wahl stehen.")
    n_b = st.slider("Lkw b (Verkäuferseite)", *bounds("b_slider"), key="b_slider", help="Wie viele Lkw zur Wahl stehen.")
    dist = st.selectbox(
        "Verteilung der Paarwerte", list(C.DISTS), key="dist_select", format_func=C.DISTS.get,
        help="Zufallsverteilungen ziehen Paarwerte mit Zufalls-Seed und Dichte; Handbeispiele sind feste Märkte.",
    )
    st.caption(T.DIST_HELP[dist])
    fixed = dist in C.HAND_DISTS
    q = st.slider(
        "Dichte der Paarwerte (%)", *bounds("q_slider"), key="q_slider", disabled=fixed,
        help="Anteil der Paare, die einen Wert ungleich 0 haben dürfen. Bei Handbeispielen ohne Wirkung (deshalb ausgegraut).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1, disabled=fixed)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, disabled=fixed)

sync_query_params({"a_slider": int(n_a), "b_slider": int(n_b), "dist_select": dist, "q_slider": int(q), "seed_input": int(seed)})

settings = ev.Settings(int(n_a), int(n_b), dist, int(q), int(seed))
res = _analysis(settings)
a, b = res.a, res.b
names = ev.player_names(a, b)
load_names, truck_names = names[:a], names[a:]
low = np.minimum(res.buyer, res.seller)
high = np.maximum(res.buyer, res.seller)
n_free = int((res.width > 1e-9).sum())

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gesamtwert v(N)", f"{ev.de(res.v_n)} €")
m2.metric("Paare in der besten Zuordnung", f"{len(res.matching)} von {min(a, b)}")
m3.metric("Kern-Spielraum (Summe)", f"{ev.de(float(res.width.sum()))} €")
m4.metric("Shapley-Wert im Kern?", "ja" if res.in_core else "nein", delta=f"{len(res.blocking)} blockierende Paare", delta_color="off")

tab_market, tab_core, tab_shapley, tab_prices = st.tabs(["Markt", "Kern als Preisspanne", "Shapley gegen Kern", "Preise finden"])

with tab_market:
    st.caption("Paarwerte in €; rot umrahmt: die Paare der besten Zuordnung (eine optimale, bei Gleichstand die, die scipy zuerst findet).")
    st.plotly_chart(V.build_matrix(res.w, res.matching, load_names, truck_names), width="stretch", key="chart_market")
    pair_rows = [{"Ladung": load_names[i], "Lkw": truck_names[j], "Paarwert (€)": ev.de(res.w[i, j])} for i, j in res.matching]
    if pair_rows:
        st.dataframe(pd.DataFrame(pair_rows), hide_index=True, width="stretch")
    else:
        st.info("Kein Paar hat einen Wert über 0: Die beste Zuordnung ist leer, v(N) = 0.")
    st.markdown("**Beitrag zur großen Koalition** (v(N) minus v(N ohne den Spieler)): zugleich das Maximum, das der Spieler im Kern bekommen kann.")
    st.dataframe(
        pd.DataFrame({"Spieler": names, "Beitrag (€)": [ev.de(x) for x in res.marginal]}), hide_index=True, width="stretch",
    )

with tab_core:
    st.markdown(
        "Die beiden Ecken des Kerns: In der **Käuferecke** bekommen die Ladungen ihr Maximum und die Lkw ihr Minimum, in der **Verkäuferecke** umgekehrt. "
        "Jeder Punkt dazwischen ist ein Kernpunkt, ebenso wie weitere Mischungen der beiden Ecken."
    )
    t_pct = st.slider("Lage zwischen den Ecken (0 % = Käuferecke, 100 % = Verkäuferecke)", 0, 100, 50, key="t_slider", format="%d %%")
    buyer_c = np.rint(res.buyer * C.SCALE).astype(np.int64)
    seller_c = np.rint(res.seller * C.SCALE).astype(np.int64)
    point_c = (100 - t_pct) * buyer_c + t_pct * seller_c                    # in Einheiten von 1/100 Hundertstel: exakt
    point = point_c / 100.0 / C.SCALE
    st.plotly_chart(V.build_core_ranges(names, a, low, high, point=point), width="stretch", key="chart_core")
    st.dataframe(
        pd.DataFrame({
            "Spieler": names, "Minimum im Kern (€)": [ev.de(x) for x in low], "Maximum im Kern (€)": [ev.de(x) for x in high],
            f"Punkt bei {t_pct} % (€)": [ev.de(x) for x in point],
        }),
        hide_index=True, width="stretch",
    )
    point_in_core = G.in_core_pairs(res.wi, point_c, denom=100)
    n_block_point = len(G.blocking_pairs(res.wi, point_c, denom=100))
    _show_verdict(ev.verdict_slider_point(point_in_core, t_pct, n_block_point))
    _show_verdict(ev.verdict_core_span(res.v_n, float(res.width.sum()), n_free, a + b))
    dim = _dimension(S.make_market(a, b, dist, int(q), int(seed)))
    st.caption(
        f"Dimension des Kerns: {dim} (ein Kern ohne Struktur hätte höchstens {a + b - 1}). In jedem Kernpunkt gilt für jedes Paar der besten Zuordnung "
        "u + y = Paarwert: Der Kern hat kein Inneres."
    )

    st.markdown("**Verband: zwei Kernpunkte verbinden**")
    pick = st.number_input("Auswahl der beiden Kernpunkte (Zufallszahl)", 0, 999, 0, step=1, key="pick_input",
                           help="Zwei Ecken des Kerns zu zufälligen Zielvektoren (SplitMix64); eine andere Zahl wählt zwei andere Punkte.")
    pt_x, pt_z = _two_core_points(S.make_market(a, b, dist, int(q), int(seed)), int(pick))
    join, meet = K.join_meet(pt_x, pt_z, a)
    neg = K.negative_probe(pt_x, pt_z, a)

    def inc(x):
        """Kernpunkte aus dem LP sind Gleitkommawerte: Test in Hundertsteln mit Toleranz statt exakt."""
        return G.in_core_pairs(res.wi, np.asarray(x) * C.SCALE, 1, 1e-4)

    comparable = bool((pt_x[:a] >= pt_z[:a] - 1e-9).all() or (pt_z[:a] >= pt_x[:a] - 1e-9).all())
    st.dataframe(
        pd.DataFrame({
            "Spieler": names, "Punkt X (€)": [ev.de(x) for x in pt_x], "Punkt Z (€)": [ev.de(x) for x in pt_z],
            "Join (€)": [ev.de(x) for x in join], "Meet (€)": [ev.de(x) for x in meet], "Gegenprobe (€)": [ev.de(x) for x in neg],
        }),
        hide_index=True, width="stretch",
    )
    _show_verdict(ev.verdict_lattice(inc(join), inc(meet), inc(neg), comparable))

with tab_shapley:
    st.markdown("Der Shapley-Wert (exakt, über alle Koalitionen gerechnet) gegen die Spanne des Kerns. Ein Spieler außerhalb seiner Spanne zeigt, wo die Aufteilung nicht stabil sein kann.")
    status = []
    for idx in range(a + b):
        if res.phi[idx] > high[idx] + 1e-9:
            status.append("über dem Kern-Maximum")
        elif res.phi[idx] < low[idx] - 1e-9:
            status.append("unter dem Kern-Minimum")
        else:
            status.append("innerhalb der Spanne")
    st.plotly_chart(V.build_core_ranges(names, a, low, high, shapley=res.phi), width="stretch", key="chart_shapley")
    st.dataframe(
        pd.DataFrame({
            "Spieler": names, "Shapley-Wert (€)": [ev.de(x) for x in res.phi], "als Bruch (€)": [str(f) for f in res.phi_frac],
            "Spanne im Kern (€)": [f"{ev.de(lo)} bis {ev.de(hi)}" for lo, hi in zip(low, high)], "Lage": status,
        }),
        hide_index=True, width="stretch",
    )
    if res.blocking:
        st.markdown("**Blockierende Paare** (Lücke = Paarwert minus Summe der beiden Shapley-Werte):")
        st.dataframe(
            pd.DataFrame([{"Ladung": load_names[i], "Lkw": truck_names[j], "Paarwert (€)": ev.de(res.w[i, j]),
                           "beide bekommen zusammen (€)": ev.de(res.phi[i] + res.phi[a + j]), "Lücke (€)": ev.de(gap)} for i, j, gap in res.blocking]),
            hide_index=True, width="stretch",
        )
        dist_core = _distance(S.make_market(a, b, dist, int(q), int(seed)), tuple(float(x) for x in res.phi))
        st.caption(f"Kürzester Abstand zum Kern (Summe der Beträge je Spieler, per Linearprogramm): {ev.de(dist_core)} € = {ev.pct(dist_core / res.v_n)} von v(N).")
    with st.expander("Handrechnung: zwei Ladungen (10 € und 8 €) und ein Lkw"):
        hand = ev.hand_calculation()
        st.markdown("Der Shapley-Wert ist der Mittelwert der Beiträge über alle sechs Reihenfolgen. Beitrag = Zuwachs von v, wenn der Spieler eintritt.")
        st.dataframe(
            pd.DataFrame([{"Reihenfolge": o["order"], **{n: ev.de(float(c)) for n, c in zip(hand["names"], o["contrib"])}} for o in hand["orders"]]),
            hide_index=True, width="stretch",
        )
        st.markdown(
            "Mittel: " + ", ".join(f"{n} = {f} € ({ev.de(float(f))})" for n, f in zip(hand["names"], hand["shapley"])) + ". "
            "Der Kern verlangt dagegen: Lkw 8 bis 10 €, Ladung 1 0 bis 2 €, Ladung 2 genau 0 €. Ladung 2 bekommt im Shapley-Wert mehr als 0, der Lkw weniger als 8: "
            "Beide Paarungen (Lkw mit Ladung 1, Lkw mit Ladung 2) blockieren."
        )

with tab_prices:
    st.markdown("Zwei Verfahren, die einen Kernpunkt als **Preise** der Lkw finden: die aufsteigende Auktion und die Ungarische Methode.")
    st.markdown("**Aufsteigende Auktion** (Demange, Gale, Sotomayor)")
    st.markdown(
        "Alle Lkw starten bei Preis 0. In jeder Runde fragt jede Ladung ihre besten Lkw nach (Wert minus Preis, nur wenn größer als 0). Gibt es eine Menge von Lkw, "
        "die mehr Ladungen ausschließlich nachfragen, als sie Lkw hat, wird die kleinste solche Menge um 1 € teurer. Sonst endet die Auktion."
    )
    auction, buyer_int = _auction(S.make_market(a, b, dist, int(q), int(seed)))
    rounded = bool((res.wi % C.SCALE != 0).any())
    history = [t["price"] for t in auction["trace"]] + [auction["price"]]
    st.plotly_chart(V.build_auction(history, truck_names), width="stretch", key="chart_auction")
    if auction["trace"]:
        st.dataframe(
            pd.DataFrame([{"Runde": t["round"], "überfragte Lkw": ", ".join(truck_names[j] for j in t["set"]),
                           "Preise vor der Erhöhung (€)": ", ".join(str(int(p)) for p in t["price"])} for t in auction["trace"][:40]]),
            hide_index=True, width="stretch",
        )
        if len(auction["trace"]) > 40:
            st.caption(f"Die ersten 40 von {len(auction['trace'])} Runden.")
    same = bool(auction["converged"] and (auction["price"] == buyer_int[a:]).all())
    _show_verdict(ev.verdict_auction(auction["converged"], auction["rounds"], same, rounded))

    st.markdown("**Ungarische Methode**")
    st.markdown(
        "Die Ungarische Methode löst das Zuordnungsproblem und liefert dabei Potenziale, die Preise im Sinne des Dualproblems sind. Welche der vielen Preissysteme "
        "des Kerns sie liefert, hängt daran, welche Seite als Zeilen geführt wird (die Zeilenseite kommt einzeln an und hält ihr Potenzial möglichst klein)."
    )
    rows_side = st.radio("Zeilen =", ["Ladungen", "Lkw"], key="hung_rows", horizontal=True)
    hx = AU.hungarian_prices(res.wi, rows_side)
    h_in = G.in_core_pairs(res.wi, hx, 1)
    h_buyer = bool((hx == buyer_c).all())
    h_seller = bool((hx == seller_c).all())
    st.dataframe(
        pd.DataFrame({
            "Spieler": names, "Ungarische Methode (€)": [ev.de(x / C.SCALE) for x in hx], "Käuferecke (€)": [ev.de(x) for x in res.buyer],
            "Verkäuferecke (€)": [ev.de(x) for x in res.seller],
        }),
        hide_index=True, width="stretch",
    )
    _show_verdict(ev.verdict_hungarian(rows_side, h_in, h_buyer, h_seller))

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die „faire“ Aufteilung nach Shapley stabil?")
max_gap = max((g for _, _, g in res.blocking), default=0.0)
_show_verdict(ev.verdict_shapley_core(a, b, res.v_n, res.in_core, len(res.blocking), max_gap))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

DIST_NAMES = C.DISTS

st.subheader("🔬 Experiment 1: Wie oft liegt der Shapley-Wert im Kern?")
st.caption(
    f"Quadratische Märkte k × k, k = {C.E1_KS[0]} bis {C.E1_KS[-1]}, acht Zufallsverteilungen. Eine **Zelle** ist eine Kombination aus Verteilung und k (zum Beispiel „stetig, k = 3“); je Zelle {C.EXP_SEEDS} Zufalls-Seeds mit "
    f"{', '.join(str(C.E1_PER_SEED[k]) for k in C.E1_KS)} Instanzen je Seed (für k = {', '.join(map(str, C.E1_KS))}). Mittel über die Seeds, Instanzen ohne Wert ausgeschlossen."
)
out = _experiment("exp1", "Messreihe 1 rechnen", EX.shapley_share_table, "Rechnet alle Märkte (Shapley-Wert über alle Koalitionen) …")
if out:
    rows, secs = out
    st.plotly_chart(V.build_share(rows, DIST_NAMES), width="stretch", key="chart_exp1")
    table = {C.DISTS[d]: {} for d in dict.fromkeys(r["dist"] for r in rows)}
    for r in rows:
        table[C.DISTS[r["dist"]]][f"k = {r['k']}"] = f"{ev.pct(r['share'])} ({r['n_in']}/{r['n_total']}; Std {ev.pct(r['std'], 1)})"
    st.dataframe(pd.DataFrame(table).T, width="stretch")
    st.caption(f"Rechenzeit: {ev.de(secs, 1)} s. Jede Zelle der Tabelle (Verteilung, k) zeigt den Anteil (Märkte mit Shapley im Kern / gerechnete Märkte; Std über die Seeds).")
    _show_verdict(ev.verdict_share(rows))

st.markdown("---")

st.subheader("🔬 Experiment 2: Warum? Der Mechanismus")
st.caption(
    "Ein Markt heißt **entkoppelt**, wenn es eine vollständige Zuordnung aus gegenseitig besten Partnern gibt (jeder Paarwert ist das Maximum seiner Zeile und seiner Spalte). "
    "Die Definition ist streng: Bei Gleichständen kann der Shapley-Wert im Kern liegen, obwohl der Markt nicht entkoppelt ist (Beispiel: Paarwerte 0, 3 | 3, 4 €, die beste Zuordnung "
    "nutzt die beiden Einträge 3 und 3); solche Märkte zählen als „im Kern, nicht entkoppelt“. "
    "Gezählt wird, wie sich „Shapley im Kern“ und „entkoppelt“ zueinander verhalten, dazu, ob die Paargleichungen (Shapley-Auszahlungen der Paare der besten Zuordnung "
    f"summieren sich auf den Paarwert) mit „im Kern“ übereinstimmen. k = {', '.join(map(str, C.E2_KS))}, vier Verteilungen."
)
out = _experiment("exp2", "Messreihe 2 rechnen", EX.mechanism_table, "Rechnet die Märkte und prüft die Entkopplung …")
if out:
    rows, secs = out
    st.dataframe(
        pd.DataFrame([{
            "Verteilung": C.DISTS[r["dist"]], "k": r["k"], "im Kern, entkoppelt": r["n11"], "im Kern, nicht entkoppelt": r["n10"],
            "außerhalb, entkoppelt": r["n01"], "außerhalb, nicht entkoppelt": r["n00"], "Paargleichungen gelten, nicht im Kern": r["eq_not_core"], "im Kern, Paargleichungen verletzt": r["core_not_eq"], "Märkte": r["n_total"],
        } for r in rows]),
        hide_index=True, width="stretch",
    )
    st.caption(f"Rechenzeit: {ev.de(secs, 1)} s.")
    _show_verdict(ev.verdict_mechanism(rows))

st.markdown("---")

st.subheader("🔬 Experiment 3: Dünne und ungleiche Märkte")
st.caption(
    f"Links: k = {C.E3_K}, stetige Werte, Dichte = Anteil der Paare mit Wert ungleich 0. Rechts: ungleich große Märkte a × b, stetige Werte. "
    f"{C.EXP_SEEDS} Zufalls-Seeds mit je {C.E3_PER_SEED} bzw. {C.E3_MARKET_PER_SEED} Instanzen."
)
out = _experiment("exp3", "Messreihe 3 rechnen", lambda: (EX.density_table(), EX.unequal_market_table()), "Rechnet dünne und ungleiche Märkte …")
if out:
    (dens_rows, uneq_rows), secs = out
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(V.build_density(dens_rows), width="stretch", key="chart_exp3")
        st.dataframe(
            pd.DataFrame([{"Dichte (%)": r["q"], "Shapley im Kern": f"{ev.pct(r['share'])} ({r['n_in']}/{r['n_total']})", "ohne Wert ausgeschlossen": r["n_zero"]} for r in dens_rows]),
            hide_index=True, width="stretch",
        )
    with c2:
        st.dataframe(
            pd.DataFrame([{
                "Markt a × b": f"{r['a']} × {r['b']}", "Shapley im Kern": f"{r['n_in']}/{r['n_total']}", "Spieler mit Kern-Maximum 0": f"{r['with_zero_max']}/{r['n_total']}",
                "… dem Shapley > 0 zahlt": f"{r['zero_paid']}/{r['n_total']}", "blockierende Paare (Mittel)": ev.de(r["blocking_mean"]),
            } for r in uneq_rows]),
            hide_index=True, width="stretch",
        )
    st.caption(f"Rechenzeit: {ev.de(secs, 1)} s.")
    _show_verdict(ev.verdict_density(dens_rows))
    _show_verdict(ev.verdict_unequal(uneq_rows))

st.markdown("---")

st.subheader("🔬 Experiment 4: Wie breit ist der Kern bei großen Märkten?")
st.caption(
    f"Kernbreite = Abstand der beiden Ecken je Spieler, im Mittel über die Spieler, geteilt durch v(N)/k. k = {', '.join(map(str, C.E4_KS))} ohne Shapley-Wert (er braucht alle Koalitionen); "
    f"{C.EXP_SEEDS} Zufalls-Seeds mit je {C.E4_PER_SEED} Instanzen, drei Verteilungen. Die Ecken kommen aus den Beiträgen zur großen Koalition; für k bis 20 gegen ein Linearprogramm geprüft."
)
out = _experiment("exp4", "Messreihe 4 rechnen", EX.width_table, "Rechnet Ecken bis k = 80 …")
if out:
    rows, secs = out
    st.plotly_chart(V.build_width(rows, DIST_NAMES), width="stretch", key="chart_exp4")
    st.dataframe(
        pd.DataFrame([{
            "Verteilung": C.DISTS[r["dist"]], "k": r["k"], "mittlere Kernbreite": ev.de(r["width"]), "Spieler mit eindeutigem Betrag": ev.pct(r["unique"]), "Märkte": r["n"],
        } for r in rows]),
        hide_index=True, width="stretch",
    )
    st.caption(f"Rechenzeit: {ev.de(secs, 1)} s.")
    _show_verdict(ev.verdict_width(rows))

st.markdown("---")

st.subheader("🔬 Experiment 5: Verband, Auktion und Ungarische Methode gegen die Ecken")
st.caption(
    f"Je Zelle (hier: Marktform und Wertebereich) {C.EXP_SEEDS} Zufalls-Seeds mit {C.E5_PER_SEED} Instanzen; Märkte {', '.join(f'{x} × {y}' for x, y in C.E5_SHAPES)}, ganzzahlige Werte (0 bis 19, 0 bis 99, 0 bis 3). "
    f"Verband: je Markt {C.E5_PAIRS_PER_INSTANCE} Paare aus fünf zufälligen Kern-Ecken."
)
out = _experiment("exp5", "Messreihe 5 rechnen", EX.lattice_price_table, "Rechnet Kernpunkte, Auktionen und Ungarische Methode …")
if out:
    rows, secs = out
    st.dataframe(
        pd.DataFrame([{
            "Verteilung": C.DISTS[r["dist"]], "Markt": f"{r['a']} × {r['b']}", "Märkte": r["n"], "Join im Kern": f"{r['join_ok']}/{r['pairs']}",
            "Meet im Kern": f"{r['meet_ok']}/{r['pairs']}", "Gegenprobe im Kern": f"{r['neg_in']}/{r['pairs']}", "Auktion = Käuferecke": f"{r['auction_ok']}/{r['n']}",
            "Runden (Mittel / Max)": f"{ev.de(r['rounds_mean'], 1)} / {r['rounds_max']}", "Ungarisch (Ladungen) = Käuferecke": f"{r['hung_buyer']}/{r['n']}",
            "Ungarisch (Lkw) = Verkäuferecke": f"{r['hung_seller']}/{r['n']}",
        } for r in rows]),
        hide_index=True, width="stretch",
    )
    st.caption(f"Rechenzeit: {ev.de(secs, 1)} s.")
    _show_verdict(ev.verdict_lattice_prices(rows))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jede Ladung fährt mit höchstens einem Lkw, jeder Lkw nimmt höchstens eine Ladung** | Mit mehreren Partnern je Spieler (Bündel) gilt „der Kern ist nie leer“ nicht mehr allgemein; hier nicht untersucht. | `standortspiel-demo` (dort kann der Kern leer sein) |
| **Paarwerte hängen nur am Paar** | Wirken Paare aufeinander (Touren, Kapazitäten), gibt es kein Zuordnungsspiel mehr. | `kern-demo`, `bestandsallianz-demo` |
| **Nutzen ist übertragbar (Geld)** | Ohne Geldübertragung bleibt die Frage nach stabilen Paarungen mit Vorlieben. | `gale-shapley-demo`, `stabile-mitbewohner-demo` |
| **Kein Nukleolus** | Eine „gewählte“ Aufteilung im Kern (Nukleolus) zeigt diese Demo nicht. | `kern-demo` |
| **Zweiseitiger Markt** | Auf allgemeinen Graphen (jeder mit jedem) ist ein stabiles Ergebnis nicht garantiert. | `austauschnetz-demo` |
| **Ein Paar, ein Wert (kein Netzfluss)** | Teilen sich Spieler Kanten mit Kapazität, entsteht ein Flussspiel. | `flussspiel-demo` |
| **Separable Werte (Klickrate mal Gebot)** | Dann ist die Käuferecke der VCG-Preis der Positionsauktion (Literatur; in dieser Demo nicht nachgerechnet); diese Demo zeigt allgemeine Paarwerte. | `positionsauktion-demo` |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(T.ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/zuordnungsspiel-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Markt aus $a$ Ladungen und $b$ Lkw mit Paarwerten; gesucht sind stabile Aufteilungen des Gesamtwerts (der Kern) und der Shapley-Wert.

| Symbol | Bedeutung |
|---|---|
| $L$, $T$ | Menge der Ladungen ($\lvert L\rvert = a$) und der Lkw ($\lvert T\rvert = b$); $N = L \cup T$, $n = a + b$ |
| $w_{ij} \ge 0$ | Paarwert von Ladung $i$ und Lkw $j$ |
| $v(S)$ | Wert der Koalition $S \subseteq N$ |
| $u_i$, $y_j$ | Auszahlung der Ladung $i$ (Überschuss) und des Lkw $j$ (Preis) |
| $\varphi_i$ | Shapley-Wert des Spielers $i$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/zuordnungsspiel-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Markt aus a Ladungen und b Lkw mit Paarwerten; gesucht sind stabile Aufteilungen des Gesamtwerts (der Kern) und der Shapley-Wert.

| Symbol | Bedeutung |
|---|---|
| $L$, $T$ | Menge der Ladungen ($\lvert L\rvert = a$) und der Lkw ($\lvert T\rvert = b$); $N = L \cup T$, $n = a + b$ |
| $w_{ij} \ge 0$ | Paarwert von Ladung $i$ und Lkw $j$ |
| $v(S)$ | Wert der Koalition $S \subseteq N$ |
| $u_i$, $y_j$ | Auszahlung der Ladung $i$ (Überschuss) und des Lkw $j$ (Preis) |
| $\varphi_i$ | Shapley-Wert des Spielers $i$ |

**Wert einer Koalition** (bestes Matching in $S$, $x_{ij}$ gleich 1, wenn $i$ und $j$ gepaart sind):
        """
    )
    st.markdown(r"$$v(S)=\max_{x\ge 0}\ \sum_{i\in L\cap S}\ \sum_{j\in T\cap S} w_{ij}\,x_{ij}$$")
    st.markdown(r"$$\sum_{j\in T\cap S} x_{ij}\le 1\quad (i\in L\cap S)$$")
    st.markdown(r"$$\sum_{i\in L\cap S} x_{ij}\le 1\quad (j\in T\cap S)$$")
    st.markdown("**Kern** (stabile Aufteilungen von $v(N)$); der Test über die Paare genügt, die Tests prüfen ihn gegen alle Koalitionen:")
    st.markdown(r"$$\sum_{i\in L} u_i+\sum_{j\in T} y_j=v(N)$$")
    st.markdown(r"$$u_i+y_j\ \ge\ w_{ij}\quad (i\in L,\ j\in T)$$")
    st.markdown(r"$$u_i\ \ge\ 0,\quad y_j\ \ge\ 0$$")
    st.markdown(
        "Das ist das Dualproblem des Matchings: Der Kern ist die Menge der optimalen Duallösungen, also nie leer. Für jedes Paar der besten Zuordnung gilt in jedem Kernpunkt Gleichheit. "
        "**Preise:** Ein Kernpunkt ist ein Preissystem, in dem jede Ladung ihren besten Lkw nimmt:"
    )
    st.markdown(r"$$u_i=\max\Bigl(0,\ \max_{j\in T}\,(w_{ij}-y_j)\Bigr)$$")
    st.markdown("**Maximum im Kern** (gemessen, siehe README; die Quellen nennen es als Satz, hier nicht daraus belegt):")
    st.markdown(r"$$\max_{x\in\text{Kern}} x_i = v(N)-v(N\setminus\{i\})$$")
    st.markdown("**Verband:** Zu zwei Kernpunkten $(u,y)$ und $(u',y')$ gehören Join und Meet; beide liegen im Kern (gemessen):")
    st.markdown(r"$$\text{Join}=\bigl(\max(u,u'),\ \min(y,y')\bigr)$$")
    st.markdown(r"$$\text{Meet}=\bigl(\min(u,u'),\ \max(y,y')\bigr)$$")
    st.markdown("**Shapley-Wert** (Mittel der Beiträge über alle Reihenfolgen):")
    st.markdown(r"$$\varphi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{\lvert S\rvert!\,(n-\lvert S\rvert-1)!}{n!}\,\bigl(v(S\cup\{i\})-v(S)\bigr)$$")
    st.markdown("**Aufsteigende Auktion:** Start $y_j=0$; ist eine Menge $M$ von Lkw überfragt (mehr Ladungen fragen nur Lkw aus $M$ nach, als $M$ Lkw hat), so gilt für jeden Lkw in $M$:")
    st.markdown(r"$$y_j \leftarrow y_j+1\quad (j\in M)$$")
    st.markdown(
        "Das Modell ist vollständig bis auf zwei Dinge, die hier nur beschrieben werden: die Menge der Nachfragen je Ladung (die besten Lkw bei positivem Überschuss) "
        "und die Wahl der kleinsten überfragten Menge."
    )
    st.markdown(T.LITERATUR_HEAD)
    st.markdown("\n".join(T.LITERATUR))

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html)."
)
