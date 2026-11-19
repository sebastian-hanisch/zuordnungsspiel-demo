"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Verteilungen und Presets.

Rechenaufwand am Regler-Maximum (a = b = 7, n = 14 Spieler): Teilmengen-Tabelle 2^14 = 16 384 Koalitionswerte, exakter Shapley-Wert
über alle Koalitionen, O(n * 2^n); Kern über die a*b + 2n = 77 Paarbedingungen; Auktion höchstens 2^b - 1 = 127 Mengen je Runde."""

A_MIN, A_MAX, DEFAULT_A = 1, 7, 3          # Ladungen (Käuferseite)
B_MIN, B_MAX, DEFAULT_B = 1, 7, 3          # Lkw (Verkäuferseite); a + b <= 14, also gibt es immer einen exakten Shapley-Wert
Q_MIN, Q_MAX, DEFAULT_Q = 5, 100, 100      # Dichte in Prozent: Anteil der Paarwerte, die von 0 verschieden sein dürfen
SEED_MAX, DEFAULT_SEED = 999999, 0
SHAPLEY_MAX_PLAYERS = 14                   # 2^14 Koalitionen: darüber gibt es hier keinen Shapley-Wert (Aufzählung)
SCALE = 100                                # Paarwerte werden in Hundertsteln (Cent) ganzzahlig gerechnet: Shapley-Wert und Kerntest sind dann exakt
PARETO_CAP = 300.0                         # Pareto-Werte werden bei 300 gekappt (Wahrscheinlichkeit des Kappens (30)^(-1,5) = 0,6 %)
AUCTION_ROUND_CAP = 5000                   # Obergrenze der Auktionsrunden (Schritt 1 auf ganzzahligen Werten bis 300: höchstens b * 300 Runden)

# Verteilungen der Paarwerte: Schlüssel -> Anzeigename. Zufallsverteilungen nutzen Zufalls-Seed und Dichte, Handbeispiele nicht.
RANDOM_DISTS = {
    "int20": "ganzzahlig 0 bis 19",
    "int100": "ganzzahlig 0 bis 99",
    "int3": "ganzzahlig 0 bis 3",
    "bin": "0 oder 1",
    "unif": "stetig gleichverteilt 0 bis 100",
    "pareto": "Pareto ab 10, gekappt bei 300",
    "prod": "Rang eins: 50 · s_i · t_j (Zufallswerte s, t)",
    "geo": "Geografie: Erlös 40 bis 100 minus 60 · Abstand",
}
HAND_DISTS = {
    "h_eins": "Handbeispiel: alle Paarwerte 1",
    "h_spalte": "Handbeispiel: Wert hängt nur an der Ladung (10, 8, 6, …)",
    "h_diag": "Handbeispiel: Diagonale 10, 8, 6, … sonst 1",
    "h_rang": "Handbeispiel: Rang eins mit s_i = a − i + 1, t_j = b − j + 1 (i, j ab 1)",
    "h_getrennt": "Handbeispiel: Diagonale 10, sonst 0 (getrennte Paare)",
}
DISTS = {**RANDOM_DISTS, **HAND_DISTS}
DEFAULT_DIST = "int20"

# Experimente: feste Raster, Zufalls-Seeds und Instanzzahlen (Mittel über vier Zufalls-Seeds, Std über die Seeds)
EXP_SEEDS = 4
E1_KS = (2, 3, 4, 5, 6)
E1_PER_SEED = {2: 100, 3: 100, 4: 50, 5: 25, 6: 10}
E2_KS = (2, 3, 4)
E2_PER_SEED = {2: 100, 3: 100, 4: 50}
E3_DENSITIES = (10, 20, 30, 50, 80, 100)
E3_K = 4
E3_PER_SEED = 50
E3_MARKETS = ((1, 2), (1, 4), (2, 3), (3, 5), (5, 3))
E3_MARKET_PER_SEED = 50
E4_KS = (3, 6, 10, 20, 40, 80)
E4_PER_SEED = 3
E5_SHAPES = ((3, 3), (5, 5), (4, 6), (6, 4))
E5_PER_SEED = 8
E5_PAIRS_PER_INSTANCE = 10

COLOR_LOAD = "#2F6B65"       # Ladungen (Käuferseite)
COLOR_TRUCK = "#C8741B"      # Lkw (Verkäuferseite)
COLOR_SHAPLEY = "#7a3b8f"
COLOR_RANGE = "rgba(76,120,168,0.35)"
COLOR_LINE = "#4c78a8"
COLOR_BAD = "#c0392b"
COLOR_OK = "#2e8b57"

PRESETS = {
    "Zufallsmarkt 3 × 3 (0 bis 19)": {"a": 3, "b": 3, "dist": "int20", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Eine Ladung, zwei Lkw": {"a": 1, "b": 2, "dist": "h_eins", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Zwei Ladungen, ein Lkw (10 und 8)": {"a": 2, "b": 1, "dist": "h_spalte", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Entkoppelt 3 × 3 (Diagonale 10, 8, 6)": {"a": 3, "b": 3, "dist": "h_diag", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Rang eins 3 × 3 (s = t = 3, 2, 1)": {"a": 3, "b": 3, "dist": "h_rang", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Zwei getrennte Paare (10 und 10)": {"a": 2, "b": 2, "dist": "h_getrennt", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Geografie 3 × 3": {"a": 3, "b": 3, "dist": "geo", "q": DEFAULT_Q, "seed": DEFAULT_SEED},
}
DEFAULT_PRESET = next(iter(PRESETS))
PRESET_HELP = {
    "Zufallsmarkt 3 × 3 (0 bis 19)": "Drei Ladungen, drei Lkw, ganzzahlige Paarwerte von 0 bis 19 € mit Zufalls-Seed 0. Der Shapley-Wert liegt außerhalb des Kerns: drei Paare blockieren.",
    "Eine Ladung, zwei Lkw": "Alle Paarwerte 1 €. Der Kern ist ein einziger Punkt (die Ladung bekommt alles), der Shapley-Wert teilt dagegen 2/3, 1/6, 1/6.",
    "Zwei Ladungen, ein Lkw (10 und 8)": "Handrechnung: Shapley-Wert 7/3, 4/3 und 19/3 €. Der Kern verlangt für den Lkw 8 bis 10 €, für die zweite Ladung genau 0 €.",
    "Entkoppelt 3 × 3 (Diagonale 10, 8, 6)": "Jede Ladung hat einen eindeutigen besten Partner. Der Shapley-Wert teilt jedes Paar in der Mitte (5, 4, 3 € je Seite) und liegt im Kern.",
    "Rang eins 3 × 3 (s = t = 3, 2, 1)": "Paarwert = s_i · t_j. Der Shapley-Wert liegt außerhalb des Kerns: ein Paar (die beiden besten) blockiert mit einer Lücke von 0,90 €.",
    "Zwei getrennte Paare (10 und 10)": "Zwei Paare, die nur miteinander etwas schaffen. Der Shapley-Wert (5 € je Spieler) liegt im Kern, und der Kern lässt Spielraum.",
    "Geografie 3 × 3": "Paarwert = Erlös minus 60 € mal Abstand, Zufalls-Seed 0. Der Shapley-Wert liegt außerhalb des Kerns: drei Paare blockieren.",
}
