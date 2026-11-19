"""Gemessene Werte der Experimente (Stand des Baus), aus `zsp_experiments` mit den festen Zufalls-Seeds erzeugt; `test_claims.py` rechnet sie neu und vergleicht.

Format je Zeile siehe Kommentar vor dem Wörterbuch. Zahlen aus Ganzzahl-Rechnung sind exakt; nur Pareto (Potenz) und LP-Ecken bekommen Bänder im Test."""

# Schlüssel 'Verteilung|k': [Märkte mit Shapley im Kern, gerechnete Märkte, ohne Wert ausgeschlossen, Anteil (Mittel über 4 Zufalls-Seeds), Std über die Seeds, mittlere Zahl blockierender Paare außerhalb]
E1 = {'int20|2': [146, 400, 0, 0.365, 0.016583, 1.3346],
 'int20|3': [24, 400, 0, 0.06, 0.007071, 2.0984],
 'int20|4': [0, 200, 0, 0.0, 0.0, 3.085],
 'int20|5': [0, 100, 0, 0.0, 0.0, 4.0],
 'int20|6': [0, 40, 0, 0.0, 0.0, 4.575],
 'int100|2': [142, 400, 0, 0.355, 0.058949, 1.3333],
 'int100|3': [18, 400, 0, 0.045, 0.018028, 2.1963],
 'int100|4': [1, 200, 0, 0.005, 0.00866, 3.0201],
 'int100|5': [0, 100, 0, 0.0, 0.0, 3.89],
 'int100|6': [0, 40, 0, 0.0, 0.0, 5.1],
 'int3|2': [210, 398, 2, 0.527828, 0.041049, 1.266],
 'int3|3': [76, 400, 0, 0.19, 0.024495, 2.3117],
 'int3|4': [4, 200, 0, 0.02, 0.014142, 3.3112],
 'int3|5': [1, 100, 0, 0.01, 0.017321, 4.0909],
 'int3|6': [0, 40, 0, 0.0, 0.0, 5.1],
 'bin|2': [263, 366, 34, 0.719413, 0.048232, 2.0],
 'bin|3': [160, 400, 0, 0.4, 0.069642, 3.1],
 'bin|4': [30, 200, 0, 0.15, 0.022361, 4.0588],
 'bin|5': [0, 100, 0, 0.0, 0.0, 5.26],
 'bin|6': [0, 40, 0, 0.0, 0.0, 7.35],
 'unif|2': [129, 400, 0, 0.3225, 0.031125, 1.3063],
 'unif|3': [18, 400, 0, 0.045, 0.015, 2.2592],
 'unif|4': [0, 200, 0, 0.0, 0.0, 3.065],
 'unif|5': [0, 100, 0, 0.0, 0.0, 4.01],
 'unif|6': [0, 40, 0, 0.0, 0.0, 4.725],
 'pareto|2': [147, 400, 0, 0.3675, 0.020463, 1.1383],
 'pareto|3': [32, 400, 0, 0.08, 0.025495, 1.7745],
 'pareto|4': [5, 200, 0, 0.025, 0.016583, 2.4564],
 'pareto|5': [0, 100, 0, 0.0, 0.0, 3.02],
 'pareto|6': [0, 40, 0, 0.0, 0.0, 3.8],
 'prod|2': [0, 400, 0, 0.0, 0.0, 1.5375],
 'prod|3': [0, 400, 0, 0.0, 0.0, 3.025],
 'prod|4': [0, 200, 0, 0.0, 0.0, 5.21],
 'prod|5': [0, 100, 0, 0.0, 0.0, 7.66],
 'prod|6': [0, 40, 0, 0.0, 0.0, 10.5],
 'geo|2': [70, 400, 0, 0.175, 0.018028, 1.4273],
 'geo|3': [3, 400, 0, 0.0075, 0.008292, 2.471],
 'geo|4': [0, 200, 0, 0.0, 0.0, 3.85],
 'geo|5': [0, 100, 0, 0.0, 0.0, 5.02],
 'geo|6': [0, 40, 0, 0.0, 0.0, 5.95]}

# 'Verteilung|k': [im Kern & entkoppelt, im Kern & nicht entkoppelt, außerhalb & entkoppelt, außerhalb & nicht, Paargleichungen gelten aber nicht im Kern, im Kern aber Paargleichungen verletzt, Märkte]
E2 = {'unif|2': [127, 0, 0, 273, 0, 0, 400],
 'unif|3': [27, 0, 24, 349, 0, 0, 400],
 'unif|4': [2, 0, 5, 193, 0, 0, 200],
 'int20|2': [161, 2, 0, 237, 2, 0, 400],
 'int20|3': [22, 0, 37, 341, 0, 0, 400],
 'int20|4': [0, 0, 6, 194, 0, 0, 200],
 'geo|2': [60, 0, 0, 340, 0, 0, 400],
 'geo|3': [2, 0, 6, 392, 0, 0, 400],
 'geo|4': [0, 0, 0, 200, 0, 0, 200],
 'pareto|2': [129, 0, 0, 271, 0, 0, 400],
 'pareto|3': [30, 0, 15, 355, 0, 0, 400],
 'pareto|4': [5, 0, 2, 193, 0, 0, 200]}

# 'Dichte in %': [im Kern, gerechnete Märkte, ausgeschlossen, Anteil]
E3A = {'10': [94, 154, 46, 0.610052],
 '20': [73, 194, 6, 0.376169],
 '30': [30, 200, 0, 0.15],
 '50': [5, 200, 0, 0.025],
 '80': [0, 200, 0, 0.0],
 '100': [1, 200, 0, 0.005]}

# 'a x b': [im Kern, Märkte, Märkte mit Spieler ohne Beitrag, davon Shapley zahlt ihm > 0, blockierende Paare im Mittel]
E3B = {'1x2': [0, 200, 200, 200, 1.265],
 '1x4': [0, 200, 200, 200, 1.69],
 '2x3': [0, 200, 200, 200, 2.345],
 '3x5': [0, 200, 200, 200, 4.5],
 '5x3': [0, 200, 200, 200, 4.465]}

# 'Verteilung|k': [mittlere Kernbreite, Anteil Spieler mit eindeutigem Betrag, Märkte]
E4 = {'unif|3': [0.811274, 0.0, 12],
 'unif|6': [0.684204, 0.0, 12],
 'unif|10': [0.734076, 0.0, 12],
 'unif|20': [0.8214, 0.0, 12],
 'unif|40': [0.841528, 0.0, 12],
 'unif|80': [0.921302, 0.0, 12],
 'int20|3': [0.774423, 0.0, 12],
 'int20|6': [0.63317, 0.0, 12],
 'int20|10': [0.714644, 0.0, 12],
 'int20|20': [0.81374, 0.0, 12],
 'int20|40': [0.88517, 0.0, 12],
 'int20|80': [0.933301, 0.0, 12],
 'geo|3': [0.733463, 0.0, 12],
 'geo|6': [0.444962, 0.041667, 12],
 'geo|10': [0.280128, 0.016667, 12],
 'geo|20': [0.365277, 0.0, 12],
 'geo|40': [0.257808, 0.0, 12],
 'geo|80': [0.156899, 0.004167, 12]}

# 'Verteilung|axb': [Märkte, Paare, Join im Kern, Meet im Kern, Gegenprobe im Kern, Auktion = Käuferecke, Runden Mittel, Runden Max, Ungarisch im Kern, Ungarisch (Ladungen) = Käuferecke, Ungarisch (Lkw) = Verkäuferecke]
E5 = {'int20|3x3': [32, 320, 320, 320, 49, 32, 3.0, 11, 32, 32, 32],
 'int20|5x5': [32, 320, 320, 320, 24, 32, 6.5, 20, 32, 32, 32],
 'int20|4x6': [32, 320, 320, 320, 52, 32, 2.125, 8, 32, 32, 32],
 'int20|6x4': [32, 320, 320, 320, 35, 32, 25.7188, 32, 32, 32, 32],
 'int100|3x3': [32, 320, 320, 320, 50, 32, 17.2812, 65, 32, 32, 32],
 'int100|5x5': [32, 320, 320, 320, 17, 32, 34.9688, 83, 32, 32, 32],
 'int100|4x6': [32, 320, 320, 320, 31, 32, 8.9375, 36, 32, 32, 32],
 'int100|6x4': [32, 320, 320, 320, 26, 32, 139.0, 192, 32, 32, 32],
 'int3|3x3': [32, 320, 320, 320, 78, 32, 0.625, 3, 32, 32, 32],
 'int3|5x5': [32, 320, 320, 320, 46, 32, 0.8438, 2, 32, 32, 32],
 'int3|4x6': [32, 320, 320, 320, 143, 32, 0.0938, 2, 32, 32, 32],
 'int3|6x4': [32, 320, 320, 320, 142, 32, 4.5, 7, 32, 32, 32]}

# 'Verteilung|k': [Märkte mit Dimension des Kerns = k, Märkte] (je vier Zufalls-Seeds mit acht Instanzen, Tag 9)
DIMENSION = {'unif|2': [32, 32],
 'unif|3': [32, 32],
 'unif|4': [32, 32],
 'int20|2': [30, 32],
 'int20|3': [29, 32],
 'int20|4': [27, 32],
 'geo|2': [31, 32],
 'geo|3': [30, 32],
 'geo|4': [28, 32]}

# Shapley im Kern bei k = 2, stetig, vier Zufalls-Seeds mit je 400 Instanzen: [im Kern, Märkte]
UNIF2_BIG = [515, 1600]

# Kontingenz bei k = 2, stetig, 4 x 400 Instanzen (n11, n10, n01, n00)
UNIF2_MECH = [521, 0, 0, 1079]

# alle 2 x 2 Märkte mit ganzzahligen Werten 0 bis 3 und v(N) > 0: [Märkte, Shapley außerhalb des Kerns]
SMALL = [255, 120]

