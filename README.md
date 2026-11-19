# Wer fährt mit wem – das Zuordnungsspiel – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-zuordnungsspiel-demo.streamlit.app/)**

Teil der Linie [Koalitionen und Teilen](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html). Vorläufer, die hier nicht umgezogen, sondern verlinkt werden: `shapley-demo` und `kern-demo` (Kostenteilung von Touren), `hungarian-demo` und `lp-dualitaet-demo` (dieselbe Mathematik ohne Spiel), `gale-shapley-demo` und `stabile-mitbewohner-demo` (Paarung ohne Geldübertragung). **Die Demo setzt `austauschnetz-demo` fort** (Matchingspiel auf allgemeinen Graphen mit ganzzahligen Kantenwerten, nach dessen README der Shapley-Wert dort selten im Kern liegt): hier zweiseitig, mit stetigen Werten, mit dem Kern als Preisspanne samt Verband, mit einem Mechanismus für den Befund und mit der Kernbreite bis 80 Spielern je Seite. Für den Spezialfall separabler Werte (Positionsauktion, VCG) siehe `positionsauktion-demo`; Nachbarn der Karte sind außerdem `bestandsallianz-demo` und `auftragsallianz-demo`.

## Kernfrage

*Spediteure und Lkw bilden Paare mit einem Paarwert. Welche Aufteilung des Gesamtgewinns ist stabil, wie groß ist der Spielraum, und warum verfehlt die „faire“ Aufteilung nach Shapley den Kern bei größeren Märkten mit stetigen oder breit gestreuten ganzzahligen Werten fast immer (bei 0/1-Werten und dünnen Märkten deutlich seltener)?* Die App rechnet zu jedem Markt (bis 7 Ladungen und 7 Lkw) den Shapley-Wert exakt, die beiden Ecken des Kerns, blockierende Paare und Preise, und beantwortet die Frage live in einem Verdikt. Fünf Experimente messen die Antwort über Verteilungen und Größen.

## Modell

Ladungen (Käuferseite, Überschuss u) und Lkw (Verkäuferseite, Preis y); ein Paar schafft den Paarwert w_ij ≥ 0 (gedacht als Erlös minus Fahrkosten in €, eine Modellannahme). Der Wert einer Koalition ist ihr bestes Matching. Der **Kern** besteht aus den Aufteilungen von v(N) mit u_i + y_j ≥ w_ij für alle Paare und u, y ≥ 0; er ist das Dualproblem des Matchings und damit nie leer. Alles wird ganzzahlig in Hundertsteln gerechnet; der Shapley-Wert ist exakt (Brüche, n! mal den Wert als int64), Kerntests laufen ohne Toleranz. Formeln: Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Shapley-Wert** über alle 2^n Koalitionen (n ≤ 14); **Kern** über die a · b Paarbedingungen (Orakel: alle 2^n Koalitionen).
- **Ecken** in geschlossener Form: das Kern-Maximum jedes Spielers ist v(N) − v(N ohne ihn), die Gegenseite bekommt in jedem optimalen Paar den Rest (n + 1 Zuordnungsprobleme, ganzzahlig); Gegenprobe per Linearprogramm (HiGHS) und über alle Koalitionen.
- **Verband**: zwei Ecken des Kerns zu zufälligen Zielvektoren (LP), Join (größeres u, kleineres y) und Meet; Gegenprobe (größeres u und größeres y).
- **Aufsteigende Auktion** (ganze Euro, Schritt 1, kleinste überfragte Menge um 1 erhöhen) und **Ungarische Methode** (Potenziale, Zeilen = Ladungen oder Lkw; quadratisch aufgefüllt bei a ≠ b).
- Zufall nur über SplitMix64 (`zsp_rng.py`, wörtliche Kopie aus `ab-test-demo`, Kopiertest); Instanzen je Zelle (Kombination aus Verteilung und Größe) aus einem Zellschlüssel (`cell_seed`). Mittel und Standardabweichung der Experimente laufen über vier Zufalls-Seeds; Instanzen ohne Wert (v(N) = 0) werden ausgeschlossen und gezählt.
- **Entkoppelt** (nur quadratische Märkte): es gibt eine vollständige Zuordnung gegenseitig bester Partner (jeder Paarwert ist das Maximum seiner Zeile und seiner Spalte); gesucht per Matching im Graphen dieser Kanten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind mit dem Code der Demo gemessen (Zufalls-Seeds und Instanzzahlen in der Spalte „Voraussetzung“) und stehen in `tests/test_claims.py` (gemessene Werte in `tests/claims_expected.py`). Ganzzahlrechnung wird exakt verglichen; Bänder gelten nur für die Pareto-Verteilung, die Eckenwahl des LP und die Paargleichungen bei Gleichstand.

| Frage | Befund | Voraussetzung | Test |
|---|---|---|---|
| Handrechnung: eine Ladung, zwei Lkw (Paarwerte 1 €) | Kern = ein Punkt (1, 0, 0); Shapley (2/3, 1/6, 1/6) liegt außerhalb | exakt | `test_hand_one_load_two_trucks_core_is_a_point_and_shapley_splits` |
| Handrechnung: zwei Ladungen (10 €, 8 €), ein Lkw | Shapley (7/3, 4/3, 19/3) €; Kern: Lkw 8 bis 10, Ladung 1 von 0 bis 2, Ladung 2 genau 0; zwei blockierende Paare (Lücken 4/3 und 1/3 €); Mittel der Ecken (1, 0, 9) ist **nicht** der Shapley-Wert | exakt | `test_hand_two_loads_one_truck_numbers_and_the_midpoint_myth` |
| Entkoppelt 3 × 3 (Diagonale 10, 8, 6, sonst 1) | Shapley (5, 4, 3 \| 5, 4, 3) im Kern | exakt | `test_hand_decoupled_three_by_three_halves_each_pair_and_is_in_the_core` |
| Rang eins 3 × 3 (s = t = 3, 2, 1) | Shapley (4,05; 2,13; 0,82 \| gleich) außerhalb; ein blockierendes Paar, Lücke 0,90 € | exakt | `test_hand_rank_one_three_by_three_has_one_blocking_pair_of_gap_ninety_cents` |
| Zwei getrennte Paare (10 €, 10 €) | Shapley 5 € je Spieler im Kern; Spielraum 40 € (Summe der Breiten) | exakt | `test_hand_two_separate_pairs_have_shapley_five_in_the_core_and_room_to_move` |
| Kern nie leer, ohne Inneres (Least-Core-Wert 0)? | Ja: Wert 0 bis auf 7,6e-14 in 344 Märkten (2 × 2 bis 4 × 4 und gemischt, acht Verteilungen) | Scratch-Lauf `tools/orakel_lauf.py` (nicht im Test); im Test 30+ Märkte | `test_least_core_value_is_zero_so_the_core_has_no_interior` |
| Dimension des Kerns (stetig) | genau k in 32 von 32 Märkten je k = 2, 3, 4; ganzzahlig 0 bis 19: 30, 29, 27 von 32; Geografie: 31, 30, 28 von 32 | quadratisch, vier Zufalls-Seeds mit je acht Instanzen | `test_core_dimension_counts` |
| Shapley im Kern, k = 2 | stetig 129 von 400 (32,3 %), Handrechnung 1/3 (1 600 Märkte: 515, Erwartung 533 ± 19); ganzzahlig 0 bis 19 146 von 400; Pareto 147 von 400; 0/1 263 von 366 (34 ohne Wert ausgeschlossen); **Rang eins 0 von 400**; Geografie 70 von 400 | je 4 Zufalls-Seeds mit 100 Instanzen | `test_e1_every_cell_matches_the_pinned_counts`, `test_e1_stetig_share_at_k2_is_a_third_by_hand_and_by_count` |
| Shapley im Kern, k = 3 | 0,0 % (Rang eins) bis 40,0 % (0/1); stetig 4,5 % (18 von 400), ganzzahlig 0 bis 19 6,0 % (24 von 400), ganzzahlig 0 bis 99 4,5 % (18 von 400), Pareto 8,0 % (32 von 400), Geografie 0,75 % (3 von 400); ganzzahlig 0 bis 3 19,0 % (76 von 400) | je 4 Seeds mit 100 Instanzen | `test_e1_every_cell_matches_the_pinned_counts` |
| Shapley im Kern, k ≥ 4 | in 19 von 24 Zellen (acht Verteilungen, k = 4, 5, 6) keine einzige Instanz (bei k = 6 nur 40 Instanzen je Zelle); Ausnahmen: 0/1 (15,0 % bei k = 4, 0 bei k = 5, 6), ganzzahlig 0 bis 3 (2,0 % und 1,0 %), Pareto (2,5 % bei k = 4), ganzzahlig 0 bis 99 (0,5 % bei k = 4) | k = 4: 4 × 50, k = 5: 4 × 25, k = 6: 4 × 10 Instanzen | `test_e1_fast_never_from_k3_zero_cells_and_the_01_exception` |
| Rang eins nie entkoppelt | 0 von 400 Märkten 3 × 3 sind entkoppelt, passend zu 0 im Kern | 4 Seeds × 100 | `test_e1_rank_one_is_never_decoupled` |
| Mechanismus: im Kern ⇒ entkoppelt? | Ja bis auf 2 von 567 Märkten mit Shapley im Kern (ganzzahlig 0 bis 19, k = 2); 565 von 567 sind entkoppelt. Ein Satz ist das nicht: bei ganzzahligen Werten gibt es Ausnahmen auch bei k = 3, z. B. 3, 10, 9 \| 7, 2, 4 \| 5, 9, 0 (Shapley im Kern, aber nicht entkoppelt; mit dem Code der Demo nachgerechnet). Gleichstände: In allen vier genannten Fällen (die zwei Märkte bei k = 2, 18, 15 \| 15, 4 und 13, 9 \| 14, 13, das 3 × 3-Beispiel und der Markt 0, 3 \| 3, 4) hat die beste Zuordnung zwei gleiche Einträge; ob das die Ursache ist, wurde nicht untersucht. „Entkoppelt“ ist hier streng definiert (Maximum der Zeile und der Spalte) und bleibt es: mit Gleichständen ist es für „Shapley im Kern“ nicht notwendig, und die Zählungen oben gelten für diese Definition | 4 000 Märkte (vier Verteilungen, k = 2 bis 4) | `test_e2_every_cell_matches_the_pinned_counts`, `test_e2_in_the_core_implies_decoupled_except_two_integer_cases_at_k2` |
| Mechanismus: entkoppelt ⇒ im Kern? | k = 2 stetig: im Kern ⇔ entkoppelt in 1 600 von 1 600 (521 beide, 1 079 keines); ab k = 3 nur notwendig: 95 entkoppelte Märkte mit Shapley außerhalb, 0 im Kern ohne entkoppelt | 4 × 400 Märkte (k = 2); k = 3, 4: 4 000er Bilanz | `test_e2_continuous_k2_in_core_iff_decoupled_in_1600_of_1600`, `test_e2_from_k3_decoupled_is_necessary_not_sufficient` |
| Paargleichungen | Im Kern gelten sie in jedem Markt (0 Ausnahmen in 4 000). Umgekehrt reichen sie nicht: 2 Märkte erfüllen sie und haben den Shapley-Wert doch außerhalb (z. B. 10, 5 \| 13, 10) | 4 000 Märkte | `test_e2_every_cell_matches_the_pinned_counts`, `test_core_always_satisfies_the_pair_equations_of_the_matching` |
| Kleine Zahlenräume | Bei ganzzahligen 2 × 2 Märkten mit Werten 0 bis 3 liegt der Shapley-Wert in 120 von 255 Märkten (v(N) > 0) außerhalb des Kerns | volle Aufzählung | `test_smallest_integer_squares_with_shapley_outside_the_core` |
| Dichte (k = 4, stetig) | Anteil im Kern 61,0 % bei Dichte 10 % (94 von 154 Märkten, 46 ohne Wert ausgeschlossen), 37,6 % bei 20 % (73 von 194), 15,0 % bei 30 % (30 von 200), 2,5 % bei 50 %, 0 bei 80 %, 0,5 % bei 100 % (1 von 200); der Anstieg 80 → 100 % (0 auf 1 von 200) ist bei dieser Stichprobe nicht von Rauschen zu unterscheiden | je 4 Seeds mit 50 Instanzen | `test_e3_density_table_matches_the_pinned_counts` |
| Ungleich große Märkte (stetig) | Shapley im Kern in 0 von 1 000 Märkten (1 × 2, 1 × 4, 2 × 3, 3 × 5, 5 × 3, je 200); jeder hat einen Spieler mit Kern-Maximum 0, und in jedem zahlt der Shapley-Wert einem solchen Spieler mehr als 0; bei Nullwerten kann der Shapley-Wert im Kern liegen (1 Ladung, 2 Lkw, Paarwerte 5 und 0: Shapley (2,5; 2,5; 0), im Kern) | 4 Seeds × 50 je Form | `test_e3_unequal_markets_never_have_shapley_in_the_core` |
| Kernbreite (Abstand der Ecken je Spieler / (v(N)/k)) | stetig 0,81 (k = 3), 0,92 (k = 80); ganzzahlig 0 bis 19 0,77 → 0,93; **Geografie 0,73 (k = 3), 0,26 (k = 40), 0,16 (k = 80)** (Tendenz fallend, nicht monoton); Ecken gegen LP (k ≤ 20) bis auf 1e-9 gleich | 4 Seeds × 3 Märkte = 12 je Zelle (Größenordnung, keine Verteilungsaussage) | `test_e4_width_table_matches_the_pinned_values_and_the_lp_agrees` |
| Verband | Join und Meet zweier Kern-Ecken liegen im Kern in 3 840 von 3 840 Paaren; die Gegenprobe (beide Maxima) nur in 693 (im Test ein Band, die Eckenwahl des LP kann je Plattform abweichen) | 384 Märkte ganzzahlig, 10 Paare je Markt | `test_e5_lattice_auction_and_hungarian_counts` |
| Gegenläufige Ordnung | Bei 48 Märkten sind 521 geordnete Paare von Kernpunkten bei den Ladungen vergleichbar; bei allen haben die Lkw die umgekehrte Ordnung | fünf Kern-Ecken je Markt (Band, Zahl plattformabhängig) | `test_opposite_order_of_loads_and_trucks_on_comparable_core_points` |
| Preise = Kernpunkte | In 180 Kernpunkten (4 × 4 Märkte, drei Verteilungen, je Markt vier LP-Ecken und die beiden geschlossenen Ecken) gilt u_i = max(0, max_j (w_ij − y_j)) bis 1e-7; bei 180 zufälligen Preisvektoren ist die Summe u + y nur in 4 gleich v(N), sonst größer | 30 Märkte; Preise aus SplitMix64 | `test_core_points_are_price_systems_and_random_prices_are_not` |
| Maximum je Spieler im Kern | gleich dem Beitrag zur großen Koalition, beide Ecken erreichen alle Maxima gleichzeitig: 255 Märkte, Abweichung 9,9e-14 | Scratch-Lauf; im Test kleinere Zahl | `test_corners_equal_the_lp_over_pairs_and_the_lp_over_all_coalitions` |
| Aufsteigende Auktion | endet in 384 von 384 Märkten auf den niedrigsten Kernpreisen (Käuferecke); Runden im Mittel 0,09 (4 × 6, 0 bis 3) bis 139,0 (6 × 4, 0 bis 99), höchstens 192 | 3 × 3, 5 × 5, 4 × 6, 6 × 4; ganzzahlig 0 bis 19, 0 bis 99, 0 bis 3 | `test_e5_lattice_auction_and_hungarian_counts`, `test_e5_round_counts_range_from_few_to_over_a_hundred` |
| Ungarische Methode | Dualpunkt im Kern in 384 von 384; mit Ladungen als Zeilen **Käuferecke** (384 von 384), mit Lkw als Zeilen **Verkäuferecke** (384 von 384) | dieselben Märkte; **eine Fassung** (Potenziale, Zeilen kommen einzeln an) | `test_e5_lattice_auction_and_hungarian_counts` |
| 0/1-Zuordnung = Flussspiel | 10 560 Koalitionswerte gleich dem Flusswert (Quelle → Ladung → Lkw → Senke, Kapazität 1) | Scratch-Lauf (`tools/orakel_lauf.py`); im Test 1 280 Werte | `test_zero_one_assignment_game_equals_a_flow_game` |

Messung dieser README, nicht im Test: Rechenzeit der Analyse bei 7 Ladungen und 7 Lkw (Teilmengen-Tabelle, exakter Shapley-Wert, Ecken) rund 0,01 s; die Messreihen 1 bis 5 brauchen 5,9 s, 1,8 s, 1,3 s, 2,5 s und 4,0 s (ein Lauf, Windows).

## Hypothesen des Einzelplans mit Ergebnis

| Nr | Hypothese | Ergebnis |
|---|---|---|
| H1 | Kern nie leer, ε* = 0, ohne Inneres, Dimension k (stetig) | **ja** (344 Märkte; Dimension 32 von 32 je Zelle) |
| H2 | Kern ist ein Verband; Ladungen und Lkw laufen gegenläufig | **ja** (3 840 von 3 840; 521 von 521) |
| H3 | Maximum je Spieler = Beitrag zur großen Koalition; Käuferecke = VCG-Ergebnis | Maximum **ja**; die Gleichheit mit VCG wurde **nicht** nachgerechnet |
| H4 | Anteil Shapley im Kern fällt mit k | **ja**, aber „fast nie“ erst ab k = 3 und dort nur für stetige, Pareto-, ganzzahlige (0 bis 19, 0 bis 99) und Geografie-Werte (höchstens 8 %), nicht für 0/1 (40,0 %) und 0 bis 3 (19,0 %); k = 2: rund ein Drittel |
| H5 | Mechanismus: Paargleichungen, entkoppelt, Dimension k | **gemessen, nicht bewiesen**; Paargleichungen sind notwendig, nicht hinreichend |
| H6 | Bei a ≠ b nie im Kern; Shapley zahlt Überzähligen | **ja für durchweg positive stetige Paarwerte** (0 von 1 000); bei Nullwerten kann der Shapley-Wert im Kern liegen (1 Ladung, 2 Lkw, Paarwerte 5 und 0) |
| H7 | Auktion endet an der niedrigsten Ecke; Ungarisch wählt die Ecke der Zeilenseite | **ja** (384 von 384; für diese eine Fassung) |
| H8 | Zufallsmärkte: Kern bleibt breit, Geografie schrumpft | **ja** als Tendenz (Geografie nicht monoton), Größenordnung (12 Märkte je Zelle) |
| H9 | Dichte q sinkt ⇒ Shapley öfter im Kern | **ja** (61 % bei 10 %) |
| H10 | 0/1-Zuordnung ist ein Flussspiel | **ja** (10 560 Werte); allgemeine Paarwerte nicht untersucht |

## Befunde und Korrekturen gegenüber dem Plan

- **Aufhänger nicht neu.** `austauschnetz-demo` zeigt „Shapley meist nicht im Kern“ schon für Einheitswerte auf Graphen (nach seiner README in 10 von 96 Netzen im Kern; Stand der Abstimmung vom 2026-11-18). Diese Demo setzt fort und hängt den Befund nicht als neu auf.
- **„Fast nie“ gilt erst ab k = 3 und dort nicht für 0/1 (40,0 %) und ganzzahlig 0 bis 3 (19,0 %)**: bei k = 2 liegt der Shapley-Wert bei stetigen, ganzzahligen (0 bis 19, 0 bis 99) und Pareto-Werten in 32 bis 37 % der Märkte im Kern, bei Geografie in 17,5 %, bei 0/1 in 71,9 %, bei Rang eins in keinem. Bei 0/1-Werten und dünn besetzten Märkten ist der Anteil auch ab k = 4 größer (15 % bei k = 4; 61 % bei Dichte 10 %).
- **Paargleichungen**: Der Plan nannte „im Kern genau dann, wenn alle Paargleichungen gelten (bis auf Gleichstände)“. Gemessen: Im Kern gelten sie immer (0 Ausnahmen), umgekehrt gibt es 2 Märkte in 4 000, in denen sie gelten, obwohl der Shapley-Wert außerhalb liegt, ohne Gleichstand der besten Zuordnung (10, 5 \| 13, 10 und 12, 0 \| 18, 12). Die Aussage ist also einseitig.
- **„Im Kern ⇒ entkoppelt“** in diesen Stichproben bestätigt für k ≥ 3 (0 Ausnahmen); bei k = 2 ganzzahlig 2 Ausnahmen in 400 (Plan: 7 in 3 200). Es ist kein Satz: bei ganzzahligen Werten gibt es auch bei k = 3 Ausnahmen (Beispiel in der Tabelle oben). Alle bekannten Ausnahmen haben Gleichstände in der besten Zuordnung (Tabelle oben); die Definition von „entkoppelt“ ist deshalb nicht erweitert worden, die Zählungen der Messreihe 2 bleiben unverändert.
- **Zahlen der Vormessung wichen ab** (anderer Zufallsgenerator, andere Instanzzahlen): 0/1 bei k = 5 hier 0 von 100 statt 7 %, bei k = 4 15,0 % statt 11 %; Geografie-Kernbreite bei k = 3 hier 0,73 statt 0,56 (12 Märkte je Zelle); Auktionsrunden 3,0 (3 × 3, 0 bis 19) und 139,0 (6 × 4, 0 bis 99) statt 4,0 und 129,3. Nur die hier gemessenen Zahlen stehen in Text und App.
- **Ecken ohne LP**: Der Plan rechnete die Ecken per LP; hier stehen sie in geschlossener Form (Beiträge zur großen Koalition), das LP ist die Gegenprobe. Dadurch sind die Ecken auch bei k = 80 exakt und schnell.
- **Regler**: Der Plan nannte einen Auktions-Schritt-Regler. Die Auktion zeigt stattdessen ihre ganze Spur als Tabelle (höchstens 40 Runden) und als Preiskurve; ein Regler mit berechneter Obergrenze wäre ein Fallstrick (siehe DEMO-PLAYBOOK §3).
- **Bei a ≠ b** füllt die Ungarische Methode den Markt mit Nullwerten quadratisch auf (neu gegenüber dem Plan; gemessen in 4 × 6 und 6 × 4 wie bei quadratischen Märkten).
- **Kein PDF-Export** (der Plan nannte ihn): siehe „Bewusst nicht umgesetzt“.
- **Flussspiel kein Tab** (Plan: nur Verweis auf `flussspiel-demo`), **kein Nukleolus** (Verweis `kern-demo`).
- Die Behauptung eines Suchergebnisses, der Shapley-Wert sei das Mittel beider Ecken, ist durch die Handrechnung zwei Ladungen und ein Lkw widerlegt (Mittel (1, 0, 9) gegen Shapley (7/3, 4/3, 19/3)).

## Ehrliche Grenzen

- Alle Anteile gelten für die acht Zufallsverteilungen mit unabhängigen Paarwerten und die genannten Größen; für reale Märkte sagt die Demo nichts.
- Der Mechanismus ist **gemessen, nicht bewiesen**: „entkoppelt“ ist in den Stichproben notwendig für Shapley im Kern (bei Gleichständen gibt es Ausnahmen, siehe Tabelle), aber ab k = 3 nicht hinreichend. Die Dimension des Kerns (k statt n − 1) ist die Deutung der Zahlen, kein Beweis.
- Die Wahl der Ecke durch die Ungarische Methode gilt für die hier implementierte Fassung; andere Implementierungen (zum Beispiel solche ohne Dualausgabe) sagen nichts über den Kern.
- Die Kernbreite bei großen Märkten beruht auf 12 Märkten je Zelle: Größenordnung, keine Verteilungsaussage. Die Aussage „Geografie schrumpft“ ist eine Tendenz bei diesem Generator (die Reihe ist nicht monoton).
- Die Literatur ist nur über Bibliographien und Abstracts bestätigt; ob die Käuferecke dem VCG-Ergebnis der Quellen entspricht, ist hier nicht nachgerechnet.
- Gleichstände: Die Ecken und das Kernurteil hängen nicht an der gewählten besten Zuordnung, die „Paargleichungen“ schon.

## Annahmen, Grenzen und bekannte Probleme

**Annahmen**

- Zwei Seiten: a Ladungen (Käuferseite, bekommen den Überschuss u) und b Lkw (Verkäuferseite, bekommen den Preis y). Ein Paar aus Ladung i und Lkw j schafft den Paarwert w_ij ≥ 0 (gedacht als Erlös minus Fahrkosten, in €). Jede Ladung fährt mit höchstens einem Lkw, jeder Lkw nimmt höchstens eine Ladung; unverpaart bleiben bringt 0.
- Der Wert einer Koalition ist das beste Matching in ihr, Nutzen ist übertragbar (Geld lässt sich frei zwischen den Spielern verschieben). Der Paarwert hängt nur am Paar, nicht an anderen Paaren.
- Der Kern ist die Menge der Aufteilungen des Gesamtwerts v(N), bei denen keine Koalition sich allein besserstellen kann. Hier genügt der Test über die a · b Paare (die Tests prüfen das gegen alle 2^n Koalitionen).
- Gerechnet wird ganzzahlig in Hundertsteln: Shapley-Wert und Kerntests sind exakt (Brüche), keine Toleranzen.
- Eigennutz, Paarwerte und Koalitionswerte sind Modellannahmen, keine Beobachtungen. Die App macht keine Anlage-, Rechts- oder Abstimmungsberatung.

**Grenzen**

- Der Shapley-Wert wird über alle Koalitionen gerechnet und gibt es daher nur bis 14 Spieler (2^14 Koalitionen); die Regler enden bei 7 Ladungen und 7 Lkw. Nur Experiment 4 rechnet den Kern größer, ohne Shapley-Wert.
- Alle Anteile und Zählungen gelten für die acht Zufallsverteilungen der App, die genannten Märkte (Größen, Instanzzahlen, Zufalls-Seeds) und die Voraussetzung unabhängiger Paarwerte; sie sind keine Aussage über reale Märkte.
- Der Mechanismus hinter „Shapley meist nicht im Kern“ ist gemessen (Zählung, Dimension des Kerns), nicht bewiesen.
- Die aufsteigende Auktion rechnet auf ganzen Euro (die Paarwerte werden gerundet) und in Schritten von 1; die Ungarische Methode ist eine einzige Fassung (Potenziale mit Zeilen als Neuankömmlingen). Aussagen über die Wahl der Ecke gelten für diese Fassung, nicht für jede Implementierung.
- Nicht enthalten: Nukleolus (siehe `kern-demo`), Flussspiele (siehe `flussspiel-demo`), Austauschnetze auf allgemeinen Graphen (siehe `austauschnetz-demo`), Märkte mit mehreren Partnern je Spieler.
- „Ladung“ und „Lkw“ sind eine Kulisse für das Modell; die Demo macht keine Aussage über reale Frachtmärkte.

**Bekannte Probleme und Unsicherheiten**

- Bei ganzzahligen Werten gibt es oft mehrere optimale Zuordnungen. Der Kerntest hängt nicht daran; die „Paargleichungen“ und die Eckenberechnung benutzen die eine Zuordnung, die scipy zuerst findet, und liefern in jedem Fall dieselben Ecken (Tests prüfen das gegen ein Linearprogramm).
- Paarwerte sind auf Hundertstel gerundet; Pareto-Werte sind bei 300 € gekappt (das betrifft nur rund 0,6 % der Paare), und die Potenz dort kann auf anderen Plattformen in der letzten Stelle abweichen. Die Tests prüfen für diese Verteilung Bänder.
- Die Dimension des Kerns wird per Linearprogramm bestimmt (Toleranz 1e-7); die Ecken des Kerns dagegen exakt ganzzahlig.
- Die Zählungen der Experimente sind Mittel über vier Zufalls-Seeds mit den genannten Instanzzahlen; kleine Zellen schwanken (die Std über die Seeds steht dabei).
- Die Literaturangaben sind über Bibliographien bestätigt, Volltexte nicht gelesen; Aussagen der App stützen sich zuerst auf die Messung, nicht auf die Quelle.

## Literatur

Autoren, Jahr, Zeitschrift und Seiten über Bibliographien bestätigt; Volltexte nicht gelesen, nur Abstracts und Zusammenfassungen.

- Shapley, L. S., Shubik, M. (1971 oder 1972, Bibliographien nennen beides): The Assignment Game I: The Core. International Journal of Game Theory 1, 111–130.
- Demange, G., Gale, D., Sotomayor, M. (1986): Multi-Item Auctions. Journal of Political Economy 94(4), 863–872 (Abstract gelesen: zwei dynamische Auktionen, die erste endet auf dem für die Bieter besten Preisgleichgewicht).
- Leonard, H. B. (1983): Elicitation of Honest Preferences for the Assignment of Individuals to Positions. Journal of Political Economy 91(3), 461–479 (Abstract gelesen; ob die Käuferecke dieser Preise gleich dem Beitrag zur großen Koalition ist, wurde hier nur gemessen, nicht aus der Quelle belegt).
- Núñez, M., Rafels, C. (2003): The Assignment Game: The τ-Value. International Journal of Game Theory 31, 411–422 (nach Suchauszug: der τ-Wert ist der Mittelpunkt der beiden Ecken des Kerns; Volltext nicht gelesen; Jahr 2003 nach dem Springer-Eintrag).

## Tests

201 Tests (`pytest --collect-only -q tests`), Laufzeit 197 s (3:16 min) bei hoher Last durch parallele Sitzungen (`-n 8`, Windows). Zusammensetzung:

- **Einheiten gegen Handrechnung** (`test_game.py`, Mini-Instanzen in `conftest.py`): Koalitionswerte, Shapley-Brüche, Ecken, blockierende Paare, Auktion, Ungarische Methode, Dimension.
- **Orakel** (`test_oracle_zuordnungsspiel.py`): v(S) gegen scipy und Aufzählung; Shapley gegen alle Reihenfolgen in Brüchen; Kern über Paare gegen alle Koalitionen; Ecken gegen LP (Paare und alle Koalitionen); Least-Core-Wert; Auktion gegen LP-Ecke; Ungarische Methode; 0/1 gegen Fluss (networkx). Der große Lauf steht in `tools/orakel_lauf.py` (Ausgabe unten).
- **Randfälle** (`test_edge_cases.py`): 1 × 1, v(N) = 0, Gleichstände, a ≠ b in beide Richtungen, Reglergrenzen, Handbeispiele ohne Zufalls-Seed und Dichte.
- **Zahlen der README** (`test_claims.py` mit `claims_expected.py`), **Verdikte** (`test_verdicts.py`: jede Schwelle, jedes Preset, die Ränder), **Szenario** (`test_scenario.py`: Wertebereiche, jede Verteilung weicht von jeder anderen ab), **Messreihen** (`test_experiments.py`), **Worst-Case** a = b = 7 (`test_algorithm.py`), **Kopie des Zufallsgenerators** (`test_copies.py`), **Texte** (`test_texts.py`: Annahmen-Text in App und README gleich, Formelblöcke).
- **Oberfläche** (`test_app.py`, `test_layout.py`, `test_presets.py`, `test_footer.py`): jeder Knopf, jeder Regler an Minimum und Maximum, jede Verteilung, jedes Preset mit erwartetem Verdikt, jeder Experiment-Knopf einzeln, Permalink, Hervorhebung des aktiven Presets.

Ausgabe des großen Orakel-Laufs (`python tools/orakel_lauf.py`, ein Lauf, rund 10 s):

```
O1 v(S): 256 Instanzen, 9344 Koalitionswerte, DP = scipy = Aufzählung
O2 Shapley-Wert: 240 Instanzen (n = 4 bis 6), Teilmengenformel = alle n! Reihenfolgen in exakten Brüchen
O3 Kern über Paare = Kern über alle Koalitionen: 382 Instanzen, 3820 Punkte (1232 im Kern, 2588 außerhalb)
O4 Least-Core-Wert über alle Koalitionen: 344 Instanzen, größter Betrag 7.6e-14 (null bis auf die LP-Genauigkeit)
O5 Ecken: 255 Instanzen, Maximum je Spieler im Kern = Beitrag zur großen Koalition = LP (über Paare, bei n <= 6 auch über alle Koalitionen); größte Abweichung 9.9e-14
O6 Auktion = LP-Ecke: 238 Märkte; Ungarische Methode (beide Rollen) im Kern und an der Ecke der Zeilenseite: 280 Märkte
O7 0/1-Zuordnungsspiel = Flussspiel (networkx): 10560 Koalitionswerte
GESAMT: 27913 Prüfungen, 0 Fehler, 8 s
```

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Ansichten, Kernfrage, fünf Experimente, Grenzen, Formeln |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `zsp_game.py` | Koalitionswerte, exakter Shapley-Wert, Kerntests (Paare und alle Koalitionen), blockierende Paare |
| `zsp_core.py` | Ecken (geschlossen), Kern-LP, Verband, Dimension, Entkopplung, Abstand zum Kern |
| `zsp_auction.py` | aufsteigende Auktion, Ungarische Methode |
| `zsp_scenario.py` | Märkte aus Verteilungen und Handbeispielen, Zellschlüssel |
| `zsp_experiments.py` | Messreihen der fünf Experimente und die Aufzählung kleiner Märkte |
| `zsp_evaluation.py` | Einstellungen, Analyse, Verdikte als reine Funktionen |
| `zsp_texts.py` | feste Texte (Annahmen, Begriffe, Literatur) |
| `zsp_constants.py` | Regler-Grenzen, Verteilungen, Raster der Experimente, Presets |
| `zsp_presets.py` | Permalink, Presets, Aktivmarkierung, Zufalls-Seed-Button |
| `zsp_layout.py` | fester Kopf (Stylesheet) |
| `zsp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `zsp_rng.py` | SplitMix64, wörtliche Kopie aus `ab-test-demo` |
| `tools/orakel_lauf.py` | großer Orakel-Lauf (nicht Teil der Suite) |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

- **PDF-Export** der Ergebnisse: der Plan nannte ihn, die Demo kommt ohne aus; die Verdikte und Tabellen stehen in der App.
- **Nukleolus** und andere Lösungskonzepte im Kern: Verweis auf `kern-demo`.
- **Flussspiel als Tab**: nur der 0/1-Fall ist ein Flussspiel (Messung oben); allgemeine Paarwerte sind ein Min-Cost-Fluss, das ist Gegenstand von `flussspiel-demo`.
- **Beweis des Mechanismus**: die Demo zeigt Zählungen und die Dimension des Kerns, keinen Beweis.
- **Märkte mit mehreren Partnern** und Auktionen mit Bündelgeboten.
- Weitere Verteilungen oder echte Daten.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Cloud-Version für Python ist in den Streamlit-Einstellungen gewählt; die CI testet mit Python 3.12, lokal läuft 3.14. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html).
