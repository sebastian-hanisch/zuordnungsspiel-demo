"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht sie): Annahmen, Grenzen, bekannte Probleme, Begriffe, Literatur.

Zahlen stehen hier nur, wenn sie Einstellungen oder feste Konstanten sind (Regler, Raster, Kappung); Messzahlen kommen mit f-String aus der Rechnung
oder stehen in der README mit Testverweis."""

DIST_HELP = {
    "int20": "Jeder Paarwert ist eine ganze Zahl von 0 bis 19 €, gleichverteilt und unabhängig.",
    "int100": "Jeder Paarwert ist eine ganze Zahl von 0 bis 99 €, gleichverteilt und unabhängig.",
    "int3": "Jeder Paarwert ist eine ganze Zahl von 0 bis 3 €, gleichverteilt und unabhängig: viele Gleichstände.",
    "bin": "Jeder Paarwert ist 0 oder 1 €, je mit Wahrscheinlichkeit ein halb: eine Paarung lohnt sich oder nicht.",
    "unif": "Jeder Paarwert ist gleichverteilt zwischen 0 und 100 €, auf Hundertstel gerundet, unabhängig.",
    "pareto": "Pareto-Verteilung mit Index 1,5 ab 10 €, bei 300 € gekappt: meist kleine, selten sehr große Paarwerte.",
    "prod": "Paarwert = 50 · s_i · t_j: Jede Ladung hat eine Güte s_i, jeder Lkw eine Güte t_j (gleichverteilt zwischen 0,5 und 1,5), der Wert ist ihr Produkt.",
    "geo": "Ladung i bringt einen Erlös R_i zwischen 40 und 100 €. Ladungen und Lkw liegen zufällig im Einheitsquadrat; Paarwert = max(0, R_i − 60 · Abstand).",
    "h_eins": "Fester Markt, unabhängig von Zufalls-Seed und Dichte: jeder Paarwert ist 1 €.",
    "h_spalte": "Fester Markt: Ladung i bringt bei jedem Lkw den Wert 10 − 2·(i − 1) € (10, 8, 6, 4, 2, 0, 0), der Wert hängt nur an der Ladung.",
    "h_diag": "Fester Markt: Ladung i und Lkw i passen am besten zusammen (10, 8, 6, 4, 2, 2, 2 €), jedes andere Paar bringt 1 €.",
    "h_rang": "Fester Markt: Paarwert = (a − i + 1) · (b − j + 1) €, also s = (a, a − 1, …) und t = (b, b − 1, …).",
    "h_getrennt": "Fester Markt: Ladung i und Lkw i bringen zusammen 10 €, jedes andere Paar 0 €.",
}

ASSUMPTIONS_MARKDOWN = """\
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
"""

GLOSSARY_MARKDOWN = """\
| Begriff | Bedeutung hier |
|---|---|
| **Paarwert** w_ij | Was Ladung i und Lkw j zusammen mehr erwirtschaften als einzeln (gedacht als Erlös minus Fahrkosten). Nicht gepaart zu bleiben bringt 0. |
| **Wert einer Koalition** v(S) | Das beste Matching, das nur Mitglieder von S benutzt. v(N) ist der Wert aller Spieler zusammen und wird aufgeteilt. |
| **Kern** (englisch Core) | Alle Aufteilungen von v(N), bei denen keine Gruppe von Spielern allein mehr erreichen könnte, als sie bekommt. Hier ist er nie leer. |
| **Blockierendes Paar** | Ein Paar aus Ladung und Lkw, das zusammen mehr schaffen würde, als beide in der Aufteilung bekommen: Die Aufteilung ist dann nicht stabil. |
| **Shapley-Wert** | Der Mittelwert des Beitrags eines Spielers über alle Reihenfolgen, in denen die Koalition wächst („fair“ im Sinne der Gleichbehandlung). Er muss nicht stabil sein. |
| **Preis (Dualpreis)** | Der Betrag y_j eines Lkw; der Überschuss einer Ladung ist dann der Wert ihres besten Lkw minus dessen Preis. Ein Kernpunkt entspricht genau solchen Preisen. |
| **Zelle** | Eine Kombination aus Verteilung und Größe (in Messreihe 5: Marktform und Wertebereich). Je Zelle laufen die Instanzen aller Zufalls-Seeds; Zahlen und Anteile der Tabellen beziehen sich auf eine Zelle. |
| **Käuferecke / Verkäuferecke** | Die beiden äußersten Punkte des Kerns: Die Ladungen bekommen jeweils ihr Maximum (Käuferecke) oder die Lkw (Verkäuferecke). Das Maximum eines Spielers ist sein Beitrag zur großen Koalition. |
"""

LITERATUR_HEAD = "**Literatur** (Autoren, Jahr, Zeitschrift und Seiten über Bibliographien bestätigt; Volltexte nicht gelesen, nur Abstracts und Zusammenfassungen)"
LITERATUR = [
    "- Shapley, L. S., Shubik, M. (1971 oder 1972, Bibliographien nennen beides): The Assignment Game I: The Core. International Journal of Game Theory 1, 111–130.",
    "- Demange, G., Gale, D., Sotomayor, M. (1986): Multi-Item Auctions. Journal of Political Economy 94(4), 863–872 (Abstract gelesen: zwei dynamische Auktionen, die erste endet auf dem für die Bieter besten Preisgleichgewicht).",
    "- Leonard, H. B. (1983): Elicitation of Honest Preferences for the Assignment of Individuals to Positions. Journal of Political Economy 91(3), 461–479 (Abstract gelesen; ob die Käuferecke dieser Preise gleich dem Beitrag zur großen Koalition ist, wurde hier nur gemessen, nicht aus der Quelle belegt).",
    "- Núñez, M., Rafels, C. (2003): The Assignment Game: The τ-Value. International Journal of Game Theory 31, 411–422 (nach Suchauszug: der τ-Wert ist der Mittelpunkt der beiden Ecken des Kerns; Volltext nicht gelesen; Jahr 2003 nach dem Springer-Eintrag).",
]
