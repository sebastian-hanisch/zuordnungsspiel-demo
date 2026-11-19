"""Texte: App und README haben denselben Wortlaut der Annahmen (DEMO-PLAYBOOK §5), keine Platzhalter, Pflichtabschnitte, Formelblöcke."""

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

import zsp_constants as C
from zsp_texts import ASSUMPTIONS_MARKDOWN, DIST_HELP, GLOSSARY_MARKDOWN

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_readme_contains_the_assumptions_text_of_the_app_verbatim():
    assert ASSUMPTIONS_MARKDOWN.rstrip("\n") in (ROOT / "README.md").read_text(encoding="utf-8")


def test_app_shows_the_assumptions_text_in_its_expander():
    at = _app()
    expander = next(e for e in at.expander if e.label == "⚠️ Annahmen, Grenzen und bekannte Probleme")
    assert ASSUMPTIONS_MARKDOWN.strip() in " ".join(m.value for m in expander.markdown)


def test_app_has_the_required_sections_by_title():
    at = _app()
    labels = {e.label for e in at.expander}
    assert {"💬 In einfachen Worten", "📐 Mathematische Formulierung", "⚠️ Annahmen, Grenzen und bekannte Probleme", "So funktioniert das Zuordnungsspiel"} <= labels
    subheads = [s.value for s in at.subheader]
    assert sum(1 for s in subheads if s.startswith("🔬")) == 5 and any(s.startswith("🚧") for s in subheads)
    assert any("Kernfrage" in m.value for m in at.markdown)


def test_every_distribution_has_a_help_text_and_a_name():
    assert set(DIST_HELP) == set(C.DISTS)
    assert all(DIST_HELP[d] for d in C.DISTS)


def test_rank_one_label_and_help_text_use_the_same_index_convention():
    """Label und Hilfetext von „Rang eins“ nennen dieselbe Formel (i, j ab 1: s_i = a − i + 1), wie die Matrix in zsp_scenario sie baut."""
    assert "a − i + 1" in C.DISTS["h_rang"] and "a − i + 1" in DIST_HELP["h_rang"] and "b − j + 1" in C.DISTS["h_rang"] and "b − j + 1" in DIST_HELP["h_rang"]


def test_unit_cell_is_explained_in_the_app_glossary_and_readme():
    at = _app()
    assert any("**Zelle**" in c.value for c in at.caption)                       # erste Verwendung (Experiment 1) erklärt den Begriff
    assert "**Zelle**" in GLOSSARY_MARKDOWN and "Kombination aus Verteilung und Größe" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_glossary_names_the_core_terms():
    for term in ("Zelle", "Paarwert", "Kern", "Blockierendes Paar", "Shapley-Wert", "Preis", "Käuferecke"):
        assert term in GLOSSARY_MARKDOWN


def test_no_placeholders_or_forbidden_words_remain():
    for path in list(ROOT.glob("*.py")) + [ROOT / "README.md"] + list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "README.md":
            text = re.sub(r"<!-- TODO nach dem Deploy.*?-->", "", text, flags=re.S)      # der Live-Link-Platzhalter des Gerüsts bleibt bis zum Deploy
        if path.name == "test_texts.py":
            continue
        assert "TODO" not in text, path.name
        assert "Saatwert" not in text and "Schwanz" not in text, path.name
        assert "Arbeitsname" not in text and "Stück der Linie" not in text and "eigenes Stück" not in text, path.name
        assert not re.search(r"\bträgt\b|\btragen\b", text.replace("Kosten tragen", "")), path.name


def test_math_expander_has_one_display_equation_per_block():
    """Je Display-Gleichung ein eigener $$-Block (sonst KaTeX-Fehler): kein Block enthält ein zweites $$ oder eine Leerzeile; kein ungeschütztes %."""
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    blocks = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
    assert len(blocks) >= 11
    for b in blocks:
        assert "\n\n" not in b and "$" not in b and not re.search(r"(?<!\\)%", b)


def test_math_expander_starts_with_the_reading_aid_and_ends_with_the_literature():
    at = _app()
    expander = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = " ".join(m.value for m in expander.markdown)
    assert text.index("Lesehilfe") < text.index("| Symbol | Bedeutung |") < text.index("Literatur")
    assert "Shapley, L. S., Shubik, M." in text
