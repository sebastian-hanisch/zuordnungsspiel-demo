"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Alle Paarwerte in Hundertsteln (ganze Zahlen); die Handrechnungen stehen in den Kommentaren in Euro."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

S100 = 100


def arr(rows):
    """Paarwerte in Euro als Matrix in Hundertsteln."""
    return np.array(rows, dtype=np.int64) * S100


@pytest.fixture
def one_load_two_trucks():
    """1 Ladung, 2 Lkw, beide Paarwerte 1. Handrechnung: v(N) = 1; Kern = {(1, 0, 0)}; Shapley (2/3, 1/6, 1/6)."""
    return arr([[1, 1]])


@pytest.fixture
def two_loads_one_truck():
    """2 Ladungen (10 und 8), 1 Lkw. Handrechnung: v(N) = 10; Shapley (7/3, 4/3, 19/3); Ecken (2, 0, 8) und (0, 0, 10)."""
    return arr([[10], [8]])


@pytest.fixture
def square_2x2():
    """[[10, 6], [6, 10]]. Handrechnung: v(N) = 20; Shapley 5 je Spieler im Kern; Ecken (10, 10, 0, 0) und (0, 0, 10, 10)."""
    return arr([[10, 6], [6, 10]])


@pytest.fixture
def diagonal_2x2():
    """Diagonale 10 / 0 / 0 / 10: zwei getrennte Paare, Shapley 5 je Spieler."""
    return arr([[10, 0], [0, 10]])


@pytest.fixture
def decoupled_3x3():
    """Diagonale 10, 8, 6, sonst 1. Handrechnung: Shapley (5, 4, 3 | 5, 4, 3), im Kern."""
    return arr([[10, 1, 1], [1, 8, 1], [1, 1, 6]])


@pytest.fixture
def rank_one_3x3():
    """s = t = (3, 2, 1): w_ij = s_i t_j. Shapley (4,05; 6,40/3; 2,45/3 | gleich), eine blockierende Paarung (Ladung 1, Lkw 1) mit Lücke 0,90."""
    return arr(np.outer([3, 2, 1], [3, 2, 1]))


def brute_value(wi, mask):
    """Wert einer Koalition per Aufzählung aller Teilzuordnungen (Rechenweg ohne scipy, ohne DP)."""
    a, b = wi.shape
    loads = [i for i in range(a) if mask >> i & 1]
    trucks = [j for j in range(b) if mask >> (a + j) & 1]
    best = 0
    for choice in itertools.product([None] + trucks, repeat=len(loads)):
        used = [c for c in choice if c is not None]
        if len(used) != len(set(used)):
            continue
        best = max(best, sum(int(wi[i, c]) for i, c in zip(loads, choice) if c is not None))
    return best
