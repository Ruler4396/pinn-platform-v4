"""Finite-difference stencils shared by the local self-test and the on-instance gate.

The same formulas must produce the truth's residual on the instance and the
reference values checked locally, otherwise "score(truth) vs score(model)" would
compare two different estimators.

Index convention everywhere: a 2-D field is ``field[i][j]`` where i walks the
first axis (spacing ``h0``) and j the second axis (spacing ``h1``).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

NAN = float("nan")


def central_first(values: Sequence[float], step: float) -> List[float]:
    n = len(values)
    out = [NAN] * n
    for i in range(1, n - 1):
        out[i] = (values[i + 1] - values[i - 1]) / (2.0 * step)
    return out


def central_second(values: Sequence[float], step: float) -> List[float]:
    n = len(values)
    out = [NAN] * n
    for i in range(1, n - 1):
        out[i] = (values[i + 1] - 2.0 * values[i] + values[i - 1]) / (step * step)
    return out


def grid_2d(rows: int, cols: int, fn: Callable[[int, int], float]) -> List[List[float]]:
    return [[fn(i, j) for j in range(cols)] for i in range(rows)]


def laplacian_2d(field: Sequence[Sequence[float]], h0: float, h1: float) -> List[List[float]]:
    rows, cols = len(field), len(field[0])
    out = [[NAN] * cols for _ in range(rows)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            out[i][j] = ((field[i + 1][j] - 2.0 * field[i][j] + field[i - 1][j]) / (h0 * h0)
                         + (field[i][j + 1] - 2.0 * field[i][j] + field[i][j - 1]) / (h1 * h1))
    return out


def gradient_2d(field: Sequence[Sequence[float]], h0: float, h1: float
                ) -> List[List[Tuple[float, float]]]:
    """Returns (d/d(axis0), d/d(axis1))."""
    rows, cols = len(field), len(field[0])
    out: List[List[Tuple[float, float]]] = [[(NAN, NAN)] * cols for _ in range(rows)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            out[i][j] = ((field[i + 1][j] - field[i - 1][j]) / (2.0 * h0),
                         (field[i][j + 1] - field[i][j - 1]) / (2.0 * h1))
    return out


def rms(values: Sequence[float]) -> float:
    xs = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not xs:
        return NAN
    return (sum(v * v for v in xs) / len(xs)) ** 0.5


def flatten(field: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for row in field:
        out.extend(row)
    return out
