from __future__ import annotations

from dataclasses import dataclass


@dataclass
class History:
    prev_a: int | None = None
    prev2_a: int | None = None
    prev_b: int | None = None
    prev_r: int | None = None


def self_context(history: History, kind: str = "x3") -> tuple[int, ...]:
    a1 = -1 if history.prev_a is None else int(history.prev_a)
    a2 = -1 if history.prev2_a is None else int(history.prev2_a)
    b1 = -1 if history.prev_b is None else int(history.prev_b)
    r1 = 9 if history.prev_r is None else int(history.prev_r)

    if kind == "x1":
        return (a1, r1)
    if kind == "x2":
        return (a1, b1)
    if kind == "x3":
        return (a1, b1, r1)
    if kind == "x4":
        return (a1, a2)
    raise ValueError(f"unknown context kind: {kind}")


def opponent_context(history: History) -> tuple[int]:
    return (-1 if history.prev_b is None else int(history.prev_b),)


def advance(history: History, a: int, b: int, r: int) -> None:
    history.prev2_a = history.prev_a
    history.prev_a = int(a)
    history.prev_b = int(b)
    history.prev_r = int(r)
