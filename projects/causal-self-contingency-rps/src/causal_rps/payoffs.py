import numpy as np

# Participant payoff U[a, b], with action coding 0=R,1=P,2=S.
U = np.array(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    dtype=float,
)


def payoff(a: int, b: int) -> int:
    return int(U[a, b])


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def exploitability(p: np.ndarray) -> float:
    """Objective zero-sum RPS exploitability for participant mixed strategy p."""
    p = np.asarray(p, dtype=float)
    return float(max(p[2] - p[1], p[0] - p[2], p[1] - p[0]))


def rotate_plus_one(a: int) -> int:
    return (a + 1) % 3
