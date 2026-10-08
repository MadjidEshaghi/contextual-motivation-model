from __future__ import annotations

import numpy as np


def normalize_weights(weights: np.ndarray) -> None:
    s = float(weights.sum())
    if not np.isfinite(s) or s <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights[:] = weights / s


def ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def systematic_resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right")


def maybe_resample(weights: np.ndarray, rng: np.random.Generator, threshold: float = 0.5):
    if ess(weights) >= threshold * len(weights):
        return None
    idx = systematic_resample_indices(weights, rng)
    weights[:] = 1.0 / len(weights)
    return idx
