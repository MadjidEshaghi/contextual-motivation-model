from __future__ import annotations

import numpy as np
from scipy.special import digamma

from causal_rps.payoffs import softmax


def normalize(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = p.sum(axis=axis, keepdims=True)
    return p / np.maximum(s, 1e-300)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def ce_kernel(p_hat: np.ndarray, beta_c: float) -> np.ndarray:
    score = np.array([p_hat[(b - 1) % 3] for b in range(3)], dtype=float)
    return softmax(float(beta_c) * score)


def cb_kernel(p_hat: np.ndarray, epsilon_b: float) -> np.ndarray:
    return (1.0 - epsilon_b) * np.asarray(p_hat, dtype=float) + epsilon_b / 3.0


def theta_information_gain(q_theta: np.ndarray, kernels: np.ndarray) -> float:
    q = normalize(q_theta)
    k = normalize(kernels, axis=1)
    pred = q @ k
    return entropy(pred) - float(sum(q[j] * entropy(k[j]) for j in range(len(q))))


def dirichlet_information_gain(alpha: np.ndarray) -> float:
    a = np.asarray(alpha, dtype=float)
    a0 = float(a.sum())
    p = a / a0
    h_pred = entropy(p)
    expected_h = float(digamma(a0 + 1.0) - np.sum(p * digamma(a + 1.0)))
    return max(0.0, h_pred - expected_h)
