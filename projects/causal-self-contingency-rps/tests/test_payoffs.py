import numpy as np

from causal_rps.payoffs import U, exploitability


def test_antisymmetric():
    assert np.allclose(U.T, -U)


def test_uniform_not_exploitable():
    assert abs(exploitability(np.ones(3) / 3)) < 1e-12


def test_counterexample_values():
    assert np.isclose(exploitability(np.array([0.6, 0.3, 0.1])), 0.5)
    assert np.isclose(exploitability(np.array([0.6, 0.1, 0.3])), 0.3)
