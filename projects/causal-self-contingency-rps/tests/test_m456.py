import numpy as np

from causal_rps.models.m4 import M4
from causal_rps.models.m5 import M5
from causal_rps.models.m6 import M6
from causal_rps.schema import ModelObservation, RunInData, RunInSummary


def _runin():
    obs = [
        ModelObservation(t=i, a=i % 3, b=(2 * i + 1) % 3, r=0)
        for i in range(12)
    ]
    return RunInData(obs, RunInSummary(0.5, 0.1, 0.1))


def _check(model):
    s = model.initialize({}, _runin())
    lp = np.array([model.predict_log_prob(s, a) for a in range(3)])
    p = np.exp(lp)
    assert np.all(np.isfinite(lp))
    assert np.isclose(p.sum(), 1.0, atol=1e-8)
    s = model.update(s, ModelObservation(t=13, a=0, b=1, r=-1))
    assert np.isfinite(model.predict_log_prob(s, 1))


def test_m4_probabilities():
    _check(M4(n_particles=24, seed=1))


def test_m5_probabilities():
    _check(M5(n_particles=24, seed=2))


def test_m6_probabilities():
    _check(M6(n_particles=24, seed=3))


def test_m4_theta_normalizes():
    m = M4(n_particles=16, seed=4)
    s = m.initialize({}, _runin())
    assert np.allclose(s.q_theta.sum(axis=1), 1.0)
