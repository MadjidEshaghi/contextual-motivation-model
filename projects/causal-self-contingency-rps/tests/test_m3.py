import numpy as np

from causal_rps.models.m3 import M3
from causal_rps.schema import ModelObservation, RunInData, RunInSummary


def _runin():
    obs = [
        ModelObservation(t=i, a=i % 3, b=(i + 1) % 3, r=-1)
        for i in range(12)
    ]
    return RunInData(obs, RunInSummary(0.5, 0.1, 0.1))


def test_m3_predictive_is_probability_distribution():
    m = M3(n_particles=64, seed=1)
    s = m.initialize({}, _runin())
    lp = np.array([m.predict_log_prob(s, a) for a in range(3)])
    p = np.exp(lp)
    assert np.isclose(p.sum(), 1.0, atol=1e-10)


def test_m3_updates_without_nonfinite_values():
    m = M3(n_particles=64, seed=2)
    s = m.initialize({}, _runin())
    obs = ModelObservation(t=13, a=0, b=1, r=-1)
    s = m.update(s, obs)
    assert np.isfinite(m.predict_log_prob(s, 1))
