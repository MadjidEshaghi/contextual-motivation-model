import numpy as np

from causal_rps.environment import OnlineCommonEnvironment, forced_probe_action, simulate_runin
from causal_rps.schema import EnvironmentConfig


def test_runin_length():
    r = simulate_runin(1, 0.6)
    assert len(r.observations) == 90
    assert 0 <= r.summary.adherence <= 1


def test_probe_forced_exactly_24_actual_design():
    vals = [forced_probe_action(t, 196) for t in range(196)]
    assert sum(v is not None for v in vals) == 24


def test_replay_is_exact():
    seq = [0, 1, 2, 2]
    e = OnlineCommonEnvironment("NE", EnvironmentConfig(), seed=3, replay_sequence=seq)
    assert [e.sample_opponent(t) for t in range(4)] == seq


def test_contingent_distribution_normalizes():
    e = OnlineCommonEnvironment("CE", EnvironmentConfig(), seed=4)
    assert np.isclose(e.opponent_distribution(0).sum(), 1.0)
