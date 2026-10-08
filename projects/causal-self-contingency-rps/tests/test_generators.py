from causal_rps.environment import simulate_yoked_pair
from causal_rps.generators import make_generator
from causal_rps.schema import EnvironmentConfig


def test_yoked_generator_opponent_sequences_match():
    d, r = simulate_yoked_pair(
        make_generator("M0", 1),
        make_generator("M0", 2),
        "CE",
        40,
        10,
        EnvironmentConfig(),
    )
    assert [o.b for o in d.observations] == [o.b for o in r.observations]
