from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from causal_rps.models import M2, M3, M4, M5, M6
from causal_rps.models.base import OnlineModel


@dataclass
class FixedM0State:
    p: np.ndarray


class FixedM0(OnlineModel):
    name = "M0"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def initialize(self, priors, runin_data):
        return FixedM0State(self.rng.dirichlet(np.ones(3)))

    def predict_log_prob(self, state, next_a):
        return math.log(max(float(state.p[next_a]), 1e-300))

    def update(self, state, obs):
        return state


@dataclass
class FixedM1State:
    p_dir: np.ndarray
    prev_a: int | None = None
    prev_r: int | None = None


class FixedM1(OnlineModel):
    name = "M1"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def initialize(self, priors, runin_data):
        state = FixedM1State(np.stack([self.rng.dirichlet(np.ones(3)) for _ in range(3)]))
        if runin_data.observations:
            state.prev_a = runin_data.observations[-1].a
            state.prev_r = runin_data.observations[-1].r
        return state

    def _p_action(self, state):
        if state.prev_a is None or state.prev_r is None:
            return np.ones(3) / 3
        row = {-1: 0, 0: 1, 1: 2}[state.prev_r]
        p = np.zeros(3)
        for d in range(3):
            p[(state.prev_a + d) % 3] = state.p_dir[row, d]
        return p

    def predict_log_prob(self, state, next_a):
        return math.log(max(float(self._p_action(state)[next_a]), 1e-300))

    def update(self, state, obs):
        state.prev_a, state.prev_r = obs.a, obs.r
        return state


def make_generator(model_name, seed, context_kind="x3"):
    if model_name == "M0": return FixedM0(seed)
    if model_name == "M1": return FixedM1(seed)
    if model_name == "M2": return M2(n_particles=1, seed=seed)
    if model_name == "M3": return M3(n_particles=1, seed=seed, context_kind=context_kind)
    if model_name == "M4": return M4(n_particles=1, seed=seed, context_kind=context_kind)
    if model_name == "M5": return M5(n_particles=1, seed=seed, context_kind=context_kind)
    if model_name == "M6": return M6(n_particles=1, seed=seed, context_kind=context_kind)
    raise ValueError(model_name)
