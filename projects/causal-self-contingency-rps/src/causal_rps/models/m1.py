from dataclasses import dataclass
import math
import numpy as np

from causal_rps.models.base import OnlineModel
from causal_rps.schema import ModelObservation, RunInData

OUTCOME_TO_ROW = {-1: 0, 0: 1, 1: 2}  # L,T,W


@dataclass
class M1State:
    alpha: np.ndarray
    prev_a: int | None
    prev_r: int | None


class M1(OnlineModel):
    name = "M1"

    def initialize(self, priors, runin_data: RunInData) -> M1State:
        state = M1State(np.ones((3, 3), dtype=float), None, None)
        for obs in runin_data.observations:
            self.update(state, obs)
        return state

    def _action_probs(self, state: M1State) -> np.ndarray:
        if state.prev_a is None or state.prev_r is None:
            return np.ones(3) / 3
        row = state.alpha[OUTCOME_TO_ROW[state.prev_r]]
        p_dir = row / row.sum()
        p_action = np.zeros(3)
        for d in range(3):
            p_action[(state.prev_a + d) % 3] += p_dir[d]
        return p_action

    def predict_log_prob(self, state: M1State, next_a: int) -> float:
        p = self._action_probs(state)
        return math.log(max(float(p[next_a]), 1e-300))

    def update(self, state: M1State, obs: ModelObservation) -> M1State:
        if state.prev_a is not None and state.prev_r is not None:
            d = (obs.a - state.prev_a) % 3
            state.alpha[OUTCOME_TO_ROW[state.prev_r], d] += 1.0
        state.prev_a, state.prev_r = obs.a, obs.r
        return state
