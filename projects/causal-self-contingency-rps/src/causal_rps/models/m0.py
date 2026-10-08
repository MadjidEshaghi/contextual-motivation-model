from dataclasses import dataclass
import math
import numpy as np

from causal_rps.models.base import OnlineModel
from causal_rps.schema import ModelObservation, RunInData


@dataclass
class M0State:
    alpha: np.ndarray


class M0(OnlineModel):
    name = "M0"

    def initialize(self, priors, runin_data: RunInData) -> M0State:
        alpha = np.ones(3, dtype=float)
        for obs in runin_data.observations:
            alpha[obs.a] += 1.0
        return M0State(alpha)

    def predict_log_prob(self, state: M0State, next_a: int) -> float:
        p = state.alpha / state.alpha.sum()
        return math.log(max(float(p[next_a]), 1e-300))

    def update(self, state: M0State, obs: ModelObservation) -> M0State:
        state.alpha[obs.a] += 1.0
        return state
