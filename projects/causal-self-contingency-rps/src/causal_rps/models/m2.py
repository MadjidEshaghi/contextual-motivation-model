from dataclasses import dataclass
import math
import numpy as np

from causal_rps.models.base import OnlineModel
from causal_rps.payoffs import softmax
from causal_rps.schema import ModelObservation, RunInData


@dataclass
class M2State:
    alpha_q: np.ndarray
    beta_q: np.ndarray
    weights: np.ndarray
    q: np.ndarray


class M2(OnlineModel):
    """SMC approximation to posterior uncertainty over alpha_Q and beta_Q."""

    name = "M2"

    def __init__(self, n_particles: int = 256, seed: int = 0):
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)

    def initialize(self, priors, runin_data: RunInData) -> M2State:
        a = self.rng.beta(2, 2, size=self.n_particles)
        b = np.exp(self.rng.normal(np.log(2), 0.6, size=self.n_particles))
        state = M2State(
            a,
            b,
            np.ones(self.n_particles) / self.n_particles,
            np.zeros((self.n_particles, 3)),
        )
        for obs in runin_data.observations:
            self._assimilate(state, obs)
        return state

    def _particle_probs(self, state: M2State) -> np.ndarray:
        return np.stack([
            softmax(state.beta_q[i] * state.q[i])
            for i in range(self.n_particles)
        ])

    def predict_log_prob(self, state: M2State, next_a: int) -> float:
        pp = self._particle_probs(state)[:, next_a]
        p = float(np.dot(state.weights, pp))
        return math.log(max(p, 1e-300))

    def _resample_if_needed(self, state: M2State) -> None:
        ess = 1.0 / np.sum(state.weights ** 2)
        if ess < self.n_particles / 2:
            idx = self.rng.choice(self.n_particles, self.n_particles, p=state.weights)
            state.alpha_q = state.alpha_q[idx]
            state.beta_q = state.beta_q[idx]
            state.q = state.q[idx]
            state.weights[:] = 1.0 / self.n_particles

    def _assimilate(self, state: M2State, obs: ModelObservation) -> None:
        probs = self._particle_probs(state)[:, obs.a]
        state.weights *= np.maximum(probs, 1e-300)
        s = state.weights.sum()
        state.weights[:] = state.weights / s if s > 0 else 1.0 / self.n_particles

        idx = np.arange(self.n_particles)
        old = state.q[idx, obs.a]
        state.q[idx, obs.a] = old + state.alpha_q * (obs.r - old)
        self._resample_if_needed(state)

    def update(self, state: M2State, obs: ModelObservation) -> M2State:
        self._assimilate(state, obs)
        return state
