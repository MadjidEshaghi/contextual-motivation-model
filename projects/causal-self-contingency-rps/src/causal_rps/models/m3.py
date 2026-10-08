from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from causal_rps.constants import ALPHA_O
from causal_rps.contexts import History, advance, self_context
from causal_rps.models.base import OnlineModel
from causal_rps.payoffs import U, softmax
from causal_rps.schema import ModelObservation, RunInData
from causal_rps.smc import maybe_resample, normalize_weights


@dataclass
class M3State:
    gamma: np.ndarray
    lambda_o: np.ndarray
    weights: np.ndarray
    evidence: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    history: History = field(default_factory=History)


class M3(OnlineModel):
    name = "M3"

    def __init__(self, n_particles=256, seed=0, context_kind="x3"):
        self.n_particles = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.context_kind = context_kind

    def initialize(self, priors, runin_data: RunInData):
        # Run-in opponent is different; M3 has no participant-specific habit
        # state, so opponent statistics and history reset at block start.
        return M3State(
            gamma=np.exp(self.rng.normal(np.log(2.0), 0.5, self.n_particles)),
            lambda_o=self.rng.beta(8.0, 2.0, self.n_particles),
            weights=np.ones(self.n_particles) / self.n_particles,
        )

    def _get_evidence(self, state, ctx):
        if ctx not in state.evidence:
            state.evidence[ctx] = np.zeros((self.n_particles, 3), dtype=float)
        return state.evidence[ctx]

    def _opponent_probs(self, state):
        ctx = self_context(state.history, self.context_kind)
        alpha = self._get_evidence(state, ctx) + ALPHA_O
        return alpha / alpha.sum(axis=1, keepdims=True)

    def _choice_probs(self, state):
        q_opp = self._opponent_probs(state)
        ev = q_opp @ U.T
        out = np.empty_like(ev)
        for i in range(self.n_particles):
            out[i] = softmax(state.gamma[i] * ev[i])
        return out

    def predict_log_prob(self, state, next_a):
        p = float(np.dot(state.weights, self._choice_probs(state)[:, next_a]))
        return math.log(max(p, 1e-300))

    def _decay(self, state):
        for arr in state.evidence.values():
            arr *= state.lambda_o[:, None]

    def _assimilate(self, state, obs, update_parameter_weights=True):
        ctx = self_context(state.history, self.context_kind)
        if update_parameter_weights:
            state.weights *= np.maximum(self._choice_probs(state)[:, obs.a], 1e-300)
            normalize_weights(state.weights)
        self._decay(state)
        self._get_evidence(state, ctx)[:, obs.b] += 1.0

        idx = maybe_resample(state.weights, self.rng)
        if idx is not None:
            state.gamma = state.gamma[idx]
            state.lambda_o = state.lambda_o[idx]
            for key in list(state.evidence):
                state.evidence[key] = state.evidence[key][idx]
        advance(state.history, obs.a, obs.b, obs.r)

    def update(self, state, obs):
        self._assimilate(state, obs, True)
        return state
