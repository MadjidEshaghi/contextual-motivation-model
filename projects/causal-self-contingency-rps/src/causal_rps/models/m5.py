from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from causal_rps.beliefs import dirichlet_information_gain_batch
from causal_rps.constants import ALPHA_H, ALPHA_O
from causal_rps.contexts import History, advance, self_context
from causal_rps.models.base import OnlineModel
from causal_rps.payoffs import U
from causal_rps.schema import ModelObservation, RunInData
from causal_rps.smc import maybe_resample, normalize_weights


@dataclass
class M5State:
    gamma: np.ndarray
    lambda_h: np.ndarray
    lambda_o: np.ndarray
    weights: np.ndarray
    habit: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    opponent: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    history: History = field(default_factory=History)


class M5(OnlineModel):
    name = "M5"

    def __init__(self, n_particles=128, seed=0, context_kind="x3", epistemic_weight=1.0):
        self.n_particles = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.context_kind = context_kind
        self.epistemic_weight = float(epistemic_weight)

    def initialize(self, priors, runin_data: RunInData):
        state = M5State(
            gamma=np.exp(self.rng.normal(np.log(2.0), 0.5, self.n_particles)),
            lambda_h=self.rng.beta(8, 2, self.n_particles),
            lambda_o=self.rng.beta(8, 2, self.n_particles),
            weights=np.ones(self.n_particles) / self.n_particles,
        )
        h = History()
        for obs in runin_data.observations:
            ctx = self_context(h, self.context_kind)
            for arr in state.habit.values():
                arr *= state.lambda_h[:, None]
            self._arr(state.habit, ctx)[:, obs.a] += 1.0
            advance(h, obs.a, obs.b, obs.r)
        state.history = History()
        state.opponent.clear()
        return state

    def _arr(self, table, key):
        if key not in table:
            table[key] = np.zeros((self.n_particles, 3), dtype=float)
        return table[key]

    def _habit_p(self, state, ctx):
        a = self._arr(state.habit, ctx) + ALPHA_H
        return a / a.sum(axis=1, keepdims=True)

    def _opp_alpha(self, state, ctx):
        return self._arr(state.opponent, ctx) + ALPHA_O

    def _opp_p(self, state, ctx):
        a = self._opp_alpha(state, ctx)
        return a / a.sum(axis=1, keepdims=True)

    @staticmethod
    def _next_history(history, a, b, r):
        return History(prev_a=a, prev2_a=history.prev_a, prev_b=b, prev_r=r)

    def _choice_probs(self, state):
        n = self.n_particles
        ctx = self_context(state.history, self.context_kind)
        h_now = self._habit_p(state, ctx)
        q_b1 = self._opp_p(state, ctx)

        logw = np.empty((n, 9))
        policy_a1 = np.empty(9, dtype=int)
        col = 0

        for a1 in range(3):
            r1_exp = q_b1 @ U[a1]
            for a2 in range(3):
                h2_exp = np.zeros(n)
                r2_exp = np.zeros(n)
                ig_exp = np.zeros(n)

                for b1 in range(3):
                    pb1 = q_b1[:, b1]
                    r1 = int(U[a1, b1])
                    h_next = self._next_history(state.history, a1, b1, r1)
                    ctx2 = self_context(h_next, self.context_kind)
                    h2_exp += pb1 * self._habit_p(state, ctx2)[:, a2]
                    alpha2 = self._opp_alpha(state, ctx2)
                    q_b2 = alpha2 / alpha2.sum(axis=1, keepdims=True)
                    r2_exp += pb1 * (q_b2 @ U[a2])
                    ig_exp += pb1 * dirichlet_information_gain_batch(alpha2)

                prior = np.maximum(h_now[:, a1] * np.maximum(h2_exp, 1e-12), 1e-300)
                value = r1_exp + r2_exp + self.epistemic_weight * ig_exp
                logw[:, col] = np.log(prior) + state.gamma * value
                policy_a1[col] = a1
                col += 1

        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        out = np.zeros((n, 3))
        for j in range(9):
            out[:, policy_a1[j]] += w[:, j]
        return out

    def predict_log_prob(self, state, next_a):
        p = float(np.dot(state.weights, self._choice_probs(state)[:, next_a]))
        return math.log(max(p, 1e-300))

    def _decay(self, state):
        for arr in state.habit.values():
            arr *= state.lambda_h[:, None]
        for arr in state.opponent.values():
            arr *= state.lambda_o[:, None]

    def _assimilate(self, state, obs, update_parameter_weights=True):
        ctx = self_context(state.history, self.context_kind)
        if update_parameter_weights:
            state.weights *= np.maximum(self._choice_probs(state)[:, obs.a], 1e-300)
            normalize_weights(state.weights)

        self._decay(state)
        self._arr(state.habit, ctx)[:, obs.a] += 1.0
        self._arr(state.opponent, ctx)[:, obs.b] += 1.0

        idx = maybe_resample(state.weights, self.rng)
        if idx is not None:
            state.gamma = state.gamma[idx]
            state.lambda_h = state.lambda_h[idx]
            state.lambda_o = state.lambda_o[idx]
            for table in (state.habit, state.opponent):
                for key in list(table):
                    table[key] = table[key][idx]

        advance(state.history, obs.a, obs.b, obs.r)

    def update(self, state, obs):
        self._assimilate(state, obs, True)
        return state
