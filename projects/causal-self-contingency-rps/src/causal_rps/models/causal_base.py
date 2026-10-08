from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from causal_rps.beliefs import cb_kernel, ce_kernel, theta_information_gain
from causal_rps.constants import ALPHA_H, ALPHA_O, EPSILON_B, Q0_THETA
from causal_rps.contexts import History, advance, opponent_context, self_context
from causal_rps.models.base import OnlineModel
from causal_rps.payoffs import U
from causal_rps.schema import ModelObservation, RunInData
from causal_rps.smc import maybe_resample, normalize_weights


@dataclass
class CausalState:
    gamma: np.ndarray
    lambda_h: np.ndarray
    lambda_o: np.ndarray
    beta_c: np.ndarray
    weights: np.ndarray
    q_theta: np.ndarray
    habit: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    opp_n: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)
    history: History = field(default_factory=History)


class CausalModelBase(OnlineModel):
    epistemic_weight = 0.0

    def __init__(self, n_particles=128, seed=0, context_kind="x3"):
        self.n_particles = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.context_kind = context_kind

    def initialize(self, priors, runin_data: RunInData):
        state = CausalState(
            gamma=np.exp(self.rng.normal(np.log(2.0), 0.5, self.n_particles)),
            lambda_h=self.rng.beta(8, 2, self.n_particles),
            lambda_o=self.rng.beta(8, 2, self.n_particles),
            beta_c=np.exp(self.rng.normal(np.log(3.0), 0.6, self.n_particles)),
            weights=np.ones(self.n_particles) / self.n_particles,
            q_theta=np.tile(np.asarray(Q0_THETA, dtype=float), (self.n_particles, 1)),
        )
        for obs in runin_data.observations:
            self._assimilate(state, obs, False)
        return state

    def _arr(self, table, key):
        if key not in table:
            table[key] = np.zeros((self.n_particles, 3), dtype=float)
        return table[key]

    def _habit_p(self, state, ctx):
        a = self._arr(state.habit, ctx) + ALPHA_H
        return a / a.sum(axis=1, keepdims=True)

    def _n_p(self, state, wctx):
        a = self._arr(state.opp_n, wctx) + ALPHA_O
        return a / a.sum(axis=1, keepdims=True)

    def _kernels_particle(self, state, i, history):
        x = self_context(history, self.context_kind)
        w = opponent_context(history)
        p_hat = self._habit_p(state, x)[i]
        return np.stack([
            self._n_p(state, w)[i],
            ce_kernel(p_hat, state.beta_c[i]),
            cb_kernel(p_hat, EPSILON_B),
        ])

    @staticmethod
    def _next_history(history, a, b, r):
        return History(prev_a=a, prev2_a=history.prev_a, prev_b=b, prev_r=r)

    def _policy_probs_particle(self, state, i):
        x = self_context(state.history, self.context_kind)
        h_now = self._habit_p(state, x)[i]
        k_now = self._kernels_particle(state, i, state.history)
        q0 = state.q_theta[i]
        p_b1 = q0 @ k_now
        logw, policies = [], []

        for a1 in range(3):
            r1_exp = float(np.dot(p_b1, U[a1]))
            for a2 in range(3):
                h2_exp = r2_exp = ig2_exp = 0.0
                for b1 in range(3):
                    pb1 = float(p_b1[b1])
                    r1 = int(U[a1, b1])
                    q1 = q0 * k_now[:, b1]
                    q1 = q1 / max(q1.sum(), 1e-300)
                    h_next = self._next_history(state.history, a1, b1, r1)
                    x2 = self_context(h_next, self.context_kind)
                    h2_exp += pb1 * float(self._habit_p(state, x2)[i, a2])
                    k2 = self._kernels_particle(state, i, h_next)
                    p_b2 = q1 @ k2
                    r2_exp += pb1 * float(np.dot(p_b2, U[a2]))
                    if self.epistemic_weight:
                        ig2_exp += pb1 * theta_information_gain(q1, k2)

                prior = max(float(h_now[a1]) * max(h2_exp, 1e-12), 1e-300)
                value = r1_exp + r2_exp + self.epistemic_weight * ig2_exp
                logw.append(math.log(prior) + state.gamma[i] * value)
                policies.append((a1, a2))

        z = np.asarray(logw)
        z -= z.max()
        w = np.exp(z); w /= w.sum()
        out = np.zeros(3)
        for prob, (a1, _) in zip(w, policies):
            out[a1] += prob
        return out

    def _choice_probs(self, state):
        return np.stack([self._policy_probs_particle(state, i) for i in range(self.n_particles)])

    def predict_log_prob(self, state, next_a):
        p = float(np.dot(state.weights, self._choice_probs(state)[:, next_a]))
        return math.log(max(p, 1e-300))

    def _decay(self, state):
        for arr in state.habit.values():
            arr *= state.lambda_h[:, None]
        for arr in state.opp_n.values():
            arr *= state.lambda_o[:, None]

    def _assimilate(self, state, obs, update_parameter_weights=True):
        x = self_context(state.history, self.context_kind)
        wctx = opponent_context(state.history)

        if update_parameter_weights:
            state.weights *= np.maximum(self._choice_probs(state)[:, obs.a], 1e-300)
            normalize_weights(state.weights)

        for i in range(self.n_particles):
            k = self._kernels_particle(state, i, state.history)
            post = state.q_theta[i] * k[:, obs.b]
            state.q_theta[i] = post / max(post.sum(), 1e-300)

        self._decay(state)
        self._arr(state.habit, x)[:, obs.a] += 1.0
        self._arr(state.opp_n, wctx)[:, obs.b] += 1.0

        idx = maybe_resample(state.weights, self.rng)
        if idx is not None:
            state.gamma = state.gamma[idx]
            state.lambda_h = state.lambda_h[idx]
            state.lambda_o = state.lambda_o[idx]
            state.beta_c = state.beta_c[idx]
            state.q_theta = state.q_theta[idx]
            for table in (state.habit, state.opp_n):
                for key in list(table):
                    table[key] = table[key][idx]

        advance(state.history, obs.a, obs.b, obs.r)

    def update(self, state, obs):
        self._assimilate(state, obs, True)
        return state

    def posterior_theta_mean(self, state):
        return np.average(state.q_theta, axis=0, weights=state.weights)
