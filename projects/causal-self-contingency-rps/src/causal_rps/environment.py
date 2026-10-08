from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from causal_rps.beliefs import cb_kernel, ce_kernel
from causal_rps.constants import PROBE_FORCED, STRESS_T_PROBE_FORCED, T_BLOCK
from causal_rps.contexts import History, advance, self_context
from causal_rps.payoffs import payoff, rotate_plus_one
from causal_rps.schema import Dataset, EnvironmentConfig, ModelObservation, RunInData, RunInSummary, TrialMeta


@dataclass
class EnvironmentState:
    history: History = field(default_factory=History)
    evidence: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)


class OnlineCommonEnvironment:
    """Online CE/CB environment; it never uses the current participant action."""

    def __init__(self, condition, config, seed=0, context_kind="x3", replay_sequence=None):
        if condition not in {"CE", "CB", "NE", "NB"}:
            raise ValueError(condition)
        self.condition = condition
        self.config = config
        self.context_kind = context_kind
        self.rng = np.random.default_rng(seed)
        self.state = EnvironmentState()
        self.replay_sequence = replay_sequence

    def _evidence(self, ctx):
        if ctx not in self.state.evidence:
            self.state.evidence[ctx] = np.zeros(3, dtype=float)
        return self.state.evidence[ctx]

    def opponent_distribution(self, t):
        if self.condition in {"NE", "NB"}:
            if self.replay_sequence is None:
                raise ValueError("replay sequence required for N condition")
            p = np.zeros(3); p[int(self.replay_sequence[t])] = 1.0
            return p

        ctx = self_context(self.state.history, self.context_kind)
        e = self._evidence(ctx)
        p_hat = e + self.config.alpha_env
        p_hat = p_hat / p_hat.sum()
        if self.condition == "CE":
            return ce_kernel(p_hat, self.config.beta_env)
        return cb_kernel(p_hat, self.config.epsilon_b)

    def sample_opponent(self, t):
        p = self.opponent_distribution(t)
        return int(self.rng.choice(3, p=p))

    def update(self, a, b, r):
        if self.condition in {"CE", "CB"}:
            for arr in self.state.evidence.values():
                arr *= self.config.lambda_env
            ctx = self_context(self.state.history, self.context_kind)
            self._evidence(ctx)[a] += 1.0
        advance(self.state.history, a, b, r)


def forced_probe_action(t_zero_based, T):
    if T == T_BLOCK:
        lo, hi = PROBE_FORCED
    elif T == 480:
        lo, hi = STRESS_T_PROBE_FORCED
    else:
        return None
    t1 = t_zero_based + 1
    if not (lo <= t1 <= hi):
        return None
    j = t1 - lo
    return min(j // 8, 2)  # 8 R, 8 P, 8 S


def simulate_runin(seed, target_adherence=None):
    """Common unscored synthetic run-in used only to prime Gate-1 models."""
    rng = np.random.default_rng(seed)
    if target_adherence is None:
        target_adherence = float(rng.uniform(0.5, 0.7))

    obs = []
    prev_a = int(rng.integers(0, 3))
    prev_r = 0
    losses = follows = 0

    for t in range(90):
        frac = min((t + 1) / 30, 1.0)
        adhere = 1 / 3 + frac * (target_adherence - 1 / 3)

        if prev_r == -1:
            losses += 1
            target = rotate_plus_one(prev_a)
            if rng.random() < adhere:
                a = target; follows += 1
            else:
                a = int(rng.choice([x for x in range(3) if x != target]))
        else:
            a = int(rng.integers(0, 3))

        b = int(rng.integers(0, 3))
        r = payoff(a, b)
        obs.append(ModelObservation(t=t, a=a, b=b, r=r))
        prev_a, prev_r = a, r

    adherence = follows / losses if losses else 0.0
    counts = np.bincount([o.a for o in obs[-40:]], minlength=3).astype(float) + 1
    p = counts / counts.sum()
    h = -float(np.sum(p * np.log(p)))
    chi0 = 1.0 - h / np.log(3)
    e0 = float(max(p[2] - p[1], p[0] - p[2], p[1] - p[0]))
    return RunInData(obs, RunInSummary(adherence, chi0, e0))


def action_probs_from_model(model, state):
    lp = np.array([model.predict_log_prob(state, a) for a in range(3)], dtype=float)
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def simulate_single(participant_model, condition, T, seed, env_config,
                    replay_sequence=None, runin=None, model_name=None):
    rng = np.random.default_rng(seed)
    if runin is None:
        runin = simulate_runin(seed + 1000003)

    state = participant_model.initialize({}, runin)
    env = OnlineCommonEnvironment(condition, env_config, seed + 17, replay_sequence=replay_sequence)
    observations, meta = [], []

    for t in range(T):
        forced = forced_probe_action(t, T)
        if forced is None:
            a = int(rng.choice(3, p=action_probs_from_model(participant_model, state)))
            score_choice = True
            phase = "free" if t >= T - 12 else None
        else:
            a = int(forced); score_choice = False; phase = "forced"

        b = env.sample_opponent(t)
        r = payoff(a, b)
        ob = ModelObservation(t=t, a=a, b=b, r=r, forced_action=forced, score_choice=score_choice)
        observations.append(ob)
        meta.append(TrialMeta(condition, 0, None, phase is not None, phase, model_name, None, seed))
        state = participant_model.update(state, ob)
        env.update(a, b, r)

    return Dataset(observations, meta, runin, T)


def simulate_yoked_pair(donor_model, recipient_model, contingent_condition, T, seed,
                        env_config, donor_runin=None, recipient_runin=None, model_name=None):
    if contingent_condition not in {"CE", "CB"}:
        raise ValueError("donor condition must be CE or CB")
    replay_condition = "NE" if contingent_condition == "CE" else "NB"
    donor = simulate_single(donor_model, contingent_condition, T, seed, env_config,
                            runin=donor_runin, model_name=model_name)
    replay = [o.b for o in donor.observations]
    recipient = simulate_single(recipient_model, replay_condition, T, seed + 1, env_config,
                                replay_sequence=replay, runin=recipient_runin, model_name=model_name)
    return donor, recipient
