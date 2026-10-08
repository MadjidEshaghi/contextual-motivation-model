# simulator_spec.md — v1.2

**Status:** locked implementation contract before `gate1_runner.py`.

This document translates Final Closure Document v3.0 into a mechanical coding contract. It does not redefine the scientific architecture.

## 0. Scoring target and interfaces

The Gate 1 target is participant choice only:

[
ell_t^{(k)}=log P_{M_k}(a_tmid D_{<t},I_t).
]

Opponent actions and rewards update states after the choice has been scored; they are never included in the scored observation.

### 0.1 Online interface

```python
class OnlineModel:
    def initialize(self, priors, runin_data): ...
    def predict_log_prob(self, state, next_a: int) -> float: ...
    def update(self, state, obs): ...
```

### 0.2 Batch interface

Permitted only for conjugate/reference implementations (M0/M1):

```python
class BatchModel:
    def fit(self, train_data, priors): ...
    def predict_log_prob(self, posterior, next_a: int, history) -> float: ...
```

### 0.3 Posterior predictive requirement

Every scored probability must integrate parameter uncertainty:

[
P(a_tmid D_{<t})=int P(a_tmidartheta,D_{<t})p(arthetamid D_{<t}),dartheta.
]

Conjugate exact or sequential Monte Carlo/particle approximations are allowed. Plug-in MAP scoring is not allowed.

### 0.4 Predict-before-update

Always score `a_t` from the state built only from trials `<t`, then update with trial `t`.

### 0.5 Leakage test

Permuting all future observations after trial `t` must leave all predictions through `t` numerically unchanged.

## 1. Data schema

```python
@dataclass(frozen=True)
class ModelObservation:
    t: int
    a: int
    b: int
    r: int
    forced_action: int | None
    score_choice: bool

@dataclass(frozen=True)
class TrialMeta:
    condition: str
    block_id: int
    pair_id: int | None
    probe_flag: bool
    probe_phase: str | None
    generating_model: str | None
    params_true: dict | None
    seed: int
```

Condition labels, true parameters and generating-model labels are never passed to a cognitive model.

## 2. Environment is online and separate from cognition

[
	ext{Dataset}=	ext{ParticipantModel }M_k+	ext{OnlineCommonEnvironment}.
]

The environment configuration, RNG streams, probe sequence, condition schedule and bot parameters may be fixed in advance; **CE/CB opponent actions may not be pre-generated independently of participant actions**.

Exact replay ordering:

1. generate the self-contingent donor sequence online (CE or CB);
2. store the realized opponent-action sequence;
3. run the matched non-contingent recipient (NE or NB) against exact replay.

Environment parameters are distinct from participant beliefs:

```python
@dataclass(frozen=True)
class EnvironmentConfig:
    beta_env: float
    lambda_env: float
    alpha_env: float = 1.0
    epsilon_b: float = 0.05
```

[
eta_{m env}
eqeta_C.
]

## 3. Context semantics

Causal models distinguish:

- `x_self`: participant-related history used by CE/CB;
- `w_opp`: opponent-only/exogenous history usable by N.

Thus

[
N:P(b_tmid w_t), qquad CE/CB:P(b_tmid x_t^{self},w_t).
]

M3 is deliberately a strong correlational competitor and may estimate (P(b_tmid x_t^{self},w_t)), but it has no interventional semantics (do(a)Rightarrow b).

## 4. Run-in

Run-in consists of 90 unscored trials:

- Phase A: 30 trials, shaping bonus 0.5.
- Phase B: 20 trials, bonus linearly decays 0.5→0.
- Phase C: 40 trials, no shaping.

Directional loss-conditioned rule:

[
r_{t-1}=LRightarrow a_t=operatorname{rotate}_{+1}(a_{t-1}).
]

Every model receives the identical raw run-in observations and primes its own internal state from those observations.

## 5. Priors and fixed constants

M0:
[
psim Dirichlet(1,1,1).
]

M1, for each previous outcome:
[
p_r(	ext{stay},+1,-1)sim Dirichlet(1,1,1).
]

Primary priors:

- (gammasim LogNormal(log2,0.5))
- (lambda_Hsim Beta(8,2))
- (lambda_Osim Beta(8,2))
- (eta_Csim LogNormal(log3,0.6))
- (alpha_Qsim Beta(2,2))
- (eta_Qsim LogNormal(log2,0.6))
- Exp.2 only: (omegasim Beta(1,19))

Broad priors:

- (gammasim HalfNormal(5))
- (lambda_H,lambda_Osim Beta(2,2))
- (eta_C,eta_Qsim HalfNormal(6))
- (alpha_Qsim Uniform(0,1))
- Exp.2 only: (omegasim Beta(1,9))

Fixed:
`ALPHA_H=1`, `ALPHA_O=1`, `BETA_R=1`, `Q0_THETA=(1/3,1/3,1/3)`, `EPSILON_B=0.05`.

Gate 1 uses marginal (eta_C), not a hierarchical cohort prior.

## 6. Model definitions

### M0 — IID multinomial
Conjugate Dirichlet-multinomial participant choice model.

### M1 — directional conditional response
[
d_t=(a_t-a_{t-1})mod3,
]
with (P(d_tmid r_{t-1})) estimated by three Dirichlet posteriors.

### M2 — model-free Q-learning
[
Q_{t+1}(a_t)=Q_t(a_t)+alpha_Q[r_t-Q_t(a_t)],
]
[
P(a_t=a)propto e^{eta_QQ_t(a)}.
]
Parameter uncertainty is integrated sequentially by SMC.

### M3 — strong Bayesian correlational opponent model
Tracks (P(b_tmid x_t^{self},w_t)) with forgetting (lambda_O):
[
P(a_t=a)propto exp{gammasum_b q_O(bmid x_t,w_t)U(a,b)}.
]
No causal/interventional representation.

### M4 — hierarchical causal state, expected-value control
[
Theta_tin{N,CE,CB}.
]
N uses (P(bmid w)); CE/CB use participant-contingent likelihoods. CE uses (eta_C); CB uses `EPSILON_B`. Habit kernel decays with (lambda_H).

`POLICY_HORIZON=2`, `N_POLICIES=9`.

M4 uses two-step expected-value planning with zero epistemic bonus.

### M5 — Active Inference without causal state
Habit + opponent-statistics belief, horizon 2:
[
mathcal I_t^{M5}(pi)=IG(q_O;b_{t+1}midpi),
]
[
G_t^{M5}(pi)=-ar V_t(pi)-w_Imathcal I_t^{M5}(pi).
]
Primary (w_I=1); 0.25 and 2.0 are sensitivity bounds only.

### M6 — Active Inference with causal state
Same planning horizon and policy set:
[
mathcal I_t^{M6}(pi)=IG(Theta;b_{t+1}midpi),
]
[
G_t^{M6}(pi)=-ar V_t(pi)-mathcal I_t^{M6}(pi).
]

## 7. Probe and scored trials

Actual design block length: 196.

- 1–160 free/natural play;
- 161–184 forced perturbation, `score_choice=False`;
- 185–196 free-response identification window, `score_choice=True`.

Forced trials update beliefs/states but never enter participant-choice likelihood.

## 8. Forward prequential score

```python
state = model.initialize(priors, runin_data)
for t in range(T):
    lp_t = model.predict_log_prob(state, obs[t].a)
    if t >= W_WARM and obs[t].score_choice:
        score += lp_t
    state = model.update(state, obs[t])
```

No backward smoothing, lookahead, random leave-one-trial-out or after-score hyperparameter tuning.

## 9. Gate 1

Primary generating models are M0–M6. M5 weak/strong are sensitivity fits only.

`N_sim=500` per generating model per T.

- `T=196`: actual-design audit.
- `T=480`: stress test: 1–444 natural play, 445–468 forced probe, 469–480 free-response probe.

Failure at T=196 cannot be rescued by success at T=480.

Critical pairs:

```python
[
    ("M3","M4"),
    ("M4","M6"),
    ("M5","M6"),
]
```

Thresholds:
- overall recovery ≥0.80: Go
- 0.70≤overall<0.80: Conditional Go
- overall<0.70: No-Go
- each direction of each critical pair ≥0.75 at T=196

Primary score: argmax total forward-prequential participant-choice log predictive density.

## 10. Outputs

M4/M6 additionally store `q_N`, `q_CE`, `q_CB`, `q_C=q_CE+q_CB=1-q_N`, and signed `X_t`.

M3/M5 store opponent-action predictive probabilities.

Posterior samples are retained only for critical-pair subsets unless an audit entry authorizes wider storage.

## 11. Reproducibility

Every run records run id, spec version, branch/commit SHA, prior set, T/N_sim, seeds, environment config, hardware/runtime, context specification, confusion matrix summary and Gate decision.

Any deviation must be entered in `locked_log.md`.

## 12. Source-of-truth constants

```python
W_WARM = 10
W_PRIMARY = 30
T_BLOCK = 196
RUNIN_LEN = 90
RUNIN_PHASE_A = 30
RUNIN_PHASE_B = 20
RUNIN_PHASE_C = 40
EARLY_WINDOW = (31, 50)
LATE_WINDOW = (81, 100)
PROBE_FORCED = (161, 184)
PROBE_FREE = (185, 196)
ALPHA_H = 1.0
ALPHA_O = 1.0
BETA_R = 1.0
Q0_THETA = (1/3, 1/3, 1/3)
EPSILON_B = 0.05
POLICY_HORIZON = 2
N_POLICIES = 9
EPISTEMIC_WEIGHT_M5_STANDARD = 1.0
EPISTEMIC_WEIGHT_M5_WEAK = 0.25
EPISTEMIC_WEIGHT_M5_STRONG = 2.0
N_SIM_GATE1 = 500
T_GRID = (196, 480)
CRITICAL_PAIRS = (("M3","M4"), ("M4","M6"), ("M5","M6"))
THRESHOLD_OVERALL_GO = 0.80
THRESHOLD_OVERALL_COND = 0.70
THRESHOLD_CRITICAL = 0.75
SCORE_TARGET = "participant_action"
STRESS_T_NATURAL_END = 444
STRESS_T_PROBE_FORCED = (445, 468)
STRESS_T_PROBE_FREE = (469, 480)
```
