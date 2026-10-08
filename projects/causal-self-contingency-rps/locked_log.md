# Locked project log

This log records implementation deviations, audit findings, and Gate decisions. Entries are append-only.

## Template

```yaml
date: YYYY-MM-DD
run_id: ...
commit_sha: ...
spec_version: ...
data:
  kind: synthetic | pilot | human
  generating_model: ...
  n_sim: ...
  T: ...
metric: forward-prequential participant-choice log predictive density
result:
  overall_recovery: ...
  critical_pairs: ...
decision: Go | Conditional Go | No-Go | Implementation-only
action: ...
reason: ...
```

---

## Entry 0001

```yaml
date: 2026-10-09
run_id: repository_initialization
commit_sha: pending
spec_version: 1.2
data:
  kind: synthetic
  generating_model: null
  n_sim: 0
  T: null
metric: null
result:
  summary: "Initialized project workspace from Final Closure Document v3.0 and simulator specification v1.2."
decision: Implementation-only
action: "Implement common schemas, online environment, M0-M6 interfaces, forward-prequential scorer, leakage tests, and pre-flight runner."
reason: "Start of execution phase; no scientific architecture change."
```


---

## Entry 0002

```yaml
date: 2026-10-09
run_id: block_boundary_state_reset
commit_sha: f98416789f9312d781af074c66634871eab9f7f1
spec_version: 1.2
data:
  kind: synthetic
  generating_model: null
  n_sim: 0
  T: null
metric: null
result:
  summary: "Implementation clarification: the 90-trial run-in primes participant-specific regularity/habit state only. Opponent-specific kernels, causal-state belief q_theta, and block history reset because the experimental block begins with a new opponent."
decision: Implementation-only
action: "M3 resets opponent statistics; M5 primes habit but resets opponent statistics/history; M4/M6 prime habit but reset q_theta to uniform, N-kernel, and history."
reason: "Prevents information about the run-in opponent from leaking into the new experimental opponent."
```

---

## Entry 0003

```yaml
date: 2026-10-09
run_id: synthetic_gate1_technical_defaults
commit_sha: f98416789f9312d781af074c66634871eab9f7f1
spec_version: 1.2
data:
  kind: synthetic
  generating_model: all
  n_sim: 0
  T: [196, 480]
metric: null
result:
  summary: "Synthetic Gate-1 implementation uses target run-in adherence sampled Uniform(0.50,0.70), a 24-trial forced probe sequence of 8 R then 8 P then 8 S, and default environment parameters beta_env=3.0, lambda_env=0.8, alpha_env=1.0, epsilon_b=0.05."
decision: Implementation-only
action: "Treat these as simulator technical defaults only. They are not human-protocol results and do not change SESOI, Gate thresholds, or scientific claims."
reason: "The locked human protocol specified run-in structure and an exogenous probe but did not mechanically define the synthetic Gate-1 generator values."
```

---

## Entry 0004

```yaml
date: 2026-10-09
run_id: local_preflight_runtime
commit_sha: f98416789f9312d781af074c66634871eab9f7f1
spec_version: 1.2
data:
  kind: synthetic
  generating_model: M0-M6
  n_sim: [1, 5]
  T: 196
metric: forward-prequential participant-choice log predictive density
result:
  tests: "14/14 local pytest tests passed."
  vectorization: "M4/M5/M6 policy evaluation vectorized over particles without changing equations."
  timing_sample_seconds:
    M3: 0.141
    M4: 1.491
    M5: 0.796
    M6: 2.108
  n1_end_to_end_runtime_seconds: 21.97
  n1_overall_recovery: 0.7142857142857143
  n1_interpretation: "IMPLEMENTATION_ONLY; N=1 per generating model is scientifically meaningless."
  n5_result: "Did not complete within the local execution limits, including a parallel attempt; no recovery estimate retained or interpreted."
decision: Implementation-only
action: "Move N=5 pre-flight to CI/compute outside the constrained local execution environment before any scientific Gate-1 run."
reason: "Runtime/engineering validation only; no Go/No-Go decision is permitted from these runs."
```


---

## Entry 0005

```yaml
date: 2026-10-09
run_id: github_ci_activation
commit_sha: pending
spec_version: 1.2
data:
  kind: synthetic
  generating_model: M0-M6
  n_sim: 5
  T: 196
metric: forward-prequential participant-choice log predictive density
result:
  status: "CI workflow enabled on the default branch; this run is implementation/runtime validation only."
decision: Implementation-only
action: "Trigger draft PR CI: run pytest, then N_sim=5 preflight with 4 particles and 2 workers; archive JSON artifact."
reason: "Move preflight outside the constrained local execution environment. No scientific Gate decision is permitted."
```
