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


---

## Entry 0006

```yaml
date: 2026-10-09
run_id: github_actions_preflight_run_37858027033
commit_sha: 46e4099f42a686a3d6c82417b2fd9866b8ad8621
spec_version: 1.2
data:
  kind: synthetic
  generating_model: M0-M6
  n_sim: 5
  T: 196
metric: forward-prequential participant-choice log predictive density
result:
  github_actions_run: 37858027033
  tests: "14/14 passed"
  particles: 4
  workers: 2
  runtime_sec: 27.086199123
  overall_recovery: 0.7428571428571429
  confusion_note: "Stored in workflow artifact and logs; N=5 is too small for scientific interpretation."
  artifact_id: 11583884716
  artifact_sha256: "7c2b8a434c650903c38e9d6f51d730eb6679df9007c3e12ab6bfa6d4638eb533"
  scientific_gate: false
decision: Implementation-only
action: "Retain pipeline; do not issue Go/No-Go. Scientific Pilot Gate 1 remains N_sim=500 per generating model at T=196 and T=480."
reason: "CI establishes executable end-to-end generate→fit→forward-score→confusion pipeline, not model identifiability."
```


---

## Entry 0007

```yaml
date: 2026-10-09
run_id: smc_particle_convergence_preregister
commit_sha: a72d2a1a663ece8dd18c87b8284ffadb4553e1b4
spec_version: 1.2
data:
  kind: synthetic-fixed-dataset-numerical-audit
  generating_model: [M3, M4, M5, M6]
  n_per_true: 8
  T: 196
metric: "Forward-prequential participant-choice LPD stability across SMC particle counts"
result:
  status: "criterion registered before convergence run"
  particle_grid: [16, 64, 256]
  fitter_seeds_per_cell: 3
  technical_acceptance_for_64:
    winner_agreement_64_vs_256_overall: ">= 0.90"
    winner_agreement_within_each_true_model: ">= 0.75"
    median_abs_delta_lpd_nats_per_scored_trial: "<= 0.01"
    p95_abs_delta_lpd_nats_per_scored_trial: "<= 0.05"
decision: Implementation-only
action: "If 64 passes, use 64 particles for Pilot Gate 1. If it fails, do not infer science; increase numerical accuracy or revise the approximation implementation only."
reason: "The N=5 particle sensitivity confusion matrices were materially unstable and are too noisy to choose particle count. This audit compares identical datasets and score estimates directly."
```


---

## Entry 0008

```yaml
date: 2026-10-09
run_id: particle_sensitivity_n5
commit_sha: 209b4e3e8ca1447bc668c500513f30e6ca5ad376
spec_version: 1.2
data:
  kind: synthetic-numerical-preflight
  generating_model: M0-M6
  n_sim: 5
  T: 196
metric: "Forward-prequential participant-choice LPD; confusion used only as numerical warning"
result:
  particles_4_overall: 0.7428571428571429
  particles_16_overall: 0.6857142857142857
  particles_64_overall: 0.6285714285714286
  observation: "Critical-model classifications varied materially with particle count at N=5."
decision: Implementation-only
action: "Do not choose particle count from these noisy confusion matrices. Run the preregistered fixed-dataset 16/64/256 score-convergence audit in Entry 0007."
reason: "N=5 recovery rates are scientifically meaningless and insufficient for numerical-convergence selection."
```

---

## Entry 0009

```yaml
date: 2026-10-09
run_id: dyson_supplement_retrieval_audit
commit_sha: f340352691752b079603e076c244cc493b8cb83c
spec_version: 1.2
data:
  kind: external-public-supplement
  source: "Dyson et al. 2016, DOI 10.1038/srep20479, PMC4740902"
metric: null
result:
  first_attempt: "PMC page link resolved to HTML rather than XLS binary."
  second_attempt: "Legacy PMC OA API endpoint returned HTTP 404 in the execution environment."
  scientific_classification: "Official supplementary/summary material; not assumed trial-level raw."
decision: Implementation-only
action: "Keep Dyson as authoritative-source pointer/retrieval metadata only. Do not mirror a convenience copy and do not use it as trial-level model-validation data unless independently verified."
reason: "Mirroring is unnecessary for reproducibility and repeated retrieval workarounds would add no scientific value."
```
