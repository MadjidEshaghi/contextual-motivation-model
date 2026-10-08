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
