# Causal Self-Contingency in Repeated Strategic Choice

Reproducibility workspace for the preregistered causal self-contingency / self-exploitability Rock–Paper–Scissors project.

## Scientific status

Design architecture is locked by **Final Closure Document v3.0**. The computational implementation contract is `simulator_spec.md` v1.2. Any deviation must be recorded in `locked_log.md` before human data collection.

The three independent claims are:

1. **Behavioral:** (	au>0) for the preregistered self-contingency × response-rule interaction.
2. **Computational:** (M_4) predicts held-out choices better than (M_3).
3. **Active inference:** (M_6) adds predictive value beyond both (M_4) and (M_5).

Failure of a higher-level claim does not redefine lower-level claims after the fact.

## Repository layout

- `simulator_spec.md` — locked implementation contract.
- `locked_log.md` — immutable decision/audit log.
- `src/causal_rps/` — simulator, environment, model and scoring code.
- `scripts/gate1_runner.py` — Gate 1 orchestration.
- `data/data_manifest.csv` — authoritative external-data provenance.
- `scripts/fetch_*.{py,R}` — retrieval scripts; original repositories remain primary sources.
- `tests/` — leakage, scoring and model sanity tests.

## External data policy

GitHub is **not** treated as the primary source for external datasets. Every external dataset is cited by its original DOI/repository URL. Raw external bytes are mirrored here only when redistribution permission is clear and file size is appropriate. Otherwise this repository contains a manifest, checksum after local retrieval, and a retrieval script.

Primary external sources currently tracked:

- Wang, Xu & Zhou (2014), *Scientific Reports*, DOI: `10.1038/srep05830`. Public 72-participant trial-level subset distributed in CRAN package `stratEst` as `WXZ2014`.
- Dyson et al. (2016), *Scientific Reports*, DOI: `10.1038/srep20479`. Official supplementary XLS is linked from the article/PMC record.
- Moerel et al., *Social Cognitive and Affective Neuroscience*, DOI: `10.1093/scan/nsaf101`; canonical EEG dataset OpenNeuro `ds006761` v1.0.0, DOI: `10.18112/openneuro.ds006761.v1.0.0`.

## Gate 1

Primary model-comparison metric: **forward prequential log predictive density of participant actions only**,

[
sum_t log P(a_tmid D_{<t}, M_k).
]

Gate 1 uses 500 synthetic datasets per generating model for `T in {196, 480}`. The actual-design audit at `T=196` cannot be rescued by performance at `T=480`.

## License

Project code follows the parent repository MIT license. External data retain their original licenses and attribution requirements.
