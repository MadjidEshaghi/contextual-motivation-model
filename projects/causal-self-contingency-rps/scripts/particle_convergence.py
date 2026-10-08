#!/usr/bin/env python3
"""Numerical SMC particle-convergence audit.

This is an implementation audit, not a scientific model-recovery Gate.
It reuses exactly the same fixed synthetic datasets across particle counts
and fitter seeds, then compares forward-prequential log-score stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from gate1_runner import forward_prequential, generate_datasets, make_fit_model
from causal_rps.schema import EnvironmentConfig

TRUE_MODELS = ("M3", "M4", "M5", "M6")
FIT_MODELS = ("M3", "M4", "M5", "M6")


def scored_trials(data):
    return sum(1 for t, o in enumerate(data.observations) if t >= 10 and o.score_choice)


def run_audit(n_per_true=8, T=196, particle_grid=(16,64,256), fit_seeds=3,
              master_seed=314159, context_kind="x3"):
    env = EnvironmentConfig()
    datasets = []
    for i, true_name in enumerate(TRUE_MODELS):
        ds = generate_datasets(
            true_name, T, n_per_true, master_seed + 10000*i, env, context_kind
        )
        for j, data in enumerate(ds):
            datasets.append((true_name, j, data))

    rows = []
    for true_name, dataset_index, data in datasets:
        nscore = scored_trials(data)
        for particles in particle_grid:
            for fit_name in FIT_MODELS:
                vals = []
                for rep in range(fit_seeds):
                    seed = (
                        master_seed
                        + 1_000_000 * TRUE_MODELS.index(true_name)
                        + 10_000 * dataset_index
                        + 100 * particles
                        + 10 * FIT_MODELS.index(fit_name)
                        + rep
                    )
                    model = make_fit_model(
                        fit_name, seed=seed, particles=particles, context_kind=context_kind
                    )
                    score, _ = forward_prequential(model, data)
                    vals.append(float(score))
                rows.append({
                    "true_model": true_name,
                    "dataset_index": dataset_index,
                    "fit_model": fit_name,
                    "particles": particles,
                    "scores": vals,
                    "mean_score": float(np.mean(vals)),
                    "sd_score": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                    "n_scored_trials": nscore,
                })

    # Indexed means.
    idx = {
        (r["true_model"], r["dataset_index"], r["fit_model"], r["particles"]): r
        for r in rows
    }

    def winners(particles):
        out={}
        for true_name, dataset_index, data in datasets:
            scores={
                m: idx[(true_name,dataset_index,m,particles)]["mean_score"]
                for m in FIT_MODELS
            }
            out[(true_name,dataset_index)] = max(scores, key=scores.get)
        return out

    w64=winners(64)
    w256=winners(256)
    overall_agreement=float(np.mean([w64[k]==w256[k] for k in w64]))
    agreement_by_true={}
    for tm in TRUE_MODELS:
        ks=[k for k in w64 if k[0]==tm]
        agreement_by_true[tm]=float(np.mean([w64[k]==w256[k] for k in ks]))

    abs_diffs=[]
    seed_sd64=[]
    seed_sd256=[]
    for true_name, dataset_index, data in datasets:
        nscore=scored_trials(data)
        for fit_name in FIT_MODELS:
            r64=idx[(true_name,dataset_index,fit_name,64)]
            r256=idx[(true_name,dataset_index,fit_name,256)]
            abs_diffs.append(abs(r64["mean_score"]-r256["mean_score"])/nscore)
            seed_sd64.append(r64["sd_score"]/nscore)
            seed_sd256.append(r256["sd_score"]/nscore)

    summary={
        "mode":"IMPLEMENTATION_ONLY",
        "T":T,
        "n_per_true":n_per_true,
        "true_models":TRUE_MODELS,
        "fit_models":FIT_MODELS,
        "particle_grid":particle_grid,
        "fit_seeds":fit_seeds,
        "winner_agreement_64_vs_256":overall_agreement,
        "winner_agreement_by_true_64_vs_256":agreement_by_true,
        "median_abs_score_diff_nats_per_trial_64_vs_256":float(np.median(abs_diffs)),
        "p95_abs_score_diff_nats_per_trial_64_vs_256":float(np.quantile(abs_diffs,0.95)),
        "median_seed_sd_nats_per_trial_64":float(np.median(seed_sd64)),
        "median_seed_sd_nats_per_trial_256":float(np.median(seed_sd256)),
        "technical_acceptance_64": bool(
            overall_agreement >= 0.90
            and min(agreement_by_true.values()) >= 0.75
            and np.median(abs_diffs) <= 0.01
            and np.quantile(abs_diffs,0.95) <= 0.05
        ),
        "criteria_note": (
            "Numerical-convergence engineering criterion only: 64-vs-256 winner "
            "agreement >=0.90 overall, >=0.75 within each generating model, median "
            "|delta LPD| <=0.01 nats/scored trial, and 95th percentile <=0.05. "
            "This does not alter scientific Gate thresholds."
        ),
    }
    return {"summary":summary,"rows":rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-per-true",type=int,default=8)
    ap.add_argument("--T",type=int,default=196)
    ap.add_argument("--fit-seeds",type=int,default=3)
    ap.add_argument("--output",default="results/gate1/particle_convergence.json")
    args=ap.parse_args()
    result=run_audit(args.n_per_true,args.T,(16,64,256),args.fit_seeds)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    main()
