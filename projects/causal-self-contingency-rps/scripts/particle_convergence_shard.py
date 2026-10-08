#!/usr/bin/env python3
"""One fixed-dataset SMC particle-convergence shard."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from gate1_runner import forward_prequential, generate_datasets, make_fit_model
from causal_rps.schema import EnvironmentConfig

TRUE_MODELS=("M3","M4","M5","M6")
FIT_MODELS=("M3","M4","M5","M6")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--true-model",required=True,choices=TRUE_MODELS)
    ap.add_argument("--particles",required=True,type=int,choices=[16,64,256])
    ap.add_argument("--n-per-true",type=int,default=8)
    ap.add_argument("--T",type=int,default=196)
    ap.add_argument("--fit-seeds",type=int,default=3)
    ap.add_argument("--master-seed",type=int,default=314159)
    ap.add_argument("--context",default="x3")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    ti=TRUE_MODELS.index(args.true_model)
    data=generate_datasets(
        args.true_model,args.T,args.n_per_true,
        args.master_seed+10000*ti,EnvironmentConfig(),args.context
    )
    rows=[]
    for di,d in enumerate(data):
        nscore=sum(1 for t,o in enumerate(d.observations) if t>=10 and o.score_choice)
        for fit_name in FIT_MODELS:
            vals=[]
            for rep in range(args.fit_seeds):
                seed=(
                    args.master_seed
                    + 1_000_000*ti
                    + 10_000*di
                    + 100*args.particles
                    + 10*FIT_MODELS.index(fit_name)
                    + rep
                )
                m=make_fit_model(fit_name,seed,args.particles,args.context)
                score,_=forward_prequential(m,d)
                vals.append(float(score))
            rows.append({
                "true_model":args.true_model,
                "dataset_index":di,
                "fit_model":fit_name,
                "particles":args.particles,
                "scores":vals,
                "mean_score":float(np.mean(vals)),
                "sd_score":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,
                "n_scored_trials":nscore,
            })
    out={
        "mode":"IMPLEMENTATION_ONLY_PARTICLE_CONVERGENCE_SHARD",
        "true_model":args.true_model,
        "particles":args.particles,
        "n_per_true":args.n_per_true,
        "T":args.T,
        "fit_seeds":args.fit_seeds,
        "master_seed":args.master_seed,
        "rows":rows,
    }
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))

if __name__=="__main__":
    main()
