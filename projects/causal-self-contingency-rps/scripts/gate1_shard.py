#!/usr/bin/env python3
"""One deterministic shard of scientific Pilot Gate 1.

This script does not decide Go/No-Go. It generates a disjoint slice of the
locked 500-dataset design for one true model and one T, fits all seven models,
and writes auditable per-dataset score vectors.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from gate1_runner import MODEL_NAMES, forward_prequential, make_fit_model
from causal_rps.environment import simulate_runin, simulate_yoked_pair
from causal_rps.generators import make_generator
from causal_rps.schema import EnvironmentConfig

N_TOTAL_PER_TRUE_T=500
BATCH_SIZE=50
N_BATCHES=N_TOTAL_PER_TRUE_T//BATCH_SIZE


def generate_slice(true_model,T,batch_index,master_seed,context_kind):
    start=batch_index*BATCH_SIZE
    if start % 2:
        raise ValueError("batch start must preserve complete yoked pairs")
    pair_start=start//2
    n_pairs=BATCH_SIZE//2
    env=EnvironmentConfig()
    out=[]
    tm_idx=MODEL_NAMES.index(true_model)

    for local_pair in range(n_pairs):
        pair_global=pair_start+local_pair
        contingent="CE" if pair_global % 2 == 0 else "CB"
        seed=master_seed + tm_idx*10_000_000 + int(T)*10_000 + pair_global*100
        donor,replay=simulate_yoked_pair(
            make_generator(true_model,seed,context_kind),
            make_generator(true_model,seed+1,context_kind),
            contingent,T,seed,env,
            donor_runin=simulate_runin(seed+10),
            recipient_runin=simulate_runin(seed+20),
            model_name=true_model,
        )
        out.append((start+2*local_pair,donor))
        out.append((start+2*local_pair+1,replay))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--true-model",required=True,choices=MODEL_NAMES)
    ap.add_argument("--T",required=True,type=int,choices=[196,480])
    ap.add_argument("--batch-index",required=True,type=int,choices=range(N_BATCHES))
    ap.add_argument("--particles",required=True,type=int)
    ap.add_argument("--master-seed",type=int,default=20261009)
    ap.add_argument("--context",default="x3",choices=["x1","x2","x3","x4"])
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    datasets=generate_slice(args.true_model,args.T,args.batch_index,args.master_seed,args.context)
    rows=[]
    for global_index,data in datasets:
        scores={}
        for fi,fit_name in enumerate(MODEL_NAMES):
            seed=(
                args.master_seed
                + MODEL_NAMES.index(args.true_model)*100_000_000
                + args.T*100_000
                + global_index*100
                + fi
            )
            model=make_fit_model(fit_name,seed,args.particles,args.context)
            score,_=forward_prequential(model,data)
            scores[fit_name]=float(score)
        rows.append({
            "true_model":args.true_model,
            "T":args.T,
            "global_dataset_index":global_index,
            "condition":data.meta[0].condition,
            "predicted_model":max(scores,key=scores.get),
            "scores":scores,
        })

    payload={
        "spec_version":"1.2",
        "mode":"SCIENTIFIC_PILOT_GATE1_SHARD",
        "true_model":args.true_model,
        "T":args.T,
        "batch_index":args.batch_index,
        "batch_size":len(rows),
        "particles":args.particles,
        "master_seed":args.master_seed,
        "context":args.context,
        "rows":rows,
    }
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
