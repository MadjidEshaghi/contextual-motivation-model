#!/usr/bin/env python3
"""Aggregate 12 convergence shards using the preregistered numerical criterion."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

TRUE_MODELS=("M3","M4","M5","M6")
FIT_MODELS=("M3","M4","M5","M6")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    payloads=[]
    for p in Path(args.input_dir).rglob("*.json"):
        x=json.loads(p.read_text())
        if x.get("mode")=="IMPLEMENTATION_ONLY_PARTICLE_CONVERGENCE_SHARD":
            payloads.append(x)
    if len(payloads)!=12:
        raise RuntimeError(f"Expected 12 shards, found {len(payloads)}")
    rows=[r for x in payloads for r in x["rows"]]
    idx={(r["true_model"],r["dataset_index"],r["fit_model"],r["particles"]):r for r in rows}

    def winners(particles):
        out={}
        for tm in TRUE_MODELS:
            for di in range(8):
                scores={fm:idx[(tm,di,fm,particles)]["mean_score"] for fm in FIT_MODELS}
                out[(tm,di)]=max(scores,key=scores.get)
        return out
    w64,w256=winners(64),winners(256)
    overall=float(np.mean([w64[k]==w256[k] for k in w64]))
    by={}
    for tm in TRUE_MODELS:
        ks=[k for k in w64 if k[0]==tm]
        by[tm]=float(np.mean([w64[k]==w256[k] for k in ks]))

    diffs=[]; sd64=[]; sd256=[]
    for tm in TRUE_MODELS:
        for di in range(8):
            for fm in FIT_MODELS:
                a=idx[(tm,di,fm,64)]; b=idx[(tm,di,fm,256)]
                n=a["n_scored_trials"]
                diffs.append(abs(a["mean_score"]-b["mean_score"])/n)
                sd64.append(a["sd_score"]/n); sd256.append(b["sd_score"]/n)

    summary={
        "mode":"IMPLEMENTATION_ONLY",
        "winner_agreement_64_vs_256":overall,
        "winner_agreement_by_true_64_vs_256":by,
        "median_abs_score_diff_nats_per_trial_64_vs_256":float(np.median(diffs)),
        "p95_abs_score_diff_nats_per_trial_64_vs_256":float(np.quantile(diffs,.95)),
        "median_seed_sd_nats_per_trial_64":float(np.median(sd64)),
        "median_seed_sd_nats_per_trial_256":float(np.median(sd256)),
    }
    summary["technical_acceptance_64"]=bool(
        overall>=.90 and min(by.values())>=.75
        and summary["median_abs_score_diff_nats_per_trial_64_vs_256"]<=.01
        and summary["p95_abs_score_diff_nats_per_trial_64_vs_256"]<=.05
    )
    summary["criteria_note"]="Pre-registered in locked_log Entry 0007; numerical convergence only."
    out={"summary":summary,"rows":rows}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
