#!/usr/bin/env python3
"""Aggregate all 140 Gate-1 shard outputs and apply locked decision rules."""

from __future__ import annotations
import argparse, json
from pathlib import Path

MODELS=("M0","M1","M2","M3","M4","M5","M6")
CRITICAL=(("M3","M4"),("M4","M6"),("M5","M6"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    files=sorted(Path(args.input_dir).rglob("*.json"))
    shards=[json.loads(p.read_text()) for p in files if "rows" in json.loads(p.read_text())]
    rows=[r for s in shards for r in s["rows"]]

    expected=7*2*500
    if len(rows)!=expected:
        raise RuntimeError(f"Expected {expected} dataset rows, found {len(rows)}")

    results={}
    for T in (196,480):
        subset=[r for r in rows if r["T"]==T]
        confusion={t:{p:0 for p in MODELS} for t in MODELS}
        for r in subset:
            confusion[r["true_model"]][r["predicted_model"]]+=1
        correct=sum(confusion[m][m] for m in MODELS)
        overall=correct/len(subset)
        diag={m:confusion[m][m]/sum(confusion[m].values()) for m in MODELS}
        condition_counts={}
        for c in ("CE","NE","CB","NB"):
            condition_counts[c]=sum(r["condition"]==c for r in subset)
        results[str(T)]={
            "n":len(subset),
            "overall_recovery":overall,
            "diagonal_recovery":diag,
            "confusion":confusion,
            "condition_counts":condition_counts,
        }

    r196=results["196"]
    critical_pass={}
    for a,b in CRITICAL:
        critical_pass[f"{a}<->{b}"]={
            a:r196["diagonal_recovery"][a],
            b:r196["diagonal_recovery"][b],
            "pass":r196["diagonal_recovery"][a]>=0.75 and r196["diagonal_recovery"][b]>=0.75,
        }

    overall=r196["overall_recovery"]
    if overall>=0.80 and all(x["pass"] for x in critical_pass.values()):
        decision="GO"
    elif overall>=0.70:
        decision="CONDITIONAL_GO"
    else:
        decision="NO_GO"

    # Failure at actual design cannot be rescued by T=480.
    payload={
        "spec_version":"1.2",
        "mode":"SCIENTIFIC_PILOT_GATE1",
        "decision_basis":"T=196 actual-design audit; T=480 is stress/sensitivity only",
        "results":results,
        "critical_pairs_T196":critical_pass,
        "decision":decision,
        "locked_thresholds":{
            "overall_go":0.80,
            "overall_conditional":0.70,
            "critical_each_direction":0.75,
        },
        "n_shards":len(shards),
        "n_dataset_rows":len(rows),
    }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()
