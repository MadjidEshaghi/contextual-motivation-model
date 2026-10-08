#!/usr/bin/env python3
"""Gate 1 forward-prequential runner.

`--preflight` is implementation/runtime validation only and never issues a
scientific Go/No-Go decision. Full Gate 1 requires the locked N=500 design.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from causal_rps.constants import (
    N_SIM_GATE1,
    THRESHOLD_OVERALL_COND,
    THRESHOLD_OVERALL_GO,
    W_WARM,
)
from causal_rps.environment import simulate_runin, simulate_yoked_pair, simulate_single
from causal_rps.generators import make_generator
from causal_rps.models import M0, M1, M2, M3, M4, M5, M6
from causal_rps.schema import Dataset, EnvironmentConfig


MODEL_NAMES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


def make_fit_model(name: str, seed: int, particles: int, context_kind="x3"):
    if name == "M0":
        return M0()
    if name == "M1":
        return M1()
    if name == "M2":
        return M2(n_particles=particles, seed=seed)
    if name == "M3":
        return M3(n_particles=particles, seed=seed, context_kind=context_kind)
    if name == "M4":
        return M4(n_particles=particles, seed=seed, context_kind=context_kind)
    if name == "M5":
        return M5(n_particles=particles, seed=seed, context_kind=context_kind)
    if name == "M6":
        return M6(n_particles=particles, seed=seed, context_kind=context_kind)
    raise ValueError(name)


def forward_prequential(model, data: Dataset, priors=None):
    state = model.initialize(priors or {}, data.runin)
    score = 0.0
    lps: list[float] = []
    for t, obs in enumerate(data.observations):
        lp = model.predict_log_prob(state, obs.a)
        if t >= W_WARM and obs.score_choice:
            if not math.isfinite(lp):
                raise FloatingPointError(f"non-finite lp at t={t}")
            score += float(lp)
            lps.append(float(lp))
        state = model.update(state, obs)
    return score, lps


def future_permutation_leakage_test(model_factory, data: Dataset, cut=20):
    model_a = model_factory()
    state_a = model_a.initialize({}, data.runin)
    pa = []
    for t in range(cut + 1):
        pa.append([model_a.predict_log_prob(state_a, a) for a in range(3)])
        state_a = model_a.update(state_a, data.observations[t])

    prefix = list(data.observations[: cut + 1])
    future = list(data.observations[cut + 1 :])
    np.random.default_rng(12345).shuffle(future)
    permuted = prefix + future

    model_b = model_factory()
    state_b = model_b.initialize({}, data.runin)
    pb = []
    for t in range(cut + 1):
        pb.append([model_b.predict_log_prob(state_b, a) for a in range(3)])
        state_b = model_b.update(state_b, permuted[t])

    if not np.allclose(pa, pb, atol=1e-12, rtol=1e-12):
        raise AssertionError("future-permutation leakage test failed")
    return True


def generate_datasets(true_model, T, n_sim, master_seed, env_config, context_kind="x3"):
    out = []
    rng = np.random.default_rng(master_seed)
    pair_index = 0

    while len(out) + 1 < n_sim:
        contingent = "CE" if pair_index % 2 == 0 else "CB"
        seed = int(rng.integers(1, 2**31 - 1))
        donor, replay = simulate_yoked_pair(
            make_generator(true_model, seed, context_kind),
            make_generator(true_model, seed + 1, context_kind),
            contingent,
            T,
            seed,
            env_config,
            donor_runin=simulate_runin(seed + 100),
            recipient_runin=simulate_runin(seed + 200),
            model_name=true_model,
        )
        out.extend([donor, replay])
        pair_index += 1

    if len(out) < n_sim:
        seed = int(rng.integers(1, 2**31 - 1))
        out.append(
            simulate_single(
                make_generator(true_model, seed, context_kind),
                "CE",
                T,
                seed,
                env_config,
                runin=simulate_runin(seed + 100),
                model_name=true_model,
            )
        )
    return out[:n_sim]


def _score_one_dataset(task):
    true_name, data, dataset_index, particles, master_seed, context_kind = task
    scores = {}
    for fi, fit_name in enumerate(MODEL_NAMES):
        model = make_fit_model(
            fit_name,
            seed=master_seed
            + 1_000_000 * MODEL_NAMES.index(true_name)
            + 1000 * dataset_index
            + fi,
            particles=particles,
            context_kind=context_kind,
        )
        scores[fit_name] = forward_prequential(model, data)[0]
    return {
        "true_model": true_name,
        "predicted_model": max(scores, key=scores.get),
        "scores": scores,
        "dataset_index": dataset_index,
    }


def confusion_from_results(results):
    counts = {t: {p: 0 for p in MODEL_NAMES} for t in MODEL_NAMES}
    correct = 0
    for row in results:
        counts[row["true_model"]][row["predicted_model"]] += 1
        correct += int(row["true_model"] == row["predicted_model"])
    return counts, correct / max(len(results), 1)


def run_audit(n_sim, T, particles, master_seed, context_kind, workers=1):
    env_config = EnvironmentConfig()
    tasks = []
    for ti, true_name in enumerate(MODEL_NAMES):
        datasets = generate_datasets(
            true_name, T, n_sim, master_seed + 10000 * ti, env_config, context_kind
        )
        for di, data in enumerate(datasets):
            tasks.append((true_name, data, di, particles, master_seed, context_kind))

    start = time.perf_counter()
    if workers <= 1:
        results = [_score_one_dataset(t) for t in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_score_one_dataset, t) for t in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

    confusion, overall = confusion_from_results(results)
    return {
        "T": T,
        "n_sim_per_model": n_sim,
        "particles": particles,
        "workers": workers,
        "overall_recovery": overall,
        "confusion": confusion,
        "runtime_sec": time.perf_counter() - start,
        "scientific_gate": n_sim == N_SIM_GATE1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--n-sim", type=int, default=5)
    ap.add_argument("--T", type=int, default=196)
    ap.add_argument("--particles", type=int, default=24)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", default="x3", choices=["x1", "x2", "x3", "x4"])
    ap.add_argument("--output", default="results/gate1/preflight.json")
    args = ap.parse_args()

    if not args.preflight and args.n_sim != N_SIM_GATE1:
        raise SystemExit("Full Gate requires n_sim=500 per generating model.")

    result = run_audit(
        args.n_sim, args.T, args.particles, args.seed, args.context, args.workers
    )
    result["mode"] = "preflight" if args.preflight else "gate1"

    if args.preflight:
        result["decision"] = "IMPLEMENTATION_ONLY"
    else:
        overall = result["overall_recovery"]
        result["decision"] = (
            "GO"
            if overall >= THRESHOLD_OVERALL_GO
            else "CONDITIONAL_GO"
            if overall >= THRESHOLD_OVERALL_COND
            else "NO_GO"
        )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
