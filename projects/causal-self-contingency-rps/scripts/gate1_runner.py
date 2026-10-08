#!/usr/bin/env python3
"""Gate 1 orchestration scaffold.

The full Gate is disabled until M3-M6 have passed their reference and leakage
tests. No placeholder confusion matrix is ever emitted as a scientific result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from causal_rps.constants import W_WARM
from causal_rps.models import M0, M1, M2
from causal_rps.schema import Dataset


MODEL_REGISTRY = {
    "M0": M0,
    "M1": M1,
    "M2": M2,
}


def forward_prequential(model, data: Dataset, priors=None):
    priors = priors or {}
    state = model.initialize(priors, data.runin)
    lps: list[float] = []
    scored_t: list[int] = []

    for t, obs in enumerate(data.observations):
        # Locked rule: predict before update.
        lp = model.predict_log_prob(state, obs.a)
        if t >= W_WARM and obs.score_choice:
            if not math.isfinite(lp):
                raise FloatingPointError(f"Non-finite log predictive density at t={t}")
            lps.append(float(lp))
            scored_t.append(t)
        state = model.update(state, obs)

    return {
        "cumulative_lp": float(np.sum(lps)),
        "log_predictive": lps,
        "scored_t": scored_t,
    }


def future_permutation_leakage_test(model_factory, data: Dataset, cut: int = 20):
    """Predictions through cut must not depend on observations after cut."""
    if data.T <= cut + 2:
        raise ValueError("dataset too short for leakage test")

    model_a = model_factory()
    state_a = model_a.initialize({}, data.runin)
    pred_a = []
    for t in range(cut + 1):
        pred_a.append([model_a.predict_log_prob(state_a, a) for a in range(3)])
        state_a = model_a.update(state_a, data.observations[t])

    rng = np.random.default_rng(12345)
    prefix = list(data.observations[: cut + 1])
    future = list(data.observations[cut + 1 :])
    rng.shuffle(future)
    permuted = prefix + future

    model_b = model_factory()
    state_b = model_b.initialize({}, data.runin)
    pred_b = []
    for t in range(cut + 1):
        pred_b.append([model_b.predict_log_prob(state_b, a) for a in range(3)])
        state_b = model_b.update(state_b, permuted[t])

    if not np.allclose(pred_a, pred_b, atol=1e-12, rtol=1e-12):
        raise AssertionError("future-permutation leakage test failed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", default="results/gate1/preflight.json")
    args = parser.parse_args()

    out = {
        "status": "pipeline_initialized",
        "registered_models": sorted(MODEL_REGISTRY),
        "full_gate_enabled": False,
        "reason": "M3-M6 not yet registered; no fake Gate result is produced.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
