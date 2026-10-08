from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelObservation:
    t: int
    a: int
    b: int
    r: int
    forced_action: int | None = None
    score_choice: bool = True


@dataclass(frozen=True)
class TrialMeta:
    condition: str
    block_id: int
    pair_id: int | None
    probe_flag: bool
    probe_phase: str | None
    generating_model: str | None
    params_true: dict[str, Any] | None
    seed: int


@dataclass(frozen=True)
class RunInSummary:
    adherence: float
    chi_0: float
    exploitability_0: float


@dataclass
class RunInData:
    observations: list[ModelObservation]
    summary: RunInSummary


@dataclass
class Dataset:
    observations: list[ModelObservation]
    meta: list[TrialMeta]
    runin: RunInData
    T: int


@dataclass(frozen=True)
class EnvironmentConfig:
    beta_env: float = 3.0
    lambda_env: float = 0.8
    alpha_env: float = 1.0
    epsilon_b: float = 0.05


@dataclass
class FitResult:
    model_name: str
    dataset_id: str
    cumulative_lp: float
    log_predictive: list[float]
    params_posterior: dict[str, Any]
    converged: bool
    diagnostics: dict[str, Any]
    runtime_sec: float
