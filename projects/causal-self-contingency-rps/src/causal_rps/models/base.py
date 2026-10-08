from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from causal_rps.schema import ModelObservation, RunInData


class OnlineModel(ABC):
    name: str

    @abstractmethod
    def initialize(self, priors: dict[str, Any], runin_data: RunInData) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict_log_prob(self, state: Any, next_a: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: Any, obs: ModelObservation) -> Any:
        raise NotImplementedError
