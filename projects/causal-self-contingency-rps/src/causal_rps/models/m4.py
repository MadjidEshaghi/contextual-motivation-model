from causal_rps.models.causal_base import CausalModelBase


class M4(CausalModelBase):
    """Bayesian causal-state model with two-step expected-value control."""

    name = "M4"
    epistemic_weight = 0.0
