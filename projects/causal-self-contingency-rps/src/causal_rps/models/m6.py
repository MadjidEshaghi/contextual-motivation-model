from causal_rps.models.causal_base import CausalModelBase


class M6(CausalModelBase):
    """Causal-state active-inference model with epistemic value over Theta."""

    name = "M6"
    epistemic_weight = 1.0
