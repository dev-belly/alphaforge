"""Performance attribution: where the return came from (sectors) and what risk drove it (factors)."""

from alphaforge.attribution.brinson import BrinsonResult, brinson_attribution
from alphaforge.attribution.factor import FactorAttributionResult, factor_attribution

__all__ = [
    "BrinsonResult",
    "brinson_attribution",
    "FactorAttributionResult",
    "factor_attribution",
]
