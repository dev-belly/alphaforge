"""Factor abstraction: every signal is a (dates x symbols) panel plus metadata.

Conventions
-----------
* A factor panel is indexed by trading date (rows) and symbol (columns).
* ``direction`` records the *a-priori economic sign*: ``+1`` means a higher raw
  value is expected to earn a higher return, ``-1`` the opposite.  The
  preprocessor multiplies by ``direction`` so every stored factor is
  "higher is better", and that assumption is documented rather than implicit.
* ``category`` groups factors for the risk model and the dashboard.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.features.fundamentals import FundamentalView
from alphaforge.features.panel import MarketPanel
from alphaforge.utils.logging import get_logger

log = get_logger("factors.base")

CATEGORIES = ("momentum", "reversal", "value", "quality", "risk", "liquidity", "size")


@dataclass(frozen=True)
class FactorSpec:
    """Static description of a factor."""

    name: str
    category: str
    direction: int = 1
    description: str = ""
    requires_fundamentals: bool = False
    # Availability is honest: some factors cannot be computed from every vendor.
    data_requirement: str = "price"

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown category {self.category!r}; expected one of {CATEGORIES}")
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")


@dataclass
class Factor:
    """A computed factor: raw values, spec and optional diagnostics."""

    spec: FactorSpec
    raw: pd.DataFrame
    notes: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def category(self) -> str:
        return self.spec.category

    def coverage(self) -> float:
        if self.raw.empty:
            return 0.0
        return float(self.raw.notna().values.mean())


class FactorContext:
    """Everything a factor function is allowed to see.

    Passing a single context object (rather than five loose frames) keeps the
    factor signatures uniform and makes it obvious that a factor cannot reach
    for data outside its own date.
    """

    def __init__(self, panel: MarketPanel, fundamentals: FundamentalView | None = None) -> None:
        self.panel = panel
        self.fundamentals = fundamentals

    @property
    def close(self) -> pd.DataFrame:
        return self.panel.close

    @property
    def returns(self) -> pd.DataFrame:
        return self.panel.returns

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.panel.market_cap

    def require_fundamentals(self) -> FundamentalView:
        if self.fundamentals is None or self.fundamentals.data.empty:
            raise FactorUnavailableError("Fundamental data is not available for this dataset")
        return self.fundamentals


class FactorUnavailableError(RuntimeError):
    """Raised when a factor cannot be computed with the configured data source."""


FactorFn = Callable[[FactorContext], pd.DataFrame]


class FactorRegistry:
    """Registry of factor specifications and their compute functions."""

    def __init__(self) -> None:
        self._specs: dict[str, FactorSpec] = {}
        self._fns: dict[str, FactorFn] = {}

    def register(self, spec: FactorSpec, fn: FactorFn | None = None):
        def deco(f: FactorFn) -> FactorFn:
            self._specs[spec.name] = spec
            self._fns[spec.name] = f
            return f

        if fn is None:
            return deco
        self._specs[spec.name] = spec
        self._fns[spec.name] = fn
        return fn

    def spec(self, name: str) -> FactorSpec:
        return self._specs[name]

    def names(self, category: str | None = None) -> list[str]:
        if category is None:
            return sorted(self._specs)
        return sorted(n for n, s in self._specs.items() if s.category == category)

    def specs(self) -> list[FactorSpec]:
        return [self._specs[n] for n in sorted(self._specs)]

    def compute(self, name: str, ctx: FactorContext) -> Factor:
        if name not in self._fns:
            raise KeyError(f"Unknown factor {name!r}")
        try:
            raw = self._fns[name](ctx)
        except FactorUnavailableError as exc:
            log.warning(f"Factor {name} unavailable: {exc}")
            raw = None
        if raw is None or raw.empty:
            raw = _empty_like(ctx)
        raw = raw.reindex(index=ctx.panel.dates, columns=ctx.panel.symbols)
        return Factor(spec=self._specs[name], raw=raw)

    def compute_all(
        self, ctx: FactorContext, names: Iterable[str] | None = None
    ) -> dict[str, Factor]:
        target = list(names) if names is not None else self.names()
        out: dict[str, Factor] = {}
        for name in target:
            try:
                fac = self.compute(name, ctx)
            except Exception as exc:  # noqa: BLE001 - one bad factor must not kill a run
                log.error(f"Factor {name} failed: {exc}")
                continue
            if fac.coverage() > 0:
                out[name] = fac
            else:
                log.warning(f"Factor {name} produced no observations - skipped")
        log.info(f"Computed {len(out)}/{len(target)} factors")
        return out

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


def _empty_like(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=ctx.panel.dates, columns=ctx.panel.symbols)


REGISTRY = FactorRegistry()


def register(spec: FactorSpec):
    """Decorator registering a factor function against the global registry."""
    return REGISTRY.register(spec)


__all__ = [
    "FactorSpec",
    "Factor",
    "FactorContext",
    "FactorRegistry",
    "FactorUnavailableError",
    "REGISTRY",
    "register",
    "CATEGORIES",
]
