"""Orchestrates factor computation, preprocessing and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.factors import (  # noqa: F401  - registration side effects
    liquidity,
    momentum,
    quality,
    risk,
    value,
)
from alphaforge.factors.base import REGISTRY, Factor, FactorContext
from alphaforge.factors.evaluation import FactorResult, evaluate_factor, factor_correlation
from alphaforge.factors.preprocessing import FactorPreprocessor, ProcessingConfig
from alphaforge.utils.logging import Timer, get_logger

log = get_logger("factors.library")


@dataclass
class FactorLibrary:
    """End-to-end factor research facade: raw -> processed -> evaluated."""

    ctx: FactorContext
    processing: ProcessingConfig
    horizon: int = 21
    n_quantiles: int = 5

    raw: dict[str, Factor] = field(default_factory=dict, init=False)
    processed: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    results: dict[str, FactorResult] = field(default_factory=dict, init=False)

    @classmethod
    def from_config(cls, ctx: FactorContext, cfg: dict) -> FactorLibrary:
        fcfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        return cls(
            ctx=ctx,
            processing=ProcessingConfig.from_dict(fcfg),
            horizon=int(fcfg.get("horizon", 21)),
            n_quantiles=int(fcfg.get("n_quantiles", 5)),
        )

    # ------------------------------------------------------------------
    @property
    def universe(self) -> pd.DataFrame:
        return self.ctx.panel.universe

    def available(self) -> list[str]:
        names = []
        for name in REGISTRY.names():
            spec = REGISTRY.spec(name)
            if spec.requires_fundamentals and (
                self.ctx.fundamentals is None or self.ctx.fundamentals.data.empty
            ):
                continue
            names.append(name)
        return names

    def compute(self, names: list[str] | None = None) -> dict[str, Factor]:
        with Timer("factors.compute", log):
            self.raw = REGISTRY.compute_all(self.ctx, names or self.available())
        return self.raw

    def preprocess(self, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
        if not self.raw:
            self.compute(names)
        targets = names or list(self.raw)
        with Timer("factors.preprocess", log):
            prep = FactorPreprocessor(self.processing, self.ctx)
            self.processed = {n: prep.process(self.raw[n]) for n in targets if n in self.raw}
        log.info(f"Preprocessed {len(self.processed)} factors")
        return self.processed

    def evaluate(self, names: list[str] | None = None) -> dict[str, FactorResult]:
        if not self.processed:
            self.preprocess(names)
        targets = names or list(self.processed)
        self.results = {}
        with Timer("factors.evaluate", log):
            for name in targets:
                spec = REGISTRY.spec(name)
                self.results[name] = evaluate_factor(
                    name=name,
                    category=spec.category,
                    direction=spec.direction,
                    factor=self.processed[name],
                    close=self.ctx.close,
                    horizon=self.horizon,
                    n_quantiles=self.n_quantiles,
                    universe=self.universe,
                )
        return self.results

    # ------------------------------------------------------------------
    def run(self, names: list[str] | None = None) -> dict[str, FactorResult]:
        self.compute(names)
        self.preprocess()
        return self.evaluate()

    def correlation(self) -> pd.DataFrame:
        if not self.processed:
            self.preprocess()
        return factor_correlation(self.processed)

    def summary_table(self) -> pd.DataFrame:
        from alphaforge.factors.evaluation import rank_summary_table

        if not self.results:
            self.evaluate()
        return rank_summary_table(self.results)

    def composite(
        self, names: list[str] | None = None, weights: dict[str, float] | None = None
    ) -> pd.DataFrame:
        """Equal-weighted (or custom-weighted) z-score composite of processed factors."""
        if not self.processed:
            self.preprocess()
        targets = names or list(self.processed)
        if not targets:
            raise ValueError("No factors available to composite")
        w = {n: float(weights.get(n, 1.0)) if weights else 1.0 for n in targets}
        total = sum(abs(v) for v in w.values()) or 1.0
        acc: pd.Series | None = None
        for n in targets:
            part = self.processed[n] * (w[n] / total)
            acc = part if acc is None else acc.add(part, fill_value=0.0)
        assert acc is not None
        z = acc.sub(acc.mean(axis=1), axis=0).div(acc.std(axis=1).replace(0, np.nan), axis=0)
        return z.fillna(0.0)

    def specs(self) -> list[dict]:
        out = []
        for name in REGISTRY.names():
            spec = REGISTRY.spec(name)
            out.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "direction": spec.direction,
                    "description": spec.description,
                    "requires_fundamentals": spec.requires_fundamentals,
                    "data_requirement": spec.data_requirement,
                    "available": name in self.available(),
                }
            )
        return out


__all__ = ["FactorLibrary"]
