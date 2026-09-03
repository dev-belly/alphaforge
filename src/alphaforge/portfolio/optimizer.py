"""Portfolio construction: from alpha scores to constrained target weights.

Why this layer exists
---------------------
A ranked list of alpha scores is not a portfolio.  Between the two sit the
decisions that actually determine the shape of the P&L: how much risk to take,
how concentrated the book may become, how much turnover you are willing to pay
for, and how far the portfolio may drift from the benchmark's industry profile.
Those are constraints, not preferences, so they are solved for explicitly
rather than patched on afterwards.

Solvers
-------
``equal_weight``    closed form with an iterative cap (no optimiser needed)
``min_variance``    convex QP, no alpha required
``mean_variance``   convex QP: ``max mu'w - 0.5*lambda*w'Sigma*w - cost*||dw||_1``
``max_sharpe``      Charnes-Cooper transform of the fractional program
``risk_parity``     equal-risk-contribution, solved by SLSQP on the RC residual

Cardinality (``max_holdings``) is a combinatorial constraint and is therefore
handled as a pre-screen: the universe is truncated by |score| before the convex
solve, and the result is re-normalised.  This is a documented heuristic, not a
claim of optimality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alphaforge.utils.logging import get_logger

log = get_logger("portfolio.optimizer")

METHODS = ("equal_weight", "min_variance", "mean_variance", "max_sharpe", "risk_parity")

# cvxpy reports a hard infeasibility in a couple of spellings depending on how
# the solver terminates; both mean "relax something and try again".
INFEASIBLE_STATUSES = {"infeasible", "infeasible_inaccurate"}


@dataclass
class OptimizerConfig:
    """Every knob that shapes the target portfolio."""

    method: str = "mean_variance"
    long_only: bool = True
    fully_invested: bool = True
    max_weight: float = 0.05
    min_weight: float = 0.0
    target_volatility: float | None = 0.12
    turnover_limit: float | None = 0.20
    max_holdings: int | None = 50
    cash_buffer: float = 0.0
    max_industry_deviation: float | None = 0.03
    # Cost (in annualised return units) per unit of industry deviation beyond
    # ``max_industry_deviation``.  Large enough to be effectively binding.
    industry_penalty: float = 10.0
    risk_aversion: float = 5.0
    tc_penalty: float = 0.5
    # One-way trading cost in bps applied to traded notional inside the
    # objective (commission + slippage), scaled by ``tc_penalty``.
    cost_bps: float = 7.0
    dust_threshold: float = 1e-4
    min_names: int = 5

    @classmethod
    def from_dict(cls, cfg: dict | None) -> OptimizerConfig:
        cfg = cfg or {}
        tv = cfg.get("target_volatility")
        return cls(
            method=str(cfg.get("method", "mean_variance")),
            long_only=bool(cfg.get("long_only", True)),
            fully_invested=bool(cfg.get("fully_invested", True)),
            max_weight=float(cfg.get("max_weight", 0.05)),
            min_weight=float(cfg.get("min_weight", 0.0)),
            target_volatility=None if tv in (None, "null") else float(tv),
            turnover_limit=(
                None
                if cfg.get("turnover_limit") in (None, "null")
                else float(cfg["turnover_limit"])
            ),
            max_holdings=(
                None if cfg.get("max_holdings") in (None, "null") else int(cfg["max_holdings"])
            ),
            cash_buffer=float(cfg.get("cash_buffer", 0.0)),
            max_industry_deviation=(
                None
                if cfg.get("max_industry_deviation") in (None, "null")
                else float(cfg["max_industry_deviation"])
            ),
            risk_aversion=float(cfg.get("risk_aversion", 5.0)),
            tc_penalty=float(cfg.get("tc_penalty", 0.5)),
            industry_penalty=float(cfg.get("industry_penalty", 10.0)),
            min_names=int(cfg.get("min_names", 5)),
        )


@dataclass
class OptimizationResult:
    """Target weights plus everything needed to audit the solve."""

    weights: pd.Series
    method: str
    status: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_holdings(self) -> int:
        return int((self.weights.abs() > 1e-8).sum())

    def to_frame(self) -> pd.DataFrame:
        return self.weights.rename("weight").to_frame()


class PortfolioOptimizer:
    """Solves for target weights under the configured constraint set."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()
        if self.config.method not in METHODS:
            raise ValueError(f"Unknown portfolio method {self.config.method!r}; expected {METHODS}")

    # ------------------------------------------------------------------
    def solve(
        self,
        mu: pd.Series | None,
        cov: pd.DataFrame,
        prev_weights: pd.Series | None = None,
        industry: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        """Return target weights over the columns of ``cov``.

        Parameters
        ----------
        mu:
            Annualised expected returns. Required for every method except
            ``min_variance`` and ``risk_parity``.
        cov:
            Annualised covariance matrix (index and columns = asset ids).
        prev_weights:
            Current book. Enables the turnover constraint and the trading-cost
            term; without it the solve is a cold start.
        industry:
            ``asset -> sector`` map used for the industry-deviation constraint.
        benchmark_weights:
            Benchmark weights the industry constraint is measured against.
        """
        cfg = self.config
        cov = cov.dropna(how="all").dropna(axis=1, how="all")
        assets = list(cov.columns)

        if mu is not None:
            assets = [a for a in assets if a in mu.index]
        if len(assets) < cfg.min_names:
            raise ValueError(
                f"Only {len(assets)} eligible names (min_names={cfg.min_names}) - cannot optimise"
            )

        sigma = cov.reindex(index=assets, columns=assets).to_numpy(dtype=float)
        sigma = 0.5 * (sigma + sigma.T)
        # Numerical floor: a near-singular Sigma makes every QP solver unhappy
        # and the resulting weights meaningless.
        sigma += np.eye(len(assets)) * 1e-8

        mu_vec = (
            mu.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            if mu is not None
            else np.zeros(len(assets))
        )
        w0 = (
            self._feasible_reference(prev_weights.reindex(assets).fillna(0.0), len(assets))
            if prev_weights is not None
            else np.zeros(len(assets))
        )

        assets, sigma, mu_vec, w0 = self._prescreen(assets, sigma, mu_vec, w0)

        bounds = self._bounds(len(assets))
        industry_matrix, industry_target = self._industry_block(assets, industry, benchmark_weights)

        failed = False
        try:
            raw = self._dispatch(
                assets, sigma, mu_vec, w0, bounds, industry_matrix, industry_target
            )
            status = "optimal"
        except Exception as exc:  # noqa: BLE001 - degrade to the last known book
            log.warning(f"{cfg.method} solve failed ({exc}) - holding the current book")
            raw = w0 if w0.any() else np.ones(len(assets)) / len(assets)
            status = f"fallback:{type(exc).__name__}"
            failed = True

        weights = pd.Series(raw, index=assets, dtype=float)
        # A failed solve means "do not trade": returning the previous book
        # untouched is safer than re-applying caps the solver could not satisfy.
        if not failed:
            weights = self._post_process(weights, sigma, assets)
        else:
            weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        diag = self._diagnostics(weights, sigma, mu_vec, w0, assets)
        log.debug(
            f"{cfg.method}: {self.n_holdings(weights)} holdings, ex-ante vol {diag['ex_ante_vol']:.3f}"
        )
        return OptimizationResult(
            weights=weights, method=cfg.method, status=status, diagnostics=diag
        )

    # ------------------------------------------------------------------
    def _dispatch(
        self,
        assets: list[str],
        sigma: np.ndarray,
        mu_vec: np.ndarray,
        w0: np.ndarray,
        bounds: tuple[float, float],
        industry_matrix: np.ndarray | None,
        industry_target: np.ndarray | None,
    ) -> np.ndarray:
        method = self.config.method
        if method == "equal_weight":
            return self._equal_weight(len(assets), mu_vec)
        if method == "min_variance":
            return self._qp(
                assets,
                sigma,
                np.zeros_like(mu_vec),
                w0,
                bounds,
                industry_matrix,
                industry_target,
                risk_aversion=1.0,
            )
        if method == "mean_variance":
            return self._qp(
                assets,
                sigma,
                mu_vec,
                w0,
                bounds,
                industry_matrix,
                industry_target,
                risk_aversion=self.config.risk_aversion,
            )
        if method == "max_sharpe":
            return self._max_sharpe(
                assets, sigma, mu_vec, w0, bounds, industry_matrix, industry_target
            )
        return self._risk_parity(sigma, bounds)

    # -- individual methods ----------------------------------------------
    def _equal_weight(self, n: int, mu_vec: np.ndarray) -> np.ndarray:
        w = np.ones(n) / n
        return self._cap_and_normalise(w)

    def _solve_relaxing_vol(self, build) -> np.ndarray:
        """Solve, relaxing constraints that make the QP infeasible.

        The relaxation ladder, hardest to softest:

        1. volatility ceiling + budget equality + turnover limit;
        2. drop the volatility ceiling (it is re-applied exactly in
           ``_post_process``, since volatility is homogeneous of degree one);
        3. drop the budget equality, leaving ``sum(w) <= budget`` - the only way
           to satisfy a turnover limit when the current book is itself
           under-invested, which happens whenever the previous solve de-levered.

        A solve that still fails after step 3 is genuinely contradictory and is
        reported as such; the caller then holds the current book.
        """
        cfg = self.config
        ladder = [(True, True), (False, True), (False, False)]
        last = "unknown"
        for enforce_vol, eq_budget in ladder:
            prob, var = build(enforce_vol, eq_budget)
            prob.solve(solver=cp.CLARABEL, warm_start=False)
            last = str(prob.status)
            if var.value is None or not np.all(np.isfinite(var.value)):
                if prob.status in INFEASIBLE_STATUSES:
                    continue
                raise RuntimeError(f"solver returned {prob.status}")
            relaxed = [] if (enforce_vol and cfg.target_volatility) else ["volatility"]
            if not eq_budget:
                relaxed.append("budget")
            if relaxed:
                log.debug(f"{cfg.method}: relaxed {relaxed} to obtain a feasible book")
            return np.asarray(var.value, dtype=float).ravel()
        raise RuntimeError(f"infeasible (last status {last})")

    def _qp(
        self,
        assets: list[str],
        sigma: np.ndarray,
        mu_vec: np.ndarray,
        w0: np.ndarray,
        bounds: tuple[float, float],
        industry_matrix: np.ndarray | None,
        industry_target: np.ndarray | None,
        risk_aversion: float,
    ) -> np.ndarray:
        cfg = self.config
        n = len(assets)
        lo, hi = bounds

        def build(enforce_vol: bool, eq_budget: bool):
            w = cp.Variable(n)
            constraints = [w >= lo, w <= hi]

            if cfg.fully_invested and eq_budget:
                constraints.append(cp.sum(w) == 1.0 - cfg.cash_buffer)
            else:
                constraints.append(cp.sum(w) <= 1.0 - cfg.cash_buffer)

            if cfg.turnover_limit is not None and np.any(w0):
                constraints.append(cp.sum(cp.abs(w - w0)) <= cfg.turnover_limit)

            if enforce_vol and cfg.target_volatility:
                constraints.append(cp.quad_form(w, cp.psd_wrap(sigma)) <= cfg.target_volatility**2)

            # Industry exposure is enforced through a penalised slack rather
            # than a hard cap: with a position cap and a turnover limit in play,
            # a hard industry constraint is frequently infeasible for reasons
            # that have nothing to do with risk (an industry with a single
            # eligible name cannot be brought to a uniform target).  The slack
            # keeps the problem solvable and charges for the excess.
            industry_penalty = 0.0
            if industry_matrix is not None:
                k = industry_matrix.shape[0]
                slack = cp.Variable(k, nonneg=True)
                deviation = industry_matrix @ w - industry_target
                constraints += [
                    deviation <= cfg.max_industry_deviation + slack,
                    -deviation <= cfg.max_industry_deviation + slack,
                ]
                industry_penalty = cfg.industry_penalty * cp.sum(slack)

            tc = (cfg.tc_penalty * cfg.cost_bps / 10_000.0) * cp.sum(cp.abs(w - w0))
            objective = cp.Maximize(
                mu_vec @ w
                - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
                - tc
                - industry_penalty
            )
            return cp.Problem(objective, constraints), w

        return self._solve_relaxing_vol(build)

    def _max_sharpe(
        self,
        assets: list[str],
        sigma: np.ndarray,
        mu_vec: np.ndarray,
        w0: np.ndarray,
        bounds: tuple[float, float],
        industry_matrix: np.ndarray | None,
        industry_target: np.ndarray | None,
    ) -> np.ndarray:
        """Charnes-Cooper transform: maximise ``mu'y`` s.t. ``y'Sigma y <= 1``.

        Valid for a zero risk-free rate and a long-only book: the recovered
        weights are ``w = y / sum(y)``, which restores the budget constraint the
        transform drops.
        """
        cfg = self.config
        n = len(assets)
        lo, hi = bounds

        def build(use_industry: bool):
            y = cp.Variable(n)
            constraints = [y >= lo, cp.quad_form(y, cp.psd_wrap(sigma)) <= 1.0]
            if use_industry and industry_matrix is not None:
                # The deviation constraint is not scale invariant, so the budget
                # is reintroduced here and the transform degrades to a
                # Sharpe-tilted program - still correct, just not the pure
                # fractional one.
                constraints += [
                    cp.sum(y) == 1.0 - cfg.cash_buffer,
                    cp.abs(industry_matrix @ y - industry_target) <= cfg.max_industry_deviation,
                ]
            if cfg.turnover_limit is not None and np.any(w0):
                constraints.append(cp.sum(cp.abs(y - w0)) <= cfg.turnover_limit)
            return cp.Problem(cp.Maximize(mu_vec @ y), constraints), y

        attempts = [True, False] if industry_matrix is not None else [False]
        last = "unknown"
        for use_industry in attempts:
            prob, y = build(use_industry)
            prob.solve(solver=cp.CLARABEL)
            last = str(prob.status)
            if y.value is None or not np.all(np.isfinite(y.value)):
                if prob.status in INFEASIBLE_STATUSES:
                    continue
                raise RuntimeError(f"solver returned {prob.status}")
            yv = np.asarray(y.value, dtype=float).ravel()
            total = yv.sum()
            if abs(total) < 1e-8:
                raise RuntimeError("degenerate max-sharpe solution (zero notional)")
            return self._cap_and_normalise(yv / total)
        raise RuntimeError(f"infeasible (last status {last})")

    def _risk_parity(self, sigma: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        """Equal risk contribution: minimise the dispersion of ``w_i (Σw)_i``."""
        n = sigma.shape[0]
        lo, hi = bounds
        lo = max(lo, 0.0)

        def objective(w: np.ndarray) -> float:
            w = np.clip(w, lo, None)
            var = float(w @ sigma @ w)
            if var <= 1e-16:
                return 0.0
            rc = w * (sigma @ w)
            target = var / n
            return float(np.sum((rc - target) ** 2) / (target**2 + 1e-18))

        x0 = np.full(n, max(1.0 / n, lo))
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(lo, hi)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 400, "ftol": 1e-12},
        )
        w = np.clip(res.x, lo, None)
        total = w.sum()
        return w / total if total > 1e-12 else np.ones(n) / n

    # -- helpers ----------------------------------------------------------
    def _prescreen(
        self, assets: list[str], sigma: np.ndarray, mu_vec: np.ndarray, w0: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """Truncate the universe by conviction so the QP stays tractable.

        Cardinality is combinatorial; ranking by |alpha| and keeping a generous
        multiple of ``max_holdings`` is the standard industry heuristic.
        """
        cfg = self.config
        if not cfg.max_holdings or len(assets) <= cfg.max_holdings:
            return assets, sigma, mu_vec, w0
        keep_n = min(len(assets), max(cfg.max_holdings * 3, cfg.max_holdings + 30))
        # Keep names we already own (selling only to satisfy a hard cap is the
        # expensive way to reduce turnover), then fill by conviction.
        held = np.where(np.abs(w0) > 1e-8)[0]
        order = np.argsort(-np.abs(mu_vec))
        selected = list(held[: cfg.max_holdings])
        for idx in order:
            if len(selected) >= keep_n:
                break
            if idx not in selected:
                selected.append(int(idx))
        selected = sorted(set(int(i) for i in selected))[:keep_n]
        if len(selected) < cfg.min_names:
            return assets, sigma, mu_vec, w0
        sel = np.asarray(selected, dtype=int)
        return (
            [assets[i] for i in sel],
            sigma[np.ix_(sel, sel)],
            mu_vec[sel],
            w0[sel],
        )

    def _bounds(self, n: int) -> tuple[float, float]:
        cfg = self.config
        lo = 0.0 if cfg.long_only else -cfg.max_weight
        hi = float(cfg.max_weight)
        if hi <= lo:
            hi = max(abs(lo) + 1e-6, 1e-6)
        return (lo, hi)

    def _industry_block(
        self,
        assets: list[str],
        industry: pd.Series | None,
        benchmark_weights: pd.Series | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        cfg = self.config
        if industry is None or cfg.max_industry_deviation is None:
            return None, None
        ind = industry.reindex(assets).astype("object")
        if ind.isna().all():
            return None, None
        ind = ind.fillna("UNKNOWN")
        labels = sorted(set(ind.astype(str)))
        matrix = np.zeros((len(labels), len(assets)))
        for k, lab in enumerate(labels):
            matrix[k] = (ind.astype(str) == lab).to_numpy(dtype=float)

        if benchmark_weights is not None:
            bench = benchmark_weights.reindex(assets).fillna(0.0)
            target = np.array(
                [float(bench[(ind.astype(str) == lab).to_numpy()].sum()) for lab in labels]
            )
        else:
            # Neutral reference = the industry composition of an equal-weighted
            # portfolio over the eligible names.  A uniform 1/K target looks
            # tidier but is *infeasible* whenever an industry cannot supply
            # enough names to reach it under the position cap.
            target = matrix.mean(axis=1)
        return matrix, target

    def _cap_and_normalise(self, w: np.ndarray) -> np.ndarray:
        cfg = self.config
        w = np.where(np.isfinite(w), w, 0.0)
        if cfg.long_only:
            w = np.clip(w, 0.0, None)
        budget = 1.0 - cfg.cash_buffer
        hi = float(cfg.max_weight)
        for _ in range(64):
            total = w.sum()
            if total <= 1e-12:
                w = np.ones_like(w) / len(w)
                continue
            w = w * (budget / total)
            over = w > hi + 1e-12
            if not over.any():
                break
            excess = float((w[over] - hi).sum())
            w[over] = hi
            free = ~over
            if free.sum() == 0 or excess <= 0:
                break
            w[free] += excess * (w[free] / max(w[free].sum(), 1e-12))
        return w

    def _feasible_reference(self, prev: pd.Series, n: int) -> np.ndarray:
        """Clip the current book onto the feasible box before using it.

        Turnover is measured from the *feasible* reference, not from a book that
        itself violates the position limits - otherwise the turnover constraint
        and the box are mutually unsatisfiable and every solve comes back
        infeasible for a reason that has nothing to do with the market.
        """
        w = np.asarray(prev.to_numpy(dtype=float), dtype=float).copy()
        w = np.where(np.isfinite(w), w, 0.0)
        return self._cap_and_normalise(w)

    def _post_process(self, weights: pd.Series, sigma: np.ndarray, assets: list[str]) -> pd.Series:
        cfg = self.config
        w = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if cfg.long_only:
            w = w.clip(lower=0.0)
        # Dust removal: a 3bp position costs as much to trade as it contributes.
        w[w.abs() < cfg.dust_threshold] = 0.0
        if cfg.max_holdings and (w.abs() > 0).sum() > cfg.max_holdings:
            keep = w.abs().sort_values(ascending=False).index[: cfg.max_holdings]
            w = w.where(w.index.isin(keep), 0.0)
        budget = 1.0 - cfg.cash_buffer
        total = float(w.sum())
        if cfg.fully_invested and total > 1e-12:
            w = w * (budget / total)
            w = pd.Series(self._cap_and_normalise(w.to_numpy(dtype=float)), index=w.index)

        # The volatility budget is applied *last*, because both dust removal and
        # cardinality trimming concentrate the book and can push it back over
        # the risk the QP was asked to respect.  De-levering into cash is the
        # only action that cannot manufacture risk.
        if cfg.target_volatility:
            v = w.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            vol = float(np.sqrt(max(v @ sigma @ v, 0.0)))
            if vol > cfg.target_volatility:
                w = w * (cfg.target_volatility / vol)
        return w

    @staticmethod
    def n_holdings(weights: pd.Series) -> int:
        return int((weights.abs() > 1e-8).sum())

    def _diagnostics(
        self,
        weights: pd.Series,
        sigma: np.ndarray,
        mu_vec: np.ndarray,
        w0: np.ndarray,
        assets: list[str],
    ) -> dict:
        w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        var = float(w @ sigma @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        exp_ret = float(mu_vec @ w)
        rc = w * (sigma @ w)
        return {
            "method": self.config.method,
            "n_assets_universe": len(assets),
            "n_holdings": int((np.abs(w) > 1e-8).sum()),
            "max_weight": float(np.max(np.abs(w))) if w.size else 0.0,
            "top10_weight": float(np.sum(np.sort(np.abs(w))[::-1][:10])),
            "effective_n": float(1.0 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0,
            "ex_ante_vol": vol,
            "ex_ante_return": exp_ret,
            "ex_ante_sharpe": float(exp_ret / vol) if vol > 1e-12 else float("nan"),
            "turnover": float(np.sum(np.abs(w - w0))),
            "gross_exposure": float(np.sum(np.abs(w))),
            "net_exposure": float(np.sum(w)),
            "cash_weight": float(1.0 - np.sum(w)),
            "max_risk_contribution_share": float(np.max(rc) / var) if var > 1e-16 else 0.0,
        }


def optimize(
    mu: pd.Series | None,
    cov: pd.DataFrame,
    config: OptimizerConfig | dict | None = None,
    **kwargs,
) -> OptimizationResult:
    """Functional entry point used by the pipeline, CLI and tests."""
    cfg = config if isinstance(config, OptimizerConfig) else OptimizerConfig.from_dict(config or {})
    return PortfolioOptimizer(cfg).solve(mu, cov, **kwargs)


__all__ = [
    "METHODS",
    "OptimizerConfig",
    "OptimizationResult",
    "PortfolioOptimizer",
    "optimize",
]
