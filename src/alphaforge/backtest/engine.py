"""Event-driven backtest engine.

Design contract
---------------
1. **Signals are generated on the rebalance date using only data available on
   that date.**  The constructor receives the panel up to and including ``t``.
2. **Orders are executed ``execution_lag_days`` sessions later**, at that
   session's close.  This is the structural guard against look-ahead: a signal
   can never be traded on the same close that produced it.
3. **The day's return is earned on the pre-trade book**, and trading costs are
   deducted from NAV at the end of the session.  Costs therefore reduce the
   compounding base exactly as they do in production.
4. **Untradeable names are not silently dropped.**  A name with no price on the
   execution date keeps its position (marked at the last valid close); a name
   that has been dark for ``delist_grace_days`` sessions is force-liquidated at
   the last valid price.  Dropping them instead is how backtests acquire an
   undeclared survivorship bias.

The engine is deliberately a pure accounting loop: it owns no alpha logic.  All
portfolio decisions are delegated to a ``weight_fn`` (by default a
:class:`~alphaforge.portfolio.constructor.PortfolioConstructor`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.backtest.metrics import MetricsConfig, performance_stats
from alphaforge.data.calendar import execution_dates, rebalance_dates
from alphaforge.execution.broker import BrokerSimulator
from alphaforge.execution.costs import CostModel
from alphaforge.features.panel import MarketPanel
from alphaforge.utils.logging import Timer, get_logger

log = get_logger("backtest.engine")

WeightFn = Callable[[pd.Timestamp, pd.Series | None], pd.Series | None]


@dataclass
class BacktestConfig:
    """Backtest mechanics - everything that is not alpha."""

    start_date: str | None = None
    end_date: str | None = None
    rebalance: str = "monthly"
    initial_capital: float = 10_000_000.0
    execution_lag_days: int = 1
    allow_short: bool = False
    min_history_days: int = 252
    adv_window: int = 20
    max_stale_days: int = 5
    delist_grace_days: int = 5
    max_gross_leverage: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> BacktestConfig:
        cfg = cfg or {}
        return cls(
            start_date=cfg.get("start_date"),
            end_date=cfg.get("end_date"),
            rebalance=str(cfg.get("rebalance", "monthly")),
            initial_capital=float(cfg.get("initial_capital", 10_000_000.0)),
            execution_lag_days=int(cfg.get("execution_lag_days", 1)),
            allow_short=bool(cfg.get("allow_short", False)),
            min_history_days=int(cfg.get("min_history_days", 252)),
            adv_window=int(cfg.get("adv_window", 20)),
            max_stale_days=int(cfg.get("max_stale_days", 5)),
            delist_grace_days=int(cfg.get("delist_grace_days", 5)),
            max_gross_leverage=float(cfg.get("max_gross_leverage", 1.0)),
        )


@dataclass
class BacktestResult:
    """Everything the reporting layer and the tests need."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # (dates x symbols) end-of-day held weights
    target_weights: pd.DataFrame  # (rebalance dates x symbols)
    trades: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    metrics: dict
    benchmark: pd.Series | None = None
    diagnostics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def nav(self) -> pd.Series:
        return self.equity

    def summary(self) -> dict:
        keys = [
            "total_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "information_ratio",
            "beta",
            "alpha_ann",
            "avg_turnover",
            "hit_rate",
        ]
        return {k: self.metrics.get(k) for k in keys if k in self.metrics}

    def trades_by_date(self) -> pd.DataFrame:
        return self.trades


class BacktestEngine:
    """Runs the accounting loop for one strategy."""

    def __init__(
        self,
        panel: MarketPanel,
        weight_fn: WeightFn | None = None,
        constructor=None,
        signals: pd.DataFrame | None = None,
        ic: float = 0.05,
        cost_model: CostModel | dict | None = None,
        broker: BrokerSimulator | None = None,
        config: BacktestConfig | dict | None = None,
        metrics_config: MetricsConfig | dict | None = None,
        benchmark: pd.Series | None = None,
    ) -> None:
        self.panel = panel
        self.config = (
            config if isinstance(config, BacktestConfig) else BacktestConfig.from_dict(config or {})
        )
        self.metrics_config = metrics_config
        self.cost_model = (
            cost_model if isinstance(cost_model, CostModel) else CostModel(cost_model or {})
        )
        self.broker = broker or BrokerSimulator(self.cost_model)
        self.benchmark = benchmark if benchmark is not None else panel.benchmark
        self.signals = signals
        self.ic = ic
        self.constructor = constructor
        self.weight_fn = weight_fn or self._default_weight_fn
        if self.weight_fn is self._default_weight_fn and constructor is None and signals is None:
            raise ValueError("Provide either `constructor` + `signals`, or an explicit `weight_fn`")

    # ------------------------------------------------------------------
    def _default_weight_fn(
        self, date: pd.Timestamp, prev_weights: pd.Series | None
    ) -> pd.Series | None:
        if self.signals is None or date not in self.signals.index:
            return None
        scores = self.signals.loc[date].dropna()
        if scores.empty:
            return None
        try:
            return self.constructor.construct(
                date=date, scores=scores, prev_weights=prev_weights, ic=self.ic
            ).weights
        except Exception as exc:  # noqa: BLE001 - a failed rebalance must not kill the run
            log.warning(f"Construction failed on {date.date()}: {exc}")
            return None

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        cfg = self.config
        panel = self.panel
        dates = panel.dates
        if cfg.start_date:
            dates = dates[dates >= pd.Timestamp(cfg.start_date)]
        if cfg.end_date:
            dates = dates[dates <= pd.Timestamp(cfg.end_date)]
        if len(dates) < cfg.min_history_days + 5:
            raise ValueError(
                f"Backtest window has {len(dates)} sessions; need at least "
                f"{cfg.min_history_days + 5} for the risk-model warm-up"
            )

        rb_dates = rebalance_dates(dates, cfg.rebalance)
        # The warm-up block is not tradable: the covariance window has to fill
        # first, otherwise the first rebalance is estimated on a handful of rows.
        warmup_end = dates[cfg.min_history_days]
        rb_dates = rb_dates[rb_dates >= warmup_end]
        exec_map = execution_dates(rb_dates, dates, cfg.execution_lag_days)
        exec_to_signal = {
            pd.Timestamp(v): pd.Timestamp(k) for k, v in exec_map.items() if v <= dates[-1]
        }

        adv = panel.dollar_volume.rolling(cfg.adv_window, min_periods=5).mean()
        valuation_px = panel.close.ffill(limit=cfg.max_stale_days)
        stale = self._stale_streak(panel.close)

        symbols = panel.symbols
        shares = pd.Series(0.0, index=symbols, dtype=float)
        cash = float(cfg.initial_capital)
        prev_nav = float(cfg.initial_capital)

        equity: dict[pd.Timestamp, float] = {}
        rets: dict[pd.Timestamp, float] = {}
        weight_rows: dict[pd.Timestamp, pd.Series] = {}
        target_rows: dict[pd.Timestamp, pd.Series] = {}
        turnover_rows: dict[pd.Timestamp, float] = {}
        cost_rows: dict[pd.Timestamp, float] = {}
        trades_frames: list[pd.DataFrame] = []
        pending: pd.Series | None = None
        n_rebalances = 0
        n_failed = 0

        with Timer(f"backtest[{cfg.rebalance}]", log):
            for date in dates:
                raw_px = panel.close.loc[date]
                mark_px = valuation_px.loc[date]
                held = shares.abs() > 1e-12

                # -- 1. force-liquidate names that have gone dark -----------
                dead = held & (stale.loc[date] >= cfg.delist_grace_days)
                if bool(dead.any()):
                    proceeds = float((shares[dead] * mark_px[dead].fillna(0.0)).sum())
                    notional = float((shares[dead].abs() * mark_px[dead].fillna(0.0)).sum())
                    cost = self.cost_model.estimate(
                        notional, float(adv.loc[date].reindex(symbols[dead]).median() or 0.0)
                    ).total
                    shares[dead] = 0.0
                    cash += proceeds - cost
                    cost_rows[date] = cost_rows.get(date, 0.0) + cost
                    log.debug(f"{date.date()}: liquidated {int(dead.sum())} delisted names")

                # -- 2. mark to market on the pre-trade book ----------------
                holdings_value = float(
                    np.nansum(shares.to_numpy(dtype=float) * mark_px.to_numpy(dtype=float))
                )
                nav_t = holdings_value + cash
                if prev_nav > 0:
                    rets[date] = nav_t / prev_nav - 1.0
                equity[date] = nav_t

                # -- 3. execute anything signaled `lag` sessions ago --------
                if date in exec_to_signal and pending is not None:
                    current_w = (
                        (shares * mark_px) / nav_t if nav_t > 0 else pd.Series(0.0, index=symbols)
                    )
                    result = self.broker.rebalance(
                        target_weights=pending,
                        current_weights=current_w,
                        nav=nav_t,
                        prices=raw_px,
                        adv_value=adv.loc[date],
                    )
                    if not result.trades.empty:
                        frame = result.trades.copy()
                        frame.insert(0, "date", date)
                        trades_frames.append(frame.reset_index(names="symbol"))
                        turnover_rows[date] = float(result.trades["trade_weight"].abs().sum())
                        cost_rows[date] = cost_rows.get(date, 0.0) + result.cost_total

                    nav_after = nav_t - result.cost_total
                    tradable = raw_px.notna() & (raw_px > 0)
                    new_shares = pd.Series(0.0, index=symbols, dtype=float)
                    if nav_after > 0:
                        new_shares[tradable] = (
                            result.weights[tradable].to_numpy(dtype=float)
                            * nav_after
                            / raw_px[tradable].to_numpy(dtype=float)
                        )
                    new_shares[~tradable] = shares[~tradable]
                    shares = new_shares.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    cash = nav_after - float(
                        np.nansum(shares.to_numpy(dtype=float) * mark_px.to_numpy(dtype=float))
                    )
                    nav_t = nav_after
                    # The day's return must be earned on the *post-cost* book, so that
                    # transaction costs actually depress the return series and the equity
                    # curve stays consistent with it. Without this, costs are deducted
                    # from NAV (step 2 above used the pre-trade book) but never reflected
                    # in `returns`, which inflates the net Sharpe/CAGR.
                    if prev_nav > 0:
                        rets[date] = nav_after / prev_nav - 1.0
                    n_rebalances += 1
                    pending = None
                elif date in exec_to_signal and pending is None:
                    n_failed += 1

                # -- 4. record end-of-day book ------------------------------
                if nav_t > 0:
                    held_w = (shares * mark_px) / nav_t
                    weight_rows[date] = held_w
                equity[date] = nav_t
                prev_nav = nav_t

                # -- 5. generate the next signal ----------------------------
                if date in set(rb_dates):
                    if not cfg.allow_short:
                        prev_w = weight_rows.get(date)
                    else:
                        prev_w = weight_rows.get(date)
                    try:
                        target = self.weight_fn(date, prev_w)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(f"weight_fn raised on {date.date()}: {exc}")
                        target = None
                    if target is not None and len(target):
                        target = target.reindex(symbols).fillna(0.0)
                        if not cfg.allow_short:
                            target = target.clip(lower=0.0)
                        gross = float(target.abs().sum())
                        if gross > cfg.max_gross_leverage > 0:
                            target = target * (cfg.max_gross_leverage / gross)
                        target_rows[date] = target
                        pending = target
                    else:
                        n_failed += 1

        equity_s = pd.Series(equity, dtype=float).sort_index()
        returns_s = pd.Series(rets, dtype=float).sort_index()
        weights_df = pd.DataFrame.from_dict(weight_rows, orient="index").sort_index()
        weights_df = weights_df.reindex(columns=symbols).fillna(0.0)
        targets_df = pd.DataFrame.from_dict(target_rows, orient="index").sort_index()
        trades_df = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
        turnover_s = pd.Series(turnover_rows, dtype=float).sort_index()
        costs_s = pd.Series(cost_rows, dtype=float).sort_index()

        bench = None
        if self.benchmark is not None:
            bench = self.benchmark.reindex(returns_s.index).dropna()
            bench = bench.pct_change(fill_method=None).dropna()

        years = len(returns_s) / 252.0
        cost_drag = (
            float(costs_s.sum() / max(equity_s.mean(), 1.0) / max(years, 1e-9))
            if years > 0
            else 0.0
        )
        metrics = performance_stats(
            returns_s,
            benchmark=bench,
            config=self.metrics_config,
            turnover=turnover_s,
            cost_drag=cost_drag,
        )

        log.info(
            f"Backtest {returns_s.index[0].date()} -> {returns_s.index[-1].date()} | "
            f"CAGR {metrics.get('cagr', float('nan')):+.2%} | "
            f"Sharpe {metrics.get('sharpe', float('nan')):.2f} | "
            f"MaxDD {metrics.get('max_drawdown', float('nan')):.2%} | "
            f"rebalances {n_rebalances}"
        )
        return BacktestResult(
            equity=equity_s,
            returns=returns_s,
            weights=weights_df,
            target_weights=targets_df,
            trades=trades_df,
            turnover=turnover_s,
            costs=costs_s,
            metrics=metrics,
            benchmark=bench,
            diagnostics={
                "n_rebalances": int(n_rebalances),
                "n_skipped_rebalances": int(n_failed),
                "n_trades": int(len(trades_df)),
                "total_costs": float(costs_s.sum()),
                "cost_drag_ann": float(cost_drag),
                "avg_turnover": float(turnover_s.mean()) if len(turnover_s) else 0.0,
                "avg_holdings": float((weights_df.abs() > 1e-6).sum(axis=1).mean()),
                "avg_gross_exposure": float(weights_df.abs().sum(axis=1).mean()),
                "execution_lag_days": cfg.execution_lag_days,
                "rebalance": cfg.rebalance,
            },
            config=cfg.__dict__,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stale_streak(close: pd.DataFrame) -> pd.DataFrame:
        """Consecutive-session count of missing prices, per name."""
        arr = close.isna().to_numpy(dtype=bool)
        streak = np.zeros(arr.shape, dtype=np.int32)
        running = np.zeros(arr.shape[1], dtype=np.int32)
        for i in range(arr.shape[0]):
            running = np.where(arr[i], running + 1, 0)
            streak[i] = running
        return pd.DataFrame(streak, index=close.index, columns=close.columns)


def run_backtest(
    panel: MarketPanel,
    signals: pd.DataFrame | None = None,
    constructor=None,
    cfg: dict | None = None,
    ic: float = 0.05,
    **kwargs,
) -> BacktestResult:
    """Convenience wrapper used by the pipeline, the CLI and the API."""
    cfg = cfg or {}
    return BacktestEngine(
        panel=panel,
        constructor=constructor,
        signals=signals,
        ic=ic,
        cost_model=cfg.get("cost", {}),
        config=BacktestConfig.from_dict(cfg.get("backtest", {})),
        metrics_config=cfg.get("metrics", {}),
        **kwargs,
    ).run()


__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult", "run_backtest"]
