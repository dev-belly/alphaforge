"""Time-series aware cross-validation for cross-sectional alpha models.

``sklearn.model_selection.train_test_split(shuffle=True)`` is the canonical
mistake in financial ML: it leaks the future into the training set and produces
backtests that cannot lose.  Everything here is chronological.

Purge and embargo
-----------------
With an ``H``-day forward-return label, the label of an observation at date
``t`` is realised at ``t + H``.  A training sample whose label is still forming
at the start of the validation window leaks information, so we

* **purge** the last ``H`` observations of every training block, and
* **embargo** ``embargo_days`` observations immediately after the training
  block, which additionally kills the serial-correlation leakage that purging
  alone does not address (López de Prado, *Advances in Financial ML*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("models.split")


@dataclass
class Fold:
    """One walk-forward fold."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Fold({self.index}: train {self.train_start.date()}->{self.train_end.date()} | "
            f"test {self.test_start.date()}->{self.test_end.date()} | "
            f"n_train={int(self.train_mask.sum())} n_test={int(self.test_mask.sum())})"
        )


@dataclass
class WalkForwardConfig:
    train_years: int = 4
    test_years: int = 1
    step_years: int = 1
    purge_days: int = 21
    embargo_days: int = 21
    expanding: bool = True
    min_train_days: int = 250


class WalkForwardSplitter:
    """Expanding / rolling walk-forward split with purge and embargo."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, dates: pd.DatetimeIndex) -> list[Fold]:
        cfg = self.config
        if len(dates) == 0:
            return []
        dates = pd.DatetimeIndex(sorted(pd.unique(dates)))

        # Fold boundaries anchored on calendar years for interpretability.
        start_year = dates.min().year
        end_year = dates.max().year
        folds: list[Fold] = []
        idx = 0
        first_train_end = start_year + cfg.train_years
        for test_start_year in range(first_train_end, end_year + 1, cfg.step_years):
            train_start_year = start_year if cfg.expanding else test_start_year - cfg.train_years
            test_end_year = test_start_year + cfg.test_years

            train_start = max(pd.Timestamp(year=train_start_year, month=1, day=1), dates.min())
            train_end = min(
                pd.Timestamp(year=test_start_year, month=1, day=1) - pd.Timedelta(days=1),
                dates.max(),
            )
            test_start = pd.Timestamp(year=test_start_year, month=1, day=1)
            test_end = min(
                pd.Timestamp(year=test_end_year, month=1, day=1) - pd.Timedelta(days=1),
                dates.max(),
            )
            if train_start >= dates.max() or test_start > dates.max():
                break

            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            if train_mask.sum() < cfg.min_train_days or test_mask.sum() == 0:
                continue

            train_mask = self._purge(train_mask, dates, cfg.purge_days)
            test_mask = self._embargo(train_mask, test_mask, dates, cfg.embargo_days)
            if train_mask.sum() < cfg.min_train_days or test_mask.sum() == 0:
                continue

            folds.append(
                Fold(
                    index=idx,
                    train_start=dates[train_mask][0],
                    train_end=dates[train_mask][-1],
                    test_start=dates[test_mask][0],
                    test_end=dates[test_mask][-1],
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            )
            idx += 1

        log.info(
            f"WalkForwardSplitter: {len(folds)} folds "
            + (f"[{folds[0].train_start.date()} .. {folds[-1].test_end.date()}]" if folds else "")
        )
        return folds

    # ------------------------------------------------------------------
    @staticmethod
    def _purge(train_mask: np.ndarray, dates: pd.DatetimeIndex, purge_days: int) -> np.ndarray:
        """Drop the tail of the training block whose labels overlap the test set."""
        if purge_days <= 0:
            return train_mask
        out = train_mask.copy()
        pos = np.where(train_mask)[0]
        if pos.size == 0:
            return out
        cutoff = dates[pos[-1]] - pd.Timedelta(days=int(purge_days))
        out &= np.asarray(dates <= cutoff)
        return out

    @staticmethod
    def _embargo(
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        dates: pd.DatetimeIndex,
        embargo_days: int,
    ) -> np.ndarray:
        """Remove the first ``embargo_days`` sessions of the test block."""
        if embargo_days <= 0:
            return test_mask
        out = test_mask.copy()
        pos = np.where(train_mask)[0]
        if pos.size == 0:
            return out
        cutoff = dates[pos[-1]] + pd.Timedelta(days=int(embargo_days))
        out &= np.asarray(dates > cutoff)
        return out

    def get_n_splits(self) -> int:  # sklearn-compatible hook
        return -1


class PurgedKFold:
    """K-fold with purge + embargo, for hyper-parameter search *within* a fold.

    Only used for model selection; never for reporting out-of-sample results.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 21, embargo_days: int = 10) -> None:
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, dates: pd.DatetimeIndex) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(dates)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        out = []
        for i in range(self.n_splits):
            test_idx = np.zeros(n, dtype=bool)
            test_idx[bounds[i] : bounds[i + 1]] = True
            train_idx = ~test_idx
            train_idx = WalkForwardSplitter._purge(train_idx, dates, self.purge_days)
            test_idx = WalkForwardSplitter._embargo(train_idx, test_idx, dates, self.embargo_days)
            if train_idx.sum() and test_idx.sum():
                out.append((train_idx, test_idx))
        return out


__all__ = ["WalkForwardSplitter", "WalkForwardConfig", "Fold", "PurgedKFold"]
