"""Covariance estimation for portfolio construction.

A sample covariance matrix over 120 names estimated from 252 daily observations
is close to singular: the largest eigenvalue is inflated and the smallest is
biased toward zero, which is precisely where a mean-variance optimiser puts its
money.  Every estimator here is a shrinkage / weighting scheme that trades a
little bias for a large reduction in estimation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.utils.logging import get_logger

log = get_logger("risk.covariance")


@dataclass
class CovarianceEstimate:
    matrix: pd.DataFrame
    method: str
    diagnostics: dict

    def to_numpy(self) -> np.ndarray:
        return self.matrix.to_numpy(dtype=float)


class CovarianceEstimator:
    """Factory for the supported covariance estimators."""

    METHODS = ("sample", "ledoit_wolf", "shrinkage", "ewma", "factor")

    def __init__(self, method: str = "ledoit_wolf", **kwargs) -> None:
        method = (method or "ledoit_wolf").lower()
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown covariance method {method!r}; expected one of {self.METHODS}"
            )
        self.method = method
        self.kwargs = kwargs

    def estimate(
        self,
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame | None = None,
        factor_cov: pd.DataFrame | None = None,
    ) -> CovarianceEstimate:
        rets = returns.dropna(how="all")
        if rets.shape[1] < 2:
            raise ValueError("Need at least two assets to estimate a covariance matrix")
        fn = {
            "sample": self._sample,
            "ledoit_wolf": self._ledoit_wolf,
            "shrinkage": self._shrinkage,
            "ewma": self._ewma,
            "factor": self._factor,
        }[self.method]
        if self.method == "factor":
            return fn(rets, factor_exposures, factor_cov)
        return fn(rets)

    # ------------------------------------------------------------------
    def _sample(self, rets: pd.DataFrame) -> CovarianceEstimate:
        cov = rets.cov(min_periods=max(int(0.5 * len(rets)), 20)) * 252.0
        return CovarianceEstimate(
            matrix=self._psd(cov),
            method="sample",
            diagnostics=self._diagnostics(cov, rets),
        )

    def _ewma(self, rets: pd.DataFrame) -> CovarianceEstimate:
        halflife = float(self.kwargs.get("halflife", 90))
        lam = 0.5 ** (1.0 / max(halflife, 1.0))
        x = rets.fillna(0.0).to_numpy(dtype=float)
        n = x.shape[0]
        weights = lam ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()
        xm = x - (weights[:, None] * x).sum(axis=0)
        cov = (xm * weights[:, None]).T @ xm * 252.0
        cov = pd.DataFrame(cov, index=rets.columns, columns=rets.columns)
        return CovarianceEstimate(
            matrix=self._psd(cov),
            method=f"ewma(halflife={halflife:.0f})",
            diagnostics=self._diagnostics(cov, rets),
        )

    def _shrinkage(self, rets: pd.DataFrame) -> CovarianceEstimate:
        """Constant-variance / constant-correlation shrinkage target."""
        intensity = self.kwargs.get("intensity")
        cov = rets.cov(min_periods=max(int(0.5 * len(rets)), 20)).to_numpy(dtype=float) * 252.0
        var = np.diag(cov).copy()
        std = np.sqrt(np.clip(var, 1e-12, None))
        corr = cov / np.outer(std, std)
        n = corr.shape[0]
        off = corr[~np.eye(n, dtype=bool)]
        rbar = float(np.nanmean(off)) if off.size else 0.0
        target_corr = np.full((n, n), rbar)
        np.fill_diagonal(target_corr, 1.0)
        target = target_corr * np.outer(std, std)

        if intensity is None:
            intensity = self._optimal_intensity(rets, cov, target)
        shrunk = (1.0 - intensity) * cov + intensity * target
        out = pd.DataFrame(shrunk, index=rets.columns, columns=rets.columns)
        return CovarianceEstimate(
            matrix=self._psd(out),
            method=f"shrinkage(intensity={intensity:.3f})",
            diagnostics={**self._diagnostics(out, rets), "intensity": float(intensity)},
        )

    def _ledoit_wolf(self, rets: pd.DataFrame) -> CovarianceEstimate:
        """Ledoit-Wolf shrinkage to a constant-correlation target.

        Implemented directly (rather than via ``sklearn``) so that the target is
        the constant-correlation matrix standard in risk systems, and so the
        estimator tolerates the ragged NaN patterns that delistings produce.
        """
        x = rets.to_numpy(dtype=float)
        mask = np.isfinite(x)
        x = np.where(mask, x, 0.0)
        t, n = x.shape
        if t < 2 or n < 2:
            raise ValueError("Not enough observations for Ledoit-Wolf shrinkage")

        means = x.sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
        xc = np.where(mask, x - means, 0.0)

        sample = (
            (xc.T @ xc) / np.maximum((mask.astype(float).T @ mask.astype(float)) - 1.0, 1.0) * 252.0
        )
        var = np.diag(sample).copy()
        std = np.sqrt(np.clip(var, 1e-12, None))
        corr = sample / np.outer(std, std)
        off_diag = corr[~np.eye(n, dtype=bool)]
        rbar = float(np.nanmean(off_diag)) if off_diag.size else 0.0
        target_corr = np.full((n, n), rbar)
        np.fill_diagonal(target_corr, 1.0)
        target = target_corr * np.outer(std, std)

        # Ledoit-Wolf (2003) constant-correlation intensity estimator.
        y = xc**2
        pi_mat = (y.T @ y) / np.maximum(mask.astype(float).T @ mask.astype(float), 1.0) * (252.0**2)
        pi_hat = float(np.sum(pi_mat - sample**2))

        term = (
            ((xc**3).T @ xc)
            / np.maximum(mask.astype(float).T @ mask.astype(float), 1.0)
            * (252.0**2)
        )
        rho_diag = np.sum(np.diag(pi_mat) - np.diag(sample) ** 2)

        rbar_mat = rbar * np.outer(std, std)
        term_target = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sij = std[i] * std[j]
                term_target[i, j] = rbar * (
                    term[i, j]
                    - 0.5
                    * sij
                    * (pi_mat[i, i] / max(var[i], 1e-12) + pi_mat[j, j] / max(var[j], 1e-12))
                )
        rho_off = float(np.sum(term_target - rbar_mat * sample))
        rho_hat = rho_diag + rho_off
        gamma_hat = float(np.linalg.norm(sample - target, "fro") ** 2)
        kappa = (pi_hat - rho_hat) / max(gamma_hat, 1e-18)
        intensity = float(np.clip(kappa / max(t, 1), 0.0, 1.0))

        shrunk = (1.0 - intensity) * sample + intensity * target
        out = pd.DataFrame(shrunk, index=rets.columns, columns=rets.columns)
        return CovarianceEstimate(
            matrix=self._psd(out),
            method=f"ledoit_wolf(intensity={intensity:.3f})",
            diagnostics={**self._diagnostics(out, rets), "intensity": intensity},
        )

    def _factor(
        self,
        rets: pd.DataFrame,
        exposures: pd.DataFrame | None,
        factor_cov: pd.DataFrame | None,
    ) -> CovarianceEstimate:
        """Structured covariance: ``B F B' + D`` (specific variance diagonal)."""
        if exposures is None or factor_cov is None:
            raise ValueError("Factor covariance requires exposures and a factor covariance matrix")
        cols = list(rets.columns)
        B = exposures.reindex(cols).fillna(0.0).to_numpy(dtype=float)
        F = factor_cov.to_numpy(dtype=float)
        resid = rets.to_numpy(dtype=float) - rets.to_numpy(dtype=float) @ np.linalg.pinv(B) @ B
        specific = np.nanvar(resid, axis=0) * 252.0
        # Floor specific variance: a zero-variance name makes Sigma singular and
        # the optimiser will happily put the whole book in it.
        floor = float(np.nanmedian(np.clip(specific, 1e-8, None)) * 0.10)
        specific = np.clip(specific, floor, None)
        cov = B @ F @ B.T + np.diag(specific)
        out = pd.DataFrame(cov, index=cols, columns=cols)
        return CovarianceEstimate(
            matrix=self._psd(out),
            method="factor",
            diagnostics={
                **self._diagnostics(out, rets),
                "n_factors": int(B.shape[1]),
                "median_specific_var": float(np.median(specific)),
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _psd(cov: pd.DataFrame) -> pd.DataFrame:
        """Nearest PSD projection with a small ridge for numerical stability."""
        m = cov.to_numpy(dtype=float)
        m = 0.5 * (m + m.T)
        vals, vecs = np.linalg.eigh(m)
        vals = np.clip(vals, 1e-10, None)
        out = vecs @ np.diag(vals) @ vecs.T
        out = 0.5 * (out + out.T)
        return pd.DataFrame(out, index=cov.index, columns=cov.columns)

    @staticmethod
    def _diagnostics(cov: pd.DataFrame, rets: pd.DataFrame) -> dict:
        m = cov.to_numpy(dtype=float)
        vals = np.linalg.eigvalsh(0.5 * (m + m.T))
        vals = np.sort(vals)[::-1]
        return {
            "n_assets": int(m.shape[0]),
            "n_obs": int(rets.notna().sum().max()),
            "condition_number": float(vals[0] / max(vals[-1], 1e-12)),
            "max_eigenvalue": float(vals[0]),
            "min_eigenvalue": float(vals[-1]),
            "avg_annual_vol": float(np.sqrt(np.mean(np.clip(np.diag(m), 0, None)))),
        }


def compare_estimators(returns: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    """Compare estimators on conditioning and out-of-sample log-likelihood style stats."""
    methods = methods or ["sample", "ledoit_wolf", "ewma", "shrinkage"]
    rows = []
    for m in methods:
        try:
            est = CovarianceEstimator(m).estimate(returns)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Covariance estimator {m} failed: {exc}")
            continue
        rows.append({"method": est.method, **est.diagnostics})
    return pd.DataFrame(rows)


__all__ = ["CovarianceEstimator", "CovarianceEstimate", "compare_estimators"]
