"""Unit tests for covariance estimation."""

from __future__ import annotations

import numpy as np

from alphaforge.risk.covariance import CovarianceEstimator


def test_ledoit_wolf_is_symmetric_and_psd(synthetic_returns):
    est = CovarianceEstimator("ledoit_wolf").estimate(synthetic_returns)
    m = est.matrix.to_numpy(dtype=float)
    assert np.allclose(m, m.T, atol=1e-8)
    eig = np.linalg.eigvalsh(m)
    # Shrinkage should keep the matrix (numerically) non-negative definite.
    assert eig.min() >= -1e-8


def test_ewma_blocks_are_consistent_shape(synthetic_returns):
    est = CovarianceEstimator("ewma", halflife=90).estimate(synthetic_returns)
    assert est.matrix.shape == (synthetic_returns.shape[1], synthetic_returns.shape[1])
    assert np.allclose(est.matrix.to_numpy(), est.matrix.to_numpy().T, atol=1e-8)


def test_sample_covariance_runs(synthetic_returns):
    est = CovarianceEstimator("sample").estimate(synthetic_returns)
    assert est.matrix.index.tolist() == synthetic_returns.columns.tolist()
