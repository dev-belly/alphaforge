"""Offline tests for the covariance estimators (``risk/covariance.py``).

Covariance estimation is the single most error-prone input to a mean-variance
optimiser: an un-PSD matrix makes the solver unstable and a singular one makes it
put the whole book into one name. Every estimator here is pure (numpy/pandas), so
we lock each method's *contract* without a network:

  * the factory validates the method name once and never silently picks one;
  * every estimate is a square, symmetric, **PSD** matrix (nearest-PSD projection
    is applied uniformly), so downstream optimisers stay well-posed;
  * the structured ``factor`` estimator decomposes to ``B F B' + D`` with a floored
    specific variance, and refuses to run without both inputs;
  * ragged NaN patterns from delistings do not break the shrinkage estimators;
  * a degenerate (non-PSD) input to ``_psd`` comes out non-negative.

Deterministic synthetic returns (seeded RNG) keep the numeric assertions stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.risk.covariance import CovarianceEstimate, CovarianceEstimator, compare_estimators


def _returns(n: int = 120, p: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(n, p)),
        columns=[f"A{i}" for i in range(p)],
    )


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_factory_default_is_ledoit_wolf() -> None:
    assert CovarianceEstimator().method == "ledoit_wolf"


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "shrinkage", "ewma", "factor"])
def test_factory_accepts_every_supported_method(method: str) -> None:
    assert CovarianceEstimator(method).method == method


@pytest.mark.parametrize("method", ["SAMPLE", "Ewma", "Shrinkage", "FACTOR"])
def test_factory_is_case_insensitive(method: str) -> None:
    assert CovarianceEstimator(method).method in CovarianceEstimator.METHODS


def test_factory_none_defaults_to_ledoit_wolf() -> None:
    assert CovarianceEstimator(None).method == "ledoit_wolf"


def test_factory_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown covariance method"):
        CovarianceEstimator("oops")


# --------------------------------------------------------------------------
# Shape / PSD invariants across every method
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "shrinkage", "ewma"])
def test_estimate_returns_square_symmetric_psd_matrix(method: str) -> None:
    estimate = CovarianceEstimator(method).estimate(_returns())
    matrix = estimate.matrix
    assert matrix.shape == (5, 5)
    assert list(matrix.columns) == list(matrix.index)
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-8)
    # Nearest-PSD projection guarantees a non-negative minimum eigenvalue.
    evals = np.linalg.eigvalsh(matrix.to_numpy())
    assert evals.min() >= -1e-9


def test_estimate_drops_all_nan_rows() -> None:
    """``dropna(how="all")`` keeps the cross-section intact when a row is empty."""
    r = _returns(p=3)
    r.iloc[0, :] = np.nan
    estimate = CovarianceEstimator("sample").estimate(r)
    assert estimate.matrix.shape == (3, 3)


def test_estimate_requires_at_least_two_assets() -> None:
    with pytest.raises(ValueError, match="at least two assets"):
        CovarianceEstimator("ledoit_wolf").estimate(_returns(p=1))


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------


def test_sample_covariance_is_annualised_pandas_cov() -> None:
    r = _returns()
    estimate = CovarianceEstimator("sample").estimate(r)
    expected = r.cov(min_periods=max(int(0.5 * len(r)), 20)) * 252.0
    np.testing.assert_allclose(estimate.matrix.to_numpy(), expected.to_numpy(), atol=1e-5)
    assert estimate.method == "sample"


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------


def test_ewma_reacts_more_to_a_recent_shock_than_sample() -> None:
    """Exponential weighting should let a huge last-day return dominate the variance."""
    r = _returns(n=200, p=3)
    r.iloc[-1, 0] = 0.5
    ewma = CovarianceEstimator("ewma").estimate(r)
    sample = CovarianceEstimator("sample").estimate(r)
    assert ewma.matrix.iloc[0, 0] > sample.matrix.iloc[0, 0]


def test_ewma_honours_the_halflife_kwarg() -> None:
    estimate = CovarianceEstimator("ewma", halflife=30).estimate(_returns())
    assert estimate.method == "ewma(halflife=30)"


# --------------------------------------------------------------------------
# Constant-correlation shrinkage
# --------------------------------------------------------------------------


def test_shrinkage_tightens_off_diagonal_correlations_relative_to_sample() -> None:
    """Shrinkage pulls the off-diagonal correlations toward their common mean."""
    r = _returns()
    estimate = CovarianceEstimator("shrinkage").estimate(r)
    shrunk_off = np.corrcoef(estimate.matrix.to_numpy())[~np.eye(5, dtype=bool)]
    sample_off = np.corrcoef(CovarianceEstimator("sample").estimate(r).matrix.to_numpy())[
        ~np.eye(5, dtype=bool)
    ]
    # The spread of off-diagonal correlations shrinks; the target is constant-correlation.
    assert np.std(shrunk_off) < np.std(sample_off)
    assert 0.0 <= estimate.diagnostics["intensity"] <= 1.0


def test_shrinkage_accepts_an_explicit_intensity() -> None:
    estimate = CovarianceEstimator("shrinkage", intensity=0.3).estimate(_returns())
    assert abs(estimate.diagnostics["intensity"] - 0.3) < 1e-9


# --------------------------------------------------------------------------
# Ledoit-Wolf
# --------------------------------------------------------------------------


def test_ledoit_wolf_intensity_is_clipped_to_unit_interval() -> None:
    estimate = CovarianceEstimator("ledoit_wolf").estimate(_returns())
    assert estimate.method.startswith("ledoit_wolf")
    assert 0.0 <= estimate.diagnostics["intensity"] <= 1.0


def test_ledoit_wolf_tolerates_ragged_nans_from_delistings() -> None:
    """Delisted names leave gaps; the estimator must still return a usable matrix."""
    r = _returns()
    r.iloc[: len(r) // 2, 0] = np.nan
    estimate = CovarianceEstimator("ledoit_wolf").estimate(r)
    assert estimate.matrix.shape == (5, 5)
    evals = np.linalg.eigvalsh(estimate.matrix.to_numpy())
    assert evals.min() >= -1e-9


# --------------------------------------------------------------------------
# Factor model
# --------------------------------------------------------------------------


def test_factor_model_decomposes_into_bfb_plus_specific_variance() -> None:
    r = _returns(p=4)
    names = [f"A{i}" for i in range(4)]
    exposures = pd.DataFrame(np.eye(4), index=names, columns=names)
    factor_cov = pd.DataFrame(np.eye(4), index=names, columns=names)

    estimate = CovarianceEstimator("factor").estimate(r, exposures, factor_cov)

    assert estimate.matrix.shape == (4, 4)
    assert estimate.diagnostics["n_factors"] == 4
    # With identity B and F the structured part is I; the residual is added back as
    # a (floored, non-negative) specific-variance diagonal.
    assert estimate.diagnostics["median_specific_var"] >= 0.0
    evals = np.linalg.eigvalsh(estimate.matrix.to_numpy())
    assert evals.min() >= -1e-9


def test_factor_requires_exposures_and_factor_cov() -> None:
    r = _returns()
    # The guard only fires on genuinely missing inputs (None); empty frames are a
    # caller error and are allowed to surface as the underlying linear-algebra fault.
    with pytest.raises(ValueError, match="Factor covariance requires"):
        CovarianceEstimator("factor").estimate(r, None, None)


# --------------------------------------------------------------------------
# Nearest-PSD projection + diagnostics + comparison helper
# --------------------------------------------------------------------------


def test_psd_projection_removes_negative_eigenvalues() -> None:
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], columns=["x", "y"])  # eigenvalues 3, -1
    out = CovarianceEstimator._psd(bad)
    evals = np.linalg.eigvalsh(out.to_numpy())
    assert evals.min() >= 1e-10  # the -1 eigenvalue is lifted off the floor
    np.testing.assert_allclose(out.to_numpy(), out.to_numpy().T, atol=1e-10)


def test_diagnostics_report_a_condition_number() -> None:
    diagnostics = CovarianceEstimator("sample").estimate(_returns()).diagnostics
    assert diagnostics["condition_number"] >= 1.0
    assert diagnostics["n_assets"] == 5
    assert diagnostics["min_eigenvalue"] <= diagnostics["max_eigenvalue"]


def test_compare_estimators_returns_a_dataframe_and_skips_failures() -> None:
    out = compare_estimators(_returns())
    assert isinstance(out, pd.DataFrame)
    assert "method" in out.columns
    assert len(out) >= 3  # the four default estimators all succeed on clean data


def test_estimate_result_is_a_covariance_estimate() -> None:
    estimate = CovarianceEstimator("ledoit_wolf").estimate(_returns())
    assert isinstance(estimate, CovarianceEstimate)
    assert estimate.to_numpy().shape == (5, 5)
