"""Offline tests for the model factory and its estimators.

The failure mode this layer owns is *silent substitution*: the run reports the
model type you configured, trains something else, and nothing anywhere says so.
So the central test is not "does it fit" - it is "does the configured parameter
actually reach the underlying estimator". A Ridge with ``alpha=1e-6`` and one
with ``alpha=1e9`` must produce wildly different coefficients; if ``params`` were
dropped somewhere between the config and ``sklearn``, both would come back
identical and every downstream number would still look plausible.

Everything else follows the same principle:

  * the same seed must give bit-identical predictions, and a different seed must
    not - otherwise the walk-forward protocol is not reproducible;
  * the scaling lives *inside* the pipeline, proved by predicting on ``10 * X``
    and getting the same answer (an unscaled model would scale its output);
  * feature importance must rank the features that carry the signal first, and
    must be ``None`` before a fit rather than an empty frame that reads as
    "no features matter";
  * an unknown model type must raise with the available options, not fall back.

No network. Sizes are kept small so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.models.estimators import (
    REGISTRY,
    BaseEstimator,
    LightGBMModel,
    ModelConfig,
    RandomForestModel,
    RidgeModel,
    get_estimator,
)

N = 800
P = 5
NAMES = [f"f{i}" for i in range(P)]


def _data(seed: int = 7, n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """``f0`` and ``f2`` carry the signal; the rest are pure noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 2] + rng.normal(0.0, 0.1, size=n)
    return X, y


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------
def test_every_registry_entry_is_a_named_estimator() -> None:
    for key, cls in REGISTRY.items():
        assert issubclass(cls, BaseEstimator), f"{key} is not an estimator"
        assert cls().name == key, f"{cls.__name__}.name does not match its key"


def test_get_estimator_returns_the_registered_class() -> None:
    for key, cls in REGISTRY.items():
        assert isinstance(get_estimator(key), cls)


@pytest.mark.parametrize("alias", ["ridge", "RIDGE", "Ridge", "  ridge  ".strip()])
def test_the_model_type_is_case_insensitive(alias: str) -> None:
    assert get_estimator(alias).name == "ridge"


@pytest.mark.parametrize("blank", ["", None])
def test_a_blank_model_type_falls_back_to_ridge(blank) -> None:
    assert get_estimator(blank).name == "ridge"


def test_an_unknown_model_type_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Unknown model type") as excinfo:
        get_estimator("gradient_boosted_magic")
    for key in REGISTRY:
        assert key in str(excinfo.value)


def test_params_and_seed_reach_the_estimator() -> None:
    got = get_estimator("elasticnet", {"alpha": 0.25, "l1_ratio": 0.75}, seed=11)
    assert got.params == {"alpha": 0.25, "l1_ratio": 0.75}
    assert got.seed == 11
    assert got.describe() == {
        "model": "elasticnet",
        "params": {"alpha": 0.25, "l1_ratio": 0.75},
        "seed": 11,
    }


def test_params_are_copied_not_aliased() -> None:
    """A caller mutating its own dict must not reconfigure a live estimator."""
    supplied = {"alpha": 0.5}
    got = get_estimator("ridge", supplied)
    supplied["alpha"] = 999.0
    assert got.params == {"alpha": 0.5}


# ----------------------------------------------------------------------
# ModelConfig
# ----------------------------------------------------------------------
def test_model_config_normalises_type_and_seed() -> None:
    got = ModelConfig.from_dict({"type": "ELASTICNET", "params": {"alpha": 0.2}, "seed": "9"})
    assert got.type == "elasticnet"
    assert got.params == {"alpha": 0.2}
    assert got.seed == 9


@pytest.mark.parametrize("blank", [None, {}])
def test_an_empty_model_config_is_the_ridge_default(blank) -> None:
    got = ModelConfig.from_dict(blank)
    assert got.type == "ridge"
    assert got.params == {}
    assert got.seed == 42


def test_a_null_params_block_is_tolerated() -> None:
    assert ModelConfig.from_dict({"type": "ridge", "params": None}).params == {}


# ----------------------------------------------------------------------
# The base contract
# ----------------------------------------------------------------------
def test_predicting_before_a_fit_is_an_error_not_a_guess() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        BaseEstimator().predict(np.zeros((2, 3)))


def test_the_base_estimator_refuses_to_fit() -> None:
    with pytest.raises(NotImplementedError):
        BaseEstimator().fit(np.zeros((2, 3)), np.zeros(2))


def test_the_base_estimator_reports_no_importance() -> None:
    assert BaseEstimator().feature_importance() is None


def test_the_base_describe_records_the_seed() -> None:
    assert BaseEstimator({"a": 1}, seed=3).describe() == {
        "model": "base",
        "params": {"a": 1},
        "seed": 3,
    }


# ----------------------------------------------------------------------
# Ridge
# ----------------------------------------------------------------------
def test_ridge_recovers_a_linear_relationship() -> None:
    X, y = _data()
    got = get_estimator("ridge", {"alpha": 1e-6}).fit(X, y, NAMES)
    assert _r2(y, got.predict(X)) > 0.99


def test_ridge_alpha_actually_reaches_the_estimator() -> None:
    """If ``params`` were dropped, both fits would come back identical."""
    X, y = _data()
    loose = get_estimator("ridge", {"alpha": 1e-6}).fit(X, y, NAMES)
    tight = get_estimator("ridge", {"alpha": 1e9}).fit(X, y, NAMES)
    loose_norm = float(np.linalg.norm(loose.feature_importance().to_numpy()))
    tight_norm = float(np.linalg.norm(tight.feature_importance().to_numpy()))
    assert loose_norm > 1.0
    assert tight_norm < 1e-3


def test_ridge_standardises_its_input_inside_the_pipeline() -> None:
    """Fitting on ``10 * X`` must reproduce the model fitted on ``X``.

    The scaler sits inside the pipeline, so rescaling the features is absorbed
    by it and the fitted coefficients are unchanged. A model without that step
    would need coefficients ten times smaller and would not reproduce this.
    """
    X, y = _data()
    plain = get_estimator("ridge", {"alpha": 1e-6}).fit(X, y, NAMES)
    scaled = get_estimator("ridge", {"alpha": 1e-6}).fit(10.0 * X, y, NAMES)
    np.testing.assert_allclose(plain.predict(X), scaled.predict(10.0 * X), rtol=1e-6)
    assert type(plain.model[0]).__name__ == "StandardScaler"


def test_ridge_importance_ranks_the_signal_first_and_signed() -> None:
    X, y = _data()
    got = get_estimator("ridge", {"alpha": 1e-6}).fit(X, y, NAMES)
    imp = got.feature_importance()
    assert list(imp.index[:2]) == ["f0", "f2"]
    assert imp["f0"] > 0 > imp["f2"]  # +3.0 and -1.5
    assert (imp.abs().diff().dropna() <= 1e-12).all(), "not sorted by magnitude"


def test_importance_is_none_when_no_names_were_supplied() -> None:
    X, y = _data()
    got = get_estimator("ridge").fit(X, y)
    assert got.feature_importance() is None


def test_ridge_fit_returns_self_for_chaining() -> None:
    X, y = _data()
    est = RidgeModel()
    assert est.fit(X, y, NAMES) is est


# ----------------------------------------------------------------------
# ElasticNet
# ----------------------------------------------------------------------
def test_elasticnet_recovers_a_sparse_linear_relationship() -> None:
    X, y = _data()
    got = get_estimator("elasticnet", {"alpha": 1e-4, "l1_ratio": 1.0}, seed=3).fit(X, y, NAMES)
    assert _r2(y, got.predict(X)) > 0.9
    imp = got.feature_importance()
    assert list(imp.index[:2]) == ["f0", "f2"]


def test_elasticnet_alpha_actually_reaches_the_estimator() -> None:
    X, y = _data()
    weak = get_estimator("elasticnet", {"alpha": 1e-4, "l1_ratio": 1.0}, seed=3).fit(X, y, NAMES)
    strong = get_estimator("elasticnet", {"alpha": 1.0, "l1_ratio": 1.0}, seed=3).fit(X, y, NAMES)
    weak_norm = float(np.linalg.norm(weak.feature_importance().to_numpy()))
    strong_norm = float(np.linalg.norm(strong.feature_importance().to_numpy()))
    assert weak_norm > strong_norm, "a stronger penalty did not shrink the coefficients"


def test_elasticnet_importance_is_none_without_feature_names() -> None:
    X, y = _data()
    assert get_estimator("elasticnet", {"alpha": 1e-4}).fit(X, y).feature_importance() is None


def test_elasticnet_is_reproducible_for_a_seed() -> None:
    X, y = _data()
    params = {"alpha": 1e-3, "l1_ratio": 0.5}
    a = get_estimator("elasticnet", params, seed=5).fit(X, y, NAMES)
    b = get_estimator("elasticnet", params, seed=5).fit(X, y, NAMES)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# ----------------------------------------------------------------------
# Random forest
# ----------------------------------------------------------------------
# The production defaults (min_samples_leaf=500, max_samples=0.4) assume a
# portfolio-scale sample; on these few hundred rows they would leave every tree
# a single unsplittable leaf and the forest a constant. Loosened here so the
# tests exercise a forest that can actually learn.
_FOREST_PARAMS = {"n_estimators": 24, "min_samples_leaf": 20, "max_samples": 0.8}


def _forest(params: dict | None = None, seed: int = 1) -> RandomForestModel:
    X, y = _data(seed=101, n=400)
    est = get_estimator("random_forest", {**_FOREST_PARAMS, **(params or {})}, seed=seed)
    return est.fit(X, y, NAMES)


def test_random_forest_fits_and_predicts() -> None:
    X, y = _data(seed=101, n=400)
    got = _forest()
    assert _r2(y, got.predict(X)) > 0.5
    assert np.isfinite(got.predict(X)).all()


def test_random_forest_is_reproducible_for_a_seed() -> None:
    """Single-threaded is bit-identical; the default parallel path agrees to ~1e-15.

    ``n_jobs`` defaults to 2, so the per-tree sums are accumulated in a
    different order and the last ulp can move. That is not a reproducibility
    failure - no reported metric is affected - but it is worth pinning so a
    future change that made the two runs *materially* differ would be caught.
    """
    X, _ = _data(seed=101, n=400)
    serial = _forest({"n_jobs": 1}, seed=1)
    serial_again = _forest({"n_jobs": 1}, seed=1)
    np.testing.assert_array_equal(serial.predict(X), serial_again.predict(X))

    parallel = _forest(seed=1)
    parallel_again = _forest(seed=1)
    np.testing.assert_allclose(
        parallel.predict(X), parallel_again.predict(X), rtol=1e-12, atol=1e-14
    )


def test_random_forest_changes_with_the_seed() -> None:
    X, _ = _data(seed=101, n=400)
    a = _forest(seed=1)
    c = _forest(seed=2)
    assert not np.allclose(a.predict(X), c.predict(X))


def test_random_forest_tree_count_reaches_the_estimator() -> None:
    small = _forest({"n_estimators": 4})
    large = _forest({"n_estimators": 24})
    assert small.model.n_estimators == 4
    assert large.model.n_estimators == 24


def test_random_forest_importance_is_non_negative_and_ranked() -> None:
    imp = _forest().feature_importance()
    assert (imp >= 0).all()
    assert imp.index[0] == "f0"
    assert imp.is_monotonic_decreasing


def test_random_forest_importance_is_none_without_feature_names() -> None:
    X, y = _data(seed=101, n=400)
    got = get_estimator("random_forest", _FOREST_PARAMS, seed=1).fit(X, y)
    assert got.feature_importance() is None


# ----------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------
# Same story as the forest: the production default of min_data_in_leaf=200
# caps a 500-row sample at a two-leaf stump per tree, so the boosted model
# cannot express a linear relationship and R^2 stays near zero.
_BOOSTER_PARAMS = {"n_estimators": 60, "min_data_in_leaf": 20}


def _booster(params: dict | None = None, seed: int = 4) -> LightGBMModel:
    X, y = _data(seed=202, n=500)
    est = get_estimator("lightgbm", {**_BOOSTER_PARAMS, **(params or {})}, seed=seed)
    return est.fit(X, y, NAMES)


def test_lightgbm_fits_and_predicts() -> None:
    X, y = _data(seed=202, n=500)
    got = _booster()
    assert _r2(y, got.predict(X)) > 0.5
    assert np.isfinite(got.predict(X)).all()


def test_lightgbm_boost_rounds_reach_the_estimator() -> None:
    assert _booster({"n_estimators": 7}).model.num_trees() == 7


def test_lightgbm_is_reproducible_for_a_seed() -> None:
    X, _ = _data(seed=202, n=500)
    a = _booster(seed=4)
    b = _booster(seed=4)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_lightgbm_importance_supports_both_types() -> None:
    got = _booster()
    gain = got.feature_importance(importance_type="gain")
    split = got.feature_importance(importance_type="split")
    assert isinstance(gain, pd.Series)
    assert set(gain.index) == set(NAMES)
    assert gain.index[0] == "f0", "the signal feature should carry the most gain"
    assert gain.is_monotonic_decreasing
    assert (split >= 0).all()


def test_lightgbm_fits_without_feature_names() -> None:
    """Regression: an empty name list used to crash inside ``lgb.train``.

    LightGBM validates ``len(feature_name) == num_feature`` and rejects ``[]``,
    so the estimator raised "Length of feature_name(0) and num_feature(5) don't
    match" even though ``feature_names`` is optional on every other estimator.
    """
    X, y = _data(seed=202, n=500)
    got = get_estimator("lightgbm", {"n_estimators": 5}).fit(X, y)
    assert got.feature_importance() is None
    assert np.isfinite(got.predict(X)).all()
