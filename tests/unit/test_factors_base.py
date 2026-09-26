"""Offline tests for the factor abstraction and the registry.

This module is where a factor's *metadata* is validated, and the campaign has
already found one bug of exactly that kind nearby: `value_composite` and
`quality_composite` raised a plain `RuntimeError` where the registry catches only
`FactorUnavailableError`, so instead of degrading they aborted the call. The
tests here pin the contract that made that bug matter:

  * ``FactorSpec`` validates what it can - category, direction, and (newly) a
    non-empty name, which is the registry key. An empty name used to register
    under ``""`` and show up as a nameless row in the factor table.
  * ``compute`` degrades on ``FactorUnavailableError`` and raises on anything
    else; ``compute_all`` is the layer that isolates a genuinely broken factor,
    and it must also drop a factor that produced no observations rather than
    carrying a column of NaN into the model.
  * the registry is a mapping keyed by name, so the tests pin its ordering,
    membership and length rather than leaving them to chance.

Everything is deterministic: no network, no panel data beyond a tiny stub.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.factors.base import (
    CATEGORIES,
    REGISTRY,
    Factor,
    FactorContext,
    FactorRegistry,
    FactorSpec,
    FactorUnavailableError,
    _empty_like,
    register,
)


class _StubPanel:
    """The smallest thing that satisfies FactorContext's property proxies."""

    def __init__(self, n_dates: int = 3, symbols: tuple[str, ...] = ("A", "B")) -> None:
        self.dates = pd.date_range("2024-01-01", periods=n_dates)
        self.symbols = pd.Index(symbols)
        self.close = pd.DataFrame(1.0, index=self.dates, columns=self.symbols)
        self.returns = self.close
        self.market_cap = self.close


def _ctx(n_dates: int = 3) -> FactorContext:
    return FactorContext(_StubPanel(n_dates))


def _constant(value: float = 1.0):
    def fn(ctx: FactorContext) -> pd.DataFrame:
        return pd.DataFrame(value, index=ctx.panel.dates, columns=ctx.panel.symbols)

    return fn


# ----------------------------------------------------------------------
# FactorSpec validation
# ----------------------------------------------------------------------
def test_a_valid_spec_keeps_its_fields() -> None:
    spec = FactorSpec(name="mom", category="momentum", direction=-1, description="d")
    assert spec.name == "mom"
    assert spec.category == "momentum"
    assert spec.direction == -1
    assert spec.description == "d"
    assert spec.requires_fundamentals is False
    assert spec.data_requirement == "price"


def test_the_defaults_are_the_safe_ones() -> None:
    spec = FactorSpec(name="x", category="value")
    assert spec.direction == 1
    assert spec.requires_fundamentals is False


def test_an_unknown_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown category"):
        FactorSpec(name="x", category="not_a_category")


@pytest.mark.parametrize("direction", [0, 2, -2, 100])
def test_a_direction_other_than_plus_or_minus_one_is_rejected(direction: int) -> None:
    """A direction of 0 would silently zero the factor in the preprocessor."""
    with pytest.raises(ValueError, match="direction must be"):
        FactorSpec(name="x", category="value", direction=direction)


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_an_empty_name_is_rejected(name: str) -> None:
    """The name is the registry key - a blank one becomes a nameless factor row."""
    with pytest.raises(ValueError, match="non-empty"):
        FactorSpec(name=name, category="value")


def test_every_category_in_the_tuple_is_accepted() -> None:
    for category in CATEGORIES:
        assert FactorSpec(name=f"f_{category}", category=category).category == category


def test_the_category_tuple_is_the_documented_one() -> None:
    assert CATEGORIES == (
        "momentum",
        "reversal",
        "value",
        "quality",
        "risk",
        "liquidity",
        "size",
    )


def test_a_spec_is_hashable_and_frozen() -> None:
    """Frozen so a registered spec cannot be edited behind the registry's back."""
    import dataclasses

    spec = FactorSpec(name="x", category="value")
    assert hash(spec) is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "y"  # type: ignore[misc]


# ----------------------------------------------------------------------
# Factor
# ----------------------------------------------------------------------
def test_a_factor_exposes_its_spec() -> None:
    spec = FactorSpec(name="mom", category="momentum")
    factor = Factor(spec=spec, raw=pd.DataFrame())
    assert factor.name == "mom"
    assert factor.category == "momentum"
    assert factor.notes == {}


def test_coverage_is_the_share_of_observed_cells() -> None:
    raw = pd.DataFrame([[1.0, np.nan], [1.0, 1.0]], index=pd.date_range("2024-01-01", periods=2))
    assert Factor(spec=FactorSpec(name="x", category="value"), raw=raw).coverage() == 0.75


def test_coverage_of_an_empty_frame_is_zero() -> None:
    factor = Factor(spec=FactorSpec(name="x", category="value"), raw=pd.DataFrame())
    assert factor.coverage() == 0.0


def test_coverage_of_an_all_nan_frame_is_zero() -> None:
    raw = pd.DataFrame(np.nan, index=pd.date_range("2024-01-01", periods=2), columns=["A"])
    assert Factor(spec=FactorSpec(name="x", category="value"), raw=raw).coverage() == 0.0


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def test_the_decorator_path_registers_and_returns_the_function() -> None:
    registry = FactorRegistry()

    @registry.register(FactorSpec(name="c", category="quality"))
    def fn(ctx: FactorContext) -> pd.DataFrame:
        return pd.DataFrame()

    assert "c" in registry
    assert registry.spec("c").category == "quality"
    assert registry._fns["c"] is fn


def test_the_direct_path_registers_and_returns_the_function() -> None:
    """`register(spec, fn)` is the non-decorator form; it must return `fn`."""
    registry = FactorRegistry()

    def fn(ctx: FactorContext) -> pd.DataFrame:
        return pd.DataFrame()

    assert registry.register(FactorSpec(name="a", category="value"), fn) is fn
    assert "a" in registry


def test_registering_a_name_twice_overwrites_silently() -> None:
    """Documented, not endorsed: a name collision drops a factor with no trace.

    Both paths write straight into the name-keyed dicts, so the second
    registration wins and the factor count does not change. Worth knowing when
    two modules pick the same name - nothing warns.
    """
    registry = FactorRegistry()
    registry.register(FactorSpec(name="dup", category="quality"), _constant())
    registry.register(FactorSpec(name="dup", category="momentum"), _constant())

    assert len(registry) == 1
    assert registry.spec("dup").category == "momentum"


# ----------------------------------------------------------------------
# Lookups
# ----------------------------------------------------------------------
def _populated() -> FactorRegistry:
    registry = FactorRegistry()
    for name, category in [("z", "value"), ("a", "value"), ("m", "momentum")]:
        registry.register(FactorSpec(name=name, category=category), _constant())
    return registry


def test_names_are_sorted() -> None:
    assert _populated().names() == ["a", "m", "z"]


def test_names_can_be_filtered_by_category() -> None:
    assert _populated().names("value") == ["a", "z"]
    assert _populated().names("momentum") == ["m"]


def test_an_unknown_category_yields_no_names() -> None:
    """Not an error: a caller asking about a category with no factors is normal."""
    assert _populated().names("reversal") == []


def test_specs_are_sorted_by_name() -> None:
    specs = _populated().specs()
    assert [s.name for s in specs] == ["a", "m", "z"]
    assert all(isinstance(s, FactorSpec) for s in specs)


def test_an_unknown_spec_raises_a_key_error() -> None:
    with pytest.raises(KeyError):
        _populated().spec("nope")


def test_membership_and_length() -> None:
    registry = _populated()
    assert "a" in registry
    assert "nope" not in registry
    assert len(registry) == 3
    assert len(FactorRegistry()) == 0


# ----------------------------------------------------------------------
# compute
# ----------------------------------------------------------------------
def test_compute_returns_a_factor_carrying_its_spec() -> None:
    registry = FactorRegistry()
    registry.register(FactorSpec(name="one", category="value"), _constant(2.0))
    factor = registry.compute("one", _ctx())
    assert factor.name == "one"
    assert (factor.raw.to_numpy() == 2.0).all()
    assert factor.coverage() == 1.0


def test_compute_rejects_an_unknown_factor() -> None:
    """A typo must surface, not silently produce an empty panel."""
    with pytest.raises(KeyError, match="Unknown factor"):
        _populated().compute("nope", _ctx())


def test_compute_degrades_on_factor_unavailable() -> None:
    """The registry catches FactorUnavailableError and hands back an empty panel."""
    registry = FactorRegistry()

    def unavailable(ctx: FactorContext) -> pd.DataFrame:
        raise FactorUnavailableError("no fundamentals here")

    registry.register(FactorSpec(name="needs_fund", category="quality"), unavailable)
    factor = registry.compute("needs_fund", _ctx())
    assert factor.raw.isna().all().all()
    assert factor.coverage() == 0.0


def test_compute_does_not_swallow_other_errors() -> None:
    """Only the documented unavailable error is caught; a bug must be visible."""
    registry = FactorRegistry()

    def broken(ctx: FactorContext) -> pd.DataFrame:
        raise RuntimeError("a genuine bug")

    registry.register(FactorSpec(name="broken", category="value"), broken)
    with pytest.raises(RuntimeError, match="a genuine bug"):
        registry.compute("broken", _ctx())


def test_compute_treats_an_empty_frame_as_unavailable() -> None:
    registry = FactorRegistry()
    registry.register(FactorSpec(name="empty", category="value"), lambda ctx: pd.DataFrame())
    assert registry.compute("empty", _ctx()).coverage() == 0.0


def test_compute_reindexes_onto_the_panel_grid() -> None:
    """A factor that returns a different grid must land on the panel's."""
    registry = FactorRegistry()

    def partial(ctx: FactorContext) -> pd.DataFrame:
        return pd.DataFrame({"B": [1.0, 2.0, 3.0]}, index=ctx.panel.dates)

    registry.register(FactorSpec(name="partial", category="value"), partial)
    raw = registry.compute("partial", _ctx()).raw
    assert list(raw.columns) == ["A", "B"]
    assert raw["A"].isna().all()
    assert raw["B"].notna().all()


def test_compute_handles_a_factor_returning_none() -> None:
    registry = FactorRegistry()
    registry.register(FactorSpec(name="none", category="value"), lambda ctx: None)
    assert registry.compute("none", _ctx()).coverage() == 0.0


# ----------------------------------------------------------------------
# compute_all
# ----------------------------------------------------------------------
def test_compute_all_defaults_to_every_registered_factor() -> None:
    registry = FactorRegistry()
    registry.register(FactorSpec(name="a", category="value"), _constant())
    registry.register(FactorSpec(name="b", category="value"), _constant())
    assert sorted(registry.compute_all(_ctx())) == ["a", "b"]


def test_compute_all_honours_an_explicit_name_list() -> None:
    registry = _populated()
    assert sorted(registry.compute_all(_ctx(), names=["a", "m"])) == ["a", "m"]


def test_compute_all_isolates_a_broken_factor() -> None:
    """One bad factor must not cost the run - it is logged and skipped."""
    registry = FactorRegistry()
    registry.register(FactorSpec(name="ok", category="value"), _constant())

    def broken(ctx: FactorContext) -> pd.DataFrame:
        raise RuntimeError("boom")

    registry.register(FactorSpec(name="broken", category="value"), broken)
    out = registry.compute_all(_ctx())
    assert sorted(out) == ["ok"]


def test_compute_all_drops_a_factor_with_no_observations() -> None:
    """A column of NaN must not reach the model as if it were a signal."""
    registry = FactorRegistry()
    registry.register(FactorSpec(name="ok", category="value"), _constant())
    registry.register(FactorSpec(name="blank", category="value"), _constant(np.nan))
    assert sorted(registry.compute_all(_ctx())) == ["ok"]


def test_compute_all_returns_factors_not_frames() -> None:
    out = _populated().compute_all(_ctx())
    assert all(isinstance(f, Factor) for f in out.values())


def test_compute_all_on_an_empty_registry_is_empty() -> None:
    assert FactorRegistry().compute_all(_ctx()) == {}


def test_compute_all_accepts_any_iterable_of_names() -> None:
    registry = _populated()
    assert sorted(registry.compute_all(_ctx(), names=("a", "z"))) == ["a", "z"]
    assert sorted(registry.compute_all(_ctx(), names=iter(["m"]))) == ["m"]


# ----------------------------------------------------------------------
# FactorContext
# ----------------------------------------------------------------------
def test_the_context_proxies_the_panel_frames() -> None:
    ctx = _ctx()
    assert ctx.close is ctx.panel.close
    assert ctx.returns is ctx.panel.returns
    assert ctx.market_cap is ctx.panel.market_cap


def test_require_fundamentals_raises_without_a_view() -> None:
    with pytest.raises(FactorUnavailableError, match="not available"):
        _ctx().require_fundamentals()


def test_require_fundamentals_raises_when_the_view_is_empty() -> None:
    ctx = FactorContext(_StubPanel(), fundamentals=SimpleNamespace(data=pd.DataFrame()))
    with pytest.raises(FactorUnavailableError, match="not available"):
        ctx.require_fundamentals()


def test_require_fundamentals_returns_the_view_when_it_has_data() -> None:
    view = SimpleNamespace(data=pd.DataFrame({"x": [1]}))
    ctx = FactorContext(_StubPanel(), fundamentals=view)
    assert ctx.require_fundamentals() is view


def test_empty_like_is_all_nan_on_the_panel_grid() -> None:
    raw = _empty_like(_ctx())
    assert raw.shape == (3, 2)
    assert raw.isna().all().all()
    assert list(raw.columns) == ["A", "B"]


# ----------------------------------------------------------------------
# The global registry
# ----------------------------------------------------------------------
def test_the_global_registry_is_populated_by_importing_the_factor_modules() -> None:
    import alphaforge.factors  # noqa: F401  - registration side effects

    assert len(REGISTRY) > 0
    assert REGISTRY.names() == sorted(REGISTRY.names())


def test_every_registered_factor_has_a_usable_spec() -> None:
    """The invariant the factor library and the report both rely on."""
    import alphaforge.factors  # noqa: F401

    for spec in REGISTRY.specs():
        assert spec.name
        assert spec.category in CATEGORIES
        assert spec.direction in (1, -1)
        assert spec.name in REGISTRY
        assert REGISTRY.spec(spec.name) is spec


def test_no_two_registered_factors_share_a_name() -> None:
    """Names are dict keys, so a collision would be invisible in the count."""
    import alphaforge.factors  # noqa: F401

    names = [spec.name for spec in REGISTRY.specs()]
    assert len(names) == len(set(names))


def test_the_module_level_register_targets_the_global_registry() -> None:
    assert register.__module__ == "alphaforge.factors.base"
    assert REGISTRY is not None
