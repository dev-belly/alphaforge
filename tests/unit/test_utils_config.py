"""Offline tests for configuration loading, merging and seeding.

Every run reads this module, so a defect here is not local: it silently changes
what the whole platform is configured to do. The failure that matters is the
quiet one - an override that lands on a key nobody reads, an environment
variable whose dotted path is misspelt, a merge that mutates the base it was
given. None of those raise.

``test_every_env_var_maps_onto_a_real_setting`` is the guard for the first of
those: the ``ALPHAFORGE_*`` mapping is a hand-written list of dotted paths, and
a typo in it produces an environment variable that parses fine, merges fine, and
has no effect whatsoever.

The precedence is pinned too, because it is a decision rather than an accident:
an explicit ``overrides`` argument is merged first and the environment last, so
the environment wins.

Everything is deterministic: no network, tmp files only, and the environment is
restored by monkeypatch.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from alphaforge.utils.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    _coerce,
    _env_overrides,
    deep_get,
    deep_merge,
    load_yaml,
    set_global_seed,
)

ENV_MAPPING = {
    "ALPHAFORGE_DATA_PROVIDER": "data.provider",
    "ALPHAFORGE_MODEL_TYPE": "model.type",
    "ALPHAFORGE_PORTFOLIO_METHOD": "portfolio.method",
    "ALPHAFORGE_BACKTEST_REBALANCE": "backtest.rebalance",
    "ALPHAFORGE_LOG_LEVEL": "logging.level",
}


# ----------------------------------------------------------------------
# load_yaml
# ----------------------------------------------------------------------
def test_a_missing_file_is_an_error_not_an_empty_config() -> None:
    """Silently returning {} would run the whole platform on defaults."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml("/tmp/alphaforge-definitely-not-here.yaml")


def test_a_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml(path)


def test_an_empty_file_loads_as_an_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_yaml(path) == {}


def test_a_normal_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump({"model": {"type": "ridge", "params": {"alpha": 1.0}}}))
    assert load_yaml(path) == {"model": {"type": "ridge", "params": {"alpha": 1.0}}}


def test_the_default_config_exists_and_is_loadable() -> None:
    raw = load_yaml(DEFAULT_CONFIG_PATH)
    assert isinstance(raw, dict) and raw


# ----------------------------------------------------------------------
# deep_merge
# ----------------------------------------------------------------------
def test_deep_merge_recurses_into_nested_mappings() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    assert deep_merge(base, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}, "d": 3}


def test_deep_merge_does_not_mutate_the_base() -> None:
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}, "c": 3})
    assert base == {"a": {"b": 1}}


def test_a_scalar_overrides_a_mapping_and_vice_versa() -> None:
    """A type change replaces rather than trying to merge."""
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert deep_merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}


def test_merging_nothing_is_a_no_op() -> None:
    base = {"a": {"b": 1}}
    assert deep_merge(base, {}) == base
    assert deep_merge(base, None) == base


def test_new_keys_are_added() -> None:
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ----------------------------------------------------------------------
# deep_get
# ----------------------------------------------------------------------
def test_deep_get_walks_a_dotted_path() -> None:
    cfg = {"a": {"b": {"c": 1}}}
    assert deep_get(cfg, "a.b.c") == 1
    assert deep_get(cfg, "a.b") == {"c": 1}


@pytest.mark.parametrize("path", ["a.x", "x.y", "a.b.c.d", ""])
def test_deep_get_falls_back_to_the_default(path: str) -> None:
    assert deep_get({"a": {"b": {"c": 1}}}, path, "MISSING") == "MISSING"


@pytest.mark.parametrize("path", ["scalar.x", "list.0"])
def test_deep_get_stops_at_a_non_mapping_intermediate(path: str) -> None:
    """Indexing into a list is not supported; say so with the default."""
    assert deep_get({"scalar": 5, "list": [1, 2]}, path, "MISSING") == "MISSING"


def test_deep_get_returns_none_by_default() -> None:
    assert deep_get({}, "anything") is None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def test_config_get_uses_the_dotted_path() -> None:
    cfg = Config(raw={"model": {"type": "ridge"}})
    assert cfg.get("model.type") == "ridge"
    assert cfg.get("model.params", {}) == {}


def test_config_load_reads_the_default_file() -> None:
    cfg = Config.load()
    assert cfg.get("data.provider")
    assert cfg.get("model.type")


def test_config_load_accepts_a_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"model": {"type": "elasticnet"}}))
    assert Config.load(path).get("model.type") == "elasticnet"


def test_explicit_overrides_are_applied() -> None:
    cfg = Config.load(overrides={"model": {"type": "random_forest"}})
    assert cfg.get("model.type") == "random_forest"
    # ...and the rest of the section survives the merge.
    assert "params" in cfg.section("model")


def test_the_environment_beats_an_explicit_override(monkeypatch) -> None:
    """Precedence is a decision: overrides merge first, the environment last."""
    monkeypatch.setenv("ALPHAFORGE_MODEL_TYPE", "elasticnet")
    cfg = Config.load(overrides={"model": {"type": "random_forest"}})
    assert cfg.get("model.type") == "elasticnet"


def test_section_returns_a_copy_of_a_mapping() -> None:
    cfg = Config(raw={"model": {"type": "ridge"}})
    section = cfg.section("model")
    section["type"] = "MUTATED"
    assert cfg.get("model.type") == "ridge"


@pytest.mark.parametrize("name", ["missing", ""])
def test_section_of_an_absent_key_is_empty(name: str) -> None:
    assert Config(raw={"model": {}}).section(name) == {}


def test_section_of_a_scalar_is_empty_not_a_type_error() -> None:
    assert Config(raw={"x": 5}).section("x") == {}


def test_to_dict_is_a_deep_copy() -> None:
    cfg = Config(raw={"model": {"params": {"alpha": 1.0}}})
    out = cfg.to_dict()
    out["model"]["params"]["alpha"] = 99.0
    assert cfg.get("model.params.alpha") == 1.0


def test_an_empty_config_is_valid() -> None:
    cfg = Config()
    assert cfg.get("anything", "fallback") == "fallback"
    assert cfg.to_dict() == {}


# ----------------------------------------------------------------------
# Environment overrides
# ----------------------------------------------------------------------
def test_every_env_var_maps_onto_a_real_setting() -> None:
    """A misspelt path yields an env var that parses, merges and does nothing."""
    raw = load_yaml(DEFAULT_CONFIG_PATH)
    sentinel = object()
    for env_key, dotted in ENV_MAPPING.items():
        assert deep_get(raw, dotted, sentinel) is not sentinel, (
            f"{env_key} maps to {dotted!r}, which is not in the default config"
        )


def test_env_overrides_build_the_nested_tree(monkeypatch) -> None:
    for key in ENV_MAPPING:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPHAFORGE_MODEL_TYPE", "elasticnet")
    monkeypatch.setenv("ALPHAFORGE_LOG_LEVEL", "DEBUG")
    got = _env_overrides()
    assert got == {"model": {"type": "elasticnet"}, "logging": {"level": "DEBUG"}}


def test_unset_env_vars_contribute_nothing(monkeypatch) -> None:
    for key in ENV_MAPPING:
        monkeypatch.delenv(key, raising=False)
    assert _env_overrides() == {}


def test_every_mapped_env_var_reaches_the_config(monkeypatch) -> None:
    for key in ENV_MAPPING:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPHAFORGE_DATA_PROVIDER", "sample")
    monkeypatch.setenv("ALPHAFORGE_MODEL_TYPE", "elasticnet")
    monkeypatch.setenv("ALPHAFORGE_PORTFOLIO_METHOD", "equal_weight")
    monkeypatch.setenv("ALPHAFORGE_BACKTEST_REBALANCE", "weekly")
    monkeypatch.setenv("ALPHAFORGE_LOG_LEVEL", "DEBUG")
    cfg = Config.load()
    for env_key, dotted in ENV_MAPPING.items():
        assert cfg.get(dotted) == os.environ[env_key], f"{env_key} did not land on {dotted}"


# ----------------------------------------------------------------------
# _coerce
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("42", 42),
        ("-7", -7),
        (" 42 ", 42),
        ("007", 7),
        ("3.14", 3.14),
        ("1e3", 1000.0),
        ("sample", "sample"),
        ("yes", "yes"),
        ("", ""),
    ],
)
def test_coerce_maps_a_string_to_the_obvious_type(raw: str, expected) -> None:
    got = _coerce(raw)
    assert got == expected
    assert type(got) is type(expected)


def test_coerce_prefers_int_over_float() -> None:
    """A count written as "60" must not become 60.0 and leak into a format string."""
    assert isinstance(_coerce("60"), int)
    assert isinstance(_coerce("60.0"), float)


def test_a_bool_string_never_becomes_a_number() -> None:
    """``int("true")`` would fail anyway, but the order is what makes it explicit."""
    assert _coerce("true") is True
    assert _coerce("false") is False


# ----------------------------------------------------------------------
# set_global_seed
# ----------------------------------------------------------------------
def test_setting_the_seed_makes_the_process_reproducible() -> None:
    set_global_seed(7)
    first = (random.random(), float(np.random.rand()))
    set_global_seed(7)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_a_different_seed_gives_a_different_draw() -> None:
    set_global_seed(1)
    first = float(np.random.rand())
    set_global_seed(2)
    second = float(np.random.rand())
    assert first != second


def test_setting_the_seed_records_the_hash_seed() -> None:
    """Harmless here (it must be set before start-up to bite) but recorded."""
    set_global_seed(11)
    assert os.environ["PYTHONHASHSEED"] == "11"


def test_the_default_seed_is_42() -> None:
    set_global_seed()
    assert os.environ["PYTHONHASHSEED"] == "42"
