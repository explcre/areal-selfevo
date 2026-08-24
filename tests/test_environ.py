# SPDX-License-Identifier: Apache-2.0

from areal.utils.environ import (
    get_bool_env_var,
    get_float_env_var,
    get_int_env_var,
)


def test_typed_env_helpers_parse_supported_values(monkeypatch):
    monkeypatch.setenv("AREAL_TEST_BOOL", "on")
    monkeypatch.setenv("AREAL_TEST_FLOAT", "1.25")
    monkeypatch.setenv("AREAL_TEST_INT", "7")

    assert get_bool_env_var("AREAL_TEST_BOOL") is True
    assert get_float_env_var("AREAL_TEST_FLOAT", 0.0) == 1.25
    assert get_int_env_var("AREAL_TEST_INT", 0) == 7


def test_typed_env_helpers_use_defaults_for_invalid_values(monkeypatch):
    monkeypatch.setenv("AREAL_TEST_BOOL", "maybe")
    monkeypatch.setenv("AREAL_TEST_FLOAT", "many")
    monkeypatch.setenv("AREAL_TEST_INT", "several")

    assert get_bool_env_var("AREAL_TEST_BOOL", "true") is False
    assert get_float_env_var("AREAL_TEST_FLOAT", 2.5) == 2.5
    assert get_int_env_var("AREAL_TEST_INT", 3) == 3
