# SPDX-License-Identifier: Apache-2.0

import os

from areal.utils import logging

logger = logging.getLogger("EnvironUtils")

_warned_bool_env_var_keys = set()
_warned_numeric_env_var_values = set()
_warned_rank_env_var_values = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1", "yes", "y", "on")
    falsy_values = ("false", "0", "no", "n", "off")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def get_float_env_var(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        warn_key = (name, value)
        if warn_key not in _warned_numeric_env_var_values:
            logger.warning("Invalid %s=%r; using %s", name, value, default)
            _warned_numeric_env_var_values.add(warn_key)
        return default


def get_int_env_var(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        warn_key = (name, value)
        if warn_key not in _warned_numeric_env_var_values:
            logger.warning("Invalid %s=%r; using %s", name, value, default)
            _warned_numeric_env_var_values.add(warn_key)
        return default


def is_in_ci():
    return get_bool_env_var("AREAL_IS_IN_CI")


def is_single_controller():
    return not get_bool_env_var("AREAL_SPMD_MODE")


def rank_in_env_filter(name: str, rank: int) -> bool:
    """Return whether rank is selected by a comma/range env filter.

    Empty or unset values mean all ranks. Accepted examples: ``0``, ``0,2``,
    ``0-3``, and ``all``.
    """

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return True

    value = value.strip().lower()
    if value == "all":
        return True

    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
        except ValueError:
            warn_key = (name, value)
            if warn_key not in _warned_rank_env_var_values:
                logger.warning(
                    "rank_in_env_filter(%s) got invalid value=%s and will ignore part=%s",
                    name,
                    value,
                    part,
                )
                _warned_rank_env_var_values.add(warn_key)

    return rank in selected
