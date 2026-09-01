"""A cap larger than a model's context makes every request fail and the run report acc=nan.

Three machines have now produced that non-result. These pin the clamp that prevents it, read
from the model's own config rather than from a rule anyone has to remember.
"""
import json
import sys

sys.path.insert(0, "experiments/bench")
import math_bench as M


def _write_cfg(tmp_path, **kw):
    (tmp_path / "config.json").write_text(json.dumps(kw))
    return str(tmp_path)


def test_the_context_is_read_from_the_model_config(tmp_path):
    assert M.model_context_limit(_write_cfg(tmp_path, max_position_embeddings=4096)) == 4096


def test_alternative_config_keys_are_understood(tmp_path):
    assert M.model_context_limit(_write_cfg(tmp_path, n_positions=2048)) == 2048


def test_an_unreadable_config_yields_None_not_a_small_number(tmp_path):
    """None must mean 'unknown'. Returning 0 or a default would clamp every model to nothing."""
    assert M.model_context_limit(str(tmp_path)) is None
    (tmp_path / "config.json").write_text("{not json")
    assert M.model_context_limit(str(tmp_path)) is None


def test_a_cap_beyond_the_context_is_clamped_with_a_reason():
    eff, why = M.clamp_max_tokens(32768, 4096)
    assert eff == 4096 - M.PROMPT_HEADROOM
    assert why and "acc=nan" in why, "the reason must name the failure it prevents"


def test_the_exact_case_that_failed_on_three_machines():
    """A 32768 cap against a 32768 context: fits numerically, fails in practice."""
    eff, why = M.clamp_max_tokens(32768, 32768)
    assert eff < 32768 and why is not None


def test_a_cap_inside_the_context_is_untouched():
    assert M.clamp_max_tokens(4096, 32768) == (4096, None)


def test_an_unknown_context_never_clamps():
    """Not knowing the context is not evidence that it is small."""
    assert M.clamp_max_tokens(65536, None) == (65536, None)
    assert M.clamp_max_tokens(65536, 0) == (65536, None)
