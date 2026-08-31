"""Adapter for mini-swe-agent, the first concrete harness.

Chosen over SWE-agent, OpenHands and deepseek-harness for one reason: its whole agent is a
190-line ``DefaultAgent`` whose ``AgentConfig`` IS the harness. A harness we need many
VARIANTS of has to be one we can actually vary, and a 453K-line runtime is not that. The
others remain addable -- :mod:`selfevo.harness.base` is what they would implement, and nothing
here is imported by the routing code.

**This measures; it does not train.** On math the solve rate falls out of the reward function.
On agentic software tasks there is no reward without an agent loop, so the first question an
adapter answers is whether the composition law measured on math -- silence governed by solve
rate, majority-unsolved when the solve rate is low -- also holds where solve rates are low by
construction. That is a measurement, and it is cheaper and more informative than trying to
beat a leaderboard with a 1.5B model.

mini-swe-agent is imported lazily and is NOT a dependency of this package; without it the
adapter raises with the install command rather than failing at import time and taking the
routing tests with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from selfevo.harness.base import HarnessRollout, HarnessVariant

__all__ = ["MiniSweAdapter", "HarnessUnavailable"]


class HarnessUnavailable(ImportError):
    """Raised when mini-swe-agent is not importable, naming how to get it."""


@dataclass
class MiniSweAdapter:
    """Runs one SWE-bench-style instance under one variant.

    Args:
        model_name: Model id passed to mini-swe-agent's model layer.
        env_kind: ``"local"`` or ``"docker"``. Docker is the honest choice for SWE-bench --
            local execution lets an agent see files the benchmark assumes it cannot -- but
            local is useful for a dry run that only exercises this adapter.
        repo_path: Where the mini-swe-agent checkout lives, added to ``sys.path`` if it is not
            installed.
        timeout_s: Wall-clock ceiling per rollout. An agent that hangs must not hang the sweep.

    Raises:
        ValueError: If ``env_kind`` is not recognised.
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    env_kind: str = "docker"
    repo_path: str = "~/baselines/mini-swe-agent/src"
    timeout_s: float = 1800.0
    _variants: tuple[HarnessVariant, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.env_kind not in ("local", "docker"):
            raise ValueError(f"env_kind must be local or docker, got {self.env_kind!r}")
        if not self._variants:
            from selfevo.harness.base import VARIANTS

            object.__setattr__(self, "_variants", tuple(VARIANTS.values()))

    def variants(self) -> tuple[HarnessVariant, ...]:
        """Variants this adapter can run."""
        return self._variants

    def _load(self):
        """Import mini-swe-agent, or explain how to get it."""
        import os
        import sys

        p = os.path.expanduser(self.repo_path)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
        try:
            from minisweagent.agents.default import AgentConfig, DefaultAgent
        except ImportError as exc:
            raise HarnessUnavailable(
                f"mini-swe-agent is not importable ({exc}). "
                "git clone https://github.com/SWE-agent/mini-swe-agent, or set repo_path to "
                "an existing checkout's src/ directory."
            ) from exc
        return DefaultAgent, AgentConfig

    def run(self, task_id: str, variant: HarnessVariant) -> HarnessRollout:
        """Attempt one instance and report the outcome.

        Errors are reported as ``error``, never as ``solved=False``. An agent that could not
        run is not an agent that failed the task, and scoring infrastructure failures as
        reward 0 is a mistake this project has already made and had to retract.

        Args:
            task_id: SWE-bench instance id.
            variant: Which harness configuration to run under.

        Returns:
            A :class:`HarnessRollout`. ``truncated`` is True when the step limit was reached,
            which on agentic tasks is far more common than on math and is the signal
            ``truncated_fraction`` exists to route on.
        """
        started = time.monotonic()
        try:
            DefaultAgent, AgentConfig = self._load()
        except HarnessUnavailable as exc:
            return HarnessRollout(task_id, variant.name, False, 0, error=str(exc))

        try:
            env = self._make_env(task_id)
            # Start from THEIR shipped config -- AgentConfig requires system_template and
            # instance_template, and inventing our own prompts would make every comparison
            # against published mini-swe-agent numbers meaningless. A variant overrides
            # fields on top of it; that is precisely what "a harness variant" means here.
            cfg = dict(self._base_config())
            cfg["step_limit"] = variant.step_limit
            cfg.update(variant.settings)
            agent = DefaultAgent(self._make_model(), env, **cfg)
            result = agent.run(task=self._task_text(task_id))
            steps = int(result.get("n_steps", getattr(agent, "n_steps", 0)) or 0)
            return HarnessRollout(
                task_id=task_id,
                variant=variant.name,
                solved=bool(result.get("submitted") and result.get("exit_status") == "submitted"),
                steps=steps,
                truncated=steps >= variant.step_limit,
                cost=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - any harness failure is reported, not scored
            return HarnessRollout(
                task_id=task_id,
                variant=variant.name,
                solved=False,
                steps=0,
                error=f"{type(exc).__name__}: {exc}",
                cost=time.monotonic() - started,
            )


    def _base_config(self) -> dict:
        """The agent settings from mini-swe-agent's own shipped config.

        Returns:
            The ``agent`` section of ``mini_textbased.yaml``.

        Raises:
            HarnessUnavailable: If the config cannot be found, since running without their
                templates would silently produce a different agent than the one whose numbers
                are published.
        """
        import os
        import pathlib as _pl

        import yaml

        root = _pl.Path(os.path.expanduser(self.repo_path)) / "minisweagent" / "config"
        f = root / "mini_textbased.yaml"
        if not f.exists():
            raise HarnessUnavailable(
                f"{f} not found. Point repo_path at a mini-swe-agent checkout's src/ "
                "directory; running without their templates would be a different agent."
            )
        data = yaml.safe_load(f.read_text()) or {}
        return data.get("agent", data)

    # The three seams a different benchmark or model layer would replace.
    def _make_env(self, task_id: str):
        """Execution environment for one instance."""
        DefaultAgent, _ = self._load()  # ensures sys.path is set
        if self.env_kind == "docker":
            from minisweagent.environments.docker import DockerEnvironment

            return DockerEnvironment(image=self._image_for(task_id))
        from minisweagent.environments.local import LocalEnvironment

        return LocalEnvironment()

    def _make_model(self):
        """Model wrapper mini-swe-agent will call."""
        from minisweagent.models import get_model

        return get_model(self.model_name)

    def _image_for(self, task_id: str) -> str:
        """SWE-bench publishes one image per instance under a stable naming scheme."""
        return f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_')}:latest"

    def _task_text(self, task_id: str) -> str:
        """Problem statement for the instance, read from the benchmark."""
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        for row in ds:
            if row["instance_id"] == task_id:
                return row["problem_statement"]
        raise KeyError(f"instance {task_id!r} not in SWE-bench_Verified")
