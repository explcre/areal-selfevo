"""One launch, one experiment-tracking identifier -- and a loud refusal when one is reused.

WHY THIS EXISTS. On 2026-09-02 A0 trained for hours while every metric it produced was
thrown away. ``StatsLoggerConfig.wandb.id_suffix`` defaults to the constant ``"train"``, so
the W&B run id is ``<experiment>_<trial>_train`` for EVERY launch of that experiment and
trial; combined with ``resume="allow"`` the fresh run RESUMED the previous one, whose step
counter was already far ahead. W&B answered all seventy of the run's commits with

    Tried to log to step N that is less than the current step M. Steps must be
    monotonically increasing, so this data will be ignored.

Every training metric and every periodic evaluation point was dropped. The on-disk
artefacts were untouched, which is exactly why it went unnoticed: from every side except
the one the project needs -- the online curve -- the run looked healthy.

TWO RULES, because "never resume" is the wrong fix. A resumed run is legitimate: AReaL's
recovery relaunches the same trial and its curve should continue rather than fork.

1. A launch that does not ASK to resume gets an identifier no earlier launch can have held
   -- the configured id with a launch-unique token appended. :func:`resolve_run_id`.
2. A launch that DOES ask names the identifier it is resuming, in
   :data:`ENV_RUN_ID`. Intent costs one environment variable; a collision costs nothing at
   all, which is why the default must not be the collision.

AND TWO CHECKS, because rule 1 is a hope until something fires when it is broken. Neither
is derived from the other and they fail at different moments:

* :func:`assert_id_is_fresh` runs at startup, right after the tracker is initialised, and
  refuses a launch that did not ask to resume but came back attached to history. It cannot
  false-positive: a genuinely fresh id has no step behind it.
* :func:`assert_step_advances` runs on the FIRST commit -- the first moment at which the
  step this run will write and the step the tracker sits at are both known -- and refuses
  any launch whose first write would land in the past, INCLUDING an intended resume that
  restarted from an earlier step. That is the case rule 2 cannot cover, because an intended
  resume is still silently dropped if it rewinds.

Both raise. This is a metrics-only failure and killing the run over it looks disproportionate
until you price it: the alternative already cost this project a four-hour run's entire
curve, and the run had to be restarted anyway to get one.
"""

from __future__ import annotations

import os
import time

__all__ = [
    "ENV_RUN_ID",
    "RunIdCollision",
    "assert_id_is_fresh",
    "assert_step_advances",
    "launch_token",
    "resolve_run_id",
]

#: The identifier to RESUME, when resuming is what the operator means. Set it and the id is
#: used verbatim; leave it unset and every launch gets its own.
ENV_RUN_ID = "SELFEVO_WANDB_RUN_ID"


class RunIdCollision(RuntimeError):
    """This launch is writing metrics into a run that is already ahead of it.

    Raised rather than warned. The failure it replaces does warn -- once per commit, in the
    tracker's own words, on a line that scrolled past seventy times in four hours while the
    run looked perfectly healthy.
    """


def launch_token(now: float | None = None, entropy: int | None = None) -> str:
    """A token no other launch of this experiment can hold.

    Time alone is not enough: two arms of a sweep can start inside the same second, and a
    relaunch after a crash routinely does. The process id disambiguates those without
    needing a random source a test cannot pin.

    Args:
        now: Unix time; defaults to now.
        entropy: A per-process number; defaults to this process' pid.

    Returns:
        A token like ``20260902T190501Z_50208``.
    """
    now = time.time() if now is None else now
    entropy = os.getpid() if entropy is None else entropy
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now)) + f"_{int(entropy) % 100000:05d}"


def resolve_run_id(
    experiment: str, trial: str, suffix: str = "", env=None, token: str | None = None
) -> tuple[str, bool]:
    """The tracker id this launch must use, and whether resuming is intended.

    Args:
        experiment: Experiment name.
        trial: Trial name.
        suffix: The configured id suffix, kept so existing ids keep their readable shape.
        env: Environment mapping; defaults to ``os.environ``.
        token: The launch-unique token; defaults to :func:`launch_token`.

    Returns:
        ``(run_id, intended_resume)``. ``intended_resume`` is True only when the operator
        named an id to resume, which is the single way to get a shared identifier.
    """
    env = os.environ if env is None else env
    pinned = (env.get(ENV_RUN_ID, "") or "").strip()
    if pinned:
        return pinned, True
    stem = "_".join(str(p) for p in (experiment, trial, suffix) if p)
    return f"{stem}_{launch_token() if token is None else token}", False


def assert_id_is_fresh(run_id: str, resumed_step: int, intended_resume: bool) -> None:
    """Refuse a launch that silently attached itself to an earlier run's step counter.

    Args:
        run_id: The identifier the tracker was initialised with.
        resumed_step: The step the tracker came back sitting at. Zero for a new run.
        intended_resume: Whether the operator asked to resume this id.

    Raises:
        RunIdCollision: If a launch that did not ask to resume resumed anyway.
    """
    if intended_resume or int(resumed_step) <= 0:
        return
    raise RunIdCollision(
        f"the tracker id {run_id!r} already exists and came back at step {resumed_step}, but "
        f"this launch did not ask to resume anything. Every metric this run writes below "
        f"step {resumed_step} will be silently discarded -- which is how A0 trained for "
        f"hours on 2026-09-02 with no curve at all. Either let the id be generated per "
        f"launch (unset {ENV_RUN_ID}) or say that resuming {run_id!r} is what you meant by "
        f"setting {ENV_RUN_ID}={run_id}."
    )


def assert_step_advances(run_id: str, resumed_step: int, log_step: int) -> None:
    """Refuse the first write of a launch whose metrics would land in the tracker's past.

    Deliberately independent of :func:`assert_id_is_fresh` and not derived from it: this one
    compares the step actually about to be WRITTEN against the step the tracker actually
    sits at, so it fires even when the id was resumed on purpose. An intended resume that
    rewinds drops metrics exactly as an accidental one does.

    The comparison is the tracker's own rule, quoted from its own warning -- a write to a
    step LESS THAN the current one is ignored; a write to the current step is merged into it
    -- so a fresh run (which sits at no step at all) is never refused, and neither is a
    resume that picks up exactly where it left off.

    Args:
        run_id: The identifier in use.
        resumed_step: The step the tracker sat at when this launch attached to it. Zero when
            this launch started the run, in which case there is no past to land in.
        log_step: The step of this launch's first commit.

    Raises:
        RunIdCollision: If the first write would land behind the tracker.
    """
    if int(resumed_step) <= 0 or int(log_step) >= int(resumed_step):
        return
    raise RunIdCollision(
        f"the first commit of this launch writes step {log_step} to tracker run {run_id!r}, "
        f"which is already at step {resumed_step}. A write to a step in the past is IGNORED "
        f"by the tracker, so this metric and every one until step {resumed_step} would be "
        f"discarded without failing anything. Resume from where the run actually is, or "
        f"unset {ENV_RUN_ID} so this launch gets an identifier of its own."
    )
