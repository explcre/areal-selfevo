"""Judges producing V, C, F and H.

THIS IS THE LARGEST RECONSTRUCTION GAP IN THE WHOLE LOOP (ambiguity A9).

The method page names the four factors -- validity, alignment, reward fidelity, hack
resistance -- but defines none of them and supplies no evaluator for any of them. Every
number this module produces is therefore conditional on OUR rubrics, not on Ornith's, and
no result computed with it is evidence about Ornith's models.

The judges are FROZEN and never trained (ambiguity A11). Training them on the same
rollouts that produce the rewards would close the last loop and make reward hacking
non-identifiable even in principle: if every check is a function of the endogenous
transcript, two worlds with opposite ground-truth labels for a shortcut have identical
distributions for everything the loop can observe.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import Rollout, RolloutOutcome, Scaffold, Task


class Judges:
    """Deterministic rubric judges. Frozen by construction: no update method exists.

    Args:
        min_task_words: Tasks shorter than this are treated as invalid.
        banned_tokens: Substrings that make a task invalid (e.g. a leaked answer).
    """

    def __init__(
        self,
        min_task_words: int = 4,
        banned_tokens: Sequence[str] = ("ANSWER:",),
    ) -> None:
        self.min_task_words = min_task_words
        self.banned_tokens = tuple(banned_tokens)

    def validity(self, task: Task, scaffold: Scaffold, mode: str = "soft") -> float:
        """V(q,s): is the task well-formed and self-contained?

        Args:
            task: The task.
            scaffold: The scaffold (unused by this rubric; present to match the published
                signature V(q,s)).
            mode: "soft" returns a graded value in [0,1]; "gate" returns 0.0 or 1.0 (A4).

        Returns:
            Validity in [0,1].

        Raises:
            ValueError: if `mode` is unknown.
        """
        if mode not in ("soft", "gate"):
            raise ValueError(f"mode must be soft|gate, got {mode}")
        words = task.text.split()
        leaked = any(b in task.text for b in self.banned_tokens)
        long_enough = len(words) >= self.min_task_words
        if mode == "gate":
            return 1.0 if (long_enough and not leaked) else 0.0
        score = 1.0
        if not long_enough:
            score *= len(words) / max(1, self.min_task_words)
        if leaked:
            score *= 0.0
        return max(0.0, min(1.0, score))

    def alignment(self, task: Task, scaffold: Scaffold) -> float:
        """C(q,h): does the scaffold address this task rather than a generic one?"""
        tq = set(task.text.lower().split())
        th = set(scaffold.instructions.lower().split())
        if not tq:
            return 0.0
        overlap = len(tq & th) / len(tq)
        return max(0.0, min(1.0, overlap))

    def reward_fidelity(self, scaffold: Scaffold, rollouts: Sequence[Rollout]) -> float:
        """F(h,{tau_i}): does the scaffold's grading discriminate among rollouts?

        A grader that returns the same verdict for every rollout carries no information,
        so fidelity is the fraction of the maximum possible discrimination achieved. This
        is our operationalisation, not Ornith's.
        """
        graded = [r for r in rollouts if r.outcome is not RolloutOutcome.ABORTED]
        if len(graded) < 2:
            return 0.0
        n_succ = sum(1 for r in graded if r.outcome is RolloutOutcome.SUCCESS)
        frac = n_succ / len(graded)
        # 1.0 at a 50/50 split, 0.0 when unanimous.
        return 1.0 - abs(2.0 * frac - 1.0)

    def hack_resistance(self, scaffold: Scaffold) -> float:
        """H(h): does the scaffold avoid trivially gameable grading?

        Penalises graders that accept any non-empty output. Our rubric; Ornith gives none.
        """
        return 1.0 if scaffold.grader_kind == "exact" else 0.3
