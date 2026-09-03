"""The task buffer `B` used by the novelty term.

The method page calls `B` a buffer of previously generated or trained-on tasks. It does
not state a capacity, a retention rule, or an insertion order (ambiguity A8). Ours is
append-only and unbounded, and the loop inserts a task only *after* it has been scored,
because inserting first makes the task its own nearest neighbour and forces `N = 0`
(guard G5).
"""

from __future__ import annotations

from collections.abc import Iterator

from .types import Task


class TaskBuffer:
    """Append-only buffer of tasks.

    Args:
        capacity: Optional cap. `None` (default) means unbounded, matching our A8 choice.
            When set, the oldest task is evicted first; the eviction is recorded so an
            analysis can tell a bounded run from an unbounded one.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")
        self.capacity = capacity
        self._tasks: list[Task] = []
        self.evictions = 0

    def add(self, task: Task) -> None:
        """Append a task, evicting the oldest if a capacity is set and reached."""
        self._tasks.append(task)
        if self.capacity is not None and len(self._tasks) > self.capacity:
            self._tasks.pop(0)
            self.evictions += 1

    def texts(self) -> list[str]:
        """Return the task texts, which are what `sim` compares."""
        return [t.text for t in self._tasks]

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)
