#!/usr/bin/env python3
"""Collapse repeated log lines so a retry storm costs bounded disk instead of O(retries).

Reads stdin, writes stdout. Two lines are "the same" if they match after digits and
hex blobs are replaced by placeholders. The first CAP occurrences of each signature pass
through verbatim; after that only a periodic tally is emitted.

Rationale: a previous run wrote 3.6 GB, of which 3,468,076 lines were one repeating
sglang 500. That volume destroyed the evidence it was supposed to preserve.

Audit fixes (see EXPERIMENTS.md):
  D2  progress lines are NEVER suppressed -- the stall watchdog reads that counter, and
      capping it made a healthy run look frozen from step 201 onward.
  D9  the signature key is length-bounded and the table has a safety valve, so a storm of
      unique-per-line messages cannot grow `seen` without bound.
  D11 the tally cadence is tight enough to be visible, and a SIGTERM handler writes the
      summary -- otherwise killing the process group loses it exactly when it matters.
  D13 a hex blob must contain a digit, so ordinary words ("facade", "deadbeef") are not
      collapsed into placeholders.
"""
from __future__ import annotations

import re
import signal
import sys

CAP = 200               # verbatim copies kept per distinct signature
TALLY_EVERY = 1000      # after CAP, emit a tally line every this many suppressed hits
MAX_SIG = 200           # signature key length bound (D9)
MAX_SIGS = 500_000      # safety valve on the signature table (D9)

# A hex blob must contain at least one digit, so "facade"/"decade" survive intact (D13).
_hexish = re.compile(r"\b(?:0x)?(?=[0-9a-fA-F]{6,}\b)[a-fA-F]*[0-9][0-9a-fA-F]*\b")
_digits = re.compile(r"[0-9]+")
_ansi = re.compile(r"\x1b\[[0-9;]*m")
# The liveness signal. Never suppressed, whatever its repeat count (D2).
_progress = re.compile(r"[Ss]tep [0-9]+/[0-9]+")

_seen: dict[str, int] = {}
_out = sys.stdout


def _summary(*_args: object) -> None:
    """Write the suppressed-signature tally. Runs at EOF and on SIGTERM (D11)."""
    try:
        for sig, n in sorted(_seen.items(), key=lambda kv: -kv[1])[:25]:
            if n > CAP:
                _out.write(f"[logfilter] TOTAL x{n}: {sig[:160].rstrip()}\n")
        _out.flush()
    except Exception:
        pass


def _on_term(signum: int, _frame: object) -> None:
    _summary()
    sys.exit(128 + signum)


def main() -> None:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    for line in sys.stdin:
        if _progress.search(line):          # D2: liveness always passes through
            _out.write(line)
            _out.flush()
            continue
        sig = _digits.sub("#", _hexish.sub("HEX", _ansi.sub("", line)))[:MAX_SIG]
        n = _seen.get(sig, 0) + 1
        if len(_seen) >= MAX_SIGS and sig not in _seen:
            _out.write(f"[logfilter] signature table hit {MAX_SIGS}; resetting\n")
            _seen.clear()
            n = 1
        _seen[sig] = n
        if n <= CAP:
            _out.write(line)
            _out.flush()
        elif n % TALLY_EVERY == 0:
            _out.write(f"[logfilter] suppressed x{n}: {sig[:120].rstrip()}\n")
            _out.flush()
    _summary()


if __name__ == "__main__":
    main()
