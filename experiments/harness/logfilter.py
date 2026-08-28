#!/usr/bin/env python3
"""Collapse repeated log lines so a retry storm costs O(1) disk instead of O(retries).

Reads stdin, writes stdout. Two lines are "the same" if they match after digits and
hex blobs are replaced by placeholders. The first CAP occurrences of each signature
pass through verbatim; after that only a periodic "(xN more)" tally is emitted.
Rationale: the previous Step 0 run wrote 3.6 GB, ~3.46M lines of which were one
repeating error. That volume destroyed the evidence it was supposed to preserve.
"""
import re, sys

CAP = 200            # verbatim copies kept per distinct signature
TALLY_EVERY = 100000  # after CAP, emit a tally line every this many suppressed hits
_num = re.compile(r"[0-9a-fA-F]{6,}|[0-9]+")
_ansi = re.compile(r"\x1b\[[0-9;]*m")

def main() -> None:
    seen: dict[str, int] = {}
    out = sys.stdout
    for line in sys.stdin:
        sig = _num.sub("#", _ansi.sub("", line))
        n = seen.get(sig, 0) + 1
        seen[sig] = n
        if n <= CAP:
            out.write(line)
        elif n % TALLY_EVERY == 0:
            out.write(f"[logfilter] signature suppressed x{n}: {sig[:120].rstrip()}\n")
        if n <= CAP or n % TALLY_EVERY == 0:
            out.flush()
    for sig, n in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
        if n > CAP:
            out.write(f"[logfilter] TOTAL x{n}: {sig[:160].rstrip()}\n")
    out.flush()

if __name__ == "__main__":
    main()
