#!/bin/bash
# Behavioural test for the recovery injection, driving the REAL block extracted from
# supervise.sh rather than a restatement of it: a test that restates the logic passes even when
# the shipped file is reverted, which has already happened once in this repo.
set -u
F=/home/ubuntu/areal-selfevo/experiments/harness/supervise.sh
# Extract the shipped case statement and run it verbatim.
BLOCK="$(sed -n '/^  case " \${EXTRA_ARGS:-} " in/,/^  esac/p' "$F")"
[ -n "$BLOCK" ] || { echo "FAIL: could not extract the block from $F"; exit 1; }

check(){ # input  expected-substring-or-NONE
  local out
  out="$(EXTRA_ARGS="$1"; eval "$BLOCK"; echo "${EXTRA_ARGS:-}")"
  if [ "$2" = "NOAUTO" ]; then
    case "$out" in *"recover.mode=auto"*) echo "FAIL  [$1] -> [$out] (should NOT add auto)"; return 1;; esac
    echo "ok    [$1] -> [$out]"; return 0
  fi
  case "$out" in *"$2"*) echo "ok    [$1] -> [$out]"; return 0;; esac
  echo "FAIL  [$1] -> [$out] (expected $2)"; return 1
}
rc=0
check ""                        "recover.mode=auto" || rc=1   # the case that was broken
check "foo=1"                   "recover.mode=auto" || rc=1   # preserves caller args
check "recover.mode=off"        "NOAUTO"            || rc=1   # explicit off must survive
check "a=1 recover.mode=auto"   "recover.mode=auto" || rc=1   # no duplicate injection
n=$(EXTRA_ARGS="a=1 recover.mode=auto"; eval "$BLOCK"; echo "$EXTRA_ARGS"|grep -o "recover.mode=" |wc -l)
[ "$n" = "1" ] && echo "ok    injected exactly once (n=$n)" || { echo "FAIL duplicate injection n=$n"; rc=1; }
exit $rc
