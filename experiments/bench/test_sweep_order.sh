#!/usr/bin/env bash
# Regression test for sweep_entropy.sh's job discovery.
#
# The guard under test is `sort -n`. With a lexical sort the step0d series comes out as
# gs115 gs144 gs173 gs028 gs057 gs086 -- every plot and every regression against step count
# would be silently wrong, with no error anywhere. Two-digit-only runs cannot detect this,
# so the test pins the three-digit step0d series specifically.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SW="$HERE/sweep_entropy.sh"
fail=0

got=$(CKPT_ROOT=$HOME/areal-runs/checkpoints/ubuntu/step0d/t1/default LIST=1 \
      bash "$SW" | grep -o "gs[0-9]*" | tr "\n" " ")
want="gs028 gs057 gs086 gs115 gs144 gs173 "
if [ "$got" != "$want" ]; then
    echo "FAIL order: got [$got] want [$want]"; fail=1
else
    echo "ok   step0d discovered in numeric step order"
fi

# The base anchor must always lead: without it a flat series cannot be read.
first=$(CKPT_ROOT=$HOME/areal-runs/checkpoints/ubuntu/step0d/t1/default LIST=1 \
        bash "$SW" | sed -n "2p" | cut -d: -f1)
if [ "$first" != "base" ]; then
    echo "FAIL anchor: first job is '$first', not 'base'"; fail=1
else
    echo "ok   base anchor leads the series"
fi

# An empty root must refuse rather than run a base-only sweep that looks like a result.
if CKPT_ROOT=/tmp/no_such_ckpt_dir_$$ LIST=1 bash "$SW" >/dev/null 2>&1; then
    echo "FAIL guard: empty checkpoint root did not refuse"; fail=1
else
    echo "ok   empty checkpoint root refuses"
fi

exit $fail
