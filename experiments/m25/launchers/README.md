# Launchers for the M25 arms

`~/harness4` on the H100 box is **not a git repository**, and `run_a0.sh` plus
`preflight_a0.py` are the files that decide *which arm actually runs*. An unversioned edit to
either is an unrecorded change to the experiment, and this project has already been bitten
once by an arm that carried the method's name and ran the baseline.

## Enforced copies

`run_a0.sh` and `preflight_a0.py` here are **enforced**, not merely archived.
`preflight_a0.py` sha256s the live file at `~/harness4/<name>` against the copy in this
directory and **refuses the launch** when they differ, naming both hashes and the fix. So:

* edit the live file, run it, and the preflight will refuse until you copy it here and commit;
* a copy here that nobody kept in sync cannot silently be the wrong one, because the run stops.

Verified 2026-09-03: the check reads `ok` when the two agree and `FAIL` (rc=1, launch refused)
after a one-line drift, with the live file restored bit-identical afterwards.

## Snapshots, not enforced

`run_a0_pe.sh` and `trainer_entry.py` are point-in-time copies taken at the same moment, for
the record only. Nothing compares them against the live files, so treat them as evidence of
what ran on 2026-09-03 rather than as the current truth. If you come to depend on either,
add it to the tuple in `preflight_a0.py` so it is enforced too.

## The arm switch

`TRUNC_ADV` selects `actor.truncated_advantage` (`keep` = the A0 baseline and the default,
`zero`, `exclude`) and `WANDB_GROUP` names the run's group. Both default to what A0 already
ran, so this launcher with no environment set is still the baseline. The preflight refuses if
the resolved config does not carry the arm the launcher asked for.
