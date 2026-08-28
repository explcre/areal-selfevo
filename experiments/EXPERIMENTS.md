
## 2026-08-28 - Step 0 rerun (step0b), and two findings that killed a planned fix

### Finding: `gconfig.min_new_tokens` is a DEAD FIELD in AReaL
Declared at `areal/api/cli_args.py:173` and listed as OpenAI-unsupported at :305, but
**never read anywhere else in the repo**:
    grep -rn "min_new_tokens" --include=*.py .   # -> only those two lines
I had planned to set `min_new_tokens=1` to prevent a degenerate all-EOS generation.
That fix would have been completely inert. Always grep for a config fields *reads*

## 2026-08-28 — Step 0 rerun (step0b), and two findings that killed a planned fix

### Finding: `gconfig.min_new_tokens` is a DEAD FIELD in AReaL
Declared at `areal/api/cli_args.py:173` and listed as OpenAI-unsupported at :305, but
**never read anywhere else in the repo**:

    grep -rn "min_new_tokens" --include=*.py .   # -> only those two lines

I had planned to set `min_new_tokens=1` to prevent a degenerate all-EOS generation from
500-ing the sglang server. That fix would have been completely inert. Always grep for a
config field's *reads* outside its own schema before building a run on it.

### Finding: the flash-attn wheel is ABI-incompatible with torch 2.9.1

    ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
    _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb

`STEP0_FAILURE.md` claimed the fix for the Step 0 stall was "install flash-attn and stop
deviating from the reference config". That was wrong on the merits: the stall was in
generation/retry handling, which `attn_impl` does not touch. sdpa stays, and the deviation
is documented in `step0b.sh` rather than mistaken for the bug.

### Evidence loss
The original 3.6 GB `step0.log` is now 0 bytes; only a 2 MB head survives. Grepping that
head for errors yields only false matches ("5.000e+00" contains the substring "500"). It
shows PPO metric tables through `step 51/233`, i.e. training *was* stepping. The stall
diagnosis therefore rests on evidence that no longer exists, so step0b re-derives the
failure instead of assuming it.

### step0b guards
- `experiments/harness/logfilter.py` collapses repeated log signatures (first 200 verbatim,
  then a periodic tally). A retry storm now costs O(1) disk instead of the ~3.46M lines that
  destroyed the last run's evidence.
- `experiments/harness/watchdog.sh` samples the `step N/233` counter TWICE, 1800s apart, and
  kills only after two consecutive no-progress strikes. Sampling a monotonic counter once
  cannot distinguish progress from a stall — that is how the last run was mis-reported as
  healthy. It kills by recorded PGID, never by pgrep pattern: a pattern kill previously
  matched the watcher's own command line and killed the controlling SSH session.
- `gconfig.max_new_tokens` is no longer overridden to 512 (that truncated chain-of-thought
  and read out as a 0 solve rate); the published 1024 stands.
