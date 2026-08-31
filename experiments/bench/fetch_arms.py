#!/usr/bin/env python3
"""Download the A/B arms' comparison checkpoints from HF so they can be scored locally.

The arms trained on the rented H200; scoring runs wherever GPUs are free. HF is already the
backup path, so it is also the transfer path -- no second mechanism to get wrong.
"""
from __future__ import annotations

import os
import pathlib
import sys

from huggingface_hub import snapshot_download

REPO = "ryan-superman/selfevo-areal-checkpoints"
STEPS = ("28", "57", "86", "115", "144")
DEST = pathlib.Path(os.path.expanduser("~/ab_ckpts"))


def main() -> int:
    """Fetch each arm into ``~/ab_ckpts/<arm>/…globalstepN`` and report what arrived."""
    token = open(os.path.expanduser("~/.hf_token_ryan.txt")).read().strip()
    for arm, prefix in (("on", "step0m-on-h200-1p5b"), ("off", "step0m-off-h200-1p5b")):
        out = DEST / arm
        out.mkdir(parents=True, exist_ok=True)
        pats = [f"{prefix}/*globalstep{s}/*" for s in STEPS]
        try:
            snapshot_download(REPO, repo_type="model", token=token, local_dir=str(out),
                              allow_patterns=pats, max_workers=8)
        except Exception as exc:  # noqa: BLE001 - a partial arm is still usable
            print(f"{arm}: download error {type(exc).__name__}: {str(exc)[:100]}")
        got = sorted(
            d.name.split("globalstep")[-1]
            for d in (out / prefix).glob("*globalstep*") if d.is_dir()
        )
        print(f"{arm}: {len(got)} checkpoints -> {sorted(got, key=int)}  at {out / prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
