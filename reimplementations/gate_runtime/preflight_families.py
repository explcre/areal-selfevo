#!/usr/bin/env python3
"""Prove the serving backend applies EACH module family, not just the ones it recognises first.

The coverage assertion in the trainer measures the adapter as PEFT built it. It cannot see what
the serving backend does with it. sglang maps PEFT module names onto its own fused buffers
(``q/k/v_proj -> qkv_proj``, ``gate/up_proj -> gate_up_proj``, ``linear_attn.out_proj ->
out_proj``) and a family it cannot map is not an error -- the adapter still loads and still
changes the output, because the other families did apply. That is a silent partial
application, and it would leave 48 of 64 layers untrained while every check upstream passed.

So each family is loaded ALONE, with LoRA-B non-zero only there, and asserted to change the
output by itself.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys

import aiohttp
import requests
from transformers import AutoTokenizer

sys.path.insert(0, "/mnt/localssd/gate/code")
from gate_lib import GenSpec, build_prompt, one_generation  # noqa: E402

BASE = r"model\.language_model\.layers\.[0-9]+\."
FAMILIES = {
    # name: (regex over module names, the sglang target list to declare, layers it should reach)
    "full_attn_qkvo": (BASE + r"self_attn\.(q_proj|k_proj|v_proj|o_proj)",
                       "q_proj,k_proj,v_proj,o_proj", 16),
    "linear_attn_out": (BASE + r"linear_attn\.out_proj", "out_proj", 48),
    "mlp": (BASE + r"mlp\.(gate_proj|up_proj|down_proj)", "gate_proj,up_proj,down_proj", 64),
}


async def texts(url: str, prompt: str, names: list[str]) -> dict:
    """Greedy completion for the base route and for each named adapter."""
    out = {}
    g = dict(max_new_tokens=32, temperature=0.0, top_p=1.0)
    async with aiohttp.ClientSession() as s:
        out["base"] = (await one_generation(s, url, prompt, GenSpec(**g)))["text"]
        for n in names:
            out[n] = (await one_generation(s, url, prompt, GenSpec(lora_path=n, **g)))["text"]
    return out


def main() -> int:
    """Build one adapter per family, load them, and assert each one moves the output alone."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--dir", default="/mnt/localssd/gate/adapters")
    ap.add_argument("--scales", default="0.02,0.05,0.2,1.0",
                    help="LoRA-B magnitudes to try, smallest first. A single scale cannot "
                         "decide this: the MLP family is 192 modules of width 17408 while the "
                         "full-attention family is 64 much narrower ones, so the same "
                         "per-element magnitude is a far smaller function perturbation there. "
                         "Measured: at 0.02 only the MLP family moved a greedy completion, at "
                         "0.2 all three did. A fixed small scale therefore reports 'not "
                         "applied' for a family that IS applied.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    prompt = build_prompt(tok, "What is 17 times 23? Give the number only.", "low")
    scales = [float(x) for x in a.scales.split(",")]
    threshold: dict = {}
    pending = list(FAMILIES)
    verdict: dict = {}
    for scale in scales:
        if not pending:
            break
        for name in list(pending):
            pattern, tlist, want = FAMILIES[name]
            path = "%s/fam_%s" % (a.dir, name)
            r = subprocess.run([sys.executable, "/mnt/localssd/gate/code/mk_probe_adapter.py",
                                "--out", path, "--scale", str(scale), "--pattern", pattern,
                                "--target-list", tlist], capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout + r.stderr, file=sys.stderr)
                return 2
            requests.post(a.url + "/unload_lora_adapter", json={"lora_name": name}, timeout=600)
            rr = requests.post(a.url + "/load_lora_adapter",
                               json={"lora_name": name, "lora_path": path}, timeout=1200)
            if rr.status_code != 200:
                print("FAIL: %s could not be loaded at all: %s" % (name, rr.text[:300]),
                      file=sys.stderr)
                return 3
            res = asyncio.run(texts(a.url, prompt, [name]))
            moved = res[name] != res["base"]
            print("[%-16s] scale %-5g modules reached %2d layers  moved=%s"
                  % (name, scale, want, moved), flush=True)
            verdict[name] = {"smallest_scale_that_moved_output": scale if moved else None,
                             "layers_expected": want, "text": res[name][:120]}
            if moved:
                threshold[name] = scale
                pending.remove(name)
            requests.post(a.url + "/unload_lora_adapter", json={"lora_name": name}, timeout=600)
    verdict["base_text"] = res["base"][:120]
    verdict["scales_tried"] = scales
    print(json.dumps(verdict, indent=1))
    if a.out:
        json.dump(verdict, open(a.out, "w"), indent=1)
    if pending:
        print("FAIL: the server never applies these families at any tried magnitude: %s. An "
              "adapter that targets them would train weights the rollouts never see."
              % pending, file=sys.stderr)
        return 1
    print("PASS: every module family is applied by the serving backend; smallest responding "
          "magnitude per family: %s" % threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
