"""Which adv_norm settings preserve the per-group zero-sum the routing rule needs?

The rule's premise is sum_i A_i = 0 within each GRPO group. An audit measured that the
repo's live setting (mean_level=batch) breaks it by 87-115% of mean |A| when generations
differ in length. cli_args allows 'batch', 'group' or None, so the question is whether a
usable setting exists -- measured, not argued.
"""
import sys
sys.path.insert(0, "/home/ubuntu/areal-selfevo")
import torch
from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization

# Two groups of two, UNEQUAL generation lengths -- the case that breaks batch centring.
adv = torch.tensor([
    [2.0, 2.0, 2.0, 0.0],
    [-2.0, -2.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0, -1.0],
])
loss_mask = torch.tensor([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
], dtype=torch.float32)
group_sizes = [2, 2]

def group_sums(a):
    """Per-group sum of the per-sequence mean advantage, over loss-carrying tokens."""
    out = []
    r = 0
    for g in group_sizes:
        rows = list(range(r, r + g))
        s = sum(float((a[i] * loss_mask[i]).sum()) for i in rows)
        out.append(s)
        r += g
    return out

print(f"{'mean_level':>12} {'std_level':>10} | per-group sums | max |sum|")
print("-" * 62)
base = group_sums(adv)
print(f"{'(raw GRPO)':>12} {'--':>10} | {[f'{x:+.4f}' for x in base]} | {max(abs(x) for x in base):.4f}")
for mean_level in ("batch", "group", None):
    for std_level in ("batch", "group", None):
        try:
            cfg = NormConfig(mean_level=mean_level, std_level=std_level,
                             mean_leave1out=False, group_size=2)
            norm = Normalization(cfg)
            out = norm(adv.clone(), loss_mask, group_sizes=group_sizes)
            s = group_sums(out)
            print(f"{str(mean_level):>12} {str(std_level):>10} | "
                  f"{[f'{x:+.4f}' for x in s]} | {max(abs(x) for x in s):.4f}")
        except Exception as e:
            print(f"{str(mean_level):>12} {str(std_level):>10} | {type(e).__name__}: {str(e)[:40]}")
