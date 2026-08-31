"""Teach sglang's LoRA manager to find vocab_size on a MULTIMODAL config.

Qwen3.8-27B is multimodal: its config has text_config / vision_config, and vocab_size lives
under text_config (248320), not at the top level. sglang's LoRA manager reads
``self.base_hf_config.vocab_size`` directly and raises
``'Qwen3_5Config' object has no attribute 'vocab_size'``.

This is a local patch to an installed package, which is invisible in a git log -- the same
class of change as the torchao upgrade that broke this environment earlier. It therefore
backs up the original and marks every edited line, so the next person can find it.
"""
import pathlib
import shutil
import sys

p = pathlib.Path("/venv/main/lib/python3.12/site-packages/sglang/srt/lora/lora_manager.py")
backup = p.with_suffix(".py.selfevo-backup")
s = p.read_text()

if "_selfevo_vocab_size" in s:
    print("already patched"); sys.exit(0)
if not backup.exists():
    shutil.copy2(p, backup)

helper = '''

def _selfevo_vocab_size(cfg):
    """vocab_size from a flat OR a multimodal config. PATCHED by selfevo.

    Multimodal configs (Qwen3.8, Qwen2.5-VL, ...) nest the language model's settings under
    ``text_config``, so a direct ``cfg.vocab_size`` raises AttributeError and LoRA cannot
    load. Falls back rather than assuming either shape.
    """
    v = getattr(cfg, "vocab_size", None)
    if v is not None:
        return v
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(cfg, attr, None)
        v = getattr(sub, "vocab_size", None) if sub is not None else None
        if v is not None:
            return v
    raise AttributeError(
        f"{type(cfg).__name__} exposes no vocab_size at the top level or under "
        "text_config/llm_config/language_config"
    )

'''

old = "                base_vocab_size=self.base_hf_config.vocab_size,"
n = s.count(old)
assert n == 2, f"expected 2 call sites, found {n}"
s = s.replace(old, "                base_vocab_size=_selfevo_vocab_size(self.base_hf_config),  # PATCHED by selfevo")

# Insert the helper after the import block.
lines = s.split("\n")
idx = max(i for i, l in enumerate(lines[:80]) if l.startswith(("import ", "from ")))
lines.insert(idx + 1, helper)
p.write_text("\n".join(lines))

import ast
ast.parse(p.read_text())
print(f"patched {n} sites; backup at {backup}")
