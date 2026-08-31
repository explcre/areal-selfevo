#!/usr/bin/env bash
# Verify N_SAMPLES / DATASET / PORTABLE_EXTRA reach the trainer command line, no GPU needed.
set -u
STUB=$(mktemp -d)
cat > "$STUB/python3" <<'EOS'
#!/usr/bin/env bash
echo "PYTHON3 ARGS: $*"
exit 0
EOS
chmod +x "$STUB/python3"
cat > "$STUB/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7; do echo "$i, 0"; done
EOS
chmod +x "$STUB/nvidia-smi"
export PATH="$STUB:$PATH"
export RUN_PORTABLE_SOURCE_ONLY=1
# shellcheck disable=SC1090
source "$HOME/areal-selfevo/experiments/harness/run_portable.sh"

RUN_NAME=knobtest MODE=train ARM=on SOLVED_ADV=0.5 EPOCHS=1
N_SAMPLES=16
DATASET=DigitalLearningGmbH/MATH-lighteval
PORTABLE_EXTRA="actor.mb_spec.max_tokens_per_mb=49152 ref.mb_spec.max_tokens_per_mb=49152"
MODEL=Qwen/Qwen2.5-1.5B-Instruct
WANDB_MODE=offline; WANDB_PROJECT=p
OUTDIR="$STUB/out"; WORKDIR="$STUB"; LOG="$STUB/log"; mkdir -p "$OUTDIR"
REPO_DIR="$HOME/areal-selfevo"
out=$(run_once "0,1,2,3,4,5,6,7" 8 2>&1; cat "$LOG" 2>/dev/null)
for want in "gconfig.n_samples=16" "train_dataset.path=DigitalLearningGmbH/MATH-lighteval" \
            "valid_dataset.path=DigitalLearningGmbH/MATH-lighteval" \
            "actor.mb_spec.max_tokens_per_mb=49152" "ref.mb_spec.max_tokens_per_mb=49152" \
            "experiment_name=knobtest"; do
  if echo "$out" | grep -q -- "$want"; then echo "  OK   $want"; else echo "  MISS $want"; fi
done
rm -rf "$STUB"
