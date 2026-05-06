#!/bin/bash
# Run BQA static and bqa_dyn back-to-back on cn-g026 (4xA100, job 9396889).
# Same step count for both so wandb curves are directly comparable; targets
# ~50 min combined wall time (well under the 1-hour budget the user asked for).
#
# Static BQA reuses the MHA-fast weight-fold path → ~640 ms/step at DBS=16.
# bqa_dyn materializes (B,H,T,T,J) so caps at DBS=8 on A100 80GB.
# Both at d12, n_kv_head=3 (J=3), --window-pattern=L (no fp8 on A100).

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
LOG=$REPO/.cache/bqa_compare.log
mkdir -p "$REPO/.cache"
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export SLURM_TMPDIR=/tmp

# Pin caches under bqa/.cache/ (NEVER $HOME)
export NANOCHAT_REPO="$REPO"
export NANOCHAT_BASE_DIR="$REPO/.cache/nanochat"
export UV_CACHE_DIR="$REPO/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$REPO/.cache/uv-python"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export XDG_CACHE_HOME="$REPO/.cache/xdg"
export HF_HOME="$REPO/.cache/hf"
export WANDB_CACHE_DIR="$REPO/.cache/wandb"
export WANDB_CONFIG_DIR="$REPO/.cache/wandb-config"
export WANDB_DIR="$REPO/.cache/wandb"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor"
mkdir -p "$NANOCHAT_BASE_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

stage() { echo "===== $(date '+%F %T') :: $* =====" | tee -a "$LOG"; }

# Activate the existing /tmp/nanochat-venv (already populated by the prior pipeline run)
stage "activate /tmp/nanochat-venv"
{ source scripts/setup_node.sh; } >> "$LOG" 2>&1

stage "python sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"

NUM_ITERS=1000
COMMON_ARGS=(
    --depth=12
    --n-kv-head=3
    --num-iterations=$NUM_ITERS
    --target-param-data-ratio=-1
    --window-pattern=L
    --core-metric-every=999999
    --core-metric-max-per-task=-1
    --sample-every=-1
    --save-every=-1
)

export OMP_NUM_THREADS=1

# --- Run 1: BQA static (fast path, weight-fold trick) ---
stage "RUN 1: bqa static (DBS=16, $NUM_ITERS iters) -> wandb run d12_bqa_static"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --attn-kind=bqa \
    --device-batch-size=16 \
    --run=d12_bqa_static \
    --model-tag=d12_bqa_static \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
RC1=${PIPESTATUS[0]}
stage "RUN 1 finished rc=$RC1"

# --- Run 2: bqa_dyn (materializes (B,H,T,T,J), DBS=8) ---
stage "RUN 2: bqa_dyn (DBS=8, $NUM_ITERS iters) -> wandb run d12_bqa_dyn"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --attn-kind=bqa_dyn \
    --device-batch-size=8 \
    --run=d12_bqa_dyn \
    --model-tag=d12_bqa_dyn \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG"
RC2=${PIPESTATUS[0]}
stage "RUN 2 finished rc=$RC2"

stage "BOTH RUNS DONE (static rc=$RC1, dyn rc=$RC2)"
