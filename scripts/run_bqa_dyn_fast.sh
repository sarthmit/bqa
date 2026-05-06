#!/bin/bash
# bqa_dyn re-run on the new SDPA Q-fold path (gpt.py DynamicBasisQueryAttention).
# Same num-iterations as the static run for direct wandb comparison; DBS=32 because
# the new path drops attention memory ~4x and unlocks larger micro-batch.
#
# Targets cn-g023 (4xA100 under SLURM job 9397730).

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
LOG=$REPO/.cache/bqa_dyn_fast.log
mkdir -p "$REPO/.cache"
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export SLURM_TMPDIR=/tmp

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

# /tmp/nanochat-venv is fresh on cn-g023 — setup_node.sh rsyncs from BeeGFS .venv.
stage "source setup_node.sh (rsync .venv -> /tmp/nanochat-venv if needed)"
{ source scripts/setup_node.sh; } >> "$LOG" 2>&1

stage "python sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"

export OMP_NUM_THREADS=1

stage "base_train d12 bqa_dyn (FAST: SDPA Q-fold, DBS=32, 1000 iters)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --attn-kind=bqa_dyn \
    --n-kv-head=3 \
    --device-batch-size=32 \
    --num-iterations=1000 \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run=d12_bqa_dyn \
    --model-tag=d12_bqa_dyn 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
stage "FAST bqa_dyn finished rc=$RC"
