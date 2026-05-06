#!/bin/bash
# Full bqa setup + tokenize + train pipeline. Designed to run on cn-g026
# (compute node with 4xA100 + 48 CPUs allocated to job 9396889).
#
# All caches/venvs/data under /home/mila/m/mittalsa/scratch/bqa/.cache/
# (BeeGFS scratch). Compile caches and the active venv copy are on
# /tmp (node-local ext4, set via SLURM_TMPDIR=/tmp).

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
LOG=$REPO/.cache/pipeline.log
mkdir -p "$REPO/.cache"

cd "$REPO"

# Make sure user-local bin (where uv lives) is on PATH — non-interactive ssh
# doesn't source ~/.bashrc.
export PATH="$HOME/.local/bin:$PATH"

# Set SLURM_TMPDIR for ssh sessions (pam_slurm_adopt sets cgroup but not env).
# /tmp on cn-g026 is local ext4 (7TB) — perfect for venv + compile caches.
export SLURM_TMPDIR=/tmp

# Pin all uv/pip/HF caches under bqa/.cache (NEVER $HOME). setup_node.sh
# normally does this, but uv needs the env vars set before its first invocation
# and setup_node.sh activates the venv (which doesn't exist yet on first run).
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
mkdir -p "$NANOCHAT_BASE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
         "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

stage() { echo "===== $(date '+%F %T') :: $* =====" | tee -a "$LOG"; }

# Wipe any stale partial local venv copy (left over from earlier failed attempt)
stage "wiping stale /tmp/nanochat-venv if present"
rm -rf /tmp/nanochat-venv

# --- Stage 1: build venv on BeeGFS (.venv inside repo, all wheels in .cache/uv) ---
# A corrupt .venv (directory exists but bin/python missing) is left over from
# any earlier killed uv sync — uv refuses to use it. Wipe only in that case.
if [ -d "$REPO/.venv" ] && [ ! -x "$REPO/.venv/bin/python" ]; then
    stage "wiping corrupt .venv (exists but no bin/python)"
    rm -rf "$REPO/.venv"
fi
stage "uv sync --extra gpu"
uv sync --extra gpu 2>&1 | tee -a "$LOG"
if [ ! -x "$REPO/.venv/bin/python" ]; then
    stage "FATAL: .venv/bin/python missing after uv sync"
    exit 1
fi

# --- Stage 2: source setup_node.sh — rsyncs full .venv -> /tmp/nanochat-venv & activates ---
# IMPORTANT: do NOT pipe `source` into tee — that runs source in a subshell and
# all PATH / VIRTUAL_ENV exports are dropped on subshell exit. Redirect instead.
stage "source setup_node.sh (rsync .venv -> /tmp/nanochat-venv, activate)"
# shellcheck source=/dev/null
{ source scripts/setup_node.sh; } >> "$LOG" 2>&1

stage "python sanity check"
python -c "import torch, pyarrow, datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"

# --- Stage 3: download 8 shards for tokenizer (~800MB) ---
stage "dataset -n 8 (tokenizer subset)"
python -m nanochat.dataset -n 8 2>&1 | tee -a "$LOG"

# --- Stage 4: kick off rest of dataset download in background ---
stage "dataset -n 240 (background full download)"
nohup python -m nanochat.dataset -n 240 > "$REPO/.cache/dataset-240.log" 2>&1 &
DATASET_PID=$!
echo "dataset bg PID=$DATASET_PID" | tee -a "$LOG"

# --- Stage 5: train tokenizer ---
stage "tok_train"
python -m scripts.tok_train 2>&1 | tee -a "$LOG"

# --- Stage 6: tok eval ---
stage "tok_eval"
python -m scripts.tok_eval 2>&1 | tee -a "$LOG"

# --- Stage 7: wait for full dataset download ---
stage "waiting for dataset bg PID=$DATASET_PID"
wait $DATASET_PID
DSTAT=$?
echo "dataset rc=$DSTAT" | tee -a "$LOG"
if [ $DSTAT -ne 0 ]; then
    stage "FATAL: dataset download failed (rc=$DSTAT)"
    tail -40 "$REPO/.cache/dataset-240.log" | tee -a "$LOG"
    exit $DSTAT
fi

# --- Stage 8: base_train (d12 MHA run on 4xA100) ---
# d12 -> n_head=6. MHA = n_kv_head == n_head (no GQA grouping).
stage "base_train d12_mha (4xA100, MHA: n_kv_head=6=n_head)"
export OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --attn-kind=gqa \
    --n-kv-head=6 \
    --device-batch-size=16 \
    --window-pattern=L \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run=d12_mha \
    --model-tag=d12_mha 2>&1 | tee -a "$LOG"

stage "pipeline complete"
