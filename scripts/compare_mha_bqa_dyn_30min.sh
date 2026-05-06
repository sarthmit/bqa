#!/bin/bash
# Iso-wallclock comparison: 30 min MHA vs 30 min bqa_dyn at d12.
# Same DBS, same window-pattern, same depth — only the attention differs.
# val bpb every 200 steps so we get a curve, not just an endpoint.

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
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
export WANDB_PROJECT=FlexHybrid
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor"
mkdir -p "$NANOCHAT_BASE_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LOG_DIR="$REPO/.cache"
MHA_LOG="$LOG_DIR/cmp30_mha.log"
BQA_LOG="$LOG_DIR/cmp30_bqa_dyn.log"
DRIVER_LOG="$LOG_DIR/cmp30_driver.log"

stage() { echo "===== $(date '+%F %T') :: $* =====" | tee -a "$DRIVER_LOG"; }

stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
{ source scripts/setup_node.sh; } >> "$DRIVER_LOG" 2>&1
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" 2>&1 | tee -a "$DRIVER_LOG"

export OMP_NUM_THREADS=1

COMMON=(
    --depth=12
    --device-batch-size=32
    --num-iterations=500
    --target-param-data-ratio=-1
    --window-pattern=L
    --core-metric-every=-1
    --core-metric-max-per-task=-1
    --sample-every=-1
    --save-every=-1
    --eval-every=100
)

stage "MHA: 500 iters, attn-kind=gqa n-kv-head=6 (= MHA), DBS=32"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=gqa \
    --n-kv-head=6 \
    --run=d12_mha_500 \
    --model-tag=d12_mha_500 2>&1 | tee -a "$MHA_LOG"
MHA_RC=${PIPESTATUS[0]}
stage "MHA finished rc=$MHA_RC"

# Settle GPUs (wandb finalize, NCCL teardown).
sleep 10

stage "bqa_dyn: 500 iters, attn-kind=bqa_dyn n-kv-head=3, DBS=32"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa_dyn \
    --n-kv-head=3 \
    --run=d12_bqa_dyn_500 \
    --model-tag=d12_bqa_dyn_500 2>&1 | tee -a "$BQA_LOG"
BQA_RC=${PIPESTATUS[0]}
stage "bqa_dyn finished rc=$BQA_RC"

stage "DONE — MHA log: $MHA_LOG ; bqa_dyn log: $BQA_LOG"
