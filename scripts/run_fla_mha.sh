#!/bin/bash
#SBATCH -J fla_mha
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/fla_mha_%j.log
#
# Pure MHA at d=12 — runs Dense first then MoE, both at target_flops=2.15e18.
# `--attn-kind=gqa` maps to fla MHA (n_kv_head==n_head per branch constraint).
# 4xH100; expected wall time ~85 min total for both runs.
set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
# setup_node.sh sets cache redirects (NANOCHAT_BASE_DIR / UV_CACHE_DIR / etc),
# rsyncs .venv -> $SLURM_TMPDIR/nanochat-venv (ext4) under SLURM, and activates it.
source scripts/setup_node.sh

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | head -1

# Common arguments. --target-param-data-ratio left at default (12) since base_train
# uses it for the auto batch-size and weight-decay scaling; --target-flops controls
# num_iterations directly (overrides the data-ratio for iteration count).
COMMON=(
    --model-kind=fla
    --depth=12
    --aspect-ratio=64
    --head-dim=64
    --max-seq-len=2048
    --target-flops=2.15e18
    --window-pattern=L
    --eval-every=250
    --core-metric-every=-1
    --sample-every=-1
    --save-every=1000
    --warmup-steps=40
    --warmdown-ratio=0.65
)

stage "DENSE MHA (target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    "${COMMON[@]}" \
    --attn-kind=gqa \
    --device-batch-size=32 \
    --run=dummy \
    --model-tag=d12_fla_mha_dense

stage "MoE MHA (8 experts, top_k=2, target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    "${COMMON[@]}" \
    --attn-kind=gqa \
    --device-batch-size=16 \
    --moe-num-experts=8 \
    --moe-top-k=2 \
    --moe-lbl-loss-weight=0.01 \
    --run=dummy \
    --model-tag=d12_fla_mha_moe

stage "DONE"
