#!/bin/bash
#SBATCH -J fla_hyb025
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/fla_hybrid025_%j.log
#
# Hybrid 3:1 GDN:MHA at d=12 — alpha=0.25 (3 GDN per MHA, MHA at indices 3, 7, 11).
# Layout: GGGM GGGM GGGM. Runs Dense then MoE at target_flops=2.15e18.
set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
source scripts/setup_node.sh

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

COMMON=(
    --model-kind=fla
    --attn-kind=hybrid
    --alpha=0.25
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

stage "DENSE Hybrid alpha=0.25 (target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --device-batch-size=32 \
    --run=dummy \
    --model-tag=d12_fla_hybrid025_dense

stage "MoE Hybrid alpha=0.25 (8 experts, top_k=2, target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --device-batch-size=16 \
    --moe-num-experts=8 \
    --moe-top-k=2 \
    --moe-lbl-loss-weight=0.01 \
    --run=dummy \
    --model-tag=d12_fla_hybrid025_moe

stage "DONE"
