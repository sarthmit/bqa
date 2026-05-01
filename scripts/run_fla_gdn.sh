#!/bin/bash
#SBATCH -J fla_gdn
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/fla_gdn_%j.log
#
# Pure GDN at d=12 — runs Dense first then MoE, both at target_flops=2.15e18.
# 4xH100; expected wall time ~90 min total. GDN has no quadratic attention term
# but the Triton chunk kernel JITs on first run (adds ~60s on a cold node).
set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
source scripts/setup_node.sh

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
python -c "import fla; print('fla', fla.__version__)" 2>&1 | head -1

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

stage "DENSE GDN (target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=gdn \
    --device-batch-size=32 \
    --run=dummy \
    --model-tag=d12_fla_gdn_dense

stage "MoE GDN (8 experts, top_k=2, target_flops=2.15e18)"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=gdn \
    --device-batch-size=16 \
    --moe-num-experts=8 \
    --moe-top-k=2 \
    --moe-lbl-loss-weight=0.01 \
    --run=dummy \
    --model-tag=d12_fla_gdn_moe

stage "DONE"
