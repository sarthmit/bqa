#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=/scratch/sarthmit/bqa/runs/logs/scaling/%x_%j.log

set -uo pipefail

cd /scratch/sarthmit/bqa
# setup_node.sh activates the venv and points NANOCHAT_BASE_DIR (and HF/UV/wandb
# caches) at scratch under the repo, so nothing is written under $HOME.
source scripts/setup_node.sh
export OMP_NUM_THREADS=1

: "${FLOPS:?FLOPS env var is required}"
: "${DEPTHS:?DEPTHS env var is required (space-separated, e.g. '12 16 20 24 28 32')}"
LABEL="${LABEL:-scaling_apr28}"
ATTN_KIND="${ATTN_KIND:-bqa}"
EVAL_TOKENS=$((100 * 524288))

echo "=== $(date -Is) | host=$(hostname) | FLOPS=$FLOPS | DEPTHS='$DEPTHS' | ATTN_KIND=$ATTN_KIND ==="
nvidia-smi -L

run_one() {
    local depth="$1"
    local tag="${LABEL}_${ATTN_KIND}_d${depth}_f${FLOPS}"

    if   [ "$depth" -ge 32 ]; then DBS=4
    elif [ "$depth" -ge 28 ]; then DBS=8
    elif [ "$depth" -ge 20 ]; then DBS=16
    else                            DBS=32
    fi

    echo "--- $(date -Is) | starting $tag (depth=$depth, dbs=$DBS, attn=$ATTN_KIND) ---"
    torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
        --depth="$depth" \
        --attn-kind="$ATTN_KIND" \
        --target-flops="$FLOPS" \
        --target-param-data-ratio=-1 \
        --device-batch-size="$DBS" \
        --fp8 \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --run="$tag" \
        --model-tag="$tag"
    rc=$?
    echo "--- $(date -Is) | finished $tag (exit=$rc) ---"
}

for d in $DEPTHS; do
    run_one "$d"
done

echo "=== $(date -Is) | job complete ==="
