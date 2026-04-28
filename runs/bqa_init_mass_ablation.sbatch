#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=/scratch/sarthmit/bqa/runs/logs/attn_compare/%x_%j.log

# Single (depth, flops, mass) BQA run with a configurable --bqa-init-mass.
# `mass` is the target softmax probability on the GQA-assigned basis at init;
# the actual logit is derived per n_kv_head so the init shape is depth-invariant.
#
# Required env: FLOPS, DEPTH, BQA_INIT_MASS, LABEL.
# Reuses MHA num_iters cached at $NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv.

set -uo pipefail

cd /scratch/sarthmit/bqa
source scripts/setup_node.sh
export OMP_NUM_THREADS=1

: "${FLOPS:?FLOPS env var is required (e.g. 4.64e18)}"
: "${DEPTH:?DEPTH env var is required (e.g. 16)}"
: "${BQA_INIT_MASS:?BQA_INIT_MASS env var is required (e.g. 0.58 or 0.85)}"
LABEL="${LABEL:-attncmp_apr28}"
EVAL_TOKENS=$((100 * 524288))

mkdir -p runs/logs/attn_compare "$NANOCHAT_BASE_DIR"
ITERS_FILE="$NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv"
[ -f "$ITERS_FILE" ] || { echo "ERROR: iters cache not found: $ITERS_FILE" >&2; exit 1; }

NUM_ITERS=$(awk -F, -v F="$FLOPS" -v D="$DEPTH" '$1==F && $2==D {v=$3} END {print v}' "$ITERS_FILE")
[ -n "$NUM_ITERS" ] || { echo "ERROR: no MHA iters cached for flops=$FLOPS depth=$DEPTH in $ITERS_FILE" >&2; exit 1; }

# dbs matches runs/attn_compare_4xh100.sbatch
if   [ "$DEPTH" -ge 28 ]; then DBS=8
elif [ "$DEPTH" -ge 20 ]; then DBS=16
else                           DBS=32
fi

# Tag uses an underscore-safe encoding of the mass value.
MASS_TAG=$(printf '%s' "$BQA_INIT_MASS" | tr '.' 'p')
TAG="${LABEL}_bqa_d${DEPTH}_f${FLOPS}_m${MASS_TAG}"
LOG="runs/logs/attn_compare/${TAG}.log"

echo "=== $(date -Is) | host=$(hostname) | TAG=$TAG | NUM_ITERS=$NUM_ITERS | DBS=$DBS ==="
nvidia-smi -L

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --num-iterations="$NUM_ITERS" \
    --device-batch-size="$DBS" \
    --fp8 \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=999999 \
    --core-metric-max-per-task=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run="$TAG" \
    --model-tag="$TAG" \
    --attn-kind=bqa \
    --bqa-init-mass="$BQA_INIT_MASS" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "=== $(date -Is) | finished $TAG (exit=$rc) ==="
exit $rc
