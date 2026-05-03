#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=/scratch/sarthmit/bqa/runs/logs/attn_compare/%x_%j.log

set -uo pipefail

cd /scratch/sarthmit/bqa
# setup_node.sh activates the venv and points NANOCHAT_BASE_DIR (and HF/UV/wandb
# caches) at scratch under the repo, so nothing is written under $HOME.
source scripts/setup_node.sh
export OMP_NUM_THREADS=1

: "${FLOPS:?FLOPS env var is required (e.g. 1e18)}"
: "${DEPTHS:?DEPTHS env var is required (space-separated, e.g. '12 16 20')}"
: "${ARCHS:?ARCHS env var is required (space-separated subset of 'mha gqa bqa')}"
LABEL="${LABEL:-attncmp_apr28}"
EVAL_TOKENS=$((100 * 524288))

mkdir -p runs/logs/attn_compare "$NANOCHAT_BASE_DIR"
ITERS_FILE="$NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv"
[ -f "$ITERS_FILE" ] || echo "flops,depth,num_iters" > "$ITERS_FILE"

echo "=== $(date -Is) | host=$(hostname) | FLOPS=$FLOPS | DEPTHS='$DEPTHS' | ARCHS='$ARCHS' | LABEL=$LABEL ==="
nvidia-smi -L

dbs_for_depth() {
    local d=$1
    if   [ "$d" -ge 28 ]; then echo 8
    elif [ "$d" -ge 20 ]; then echo 16
    else                       echo 32
    fi
}

# Per-arch CLI flags. n_head = (depth*64)/128 = depth/2 with default aspect_ratio/head_dim.
arch_args() {
    local arch=$1 depth=$2
    case "$arch" in
        mha) echo "--attn-kind=gqa --n-kv-head=$(( depth / 2 ))" ;;
        gqa) echo "--attn-kind=gqa" ;;
        bqa) echo "--attn-kind=bqa" ;;
        *) echo "ERROR: unknown arch '$arch'" >&2; return 1 ;;
    esac
}

lookup_mha_iters() {
    local flops=$1 depth=$2
    awk -F, -v F="$flops" -v D="$depth" '$1==F && $2==D {v=$3} END {print v}' "$ITERS_FILE"
}

run_one() {
    local arch=$1 depth=$2 flops=$3
    local tag="${LABEL}_${arch}_d${depth}_f${flops}"
    local dbs; dbs=$(dbs_for_depth "$depth")
    local extra_args; extra_args=$(arch_args "$arch" "$depth") || return 1
    local horizon_args

    if [ "$arch" = "mha" ]; then
        horizon_args="--target-flops=$flops --target-param-data-ratio=-1"
    else
        local mha_iters; mha_iters=$(lookup_mha_iters "$flops" "$depth")
        if [ -z "$mha_iters" ]; then
            echo "ERROR: MHA num_iters not found for flops=$flops depth=$depth in $ITERS_FILE" >&2
            return 1
        fi
        horizon_args="--num-iterations=$mha_iters"
    fi

    local log_file="runs/logs/attn_compare/${tag}.log"
    echo "--- $(date -Is) | starting $tag (arch=$arch, depth=$depth, dbs=$dbs, horizon='$horizon_args') ---"
    torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
        --depth="$depth" \
        $horizon_args \
        --device-batch-size="$dbs" \
        --fp8 \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --run="$tag" \
        --model-tag="$tag" \
        $extra_args \
        2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    echo "--- $(date -Is) | finished $tag (exit=$rc) ---"

    if [ "$arch" = "mha" ] && [ "$rc" = "0" ]; then
        local iters
        iters=$(grep "Calculated number of iterations from target FLOPs" "$log_file" \
                | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        if [ -n "$iters" ]; then
            # Atomic single-line append; safe across parallel MHA jobs writing different (flops,depth)
            echo "$flops,$depth,$iters" >> "$ITERS_FILE"
            echo "Cached MHA iters: flops=$flops depth=$depth num_iters=$iters"
        else
            echo "WARNING: could not extract MHA num_iters from $log_file" >&2
        fi
    fi
    return $rc
}

# Per depth, run requested archs in order. If 'mha' is in ARCHS, it must come first
# so it caches num_iters before gqa/bqa look it up.
order_archs() {
    local archs="$1"
    local out=""
    for want in mha gqa bqa; do
        for a in $archs; do
            if [ "$a" = "$want" ]; then out="$out $want"; fi
        done
    done
    echo "$out"
}

ORDERED_ARCHS=$(order_archs "$ARCHS")
echo "Run order per depth: '$ORDERED_ARCHS'"

failures=0
for d in $DEPTHS; do
    for a in $ORDERED_ARCHS; do
        if ! run_one "$a" "$d" "$FLOPS"; then
            failures=$((failures + 1))
            echo "FAILED: arch=$a depth=$d flops=$FLOPS (continuing)"
        fi
    done
done

echo "=== $(date -Is) | job complete | failures=$failures ==="
exit $failures
