#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=/scratch/sarthmit/bqa/runs/logs/attn_compare/%x_%j.log

# Packed runner: executes a sequence of base_train invocations sequentially in
# a single slurm allocation. The CONFIGS env var holds a `;`-separated list of
# `arch:depth:flops:m_k:m_v:seed[:alpha_lr_mult:alpha_beta1:alpha_wd]` tuples.
# m_k/m_v ignored when arch=gqa/mha; bqa_dyn ignores m_v; alpha_* fields are
# optional and default to (1.0, 0.9, 0.0) which reproduces the baseline.
#
# Required env: CONFIGS, LABEL.
# Example:
#   CONFIGS="bqa:16:1e18:0.50:0.85:42;bqa:16:1e18:0.50:0.85:42:2.0:0.5:0.0"
# Reuses MHA num_iters cached at $NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv.

set -uo pipefail

cd /scratch/sarthmit/bqa
source scripts/setup_node.sh
export OMP_NUM_THREADS=1

: "${CONFIGS:?CONFIGS env var is required, semicolon-separated arch:depth:flops:mass:seed tuples}"
LABEL="${LABEL:-attncmp_apr28}"
EVAL_TOKENS=$((100 * 524288))

mkdir -p runs/logs/attn_compare "$NANOCHAT_BASE_DIR"
ITERS_FILE="$NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv"
[ -f "$ITERS_FILE" ] || { echo "ERROR: iters cache not found: $ITERS_FILE" >&2; exit 1; }

dbs_for_depth() {
    local d=$1
    if   [ "$d" -ge 28 ]; then echo 8
    elif [ "$d" -ge 20 ]; then echo 16
    else                       echo 32
    fi
}

run_one() {
    local arch=$1 depth=$2 flops=$3 m_k=$4 m_v=$5 seed=$6
    local alpha_lr_mult="${7:-1.0}" alpha_beta1="${8:-0.9}" alpha_wd="${9:-0.0}"

    local num_iters
    num_iters=$(awk -F, -v F="$flops" -v D="$depth" '$1==F && $2==D {v=$3} END {print v}' "$ITERS_FILE")
    if [ -z "$num_iters" ]; then
        echo "ERROR: no MHA iters cached for flops=$flops depth=$depth in $ITERS_FILE" >&2
        return 1
    fi
    local dbs; dbs=$(dbs_for_depth "$depth")

    # Suffix added to the tag only when an alpha-group optimizer override is non-default,
    # so vanilla (m_k, m_v) runs keep the existing short tag form.
    local opt_tag=""
    [ "$alpha_lr_mult" = "1.0" ] || opt_tag="${opt_tag}_lrm$(printf '%s' "$alpha_lr_mult" | tr '.' 'p')"
    [ "$alpha_beta1"   = "0.9" ] || opt_tag="${opt_tag}_b1$(printf '%s' "$alpha_beta1"   | tr '.' 'p')"
    [ "$alpha_wd"      = "0.0" ] || opt_tag="${opt_tag}_wd$(printf '%s' "$alpha_wd"      | tr '.' 'p')"

    local extra=""
    local tag
    if [ "$arch" = "bqa" ]; then
        local mk_tag mv_tag; mk_tag=$(printf '%s' "$m_k" | tr '.' 'p'); mv_tag=$(printf '%s' "$m_v" | tr '.' 'p')
        tag="${LABEL}_bqa_d${depth}_f${flops}_mk${mk_tag}_mv${mv_tag}_s${seed}${opt_tag}"
        extra="--attn-kind=bqa --bqa-init-mass-k=${m_k} --bqa-init-mass-v=${m_v} --alpha-lr-mult=${alpha_lr_mult} --alpha-beta1=${alpha_beta1} --alpha-wd=${alpha_wd}"
    elif [ "$arch" = "gqa" ]; then
        tag="${LABEL}_gqa_d${depth}_f${flops}_s${seed}"
        extra="--attn-kind=gqa"
    elif [ "$arch" = "mha" ]; then
        tag="${LABEL}_mha_d${depth}_f${flops}_s${seed}"
        extra="--attn-kind=gqa --n-kv-head=$(( depth / 2 ))"
    else
        echo "ERROR: unknown arch '$arch'" >&2
        return 1
    fi

    local log_file="runs/logs/attn_compare/${tag}.log"
    echo "--- $(date -Is) | starting $tag (arch=$arch d=$depth f=$flops m_k=$m_k m_v=$m_v seed=$seed iters=$num_iters dbs=$dbs alpha_lr_mult=$alpha_lr_mult alpha_beta1=$alpha_beta1 alpha_wd=$alpha_wd) ---"

    torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
        --depth="$depth" \
        --num-iterations="$num_iters" \
        --device-batch-size="$dbs" \
        --fp8 \
        --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --run="$tag" \
        --model-tag="$tag" \
        --seed="$seed" \
        $extra \
        2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    echo "--- $(date -Is) | finished $tag (exit=$rc) ---"
    return $rc
}

echo "=== $(date -Is) | host=$(hostname) | LABEL=$LABEL | CONFIGS='$CONFIGS' ==="
nvidia-smi -L

failures=0
IFS=';' read -ra CFGS <<< "$CONFIGS"
for cfg in "${CFGS[@]}"; do
    IFS=':' read -ra parts <<< "$cfg"
    arch="${parts[0]}"; depth="${parts[1]}"; flops="${parts[2]}"
    m_k="${parts[3]:-0.58}"; m_v="${parts[4]:-0.58}"; seed="${parts[5]:-42}"
    alpha_lr_mult="${parts[6]:-1.0}"; alpha_beta1="${parts[7]:-0.9}"; alpha_wd="${parts[8]:-0.0}"
    if ! run_one "$arch" "$depth" "$flops" "$m_k" "$m_v" "$seed" "$alpha_lr_mult" "$alpha_beta1" "$alpha_wd"; then
        failures=$((failures + 1))
        echo "FAILED: $cfg (continuing)"
    fi
done

echo "=== $(date -Is) | job complete | failures=$failures ==="
exit $failures
