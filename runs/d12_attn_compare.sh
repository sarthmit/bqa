#!/bin/bash
set -uo pipefail

cd /scratch/sarthmit/bqa
source .venv/bin/activate
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

mkdir -p runs/logs

COMMON=(--depth=12 --num-iterations=500 --core-metric-every=-1 --fp8)

run_one() {
    local tag="$1"; shift
    echo "=== $(date -Is) | starting $tag ==="
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        "${COMMON[@]}" --run="$tag" --model-tag="$tag" "$@" \
        2>&1 | tee "runs/logs/$tag.log"
    echo "=== $(date -Is) | finished $tag ==="
}

run_one d12-mha-500 --attn-kind=gqa --n-kv-head=6
run_one d12-gqa-500 --attn-kind=gqa
run_one d12-bqa-500 --attn-kind=bqa
