#!/bin/bash
# Extension sweep launched 2026-04-30:
#   d20: f=2.15e19, 4.64e19
#   d24, d28: f=1e19, 2.15e19, 4.64e19
# All 3 archs (mha, gqa, bqa). One sbatch per (arch, depth, flops).

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

LABEL="${LABEL:-attncmp_apr30}"
PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
ARCHS_ORDERED="mha gqa bqa"

# Pre-cache MHA num_iters for every (flops, depth) cell we plan to launch.
echo ">>> Pre-caching MHA num_iters (LABEL=$LABEL) ..."
source scripts/setup_node.sh
python -m scripts.cache_mha_iters \
    --label "$LABEL" \
    --flops 1e19 2.15e19 4.64e19 \
    --depths 20 24 28

submit_one() {
    local arch=$1 depth=$2 flops=$3 part=$4 wt=$5
    local name="${LABEL}_${arch}_d${depth}_f${flops}"
    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ARCHS="$arch",LABEL="$LABEL" \
        runs/attn_compare_4xh100.sbatch | awk '{print $NF}')
    printf "  %-12s arch=%-3s d=%-2s flops=%-8s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$arch" "$depth" "$flops" "$part" "$wt" "$jid"
}

echo ">>> Submitting jobs"

# 1e19  : easy fit -> 3 h partition, --time=02:00:00.   (d24, d28) x 3 archs = 6 jobs
for d in 24 28; do
    for a in $ARCHS_ORDERED; do
        submit_one "$a" "$d" "1e19" "$PART_SHORT" "02:00:00"
    done
done

# 2.15e19: ~3 h estimated -> 12 h partition with 4 h cap. (d20, d24, d28) x 3 archs = 9 jobs
for d in 20 24 28; do
    for a in $ARCHS_ORDERED; do
        submit_one "$a" "$d" "2.15e19" "$PART_LONG" "04:00:00"
    done
done

# 4.64e19: ~6 h estimated -> 12 h partition with 8 h cap. (d20, d24, d28) x 3 archs = 9 jobs
for d in 20 24 28; do
    for a in $ARCHS_ORDERED; do
        submit_one "$a" "$d" "4.64e19" "$PART_LONG" "08:00:00"
    done
done

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
