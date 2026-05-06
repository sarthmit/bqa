#!/bin/bash
# Submitted 2026-05-03: BQA with n_kv_head = n_head (full-rank K/V, learned
# softmax K/V remix starting near identity from m_k=0.70, m_v=0.85 defaults).
# Coverage: every (depth, flops) cell where mha/gqa/bqa have been run or
# are queued, namely:
#     d12 / d16 / d20  x  1e18, 2.15e18, 4.64e18, 1e19
#     d20 / d24 / d28  x  2.15e19, 4.64e19
#     d24 / d28        x  1e19
# = 20 cells total.
#
# The MHA num_iters for these cells are split across two iter-cache files:
#     attn_compare_attncmp_apr28_iters.csv : d12/d16/d20 x {1e18 ... 1e19}
#     attn_compare_attncmp_apr30_iters.csv : d20/d24/d28 x {1e19 ... 4.64e19}
# This script picks the right LABEL per cell so the cache lookup hits.
# Tags `attncmp_<LABEL>_bqafull_*` don't collide with existing arch tags.

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
ARCH="bqafull"

submit_one() {
    local label=$1 depth=$2 flops=$3 part=$4 wt=$5
    local name="${label}_${ARCH}_d${depth}_f${flops}"
    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ARCHS="$ARCH",LABEL="$label" \
        runs/attn_compare_4xh100.sbatch | awk '{print $NF}')
    printf "  %-50s d=%-2s flops=%-8s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$part" "$wt" "$jid"
}

echo ">>> Submitting bqafull jobs (ARCH=$ARCH)"

# --- attncmp_apr28 cache: d12/d16/d20 × {1e18, 2.15e18, 4.64e18, 1e19} ---
# Sub-1e19 budgets at d12/d16/d20 are fast (≤1hr), use PART_SHORT.
for d in 12 16 20; do
    submit_one "attncmp_apr28" "$d" "1e18"    "$PART_SHORT" "01:00:00"
    submit_one "attncmp_apr28" "$d" "2.15e18" "$PART_SHORT" "01:30:00"
    submit_one "attncmp_apr28" "$d" "4.64e18" "$PART_SHORT" "02:30:00"
    submit_one "attncmp_apr28" "$d" "1e19"    "$PART_SHORT" "02:30:00"
done

# --- attncmp_apr30 cache: d20/d24/d28 × {1e19, 2.15e19, 4.64e19} ---
# 1e19 d24/d28 (d20/1e19 already covered above under attncmp_apr28).
for d in 24 28; do
    submit_one "attncmp_apr30" "$d" "1e19" "$PART_SHORT" "02:00:00"
done
# 2.15e19 d20/d24/d28
for d in 20 24 28; do
    submit_one "attncmp_apr30" "$d" "2.15e19" "$PART_LONG" "04:00:00"
done
# 4.64e19 d20/d24/d28
for d in 20 24 28; do
    submit_one "attncmp_apr30" "$d" "4.64e19" "$PART_LONG" "08:00:00"
done

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
