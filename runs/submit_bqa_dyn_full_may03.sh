#!/bin/bash
# Submitted 2026-05-03: bqa_dyn at full rank (n_kv_head = n_head = depth/2),
# parallel to bqafull but with per-query dynamic alpha. Same defaults as bqa:
# bqa_init_mass_k=0.70, bqa_init_mass_v=0.85.
#
# Walltime sized from empirical data:
#   - bqa_dyn ≈ 5x static bqa (per-token alpha projection adds compute)
#   - bqafull ≈ 1.0-1.2x static bqa half-rank (full-rank K/V remix is cheap
#     since QK/AV matmul is unchanged)
# So bqa_dyn_full ≈ 5-6x static bqa half-rank. Walltimes here mirror the
# bqa_dyn (half-rank) submission with a small bump.
#
# MHA num_iters cache: attn_compare_attncmp_apr28_iters.csv (already populated).

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
ARCH="bqa_dyn_full"
LABEL="${LABEL:-attncmp_apr28}"

submit_one() {
    local depth=$1 flops=$2 part=$3 wt=$4
    local name="${LABEL}_${ARCH}_d${depth}_f${flops}"
    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ARCHS="$ARCH",LABEL="$LABEL" \
        runs/attn_compare_4xh100.sbatch | awk '{print $NF}')
    printf "  %-50s d=%-2s flops=%-8s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$part" "$wt" "$jid"
}

echo ">>> Submitting bqa_dyn_full jobs (ARCH=$ARCH, LABEL=$LABEL)"

# d=12 — bqafull baseline 11.1/27.9/61.2/112.2 min for 1e18/2.15e18/4.64e18/1e19
submit_one 12 "1e18"    "$PART_SHORT" "01:30:00"
submit_one 12 "2.15e18" "$PART_LONG"  "03:30:00"
submit_one 12 "4.64e18" "$PART_LONG"  "06:30:00"
submit_one 12 "1e19"    "$PART_LONG"  "11:30:00"

# d=16 — bqafull baseline 8.9/19.6/42.0/89.4 min
submit_one 16 "1e18"    "$PART_SHORT" "01:30:00"
submit_one 16 "2.15e18" "$PART_SHORT" "02:30:00"
submit_one 16 "4.64e18" "$PART_LONG"  "05:00:00"
submit_one 16 "1e19"    "$PART_LONG"  "09:30:00"

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
