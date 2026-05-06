#!/bin/bash
# 2026-05-04: re-submit the 8 bqa_dyn / bqa_dyn_full cells that failed on
# 2026-05-03 with FA3's "head dimension at most 256" error. Fix landed in
# nanochat/flash_attention.py: route by head_dim, not just hardware — calls
# with head_dim > 256 fall to SDPA (mem_efficient or math backend).
#
# Pending 4.64e18 / 1e19 cells (jids 38517347/8/51/52, 38519364/5/8/9) will
# pick up the fix automatically when they start; we only re-submit the 8
# that already finished with exit=1.
#
# Walltimes:
#   - bqa_dyn (head_dim_eff = depth*32 → 384 at d12, 512 at d16): SDPA's
#     mem_efficient backend handles ≤512 cleanly. Expect ~1.5-2x slower than
#     FA3, so original walltimes (sized at ~5x bqa) get a 1.5x bump.
#   - bqa_dyn_full (head_dim_eff = depth*64 → 768 at d12, 1024 at d16): may
#     fall to SDPA's math backend (mem_efficient cap is around 512 in some
#     PyTorch builds). Math is 5-10x slower than FA3 — large bump, all on
#     PART_LONG.

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
LABEL="${LABEL:-attncmp_apr28}"

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
    printf "  %-50s d=%-2s flops=%-8s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$part" "$wt" "$jid"
}

echo ">>> Re-submitting 8 SDPA-fallback cells (4 bqa_dyn + 4 bqa_dyn_full)"

# bqa_dyn: head_dim_eff 384/512, mem_efficient backend, ~1.5-2x FA3.
# Original walltimes were 90/175/90/150 min; bump to 150/270/150/240.
submit_one bqa_dyn      12 "1e18"    "$PART_SHORT" "02:30:00"
submit_one bqa_dyn      12 "2.15e18" "$PART_LONG"  "04:30:00"
submit_one bqa_dyn      16 "1e18"    "$PART_SHORT" "02:30:00"
submit_one bqa_dyn      16 "2.15e18" "$PART_LONG"  "04:00:00"

# bqa_dyn_full: head_dim_eff 768/1024, likely math backend, 5-10x FA3.
# Original walltimes were 90/210/90/150 min; bump to 360/540/360/480.
submit_one bqa_dyn_full 12 "1e18"    "$PART_LONG"  "06:00:00"
submit_one bqa_dyn_full 12 "2.15e18" "$PART_LONG"  "09:00:00"
submit_one bqa_dyn_full 16 "1e18"    "$PART_LONG"  "06:00:00"
submit_one bqa_dyn_full 16 "2.15e18" "$PART_LONG"  "08:00:00"

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
