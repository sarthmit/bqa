#!/bin/bash
# 2026-05-04: full bqa_dyn triton sweep across {d12, d16, d20} x {half, full}
# at all 4 flops cells {1e18, 2.15e18, 4.64e18, 1e19}.
#
# Walltime model (anchored to user-observed 40 min for d12 1e18 triton half):
#   T_half(F) = 40 min * (F / 1e18)        — total flops budget controls runtime
#   T_full(F) = 1.2 * T_half(F)            — full ~1.0-1.2x half (matmul rank-invariant)
# Times approximately constant in depth at fixed flops (per-step throughput
# scales with model size; iter count scales inversely → wallclock is constant).
#
# Packing strategy (3hr SHORT cap when feasible, LONG otherwise):
#   1e18    : pair (half+full) per depth in one sbatch → ~88 min, 1:30 SHORT
#   2.15e18 : singleton per cell → ~86-103 min, 2:30 SHORT (pair = 3.15hr, busts SHORT)
#   4.64e18 : singleton per cell → ~186-223 min, 4:30 LONG (busts SHORT alone)
#   1e19    : singleton per cell → ~400-480 min, 9:00 LONG
#
# Total: 3 (1e18 packed pairs) + 6 (2.15e18) + 6 (4.64e18) + 6 (1e19) = 21 jobs.
# All exclude fc10201 (broken node, OOM at torch.cuda.set_device 2026-05-04).

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
LABEL="${LABEL:-attncmp_apr28}"
EXCLUDE="${EXCLUDE:-fc10201}"

submit_one() {
    local archs="$1" depth="$2" flops="$3" part="$4" wt="$5" name="$6"
    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --exclude="$EXCLUDE" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ARCHS="$archs",LABEL="$LABEL" \
        runs/attn_compare_4xh100.sbatch | awk '{print $NF}')
    printf "  %-55s d=%-2s flops=%-8s archs=%-40s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$archs" "$part" "$wt" "$jid"
}

echo ">>> 1e18: 3 packed pair jobs (half+full per depth, SHORT)"
for d in 12 16 20; do
    submit_one "bqa_dyn_triton bqa_dyn_triton_full" "$d" "1e18" "$PART_SHORT" "01:30:00" \
        "${LABEL}_bqa_dyn_triton_pair_d${d}_f1e18"
done

echo ""
echo ">>> 2.15e18: 6 singleton jobs (SHORT)"
for d in 12 16 20; do
    submit_one "bqa_dyn_triton"      "$d" "2.15e18" "$PART_SHORT" "02:30:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f2.15e18"
    submit_one "bqa_dyn_triton_full" "$d" "2.15e18" "$PART_SHORT" "02:45:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f2.15e18"
done

echo ""
echo ">>> 4.64e18: 6 singleton jobs (LONG)"
for d in 12 16 20; do
    submit_one "bqa_dyn_triton"      "$d" "4.64e18" "$PART_LONG"  "04:30:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f4.64e18"
    submit_one "bqa_dyn_triton_full" "$d" "4.64e18" "$PART_LONG"  "05:30:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f4.64e18"
done

echo ""
echo ">>> 1e19: 6 singleton jobs (LONG)"
for d in 12 16 20; do
    submit_one "bqa_dyn_triton"      "$d" "1e19"    "$PART_LONG"  "09:00:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f1e19"
    submit_one "bqa_dyn_triton_full" "$d" "1e19"    "$PART_LONG"  "10:30:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f1e19"
done

echo ""
echo "Submitted. Inspect with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
