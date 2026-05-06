#!/bin/bash
# 2026-05-04 v2: bqa_dyn triton sweep extended to d={16, 20, 28} across all 4 flops
# cells, with the eval-time torch.compile recompilation fixed (base_train.py now
# uses orig_model for evaluate_bpb, matching the existing pattern for evaluate_core).
#
# d=12 is already in queue (38675485 running 1e18, 38675488/9/94/95/500/501 pending
# for the other 3 flops). This submits the remaining 3 depths.
#
# d=16, d=20: iter cache (attn_compare_attncmp_apr28_iters.csv) already has MHA
# entries, so submit bqa_dyn_triton + bqa_dyn_triton_full directly.
# d=28: MHA cache empty for that depth, so pack MHA into the same sbatch — the
# attn_compare loop runs MHA first (caches iters), then bqa_dyn_triton variants.
#
# Walltime model (anchored to user-observed 40 min for d=12 1e18 triton half):
#   T_half(F) = 40 min × (F / 1e18)        — constant in depth at fixed flops
#   T_full(F) = 1.2 × T_half(F)            — full ~1.0-1.2× half (matmul rank-invariant)
#   T_mha(F)  = 0.8 × T_half(F)            — MHA is fastest per token (no mixing)
#
# Packing strategy (each job ≤ partition cap):
#   d=16/d=20:
#     1e18    : pair (half+full)            → SHORT  1:30
#     2.15e18 : singletons (half / full)    → SHORT  2:30 / 2:45
#     4.64e18 : singletons                  → LONG   4:30 / 5:30
#     1e19    : singletons                  → LONG   9:00 / 10:30
#   d=28 (MHA must run first for cache):
#     1e18    : MHA + half + full packed    → SHORT  2:30   (~32+40+48 = 120 min)
#     2.15e18 : MHA + half + full packed    → LONG   4:30   (~258 min)
#     4.64e18 : MHA + half + full packed    → LONG   9:30   (~556 min)
#     1e19    : split (busts 12hr cap when packed):
#               MHA alone                   → LONG   5:30   (no dep)
#               half (dep on MHA)           → LONG   9:00
#               full (dep on MHA)           → LONG   10:30

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
LABEL="${LABEL:-attncmp_apr28}"
EXCLUDE="${EXCLUDE:-fc10201}"

submit_one() {
    local archs="$1" depth="$2" flops="$3" part="$4" wt="$5" name="$6" dep="${7:-}"
    local depflag=""
    if [ -n "$dep" ]; then depflag="--dependency=afterok:$dep"; fi
    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --exclude="$EXCLUDE" \
        $depflag \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ARCHS="$archs",LABEL="$LABEL" \
        runs/attn_compare_4xh100.sbatch | awk '{print $NF}')
    printf "  %-60s d=%-2s flops=%-8s archs=%-45s wt=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$archs" "$wt" "$jid" >&2
    echo "$jid"
}

echo ">>> d=16, d=20: bqa_dyn_triton + bqa_dyn_triton_full across all 4 flops"
for d in 16 20; do
    # 1e18 pair packed
    submit_one "bqa_dyn_triton bqa_dyn_triton_full" "$d" "1e18" "$PART_SHORT" "01:30:00" \
        "${LABEL}_bqa_dyn_triton_pair_d${d}_f1e18" > /dev/null

    # 2.15e18 singletons
    submit_one "bqa_dyn_triton"      "$d" "2.15e18" "$PART_SHORT" "02:30:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f2.15e18" > /dev/null
    submit_one "bqa_dyn_triton_full" "$d" "2.15e18" "$PART_SHORT" "02:45:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f2.15e18" > /dev/null

    # 4.64e18 singletons
    submit_one "bqa_dyn_triton"      "$d" "4.64e18" "$PART_LONG"  "04:30:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f4.64e18" > /dev/null
    submit_one "bqa_dyn_triton_full" "$d" "4.64e18" "$PART_LONG"  "05:30:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f4.64e18" > /dev/null

    # 1e19 singletons
    submit_one "bqa_dyn_triton"      "$d" "1e19"    "$PART_LONG"  "09:00:00" \
        "${LABEL}_bqa_dyn_triton_d${d}_f1e19" > /dev/null
    submit_one "bqa_dyn_triton_full" "$d" "1e19"    "$PART_LONG"  "10:30:00" \
        "${LABEL}_bqa_dyn_triton_full_d${d}_f1e19" > /dev/null
done

echo ""
echo ">>> d=28: pack MHA into each sbatch so iter cache is populated in-job"
echo "    (MHA cache has no d=28 entries — depends on this sweep producing them)"
# 1e18 — MHA + half + full all in one SHORT job
submit_one "mha bqa_dyn_triton bqa_dyn_triton_full" 28 "1e18" "$PART_SHORT" "02:30:00" \
    "${LABEL}_bqa_dyn_triton_mhapack_d28_f1e18" > /dev/null

# 2.15e18 — MHA + half + full LONG
submit_one "mha bqa_dyn_triton bqa_dyn_triton_full" 28 "2.15e18" "$PART_LONG" "04:30:00" \
    "${LABEL}_bqa_dyn_triton_mhapack_d28_f2.15e18" > /dev/null

# 4.64e18 — MHA + half + full LONG (~9.3hr packed)
submit_one "mha bqa_dyn_triton bqa_dyn_triton_full" 28 "4.64e18" "$PART_LONG" "09:30:00" \
    "${LABEL}_bqa_dyn_triton_mhapack_d28_f4.64e18" > /dev/null

# 1e19 — packing busts 12hr LONG cap; split into MHA-alone + half + full with dep on MHA
mha_d28_1e19_jid=$(submit_one "mha" 28 "1e19" "$PART_LONG" "05:30:00" \
    "${LABEL}_mha_d28_f1e19")
submit_one "bqa_dyn_triton"      28 "1e19" "$PART_LONG" "09:00:00" \
    "${LABEL}_bqa_dyn_triton_d28_f1e19"      "$mha_d28_1e19_jid" > /dev/null
submit_one "bqa_dyn_triton_full" 28 "1e19" "$PART_LONG" "10:30:00" \
    "${LABEL}_bqa_dyn_triton_full_d28_f1e19" "$mha_d28_1e19_jid" > /dev/null

echo ""
echo "Inspect: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
