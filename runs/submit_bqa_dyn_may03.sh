#!/bin/bash
# Submitted 2026-05-03: bqa_dyn (per-query independent K/V mixing logits)
# at d=12 and d=16 for the 1e18 / 2.15e18 / 4.64e18 / 1e19 budgets.
# Same arch defaults as bqa: bqa_init_mass_k=0.70, bqa_init_mass_v=0.85,
# n_kv_head = n_head/2 (default).
#
# Walltimes are sized for ~5x bqa training time (per empirical guidance for
# d=12) plus ~25min eval/overhead. PART_SHORT (3hr cap) where it fits to get
# faster scheduling; PART_LONG (12hr cap) only for cells that exceed 3hr.
# No packing — one job per (depth, flops) cell.
#
# MHA num_iters cache file used: attn_compare_attncmp_apr28_iters.csv
# (already populated for d12/d16 × {1e18, 2.15e18, 4.64e18, 1e19}).

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"   # 3 h cap
PART_LONG="${PART_LONG:-gpubase_bynode_b2}"     # 12 h cap
ARCH="bqa_dyn"
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
    printf "  %-45s d=%-2s flops=%-8s part=%-21s time=%s -> jid=%s\n" \
        "$name" "$depth" "$flops" "$part" "$wt" "$jid"
}

echo ">>> Submitting bqa_dyn jobs (ARCH=$ARCH, LABEL=$LABEL)"

# d=12: bqa was 11/24/51/112 min for 1e18/2.15e18/4.64e18/1e19.
# At ~5x: 55/120/255/560 + 25min overhead → 80/145/280/585 min.
submit_one 12 "1e18"    "$PART_SHORT" "01:30:00"
submit_one 12 "2.15e18" "$PART_SHORT" "02:55:00"
submit_one 12 "4.64e18" "$PART_LONG"  "05:30:00"
submit_one 12 "1e19"    "$PART_LONG"  "11:00:00"

# d=16: bqa was 9/19/40/88 min for 1e18/2.15e18/4.64e18/1e19.
# At ~5x: 45/95/200/440 + 25min overhead → 70/120/225/465 min.
submit_one 16 "1e18"    "$PART_SHORT" "01:30:00"
submit_one 16 "2.15e18" "$PART_SHORT" "02:30:00"
submit_one 16 "4.64e18" "$PART_LONG"  "04:30:00"
submit_one 16 "1e19"    "$PART_LONG"  "09:00:00"

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %P %j %T %L %R'"
