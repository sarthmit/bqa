#!/bin/bash
# Scaling-law sweep: 4 flops budgets per depth, staircase-staggered around
# each depth's Chinchilla-optimal compute. Default arch is BQA with the new
# (m_k=0.70, m_v=0.85) defaults set 2026-04-29. Override with ATTN_KIND env.
#
# Grid (24 runs, 4 budgets per depth, shift by ~2.15x per +4 depth):
#   d=12: 1e18,    2.15e18, 4.64e18, 1e19      (Chinchilla-opt ≈ 1e18)
#   d=16: 2.15e18, 4.64e18, 1e19,    2.15e19   (opt ≈ 4.64e18)
#   d=20: 4.64e18, 1e19,    2.15e19, 4.64e19   (opt ≈ 1.87e19)
#   d=24: 1e19,    2.15e19, 4.64e19, 1e20      (opt ≈ 5.5e19)
#   d=28: 2.15e19, 4.64e19, 1e20,    2.15e20   (opt ≈ 1.4e20)
#   d=32: 4.64e19, 1e20,    2.15e20, 4.64e20   (opt ≈ 3.1e20)
#
# Budgets used (8 distinct):
#   1e18, 2.15e18, 4.64e18, 1e19, 2.15e19, 4.64e19, 1e20, 2.15e20, 4.64e20
#
# Wall-time estimates on 4xH100 (~1.1e17 flops/min):
#    1e18    ≈   9-11 min
#    2.15e18 ≈  17-23 min
#    4.64e18 ≈  38-50 min
#    1e19    ≈  80-110 min   (last budget that fits 3hr per-depth)
#    2.15e19 ≈ 175-230 min   (borderline, prefer long partition)
#    4.64e19 ≈ 380-500 min   (~6-8hr, long partition)
#    1e20    ≈ 815-1075 min  (~14-18hr, long partition)
#    2.15e20 ≈ 1750-2300 min (~29-38hr, multi-day partition)
#    4.64e20 ≈ 3800-5000 min (~63-83hr, multi-day partition)
#
# Three partitions (set via env):
#   PART_SHORT (3hr)  : flops <= 1e19
#   PART_LONG         : 2.15e19, 4.64e19, 1e20
#   PART_MULTIDAY     : 2.15e20, 4.64e20
#
# Stage submissions with MAX_FLOPS to cap the sweep:
#   MAX_FLOPS=1e19  ./submit_scaling_d12_d32.sh   (12 runs, all on 3hr partition)
#   MAX_FLOPS=1e20  ./submit_scaling_d12_d32.sh   (21 runs, needs PART_LONG)
#   MAX_FLOPS=4.64e20 ./submit_scaling_d12_d32.sh (24 runs, needs PART_MULTIDAY)

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/scaling

LABEL="${LABEL:-scaling_apr29}"
ATTN_KIND="${ATTN_KIND:-bqa}"
PART_SHORT="${PART_SHORT:-gpubase_bynode_b1}"
PART_LONG="${PART_LONG:-}"            # required for f in {2.15e19, 4.64e19, 1e20}
PART_MULTIDAY="${PART_MULTIDAY:-}"    # required for f in {2.15e20, 4.64e20}
MAX_FLOPS="${MAX_FLOPS:-4.64e20}"     # cap to stage submissions

# --- Grid: per-depth flops list (env-overridable) ---
DEPTHS_FLOPS=(
    "12 : 1e18    2.15e18 4.64e18 1e19"
    "16 : 2.15e18 4.64e18 1e19    2.15e19"
    "20 : 4.64e18 1e19    2.15e19 4.64e19"
    "24 : 1e19    2.15e19 4.64e19 1e20"
    "28 : 2.15e19 4.64e19 1e20    2.15e20"
    "32 : 4.64e19 1e20    2.15e20 4.64e20"
)

# --- Per-budget walltimes (HH:MM:SS) ---
declare -A WT=(
    [1e18]="01:00:00"
    [2.15e18]="01:30:00"
    [4.64e18]="02:30:00"
    [1e19]="02:30:00"
    [2.15e19]="04:30:00"
    [4.64e19]="09:00:00"
    [1e20]="20:00:00"
    [2.15e20]="40:00:00"
    [4.64e20]="84:00:00"
)

# --- Per-budget partition assignment ---
partition_for_flops() {
    case "$1" in
        1e18|2.15e18|4.64e18|1e19) echo "$PART_SHORT" ;;
        2.15e19|4.64e19|1e20)      echo "$PART_LONG" ;;
        2.15e20|4.64e20)           echo "$PART_MULTIDAY" ;;
        *) echo "ERROR: unknown flops '$1'" >&2; return 1 ;;
    esac
}

# Numeric comparison via bc (avoids bash float issues with 2.15e18 etc.)
flops_le_max() {
    local f="$1" m="$2"
    [ "$(echo "$f <= $m" | bc -l)" -eq 1 ]
}

submit() {
    local name="$1" part="$2" flops="$3" depth="$4" wt="$5"
    sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depth",ATTN_KIND="$ATTN_KIND",LABEL="$LABEL" \
        runs/scaling_laws_4xh100.sbatch \
        | awk '{print $NF}'
}

echo ">>> Scaling-law sweep submit"
echo "    LABEL=$LABEL  ATTN_KIND=$ATTN_KIND  MAX_FLOPS=$MAX_FLOPS"
echo "    PART_SHORT=$PART_SHORT"
echo "    PART_LONG=${PART_LONG:-<unset>}"
echo "    PART_MULTIDAY=${PART_MULTIDAY:-<unset>}"
echo ""

n_submitted=0
n_skipped=0
errors=0

for row in "${DEPTHS_FLOPS[@]}"; do
    depth="${row%% :*}"
    flops_list="${row#*: }"
    depth="$(echo "$depth" | xargs)"  # trim
    for f in $flops_list; do
        if ! flops_le_max "$f" "$MAX_FLOPS"; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        part="$(partition_for_flops "$f")" || { errors=$((errors+1)); continue; }
        if [ -z "$part" ]; then
            echo "ERROR: no partition set for flops=$f (set PART_LONG / PART_MULTIDAY); skipping d=$depth f=$f" >&2
            errors=$((errors+1))
            continue
        fi
        wt="${WT[$f]}"
        name="${LABEL}_${ATTN_KIND}_d${depth}_f${f}"
        J=$(submit "$name" "$part" "$f" "$depth" "$wt")
        printf "  %-20s d=%-2s  part=%-22s wt=%s  job=%s\n" "$f" "$depth" "$part" "$wt" "$J"
        n_submitted=$((n_submitted + 1))
    done
done

echo ""
echo "submitted: $n_submitted  skipped (>MAX_FLOPS): $n_skipped  errors: $errors"
echo "check with: squeue -u $(whoami) -o '%A %j %P %T %L %R'"
echo "logs land in: /scratch/sarthmit/bqa/runs/logs/scaling/${LABEL}_*"
