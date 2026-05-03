#!/bin/bash
set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/scaling

LABEL="${LABEL:-scaling_apr28}"
DEPTHS="${DEPTHS:-12 16 20}"

# Packed: one job per FLOPs that runs all DEPTHS sequentially.
PACKED_JOBS=(
    "1e18     gpubase_bynode_b1  02:30:00"
    "2.15e18  gpubase_bynode_b1  02:45:00"
)

# Individual: one job per (FLOPs, depth) pair.
INDIVIDUAL_JOBS=(
    "4.64e18  gpubase_bynode_b1  01:30:00"
    "1e19     gpubase_bynode_b1  02:45:00"
)

submit() {
    local name="$1" part="$2" wt="$3" flops="$4" depths="$5"
    sbatch \
        --job-name="$name" \
        --partition="$part" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depths",LABEL="$LABEL" \
        runs/scaling_laws_4xh100.sbatch
}

for row in "${PACKED_JOBS[@]}"; do
    read -r flops part wt <<<"$row"
    submit "${LABEL}_f${flops}" "$part" "$wt" "$flops" "$DEPTHS"
done

for row in "${INDIVIDUAL_JOBS[@]}"; do
    read -r flops part wt <<<"$row"
    for d in $DEPTHS; do
        submit "${LABEL}_d${d}_f${flops}" "$part" "$wt" "$flops" "$d"
    done
done

echo "submitted; check with: squeue -u $(whoami) -o '%A %j %P %T %L %R'"
