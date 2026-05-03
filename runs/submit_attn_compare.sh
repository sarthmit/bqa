#!/bin/bash
# Submits MHA / GQA / BQA comparison jobs at four FLOPs budgets.
# All non-MHA runs use --num-iterations equal to MHA's auto-derived num_iters at
# the same (flops, depth). MHA num_iters are pre-computed analytically (no
# training needed) by scripts/cache_mha_iters.py and written to a shared CSV
# *before* any sbatch jobs are submitted, so all jobs can run in parallel
# without afterok dependencies.
#
# Packing per spec:
#   1e18    : single packed job, all archs x all depths
#   2.15e18 : one job per arch (mha/gqa/bqa), all depths inside
#   4.64e18 : one job per (arch, depth)
#   1e19    : same as 4.64e18
#
# Wall times are sized to fit the 3 hr partition. Override with WT_<budget>.

set -euo pipefail

cd /scratch/sarthmit/bqa
mkdir -p runs/logs/attn_compare

LABEL="${LABEL:-attncmp_apr28}"
DEPTHS="${DEPTHS:-12 16 20}"
PART="${PART:-gpubase_bynode_b1}"

# Wall times (HH:MM:SS) — env-overridable
WT_1E18="${WT_1E18:-02:55:00}"      # 1 packed job, 3 archs x 3 depths
WT_2_15E18="${WT_2_15E18:-02:45:00}" # 1 job per arch, 3 depths
WT_4_64E18="${WT_4_64E18:-01:30:00}" # 1 job per (arch, depth)
WT_1E19="${WT_1E19:-02:45:00}"       # 1 job per (arch, depth)

# ---- Pre-cache MHA num_iters analytically so non-MHA jobs don't need afterok ----
echo ">>> Pre-caching MHA num_iters for LABEL=$LABEL DEPTHS='$DEPTHS' ..."
# setup_node.sh activates the venv and points NANOCHAT_BASE_DIR at scratch under
# the repo, matching what runs/attn_compare_4xh100.sbatch does inside the job.
source scripts/setup_node.sh
python -m scripts.cache_mha_iters \
    --label "$LABEL" \
    --flops 1e18 2.15e18 4.64e18 1e19 \
    --depths $DEPTHS

submit() {
    # echoes the submitted job id
    local name="$1" archs="$2" flops="$3" depths="$4" wt="$5"
    sbatch \
        --job-name="$name" \
        --partition="$PART" \
        --time="$wt" \
        --export=ALL,FLOPS="$flops",DEPTHS="$depths",ARCHS="$archs",LABEL="$LABEL" \
        runs/attn_compare_4xh100.sbatch \
        | awk '{print $NF}'
}

echo ">>> Submitting jobs (PART=$PART)"

# 1e18: one packed job, all archs x all depths
J=$(submit "${LABEL}_f1e18_all" "mha gqa bqa" "1e18" "$DEPTHS" "$WT_1E18")
echo "  1e18 packed:           job=$J"

# 2.15e18: one job per arch, all depths inside (no afterok needed)
for a in mha gqa bqa; do
    J=$(submit "${LABEL}_f2.15e18_${a}" "$a" "2.15e18" "$DEPTHS" "$WT_2_15E18")
    echo "  2.15e18 ${a}:           job=$J"
done

# 4.64e18 and 1e19: per (arch, depth)
for row in "4.64e18 $WT_4_64E18" "1e19 $WT_1E19"; do
    read -r flops wt <<<"$row"
    for d in $DEPTHS; do
        for a in mha gqa bqa; do
            J=$(submit "${LABEL}_f${flops}_${a}_d${d}" "$a" "$flops" "$d" "$wt")
            echo "  $flops ${a} d=${d}:    job=$J"
        done
    done
done

echo ""
echo "submitted; check with: squeue -u $(whoami) -o '%A %j %P %T %L %R'"
