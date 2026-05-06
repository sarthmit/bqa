#!/bin/bash
#SBATCH -J cmp_mha_bqa_dyn_2k
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/cmp_mha_bqa_dyn_2k_%j.log
#
# Iso-iteration comparison at 2000 steps: MHA vs bqa_dyn at d12 (4x longer than the
# 500-step probe run). Same DBS, same window-pattern. MHA first, then bqa_dyn.
# Estimated wall time: ~20 min MHA + ~70 min bqa_dyn ≈ 90 min total.

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=FlexHybrid

# setup_node.sh handles cache redirects, .venv rsync to $SLURM_TMPDIR (ext4),
# and venv activation.
source scripts/setup_node.sh

LOG_DIR="$REPO/.cache"
MHA_LOG="$LOG_DIR/cmp2k_mha_${SLURM_JOB_ID}.log"
BQA_LOG="$LOG_DIR/cmp2k_bqa_dyn_${SLURM_JOB_ID}.log"

stage() { echo "===== $(date '+%F %T') :: $* ====="; }

stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l) jobid=$SLURM_JOB_ID"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

export OMP_NUM_THREADS=1

COMMON=(
    --depth=12
    --device-batch-size=32
    --num-iterations=2000
    --target-param-data-ratio=-1
    --window-pattern=L
    --core-metric-every=-1
    --core-metric-max-per-task=-1
    --sample-every=-1
    --save-every=-1
    --eval-every=250
)

stage "MHA: 2000 iters, attn-kind=gqa n-kv-head=6 (= MHA), DBS=32"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=gqa \
    --n-kv-head=6 \
    --run=d12_mha_2k \
    --model-tag=d12_mha_2k 2>&1 | tee "$MHA_LOG"
MHA_RC=${PIPESTATUS[0]}
stage "MHA finished rc=$MHA_RC"

sleep 10

stage "bqa_dyn: 2000 iters, attn-kind=bqa_dyn n-kv-head=3, DBS=32"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa_dyn \
    --n-kv-head=3 \
    --run=d12_bqa_dyn_2k \
    --model-tag=d12_bqa_dyn_2k 2>&1 | tee "$BQA_LOG"
BQA_RC=${PIPESTATUS[0]}
stage "bqa_dyn finished rc=$BQA_RC"

stage "DONE — MHA log: $MHA_LOG ; bqa_dyn log: $BQA_LOG"
