#!/bin/bash
#SBATCH -J cmp_gqa_2k
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/cmp_gqa_2k_%j.log
#
# 2000-step GQA run (n_kv_head=3) — companion to cmp_mha_bqa_dyn_2k.sbatch.
# Same DBS, same window-pattern, same depth — n_kv_head matches bqa_dyn (=3),
# so this is the direct hard-routing baseline against bqa_dyn's soft-mix.

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=FlexHybrid

source scripts/setup_node.sh

LOG="$REPO/.cache/cmp2k_gqa_${SLURM_JOB_ID}.log"

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l) jobid=$SLURM_JOB_ID"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

export OMP_NUM_THREADS=1

stage "GQA: 2000 iters, attn-kind=gqa n-kv-head=3, DBS=32"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=32 \
    --num-iterations=2000 \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
    --core-metric-every=-1 \
    --core-metric-max-per-task=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --eval-every=250 \
    --attn-kind=gqa \
    --n-kv-head=3 \
    --run=d12_gqa_2k \
    --model-tag=d12_gqa_2k 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
stage "GQA finished rc=$RC"

stage "DONE — log: $LOG"
