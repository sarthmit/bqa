#!/bin/bash
#SBATCH -J cmp_bqa_static_2k
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/cmp_bqa_static_2k_%j.log
#
# 2000-step static BQA run (n_kv_head=3, default mass 0.58) — fills the missing
# baseline alongside MHA / GQA / bqa_dyn 2k runs. Same DBS, same window-pattern,
# same depth — only attention differs. Default alpha optimizer settings.

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=FlexHybrid

source scripts/setup_node.sh

LOG="$REPO/.cache/cmp2k_bqa_static_${SLURM_JOB_ID}.log"

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l) jobid=$SLURM_JOB_ID"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

export OMP_NUM_THREADS=1

stage "BQA static: 2000 iters, attn-kind=bqa n-kv-head=3, DBS=32, mass_{k,v}=0.58"
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
    --attn-kind=bqa \
    --n-kv-head=3 \
    --run=d12_bqa_static_2k \
    --model-tag=d12_bqa_static_2k 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
stage "BQA static finished rc=$RC"

stage "DONE — log: $LOG"
