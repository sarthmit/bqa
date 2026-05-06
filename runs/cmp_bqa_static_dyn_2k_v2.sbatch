#!/bin/bash
#SBATCH -J cmp_bqa_static_dyn_2k_v2
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/cmp_bqa_static_dyn_2k_v2_%j.log
#
# 2000-step BQA static + bqa_dyn back-to-back at the upstream-updated defaults
# (m_k=0.70, m_v=0.85 from the apr28 d16 sweep). Defaults flow through the
# parser so we don't need explicit --bqa-init-mass-{k,v} flags.

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=FlexHybrid

source scripts/setup_node.sh

LOG_S="$REPO/.cache/cmp2k_bqa_static_v2_${SLURM_JOB_ID}.log"
LOG_D="$REPO/.cache/cmp2k_bqa_dyn_v2_${SLURM_JOB_ID}.log"

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l) jobid=$SLURM_JOB_ID"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
python -c "from nanochat.gpt import GPTConfig; c = GPTConfig(); print(f'init mass defaults: m_k={c.bqa_init_mass_k} m_v={c.bqa_init_mass_v}')"

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

stage "BQA static v2: 2000 iters, attn-kind=bqa n-kv-head=3, m_k=0.70 m_v=0.85"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa \
    --n-kv-head=3 \
    --run=d12_bqa_static_2k_v2 \
    --model-tag=d12_bqa_static_2k_v2 2>&1 | tee "$LOG_S"
RC_S=${PIPESTATUS[0]}
stage "BQA static v2 finished rc=$RC_S"

sleep 10

stage "bqa_dyn v2: 2000 iters, attn-kind=bqa_dyn n-kv-head=3, m_k=0.70 m_v=0.85"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa_dyn \
    --n-kv-head=3 \
    --run=d12_bqa_dyn_2k_v2 \
    --model-tag=d12_bqa_dyn_2k_v2 2>&1 | tee "$LOG_D"
RC_D=${PIPESTATUS[0]}
stage "bqa_dyn v2 finished rc=$RC_D"

stage "DONE — static log: $LOG_S ; dyn log: $LOG_D"
