#!/bin/bash
#SBATCH -J cmp_bqa_jH_2k
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=256G
#SBATCH -c 48
#SBATCH -p short-unkillable
#SBATCH -t 2:55:00
#SBATCH --output=/network/scratch/m/mittalsa/bqa/.cache/sbatch/cmp_bqa_jH_2k_%j.log
#
# 2000-step BQA static + bqa_dyn back-to-back at J = n_kv_head = n_head
# (no GQA-style basis sharing — each query head has its own basis). Uses the
# updated defaults m_k=0.70 / m_v=0.85 from the parser. DBS halved to 16 to
# keep peak memory in bounds for bqa_dyn at head_dim_eff = J*D = 768.
# Total batch size unchanged (DBS 16 × 4 ranks × T 2048 × grad_accum 4 = 524288).

set -uo pipefail

REPO=/home/mila/m/mittalsa/scratch/bqa
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=FlexHybrid

source scripts/setup_node.sh

LOG_S="$REPO/.cache/cmp2k_bqa_static_jH_${SLURM_JOB_ID}.log"
LOG_D="$REPO/.cache/cmp2k_bqa_dyn_jH_${SLURM_JOB_ID}.log"

stage() { echo "===== $(date '+%F %T') :: $* ====="; }
stage "host=$(hostname) gpus=$(nvidia-smi -L | wc -l) jobid=$SLURM_JOB_ID"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
python -c "from nanochat.gpt import GPTConfig; c = GPTConfig(); print(f'init mass defaults: m_k={c.bqa_init_mass_k} m_v={c.bqa_init_mass_v}')"

export OMP_NUM_THREADS=1

COMMON=(
    --depth=12
    --device-batch-size=16
    --num-iterations=2000
    --target-param-data-ratio=-1
    --window-pattern=L
    --core-metric-every=-1
    --core-metric-max-per-task=-1
    --sample-every=-1
    --save-every=-1
    --eval-every=250
    --n-kv-head=6
)

stage "BQA static, J=H=6: 2000 iters, attn-kind=bqa, DBS=16, m_k=0.70 m_v=0.85"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa \
    --run=d12_bqa_static_jH_2k \
    --model-tag=d12_bqa_static_jH_2k 2>&1 | tee "$LOG_S"
RC_S=${PIPESTATUS[0]}
stage "BQA static jH finished rc=$RC_S"

sleep 10

stage "bqa_dyn, J=H=6: 2000 iters, attn-kind=bqa_dyn, DBS=16, m_k=0.70 m_v=0.85"
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    "${COMMON[@]}" \
    --attn-kind=bqa_dyn \
    --run=d12_bqa_dyn_jH_2k \
    --model-tag=d12_bqa_dyn_jH_2k 2>&1 | tee "$LOG_D"
RC_D=${PIPESTATUS[0]}
stage "bqa_dyn jH finished rc=$RC_D"

stage "DONE — static log: $LOG_S ; dyn log: $LOG_D"
