#!/bin/bash
set -uo pipefail

cd /scratch/sarthmit/bqa
source .venv/bin/activate
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p runs/logs

echo "=== $(date -Is) | downloading climbmix shards ==="
python -m nanochat.dataset -n 170 -w 8 2>&1 | tee runs/logs/dataset_download.log
DL_RC=${PIPESTATUS[0]}
if [ "$DL_RC" -ne 0 ]; then
    echo "=== dataset download exited $DL_RC; aborting training ==="
    exit "$DL_RC"
fi

echo "=== $(date -Is) | starting training driver ==="
bash runs/d12_attn_compare.sh
