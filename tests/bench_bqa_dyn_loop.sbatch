#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --job-name=bench_bqa_dyn_loop
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=fc10201
#SBATCH --output=/scratch/sarthmit/bqa/tests/bench_bqa_dyn_loop_%j.log

set -euo pipefail
cd /scratch/sarthmit/bqa

# Use the project venv directly so we don't depend on uv sync at startup.
export VENV=/scratch/sarthmit/bqa/.venv
export PYTHONPATH=/scratch/sarthmit/bqa
export TAG="${TAG:-default}"

echo "=== $(date -Is) | host=$(hostname) | TAG=$TAG ==="
nvidia-smi -L

BENCH_OUT="/scratch/sarthmit/bqa/tests/bench_bqa_dyn_loop_${TAG}.json" \
  $VENV/bin/python tests/bench_bqa_dyn_loop.py
echo "=== $(date -Is) | done ==="
