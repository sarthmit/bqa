# shellcheck shell=bash
# Source this file (do not execute) before any `uv` / training command for nanochat.
#
# Goal: keep ALL caches and venv state under the repo's .cache/ directory
# (BeeGFS scratch, plenty of quota), and never under $HOME (tight quota on Mila).
#
# Usage:
#   source scripts/setup_node.sh
#   uv sync --extra gpu
#   uv run python -m nanochat.something
#
# Notes:
# - Unlike prime-rl, nanochat does NOT need a node-local ext4 venv: it doesn't
#   import the heavy `transformers` dev package whose import-time metadata scan
#   trips BeeGFS. So the `.venv` and uv cache both live under the repo on scratch.
# - If you add a new tool that writes to $HOME/.cache/<foo>, redirect it here.

# Resolve the repo root (the parent of this scripts/ directory), following symlinks.
_NANOCHAT_SETUP_SCRIPT="${BASH_SOURCE[0]:-$0}"
NANOCHAT_REPO="$(cd "$(dirname "$(readlink -f "${_NANOCHAT_SETUP_SCRIPT}")")/.." && pwd)"
unset _NANOCHAT_SETUP_SCRIPT
export NANOCHAT_REPO

NANOCHAT_CACHE="${NANOCHAT_REPO}/.cache"
# NANOCHAT_BASE_DIR is where the project writes its own artefacts (tokenizer,
# dataset shards, base_checkpoints, report/, etc.) — same idea as the script's
# default of $HOME/.cache/nanochat, just relocated under the repo.
export NANOCHAT_BASE_DIR="${NANOCHAT_REPO}/.cache/nanochat"
mkdir -p \
    "${NANOCHAT_BASE_DIR}" \
    "${NANOCHAT_CACHE}/uv" \
    "${NANOCHAT_CACHE}/uv-python" \
    "${NANOCHAT_CACHE}/pip" \
    "${NANOCHAT_CACHE}/xdg" \
    "${NANOCHAT_CACHE}/hf" \
    "${NANOCHAT_CACHE}/wandb" \
    "${NANOCHAT_CACHE}/wandb-config" \
    "${NANOCHAT_CACHE}/triton" \
    "${NANOCHAT_CACHE}/torchinductor"

# uv: where uv installs Python toolchains and where it caches wheels/sdists.
# (For nanochat we keep UV_CACHE_DIR under the repo too — different from prime-rl,
# which had to move it to ext4 because BeeGFS broke its rename(2)-based wheel build
# cache for `transformers` etc. nanochat's deps don't trip that.)
export UV_PYTHON_INSTALL_DIR="${NANOCHAT_CACHE}/uv-python"
export UV_CACHE_DIR="${NANOCHAT_CACHE}/uv"

# pip / generic XDG fallbacks — keep anything that respects them off $HOME.
export PIP_CACHE_DIR="${NANOCHAT_CACHE}/pip"
export XDG_CACHE_HOME="${NANOCHAT_CACHE}/xdg"

# HuggingFace (datasets, hub, transformers all read HF_HOME).
export HF_HOME="${NANOCHAT_CACHE}/hf"

# Weights & Biases.
export WANDB_CACHE_DIR="${NANOCHAT_CACHE}/wandb"
export WANDB_CONFIG_DIR="${NANOCHAT_CACHE}/wandb-config"
# Where wandb writes per-run logs (default is ./wandb in cwd).
export WANDB_DIR="${NANOCHAT_CACHE}/wandb"

# Triton & torch.compile / inductor.
# These caches use atomic rename(2) to publish compiled kernels — BeeGFS doesn't
# handle that reliably under concurrent (multi-rank torchrun) writes, so we redirect
# them to node-local ext4 ($SLURM_TMPDIR) under SLURM. Login-node smoke tests fall
# back to the BeeGFS .cache/ paths.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TRITON_CACHE_DIR="${SLURM_TMPDIR}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR}/torchinductor"
    mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    echo "[nanochat] compile caches on \$SLURM_TMPDIR=${SLURM_TMPDIR}"
else
    export TRITON_CACHE_DIR="${NANOCHAT_CACHE}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${NANOCHAT_CACHE}/torchinductor"
fi

# venv: under SLURM we rsync .venv to $SLURM_TMPDIR/nanochat-venv on ext4 and
# activate the local copy. This is needed because torch.compile's inductor recursively
# walks the torch source tree at first-compile time, and concurrent multi-rank reads
# trip BeeGFS metadata caches (intermittent FileNotFoundError on existing .py files).
# Outside SLURM we activate the BeeGFS .venv directly (login-node smoke tests).
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${NANOCHAT_REPO}/.venv" ]; then
    NANOCHAT_LOCAL_VENV="${SLURM_TMPDIR}/nanochat-venv"
    if [ ! -e "${NANOCHAT_LOCAL_VENV}/bin/python" ]; then
        echo "[nanochat] rsyncing .venv -> ${NANOCHAT_LOCAL_VENV} (BeeGFS->ext4)"
        time rsync -a --delete "${NANOCHAT_REPO}/.venv/" "${NANOCHAT_LOCAL_VENV}/"
        # uv hard-codes the absolute path of the source venv into bin/* scripts in
        # several places: shebangs (`#!/<repo>/.venv/bin/python` in `torchrun`, etc.)
        # AND the `VIRTUAL_ENV='<repo>/.venv'` line inside `activate`/`activate.csh`/
        # `activate.fish`. If we don't rewrite all of them, sourcing activate would
        # set VIRTUAL_ENV+PATH back to the BeeGFS venv and defeat the rsync. One
        # global sed over bin/* covers every occurrence.
        echo "[nanochat] rewriting venv-bin path references: ${NANOCHAT_REPO}/.venv -> ${NANOCHAT_LOCAL_VENV}"
        find "${NANOCHAT_LOCAL_VENV}/bin" -maxdepth 1 -type f -exec sed -i \
            "s|${NANOCHAT_REPO}/\.venv|${NANOCHAT_LOCAL_VENV}|g" {} +
    else
        echo "[nanochat] reusing existing local venv at ${NANOCHAT_LOCAL_VENV}"
    fi
    # shellcheck source=/dev/null
    source "${NANOCHAT_LOCAL_VENV}/bin/activate"
    echo "[nanochat] venv (local): ${VIRTUAL_ENV}"
elif [ -d "${NANOCHAT_REPO}/.venv" ]; then
    # shellcheck source=/dev/null
    source "${NANOCHAT_REPO}/.venv/bin/activate"
    echo "[nanochat] venv (BeeGFS): ${VIRTUAL_ENV}"
fi

echo "[nanochat] repo:  ${NANOCHAT_REPO}"
echo "[nanochat] cache: ${NANOCHAT_CACHE}"
