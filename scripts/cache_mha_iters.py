"""
Pre-compute MHA num_iterations for (target_flops, depth) pairs and append them
to the attn_compare iters cache, so GQA/BQA jobs can read it without an afterok
dependency on a real MHA training job.

Mirrors the formula in scripts/base_train.py (build_model_meta -> estimate_flops
-> auto total_batch_size -> num_iters = round(target_flops / (flops_per_token *
total_batch_size))). The MHA config is built with n_kv_head = n_head.

Usage (from repo root, .venv active):
    python -m scripts.cache_mha_iters \
        --label attncmp_apr28 \
        --flops 1e18 2.15e18 4.64e18 1e19 \
        --depths 12 16 20

Writes to: $NANOCHAT_BASE_DIR/attn_compare_${LABEL}_iters.csv
"""
import argparse
import math
import os

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

# Defaults must match the CLI defaults in scripts/base_train.py.
ASPECT_RATIO = 64
HEAD_DIM = 128
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
B_REF = 2 ** 19  # 524,288 (matches base_train.py)


def _build_meta(depth, vocab_size, n_kv_head, attn_kind="gqa"):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    if n_kv_head <= 0:
        n_kv_head = max(1, num_heads // 2)
    assert num_heads % n_kv_head == 0
    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=n_kv_head, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN, attn_kind=attn_kind,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model, num_heads


def _scaling_params(model):
    pc = model.num_scaling_params()
    return pc["transformer_matrices"] + pc["lm_head"]


def compute_mha_num_iters(target_flops, depth, vocab_size, d12_scaling_params):
    # MHA = full n_kv_head (== n_head)
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    n_head = model_dim // HEAD_DIM
    model, _ = _build_meta(depth, vocab_size, n_kv_head=n_head, attn_kind="gqa")

    flops_per_token = model.estimate_flops()
    sp = _scaling_params(model)

    # In base_train.py the target_param_data_ratio multiplier cancels in
    # batch_size_ratio = target_tokens / D_REF (both share the same multiplier),
    # so the ratio reduces to scaling_params / d12_scaling_params.
    batch_size_ratio = sp / d12_scaling_params
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))

    num_iters = round(target_flops / (flops_per_token * total_batch_size))
    return num_iters, total_batch_size, flops_per_token, sp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="experiment label (matches LABEL in submit_attn_compare.sh)")
    p.add_argument("--flops", nargs="+", required=True,
                   help="target FLOPs values, e.g. 1e18 2.15e18 4.64e18 1e19")
    p.add_argument("--depths", nargs="+", type=int, required=True,
                   help="depths, e.g. 12 16 20")
    p.add_argument("--out", default=None,
                   help="output CSV path (default: $NANOCHAT_BASE_DIR/attn_compare_{label}_iters.csv)")
    args = p.parse_args()

    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or get_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    out_path = args.out or os.path.join(base_dir, f"attn_compare_{args.label}_iters.csv")
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("flops,depth,num_iters\n")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # d12 reference must match base_train.py line 283: d12, n_kv_head=-1 (default n_head//2), gqa
    d12_ref, _ = _build_meta(12, vocab_size, n_kv_head=-1, attn_kind="gqa")
    d12_sp = _scaling_params(d12_ref)
    print(f"vocab_size={vocab_size:,}  d12_scaling_params={d12_sp:,}")

    print(f"Writing to: {out_path}")
    with open(out_path, "a") as f:
        for flops_str in args.flops:
            target_flops = float(flops_str)
            for d in args.depths:
                ni, bs, fpt, sp = compute_mha_num_iters(target_flops, d, vocab_size, d12_sp)
                f.write(f"{flops_str},{d},{ni}\n")
                f.flush()
                print(f"  flops={flops_str:>8} depth={d:>2}  num_iters={ni:>6,}  "
                      f"batch={bs:,}  flops/tok={fpt:.3e}  scaling_params={sp:,}")


if __name__ == "__main__":
    main()
