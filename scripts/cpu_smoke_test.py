"""
CPU smoke test for nanochat — builds a tiny GPT, runs a forward + backward pass
using the SDPA fallback path, and checks BQA / GQA both work.

Run from repo root:
    uv run python scripts/cpu_smoke_test.py
"""
import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON


def run_one(attn_kind: str):
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        attn_kind=attn_kind,
        window_pattern="L",
    )
    model = GPT(cfg).to("cpu")
    model.train()

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    loss = model(idx, targets=targets)
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)

    model.eval()
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    print(f"  [{attn_kind}] loss={loss.item():.4f}  params_with_grad={grad_count}  logits={tuple(logits.shape)}")


def main():
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"compute dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    torch.manual_seed(0)
    for kind in ("gqa", "bqa", "bqa_dyn"):
        run_one(kind)
    print("OK")


if __name__ == "__main__":
    main()
