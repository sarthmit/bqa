"""Minimal sanity test: model with bqa_dyn_use_triton=True + torch.compile
runs forward and backward end-to-end without graph-break busts. Run on a GPU node.
"""
import os, sys, time
sys.path.insert(0, "/scratch/sarthmit/bqa")
import torch
from nanochat.gpt import GPT, GPTConfig

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

# Small d=12 config — quickest sanity check.
cfg = GPTConfig(
    sequence_len=2048, vocab_size=32768,
    n_layer=16, n_head=8, n_kv_head=8, n_embd=1024,  # d=16 full, J=8
    attn_kind="bqa_dyn",
    bqa_dyn_use_triton=True,
)
torch.manual_seed(0)
model = GPT(cfg).to(device=device, dtype=torch.bfloat16)
model.init_weights()
print("model built: J =", cfg.n_kv_head, "n_layer =", cfg.n_layer)

# Wrap in compile, just like base_train.py does.
model_c = torch.compile(model, dynamic=False)

B, T = 4, 2048
x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

# Warmup (triggers compile).
print("warmup pass...")
out = model_c(x, y)
print("  loss:", float(out.detach()))
out.backward()
torch.cuda.synchronize()
print("compile + bwd OK")

# Time a few steps.
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    model.zero_grad()
    out = model_c(x, y)
    out.backward()
torch.cuda.synchronize()
ms_per_step = (time.perf_counter() - t0) * 100.0
print(f"avg step time (10 iters): {ms_per_step:.2f} ms")
