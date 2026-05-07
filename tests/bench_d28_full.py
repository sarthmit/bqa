"""Quick d=28 full J=14 bench to validate the J>=14 fwd config update."""
import sys, time
sys.path.insert(0, "/scratch/sarthmit/bqa")
import torch
from nanochat.gpt import GPT, GPTConfig

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
DTYPE = torch.bfloat16
T = 2048

cfg = GPTConfig(
    sequence_len=T, vocab_size=32768,
    n_layer=28, n_head=14, n_kv_head=14, n_embd=28*64,  # d=28 full J=14
    attn_kind="bqa_dyn", bqa_dyn_use_triton=True,
)
torch.manual_seed(0)
model = GPT(cfg).to(device=device, dtype=DTYPE)
model.init_weights()
model_c = torch.compile(model, dynamic=False)

dbs = 8
x = torch.randint(0, cfg.vocab_size, (dbs, T), device=device)
y = torch.randint(0, cfg.vocab_size, (dbs, T), device=device)

print("warmup...")
for _ in range(3):
    loss = model_c(x, y)
    loss.backward()
torch.cuda.synchronize()

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    for p in model.parameters():
        p.grad = None
    loss = model_c(x, y)
    loss.backward()
torch.cuda.synchronize()
ms_per_step = (time.perf_counter() - t0) * 1000.0 / iters

N = sum(p.numel() for p in model.parameters() if p.requires_grad)
tokens = dbs * T
flops = 6 * N * tokens
tflops_s = flops / (ms_per_step * 1e-3) / 1e12
mfu = tflops_s / 989 * 100

print(f"d=28 full J=14: dbs={dbs}, params={N/1e6:.1f}M, ms/step={ms_per_step:.2f}, tok/sec={tokens/(ms_per_step*1e-3):.0f}, MFU={mfu:.2f}%")
