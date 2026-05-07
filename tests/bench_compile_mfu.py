"""End-to-end model.compile() benchmark across d=16, d=20, d=28 with bqa_dyn_use_triton=True.
Reports step time + MFU. Uses proper warmup."""
import sys, time, math
sys.path.insert(0, "/scratch/sarthmit/bqa")
import torch
from nanochat.gpt import GPT, GPTConfig

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

# Configs: (label, depth, n_kv_head, dbs)  -- match the project's dbs_for_depth
CONFIGS = [
    ("d=16 half J=4", 16, 4, 32),
    ("d=16 full J=8", 16, 8, 32),
    ("d=20 half J=5", 20, 5, 16),
    ("d=20 full J=10", 20, 10, 16),
    ("d=28 half J=7", 28, 7, 8),
    ("d=28 full J=14", 28, 14, 8),
]

T = 2048
DTYPE = torch.bfloat16

def model_params(d, n_head, vocab=32768):
    """Approx; matches GPTConfig defaults."""
    n_embd = d * 64
    head_dim = 128
    # transformer blocks: 4 × n_embd² (attn proj) + 8 × n_embd² (ffn) ≈ 12 n_embd²
    p_block = 12 * n_embd * n_embd
    p_emb = vocab * n_embd
    return d * p_block + 2 * p_emb  # rough

def measure(label, depth, n_kv_head, dbs):
    head_dim = 128
    n_embd = depth * 64
    n_head = n_embd // head_dim
    cfg = GPTConfig(
        sequence_len=T, vocab_size=32768,
        n_layer=depth, n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
        attn_kind="bqa_dyn", bqa_dyn_use_triton=True,
    )
    torch.manual_seed(0)
    model = GPT(cfg).to(device=device, dtype=DTYPE)
    model.init_weights()
    model_c = torch.compile(model, dynamic=False)

    x = torch.randint(0, cfg.vocab_size, (dbs, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (dbs, T), device=device)

    # warmup (compile)
    for _ in range(3):
        loss = model_c(x, y)
        loss.backward()
    torch.cuda.synchronize()

    # timing
    iters = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for p in model.parameters():
            p.grad = None
        loss = model_c(x, y)
        loss.backward()
    torch.cuda.synchronize()
    ms_per_step = (time.perf_counter() - t0) * 1000.0 / iters

    # mfu (rough): 6 N D for fwd+bwd
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens = dbs * T
    flops = 6 * N * tokens
    tflops_s = flops / (ms_per_step * 1e-3) / 1e12
    h100_peak = 989  # bf16
    mfu = tflops_s / h100_peak * 100

    del model, model_c
    torch.cuda.empty_cache()
    return ms_per_step, tokens / (ms_per_step * 1e-3), mfu, N

print(f"Device: {torch.cuda.get_device_name(0)}, B (per-GPU dbs varies), T={T}, dtype={DTYPE}\n")
print(f"{'config':<22} {'dbs':>5} {'params':>8} {'ms/step':>10} {'tok/sec':>10} {'MFU%':>8}")
print("-" * 75)
for cfg in CONFIGS:
    label, d, jkv, dbs = cfg
    try:
        ms, toks, mfu, N = measure(label, d, jkv, dbs)
        print(f"{label:<22} {dbs:>5} {N/1e6:>7.1f}M {ms:>10.2f} {toks:>10.0f} {mfu:>7.2f}%")
    except Exception as e:
        print(f"{label:<22} FAILED: {str(e)[:200]}")
