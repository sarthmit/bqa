"""Speed comparison: MHA (CausalSelfAttention, n_kv_head=n_head) vs DynamicBasisQueryAttention.

Single-attention-layer microbench. Same n_embd / n_head / head_dim / RoPE / window
for both. Reports forward and forward+backward ms/step plus peak GPU mem.
"""
import os
import sys
import time
import torch

sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")

from nanochat.gpt import (
    GPTConfig,
    CausalSelfAttention,
    DynamicBasisQueryAttention,
)

device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_float32_matmul_precision("high")


def make_cos_sin(T, head_dim):
    d = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.einsum("t,f->tf", pos, inv_freq)
    cos = freqs.cos().to(dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype)[None, :, None, :]
    return cos, sin


def build(kind, n_layer, n_head, n_kv_head, n_embd, layer_idx):
    cfg = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        attn_kind=kind,
    )
    torch.manual_seed(0)
    if kind in ("mha", "gqa"):
        m = CausalSelfAttention(cfg, layer_idx)
    elif kind == "bqa_dyn":
        m = DynamicBasisQueryAttention(cfg, layer_idx)
    else:
        raise ValueError(kind)
    return m.to(device=device, dtype=dtype), cfg


def bench_one(name, fn, n_warmup=10, n_iters=50):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / n_iters
    return ms


def run(B, T, n_head, n_kv_head_bqa, n_embd, n_layer, label):
    head_dim = n_embd // n_head
    cos_sin = make_cos_sin(T, head_dim)
    win = (1024, 0)

    print("=" * 72)
    print(f"{label}: B={B} T={T} n_embd={n_embd} n_head={n_head} "
          f"head_dim={head_dim} J(bqa)={n_kv_head_bqa} window={win}")
    print("=" * 72)

    rows = []
    for kind, layer_idx, has_ve in [("mha", 0, False),
                                    ("mha", n_layer - 1, True),
                                    ("gqa", 0, False),
                                    ("gqa", n_layer - 1, True),
                                    ("bqa_dyn", 0, False),
                                    ("bqa_dyn", n_layer - 1, True)]:
        if kind == "mha":
            n_kv = n_head
            ve_dim = n_head * head_dim
        elif kind == "gqa":
            n_kv = n_kv_head_bqa
            ve_dim = n_kv * head_dim
        else:
            n_kv = n_kv_head_bqa
            ve_dim = n_kv * head_dim

        m, _ = build(kind, n_layer, n_head, n_kv, n_embd, layer_idx)

        def make_inputs():
            x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
            ve = torch.randn(B, T, ve_dim, device=device, dtype=dtype) if has_ve else None
            return x, ve

        def step_fwd():
            x, ve = make_inputs()
            with torch.no_grad():
                m(x, ve=ve, cos_sin=cos_sin, window_size=win, kv_cache=None)

        def step_fwdbwd():
            m.zero_grad(set_to_none=True)
            x, ve = make_inputs()
            y = m(x, ve=ve, cos_sin=cos_sin, window_size=win, kv_cache=None)
            y.float().pow(2).mean().backward()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ms_fwd = bench_one("fwd", step_fwd, n_warmup=5, n_iters=30)
        except torch.cuda.OutOfMemoryError:
            ms_fwd = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ms_fb = bench_one("fwdbwd", step_fwdbwd, n_warmup=5, n_iters=30)
            peak_fb = torch.cuda.max_memory_allocated() / 1e9
        except torch.cuda.OutOfMemoryError:
            ms_fb = None
            peak_fb = None
        torch.cuda.empty_cache()

        params = sum(p.numel() for p in m.parameters()) / 1e6
        ve_tag = "ve" if has_ve else "  "
        rows.append((kind, ve_tag, params, ms_fwd, ms_fb, peak_fb))
        del m

    print(f"  {'kind':9s} {'ve':3s} {'params(M)':>9s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'peak GB':>8s}")
    for kind, ve_tag, params, ms_fwd, ms_fb, peak in rows:
        fwd = f"{ms_fwd:8.3f}" if ms_fwd is not None else "    OOM"
        fb = f"{ms_fb:11.3f}" if ms_fb is not None else "        OOM"
        pk = f"{peak:8.2f}" if peak is not None else "     OOM"
        print(f"  {kind:9s} {ve_tag:3s} {params:9.2f} {fwd} {fb} {pk}")
    # Pairwise speedups (vs MHA baseline at matching ve setting).
    by_key = {(k, v): (fw, fb) for k, v, _, fw, fb, _ in rows}
    for ve_tag in ("  ", "ve"):
        m_fw, m_fb = by_key[("mha", ve_tag)]
        tag = "ve!=None" if ve_tag == "ve" else "ve=None"
        for other in ("gqa", "bqa_dyn"):
            o_fw, o_fb = by_key[(other, ve_tag)]
            if m_fb and o_fb:
                print(f"  [{tag}]  fwd:    {other} / mha  =  {o_fw / m_fw:.2f}x   "
                      f"fwdbwd: {other} / mha  =  {o_fb / m_fb:.2f}x")


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
    print(f"cuda mem: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # Match d12 bqa_dyn fast run shape.
    run(B=32, T=2048, n_head=6, n_kv_head_bqa=3, n_embd=768, n_layer=12, label="d12 train shape")
    # Half batch (single fwd-bwd budget).
    run(B=8, T=2048, n_head=6, n_kv_head_bqa=3, n_embd=768, n_layer=12, label="d12 small batch")
    # Bigger model proxy (d20-ish).
    run(B=8, T=2048, n_head=8, n_kv_head_bqa=4, n_embd=1024, n_layer=20, label="d20-ish")


if __name__ == "__main__":
    main()
