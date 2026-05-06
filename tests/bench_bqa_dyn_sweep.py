"""Sweep bench for bqa_dyn forward + fwd+bwd across the configurations the user
asked for: d12, d16, d20 with full and half number of kv_heads.

Configuration convention (matches scripts/base_train.py with aspect_ratio=64,
head_dim=128):
    d12 → model_dim=768,  H=6,  J_full=6,  J_half=3
    d16 → model_dim=1024, H=8,  J_full=8,  J_half=4
    d20 → model_dim=1280, H=10, J_full=10, J_half=5

Reports old (SDPA Q-fold) vs Triton wall time and peak GB.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")

from nanochat.gpt import (
    GPTConfig,
    DynamicBasisQueryAttention,
    apply_rotary_emb,
    norm,
)
from nanochat.bqa_dyn_triton import bqa_dyn_attn_triton_fwd, bqa_dyn_attn

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")


def make_cos_sin(T, head_dim, dtype):
    d = head_dim // 2
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.einsum("t,f->tf", pos, inv_freq)
    cos = freqs.cos().to(dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype)[None, :, None, :]
    return cos, sin


def make_module(layer_idx, n_layer, dtype, n_head, n_kv_head, n_embd):
    cfg = GPTConfig(
        sequence_len=2048, vocab_size=32768, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
    )
    torch.manual_seed(0)
    m = DynamicBasisQueryAttention(cfg, layer_idx=layer_idx).to(device=device, dtype=dtype)
    return m, cfg


def bench_one(fn, n_warmup=8, n_iters=20):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iters


def run_case(B, T, depth, n_head, n_kv_head, n_embd, dtype, win, ve_on, mode):
    """Returns (old_ms, old_gb, tri_ms, tri_gb)."""
    H, J, D = n_head, n_kv_head, n_embd // n_head
    layer_idx = (depth - 1) if ve_on else 0
    m, cfg = make_module(layer_idx, depth, dtype, n_head, n_kv_head, n_embd)
    cos_sin = make_cos_sin(T, D, dtype)
    x_base = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype)
    ve = torch.randn(B, T, J * D, device=device, dtype=dtype) if ve_on else None

    if mode == "fwd":
        x_in = x_base.detach()
        def run_old():
            with torch.no_grad():
                m(x_in, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)

        def run_tri():
            with torch.no_grad():
                B_, T_, _ = x_in.size()
                alpha_k = m.alpha_proj_k(x_in).view(B_, T_, H, J) + m.b_alpha_k
                alpha_v = m.alpha_proj_v(x_in).view(B_, T_, H, J) + m.b_alpha_v
                wk = F.softmax(alpha_k.float(), dim=-1).to(dtype)
                wv = F.softmax(alpha_v.float(), dim=-1).to(dtype)
                q = m.c_q(x_in).view(B_, T_, H, D)
                kb = m.c_k(x_in).view(B_, T_, J, D)
                vb = m.c_v(x_in).view(B_, T_, J, D)
                cs, sn = cos_sin
                q = apply_rotary_emb(q, cs, sn)
                kb = apply_rotary_emb(kb, cs, sn)
                q = norm(q) * 1.2
                kb = norm(kb) * 1.2
                if ve is not None:
                    ve_t = ve.view(B_, T_, J, D).contiguous()
                    gate = (3 * torch.sigmoid(m.ve_gate(x_in[..., : m.ve_gate_channels]))).to(dtype)
                else:
                    ve_t, gate = None, None
                y = bqa_dyn_attn_triton_fwd(
                    q.contiguous(), kb.contiguous(), vb.contiguous(),
                    wk.contiguous(), wv.contiguous(),
                    ve=ve_t, gate=gate, causal=True, window_size=(win, 0),
                )
                m.c_proj(y.contiguous().view(B_, T_, -1))
    elif mode == "fb":
        def run_old():
            m.zero_grad(set_to_none=True)
            x_loc = x_base.detach().clone().requires_grad_(True)
            y = m(x_loc, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)
            y.float().pow(2).mean().backward()

        def run_tri():
            m.zero_grad(set_to_none=True)
            x_loc = x_base.detach().clone().requires_grad_(True)
            B_, T_, _ = x_loc.size()
            alpha_k = m.alpha_proj_k(x_loc).view(B_, T_, H, J) + m.b_alpha_k
            alpha_v = m.alpha_proj_v(x_loc).view(B_, T_, H, J) + m.b_alpha_v
            wk = F.softmax(alpha_k.float(), dim=-1).to(dtype)
            wv = F.softmax(alpha_v.float(), dim=-1).to(dtype)
            q = m.c_q(x_loc).view(B_, T_, H, D)
            kb = m.c_k(x_loc).view(B_, T_, J, D)
            vb = m.c_v(x_loc).view(B_, T_, J, D)
            cs, sn = cos_sin
            q = apply_rotary_emb(q, cs, sn)
            kb = apply_rotary_emb(kb, cs, sn)
            q = norm(q) * 1.2
            kb = norm(kb) * 1.2
            if ve is not None:
                ve_t = ve.view(B_, T_, J, D).contiguous()
                gate = (3 * torch.sigmoid(m.ve_gate(x_loc[..., : m.ve_gate_channels]))).to(dtype)
            else:
                ve_t, gate = None, None
            y = bqa_dyn_attn(q.contiguous(), kb.contiguous(), vb.contiguous(),
                             wk.contiguous(), wv.contiguous(),
                             ve=ve_t, gate=gate, causal=True, window_size=(win, 0))
            y = m.c_proj(y.contiguous().view(B_, T_, -1))
            y.float().pow(2).mean().backward()
    else:
        raise ValueError(mode)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ms_old = bench_one(run_old)
    peak_old = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ms_tri = bench_one(run_tri)
    peak_tri = torch.cuda.max_memory_allocated() / 1e9
    return ms_old, peak_old, ms_tri, peak_tri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--mode", type=str, default="both", choices=["fwd", "fb", "both"])
    ap.add_argument("--ve", type=str, default="both", choices=["off", "on", "both"])
    ap.add_argument("--depths", type=str, default="12,16,20")
    ap.add_argument("--kv", type=str, default="half,full")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"Device: {torch.cuda.get_device_name(0)}, dtype={dtype}, B={args.B}, T={args.T}, win={args.win}")
    print(f"{'depth':>5} {'H':>3} {'J':>3} {'D':>3} {'mode':>4} {'ve':>4}  "
          f"{'old_ms':>9} {'tri_ms':>9} {'sp':>5}  {'old_GB':>7} {'tri_GB':>7}")

    depths = [int(d) for d in args.depths.split(",")]
    kvs = args.kv.split(",")
    aspect_ratio, head_dim = 64, 128
    modes = ["fwd", "fb"] if args.mode == "both" else [args.mode]
    ve_settings = [False, True] if args.ve == "both" else [args.ve == "on"]

    for depth in depths:
        n_embd = depth * aspect_ratio
        H = n_embd // head_dim
        D = head_dim
        for kv_kind in kvs:
            J = H if kv_kind == "full" else max(1, H // 2)
            for mode in modes:
                for ve_on in ve_settings:
                    try:
                        ms_old, gb_old, ms_tri, gb_tri = run_case(
                            args.B, args.T, depth, H, J, n_embd, dtype, args.win, ve_on, mode
                        )
                        ve_str = "on" if ve_on else "off"
                        print(f"{depth:>5} {H:>3} {J:>3} {D:>3} {mode:>4} {ve_str:>4}  "
                              f"{ms_old:>9.2f} {ms_tri:>9.2f} {ms_old/ms_tri:>5.2f}x  "
                              f"{gb_old:>7.2f} {gb_tri:>7.2f}")
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"{depth:>5} {H:>3} {J:>3} {D:>3} {mode:>4} {('on' if ve_on else 'off'):>4}  OOM: {e}")


if __name__ == "__main__":
    main()
