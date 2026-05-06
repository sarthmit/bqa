"""
Verify bqa_dyn refactor: equivalence with explicit forward + bench vs explicit.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure we import the nanochat in the bqa repo, not anywhere else.
sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")

from nanochat.gpt import (
    GPTConfig,
    DynamicBasisQueryAttention,
    apply_rotary_emb,
    norm,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
dtype = torch.bfloat16


def make_cos_sin(T, head_dim, device):
    # Standard rotary: half-dim angles, repeated to fill head_dim.
    # Match nanochat's apply_rotary_emb: it splits last dim into halves of size d=head_dim/2.
    d = head_dim // 2
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.einsum("t,f->tf", pos, inv_freq)  # (T, d)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    # apply_rotary_emb expects cos, sin shapes broadcastable to x[..., :d] which is (B, T, H, d).
    # Provide (T, d) — broadcasts as (1, T, 1, d).
    return cos[None, :, None, :], sin[None, :, None, :]


def explicit_forward(mod, x, ve, cos_sin, window_size):
    """OLD explicit forward — algebraic reference. Uses mod's parameters/sublayers."""
    B, T, C = x.size()
    H, J, D = mod.n_head, mod.n_kv_head, mod.head_dim

    alpha_logits_k = mod.alpha_proj_k(x).view(B, T, H, J) + mod.b_alpha_k
    alpha_logits_v = mod.alpha_proj_v(x).view(B, T, H, J) + mod.b_alpha_v
    w_k = F.softmax(alpha_logits_k.float(), dim=-1).to(x.dtype)  # (B,T,H,J)
    w_v = F.softmax(alpha_logits_v.float(), dim=-1).to(x.dtype)  # (B,T,H,J)

    q = mod.c_q(x).view(B, T, H, D)
    k_basis = mod.c_k(x).view(B, T, J, D)
    v_basis = mod.c_v(x).view(B, T, J, D)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k_basis = apply_rotary_emb(k_basis, cos, sin)
    q = norm(q) * 1.2
    k_basis = norm(k_basis) * 1.2

    # explicit basis-score path
    S = torch.einsum("bthd,bsjd->bhtsj", q, k_basis) / (D ** 0.5)
    score = torch.einsum("bhtsj,bthj->bhts", S, w_k)

    row = torch.arange(T, device=score.device).view(-1, 1)
    col = torch.arange(T, device=score.device).view(1, -1)
    mask = col <= row
    win = window_size[0]
    if win >= 0 and win < T:
        mask = mask & ((row - col) <= win)
    score = score.masked_fill(~mask, float("-inf"))
    p = F.softmax(score, dim=-1)
    O_v = torch.einsum("bhts,bsjd->bthjd", p, v_basis)
    if ve is not None:
        ve_local = ve.view(B, T, J, D)
        gate = 3 * torch.sigmoid(mod.ve_gate(x[..., : mod.ve_gate_channels]))  # (B,T,H)
        p_gated = p * gate.permute(0, 2, 1).unsqueeze(2)  # (B,H,T,T) * (B,H,T,1)
        O_ve = torch.einsum("bhts,bsjd->bthjd", p_gated, ve_local)
        O_v = O_v + O_ve
    y = torch.einsum("bthj,bthjd->bthd", w_v, O_v)
    y = y.contiguous().view(B, T, -1)
    y = mod.c_proj(y)
    return y


def explicit_forward_widened_scale(mod, x, ve, cos_sin, window_size):
    """Same as new path but with the WRONG scale 1/sqrt(J*D) — i.e., what SDPA actually computes.
    Used to confirm the only delta is the attention temperature."""
    B, T, C = x.size()
    H, J, D = mod.n_head, mod.n_kv_head, mod.head_dim

    alpha_logits_k = mod.alpha_proj_k(x).view(B, T, H, J) + mod.b_alpha_k
    alpha_logits_v = mod.alpha_proj_v(x).view(B, T, H, J) + mod.b_alpha_v
    w_k = F.softmax(alpha_logits_k.float(), dim=-1).to(x.dtype)
    w_v = F.softmax(alpha_logits_v.float(), dim=-1).to(x.dtype)

    q = mod.c_q(x).view(B, T, H, D)
    k_basis = mod.c_k(x).view(B, T, J, D)
    v_basis = mod.c_v(x).view(B, T, J, D)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k_basis = apply_rotary_emb(k_basis, cos, sin)
    q = norm(q) * 1.2
    k_basis = norm(k_basis) * 1.2

    # Use 1/sqrt(J*D) — what the SDPA call inside the new path uses.
    S = torch.einsum("bthd,bsjd->bhtsj", q, k_basis) / ((J * D) ** 0.5)
    score = torch.einsum("bhtsj,bthj->bhts", S, w_k)

    row = torch.arange(T, device=score.device).view(-1, 1)
    col = torch.arange(T, device=score.device).view(1, -1)
    mask = col <= row
    win = window_size[0]
    if win >= 0 and win < T:
        mask = mask & ((row - col) <= win)
    score = score.masked_fill(~mask, float("-inf"))
    p = F.softmax(score, dim=-1)
    O_v = torch.einsum("bhts,bsjd->bthjd", p, v_basis)
    if ve is not None:
        ve_local = ve.view(B, T, J, D)
        gate = 3 * torch.sigmoid(mod.ve_gate(x[..., : mod.ve_gate_channels]))
        p_gated = p * gate.permute(0, 2, 1).unsqueeze(2)
        O_ve = torch.einsum("bhts,bsjd->bthjd", p_gated, ve_local)
        O_v = O_v + O_ve
    y = torch.einsum("bthj,bthjd->bthd", w_v, O_v)
    y = y.contiguous().view(B, T, -1)
    y = mod.c_proj(y)
    return y


def diff_stats(a, b):
    a_f = a.float()
    b_f = b.float()
    abs_diff = (a_f - b_f).abs()
    rel_diff = abs_diff / (b_f.abs().clamp_min(1e-6))
    return abs_diff.max().item(), rel_diff.max().item(), a_f.abs().mean().item()


def make_module(layer_idx, n_layer, device, dtype):
    cfg = GPTConfig(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=n_layer,
        n_head=6,
        n_kv_head=3,
        n_embd=768,
    )
    torch.manual_seed(0)
    m = DynamicBasisQueryAttention(cfg, layer_idx=layer_idx).to(device=device, dtype=dtype)
    # Linear stores fp32 master weights and casts to x.dtype in forward — keep as is.
    # Convert weights to dtype to match what training would do.
    return m, cfg


def equivalence_test():
    print("=" * 70)
    print("EQUIVALENCE TEST")
    print("=" * 70)
    B, T = 2, 256
    n_layer = 4
    H, J, D = 6, 3, 128
    win = 1024  # sliding window

    for test_dtype in (torch.float32, torch.bfloat16):
        print(f"\n--- dtype={test_dtype} ---")
        for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
            m, cfg = make_module(layer_idx=layer_idx, n_layer=n_layer, device=device, dtype=test_dtype)
            torch.manual_seed(123)
            x = torch.randn(B, T, cfg.n_embd, device=device, dtype=test_dtype, requires_grad=False)
            d = D // 2
            base = 10000.0
            inv_freq = 1.0 / (base ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))
            pos = torch.arange(T, dtype=torch.float32, device=device)
            freqs = torch.einsum("t,f->tf", pos, inv_freq)
            cos = freqs.cos().to(test_dtype)[None, :, None, :]
            sin = freqs.sin().to(test_dtype)[None, :, None, :]
            cos_sin = (cos, sin)

            if m.ve_gate is not None:
                ve = torch.randn(B, T, J * D, device=device, dtype=test_dtype)
            else:
                ve = None

            with torch.no_grad():
                y_new = m(x, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)
                y_old = explicit_forward(m, x, ve=ve, cos_sin=cos_sin, window_size=(win, 0))
                y_widened = explicit_forward_widened_scale(m, x, ve=ve, cos_sin=cos_sin, window_size=(win, 0))

            max_abs, max_rel, mean_abs = diff_stats(y_new, y_old)
            yn = y_new.float().reshape(-1)
            yo = y_old.float().reshape(-1)
            yw = y_widened.float().reshape(-1)
            corr_no = torch.nn.functional.cosine_similarity(yn, yo, dim=0).item()
            corr_nw = torch.nn.functional.cosine_similarity(yn, yw, dim=0).item()
            ma_nw, mr_nw, _ = diff_stats(y_new, y_widened)
            print(f"[{case_name}] layer_idx={layer_idx}, has_ve_gate={m.ve_gate is not None}, dtype={test_dtype}")
            print(f"  shape: {tuple(y_new.shape)}, mean|y|={mean_abs:.4f}")
            print(f"  new vs OLD (1/sqrt(D))           : abs={max_abs:.4e}, rel={max_rel:.4e}, cos={corr_no:.6f}")
            print(f"  new vs widened-scale (1/sqrt(JD)): abs={ma_nw:.4e}, rel={mr_nw:.4e}, cos={corr_nw:.6f}")
            atol, rtol = 1e-2, 1e-2
            allclose_old = torch.allclose(y_new.float(), y_old.float(), atol=atol, rtol=rtol)
            allclose_widened = torch.allclose(y_new.float(), y_widened.float(), atol=atol, rtol=rtol)
            print(f"  allclose vs OLD: {allclose_old}; allclose vs widened-scale ref: {allclose_widened}")


def fwd_bwd_smoke():
    print("=" * 70)
    print("FORWARD+BACKWARD SMOKE")
    print("=" * 70)
    B, T = 2, 512
    n_layer = 4
    H, J, D = 6, 3, 128
    layer_idx = n_layer - 1  # ve path
    m, cfg = make_module(layer_idx=layer_idx, n_layer=n_layer, device=device, dtype=dtype)
    x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)
    cos_sin = make_cos_sin(T, D, device)
    ve = torch.randn(B, T, J * D, device=device, dtype=dtype)
    y = m(x, ve=ve, cos_sin=cos_sin, window_size=(1024, 0), kv_cache=None)
    print(f"  out shape: {tuple(y.shape)}, has NaN: {torch.isnan(y).any().item()}, has Inf: {torch.isinf(y).any().item()}")
    loss = y.float().pow(2).mean()
    loss.backward()
    missing = []
    nan_grads = []
    for name, p in m.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            nan_grads.append(name)
    print(f"  missing grads: {missing}")
    print(f"  nan/inf grads: {nan_grads}")
    print(f"  loss: {loss.item():.4f}")


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
    print(f"  {name}: {ms:.2f} ms/step")
    return ms


def bench(B, T, label):
    print("=" * 70)
    print(f"BENCH: {label} (B={B}, T={T})")
    print("=" * 70)
    n_layer = 4
    H, J, D = 6, 3, 128
    layer_idx = n_layer - 1
    m, cfg = make_module(layer_idx=layer_idx, n_layer=n_layer, device=device, dtype=dtype)
    cos_sin = make_cos_sin(T, D, device)
    win = (1024, 0)

    def make_inputs():
        x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)
        ve = torch.randn(B, T, J * D, device=device, dtype=dtype)
        return x, ve

    def step_new():
        m.zero_grad(set_to_none=True)
        x, ve = make_inputs()
        y = m(x, ve=ve, cos_sin=cos_sin, window_size=win, kv_cache=None)
        y.float().pow(2).mean().backward()

    def step_old():
        m.zero_grad(set_to_none=True)
        x, ve = make_inputs()
        y = explicit_forward(m, x, ve=ve, cos_sin=cos_sin, window_size=win)
        y.float().pow(2).mean().backward()

    print(f"  GPU mem before: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    try:
        ms_new = bench_one("NEW", step_new)
        peak_new = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  NEW: OOM ({e})")
        ms_new = None
        peak_new = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        ms_old = bench_one("OLD", step_old)
        peak_old = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OLD: OOM ({e})")
        ms_old = None
        peak_old = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if ms_new and ms_old:
        print(f"  speedup new vs old: {ms_old / ms_new:.2f}x")
    if peak_new is not None:
        print(f"  peak mem NEW: {peak_new:.2f} GB")
    if peak_old is not None:
        print(f"  peak mem OLD: {peak_old:.2f} GB")
    return ms_new, ms_old


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    print(f"cuda mem total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"cuda mem free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB / total {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")

    equivalence_test()
    if os.environ.get("VERIFY_SKIP_BENCH"):
        return
    fwd_bwd_smoke()
    try:
        bench(B=8, T=2048, label="d12-equiv B=8 T=2048")
    except torch.cuda.OutOfMemoryError:
        print("  -> fallback to smaller shape")
        torch.cuda.empty_cache()
    try:
        bench(B=32, T=2048, label="big DBS B=32 T=2048")
    except torch.cuda.OutOfMemoryError:
        print("  -> fallback to smaller shape")
        torch.cuda.empty_cache()
    bench(B=4, T=1024, label="small fallback")

    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
