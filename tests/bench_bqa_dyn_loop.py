"""
Self-contained benchmark + correctness check for the bqa_dyn triton kernel
at J ∈ {3, 6, 8, 10, 14}. Used by the /loop kernel-optimization task.

Outputs a CSV-ish summary that subsequent loop iterations can parse:

  J=3 fwd_ms=... fwd_bwd_ms=... fwd_cos=... bwd_cos=... ok=True/False
  ...
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

# Configurations to benchmark — covers the full sweep we care about.
# (J, n_head, head_dim, T) — keeps head_dim=128 (project default), varies J via n_kv_head.
# We pick n_head ≥ J consistent with: full-rank d=12,16,20,28 → J=6,8,10,14, plus J=3 (d=12 half).
CONFIGS = [
    # (label, J, n_head, T, has_ve)
    ("J=3 (d=12 half)",  3,  6, 1024, False),
    ("J=6 (d=12 full)",  6,  6, 1024, False),
    ("J=8 (d=16 full)",  8,  8, 1024, False),
    ("J=10 (d=20 full)", 10, 10, 1024, False),
    ("J=14 (d=28 full)", 14, 14, 1024, False),
]

D = 128  # head_dim
B = 2    # batch — small to keep runtime reasonable, big enough for meaningful timing
DTYPE = torch.bfloat16


def explicit_forward(q, k, v, w_k, w_v, ve, gate, T, causal=True, window=-1):
    """Reference: 1/sqrt(D), single softmax over s and J. q is (B,T,H,D), k/v (B,T,J,D)."""
    B_, T_, H_, D_ = q.shape
    J_ = k.shape[2]
    sm = 1.0 / math.sqrt(D_)
    # S[b,h,t,s,j] = q[b,t,h,d] @ k[b,s,j,d]
    S = torch.einsum("bthd,bsjd->bhtsj", q, k) * sm  # (B,H,T,T,J)
    # mix logits: weighted by w_k[b,t,h,j]
    score = torch.einsum("bhtsj,bthj->bhts", S, w_k)
    row = torch.arange(T, device=score.device).view(-1, 1)
    col = torch.arange(T, device=score.device).view(1, -1)
    mask = col <= row if causal else torch.ones_like(col, dtype=torch.bool)
    if 0 <= window < T:
        mask = mask & ((row - col) <= window)
    score = score.masked_fill(~mask, float("-inf"))
    p = F.softmax(score, dim=-1)
    # V mix: O[b,t,h,d] = sum_s p[b,h,t,s] · sum_j w_v[b,t,h,j] · v_eff[b,s,j,d]
    if ve is not None:
        v_eff = v + gate.unsqueeze(-1) * ve  # (B,T,H,J,D)? gate is (B,T,H), ve is (B,T,J,D)
    else:
        v_eff = v
    # First compute per-j attention output, then mix by w_v.
    # O_v[b,h,t,j,d] = sum_s p[b,h,t,s] · v_eff[b,s,j,d]
    if ve is not None:
        # gate is per-(b,t,h); here we mix at attended s, so apply at v side:
        # v_eff_per_h[b,s,h,j,d] = v[b,s,j,d] + gate[b,s,h] · ve[b,s,j,d]
        v_eff_per_h = v.unsqueeze(2) + gate.unsqueeze(-1).unsqueeze(-1) * ve.unsqueeze(2)  # (B,T,H,J,D)
        # rearrange for einsum
        # O[b,h,t,d] = sum_s p[b,h,t,s] · sum_j w_v[b,t,h,j] · v_eff_per_h[b,s,h,j,d]
        out = torch.einsum("bhts,bthj,bshjd->bthd", p, w_v, v_eff_per_h)
    else:
        out = torch.einsum("bhts,bthj,bsjd->bthd", p, w_v, v)
    return out


def bench_one(label, J, H, T, has_ve):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=device, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, T, J, D, device=device, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, T, J, D, device=device, dtype=DTYPE, requires_grad=True)
    # Mixing weights: softmax to make them realistic distributions.
    w_k_logits = torch.randn(B, T, H, J, device=device, dtype=torch.float32)
    w_v_logits = torch.randn(B, T, H, J, device=device, dtype=torch.float32)
    w_k = F.softmax(w_k_logits, dim=-1).to(DTYPE).requires_grad_(True)
    w_v = F.softmax(w_v_logits, dim=-1).to(DTYPE).requires_grad_(True)

    ve = gate = None
    # Skip ve path for speed in this loop.

    from nanochat.bqa_dyn_triton import bqa_dyn_attn

    # --- correctness (fwd + bwd) ---
    # Forward
    with torch.no_grad():
        y_ref_nograd = explicit_forward(q.detach(), k.detach(), v.detach(),
                                  w_k.detach(), w_v.detach(),
                                  None, None, T, causal=True, window=-1)
        y_tri_nograd = bqa_dyn_attn(
            q.detach().contiguous(), k.detach().contiguous(), v.detach().contiguous(),
            w_k.detach().contiguous(), w_v.detach().contiguous(),
            ve=None, gate=None, causal=True, window_size=None,
        )
    cos_fwd = F.cosine_similarity(y_tri_nograd.float().flatten(), y_ref_nograd.float().flatten(), dim=0).item()
    abs_max = (y_tri_nograd.float() - y_ref_nograd.float()).abs().max().item()
    fwd_ok = cos_fwd > 0.999

    # Backward — compare gradients from triton vs gradients from explicit_forward.
    # Use a fresh set of leaf tensors to avoid stale grads.
    q_a = q.detach().clone().requires_grad_(True)
    k_a = k.detach().clone().requires_grad_(True)
    v_a = v.detach().clone().requires_grad_(True)
    wk_a = w_k.detach().clone().requires_grad_(True)
    wv_a = w_v.detach().clone().requires_grad_(True)
    q_b = q.detach().clone().requires_grad_(True)
    k_b = k.detach().clone().requires_grad_(True)
    v_b = v.detach().clone().requires_grad_(True)
    wk_b = w_k.detach().clone().requires_grad_(True)
    wv_b = w_v.detach().clone().requires_grad_(True)
    grad_o = torch.randn_like(q_a)

    y_ref = explicit_forward(q_a, k_a, v_a, wk_a, wv_a, None, None, T, causal=True, window=-1)
    y_ref.backward(grad_o)
    y_tri_g = bqa_dyn_attn(q_b.contiguous(), k_b.contiguous(), v_b.contiguous(),
                            wk_b.contiguous(), wv_b.contiguous(),
                            ve=None, gate=None, causal=True, window_size=None)
    y_tri_g.backward(grad_o)

    def _cos(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    cos_dq = _cos(q_a.grad, q_b.grad)
    cos_dk = _cos(k_a.grad, k_b.grad)
    cos_dv = _cos(v_a.grad, v_b.grad)
    cos_dwk = _cos(wk_a.grad, wk_b.grad)
    cos_dwv = _cos(wv_a.grad, wv_b.grad)
    bwd_min_cos = min(cos_dq, cos_dk, cos_dv, cos_dwk, cos_dwv)
    bwd_ok = bwd_min_cos > 0.99
    fwd_ok = fwd_ok and bwd_ok

    # --- forward perf ---
    torch.cuda.synchronize()
    # warmup
    for _ in range(3):
        _ = bqa_dyn_attn(q, k, v, w_k, w_v, ve=None, gate=None, causal=True, window_size=None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        _ = bqa_dyn_attn(q, k, v, w_k, w_v, ve=None, gate=None, causal=True, window_size=None)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # --- forward+backward perf ---
    grad_out = torch.randn(B, T, H, D, device=device, dtype=DTYPE)
    torch.cuda.synchronize()
    # warmup
    for _ in range(3):
        for t in (q, k, v, w_k, w_v):
            if t.grad is not None:
                t.grad = None
        y = bqa_dyn_attn(q, k, v, w_k, w_v, ve=None, gate=None, causal=True, window_size=None)
        y.backward(grad_out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters_bwd = 10
    for _ in range(iters_bwd):
        for t in (q, k, v, w_k, w_v):
            if t.grad is not None:
                t.grad = None
        y = bqa_dyn_attn(q, k, v, w_k, w_v, ve=None, gate=None, causal=True, window_size=None)
        y.backward(grad_out)
    torch.cuda.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) * 1000.0 / iters_bwd

    return {
        "label": label,
        "J": J,
        "H": H,
        "T": T,
        "fwd_ms": fwd_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "fwd_cos": cos_fwd,
        "fwd_abs_max": abs_max,
        "fwd_ok": fwd_ok,
        "cos_dq": cos_dq,
        "cos_dk": cos_dk,
        "cos_dv": cos_dv,
        "cos_dwk": cos_dwk,
        "cos_dwv": cos_dwv,
        "bwd_min_cos": bwd_min_cos,
    }


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"B={B}, D={D}, dtype={DTYPE}\n")
    print(f"{'config':<20} {'fwd ms':>10} {'fwd+bwd ms':>12} {'fwd cos':>10} {'bwd min cos':>12} {'ok':>5}")
    print("-" * 90)

    results = []
    for cfg in CONFIGS:
        try:
            r = bench_one(*cfg)
            results.append(r)
            print(f"{r['label']:<20} {r['fwd_ms']:>10.3f} {r['fwd_bwd_ms']:>12.3f} "
                  f"{r['fwd_cos']:>10.6f} {r['bwd_min_cos']:>12.6f} {str(r['fwd_ok']):>5}")
        except Exception as e:
            err = str(e)[:200]
            print(f"{cfg[0]:<20} FAILED: {err}")
            results.append({"label": cfg[0], "J": cfg[1], "error": err})

    out_path = os.environ.get("BENCH_OUT", "/scratch/sarthmit/bqa/tests/bench_bqa_dyn_loop_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
