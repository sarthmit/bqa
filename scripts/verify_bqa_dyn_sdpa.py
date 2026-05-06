"""Verify SDPA-based bqa_dyn rewrite vs OLD explicit reference.

Tests that DynamicBasisQueryAttention.forward (SDPA path with sqrt(J)
pre-multiplication) is numerically equivalent to the explicit
einsum/softmax reference, in both fp32 and bf16, with ve=None and ve!=None.
"""
import os
import sys
import math
import torch
import torch.nn.functional as F

# Make nanochat package importable as `nanochat.*` AND `gpt`.
REPO_ROOT = "/home/mila/m/mittalsa/scratch/bqa"
PKG = f"{REPO_ROOT}/nanochat"
sys.path.insert(0, REPO_ROOT)  # so `import nanochat.common` works
sys.path.insert(0, PKG)         # so `from gpt import ...` works

from nanochat.gpt import (
    DynamicBasisQueryAttention,
    apply_rotary_emb,
    norm,
)


class GPTConfig:
    """Minimal config matching what DynamicBasisQueryAttention needs."""
    def __init__(self, n_embd=256, n_head=8, n_kv_head=4, n_layer=12,
                 sequence_len=64, attn_kind="bqa_dyn"):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_layer = n_layer
        self.sequence_len = sequence_len
        self.attn_kind = attn_kind


def explicit_reference(self, x, ve, cos_sin, window_size, kv_cache):
    """OLD explicit forward: per-basis-D scale 1/sqrt(D), einsum scoring."""
    assert kv_cache is None
    B, T, C = x.size()
    H, J, D = self.n_head, self.n_kv_head, self.head_dim

    # Same mixing weights as the SDPA path.
    alpha_logits_k = self.alpha_proj_k(x).view(B, T, H, J) + self.b_alpha_k
    alpha_logits_v = self.alpha_proj_v(x).view(B, T, H, J) + self.b_alpha_v
    w_k = F.softmax(alpha_logits_k.float(), dim=-1).to(x.dtype)
    w_v = F.softmax(alpha_logits_v.float(), dim=-1).to(x.dtype)

    q = self.c_q(x).view(B, T, H, D)
    k_basis = self.c_k(x).view(B, T, J, D)
    v_basis = self.c_v(x).view(B, T, J, D)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k_basis = apply_rotary_emb(k_basis, cos, sin)
    q = norm(q) * 1.2
    k_basis = norm(k_basis) * 1.2

    # Per-(t,s,j) scores then weighted-sum over j with w_k.
    # S[b,h,t,s,j] = (Q[b,t,h,:] · K[b,s,j,:]) / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    S = torch.einsum('bthd,bsjd->bhtsj', q, k_basis) * scale
    score = torch.einsum('bhtsj,bthj->bhts', S, w_k)

    # Causal + window mask.
    device = x.device
    row = torch.arange(T, device=device).view(T, 1)
    col = torch.arange(T, device=device).view(1, T)
    causal = (row >= col)
    if window_size is not None and window_size != (-1, -1):
        wleft, wright = window_size
        if wleft >= 0:
            causal = causal & ((row - col) <= wleft)
        if wright >= 0:
            causal = causal & ((col - row) <= wright)
    mask = ~causal  # True = mask out
    score = score.masked_fill(mask.view(1, 1, T, T), float('-inf'))

    attn = F.softmax(score.float(), dim=-1).to(x.dtype)  # (B, H, T, T)

    # V combination (per query head): V_eff[b,s,h,j,d] = V_basis[b,s,j,d] + (ve term).
    if ve is not None:
        ve_v = ve.view(B, T, J, D)
        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B,T,H)
        v_per_head = v_basis.unsqueeze(2) + gate[..., None, None] * ve_v.unsqueeze(2)
        # (B, T, H, J, D)
    else:
        v_per_head = v_basis.unsqueeze(2).expand(B, T, self.n_head, J, D)

    # Aggregate: o[b,t,h,j,d] = sum_s attn[b,h,t,s] * v_per_head[b,s,h,j,d]
    o = torch.einsum('bhts,bshjd->bthjd', attn, v_per_head)

    # w_v mix.
    y = torch.einsum('bthj,bthjd->bthd', w_v, o)
    y = y.contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y


def build_cos_sin(T, head_dim, device, dtype):
    # Mimic typical RoPE buffer shape used by apply_rotary_emb (last dim = head_dim/2).
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, half)
    # Shape (1, T, 1, half) for broadcasting against (B, T, H, half).
    cos = freqs.cos().to(dtype).view(1, T, 1, half)
    sin = freqs.sin().to(dtype).view(1, T, 1, half)
    return cos, sin


def cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def run_case(dtype, use_ve, device='cuda'):
    torch.manual_seed(0)
    cfg = GPTConfig(n_embd=384, n_head=6, n_kv_head=3, n_layer=12, sequence_len=64)
    layer_idx = 1  # has_ve(1, 12) is True for nanochat
    attn = DynamicBasisQueryAttention(cfg, layer_idx).to(device).to(dtype)

    B, T = 2, 32
    x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)

    if use_ve:
        ve = torch.randn(B, T, cfg.n_kv_head * attn.head_dim, device=device, dtype=dtype)
    else:
        ve = None

    cos, sin = build_cos_sin(T, attn.head_dim, device, dtype)
    window_size = (-1, -1)  # unbounded; just causal
    kv_cache = None

    # SDPA path (the one in the file).
    y_new = attn(x, ve, (cos, sin), window_size, kv_cache)

    # Explicit reference using the same module (so weights match exactly).
    y_old = explicit_reference(attn, x, ve, (cos, sin), window_size, kv_cache)

    diff = (y_new - y_old).float()
    max_abs = diff.abs().max().item()
    rel = (diff.abs() / (y_old.float().abs() + 1e-6)).max().item()
    cos_sim = cosine(y_new, y_old)

    # Smoke: backward.
    loss = y_new.sum()
    loss.backward()
    grad_ok = x.grad is not None and torch.isfinite(x.grad).all().item()
    nan_out = (~torch.isfinite(y_new)).any().item()

    print(f"[dtype={dtype}, ve={'on' if use_ve else 'off'}] "
          f"cos={cos_sim:.6f}  max_abs={max_abs:.3e}  max_rel={rel:.3e}  "
          f"grad_ok={grad_ok}  nan_out={nan_out}")

    # If equivalence fails, dump samples.
    threshold = 5e-3 if dtype == torch.bfloat16 else 5e-4
    if max_abs > threshold or cos_sim < 0.999:
        print(f"  FAIL — dumping samples (b,t,h,d) old vs new:")
        for (b, t, h, d) in [(0, 0, 0, 0), (1, 5, 3, 7), (0, 17, 1, 12), (1, 30, 7, 0)]:
            try:
                ov = y_old[b, t, b * 0 + h * 0 + 0].float().mean().item()  # not used
            except Exception:
                pass
            print(f"    (b={b},t={t},c={h*attn.head_dim+d}) "
                  f"old={y_old[b,t,h*attn.head_dim+d].item():+.5e}  "
                  f"new={y_new[b,t,h*attn.head_dim+d].item():+.5e}")

    return cos_sim, max_abs, rel, grad_ok, nan_out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    for dtype in (torch.float32, torch.bfloat16):
        for use_ve in (False, True):
            run_case(dtype, use_ve)
