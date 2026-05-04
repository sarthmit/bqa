"""
Equivalence + perf test for the Triton bqa_dyn forward kernel.

Compares against:
  - explicit_forward (algebraic reference, 1/sqrt(D) scale, single softmax over s),
  - DynamicBasisQueryAttention.forward (the current Q-side-fold + SDPA path).

Run on a GPU node, e.g.:
    ssh cn-g022
    export SLURM_TMPDIR=/tmp
    export PATH=$HOME/.local/bin:$PATH
    cd /network/scratch/m/mittalsa/bqa
    source scripts/setup_node.sh
    uv run python tests/test_bqa_dyn_triton.py
"""
import os
import sys
import time
import math
import torch
import torch.nn.functional as F

# Make sure we import the nanochat in the bqa repo, not anywhere else.
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


def explicit_forward(mod, x, ve, cos_sin, window_size):
    """Algebraic reference (1/sqrt(D), single softmax). Same as verify_bqa_dyn.py."""
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

    S = torch.einsum("bthd,bsjd->bhtsj", q, k_basis) / (D ** 0.5)
    score = torch.einsum("bhtsj,bthj->bhts", S, w_k)

    row = torch.arange(T, device=score.device).view(-1, 1)
    col = torch.arange(T, device=score.device).view(1, -1)
    mask = col <= row
    win = window_size[0]
    if 0 <= win < T:
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


def triton_module_forward(mod, x, ve, cos_sin, window_size):
    """Run the Triton kernel from the same module's parameters; apply c_proj at the end.

    Returns (y_after_proj, y_pre_proj) so callers can compare both.
    """
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

    if ve is not None:
        ve_t = ve.view(B, T, J, D).contiguous()
        gate = 3 * torch.sigmoid(mod.ve_gate(x[..., : mod.ve_gate_channels])).to(x.dtype)
    else:
        ve_t = None
        gate = None

    y_pre = bqa_dyn_attn_triton_fwd(
        q.contiguous(), k_basis.contiguous(), v_basis.contiguous(),
        w_k.contiguous(), w_v.contiguous(),
        ve=ve_t, gate=gate,
        causal=True, window_size=window_size,
    )
    y = y_pre.contiguous().view(B, T, -1)
    y = mod.c_proj(y)
    return y, y_pre


def diff_stats(a, b):
    af = a.float(); bf = b.float()
    abs_diff = (af - bf).abs()
    rel_diff = abs_diff / bf.abs().clamp_min(1e-6)
    return abs_diff.max().item(), rel_diff.max().item(), af.abs().mean().item()


def make_module(layer_idx, n_layer, dtype, n_head=6, n_kv_head=3, n_embd=768):
    cfg = GPTConfig(
        sequence_len=2048, vocab_size=32768, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
    )
    torch.manual_seed(0)
    m = DynamicBasisQueryAttention(cfg, layer_idx=layer_idx).to(device=device, dtype=dtype)
    return m, cfg


def equivalence(test_dtype, win, n_layer=4, n_head=6, n_kv_head=3, n_embd=768, B=2, T=256):
    H, J, D = n_head, n_kv_head, n_embd // n_head
    print(f"\n--- equivalence: dtype={test_dtype}, B={B}, T={T}, H={H}, J={J}, D={D}, win={win} ---")
    for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
        m, cfg = make_module(layer_idx, n_layer, test_dtype, n_head, n_kv_head, n_embd)
        torch.manual_seed(123)
        x = torch.randn(B, T, cfg.n_embd, device=device, dtype=test_dtype)
        cos_sin = make_cos_sin(T, D, test_dtype)
        ve = torch.randn(B, T, J * D, device=device, dtype=test_dtype) if m.ve_gate is not None else None
        with torch.no_grad():
            y_ref = explicit_forward(m, x, ve, cos_sin, (win, 0))
            y_old = m(x, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)
            y_tri, _ = triton_module_forward(m, x, ve, cos_sin, (win, 0))
        ma_ref, mr_ref, mean_abs = diff_stats(y_tri, y_ref)
        ma_old, mr_old, _ = diff_stats(y_tri, y_old)
        cos_ref = F.cosine_similarity(y_tri.float().flatten(), y_ref.float().flatten(), dim=0).item()
        cos_old = F.cosine_similarity(y_tri.float().flatten(), y_old.float().flatten(), dim=0).item()
        # fp32: 1e-3 abs is the right scale here — the reference uses a flat softmax
        # and a single big einsum over (s, j), while the kernel does online softmax with
        # J separate dot products and rescales — equivalent algebra, different fp order,
        # so differences scale with seq length (256 here) at ~ε·sqrt(T·J·D).
        atol, rtol = (1e-2, 1e-2) if test_dtype != torch.float32 else (2e-3, 2e-3)
        ok_ref = torch.allclose(y_tri.float(), y_ref.float(), atol=atol, rtol=rtol)
        ok_old = torch.allclose(y_tri.float(), y_old.float(), atol=atol, rtol=rtol)
        print(f"  [{case_name}] mean|y|={mean_abs:.4f}")
        print(f"    triton vs explicit (1/sqrt(D))    : abs={ma_ref:.3e}  rel={mr_ref:.3e}  cos={cos_ref:.6f}  allclose={ok_ref}")
        print(f"    triton vs old SDPA (Q-fold path)  : abs={ma_old:.3e}  rel={mr_old:.3e}  cos={cos_old:.6f}  allclose={ok_old}")


def bench_one(fn, n_warmup=10, n_iters=30):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iters


def bench(B, T, n_head=6, n_kv_head=3, n_embd=768, n_layer=12, win=1024, dtype=torch.bfloat16):
    H, J, D = n_head, n_kv_head, n_embd // n_head
    print(f"\n--- bench: B={B}, T={T}, H={H}, J={J}, D={D}, window={win} dtype={dtype} ---")
    for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
        m, cfg = make_module(layer_idx, n_layer, dtype, n_head, n_kv_head, n_embd)
        x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype)
        cos_sin = make_cos_sin(T, D, dtype)
        ve = torch.randn(B, T, J * D, device=device, dtype=dtype) if m.ve_gate is not None else None

        def run_old():
            with torch.no_grad():
                m(x, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)
        def run_tri():
            with torch.no_grad():
                triton_module_forward(m, x, ve, cos_sin, (win, 0))

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ms_old = bench_one(run_old)
        peak_old = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ms_tri = bench_one(run_tri)
        peak_tri = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{case_name}]  old fwd: {ms_old:7.3f} ms (peak {peak_old:.2f} GB)   "
              f"triton fwd: {ms_tri:7.3f} ms (peak {peak_tri:.2f} GB)   speedup: {ms_old/ms_tri:.2f}x")


def reference_fwd_through_module(mod, x, ve, cos_sin, window_size):
    """Run the explicit (algebraic) reference WITH AUTOGRAD on the module's inputs.

    We set up the SAME tensor graph the Triton autograd Function sees: leaves are
    (q, k_basis, v_basis, w_k, w_v, ve_view, gate). All upstream linear layers,
    rotary, RMS norm, ×1.2 are computed first under no_grad and then `requires_grad_`
    is flipped on the relevant leaves so we can compare grads of these EXACT tensors
    between the two paths. Returns (y, leaves_dict) where the dict has names mapping
    to grad-enabled tensors. Output `y` is BEFORE c_proj (matches Triton output).
    """
    B, T, C = x.size()
    H, J, D = mod.n_head, mod.n_kv_head, mod.head_dim

    with torch.no_grad():
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

        if ve is not None:
            ve_t = ve.view(B, T, J, D).contiguous()
            gate = (3 * torch.sigmoid(mod.ve_gate(x[..., : mod.ve_gate_channels]))).to(x.dtype)
        else:
            ve_t = None
            gate = None

    # Build a fresh leaf set for autograd comparison.
    leaves = {}
    leaves["q"] = q.detach().clone().requires_grad_(True)
    leaves["k_basis"] = k_basis.detach().clone().requires_grad_(True)
    leaves["v_basis"] = v_basis.detach().clone().requires_grad_(True)
    leaves["w_k"] = w_k.detach().clone().requires_grad_(True)
    leaves["w_v"] = w_v.detach().clone().requires_grad_(True)
    if ve_t is not None:
        leaves["ve"] = ve_t.detach().clone().requires_grad_(True)
        leaves["gate"] = gate.detach().clone().requires_grad_(True)

    # Now run explicit forward on the LEAVES.
    q_l = leaves["q"]; k_l = leaves["k_basis"]; v_l = leaves["v_basis"]
    wk_l = leaves["w_k"]; wv_l = leaves["w_v"]

    S = torch.einsum("bthd,bsjd->bhtsj", q_l, k_l) / (D ** 0.5)
    score = torch.einsum("bhtsj,bthj->bhts", S, wk_l)

    row = torch.arange(T, device=score.device).view(-1, 1)
    col = torch.arange(T, device=score.device).view(1, -1)
    mask = col <= row
    win = window_size[0]
    if 0 <= win < T:
        mask = mask & ((row - col) <= win)
    score = score.masked_fill(~mask, float("-inf"))
    p = F.softmax(score, dim=-1)
    O_v = torch.einsum("bhts,bsjd->bthjd", p, v_l)
    if ve_t is not None:
        ve_l = leaves["ve"]; g_l = leaves["gate"]
        # p_gated[b, h, t, s] = p[b,h,t,s] · gate[b,s,h]
        p_gated = p * g_l.permute(0, 2, 1).unsqueeze(2)
        O_ve = torch.einsum("bhts,bsjd->bthjd", p_gated, ve_l)
        O_v = O_v + O_ve
    y = torch.einsum("bthj,bthjd->bthd", wv_l, O_v)
    return y, leaves


def grad_check(test_dtype, win, n_layer=4, n_head=6, n_kv_head=3, n_embd=768, B=2, T=128):
    H, J, D = n_head, n_kv_head, n_embd // n_head
    print(f"\n--- gradcheck: dtype={test_dtype}, B={B}, T={T}, H={H}, J={J}, D={D}, win={win} ---")
    for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
        m, cfg = make_module(layer_idx, n_layer, test_dtype, n_head, n_kv_head, n_embd)
        torch.manual_seed(123)
        x = torch.randn(B, T, cfg.n_embd, device=device, dtype=test_dtype)
        cos_sin = make_cos_sin(T, D, test_dtype)
        ve = torch.randn(B, T, J * D, device=device, dtype=test_dtype) if m.ve_gate is not None else None

        # Reference forward+backward through autograd of the explicit path.
        y_ref, leaves = reference_fwd_through_module(m, x, ve, cos_sin, (win, 0))
        torch.manual_seed(0)
        dy = torch.randn_like(y_ref)
        y_ref.backward(dy)

        # Triton forward+backward on a separate set of identical leaves.
        leaves_t = {k: v.detach().clone().requires_grad_(True) for k, v in leaves.items()}
        y_tri = bqa_dyn_attn(
            leaves_t["q"], leaves_t["k_basis"], leaves_t["v_basis"],
            leaves_t["w_k"], leaves_t["w_v"],
            ve=leaves_t.get("ve"), gate=leaves_t.get("gate"),
            causal=True, window_size=(win, 0),
        )
        y_tri.backward(dy)

        # Compare outputs and grads. Use normwise relative error (‖dg‖/‖g‖) +
        # cosine — element-wise allclose is the wrong tool here because
        # (a) Triton's tl.dot uses TF32 by default on Ampere (10-bit mantissa
        #     during the matmul), so even in fp32 mode per-element abs error
        #     scales with the matmul magnitudes, not with fp32 epsilon, and
        # (b) gradient magnitudes vary widely across elements, so a single
        #     atol/rtol threshold either over- or under-tests.
        names = ["q", "k_basis", "v_basis", "w_k", "w_v"]
        if "ve" in leaves:
            names += ["ve", "gate"]
        # Normwise rel error budget: TF32 matmuls give ~1e-3 rel; bf16 gives ~3e-2.
        norm_budget = 1e-2 if test_dtype == torch.float32 else 5e-2
        cos_floor = 0.9999

        ma_y, mr_y, _ = diff_stats(y_tri, y_ref)
        y_normwise = (y_tri.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-8)
        cos_y = F.cosine_similarity(y_tri.float().flatten(), y_ref.float().flatten(), dim=0).item()
        print(f"  [{case_name}] y: abs={ma_y:.3e} normwise_rel={y_normwise.item():.3e} cos={cos_y:.6f}")
        all_ok = (cos_y >= cos_floor) and (y_normwise.item() <= norm_budget)
        for nm in names:
            g_ref = leaves[nm].grad
            g_tri = leaves_t[nm].grad
            ma, mr, _ = diff_stats(g_tri, g_ref)
            cos = F.cosine_similarity(g_tri.float().flatten(), g_ref.float().flatten(), dim=0).item()
            normwise = ((g_tri.float() - g_ref.float()).norm() / g_ref.float().norm().clamp_min(1e-8)).item()
            ok = (cos >= cos_floor) and (normwise <= norm_budget)
            all_ok = all_ok and ok
            print(f"    grad {nm:8s}: abs={ma:.3e} normwise_rel={normwise:.3e} cos={cos:.6f} ok={ok} (mean|g|={g_ref.float().abs().mean().item():.4f})")
        print(f"  [{case_name}] all-pass (normwise<{norm_budget}, cos>{cos_floor}): {all_ok}")


def bench_fb(B, T, n_head=6, n_kv_head=3, n_embd=768, n_layer=12, win=1024, dtype=torch.bfloat16):
    H, J, D = n_head, n_kv_head, n_embd // n_head
    print(f"\n--- bench fwd+bwd: B={B}, T={T}, H={H}, J={J}, D={D}, window={win} dtype={dtype} ---")
    for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
        m, cfg = make_module(layer_idx, n_layer, dtype, n_head, n_kv_head, n_embd)
        cos_sin = make_cos_sin(T, D, dtype)

        def make_inputs(ve_on):
            x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype, requires_grad=True)
            ve = torch.randn(B, T, J * D, device=device, dtype=dtype) if ve_on else None
            return x, ve

        ve_on = m.ve_gate is not None
        x, ve = make_inputs(ve_on)

        def run_old():
            m.zero_grad(set_to_none=True)
            x_local = x.detach().clone().requires_grad_(True)
            y = m(x_local, ve=ve, cos_sin=cos_sin, window_size=(win, 0), kv_cache=None)
            y.float().pow(2).mean().backward()

        # Triton fwd+bwd, applied through the same module's prelude (rotary/norm/etc),
        # using the autograd Function for attention.
        def run_tri():
            m.zero_grad(set_to_none=True)
            x_local = x.detach().clone().requires_grad_(True)
            B_, T_, C_ = x_local.size()
            alpha_k = m.alpha_proj_k(x_local).view(B_, T_, H, J) + m.b_alpha_k
            alpha_v = m.alpha_proj_v(x_local).view(B_, T_, H, J) + m.b_alpha_v
            wk = F.softmax(alpha_k.float(), dim=-1).to(x_local.dtype)
            wv = F.softmax(alpha_v.float(), dim=-1).to(x_local.dtype)
            q = m.c_q(x_local).view(B_, T_, H, D)
            kb = m.c_k(x_local).view(B_, T_, J, D)
            vb = m.c_v(x_local).view(B_, T_, J, D)
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            kb = apply_rotary_emb(kb, cos, sin)
            q = norm(q) * 1.2
            kb = norm(kb) * 1.2
            if ve is not None:
                ve_t = ve.view(B_, T_, J, D).contiguous()
                gate = (3 * torch.sigmoid(m.ve_gate(x_local[..., : m.ve_gate_channels]))).to(x_local.dtype)
            else:
                ve_t = None
                gate = None
            y = bqa_dyn_attn(q.contiguous(), kb.contiguous(), vb.contiguous(), wk.contiguous(), wv.contiguous(),
                             ve=ve_t, gate=gate, causal=True, window_size=(win, 0))
            y = y.contiguous().view(B_, T_, -1)
            y = m.c_proj(y)
            y.float().pow(2).mean().backward()

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ms_old = bench_one(run_old)
        peak_old = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ms_tri = bench_one(run_tri)
        peak_tri = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{case_name}]  old fwd+bwd: {ms_old:7.3f} ms (peak {peak_old:.2f} GB)   "
              f"triton fwd+bwd: {ms_tri:7.3f} ms (peak {peak_tri:.2f} GB)   speedup: {ms_old/ms_tri:.2f}x")


def main():
    print(f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
    print(f"cuda mem free / total: {torch.cuda.mem_get_info()[0]/1e9:.1f} / {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")

    equivalence(torch.float32, win=1024)
    equivalence(torch.float32, win=64)            # tight window stress
    equivalence(torch.bfloat16, win=1024)
    equivalence(torch.bfloat16, win=64)

    grad_check(torch.float32, win=1024)
    grad_check(torch.float32, win=32)
    grad_check(torch.bfloat16, win=1024)

    if os.environ.get("VERIFY_SKIP_BENCH"):
        return

    bench(B=8,  T=2048, win=1024)
    bench(B=32, T=2048, win=1024)
    bench(B=8,  T=2048, n_head=8, n_kv_head=4, n_embd=1024, n_layer=20, win=1024)

    bench_fb(B=8,  T=2048, win=1024)
    bench_fb(B=32, T=2048, win=1024)
    bench_fb(B=8,  T=2048, n_head=8, n_kv_head=4, n_embd=1024, n_layer=20, win=1024)


if __name__ == "__main__":
    main()
