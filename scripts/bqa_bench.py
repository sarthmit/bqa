"""
Micro-benchmark: MHA vs GQA vs BQA attention forward+backward at a single layer,
to localize the BQA slowdown observed in d12_bqa_2e18 (~73K tok/s vs ~865K tok/s for MHA).

Run with:
  source scripts/setup_node.sh
  torchrun --standalone --nproc_per_node=1 -m scripts.bqa_bench   # or plain `python -m scripts.bqa_bench`

Prints per-config: forward-only ms, fwd+bwd ms, peak memory.
Also probes the SDPA backend chosen via torch.nn.attention.SDPBackend hints.
"""
import time
import torch
import torch.nn as nn

from nanochat.gpt import GPTConfig, CausalSelfAttention, BasisQueryAttention

torch.manual_seed(0)
device = torch.device("cuda")
dtype = torch.bfloat16

B, T = 32, 2048
n_embd = 768
n_head = 6
head_dim = n_embd // n_head

# Common cos/sin
def build_cos_sin(T, head_dim, device, dtype, base=100000):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    cos = cos.view(1, T, 1, half)
    sin = sin.view(1, T, 1, half)
    return cos, sin

cos, sin = build_cos_sin(T, head_dim, device, dtype)
window_size = (-1, -1)  # full context, "L"

def make_attn(kind, n_kv_head, realistic_dtype=True):
    """If realistic_dtype, weights stay fp32 (as in real training, where the
    custom Linear casts at call time), else cast everything to compute dtype
    (legacy bench mode — convenient but hides fp32/bf16 dispatch bugs)."""
    cfg = GPTConfig(
        sequence_len=T, vocab_size=32768, n_layer=12,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
        attn_kind=kind, window_pattern="L",
    )
    if kind == "bqa":
        m = BasisQueryAttention(cfg, layer_idx=1)
    else:
        m = CausalSelfAttention(cfg, layer_idx=1)
    if realistic_dtype:
        return m.to(device=device)  # weights stay fp32, x will be bf16
    return m.to(device=device, dtype=dtype)

def time_module(name, attn, iters=20, warmup=5):
    x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
    ve = None
    # warmup
    for _ in range(warmup):
        y = attn(x, ve, (cos, sin), window_size, None)
        y.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # forward only
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            y = attn(x, ve, (cos, sin), window_size, None)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000 / iters

    # fwd+bwd
    t0 = time.perf_counter()
    for _ in range(iters):
        x.grad = None
        for p in attn.parameters():
            p.grad = None
        y = attn(x, ve, (cos, sin), window_size, None)
        y.sum().backward()
    torch.cuda.synchronize()
    fb_ms = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{name:<12} fwd: {fwd_ms:7.2f} ms | fwd+bwd: {fb_ms:7.2f} ms | peak: {peak:7.1f} MiB")

print(f"GPU: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability()}")
print(f"shape: B={B}, T={T}, n_head={n_head}, head_dim={head_dim}, dtype={dtype}\n")

for tag, kind, nkv in [
    ("mha (kv=6)", "gqa", 6),
    ("gqa (kv=3)", "gqa", 3),
    ("bqa (kv=3)", "bqa", 3),
]:
    attn = make_attn(kind, nkv)
    time_module(tag, attn)
    del attn
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# Localize the BQA backward slowdown
# ----------------------------------------------------------------------
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from nanochat.gpt import apply_rotary_emb, norm

print("\n-- BQA backward localization --")

# Build a fresh BQA module to grab its weights
bqa = make_attn("bqa", 3)

class BQAVariant(nn.Module):
    """Reimplements BQA forward with optional patches to localize the slowdown."""
    def __init__(self, src, contiguous_kv=False, w_in_compute_dtype=False, force_flash=False):
        super().__init__()
        self.c_q, self.c_k, self.c_v, self.c_proj, self.alpha = src.c_q, src.c_k, src.c_v, src.c_proj, src.alpha_k
        self.n_head, self.n_kv_head = src.n_head, src.n_kv_head
        self.head_dim = src.head_dim
        self.contiguous_kv = contiguous_kv
        self.w_in_compute_dtype = w_in_compute_dtype
        self.force_flash = force_flash

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B_, T_, _ = x.shape
        q = self.c_q(x).view(B_, T_, self.n_head, self.head_dim)
        k_basis = self.c_k(x).view(B_, T_, self.n_kv_head, self.head_dim)
        v_basis = self.c_v(x).view(B_, T_, self.n_kv_head, self.head_dim)
        if self.w_in_compute_dtype:
            w = F.softmax(self.alpha, dim=-1)  # native dtype, no fp32 detour
        else:
            w = F.softmax(self.alpha.float(), dim=-1).to(k_basis.dtype)
        k = torch.einsum('hj,btjd->bthd', w, k_basis)
        v = torch.einsum('hj,btjd->bthd', w, v_basis)
        if self.contiguous_kv:
            k = k.contiguous()
            v = v.contiguous()
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2; k = k * 1.2
        # SDPA in (B,H,T,D) layout
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.force_flash:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B_, T_, -1)
        return self.c_proj(y)

# Skip the variant sweep — none of {contiguous_kv, w_native_dtype, force_flash} helped.

# ----------------------------------------------------------------------
# Bisect: time individual stages of the BQA forward
# ----------------------------------------------------------------------
print("\n-- Stage isolation (forward+backward of just the named stage) --")

def bench_fn(name, fn, iters=20, warmup=5):
    for _ in range(warmup):
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        x_grad_zero()
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()
    fb_ms = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{name:<40} fwd+bwd: {fb_ms:7.2f} ms | peak: {peak:7.1f} MiB")

x = torch.randn(B, T, n_embd, device=device, dtype=dtype, requires_grad=True)
def x_grad_zero():
    x.grad = None

# Just linears
c_q, c_k_full, c_v_full = bqa.c_q, nn.Linear(n_embd, n_head*head_dim, bias=False).to(device, dtype), nn.Linear(n_embd, n_head*head_dim, bias=False).to(device, dtype)
c_k_basis, c_v_basis = bqa.c_k, bqa.c_v
alpha = bqa.alpha_k

def linears_only_mha():
    q = c_q(x); k = c_k_full(x); v = c_v_full(x)
    return q + k + v
def linears_only_bqa():
    q = c_q(x); k = c_k_basis(x); v = c_v_basis(x)
    return q + torch.cat([k]*2, dim=-1) + torch.cat([v]*2, dim=-1)

bench_fn("linears MHA (3 full Linear)", linears_only_mha)
bench_fn("linears BQA (q full, k/v basis)", linears_only_bqa)

# Einsum stage alone
def einsum_only():
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = torch.einsum('hj,btjd->bthd', w, k_b)
    v = torch.einsum('hj,btjd->bthd', w, v_b)
    return k.sum(0) + v.sum(0)

bench_fn("einsum stage (softmax+two einsums)", einsum_only)

# Linears + einsum, no SDPA
def pre_sdpa_bqa():
    q = c_q(x).view(B, T, n_head, head_dim)
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = torch.einsum('hj,btjd->bthd', w, k_b)
    v = torch.einsum('hj,btjd->bthd', w, v_b)
    return q.sum(0) + k.sum(0) + v.sum(0)

bench_fn("BQA pre-SDPA (linears+einsum)", pre_sdpa_bqa)

# SDPA only on (n_head=6) k/v that originate from einsum vs from a fresh linear
def sdpa_only_mha_origin():
    q = c_q(x).view(B, T, n_head, head_dim).transpose(1, 2)
    k = c_k_full(x).view(B, T, n_head, head_dim).transpose(1, 2)
    v = c_v_full(x).view(B, T, n_head, head_dim).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def sdpa_only_bqa_origin():
    q = c_q(x).view(B, T, n_head, head_dim).transpose(1, 2)
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = torch.einsum('hj,btjd->bthd', w, k_b).transpose(1, 2)
    v = torch.einsum('hj,btjd->bthd', w, v_b).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

bench_fn("SDPA(k/v from full Linear)", sdpa_only_mha_origin)
bench_fn("SDPA(k/v from einsum-mix)", sdpa_only_bqa_origin)

# Candidate fix: replace einsum with a matmul (lowered to cuBLAS GEMM).
# (B,T,J,D) @ w.T equiv: F.linear over the J dim by transposing D <-> J.
def linear_mix(basis, w_):
    # basis: (B,T,J,D); w_: (H,J). Output: (B,T,H,D).
    return F.linear(basis.transpose(-1, -2), w_).transpose(-1, -2).contiguous()

def einsum_stage_linear():
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = linear_mix(k_b, w)
    v = linear_mix(v_b, w)
    return k.sum(0) + v.sum(0)

def sdpa_with_linear_mix():
    q = c_q(x).view(B, T, n_head, head_dim).transpose(1, 2)
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = linear_mix(k_b, w).transpose(1, 2)
    v = linear_mix(v_b, w).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

bench_fn("einsum-FIX (linear_mix)", einsum_stage_linear)
bench_fn("SDPA(k/v from linear_mix)", sdpa_with_linear_mix)

# Also try a bmm variant just to see if it differs.
def bmm_mix(basis, w_):
    # basis (B,T,J,D); w_ (H,J)
    # treat (B*T, D, J) @ (J, H) -> (B*T, D, H)
    B_, T_, J_, D_ = basis.shape
    H_ = w_.shape[0]
    out = basis.permute(0, 1, 3, 2).reshape(-1, J_) @ w_.t()
    return out.view(B_, T_, D_, H_).permute(0, 1, 3, 2).contiguous()

def einsum_stage_bmm():
    k_b = c_k_basis(x).view(B, T, 3, head_dim)
    v_b = c_v_basis(x).view(B, T, 3, head_dim)
    w = F.softmax(alpha.float(), dim=-1).to(k_b.dtype)
    k = bmm_mix(k_b, w)
    v = bmm_mix(v_b, w)
    return k.sum(0) + v.sum(0)

bench_fn("einsum-FIX (bmm reshape)", einsum_stage_bmm)

# Weight-level mix: K = (sum_j w[h,j] * W_basis_j) @ x.
# Mathematically identical to the activation-level mix by linearity, but the
# expensive small-output reduction `dw` is now over (D*E)=98K instead of (B*T*D)=8.4M.
# We use the basis Linears' weights as the W_basis tensor.
def stage_weight_mix():
    # c_k_basis.weight shape: (n_kv_head*D, n_embd). Reshape to (J, D, E).
    Wk = c_k_basis.weight.view(3, head_dim, n_embd)
    Wv = c_v_basis.weight.view(3, head_dim, n_embd)
    w = F.softmax(alpha.float(), dim=-1).to(Wk.dtype)  # (H, J)
    Wk_eff = torch.einsum('hj,jde->hde', w, Wk).reshape(n_head * head_dim, n_embd)
    Wv_eff = torch.einsum('hj,jde->hde', w, Wv).reshape(n_head * head_dim, n_embd)
    k = F.linear(x, Wk_eff).view(B, T, n_head, head_dim)
    v = F.linear(x, Wv_eff).view(B, T, n_head, head_dim)
    return k.sum(0) + v.sum(0)

def sdpa_with_weight_mix():
    Wk = c_k_basis.weight.view(3, head_dim, n_embd)
    Wv = c_v_basis.weight.view(3, head_dim, n_embd)
    w = F.softmax(alpha.float(), dim=-1).to(Wk.dtype)
    Wk_eff = torch.einsum('hj,jde->hde', w, Wk).reshape(n_head * head_dim, n_embd)
    Wv_eff = torch.einsum('hj,jde->hde', w, Wv).reshape(n_head * head_dim, n_embd)
    q = c_q(x).view(B, T, n_head, head_dim).transpose(1, 2)
    k = F.linear(x, Wk_eff).view(B, T, n_head, head_dim).transpose(1, 2)
    v = F.linear(x, Wv_eff).view(B, T, n_head, head_dim).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

bench_fn("WEIGHT-mix stage (small einsum + Linear)", stage_weight_mix)
bench_fn("SDPA(k/v from WEIGHT-mix)", sdpa_with_weight_mix)

# ----------------------------------------------------------------------
# Parity + end-to-end timing of the PATCHED BQA module
# ----------------------------------------------------------------------
print("\n-- Patched BQA module --")
patched = make_attn("bqa", 3)  # uses the modified BasisQueryAttention
# Reuse the same weights from `bqa` so we can compare bit-for-bit (modulo fp rounding).
patched.load_state_dict(bqa.state_dict())

# The OLD implementation, frozen here for parity comparison.
class BQALegacy(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.c_q, self.c_k, self.c_v, self.c_proj, self.alpha = src.c_q, src.c_k, src.c_v, src.c_proj, src.alpha_k
        self.n_head, self.n_kv_head, self.head_dim = src.n_head, src.n_kv_head, src.head_dim

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B_, T_, _ = x.shape
        q = self.c_q(x).view(B_, T_, self.n_head, self.head_dim)
        k_basis = self.c_k(x).view(B_, T_, self.n_kv_head, self.head_dim)
        v_basis = self.c_v(x).view(B_, T_, self.n_kv_head, self.head_dim)
        w = F.softmax(self.alpha.float(), dim=-1).to(k_basis.dtype)
        k = torch.einsum('hj,btjd->bthd', w, k_basis)
        v = torch.einsum('hj,btjd->bthd', w, v_basis)
        from nanochat.gpt import apply_rotary_emb, norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k); q = q * 1.2; k = k * 1.2
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B_, T_, -1)
        return self.c_proj(y)

legacy = BQALegacy(bqa).to(device, dtype)

x_test = torch.randn(B, T, n_embd, device=device, dtype=dtype)
with torch.no_grad():
    y_old = legacy(x_test, None, (cos, sin), window_size, None)
    y_new = patched(x_test, None, (cos, sin), window_size, None)
diff = (y_old - y_new).abs().max().item()
rel = diff / y_old.abs().max().item()
print(f"parity (no ve):  max|Δ| = {diff:.3e}   rel = {rel:.3e}   (bf16 ~1e-2 OK)")

time_module("bqa PATCHED (no ve)", patched)

# ----------------------------------------------------------------------
# ve-residual term: einsum vs bmm reformulation
# ----------------------------------------------------------------------
print("\n-- ve term reformulation --")
gate_lin = nn.Linear(12, 3, bias=False).to(device, dtype)

def ve_einsum():
    ve_t = torch.randn(B, T, 3, head_dim, device=device, dtype=dtype, requires_grad=True)
    gate = 3 * torch.sigmoid(gate_lin(x[..., :12]))
    w = F.softmax(alpha.float(), dim=-1).to(ve_t.dtype)
    return torch.einsum('hj,btj,btjd->bthd', w, gate, ve_t).sum(0)

def ve_bmm():
    ve_t = torch.randn(B, T, 3, head_dim, device=device, dtype=dtype, requires_grad=True)
    gate = 3 * torch.sigmoid(gate_lin(x[..., :12]))
    w = F.softmax(alpha.float(), dim=-1).to(ve_t.dtype)
    mix = w.unsqueeze(0).unsqueeze(0) * gate.unsqueeze(2)  # (B,T,H,J)
    return torch.matmul(mix, ve_t).sum(0)  # (B,T,H,D) -> sum over B for the bench's loss surrogate

bench_fn("ve einsum  ('hj,btj,btjd->bthd')", ve_einsum)
bench_fn("ve bmm     (precomposed M @ ve)", ve_bmm)

# Numerical parity for the ve term
torch.manual_seed(42)
ve_test = torch.randn(B, T, 3, head_dim, device=device, dtype=dtype)
gate = 3 * torch.sigmoid(gate_lin(x[..., :12]))
w_test = F.softmax(alpha.float(), dim=-1).to(ve_test.dtype)
y_e = torch.einsum('hj,btj,btjd->bthd', w_test, gate, ve_test)
mix = w_test.unsqueeze(0).unsqueeze(0) * gate.unsqueeze(2)
y_b = torch.matmul(mix, ve_test)
print(f"ve parity:  max|Δ| = {(y_e - y_b).abs().max().item():.3e}")

# Probe which SDPA backend is selected by running a couple of variants explicitly
print("\n-- SDPA backend probe --")
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q6 = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
k6 = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
v6 = torch.randn(B, n_head, T, head_dim, device=device, dtype=dtype)
k3 = torch.randn(B, 3, T, head_dim, device=device, dtype=dtype)
v3 = torch.randn(B, 3, T, head_dim, device=device, dtype=dtype)

for label, q, k, v, gqa in [
    ("MHA-6:6", q6, k6, v6, False),
    ("GQA-6:3", q6, k3, v3, True),
]:
    for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel(backends=[backend]):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(20):
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000 / 20
                print(f"  {label} {backend.name:<22}: {ms:7.2f} ms")
        except Exception as e:
            print(f"  {label} {backend.name:<22}: ERR {type(e).__name__}: {e}")
