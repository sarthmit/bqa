"""Probe single-shape (d16 J=8 fb on) timing of bqa_dyn_attn through autograd Fn
to isolate why bench_bqa_dyn_sweep was reporting 231ms at this shape.
"""
import sys, time, torch
sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")
from nanochat.bqa_dyn_triton import bqa_dyn_attn, bqa_dyn_attn_triton_fwd, bqa_dyn_attn_triton_bwd

device = torch.device("cuda")
dtype = torch.bfloat16

B, T, H, J, D = 8, 2048, 8, 8, 128
torch.manual_seed(0)
q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True) * 0.5
k = torch.randn(B, T, J, D, device=device, dtype=dtype, requires_grad=True) * 0.5
v = torch.randn(B, T, J, D, device=device, dtype=dtype, requires_grad=True) * 0.5
wk = torch.softmax(torch.randn(B, T, H, J, device=device, dtype=torch.float32), dim=-1).to(dtype).requires_grad_(True)
wv = torch.softmax(torch.randn(B, T, H, J, device=device, dtype=torch.float32), dim=-1).to(dtype).requires_grad_(True)
ve = (torch.randn(B, T, J, D, device=device, dtype=dtype) * 0.5).requires_grad_(True)
gate = (torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 3).requires_grad_(True)


def fb_via_autograd():
    qd, kd, vd = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    wkd, wvd = wk.detach().clone().requires_grad_(True), wv.detach().clone().requires_grad_(True)
    ved, gd = ve.detach().clone().requires_grad_(True), gate.detach().clone().requires_grad_(True)
    y = bqa_dyn_attn(qd, kd, vd, wkd, wvd, ve=ved, gate=gd, causal=True, window_size=(1024, 0))
    y.float().pow(2).mean().backward()


def fwd_only():
    bqa_dyn_attn_triton_fwd(q.detach(), k.detach(), v.detach(), wk.detach(), wv.detach(),
                             ve=ve.detach(), gate=gate.detach(), causal=True, window_size=(1024, 0))


def bwd_only():
    out, L = bqa_dyn_attn_triton_fwd(q.detach(), k.detach(), v.detach(), wk.detach(), wv.detach(),
                                       ve=ve.detach(), gate=gate.detach(), causal=True, window_size=(1024, 0),
                                       return_L=True)
    dy = torch.randn_like(out)
    bqa_dyn_attn_triton_bwd(dy, q.detach(), k.detach(), v.detach(), wk.detach(), wv.detach(), out, L,
                             ve=ve.detach(), gate=gate.detach(), causal=True, window_size=(1024, 0))


def bench(fn, name, warmup=5, iters=15):
    torch.cuda.synchronize()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000 / iters
    print(f"{name:25s}: {dt:.3f} ms")


print(f"d16 J=8 ve=on shape: B={B} T={T} H={H} J={J} D={D}")
bench(fwd_only, "fwd only")
bench(bwd_only, "fwd+bwd direct kernels")
bench(fb_via_autograd, "fb via bqa_dyn_attn (autograd)")
