"""Block-size tuning for bqa_dyn_triton fwd + bwd kernels.

Sweeps BLOCK_M, BLOCK_N, num_warps, num_stages directly through the kernel-launch
helpers (not the autograd path) and measures forward/backward time on a real shape.

Use to inform the default block-size policy in bqa_dyn_triton.py.
"""
import sys, time, argparse, itertools
import torch, triton

sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")
from nanochat.bqa_dyn_triton import bqa_dyn_attn_triton_fwd, bqa_dyn_attn_triton_bwd

device = torch.device("cuda")


def make_inputs(B, T, H, J, D, dtype, ve_on):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, T, J, D, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, T, J, D, device=device, dtype=dtype) * 0.5
    wk = torch.softmax(torch.randn(B, T, H, J, device=device, dtype=torch.float32), dim=-1).to(dtype)
    wv = torch.softmax(torch.randn(B, T, H, J, device=device, dtype=torch.float32), dim=-1).to(dtype)
    if ve_on:
        ve = torch.randn(B, T, J, D, device=device, dtype=dtype) * 0.5
        gate = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 3
    else:
        ve, gate = None, None
    return q, k, v, wk, wv, ve, gate


def bench(fn, n_warmup=5, n_iters=15):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iters


def tune_fwd(B, T, H, J, D, dtype, ve_on, win):
    q, k, v, wk, wv, ve, gate = make_inputs(B, T, H, J, D, dtype, ve_on)
    print(f"\n=== FWD tune  B={B} T={T} H={H} J={J} D={D} dtype={dtype} ve={ve_on} win={win} ===")
    print(f"{'BLOCK_M':>8} {'BLOCK_N':>8} {'warps':>5} {'stages':>6}  {'ms':>8}")

    # Reasonable search space, filtered to only PoT
    cfgs = []
    for bm, bn, w, st in itertools.product([32, 64, 128], [16, 32, 64], [4, 8], [1, 2, 3]):
        if bm * bn > 64 * 64 and st > 1:
            continue  # SMEM headroom heuristic
        cfgs.append((bm, bn, w, st))

    best = (None, float("inf"))
    for bm, bn, w, st in cfgs:
        try:
            def fn():
                bqa_dyn_attn_triton_fwd(
                    q, k, v, wk, wv, ve=ve, gate=gate,
                    causal=True, window_size=(win, 0),
                    block_m=bm, block_n=bn, num_warps=w, num_stages=st,
                )
            ms = bench(fn)
            tag = ""
            if ms < best[1]:
                best = ((bm, bn, w, st), ms)
                tag = "  <- best"
            print(f"{bm:>8} {bn:>8} {w:>5} {st:>6}  {ms:>8.3f}{tag}")
        except Exception as e:
            print(f"{bm:>8} {bn:>8} {w:>5} {st:>6}  FAIL: {type(e).__name__}")

    print(f"BEST: BLOCK_M={best[0][0]} BLOCK_N={best[0][1]} warps={best[0][2]} stages={best[0][3]} -> {best[1]:.3f} ms")
    return best


def tune_bwd(B, T, H, J, D, dtype, ve_on, win):
    q, k, v, wk, wv, ve, gate = make_inputs(B, T, H, J, D, dtype, ve_on)
    o, L = bqa_dyn_attn_triton_fwd(q, k, v, wk, wv, ve=ve, gate=gate, causal=True, window_size=(win, 0), return_L=True)
    dy = torch.randn_like(o)
    print(f"\n=== BWD tune  B={B} T={T} H={H} J={J} D={D} dtype={dtype} ve={ve_on} win={win} ===")
    print(f"{'dqM':>4} {'dqN':>4} {'dkvM':>4} {'dkvN':>4} {'warps':>5} {'stages':>6}  {'ms':>8}")
    cfgs = []
    for dqm, dqn, dkvm, dkvn, w, st in itertools.product(
        [32, 64, 128], [16, 32], [16, 32, 64], [16, 32], [4, 8], [1, 2],
    ):
        if dqm * dqn > 64 * 64 and st > 1:
            continue
        if dkvm * dkvn > 64 * 64 and st > 1:
            continue
        cfgs.append((dqm, dqn, dkvm, dkvn, w, st))
    best = (None, float("inf"))
    for dqm, dqn, dkvm, dkvn, w, st in cfgs:
        try:
            def fn():
                bqa_dyn_attn_triton_bwd(
                    dy, q, k, v, wk, wv, o, L, ve=ve, gate=gate,
                    causal=True, window_size=(win, 0),
                    dq_block_m=dqm, dq_block_n=dqn,
                    dkv_block_m=dkvm, dkv_block_n=dkvn,
                    num_warps=w, num_stages=st,
                )
            ms = bench(fn)
            tag = ""
            if ms < best[1]:
                best = ((dqm, dqn, dkvm, dkvn, w, st), ms)
                tag = "  <- best"
            print(f"{dqm:>4} {dqn:>4} {dkvm:>4} {dkvn:>4} {w:>5} {st:>6}  {ms:>8.3f}{tag}")
        except Exception as e:
            print(f"{dqm:>4} {dqn:>4} {dkvm:>4} {dkvn:>4} {w:>5} {st:>6}  FAIL: {type(e).__name__}")

    print(f"BEST: dq=({best[0][0]},{best[0][1]}) dkv=({best[0][2]},{best[0][3]}) warps={best[0][4]} stages={best[0][5]} -> {best[1]:.3f} ms")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=str, required=True,
                    help="comma-separated B,T,H,J,D")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--mode", type=str, default="fwd", choices=["fwd", "bwd", "both"])
    ap.add_argument("--ve", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    B, T, H, J, D = [int(x) for x in args.shape.split(",")]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ve_on = bool(args.ve)
    print(f"GPU: {torch.cuda.get_device_name(0)}, triton {triton.__version__}")
    if args.mode in ("fwd", "both"):
        tune_fwd(B, T, H, J, D, dtype, ve_on, args.win)
    if args.mode in ("bwd", "both"):
        tune_bwd(B, T, H, J, D, dtype, ve_on, args.win)


if __name__ == "__main__":
    main()
