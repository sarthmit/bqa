"""End-to-end smoke test: DynamicBasisQueryAttention forward+backward with the
config.bqa_dyn_use_triton switch flipped — checks that both paths produce
near-identical outputs/grads through the actual module (linear projections,
rotary, RMS norm, ×1.2 scale, ve gate, c_proj all included).
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mila/m/mittalsa/scratch/bqa")

from nanochat.gpt import GPTConfig, DynamicBasisQueryAttention

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
dtype = torch.bfloat16


def make_cos_sin(T, head_dim):
    d = head_dim // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    f = torch.einsum("t,f->tf", pos, inv)
    return f.cos().to(dtype)[None, :, None, :], f.sin().to(dtype)[None, :, None, :]


def make_module(use_triton, layer_idx, n_layer=4, n_head=6, n_kv_head=3, n_embd=768):
    cfg = GPTConfig(
        sequence_len=2048, vocab_size=32768, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd,
        bqa_dyn_use_triton=use_triton,
    )
    torch.manual_seed(0)
    return DynamicBasisQueryAttention(cfg, layer_idx=layer_idx).to(device=device, dtype=dtype), cfg


def normwise(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-8)).item()


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    B, T = 4, 1024
    n_layer, n_head, n_kv_head, n_embd = 4, 6, 3, 768
    H, J, D = n_head, n_kv_head, n_embd // n_head
    cos_sin = make_cos_sin(T, D)
    win = (256, 0)

    for case_name, layer_idx in [("ve=None", 0), ("ve!=None", n_layer - 1)]:
        # Build module and clone weights so SDPA and Triton share identical params.
        m_sdpa, cfg = make_module(use_triton=False, layer_idx=layer_idx,
                                  n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd)
        m_tri, _ = make_module(use_triton=True, layer_idx=layer_idx,
                               n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd)
        m_tri.load_state_dict(m_sdpa.state_dict())

        torch.manual_seed(123)
        x = torch.randn(B, T, cfg.n_embd, device=device, dtype=dtype)
        ve = torch.randn(B, T, J * D, device=device, dtype=dtype) if m_sdpa.ve_gate is not None else None

        x1 = x.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)

        y_sdpa = m_sdpa(x1, ve=ve, cos_sin=cos_sin, window_size=win, kv_cache=None)
        y_tri = m_tri(x2, ve=ve, cos_sin=cos_sin, window_size=win, kv_cache=None)

        torch.manual_seed(0)
        dy = torch.randn_like(y_sdpa)
        y_sdpa.backward(dy)
        y_tri.backward(dy)

        print(f"\n[{case_name}]  y normwise_rel={normwise(y_tri, y_sdpa):.3e}  cos={cos(y_tri, y_sdpa):.6f}")
        print(f"  dx          normwise_rel={normwise(x2.grad, x1.grad):.3e}  cos={cos(x2.grad, x1.grad):.6f}")
        for name, p_sdpa in m_sdpa.named_parameters():
            if p_sdpa.grad is None:
                continue
            p_tri = dict(m_tri.named_parameters())[name]
            assert p_tri.grad is not None, name
            n = normwise(p_tri.grad, p_sdpa.grad)
            c = cos(p_tri.grad, p_sdpa.grad)
            print(f"  {name:24s} normwise_rel={n:.3e}  cos={c:.6f}  mean|g|={p_sdpa.grad.float().abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
