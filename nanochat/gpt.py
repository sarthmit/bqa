"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA) / basis K/V heads (BQA)
    n_embd: int = 768
    # Attention kind:
    #   "gqa"      — Grouped-Query Attention: query head i attends to basis K/V head (i // group_size)
    #   "bqa"      — Basis Query Attention: each query head attends to a learned softmax mixture
    #                of all n_kv_head basis K/V heads (alpha logits, shape (n_head, n_kv_head)).
    #                At init, alpha is set so BQA recovers GQA exactly; training can then deviate.
    #   "bqa_dyn"  — Per-query logit-mix BQA: independent K and V mixing logits are produced
    #                per token by Linear `alpha_proj_{k,v}(x)` plus learned static biases
    #                `b_alpha_{k,v}` (mirrors BQA static's alpha_k / alpha_v). Each query
    #                position picks its own (separate) K and V basis combinations, applied
    #                to every past key / past value.
    #                Implemented by materializing per-pair basis scores S[b,h,t,s,j] and
    #                summing across j with w[b,t,h,j] BEFORE softmax — single softmax per
    #                (t,h), unlike Mixture-of-Softmaxes. See DynamicBasisQueryAttention.
    attn_kind: str = "gqa"
    # BQA / bqa_dyn: target softmax probability mass placed on the GQA-assigned
    # basis at init, separately for K and V. The remaining (1 - m) is split
    # uniformly over the other (n_kv_head - 1) basis heads; the actual init
    # logit is derived as L = log(m / (1 - m)) + log(n_kv_head - 1) so the
    # init distribution shape is invariant to n_kv_head (no per-depth tuning).
    # K and V are separated because trained-entropy data shows V converges
    # much more concentrated than K (≈0.07 vs ≈0.40 ratio at d16), so m_v can
    # usefully be initialized higher than m_k. Defaults (0.70, 0.85) from the
    # upstream apr28 d16 (m_k, m_v) sweep: argmin at d16 across 1e18/2.15e18/4.64e18,
    # top-3 at d12 and d20, and beats GQA at d16/4.64e18. Set to 1/n_kv_head
    # for uniform init; n_kv_head=1 falls through to a no-op. bqa_dyn uses both
    # m_k and m_v (one per independent K/V mixing-logit head), same as static BQA.
    bqa_init_mass_k: float = 0.70
    bqa_init_mass_v: float = 0.85
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"

    @property
    def cache_kv_heads(self):
        """Number of K/V heads stored in the inference KV cache.
        For GQA the cache is sized to the (smaller) basis count; for BQA the cache holds
        the post-mix K/V, which has full n_head heads.
        For bqa_dyn the cache would need to store basis K/V (mix is per-query at inference
        time), but inference cache is not yet implemented for bqa_dyn — training-only."""
        if self.attn_kind == "bqa":
            return self.n_head
        return self.n_kv_head


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class BasisQueryAttention(nn.Module):
    """
    Basis Query Attention (BQA) — a strict generalisation of GQA.

    GQA assigns each query head to one of n_kv_head basis K/V heads (groups of size
    n_head // n_kv_head share K/V). BQA replaces that hard assignment with a learned
    convex combination, with *independent* mixings for K and V: parameters
    `alpha_k`, `alpha_v` of shape (n_head, n_kv_head) are each softmaxed along the
    basis dimension to produce w_k[h, j], w_v[h, j], and query head h attends to
    K_h = sum_j w_k[h,j] * K_basis[j], V_h = sum_j w_v[h,j] * V_basis[j].

    Recovers GQA exactly when both alpha_k and alpha_v are one-hot on the assigned
    basis (the init we use here, with high logits on the assigned head). After mixing,
    attention runs as full MHA over n_head heads — so the inference KV cache stores
    the post-mix K/V of shape (B, T, n_head, head_dim), not the basis.
    (See GPTConfig.cache_kv_heads.)
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Independent K/V mixing logits over basis heads. Real init in GPT.init_weights.
        self.alpha_k = nn.Parameter(torch.zeros(self.n_head, self.n_kv_head))
        self.alpha_v = nn.Parameter(torch.zeros(self.n_head, self.n_kv_head))
        # Cache for the basis-folded effective K/V projection weights, populated
        # only in no-grad contexts (eval / inference) where alpha and c_{k,v}
        # don't move between calls. Holds (Wk_eff, Wv_eff, w_v) in x.dtype, or
        # None when stale. Invalidated at the start of every grad-enabled
        # forward so the next no-grad pass recomputes against fresh weights.
        self._w_eff_cache = None  # (Wk_eff, Wv_eff, w_v) bf16/fp32 tensors, or None
        # ve gating is per query head (n_head outputs), applied AFTER the basis
        # mix of ve. GQA's ve_gate is per-basis-head because GQA's V lives in
        # the basis space; BQA's V lives in the post-mix per-query-head space,
        # so the natural place for the gate is there too. This lets each query
        # head independently scale its value-residual signal — under the old
        # per-basis gate, all query heads sharing nontrivial w_v on basis j
        # were forced to scale ve through the same gate[j].
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def _compute_basis_mix(self, dtype):
        """Fold the alpha-softmax mix into the K/V projection weights.

        Returns (Wk_eff, Wv_eff, w_v) all cast to `dtype`. Mixed in fp32 for
        stability (matches alpha.float() softmax), then cast at the end so
        the downstream F.linear / matmul runs in the activation dtype.
        """
        w_k = F.softmax(self.alpha_k.float(), dim=-1)  # (n_head, n_kv_head), fp32
        w_v = F.softmax(self.alpha_v.float(), dim=-1)  # (n_head, n_kv_head), fp32
        Wk_basis = self.c_k.weight.float().view(self.n_kv_head, self.head_dim, self.n_embd)
        Wv_basis = self.c_v.weight.float().view(self.n_kv_head, self.head_dim, self.n_embd)
        Wk_eff = torch.einsum('hj,jde->hde', w_k, Wk_basis).reshape(self.n_head * self.head_dim, self.n_embd)
        Wv_eff = torch.einsum('hj,jde->hde', w_v, Wv_basis).reshape(self.n_head * self.head_dim, self.n_embd)
        return Wk_eff.to(dtype), Wv_eff.to(dtype), w_v.to(dtype)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Mix at the WEIGHT level rather than the activation level: by linearity,
        #   sum_j w[h,j] * (W_basis_j @ x) == (sum_j w[h,j] W_basis_j) @ x
        # so we can fold the basis combination into an effective per-query-head
        # weight matrix, then run a normal Linear. This avoids the activation-
        # level einsum 'hj,btjd->bthd' whose backward has a degenerate dw
        # reduction over (B*T*D) into a tiny (n_head*n_kv_head) output —
        # bf16 split-K + atomic accumulate, ~17x slower than MHA in fwd+bwd
        # on A100. (See scripts/bqa_bench.py.)
        if torch.is_grad_enabled():
            # Training: alpha_{k,v} and c_{k,v}.weight all need gradients on
            # every microbatch, so the fold must run inside the autograd graph.
            # Drop any stale eval-time cache so the next no-grad pass refreshes.
            self._w_eff_cache = None
            Wk_eff, Wv_eff, w_v = self._compute_basis_mix(x.dtype)
        else:
            # Eval / inference: alpha and c_{k,v} are constant across forwards;
            # cache the fold so a multi-batch eval or a long generation only
            # pays the einsum + cast on the first call per layer.
            cache = self._w_eff_cache
            if cache is None or cache[0].dtype != x.dtype:
                cache = self._compute_basis_mix(x.dtype)
                self._w_eff_cache = cache
            Wk_eff, Wv_eff, w_v = cache

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = F.linear(x, Wk_eff).view(B, T, self.n_head, self.head_dim)
        v = F.linear(x, Wv_eff).view(B, T, self.n_head, self.head_dim)

        # Value residual: first mix the per-basis-head value embedding through
        # the same w_v as V to lift it into per-query-head space, then apply
        # an independent per-query-head gate. The matmul broadcasts w_v over
        # (B, T): w_v=(H,J), ve=(B,T,J,D) → (B,T,H,D).
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            ve_mixed = torch.matmul(w_v, ve)  # (B, T, H, D)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, H)
            v = v + gate.unsqueeze(-1) * ve_mixed

        # Rotary + QK norm + attention (identical to MHA from here on).
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Cache stores post-mix K/V at full n_head shape (see GPTConfig.cache_kv_heads).
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class DynamicBasisQueryAttention(nn.Module):
    """
    Per-query logit-mix BQA with INDEPENDENT K and V mixings (mirrors BQA static's
    alpha_k / alpha_v structure). Each query position t produces its own K and V
    mixing distributions w_k[t, h, :], w_v[t, h, :] over the n_kv_head basis K/V
    heads via softmax(alpha_proj_{k,v}(x_t) + b_alpha_{k,v}). Effective per-query
    K_eff[t,s,h] = Σ_j w_k[t,h,j] · K_basis[s,j]; effective per-query V is given
    by O[t,h,d] = Σ_j w_v[t,h,j] · (Σ_s p[t,s,h] · V_basis[s,j,d]). A SINGLE
    softmax over s is applied to the w_k-mixed scores (not mixture-of-softmaxes).

    ve gate is per-query-head (n_head outputs, matching BQA static): gate(x_s)[h]
    modulates the VE contribution from source position s through the same w_v
    path. Implemented by baking the gate into the attention weights for the VE
    branch (p_gated[t,s,h] = p[t,s,h] · gate[s,h]) and adding the resulting VE
    aggregate to the V aggregate before the per-query w_v mix.

    Implementation: Q-side fold (algebraically identical to materializing
    S[b,h,t,s,j] but uses standard attention infra). Define
        Q_w[b,t,h,j,d] = w_k[b,t,h,j] · Q[b,t,h,d],
    then
        score[t,s,h] = Σⱼ wₖ[t,h,j]·⟨Q[t,h], K_basis[s,j]⟩
                     = Σ_(j,d) Q_w[t,h,(j,d)] · K_basis_flat[s,(j,d)]
                     = ⟨Q_w_flat[t,h,:], K_basis_flat[s,:]⟩  with head_dim_eff = J·D.
    That's a standard MHA score with widened head dim — single softmax over s, no
    (B,H,T,T,J) tensor anywhere. Per-head V is `V_basis + gate(x_s)·ve` (per source,
    per head, matching BQA static's ve gate). Per-query w_v mix is applied OUTSIDE
    the kernel as a small einsum: y[t,h,d] = Σⱼ wᵥ[t,h,j]·o[t,h,j,d].
    Memory drops from O(B·H·T²·J) to O(B·T·H·J·D); DBS can be similar to MHA.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Independent K and V mixing-logit projections (mirrors BQA static's
        # alpha_k / alpha_v). alpha_proj_{k,v}.weight init to zero so step-0
        # logits = b_alpha_{k,v} (GQA-leaning init in GPT.init_weights).
        self.alpha_proj_k = Linear(self.n_embd, self.n_head * self.n_kv_head, bias=False)
        self.alpha_proj_v = Linear(self.n_embd, self.n_head * self.n_kv_head, bias=False)
        # Static biases on K and V mixing logits — initialized to recover GQA.
        self.b_alpha_k = nn.Parameter(torch.zeros(self.n_head, self.n_kv_head))
        self.b_alpha_v = nn.Parameter(torch.zeros(self.n_head, self.n_kv_head))
        # ve gate: per-query-head (matches BQA static; n_head outputs). Each source
        # position s produces gate(x_s)[h] that modulates the VE attention contribution
        # before the w_v mix.
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("bqa_dyn does not yet support inference KV cache")
        B, T, C = x.size()
        H, J, D = self.n_head, self.n_kv_head, self.head_dim

        # Per-query mixing weights: independent K and V (matches BQA static's
        # alpha_k / alpha_v). softmax in fp32 for stability, cast to compute dtype.
        alpha_logits_k = self.alpha_proj_k(x).view(B, T, H, J) + self.b_alpha_k
        alpha_logits_v = self.alpha_proj_v(x).view(B, T, H, J) + self.b_alpha_v
        w_k_f = F.softmax(alpha_logits_k.float(), dim=-1)  # (B, T, H, J), fp32
        w_v_f = F.softmax(alpha_logits_v.float(), dim=-1)
        w_k = w_k_f.to(x.dtype)
        w_v = w_v_f.to(x.dtype)
        # Mean per-token mixing entropy (over B, T, H), stashed for wandb logging.
        # Uniform-init value = log(J); collapses to 0 as alpha specialises.
        with torch.no_grad():
            self._last_h_w_k = -(w_k_f * w_k_f.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self._last_h_w_v = -(w_v_f * w_v_f.clamp_min(1e-12).log()).sum(dim=-1).mean()

        # Standard projections.
        q = self.c_q(x).view(B, T, H, D)
        k_basis = self.c_k(x).view(B, T, J, D)
        v_basis = self.c_v(x).view(B, T, J, D)

        # Rotary on q (per query position) and k_basis (per key position). QK norm + ×1.2.
        # Note: rotary acts on the last dim (D), so it MUST run before the J-fold below.
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k_basis = apply_rotary_emb(k_basis, cos, sin)
        q = norm(q) * 1.2
        k_basis = norm(k_basis) * 1.2

        # Q-side fold: bake w_k into Q so the basis-mix becomes part of a standard
        # attention score (single softmax over s, no (B,H,T,T,J) tensor).
        # Q_eff[b, t, h, j, d] = w_k[b, t, h, j] · Q[b, t, h, d], then flatten (j,d).
        # Pre-scale by sqrt(J) so SDPA's default 1/sqrt(J·D) becomes 1/sqrt(D),
        # matching the per-basis-D scale of the explicit formulation.
        q_eff = (w_k.unsqueeze(-1) * q.unsqueeze(-2)).reshape(B, T, H, J * D) * (J ** 0.5)

        # K_eff: same K_basis for every query head (no per-head mix on K side — the
        # mix lives in Q). Replicate K_basis across H so flash_attn shim sees a
        # standard num_q_heads == num_kv_heads layout. (Could use enable_gqa with
        # num_kv_heads=1, but per-head V below requires num_kv_heads=H anyway —
        # see ve handling — so we keep K and V symmetric for one SDPA call.)
        k_eff = k_basis.unsqueeze(2).expand(B, T, H, J, D).reshape(B, T, H, J * D)

        # V_eff: per-head V combines V_basis with optional ve residual. ve gate is
        # per-source-per-head (matches BQA static), applied at the source position
        # by adding gate(x_s)[h] · ve[s, j, d] to V_basis[s, j, d].
        if ve is not None:
            ve = ve.view(B, T, J, D)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, H)
            # broadcast: V_basis (B,T,1,J,D) + gate (B,T,H,1,1) * ve (B,T,1,J,D)
            v_combined = v_basis.unsqueeze(2) + gate[..., None, None] * ve.unsqueeze(2)
            v_eff = v_combined.reshape(B, T, H, J * D)
        else:
            v_eff = v_basis.unsqueeze(2).expand(B, T, H, J, D).reshape(B, T, H, J * D)

        # Standard attention with widened head_dim = J·D. The SDPA fallback in
        # nanochat.flash_attention handles head_dim > 128 via mem_efficient/math
        # backends. Causal + sliding window come for free.
        o = flash_attn.flash_attn_func(q_eff, k_eff, v_eff, causal=True, window_size=window_size)
        o = o.reshape(B, T, H, J, D)

        # Per-query w_v mix to per-query-head output (the only "outside the kernel"
        # piece — wᵥ is t-indexed, can't be folded into V at source).
        y = torch.einsum('bthj,bthjd->bthd', w_v, o)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def make_attention(config, layer_idx):
    if config.attn_kind == "gqa":
        return CausalSelfAttention(config, layer_idx)
    if config.attn_kind == "bqa":
        return BasisQueryAttention(config, layer_idx)
    if config.attn_kind == "bqa_dyn":
        return DynamicBasisQueryAttention(config, layer_idx)
    raise ValueError(f"Unknown attn_kind: {config.attn_kind!r} (expected 'gqa', 'bqa', or 'bqa_dyn')")


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = make_attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # BQA mixing logits: GQA-leaning init parameterized by per-side target
        # softmax mass on the assigned basis. Mass m → logit L = log(m/(1-m))
        # + log(J-1) where J = n_kv_head; this makes the init distribution
        # shape independent of n_kv_head, instead of letting m collapse with
        # J as a fixed-logit init does. Skip the bonus when J == 1 (degenerate,
        # single basis already has mass 1) or m == 1/J (uniform, logit = 0).
        # bqa_dyn has independent K/V mixing logits (b_alpha_k, b_alpha_v) — use both.
        if self.config.attn_kind in ("bqa", "bqa_dyn"):
            n_head, n_kv_head = self.config.n_head, self.config.n_kv_head
            assert n_head % n_kv_head == 0
            group_size = n_head // n_kv_head

            def _logit_from_mass(m):
                assert 0.0 < m < 1.0, f"bqa_init_mass must be in (0, 1), got {m}"
                if n_kv_head <= 1:
                    return 0.0
                return math.log(m / (1.0 - m)) + math.log(n_kv_head - 1)

            logit_k = _logit_from_mass(self.config.bqa_init_mass_k)
            logit_v = _logit_from_mass(self.config.bqa_init_mass_v)

            for block in self.transformer.h:
                if self.config.attn_kind == "bqa":
                    pairs = ((block.attn.alpha_k, logit_k), (block.attn.alpha_v, logit_v))
                else:
                    torch.nn.init.zeros_(block.attn.alpha_proj_k.weight)
                    torch.nn.init.zeros_(block.attn.alpha_proj_v.weight)
                    pairs = ((block.attn.b_alpha_k, logit_k), (block.attn.b_alpha_v, logit_v))
                for a, init_logit in pairs:
                    torch.nn.init.zeros_(a)
                    if init_logit != 0.0:
                        for h in range(n_head):
                            a.data[h, h // group_size] = init_logit

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5,
                        alpha_lr_mult=1.0, alpha_beta1=0.9, alpha_wd=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # BQA's logit tensors (`alpha_k`/`alpha_v` for static BQA, `b_alpha_k`/`b_alpha_v`
        # for bqa_dyn) are *not* linear operators — they're tensors of independent logits —
        # so they don't belong in the Muon bucket (Newton-Schulz on logits scrambles them)
        # and must not get the matrix-group weight decay (which would pull softmax toward
        # uniform). Pull them out of `transformer.h` into a dedicated AdamW group with
        # no WD. Note: `alpha_proj_{k,v}.weight` (bqa_dyn) ARE real linear-operator weights
        # and stay in the matrix/Muon bucket as usual.
        def _is_alpha_logit(name):
            return (name.endswith(".alpha_k") or name.endswith(".alpha_v")
                    or name.endswith(".b_alpha_k") or name.endswith(".b_alpha_v"))
        alpha_params = [p for n, p in self.transformer.h.named_parameters() if _is_alpha_logit(n)]
        matrix_params = [p for n, p in self.transformer.h.named_parameters() if not _is_alpha_logit(n)]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(alpha_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # BQA mixing logits (static `alpha_k`/`alpha_v` and bqa_dyn's `b_alpha_k`/`b_alpha_v`): tiny tensors,
        # AdamW. Default LR matches the embedding scale; default WD=0 so the
        # GQA-recovering init isn't slowly pulled toward uniform. The
        # `alpha_lr_mult`, `alpha_beta1`, `alpha_wd` overrides exist so this
        # group can be tuned independently of the rest of AdamW.
        if alpha_params:
            param_groups.append(
                dict(kind='adamw', params=alpha_params,
                     lr=embedding_lr * dmodel_lr_scale * alpha_lr_mult,
                     betas=(alpha_beta1, 0.95),
                     eps=1e-10, weight_decay=alpha_wd)
            )
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
