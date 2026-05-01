"""Clean transformer model with attention layers taken from fla.

Two attention kinds are supported:
  * "mha"  — preferred path: `fla.layers.attn.Attention` (q/k/v/o_proj, internal
             rotary, optional qk-RMSNorm, sliding-window) using the flash-attn
             kernel. Auto-falls back to a small SDPA-based port (same parameter
             layout) when the `flash_attn` package isn't importable, so the
             model stays loadable on machines where the heavy flash-attn build
             didn't complete.
  * "gdn"  — `fla.layers.gated_deltanet.GatedDeltaNet` (Gated Delta Networks,
             arXiv:2412.06464). Trained with the chunked parallel kernel.

Stripped down from `nanochat/gpt.py` — no value-residual, no smear/x0/backout,
no Muon, no per-layer sliding-window pattern, no FP8. Single AdamW group with
one shared LR/WD across the whole model. Reuses the standard warmup/warmdown
schedule from `scripts/base_train.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear  # subclass of nn.Linear: casts master weight (fp32) to input dtype on forward


def _cast_linears_inplace(module: nn.Module) -> None:
    """In-place class-swap every plain nn.Linear inside `module` to nanochat's Linear,
    so the master weight stays fp32 (good for AdamW precision on tiny updates) but the
    matmul runs in the activation dtype (bf16 throughput on A100). Only swaps exact
    nn.Linear; leaves subclasses (e.g. fla's quantized Linear variants if any) alone.
    """
    for m in module.modules():
        if type(m) is nn.Linear:
            m.__class__ = Linear


def _cast_conv1d_to_compute_dtype(module: nn.Module) -> None:
    """Cast every nn.Conv1d weight (and bias if present) inside `module` to COMPUTE_DTYPE.
    Used for fla's ShortConvolution, which passes `self.weight` straight to a triton
    kernel — it expects weight and input to share dtype, so we can't keep fp32 master
    weights with on-the-fly cast there. The conv params are tiny ((C, 1, K) per stream),
    so the precision loss vs. fp32 master is negligible.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data = m.weight.data.to(dtype=COMPUTE_DTYPE)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(dtype=COMPUTE_DTYPE)


@dataclass
class FLAConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    # Per-layer attention type:
    #   "mha"    — every layer is multi-head attention (fla.layers.attn.Attention via flash_attn,
    #              or our SDPA fallback if flash_attn isn't importable).
    #   "gdn"    — every layer is fla.layers.gated_deltanet.GatedDeltaNet (chunk for prefill,
    #              fused_recurrent for q_len ≤ 64 in eval).
    #   "hybrid" — fraction of layers given by `alpha` are MHA; the rest are GDN. MHA layers
    #              are spread evenly across depth (Bresenham, see `_is_mha_layer`). At alpha=0
    #              this degenerates to all-GDN; at alpha=1 to all-MHA.
    attn_kind: str = "mha"
    # Fraction of layers that are MHA when attn_kind == "hybrid"; the remainder are GDN.
    # The dial runs end-to-end:
    #   * alpha == 0.0 → 0 MHA, all GDN (equivalent to attn_kind="gdn")
    #   * alpha == 0.5, n_layer == 12 → 6 MHA evenly interleaved with 6 GDN (alternating)
    #   * alpha == 0.25, n_layer == 12 → 3 MHA, 9 GDN: pattern "GGGM" repeating (1 MHA per
    #                                    4 layers, matching "3 GDN for 1 MHA" alternating)
    #   * alpha == 1.0 → all MHA (equivalent to attn_kind="mha")
    # Non-divisible products are rounded to the closest integer (e.g. alpha=0.4 n_layer=12
    # → round(4.8) = 5 MHA layers, not floor=4). Placement is deterministic and depends only
    # on (round(alpha*n_layer), n_layer) — not on alpha's float value — so resuming a hybrid
    # model with a slightly different alpha (e.g. 0.50001 vs 0.5) lands at the same schedule.
    alpha: float = 0.5
    rope_theta: float = 100_000.0
    qk_norm: bool = True
    # GDN-specific (ignored when no layer is GDN). Defaults match fla.
    gdn_expand_v: float = 2.0
    gdn_use_short_conv: bool = True
    gdn_conv_size: int = 4
    gdn_use_gate: bool = True
    gdn_allow_neg_eigval: bool = False
    # Mixture-of-Experts FFN. When `moe_num_experts > 0` every block's MLP becomes a
    # token-routed top-k MoE (replacing the dense ReLU² FFN). Routing pattern mirrors
    # google-deepmind/simply MoEFeedForward (model_lib.py:642):
    #   * top_k == 1 → softmax → top-1 (avoids zero gradient at the argmax)
    #   * top_k > 1  → top-k of logits → softmax over the selected k (normalized)
    # Each expert is its own ReLU² FFN with the same up/down shape as the dense MLP.
    # A Switch-Transformer-style load-balancing auxiliary loss is added to the main
    # loss with weight `moe_lbl_loss_weight` (computed only in training mode).
    # `moe_num_experts == 0` (default) disables MoE and uses the dense MLP.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_lbl_loss_weight: float = 0.01


def _rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary_emb(x, cos, sin):
    # x: (B, T, H, D); cos, sin: (1, T, 1, D//2). Same scheme as nanochat/gpt.py.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class _SDPAAttention(nn.Module):
    """Vanilla causal MHA with rotary + optional QK-RMSNorm. Mirrors fla.Attention's
    parameter layout (so swapping in fla.Attention later is a one-import change),
    but uses F.scaled_dot_product_attention as the kernel."""

    def __init__(self, config: FLAConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qk_norm = config.qk_norm
        self.q_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin, past_key_values=None, use_cache=False):
        """Causal MHA via SDPA. When `past_key_values` is provided, this layer's k/v
        from prior calls is concatenated along the time dim — identical pattern to fla
        but using fla's Cache as the carrier. The rotary table the caller passes in
        already has the right offset applied (see FLATransformer.forward), so we don't
        re-offset here.
        """
        B, T, _ = x.size()
        H, D = self.n_head, self.head_dim
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)
        if self.qk_norm:
            q, k = _rms_norm(q), _rms_norm(k)
        cos, sin = cos_sin
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # KV-cache update — concatenate this step's k/v with the cached history along T.
        # Cache stores (k, v) flattened to (B, T_seen, H*D) per fla.Cache convention.
        if past_key_values is not None:
            new = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=T,
            )["attn_state"]
            k = new[0].view(B, -1, H, D)
            v = new[1].view(B, -1, H, D)

        # SDPA expects (B, H, T, D). When the cache contains prior context, the score
        # matrix is non-square (T_q < T_kv) so is_causal=True would mask the wrong
        # corner — pass a manual mask in that case.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        Tq, Tkv = q.size(2), k.size(2)
        if Tq == Tkv:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Causal mask aligned to the right edge of K/V (decode): position s of K
            # corresponds to absolute position (Tkv - Tq + t_q) for query t_q. Allow
            # K positions ≤ that absolute position. Since we feed only this step's q,
            # Tq is small (usually 1) and the mask cost is negligible.
            row = torch.arange(Tq, device=q.device).view(-1, 1) + (Tkv - Tq)
            col = torch.arange(Tkv, device=q.device).view(1, -1)
            attn_mask = (col <= row)  # (Tq, Tkv) bool
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(o)


class _GDNAttention(nn.Module):
    """Wraps fla.layers.gated_deltanet.GatedDeltaNet so the Block can call
    `attn(x, cos_sin)` uniformly. cos_sin is unused (GDN's recurrence + short
    conv handle ordering and locality)."""

    def __init__(self, config: FLAConfig, layer_idx: int):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_gate = config.gdn_use_gate
        self.use_short_conv = config.gdn_use_short_conv
        self.inner = GatedDeltaNet(
            hidden_size=config.n_embd,
            expand_v=config.gdn_expand_v,
            head_dim=self.head_dim,
            num_heads=config.n_head,
            num_v_heads=config.n_head,  # n_kv_head == n_head per branch constraint
            mode="chunk",
            use_gate=config.gdn_use_gate,
            use_short_conv=config.gdn_use_short_conv,
            conv_size=config.gdn_conv_size,
            allow_neg_eigval=config.gdn_allow_neg_eigval,
            layer_idx=layer_idx,
        )
        # Class-swap fla's stock nn.Linear projections to nanochat's Linear (fp32 master,
        # on-the-fly bf16 cast). Applies to q/k/v/g/a/b/o_proj inside GatedDeltaNet.
        _cast_linears_inplace(self.inner)

    def forward(self, x, cos_sin, past_key_values=None, use_cache=False):
        # fla.GatedDeltaNet uses fla.Cache for both `recurrent_state` and per-stream
        # `conv_state`. When q_len <= 64 in eval, fla auto-routes to the
        # fused_recurrent_gated_delta_rule kernel for per-token decode.
        o, _, _ = self.inner(
            hidden_states=x,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return o

    @torch.no_grad()
    def reset_parameters(self, init_std: float):
        # All linear projections and conv1d weights get a uniform init with std
        # set to match the transformer-block init scale; o_proj zeroed for a
        # neutral residual at step 0; A_log/dt_bias re-initialized using fla's
        # __init__ recipe (uniform A in (0,16), log-uniform dt then inv-softplus);
        # o_norm gain set to 1.
        s = init_std * (3 ** 0.5)
        m = self.inner
        for lin in (m.q_proj, m.k_proj, m.v_proj, m.a_proj, m.b_proj):
            nn.init.uniform_(lin.weight, -s, s)
        if self.use_gate:
            nn.init.uniform_(m.g_proj.weight, -s, s)
        nn.init.zeros_(m.o_proj.weight)
        if self.use_short_conv:
            for conv in (m.q_conv1d, m.k_conv1d, m.v_conv1d):
                bound = 1.0 / conv.weight.size(-1) ** 0.5
                nn.init.uniform_(conv.weight, -bound, bound)
                if getattr(conv, "bias", None) is not None:
                    nn.init.zeros_(conv.bias)
        nn.init.ones_(m.o_norm.weight)
        n_v = m.num_v_heads
        device = m.A_log.device
        A = torch.empty(n_v, dtype=torch.float32, device=device).uniform_(0, 16)
        m.A_log.data.copy_(torch.log(A))
        m.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(n_v, device=device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        m.dt_bias.data.copy_(inv_dt)
        m.dt_bias._no_weight_decay = True


class _FlashAttention(nn.Module):
    """Vanilla causal MHA via `fla.layers.attn.Attention` (which uses the
    flash-attn package). Adapter that hides fla's HF-style forward signature
    behind our `(x, cos_sin)` interface — fla.Attention applies rotary
    internally, so the externally-precomputed cos_sin is ignored."""

    def __init__(self, config: FLAConfig, layer_idx: int):
        super().__init__()
        from fla.layers.attn import Attention
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.qk_norm = config.qk_norm
        self.inner = Attention(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            num_kv_heads=config.n_head,  # n_kv_head == n_head per branch constraint
            qk_norm=config.qk_norm,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.sequence_len,
            layer_idx=layer_idx,
        )
        # Class-swap fla's stock nn.Linear projections to nanochat's Linear so master
        # weights stay fp32 (good for AdamW precision) but matmul runs in bf16. The fla
        # Attention rotary buffer (inv_freq) is not a Parameter and is unaffected.
        _cast_linears_inplace(self.inner)

    def forward(self, x, cos_sin, past_key_values=None, use_cache=False):
        # fla.Attention natively handles past_key_values (a fla.Cache instance) — it
        # applies rotary internally with the correct offset derived from the cache,
        # updates the attn_state in place, and runs flash_attn against the cached k/v.
        o, _, _ = self.inner(
            hidden_states=x,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return o

    @torch.no_grad()
    def reset_parameters(self, init_std: float):
        s = init_std * (3 ** 0.5)
        m = self.inner
        nn.init.uniform_(m.q_proj.weight, -s, s)
        nn.init.uniform_(m.k_proj.weight, -s, s)
        nn.init.uniform_(m.v_proj.weight, -s, s)
        nn.init.zeros_(m.o_proj.weight)
        if self.qk_norm:
            nn.init.ones_(m.q_norm.weight)
            nn.init.ones_(m.k_norm.weight)


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _is_mha_layer(layer_idx: int, n_layer: int, alpha: float) -> bool:
    """Should layer `layer_idx` (0-indexed) be MHA under the hybrid schedule?

    `alpha` is the MHA fraction: alpha=0 is all-GDN, alpha=1 is all-MHA. Places exactly
    `n_mha = round(alpha * n_layer)` MHA layers evenly across depth using a Bresenham
    accumulator parameterised by n_mha (not alpha directly):
        is_mha(i) := (i+1) * n_mha // n_layer > i * n_mha // n_layer
    Integer-only — guarantees exactly `round(alpha*n_layer)` MHA layers, deterministic,
    identical at training and inference. Examples (n_layer=12):
      * alpha == 0.00 → 0 MHA: GGGGGGGGGGGG
      * alpha == 0.25 → 3 MHA at [3, 7, 11]: GGGMGGGMGGGM   (1 MHA per 4 layers)
      * alpha == 0.40 → 5 MHA at [2, 4, 7, 9, 11]: GGMGMGGMGMGM
      * alpha == 0.50 → 6 MHA at [1, 3, 5, 7, 9, 11]: GMGMGMGMGMGM   (alternating)
      * alpha == 0.75 → 9 MHA, 3 GDN at [0, 4, 8]: GMMMGMMMGMMM
      * alpha == 1.00 → all MHA: MMMMMMMMMMMM
    """
    if alpha <= 0.0:
        return False
    if alpha >= 1.0:
        return True
    n_mha = round(alpha * n_layer)
    if n_mha == 0:
        return False
    if n_mha >= n_layer:
        return True
    return ((layer_idx + 1) * n_mha) // n_layer > (layer_idx * n_mha) // n_layer


def _make_mha_attn(config: FLAConfig, layer_idx: int) -> nn.Module:
    """Pick the best available MHA implementation: prefer fla's flash_attn-backed
    Attention, fall back to our SDPA port if the flash_attn package isn't installed."""
    if _flash_attn_available():
        return _FlashAttention(config, layer_idx)
    if layer_idx == 0:
        print0(
            "[fla] flash_attn package not importable; falling back to "
            "F.scaled_dot_product_attention. Install flash-attn for the fla.Attention path."
        )
    return _SDPAAttention(config, layer_idx)


def _make_attn(config: FLAConfig, layer_idx: int) -> nn.Module:
    if config.attn_kind == "mha":
        return _make_mha_attn(config, layer_idx)
    if config.attn_kind == "gdn":
        return _GDNAttention(config, layer_idx)
    if config.attn_kind == "hybrid":
        if not 0.0 <= config.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {config.alpha}")
        if _is_mha_layer(layer_idx, config.n_layer, config.alpha):
            return _make_mha_attn(config, layer_idx)
        return _GDNAttention(config, layer_idx)
    raise ValueError(f"Unknown attn_kind for FLATransformer: {config.attn_kind!r} (expected 'mha', 'gdn', or 'hybrid')")


class _MLP(nn.Module):
    """Pre-norm SwiGLU FFN — the dominant FFN pattern in modern LLMs (LLaMA, Mistral,
    Qwen, DeepSeek, Mixtral, Phi-3, ...) since Shazeer 2020:
        down_proj(silu(gate_proj(x)) * up_proj(x))

    Hidden dim = 4·n_embd (matches the prior ReLU² FFN size — three matmuls instead
    of two trade extra parameters for the empirically-better gated activation; we
    keep this size for direct comparability rather than dropping to LLaMA's
    parameter-equivalent 8/3·n_embd).

    Same module also serves as a single MoE expert — see _MoEMLP.
    """

    def __init__(self, config: FLAConfig):
        super().__init__()
        n_embd = config.n_embd
        hidden = 4 * n_embd
        self.gate_proj = Linear(n_embd, hidden, bias=False)
        self.up_proj = Linear(n_embd, hidden, bias=False)
        self.down_proj = Linear(hidden, n_embd, bias=False)
        self.hidden = hidden

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @torch.no_grad()
    def reset_parameters(self, s: float):
        """Init: gate and up get the modest `s * 0.4` uniform (activation-explosion
        control, matches nanochat/gpt.py's c_fc init scale); down is zero-init for
        residual stability so each block starts at identity in the residual stream.
        """
        nn.init.uniform_(self.gate_proj.weight, -s * 0.4, s * 0.4)
        nn.init.uniform_(self.up_proj.weight, -s * 0.4, s * 0.4)
        nn.init.zeros_(self.down_proj.weight)


class _MoEMLP(nn.Module):
    """Mixture-of-Experts FFN with token-specific top-k routing.

    Pattern adapted from google-deepmind/simply MoEFeedForward
    (model_lib.py:642): linear router → top_k → softmax over the selected k →
    per-expert dispatch. Each expert is its own ReLU² up/down with the same
    (n_embd, 4·n_embd) shape as the dense MLP — so a per-block MoE layer holds
    `num_experts` × the FFN params, but each token only consumes top_k of them.

    Routing modes (matching the simply convention exactly):
      * top_k == 1 — softmax → top-1. Doing the topk after softmax avoids the
                     zero-gradient pitfall at the argmax (the chosen logit's
                     probability is a function of every other logit too).
      * top_k > 1  — top_k → softmax. The selected logits are renormalized so
                     the per-token expert weights sum to 1 over the picked
                     experts; their ratios still depend on each other through
                     the localized softmax, so all selected logits get gradient.

    Dispatch: per-expert loop with `index_select` / `index_add_` — does exactly
    O(N · top_k) Linear work in total (not O(N · num_experts)). Each expert e
    pulls the tokens routed to it, runs its FFN, and writes weighted outputs
    back into the residual buffer. No expert-capacity dropping (all tokens
    are processed; suitable for the small num_experts setting).

    Auxiliary load-balancing loss (Switch Transformer style;
    https://arxiv.org/abs/2101.03961 §3): `lbl = num_experts · Σ_e f_e · P_e`
    where `f_e = fraction of routings going to expert e` (no grad) and
    `P_e = mean router probability for expert e` (grad flows through softmax,
    so the router is pushed toward uniform when high-prob experts are also the
    high-frequency ones). Stashed in `self._last_aux_loss` and consumed by
    `FLATransformer.forward`.
    """

    def __init__(self, config: FLAConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"moe_top_k must be in [1, moe_num_experts]; got top_k={self.top_k}, "
                f"num_experts={self.num_experts}"
            )
        self.lbl_loss_weight = config.moe_lbl_loss_weight
        # Router: matmul on the residual to produce per-expert logits. nanochat's
        # Linear keeps the master weight in fp32 (good for the explicit float() cast
        # of the logits below).
        self.router = Linear(config.n_embd, self.num_experts, bias=False)
        # Each expert is its own _MLP — same kind (swiglu/relu2) as the dense MLP would
        # be. Storing as a ModuleList of full _MLP instances (instead of separate per-
        # weight ModuleLists) keeps the dispatch loop kind-agnostic: `y = experts[e](x_e)`
        # works for both SwiGLU and ReLU². Could be folded into stacked (E, D, H)
        # weight tensors with grouped GEMM for higher arithmetic intensity, but at small
        # num_experts the per-expert loop is simpler and already does the ideal
        # O(N · top_k) FFN work.
        self.experts = nn.ModuleList([_MLP(config) for _ in range(self.num_experts)])
        # Communication slot for FLATransformer.forward to harvest the aux loss
        # after each forward call. Reset at the start of every forward; left as
        # None when not training or when lbl_loss_weight == 0.
        self._last_aux_loss = None

    @torch.no_grad()
    def reset_parameters(self, s: float):
        """Init the router with a small uniform (zero would lock all tokens to expert 0
        at step 0; small uniform breaks symmetry without overpowering the softmax).
        Each expert's reset_parameters does the up-uniform + down-zero pattern."""
        nn.init.uniform_(self.router.weight, -s * 0.1, s * 0.1)
        for expert in self.experts:
            expert.reset_parameters(s)

    def forward(self, x):
        B, T, D = x.size()
        N = B * T
        x_flat = x.reshape(N, D)
        # Router logits in fp32 — matches simply (`weight_dtype='float32'`) and
        # keeps the softmax + topk numerically stable independent of activation dtype.
        router_logits = self.router(x_flat).float()  # (N, E)

        if self.top_k == 1:
            # softmax → topk to keep gradient on the chosen logit non-zero.
            router_probs = F.softmax(router_logits, dim=-1)
            top_w, top_idx = torch.topk(router_probs, k=1, dim=-1)
        else:
            # topk → softmax: per-token weights normalized over the picked k.
            top_logits, top_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
            top_w = F.softmax(top_logits, dim=-1)
            router_probs = F.softmax(router_logits, dim=-1)
        # Cast routing weights back to activation dtype for the residual add.
        top_w = top_w.to(x.dtype)

        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # `mask`: which (token, k-slot) pairs picked expert e. Sum across k
            # collapses to a per-token weight (a token would be 0 here unless it
            # selected expert e in at least one of its k slots).
            mask = (top_idx == e)
            if not mask.any():
                continue
            weight_e = (top_w * mask).sum(dim=-1)  # (N,)
            tok_idx = weight_e.nonzero(as_tuple=True)[0]
            if tok_idx.numel() == 0:
                continue
            x_e = x_flat.index_select(0, tok_idx)
            y = self.experts[e](x_e)  # SwiGLU or ReLU² per the expert's own kind
            out.index_add_(0, tok_idx, y * weight_e[tok_idx, None])

        # Auxiliary load-balancing loss. Only meaningful when training; skip in eval
        # to save the full softmax + reductions when generate() is hot.
        if self.training and self.lbl_loss_weight > 0.0:
            # selection_freq[e]: fraction of (token, k-slot) routings landing on e.
            # Detached so gradient flows only through the router_probs branch.
            with torch.no_grad():
                selection_onehot = F.one_hot(top_idx, self.num_experts).float()
                selection_freq = selection_onehot.sum(dim=(0, 1)) / (N * self.top_k)
            mean_prob = router_probs.mean(dim=0)  # (E,) — gradient flows through here
            lbl_loss = self.num_experts * (selection_freq * mean_prob).sum()
            self._last_aux_loss = lbl_loss * self.lbl_loss_weight
        else:
            self._last_aux_loss = None

        return out.view(B, T, D)


def _make_mlp(config: FLAConfig) -> nn.Module:
    """Pick MLP vs MoE based on `moe_num_experts`. Token-routed top-k MoE replaces
    the dense ReLU² FFN entirely (simply's pattern); the residual stream sees the
    same `(B, T, n_embd)` output either way."""
    if config.moe_num_experts > 0:
        return _MoEMLP(config)
    return _MLP(config)


class _Block(nn.Module):
    """Pre-norm transformer block: residual = x + attn(norm(x), cos_sin); residual = x + mlp(norm(x))."""

    def __init__(self, config: FLAConfig, layer_idx: int):
        super().__init__()
        self.attn = _make_attn(config, layer_idx)
        self.mlp = _make_mlp(config)

    def forward(self, x, cos_sin, past_key_values=None, use_cache=False):
        x = x + self.attn(_rms_norm(x), cos_sin, past_key_values=past_key_values, use_cache=use_cache)
        x = x + self.mlp(_rms_norm(x))
        return x


class FLATransformer(nn.Module):
    """Minimal transformer with attn_kind ∈ {mha, gdn, hybrid}. Compatible with
    nanochat's base_train.py loop: same `forward(idx, targets=None)` contract
    returning a scalar loss (or logits if targets is None), same scheduler hooks.

    Hybrid mode (`attn_kind="hybrid"`): `alpha ∈ [0, 1]` sets the fraction of layers
    that are MHA (the remainder are GDN). alpha=0 → all GDN; alpha=1 → all MHA;
    alpha=0.5 → alternating; alpha=0.25 → 1 MHA every 4 layers; etc. MHA positions are
    picked by `_is_mha_layer` (Bresenham accumulator over depth), deterministic and
    proportional. GDN and MHA layers share the same `_Block` shell, the same
    RMSNorm/MLP/embedding setup, and the same fla.Cache object during inference
    (cache.attn_state for MHA layers, cache.recurrent_state + conv_state for GDN
    layers — addressed per-layer via layer_idx).

    NOTE — meta-device init pattern: __init__ runs under `torch.device('meta')`
    in base_train.py. All parameter data is therefore meta and must be filled
    in by `init_weights()` after `to_empty(device=...)`. Same convention as
    `nanochat.gpt.GPT`.
    """

    def mha_layer_indices(self):
        """List of layer indices that are MHA under the current config. The natural
        knob now that `alpha` is the MHA fraction. Useful for logging."""
        cfg = self.config
        if cfg.attn_kind == "mha":
            return list(range(cfg.n_layer))
        if cfg.attn_kind == "gdn":
            return []
        # hybrid
        return [i for i in range(cfg.n_layer) if _is_mha_layer(i, cfg.n_layer, cfg.alpha)]

    def gdn_layer_indices(self):
        """Complement of mha_layer_indices."""
        mha = set(self.mha_layer_indices())
        return [i for i in range(self.config.n_layer) if i not in mha]

    def get_device(self):
        """Return the device of the model (used by evaluate_bpb and the inference engine).
        Pulled from wte since it's the canonical leaf parameter on every model."""
        return self.wte.weight.device

    def __init__(self, config: FLAConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"[fla] padding vocab_size {config.vocab_size} -> {padded_vocab_size} for tensor-core alignment")
        self.padded_vocab_size = padded_vocab_size

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList([_Block(config, i) for i in range(config.n_layer)])
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Precomputed rotary table — shared across MHA layers; GDN ignores it.
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10  # over-compute, asserts on grow
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base, device=None):
        if device is None:
            device = self.wte.weight.device
        ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (ch / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = freqs.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        cfg = self.config
        # Token embeddings: normal, std=0.02 (GPT-style).
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        # lm_head: small std for a near-uniform initial logit distribution
        # (same trick as nanochat/gpt.py, helps the early loss settle).
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Per-block init: linear weights ~ Uniform(-s, s) where s = sqrt(3)/sqrt(n_embd)
        # → std = 1/sqrt(n_embd). Residual projections (o_proj, mlp.c_proj) → 0 so the
        # initial residual contribution is exactly zero.
        n_embd = cfg.n_embd
        std = 1.0 / n_embd ** 0.5
        s = std * (3 ** 0.5)
        for block in self.h:
            attn = block.attn
            if isinstance(attn, _SDPAAttention):
                nn.init.uniform_(attn.q_proj.weight, -s, s)
                nn.init.uniform_(attn.k_proj.weight, -s, s)
                nn.init.uniform_(attn.v_proj.weight, -s, s)
                nn.init.zeros_(attn.o_proj.weight)
            elif isinstance(attn, (_FlashAttention, _GDNAttention)):
                attn.reset_parameters(init_std=std)
            else:
                raise AssertionError(f"unknown attn type {type(attn).__name__}")
            # MLP / MoE init delegated to each module's reset_parameters. _MLP handles
            # both swiglu (gate/up uniform, down zeros) and relu2 (c_fc uniform, c_proj
            # zeros); _MoEMLP additionally inits a small uniform router and recurses
            # into each expert's reset_parameters.
            block.mlp.reset_parameters(s)

        # Refresh the rotary buffers on the real device after to_empty().
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=cfg.rope_theta)
        self.cos, self.sin = cos, sin

        # Cast embeddings to compute dtype (memory savings, matches gpt.py). All Linear
        # weights stay in fp32 master — nanochat's `Linear` casts on the fly. The GDN
        # path uses fla's ShortConvolution, which passes its Conv1d weight straight to
        # a triton kernel that expects matching dtype with the (bf16) activations, so
        # we cast just those Conv1d params to COMPUTE_DTYPE per-block.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)
            for block in self.h:
                if isinstance(block.attn, _GDNAttention):
                    _cast_conv1d_to_compute_dtype(block.attn.inner)

    def setup_optimizer(self, lr: float = 3e-4, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        """Single AdamW group covering every parameter (one shared LR + WD).

        Uses torch.optim.AdamW(fused=True) directly rather than MuonAdamW. nanochat's
        MuonAdamW wraps the AdamW step with @torch.compile(dynamic=False) and recompiles
        per-shape; with the variety of GDN param shapes (q/k/v/g/a/b/o_proj of differing
        sizes, conv1d filters, A_log, dt_bias, o_norm) this triggered a >2.5x slowdown
        compared to plain fused AdamW (verified on cn-g020: GDN d=12 hidden=768 B=4
        T=8192 went from 437ms/step → 239ms/step). We don't need Muon here anyway since
        none of the FLATransformer parameters are slated for Newton-Schulz orthogonalization.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr, betas=betas, eps=1e-10, weight_decay=weight_decay,
            fused=True,
        )
        # base_train.py's lr scheduler reads `initial_lr` per group to compute
        # `group["lr"] = group["initial_lr"] * lrm` at every step. Stamp it now.
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
        return optimizer

    def estimate_flops(self):
        """Coarse FLOPs/token estimate for MFU logging and `--target-flops` budgeting.
        Counts the matmul FLOPs that actually run *per token*:
          * Attention/projection params: every block contributes its attn params.
          * Dense MLP: every param contributes (the 6N rule).
          * MoE MLP: only `top_k` of `num_experts` experts run per token, plus the
            (tiny) router. Without this correction `--target-flops` would over-budget
            MoE by ~num_experts/top_k (we'd train for ~4× too few steps at 8/2).
          * Attention quadratic: 12·H·D·T per MHA layer (FA-style — Q@K + soft@V);
            GDN's recurrence is O(T) per layer and absorbed into the 6N count.
        Embeddings and the lm_head are excluded — same convention as nanochat/gpt.py.
        """
        cfg = self.config
        h = cfg.n_head
        d = cfg.n_embd // h
        t = cfg.sequence_len

        nparams_active = 0
        for block in self.h:
            for p in block.attn.parameters():
                nparams_active += p.numel()
            if isinstance(block.mlp, _MoEMLP):
                # router runs once per token regardless of top_k
                nparams_active += block.mlp.router.weight.numel()
                # only top_k experts run per token; experts are homogeneous so we
                # count the param count of expert 0 and scale.
                nparams_active += block.mlp.top_k * sum(p.numel() for p in block.mlp.experts[0].parameters())
            else:
                for p in block.mlp.parameters():
                    nparams_active += p.numel()

        n_mha = len(self.mha_layer_indices())
        attn_flops = n_mha * 12 * h * d * t
        return 6 * nparams_active + attn_flops

    def num_scaling_params(self):
        """Param-count breakdown for scaling-law analysis. Mirrors the keys returned by
        nanochat.gpt.GPT.num_scaling_params() so the rest of base_train.py works."""
        wte = self.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        transformer_matrices = sum(p.numel() for p in self.h.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": 0,
            "total": total,
        }

    @staticmethod
    def make_cache():
        """Empty fla.Cache to feed `past_key_values` for incremental decoding. Same
        object handles both MHA (`attn_state`) and GDN (`recurrent_state` + `conv_state`),
        so this is the unified cache type for both attn_kinds."""
        from fla.models.utils import Cache
        return Cache()

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,           # legacy alias for past_key_values
        past_key_values=None,
        use_cache: bool = False,
        loss_reduction: str = "mean",
    ):
        # Accept either legacy `kv_cache` or fla-style `past_key_values`. They mean the
        # same thing here — a fla.Cache instance carrying per-layer state across calls.
        if past_key_values is None and kv_cache is not None:
            past_key_values = kv_cache
        B, T = idx.size()
        # Rotary slice: when decoding step-by-step, the cache holds the prior sequence
        # length and cos/sin must be advanced past it so the new tokens get the correct
        # rotary phase. Only used by _SDPAAttention; _FlashAttention/_GDNAttention apply
        # rotary internally with their own offset accounting.
        offset = 0
        if past_key_values is not None and len(past_key_values) > 0:
            offset = past_key_values.get_seq_length()
        end = offset + T
        assert end <= self.cos.size(1), (
            f"sequence length {end} grew beyond the rotary cache ({self.cos.size(1)}); "
            f"increase rotary_seq_len in __init__"
        )
        cos_sin = self.cos[:, offset:end], self.sin[:, offset:end]

        x = self.wte(idx)
        for block in self.h:
            x = block(x, cos_sin, past_key_values=past_key_values, use_cache=use_cache)
        x = _rms_norm(x)
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        # Pull in any auxiliary losses stashed by MoE blocks during this forward.
        # Each `_MoEMLP` writes its load-balancing loss to `_last_aux_loss` (None when
        # not training or when lbl_loss_weight == 0). We sum across blocks and add to
        # the main CE loss so a single backward pass handles both. Stashed back on
        # the model (`self._last_aux_loss`) for logging.
        aux = self._collect_moe_aux_loss()
        self._last_aux_loss = aux  # python float on host or None — convenient to log
        if aux is not None:
            loss = loss + aux
        return loss

    def _collect_moe_aux_loss(self):
        total = None
        for block in self.h:
            mlp = block.mlp
            if isinstance(mlp, _MoEMLP) and mlp._last_aux_loss is not None:
                total = mlp._last_aux_loss if total is None else total + mlp._last_aux_loss
        return total

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """Greedy/temperature/top-k autoregressive sampling with caching.

        Two phases:
          1. Prefill — run the full prompt through `forward(... use_cache=True)`. The
             cache fills with: per-MHA-layer (k, v) up to len(prompt), per-GDN-layer
             (recurrent_state, conv_state). For GDN with q_len > 64 fla automatically
             routes to the chunked parallel kernel for prefill.
          2. Decode — feed one token at a time. fla.GatedDeltaNet auto-switches to
             the fused recurrent kernel (q_len ≤ 64 in eval mode), giving O(d) per
             step independent of prompt length. _SDPAAttention concatenates this
             step's k/v with the cached history and runs SDPA against the full window.

        idx: (B, T_prompt) int64. Returns (B, T_prompt + max_new_tokens) int64.
        """
        was_training = self.training
        self.eval()
        try:
            cache = self.make_cache()
            # Prefill — single forward over the whole prompt.
            logits = self.forward(idx, past_key_values=cache, use_cache=True)
            # logits is (B, T_prompt, vocab) when targets=None; take the last position.
            next_tok = self._sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
            generated = [next_tok]
            done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
            if eos_token_id is not None:
                done = done | (next_tok.squeeze(-1) == eos_token_id)
            for _ in range(max_new_tokens - 1):
                if done.all():
                    break
                logits = self.forward(next_tok, past_key_values=cache, use_cache=True)
                next_tok = self._sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
                if eos_token_id is not None:
                    done = done | (next_tok.squeeze(-1) == eos_token_id)
                generated.append(next_tok)
            out = torch.cat([idx] + generated, dim=1)
        finally:
            if was_training:
                self.train()
        return out

    @staticmethod
    def _sample(logits, temperature: float, top_k: int | None):
        """Sample one token per row from `logits` of shape (B, V). Returns (B, 1) int64."""
        if temperature <= 0.0:
            # Deterministic greedy.
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = torch.where(logits < v[..., [-1]], torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
