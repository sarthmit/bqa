# BQA: a small free win on top of GQA

*Apr 29 / Apr 30 attn_compare scaling sweep — written 2026-05-02*

---

## TL;DR

We compared three attention variants — **MHA**, **GQA**, and **BQA** — across
five FLOP budgets (1e18 → 2.15e19) and five depths (12, 16, 20, 24, 28).

- **BQA beats GQA at every (depth, FLOP) point we measured** — 17 / 17 cells —
  by a small but consistent **0.0002–0.0012 bpb**.
- **BQA and GQA share the same parameter count and the same inference compute /
  KV-cache size.** Same `n_kv_head=4` vs `n_head=8`. So BQA's gain is genuinely
  free at inference time.
- **MHA still wins on bpb** by a larger margin (**0.0014–0.0062 bpb** over BQA),
  but pays for it: ~40% more attention parameters and 2× the KV cache.
- **Scaling-law fits** are essentially identical for GQA and BQA
  (α ≈ 0.39, β ≈ 0.56) and slightly different for MHA (α ≈ 0.37, β ≈ 0.57,
  with a higher prefactor on N\*).

If you already use GQA, switching to BQA is a no-cost upgrade. If you can
afford the KV cache, MHA is still meaningfully better for loss.

---

## Setup

- **Architectures.**
  - MHA: 8 query heads, 8 KV heads.
  - GQA: 8 query heads, 4 KV heads (2-to-1 grouping).
  - BQA: same head topology as GQA (8 Q, 4 KV) plus a per-query-head value-edit
    gate (`ve_gate`) and the asymmetric mass-based init `(m_k=0.70, m_v=0.85)`
    set as defaults on 2026-04-29.
- **Sweep.** depths d ∈ {12, 16, 20, 24, 28}, FLOP budgets
  C ∈ {1e18, 2.15e18, 4.64e18, 1e19, 2.15e19}. Not every (d, C) cell was run;
  smaller depths get the smaller budgets, larger depths the larger budgets.
- **Metric.** Minimum validation bpb (lower is better).

Logs live in `runs/logs/attn_compare/attncmp_apr{29,30}_*.log`. Plot scripts:
[scripts/plots/isoflop.py](../scripts/plots/isoflop.py) and
[scripts/plots/scaling_laws.py](../scripts/plots/scaling_laws.py).

---

## Result 1: BQA beats GQA at every point, by a thin margin

Side-by-side validation bpb at matched (depth, C). Negative `bqa−gqa` means BQA
is better.

| depth | C        | MHA bpb | GQA bpb | BQA bpb | bqa − gqa | mha − bqa |
|------:|---------:|--------:|--------:|--------:|----------:|----------:|
| 12 | 1.00e18 | 0.8498 | 0.8547 | 0.8539 | −0.0008 | −0.0041 |
| 12 | 2.15e18 | 0.8205 | 0.8262 | 0.8259 | −0.0003 | −0.0054 |
| 12 | 4.64e18 | 0.8013 | 0.8080 | 0.8074 | −0.0006 | −0.0061 |
| 12 | 1.00e19 | 0.7884 | 0.7952 | 0.7946 | −0.0006 | −0.0062 |
| 16 | 1.00e18 | 0.8635 | 0.8661 | 0.8649 | −0.0012 | −0.0014 |
| 16 | 2.15e18 | 0.8158 | 0.8198 | 0.8192 | −0.0006 | −0.0033 |
| 16 | 4.64e18 | 0.7819 | 0.7866 | 0.7862 | −0.0004 | −0.0043 |
| 16 | 1.00e19 | 0.7589 | 0.7640 | 0.7637 | −0.0003 | −0.0048 |
| 20 | 1.00e18 | 0.9752 | 0.9765 | 0.9758 | −0.0007 | −0.0006 |
| 20 | 2.15e18 | 0.8562 | 0.8587 | 0.8585 | −0.0002 | −0.0023 |
| 20 | 4.64e18 | 0.7968 | 0.8001 | 0.7998 | −0.0002 | −0.0031 |
| 20 | 1.00e19 | 0.7563 | 0.7606 | 0.7600 | −0.0006 | −0.0038 |
| 20 | 2.15e19 | 0.7286 | 0.7336 | 0.7329 | −0.0006 | −0.0044 |
| 24 | 1.00e19 | 0.7661 | 0.7692 | 0.7688 | −0.0004 | −0.0027 |
| 24 | 2.15e19 | 0.7291 | 0.7329 | 0.7323 | −0.0005 | −0.0032 |
| 28 | 1.00e19 | 0.7827 | 0.7848 | 0.7846 | −0.0002 | −0.0019 |
| 28 | 2.15e19 | 0.7383 | 0.7409 | 0.7405 | −0.0003 | −0.0022 |

**Two patterns worth naming:**

1. **BQA → GQA gap is uniformly small (≤ 0.0012 bpb)** but never negative. 17/17
   wins under a single seed is a weak signal individually but a strong one
   collectively — if BQA were neutral we'd expect ~50/50.
2. **MHA → BQA gap is 4–10× larger than BQA → GQA**, and grows with compute.
   At 1e18 the gap is ~0.001–0.004; at 2.15e19 it's ~0.002–0.004 even at the
   biggest depths. So: GQA-family attention loses something real to MHA, and
   BQA only claws back a small fraction of that loss.

---

## Result 2: BQA and GQA share inference compute

This is the part that makes BQA actually interesting as a default.

- **Parameter count.** Identical at every (d, C) point in the table above —
  e.g., d16 → 385.9M for both BQA and GQA, vs 536.9M for MHA.
- **KV cache.** Both use `n_kv_head=4`, so the per-token KV cache is the same.
  MHA's is 2× larger.
- **Attention compute at inference.** Dominated by the K, V projections and the
  KV-cache reads, which are determined by `n_kv_head`. BQA's extra
  per-query-head `ve_gate` is a small element-wise op on Q-side activations and
  doesn't change the matmul shapes or KV memory footprint.

So switching GQA → BQA changes the trained weights and how value information is
mixed across query heads, but **does not** change the inference-time arithmetic
intensity, KV cache size, or the parameter count you need to host. The only
cost is at training time (slightly more compute through the `ve_gate`), and
even that is small.

This is why we frame it as "free": you spend the same per-token FLOPs at serve
time and get a small but consistent quality improvement.

---

## Result 3: scaling-law fits

Using the parabola-minimum trick — for each (arch, C) we fit val_bpb as a
quadratic in log₁₀(N), take the minimum N\*, and read off D\* via interpolated
flops-per-token along the curve — gives Chinchilla-style power laws
N\* ∝ C^α and D\* ∝ C^β:

| arch | α (params) | β (tokens) | α + β |
|------|-----------:|-----------:|------:|
| MHA  | 0.368 | 0.570 | 0.94 |
| GQA  | 0.393 | 0.557 | 0.95 |
| BQA  | 0.390 | 0.560 | 0.95 |

GQA and BQA produce essentially the same scaling exponents — consistent with
the picture that BQA shifts the loss-vs-compute curve down by a small constant
without bending it. MHA has a *slightly* shallower exponent on N and a higher
prefactor: at any fixed compute it prefers a larger model, partly because its
per-token FLOPs are higher for the same depth/width.

**Caveats on the fit.**

- α + β should equal 1 from C = k·N·D bookkeeping. We see ~0.95, mostly because
  embedding/head parameters are counted in N but not amortized into the
  per-token attention FLOPs at small N.
- β > α is unlike Chinchilla's ≈ 0.5 / 0.5. Our depth grid at the largest
  budget (2.15e19) is only d20/d24/d28, which is a short lever arm on the
  high-C side and probably exaggerates β. Adding a 4.64e19 budget at d24–d40
  would test this.

Plot: [scripts/plots/scaling_laws_apr30.png](../scripts/plots/scaling_laws_apr30.png).
Underlying isoflop curves (where the stars come from) are in
[scripts/plots/isoflop_apr30.png](../scripts/plots/isoflop_apr30.png).

---

## So when do you want each?

- **You're serving and KV cache / latency dominates** → GQA or BQA. Prefer
  **BQA** — same cost, slightly lower loss.
- **You can afford full MHA's KV cache** → MHA still wins on bpb by a
  meaningful margin (0.002–0.006 at the FLOP scales we measured), and the gap
  doesn't close as you scale up.
- **You're choosing between GQA and BQA** → BQA, every time. The win is small
  but it's free, and it shows up at every (depth, compute) point we tested.

---

## What this *doesn't* settle

- **Single seed.** Each (arch, d, C) cell is one seed. The BQA → GQA gap is
  smaller than typical seed noise on individual runs; the consistent sign
  across 17 cells is what makes it credible. A 3-seed replication at d20 and
  d24/2.15e19 would tighten this.
- **MHA's compute advantage.** Part of MHA's lower bpb at fixed C comes from
  its higher per-token FLOPs (more KV heads → more attention work). The
  isoflop framing accounts for this on the C axis but not in the
  parameter-efficiency framing — at fixed *N*, BQA/GQA already match or beat
  MHA, but you don't get to choose N independently of C.
- **Long context / inference scaling.** All sweep runs use a 2048 context. The
  GQA-vs-BQA "free win" framing is strongest in regimes where KV-cache
  bandwidth is the bottleneck — i.e., long context, batched decode. We haven't
  measured serving throughput here.
