# Kernel optimization loop — progress log

Goal: verify and maximize perf of the bqa_dyn triton kernel at J ∈ {3, 6, 8, 10, 14} on H100. 5 iterations.

## Iteration 1 (in progress, started at this turn)

**Hypothesis:** baseline measurement after the BLOCK_M=128 BLOCK_N=16 + qk-cache-revert optimization landed earlier in this session.

**Action:**
- Created `tests/bench_bqa_dyn_loop.py` — benchmarks fwd/bwd at J ∈ {3,6,8,10,14}, B=2, T=1024, D=128, bf16, vs explicit reference.
- Submitted `bench_bqa_dyn_loop.sbatch` as **jid 38830194** (1 H100, 25 min walltime).
- Output JSON: `tests/bench_bqa_dyn_loop_iter1_baseline.json`.

**Measure:** fwd ms, fwd+bwd ms, fwd cosine vs explicit reference.

**Pending:** wait for job result, parse, decide next-iteration optimization.

## Iteration 2 (planned)

**Hypothesis options to explore:**
- Restore `num_stages=2` at J=4 — pipelining might pay off if SMEM allows with the new BLOCK_M=128 BLOCK_N=16 layout.
- Try BLOCK_M=64 BLOCK_N=32 at J=4 instead of 128×16, see which is faster.
- Tune `num_warps` (currently 4) — bigger warps may help at large reductions.

## Iteration 3-5 (TBD based on iter 1-2 results)

## Iteration 1 RESULT (baseline, jid 38830194, completed)

| J  | fwd ms | fwd+bwd ms | bwd-only | fwd cos    |
|----|--------|------------|----------|------------|
| 3  | 0.095  | 0.537      | 0.44     | 0.999991   |
| 6  | 0.816  | 2.935      | 2.12     | 0.999993   |
| 8  | 1.156  | 4.985      | 3.83     | 0.999993   |
| 10 | 1.946  | 9.096      | 7.15     | 0.999994   |
| 14 | 3.500  | 25.621     | 22.1     | 0.999994   |

All correct. Backward dominates at large J (bwd/fwd = 6.3× at J=14 vs typical ~3×). Forward already healthy.

## Iteration 2 (in progress, jid 38832483)

**Hypothesis:** small BLOCK_M=128 BLOCK_N=16 tile underutilizes warps (4 warps × 32 lanes = 128 threads = 1/M-row); 8 warps gives more in-flight memory ops without SMEM cost.

**Change:** `num_warps = 8 if (bf16 and J >= 4) else 4` in both `bqa_dyn_attn_triton_fwd` and `bqa_dyn_attn_triton_bwd` wrappers. Default arg now `None`.

**Output:** `tests/bench_bqa_dyn_loop_iter2_warps8.json`

## Iteration 2 RESULT (num_warps=8, jid 38832483, completed)

| J  | iter1 fwd | iter2 fwd | speedup | iter1 fwd+bwd | iter2 fwd+bwd | speedup |
|----|-----------|-----------|---------|---------------|---------------|---------|
| 3  | 0.095     | 0.094     | 1.00×   | 0.537         | 0.501         | 1.07×   |
| 6  | 0.816     | **0.297** | 2.75×   | 2.935         | 1.612         | 1.82×   |
| 8  | 1.156     | **0.372** | 3.11×   | 4.985         | 2.611         | 1.91×   |
| 10 | 1.946     | **0.621** | 3.13×   | 9.096         | 4.102         | 2.22×   |
| 14 | 3.500     | **1.197** | 2.92×   | 25.621        | 10.781        | 2.38×   |

KEPT. 3× forward, ~2× fwd+bwd. Backward still bwd/fwd ≈ 7-9× at large J — warps helped fwd more than bwd.

## Iteration 3 (in progress, jid 38833708)

**Hypothesis:** push num_warps further at J ≥ 8 to help the backward dKV kernel's small-tile latency hiding (many active slot accumulators, BLOCK_M=32).

**Change:** num_warps tiered: 4 (J<4 / fp32), 8 (4≤J<8 bf16), 16 (J≥8 bf16). Applies to both fwd and bwd wrappers.

**Output:** `tests/bench_bqa_dyn_loop_iter3_warps16_J8.json`

## Iteration 3 RESULT (num_warps=16 at J≥8, jid 38833708)

| J  | iter2 fwd | iter3 fwd | change |
|----|-----------|-----------|--------|
| 8  | 0.372     | 2.124     | **5.7× SLOWER** |
| 10 | 0.621     | 3.714     | **5.98× SLOWER** |
| 14 | 1.197     | 7.683     | **6.42× SLOWER** |
| 14 fwd+bwd | 10.781 | 142.045 | **13× SLOWER** |

**REVERTED.** Too many warps → register spilling. Sweet spot is num_warps=8.

## Iteration 4 (in progress, jid 38835611)

**Hypothesis:** restore `num_stages=2` in forward at J ≤ 6 — pipelining K/V loads while still fitting SMEM (J=6 BLOCK_M=128 BLOCK_N=16 stages=2 = ~214 KB, fits H100). Helps d=12 full (J=6) and d=16 half (J=4) which are the most common large-J cells.

**Change:** fwd `num_stages = 2 if (bf16 and J ≤ 6) else 1`. At J=7+ stages=2 busts SMEM, stays at 1.

**Output:** `tests/bench_bqa_dyn_loop_iter4_stages2_J6.json`

## Iteration 4 RESULT (num_stages=2 in fwd at J≤6, jid 38835611)

| J | iter2 fwd | iter4 fwd | change |
|---|-----------|-----------|--------|
| 6 | 0.297     | **0.194** | 1.53× faster |
| 3/8/10/14 | unchanged (J>6 untouched) |

| J | iter2 fwd+bwd | iter4 fwd+bwd | change |
|---|---------------|---------------|--------|
| 6 | 1.612         | **1.347**     | 1.20× faster |

**KEPT.** Pipelining helps where SMEM allows. No regression elsewhere (J>6 still at stages=1).

## Iteration 5 (in progress, jid 38836289 — FINAL)

**Hypothesis:** bump `dkv_block_m` from 32 to 64 at J ≥ 4. Bigger BLOCK_M improves tensor-core utilization on the `dscore_wk^T @ q` and `p_wv^T @ dy` reductions. Currently the dKV kernel runs at small (32, 16) tiles which can't saturate H100's wmma. SMEM at J=14 with 16 active dK+dV slots fits at BLOCK_M=64.

**Change:** `dkv_block_m = 64 if (bf16 and J ≥ 4) else 32`.

**Output:** `tests/bench_bqa_dyn_loop_iter5_dkv_m64.json`

## Iteration 5 RESULT (dkv_block_m=64 at J≥4, jid 38836289)

| J | iter4 fwd+bwd | iter5 fwd+bwd | change |
|---|---------------|---------------|--------|
| 6 | 1.347 | 1.417 | 1.05× SLOWER |
| 8 | 2.627 | 3.311 | 1.26× SLOWER |
| 10 | 4.124 | 5.733 | 1.39× SLOWER |
| 14 | 10.784 | 13.646 | 1.27× SLOWER |

**REVERTED.** Bigger BLOCK_M=64 in dKV likely caused register pressure with 16 active dK/dV slots.

## Iteration 6 (in progress, jid 38836926)

**Hypothesis:** restore qk cache up to J=8 in bwd_dq. At J=4-8 cells we currently recompute qk_j in Phase 4 (1 extra `q × K_j` matmul per j per s_blk); cache eliminates that. At J > 8 still recompute (full cache busts SMEM).

**Change:** declare qk_4..qk_7 inside `if J > 4:` block. Phase 4 reads cached qk_4..qk_7 for j=4-7, recomputes only at j ≥ 8.

**Output:** `tests/bench_bqa_dyn_loop_iter6_qk_cache_j8.json`

## Iteration 7 (planned)

**Hypothesis:** eliminate the H× HBM bloat in bwd_dkv. Currently dK_per_h has shape (B, H, T, J, D) and PyTorch sums across H after the kernel. Switching to `tl.atomic_add` directly into (B, T, J, D) saves H× HBM writes and the post-kernel reduction. H ranges from 6 to 14 in our sweep, so potential 6-14× HBM savings on the dKV write side.

**Risk:** atomic_add is non-deterministic in op order. fp32 rounding order will vary run-to-run. Acceptable for training.

## Iteration 6 RESULT (qk cache to J=8 in bwd_dq, jid 38836926)

| J | iter4 fwd+bwd | iter6 fwd+bwd | change |
|---|---------------|---------------|--------|
| 6 | 1.347 | 1.337 | ~0% |
| 8 | 2.627 | 2.636 | ~0% |
| 10 | 4.124 | 4.143 | ~0% |
| 14 | 10.784 | 10.773 | ~0% |

**REVERTED.** Recompute path is essentially free; cache extension didn't move the needle. Bottleneck must be elsewhere — pointing at HBM bandwidth in dKV (H× bloat).

## Iteration 7 (in progress, jid 38837262)

**Hypothesis:** eliminate H× HBM bloat in bwd_dkv. Currently the kernel writes per-(b, h, t, j, D) into dK_per_h shape (B, H, T, J, D) then PyTorch sums across H. Switching to `tl.atomic_add` directly into (B, T, J, D) saves H× HBM bandwidth on the dKV write path (H = 6 to 14 in our sweep) and eliminates the post-kernel reduction.

**Change:**
- Kernel signature: dK_PH/dV_PH/dVE_PH (5D) → DK_OUT/DV_OUT/DVE_OUT (4D, basis-shaped).
- Stores: `tl.store` → `tl.atomic_add`.
- Wrapper: allocate `dk_basis_acc, dv_basis_acc, dve_basis_acc` as zeros (not empty) at (B, T, J, D), skip the H-sum.
- dGate path unchanged (already (B, T, H), single-writer per program).

**Risk:** atomic_add is non-deterministic in op order; fp32 rounding order varies run-to-run. Acceptable for training.

**Output:** `tests/bench_bqa_dyn_loop_iter7_atomic_dkv.json`

## Iteration 7 RESULT (atomic dK/dV in bwd_dkv, jid 38838836)

| J | iter4 fwd+bwd | iter7 fwd+bwd | change | bwd_min_cos |
|---|---------------|---------------|--------|-------------|
| 3 | 0.523 | 0.557 | 1.06× SLOWER | 0.999976 |
| 6 | 1.347 | 1.391 | 1.03× SLOWER | 0.999982 |
| 8 | 2.627 | 2.722 | 1.04× SLOWER | 0.999983 |
| 10 | 4.124 | 4.297 | 1.04× SLOWER | 0.999983 |
| 14 | 10.784 | 10.646 | 1.01× faster | 0.999984 |

**REVERTED.** Atomic_add change is correct (bwd_min_cos > 0.99997 across all J — first time we verified bwd numerics in this loop). But H100's atomic_add overhead offsets the H× HBM savings. Roughly neutral, slightly slower at small/mid J.

## Final state

- **Kept**: iter 2 (num_warps=8 at J≥4) + iter 4 (num_stages=2 in fwd at J≤6).
- **Verified**: bwd correctness (cos > 0.99997 vs autograd-through-explicit-reference at all J).
- **Confirmation bench**: jid 38840951 with the kept-only kernel state (iter 7 reverted).

### Speedup vs iter 1 baseline (microbench at B=2, T=1024, D=128, bf16, H100):

| J  | fwd ms (1→4) | speedup | fwd+bwd ms (1→4) | speedup |
|----|--------------|---------|-------------------|---------|
| 3  | 0.095 → 0.095 | 1.00× | 0.537 → 0.523 | 1.03× |
| 6  | 0.816 → **0.194** | **4.21×** | 2.935 → **1.347** | **2.18×** |
| 8  | 1.156 → **0.375** | **3.08×** | 4.985 → **2.627** | **1.90×** |
| 10 | 1.946 → **0.622** | **3.13×** | 9.096 → **4.124** | **2.21×** |
| 14 | 3.500 → **1.197** | **2.92×** | 25.621 → **10.784** | **2.38×** |

## FINAL: upstream 036a3ea wins on H100, no merge needed

Pulled upstream commit 036a3ea (60+ config A100 sweep, J-dependent block heuristic, ve-aware cases). Reset my local 73e7ff8 (iter 2 + iter 4) to upstream and re-benched on H100:

| J  | iter4 fwd | upstream fwd | iter4 fwd+bwd | upstream fwd+bwd | upstream wins? |
|----|-----------|--------------|---------------|-------------------|----------------|
| 3  | 0.095 | 0.095     | 0.523 | 0.485    | ~tied |
| 6  | 0.194 | 0.194     | 1.347 | **1.114** (1.21×) | bwd only |
| 8  | 0.375 | **0.236** (1.59×) | 2.627 | **1.988** (1.32×) | YES |
| 10 | 0.622 | **0.375** (1.66×) | 4.124 | **3.924** | YES |
| 14 | 1.197 | FAILED (SMEM 256 KB > 228 KB) | 10.784 | FAILED | upstream busts at J=14 fwd-only on H100 |

Upstream wins at J=8 and J=10 by 1.3-1.7×. My iter 2 (num_warps=8) + iter 4 (num_stages=2 at J≤6) findings are subsumed by upstream's more thorough config sweep — no merge needed.

J=14 (d=28 full, no-ve) busts H100 SMEM under upstream's `(BLOCK_M=128, BLOCK_N=16, num_warps=8, num_stages=2)` selected by the `if J >= 4:` branch in `_fwd_block_cfg`: K/V tiles × 14 × 2 stages = 224 KB just for K, blows 228 KB H100 cap. ve-on path is OK because it has a special case.

User scoped sweep to up-to-d=20 so J=14 is out of scope. Final state: upstream `036a3ea` is strictly better for the cells we run; no kernel changes needed locally. Commit my test infra (bench script with bwd check + iter result JSONs + this loop log) on top of upstream and end loop.
