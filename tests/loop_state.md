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
