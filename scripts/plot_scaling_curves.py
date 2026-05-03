"""Plot isoflop scaling curves: val BPB vs model params at fixed FLOPS budgets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import re

LOG_DIR = "runs/logs/attn_compare"
LABEL = "attncmp_apr29"
ARCHS = ["mha", "gqa", "bqa"]
DEPTHS = [12, 16, 20]
FLOPS_BUDGETS = ["1e18", "2.15e18", "4.64e18", "1e19"]
FLOPS_LABELS = {"1e18": r"$10^{18}$", "2.15e18": r"$2.15 \times 10^{18}$",
                "4.64e18": r"$4.64 \times 10^{18}$", "1e19": r"$10^{19}$"}

ARCH_COLORS = {"mha": "#1f77b4", "gqa": "#ff7f0e", "bqa": "#2ca02c"}
ARCH_LABELS = {"mha": "MHA", "gqa": "GQA", "bqa": "BQA"}
ARCH_MARKERS = {"mha": "o", "gqa": "s", "bqa": "D"}


def extract_from_log(log_path):
    """Extract final val bpb and total params from a training log."""
    last_bpb = None
    total_params = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Validation bpb:\s+([\d.]+)", line)
            if m:
                last_bpb = float(m.group(1))
            m = re.search(r"^total\s+:\s+([\d,]+)", line)
            if m:
                total_params = int(m.group(1).replace(",", ""))
    return last_bpb, total_params


def is_complete(log_path):
    """Check if a run finished cleanly."""
    with open(log_path) as f:
        tail = f.readlines()[-10:]
    tail_str = "".join(tail)
    return "Peak memory" in tail_str or "Total training time" in tail_str


# Collect data
data = {}  # (arch, depth, flops_budget) -> (params, bpb)

for arch in ARCHS:
    for depth in DEPTHS:
        for fb in FLOPS_BUDGETS:
            log_file = os.path.join(LOG_DIR, f"{LABEL}_{arch}_d{depth}_f{fb}.log")
            if not os.path.exists(log_file):
                continue
            if not is_complete(log_file):
                print(f"  SKIP (incomplete): {log_file}")
                continue
            bpb, params = extract_from_log(log_file)
            if bpb is not None and params is not None:
                data[(arch, depth, fb)] = (params, bpb)
                print(f"  {arch} d{depth} f{fb}: {params/1e6:.0f}M params, bpb={bpb:.4f}")

# --- Isoflop plot: one subplot per FLOPS budget ---
fig, axes = plt.subplots(1, len(FLOPS_BUDGETS), figsize=(20, 5), sharey=True)

for idx, fb in enumerate(FLOPS_BUDGETS):
    ax = axes[idx]
    for arch in ARCHS:
        pts = []
        for depth in DEPTHS:
            key = (arch, depth, fb)
            if key in data:
                params, bpb = data[key]
                pts.append((params, bpb, depth))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] / 1e6 for p in pts]  # in millions
        ys = [p[1] for p in pts]
        depths = [p[2] for p in pts]

        ax.plot(xs, ys,
                color=ARCH_COLORS[arch],
                marker=ARCH_MARKERS[arch],
                markersize=8,
                label=ARCH_LABELS[arch],
                linewidth=2,
                alpha=0.9)
        # Annotate with depth
        for x, y, d in zip(xs, ys, depths):
            ax.annotate(f"d{d}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=ARCH_COLORS[arch], alpha=0.7)

    ax.set_title(f"FLOPS = {FLOPS_LABELS[fb]}", fontsize=11)
    ax.set_xlabel("Parameters (M)", fontsize=10)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.set_ylabel("Validation BPB", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")

plt.suptitle("Isoflop Scaling Curves: MHA vs GQA vs BQA", fontsize=14, y=1.02)
plt.tight_layout()
out_path = "runs/isoflop_curves_apr29.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")

# --- Also make a single combined plot with compute-optimal frontier ---
fig2, ax2 = plt.subplots(figsize=(8, 5.5))

for arch in ARCHS:
    # For each FLOPS budget, pick the best depth
    frontier = []
    for fb in FLOPS_BUDGETS:
        best = None
        for depth in DEPTHS:
            key = (arch, depth, fb)
            if key in data:
                params, bpb = data[key]
                if best is None or bpb < best[1]:
                    best = (float(fb), bpb, depth)
        if best:
            frontier.append(best)

    frontier.sort()
    xs = [p[0] for p in frontier]
    ys = [p[1] for p in frontier]
    best_depths = [p[2] for p in frontier]

    ax2.plot(xs, ys,
             color=ARCH_COLORS[arch],
             marker=ARCH_MARKERS[arch],
             markersize=9,
             label=ARCH_LABELS[arch],
             linewidth=2.5)
    for x, y, d in zip(xs, ys, best_depths):
        ax2.annotate(f"d{d}", (x, y), textcoords="offset points",
                     xytext=(6, 5), fontsize=8, color=ARCH_COLORS[arch])

ax2.set_xscale("log")
ax2.set_xlabel("Training FLOPS", fontsize=12)
ax2.set_ylabel("Validation BPB", fontsize=12)
ax2.set_title("Compute-Optimal Frontier", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = "runs/compute_optimal_frontier_apr29.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path2}")
