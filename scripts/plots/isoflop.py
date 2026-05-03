"""Parse apr29/apr30 attn_compare logs and plot isoflop curves (val bpb vs params, per FLOP budget)."""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = Path("/home/sarthmit/scratch/bqa/runs/logs/attn_compare")
OUT_DIR = Path("/scratch/sarthmit/bqa/scripts/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = "apr30"

PATTERN = re.compile(r"attncmp_(?:apr29|apr30)_(?P<arch>bqa|gqa|mha)_d(?P<depth>\d+)_f(?P<flops>[\d.e+]+)\.log$")

FIELDS = {
    "n_layer":        re.compile(r'"n_layer":\s*(\d+)'),
    "n_head":         re.compile(r'"n_head":\s*(\d+)'),
    "n_kv_head":      re.compile(r'"n_kv_head":\s*(\d+)'),
    "n_embd":         re.compile(r'"n_embd":\s*(\d+)'),
    "attn_kind":      re.compile(r'"attn_kind":\s*"(\w+)"'),
    "total_params":   re.compile(r"^total\s*:\s*([\d,]+)", re.MULTILINE),
    "flops_per_tok":  re.compile(r"Estimated FLOPs per token:\s*([\deE+.\-]+)"),
    "total_flops":    re.compile(r"Total training FLOPs estimate:\s*([\deE+.\-]+)"),
    "val_bpb":        re.compile(r"Minimum validation bpb:\s*([\d.]+)"),
}

def parse(path: Path):
    text = path.read_text(errors="ignore")
    out = {}
    for k, pat in FIELDS.items():
        m = pat.search(text)
        if m is None:
            return None
        v = m.group(1).replace(",", "")
        out[k] = v
    out["total_params"] = int(out["total_params"])
    out["flops_per_tok"] = float(out["flops_per_tok"])
    out["total_flops"] = float(out["total_flops"])
    out["val_bpb"] = float(out["val_bpb"])
    for k in ("n_layer", "n_head", "n_kv_head", "n_embd"):
        out[k] = int(out[k])
    return out

rows = []
candidates = sorted(LOG_DIR.glob("attncmp_apr29_*_d*_f*.log")) + sorted(
    LOG_DIR.glob("attncmp_apr30_*_d*_f*.log")
)
for p in candidates:
    m = PATTERN.search(p.name)
    if not m:
        continue
    g = m.groupdict()
    parsed = parse(p)
    if parsed is None:
        print(f"skip (incomplete): {p.name}")
        continue
    rows.append({
        "log": p.name,
        "arch": g["arch"],
        "depth_label": int(g["depth"]),
        "flops_budget": float(g["flops"]),
        **parsed,
    })

print(f"Parsed {len(rows)} runs")
for r in rows:
    print(f"  {r['arch']:3s} d{r['depth_label']:>2d} budget={r['flops_budget']:.2e} "
          f"N={r['total_params']/1e6:6.1f}M  C={r['total_flops']:.2e}  bpb={r['val_bpb']:.4f}")

# One column per architecture, all FLOP budgets overlaid in each panel.
budgets = sorted({r["flops_budget"] for r in rows})
archs = ("mha", "gqa", "bqa")
arch_marker = {"mha": "o", "gqa": "s", "bqa": "^"}
cmap = plt.get_cmap("viridis")
budget_color = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}

# Shared y-range across panels for fair visual comparison.
all_bpb = [r["val_bpb"] for r in rows]
y_lo, y_hi = min(all_bpb), max(all_bpb)
y_pad = 0.04 * (y_hi - y_lo)

fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 5.0),
                         sharey=True)
if len(archs) == 1:
    axes = [axes]

for ax, arch in zip(axes, archs):
    for budget in budgets:
        pts = sorted(
            [r for r in rows if r["flops_budget"] == budget and r["arch"] == arch],
            key=lambda r: r["total_params"],
        )
        if not pts:
            continue
        xs = np.array([r["total_params"] / 1e6 for r in pts])
        ys = np.array([r["val_bpb"] for r in pts])
        c = budget_color[budget]
        label = f"C ≈ {budget:.2e}"
        ax.scatter(xs, ys, marker=arch_marker[arch], color=c, s=55,
                   label=label, zorder=3, edgecolor="black", linewidth=0.4)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(f"d{r['depth_label']}", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=7,
                        color=c)

        if len(pts) >= 3:
            log_n = np.log10(np.array([r["total_params"] for r in pts]))
            coeffs = np.polyfit(log_n, ys, 2)
            xs_fit_logn = np.linspace(log_n.min() - 0.15, log_n.max() + 0.15, 100)
            ys_fit = np.polyval(coeffs, xs_fit_logn)
            xs_fit = 10 ** xs_fit_logn / 1e6
            ax.plot(xs_fit, ys_fit, color=c, linewidth=1.4, alpha=0.85, zorder=2)
            a2, a1, a0 = coeffs
            if a2 > 0:
                x_star_logn = -a1 / (2 * a2)
                y_star = a0 + a1 * x_star_logn + a2 * x_star_logn ** 2
                if log_n.min() - 0.3 <= x_star_logn <= log_n.max() + 0.3:
                    ax.plot(10 ** x_star_logn / 1e6, y_star, marker="*",
                            color=c, markersize=14, markeredgecolor="black",
                            markeredgewidth=0.6, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Total params (M)")
    ax.set_title(arch.upper())
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.legend(loc="best", fontsize=8, title="FLOP budget")

axes[0].set_ylabel("Min validation bpb")
fig.suptitle(f"Isoflop curves — {TAG} attn_compare (cols: arch, lines: FLOP budget)",
             y=1.02)
fig.tight_layout()

out_png = OUT_DIR / f"isoflop_{TAG}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nWrote {out_png}")

# Also: combined view, val bpb vs total compute used, one line per (arch, depth)
arch_color = {"mha": "#1f77b4", "gqa": "#2ca02c", "bqa": "#d62728"}
all_depths = sorted({r["depth_label"] for r in rows})
d_min, d_max = (min(all_depths), max(all_depths)) if all_depths else (12, 20)
def depth_alpha(d):
    if d_max == d_min:
        return 0.85
    return 0.45 + 0.45 * (d - d_min) / (d_max - d_min)

fig2, ax2 = plt.subplots(figsize=(7, 5))
for arch in ("mha", "gqa", "bqa"):
    for depth in sorted({r["depth_label"] for r in rows if r["arch"] == arch}):
        pts = sorted(
            [r for r in rows if r["arch"] == arch and r["depth_label"] == depth],
            key=lambda r: r["total_flops"],
        )
        if len(pts) < 2:
            continue
        xs = [r["total_flops"] for r in pts]
        ys = [r["val_bpb"] for r in pts]
        ax2.plot(xs, ys, marker=arch_marker[arch], color=arch_color[arch],
                 alpha=depth_alpha(depth),
                 label=f"{arch.upper()} d{depth}", linewidth=1.4, markersize=6)
ax2.set_xscale("log")
ax2.set_xlabel("Total training FLOPs")
ax2.set_ylabel("Min validation bpb")
ax2.set_title("Compute scaling, per (arch, depth)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=8, ncol=3)
fig2.tight_layout()
out_png2 = OUT_DIR / f"compute_scaling_{TAG}.png"
fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
print(f"Wrote {out_png2}")
