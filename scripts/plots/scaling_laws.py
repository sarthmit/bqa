"""Fit Chinchilla-style scaling laws from apr29/apr30 isoflop curves: N*(C), D*(C)."""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = Path("/home/sarthmit/scratch/bqa/runs/logs/attn_compare")
OUT_DIR = Path("/scratch/sarthmit/bqa/scripts/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = "apr30"

PATTERN = re.compile(
    r"attncmp_(?:apr29|apr30)_(?P<arch>bqa|gqa|mha)_d(?P<depth>\d+)_f(?P<flops>[\d.e+]+)\.log$"
)

FIELDS = {
    "n_layer":       re.compile(r'"n_layer":\s*(\d+)'),
    "n_head":        re.compile(r'"n_head":\s*(\d+)'),
    "n_kv_head":     re.compile(r'"n_kv_head":\s*(\d+)'),
    "n_embd":        re.compile(r'"n_embd":\s*(\d+)'),
    "total_params":  re.compile(r"^total\s*:\s*([\d,]+)", re.MULTILINE),
    "flops_per_tok": re.compile(r"Estimated FLOPs per token:\s*([\deE+.\-]+)"),
    "total_flops":   re.compile(r"Total training FLOPs estimate:\s*([\deE+.\-]+)"),
    "val_bpb":       re.compile(r"Minimum validation bpb:\s*([\d.]+)"),
}

def parse(path):
    text = path.read_text(errors="ignore")
    out = {}
    for k, pat in FIELDS.items():
        m = pat.search(text)
        if m is None:
            return None
        out[k] = m.group(1).replace(",", "")
    out["total_params"] = int(out["total_params"])
    out["flops_per_tok"] = float(out["flops_per_tok"])
    out["total_flops"] = float(out["total_flops"])
    out["val_bpb"] = float(out["val_bpb"])
    for k in ("n_layer", "n_head", "n_kv_head", "n_embd"):
        out[k] = int(out[k])
    return out

rows = []
for p in sorted(LOG_DIR.glob("attncmp_apr29_*_d*_f*.log")) + sorted(
    LOG_DIR.glob("attncmp_apr30_*_d*_f*.log")
):
    m = PATTERN.search(p.name)
    if not m:
        continue
    parsed = parse(p)
    if parsed is None:
        continue
    rows.append({
        "log": p.name,
        "arch": m.group("arch"),
        "depth_label": int(m.group("depth")),
        "flops_budget": float(m.group("flops")),
        **parsed,
    })
print(f"Parsed {len(rows)} runs")

# ---- For each (arch, C): fit quadratic in log10(N), get N* and D* ----
optima = []
for arch in ("mha", "gqa", "bqa"):
    for budget in sorted({r["flops_budget"] for r in rows}):
        pts = sorted(
            [r for r in rows if r["arch"] == arch and r["flops_budget"] == budget],
            key=lambda r: r["total_params"],
        )
        if len(pts) < 3:
            continue
        log_n = np.log10([r["total_params"] for r in pts])
        bpb = np.array([r["val_bpb"] for r in pts])
        a2, a1, a0 = np.polyfit(log_n, bpb, 2)
        if a2 <= 0:
            print(f"  skip {arch} C={budget:.2e}: non-convex parabola")
            continue
        x_star = -a1 / (2 * a2)
        if not (log_n.min() - 0.3 <= x_star <= log_n.max() + 0.3):
            print(f"  skip {arch} C={budget:.2e}: optimum out of range "
                  f"(log10 N*={x_star:.2f} vs [{log_n.min():.2f},{log_n.max():.2f}])")
            continue
        N_star = 10 ** x_star
        L_star = a0 + a1 * x_star + a2 * x_star ** 2
        # D* via interp of flops_per_tok at N*; D = total_flops / flops_per_tok.
        log_fpt = np.log10([r["flops_per_tok"] for r in pts])
        fpt_star = 10 ** np.interp(x_star, log_n, log_fpt)
        # Use mean measured C across runs at this budget (slightly < nominal).
        C_meas = float(np.mean([r["total_flops"] for r in pts]))
        D_star = C_meas / fpt_star
        optima.append({
            "arch": arch, "C_nominal": budget, "C": C_meas,
            "N_star": N_star, "D_star": D_star, "L_star": L_star,
            "fpt_star": fpt_star,
        })

print("\nCompute-optimal points:")
print(f"  {'arch':3s} {'C':>10s} {'N* (M)':>10s} {'D* (B tok)':>12s} {'D*/N*':>8s} {'bpb':>7s}")
for o in optima:
    print(f"  {o['arch']:3s} {o['C']:10.2e} {o['N_star']/1e6:10.1f} "
          f"{o['D_star']/1e9:12.2f} {o['D_star']/o['N_star']:8.1f} {o['L_star']:7.4f}")

# ---- Fit power laws per arch: log10(N*) = alpha * log10(C) + b_n, etc. ----
arch_color = {"mha": "#1f77b4", "gqa": "#2ca02c", "bqa": "#d62728"}
arch_marker = {"mha": "o", "gqa": "s", "bqa": "^"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fits = {}
for arch in ("mha", "gqa", "bqa"):
    pts = [o for o in optima if o["arch"] == arch]
    if len(pts) < 2:
        continue
    Cs = np.array([o["C"] for o in pts])
    Ns = np.array([o["N_star"] for o in pts])
    Ds = np.array([o["D_star"] for o in pts])
    alpha, bn = np.polyfit(np.log10(Cs), np.log10(Ns), 1)
    beta,  bd = np.polyfit(np.log10(Cs), np.log10(Ds), 1)
    fits[arch] = {"alpha": alpha, "Nc": 10 ** bn, "beta": beta, "Dc": 10 ** bd}
    color, marker = arch_color[arch], arch_marker[arch]
    Cs_fit = np.logspace(np.log10(Cs.min()) - 0.3, np.log10(Cs.max()) + 0.3, 50)

    axes[0].scatter(Cs, Ns, color=color, marker=marker, s=70,
                    edgecolor="black", linewidth=0.4, zorder=3,
                    label=f"{arch.upper()}: N* ∝ C^{alpha:.3f}")
    axes[0].plot(Cs_fit, (10 ** bn) * Cs_fit ** alpha, color=color,
                 linestyle="--", alpha=0.7)

    axes[1].scatter(Cs, Ds, color=color, marker=marker, s=70,
                    edgecolor="black", linewidth=0.4, zorder=3,
                    label=f"{arch.upper()}: D* ∝ C^{beta:.3f}")
    axes[1].plot(Cs_fit, (10 ** bd) * Cs_fit ** beta, color=color,
                 linestyle="--", alpha=0.7)

for ax, ylabel, title in [
    (axes[0], "Compute-optimal N* (params)", "Optimal model size vs compute"),
    (axes[1], "Compute-optimal D* (tokens)", "Optimal dataset size vs compute"),
]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training compute C (FLOPs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle(f"Scaling laws — {TAG} attn_compare (Chinchilla-style fits)", y=1.02)
fig.tight_layout()
out_png = OUT_DIR / f"scaling_laws_{TAG}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nWrote {out_png}")

print("\nFitted exponents (N* = N_c · C^alpha, D* = D_c · C^beta):")
for arch, f in fits.items():
    print(f"  {arch.upper():3s}: alpha={f['alpha']:.3f}  N_c={f['Nc']:.3e}    "
          f"beta={f['beta']:.3f}  D_c={f['Dc']:.3e}")
