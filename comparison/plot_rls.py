#!/usr/bin/env python3
"""
Generate all benchmark plots for flight-solver RLS crate.

Usage:
    cd comparison/
    python3 plot.py

Reads results_rls_c.csv, results_rls_rust.csv, and results_rls_stability.csv from cwd.
Outputs PNGs to plots/ subdirectory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "figure.dpi": 150,
})

STD_COLOR = "#e3442b"
IQR_COLOR = "#9b59b6"
C_COLOR = "#2e8b57"
REF_COLOR = "#888888"

CONFIGS = ["motor_n4p1", "g1g2_n8p3", "force_n4p3"]
CONFIG_LABELS = {
    "motor_n4p1": "Motor\nn=4, p=1",
    "g1g2_n8p3": "G1/G2\nn=8, p=3",
    "force_n4p3": "Force\nn=4, p=3",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Throughput plots (from results_rls_c.csv + results_rls_rust.csv)
# ═══════════════════════════════════════════════════════════════════════════

def plot_throughput(df_c, df_r, outdir):
    """Grouped bar: ns/step for all solvers x configs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONFIGS))
    width = 0.22

    c_t, std_t, iqr_t = [], [], []
    for cfg in CONFIGS:
        c_t.append(df_c[df_c["config"] == cfg]["ns_per_step"].values[0])
        std_t.append(df_r[(df_r["config"] == cfg) & (df_r["solver"] == "standard")]["ns_per_step"].values[0])
        iqr_t.append(df_r[(df_r["config"] == cfg) & (df_r["solver"] == "inverse_qr")]["ns_per_step"].values[0])

    bars = [
        ax.bar(x - width, c_t, width, label="C (indiflight)", color=C_COLOR, edgecolor="white"),
        ax.bar(x, std_t, width, label="Rust standard", color=STD_COLOR, edgecolor="white"),
        ax.bar(x + width, iqr_t, width, label="Rust inverse QR", color=IQR_COLOR, edgecolor="white"),
    ]
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("ns / step")
    ax.set_title("RLS Update Latency (100-step sequence, per-step average)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS])
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(iqr_t) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rls_throughput.png"))
    plt.close(fig)
    print("  throughput.png")


def plot_rust_vs_c(df_c, df_r, outdir):
    """Paired bars: Rust standard vs C standard with speedup annotations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CONFIGS))
    width = 0.3

    c_t, r_t = [], []
    for cfg in CONFIGS:
        c_t.append(df_c[df_c["config"] == cfg]["ns_per_step"].values[0])
        r_t.append(df_r[(df_r["config"] == cfg) & (df_r["solver"] == "standard")]["ns_per_step"].values[0])

    ax.bar(x - width/2, c_t, width, label="C (indiflight)", color=C_COLOR, edgecolor="white")
    ax.bar(x + width/2, r_t, width, label="Rust standard", color=STD_COLOR, edgecolor="white")

    for i, (ct, rt) in enumerate(zip(c_t, r_t)):
        for val, xoff in [(ct, -width/2), (rt, width/2)]:
            ax.text(x[i] + xoff, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        speedup = ct / rt
        ax.annotate(f"{speedup:.2f}x", xy=(x[i] + width/2, rt),
                    xytext=(x[i] + width*1.2, rt * 0.7), fontsize=10, fontweight="bold",
                    color=STD_COLOR, arrowprops=dict(arrowstyle="->", color=STD_COLOR, lw=1.5))

    ax.set_ylabel("ns / step")
    ax.set_title("Rust vs C: Standard RLS (same algorithm)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS])
    ax.legend()
    ax.set_ylim(0, max(c_t) * 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rls_rust_vs_c.png"))
    plt.close(fig)
    print("  rust_vs_c.png")


def plot_accuracy(df_r, outdir):
    """Bar chart: solver agreement |standard - inverse_qr|."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    diffs = []
    for cfg in CONFIGS:
        x_std = df_r[(df_r["config"] == cfg) & (df_r["solver"] == "standard")]["final_x00"].values[0]
        x_iqr = df_r[(df_r["config"] == cfg) & (df_r["solver"] == "inverse_qr")]["final_x00"].values[0]
        diffs.append(abs(x_std - x_iqr))

    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, diffs, color=IQR_COLOR, width=0.5, edgecolor="white")
    for bar, d in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f"{d:.1e}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1e-4, color="gray", linestyle="--", linewidth=1, label="1e-4")
    ax.set_ylabel("|standard − inverse_qr| on X[0,0]")
    ax.set_title("Solver Agreement (after 100 steps, same input)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS])
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rls_accuracy.png"))
    plt.close(fig)
    print("  accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Stability plots (from results_rls_stability.csv)
# ═══════════════════════════════════════════════════════════════════════════

SCENARIO_LABELS = {
    "high_gamma": "High Initial Covariance\n(γ = 10⁶)",
    "low_excitation": "Low Excitation Burst\n(500 quiet → 500 loud)",
    "long_convergence": "Long Convergence\n(2000 steps, λ = 0.999)",
}

def plot_stability_drift(df_stab, outdir):
    """Per-scenario: f32 drift from f64 reference."""
    scenarios = df_stab["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = df_stab[df_stab["scenario"] == scenario]
        steps = sub["step"].values
        ax.semilogy(steps, sub["std_vs_ref"].values, color=STD_COLOR, linewidth=1.2, label="Standard RLS")
        ax.semilogy(steps, sub["iqr_vs_ref"].values, color=IQR_COLOR, linewidth=1.2, label="Inverse QR-RLS")
        ax.set_xlabel("Step")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontweight="bold", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")

    axes[0].set_ylabel("Max |f32 − f64 reference|")
    fig.suptitle("Numerical Drift from f64 Ground Truth", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rls_stability_drift.png"), bbox_inches="tight")
    plt.close(fig)
    print("  stability_drift.png")


def plot_stability_vs_true(df_stab, outdir):
    """Per-scenario: parameter error vs true values."""
    scenarios = df_stab["scenario"].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = df_stab[df_stab["scenario"] == scenario]
        steps = sub["step"].values
        ax.semilogy(steps, sub["ref_vs_true"].values, color=REF_COLOR, linewidth=1.5, alpha=0.7,
                    label="f64 reference", linestyle="--")
        ax.semilogy(steps, sub["std_vs_true"].values, color=STD_COLOR, linewidth=1.2, label="Standard (f32)")
        ax.semilogy(steps, sub["iqr_vs_true"].values, color=IQR_COLOR, linewidth=1.2, label="Inverse QR (f32)")
        ax.set_xlabel("Step")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontweight="bold", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel("Max |estimate − true|")
    fig.suptitle("Parameter Estimation Error vs True Values", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rls_stability_vs_true.png"), bbox_inches="tight")
    plt.close(fig)
    print("  stability_vs_true.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(df_c, df_r, df_stab):
    print("\n## Throughput (ns/step)\n")
    print("| Config | C standard | Rust standard | Rust inverse QR | Speedup |")
    print("|--------|-----------|--------------|----------------|---------|")
    for cfg in CONFIGS:
        c_ns = df_c[df_c["config"] == cfg]["ns_per_step"].values[0]
        r_std = df_r[(df_r["config"] == cfg) & (df_r["solver"] == "standard")]["ns_per_step"].values[0]
        r_iqr = df_r[(df_r["config"] == cfg) & (df_r["solver"] == "inverse_qr")]["ns_per_step"].values[0]
        print(f"| {cfg} | {c_ns:.1f} ns | {r_std:.1f} ns | {r_iqr:.1f} ns | {c_ns/r_std:.2f}x |")

    if df_stab is not None:
        print("\n## Numerical Drift at Final Step\n")
        print("| Scenario | Steps | Standard drift | Inverse QR drift | Ratio |")
        print("|----------|-------|---------------|-----------------|-------|")
        for scenario in df_stab["scenario"].unique():
            last = df_stab[df_stab["scenario"] == scenario].iloc[-1]
            std_d = last["std_vs_ref"]
            iqr_d = last["iqr_vs_ref"]
            ratio = std_d / max(iqr_d, 1e-30)
            print(f"| {scenario} | {int(last['step'])} | {std_d:.2e} | {iqr_d:.2e} | **{ratio:.0f}x** |")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # Load throughput data
    df_c = pd.read_csv("results_rls_c.csv")
    df_r = pd.read_csv("results_rls_rust.csv")
    df_c.columns = df_c.columns.str.strip()
    df_r.columns = df_r.columns.str.strip()
    print(f"Loaded throughput data: C={len(df_c)} rows, Rust={len(df_r)} rows")

    # Load stability data (optional)
    df_stab = None
    if os.path.exists("results_rls_stability.csv"):
        df_stab = pd.read_csv("results_rls_stability.csv")
        df_stab.columns = df_stab.columns.str.strip()
        print(f"Loaded stability data: {len(df_stab)} rows")

    print("\nGenerating plots:")
    plot_throughput(df_c, df_r, outdir)
    plot_rust_vs_c(df_c, df_r, outdir)
    plot_accuracy(df_r, outdir)

    if df_stab is not None:
        plot_stability_drift(df_stab, outdir)
        plot_stability_vs_true(df_stab, outdir)

    print_summary(df_c, df_r, df_stab)
    print(f"\nPlots saved to {outdir}/")


if __name__ == "__main__":
    main()
