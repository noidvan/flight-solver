#!/usr/bin/env python3
"""
Generate benchmark comparison plots: Rust flight-solver CLS vs C ActiveSetCtlAlloc.

Usage:
    python3 plot.py              # reads results_c.csv and results_rust.csv from cwd
    python3 plot.py --dir bench  # specify directory containing CSVs

Outputs PNG files in plots/ subdirectory.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "figure.dpi": 150,
})

RUST_COLOR = "#e3442b"
C_QR_NAIVE_COLOR = "#4682b4"
C_QR_COLOR = "#2e8b57"


def load_data(directory):
    c_path = os.path.join(directory, "results_wls_c.csv")
    r_path = os.path.join(directory, "results_wls_rust.csv")

    df_c = pd.read_csv(c_path)
    df_r = pd.read_csv(r_path)

    # Trim whitespace from column names
    df_c.columns = df_c.columns.str.strip()
    df_r.columns = df_r.columns.str.strip()

    return df_c, df_r


def plot_throughput(df_c, df_r, outdir):
    """Bar chart: median solves per second for each solver."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    c_naive = df_c[df_c["solver"] == "qr_naive"]["time_ns"]
    c_qr = df_c[df_c["solver"] == "qr"]["time_ns"]
    r_inc = df_r["time_ns"]

    medians_ns = [c_naive.median(), c_qr.median(), r_inc.median()]
    throughputs = [1e9 / m for m in medians_ns]
    labels = ["C qr_naive", "C qr\n(incremental)", "Rust\n(incremental)"]
    colors = [C_QR_NAIVE_COLOR, C_QR_COLOR, RUST_COLOR]

    bars = ax.bar(labels, throughputs, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{tp/1e6:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Solves / second")
    ax.set_title("Solver Throughput (median, 1000 test cases)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(0, max(throughputs) * 1.18)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wls_throughput.png"))
    plt.close(fig)
    print(f"  throughput.png  (medians: {medians_ns[0]:.0f} / {medians_ns[1]:.0f} / {medians_ns[2]:.0f} ns)")


def plot_timing_distribution(df_c, df_r, outdir):
    """Box plot of per-case solve time for each solver."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    c_naive = df_c[df_c["solver"] == "qr_naive"]["time_ns"].values
    c_qr = df_c[df_c["solver"] == "qr"]["time_ns"].values
    r_inc = df_r["time_ns"].values

    data = [c_naive, c_qr, r_inc]
    labels = ["C qr_naive", "C qr (incr.)", "Rust (incr.)"]
    colors = [C_QR_NAIVE_COLOR, C_QR_COLOR, RUST_COLOR]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Time per solve (ns)")
    ax.set_title("Solve Time Distribution (1000 test cases)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wls_timing_distribution.png"))
    plt.close(fig)
    print("  timing_distribution.png")


def plot_accuracy(df_c, df_r, outdir):
    """Scatter: max |Rust - C_qr_naive| per case, showing numerical consistency."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    c_naive = df_c[df_c["solver"] == "qr_naive"].sort_values("case").reset_index(drop=True)
    r_inc = df_r.sort_values("case").reset_index(drop=True)

    us_cols = [f"us{i}" for i in range(6)]

    max_diffs = []
    for idx in range(len(c_naive)):
        c_us = c_naive.iloc[idx][us_cols].values.astype(float)
        r_us = r_inc.iloc[idx][us_cols].values.astype(float)
        max_diffs.append(np.max(np.abs(c_us - r_us)))

    max_diffs = np.array(max_diffs)
    cases = np.arange(len(max_diffs))

    ax.scatter(cases, max_diffs, s=8, alpha=0.6, c=RUST_COLOR, edgecolors="none")
    ax.axhline(y=1e-4, color="gray", linestyle="--", linewidth=1, label="1e-4 tolerance")
    ax.set_xlabel("Test case index")
    ax.set_ylabel("Max |Rust - C| per actuator")
    ax.set_title("Numerical Consistency: Rust vs C (qr_naive)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    pct_below = 100 * np.mean(max_diffs < 1e-4)
    ax.text(0.02, 0.95, f"{pct_below:.1f}% below 1e-4",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wls_accuracy.png"))
    plt.close(fig)
    print(f"  accuracy.png  (max diff: {max_diffs.max():.2e}, median: {np.median(max_diffs):.2e})")


def plot_timing_ratio_cdf(df_c, df_r, outdir):
    """CDF of (Rust time / C time) ratio per case — shows how Rust compares."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    c_naive = df_c[df_c["solver"] == "qr_naive"].sort_values("case").reset_index(drop=True)
    c_qr = df_c[df_c["solver"] == "qr"].sort_values("case").reset_index(drop=True)
    r_inc = df_r.sort_values("case").reset_index(drop=True)

    ratio_naive = (r_inc["time_ns"].values / c_naive["time_ns"].values)
    ratio_qr = (r_inc["time_ns"].values / c_qr["time_ns"].values)

    for ratio, label, color in [
        (ratio_naive, "Rust / C qr_naive", C_QR_NAIVE_COLOR),
        (ratio_qr, "Rust / C qr (incr.)", C_QR_COLOR),
    ]:
        sorted_r = np.sort(ratio)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax.plot(sorted_r, cdf, color=color, linewidth=2, label=label)

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, label="1:1 (parity)")
    ax.set_xlabel("Time ratio (Rust / C)")
    ax.set_ylabel("Cumulative fraction of test cases")
    ax.set_title("Timing Ratio CDF")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(ratio_naive.max(), ratio_qr.max()) * 1.1)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wls_timing_ratio_cdf.png"))
    plt.close(fig)

    print(f"  timing_ratio_cdf.png  (median Rust/C_naive: {np.median(ratio_naive):.2f}, "
          f"Rust/C_qr: {np.median(ratio_qr):.2f})")


def plot_per_case_timing(df_c, df_r, outdir):
    """Overlay: per-case timing for all three solvers."""
    fig, ax = plt.subplots(figsize=(10, 4))

    c_naive = df_c[df_c["solver"] == "qr_naive"].sort_values("case").reset_index(drop=True)
    c_qr = df_c[df_c["solver"] == "qr"].sort_values("case").reset_index(drop=True)
    r_inc = df_r.sort_values("case").reset_index(drop=True)

    cases = c_naive["case"].values
    ax.plot(cases, c_naive["time_ns"].values, ".", markersize=2, alpha=0.4,
            color=C_QR_NAIVE_COLOR, label="C qr_naive")
    ax.plot(cases, c_qr["time_ns"].values, ".", markersize=2, alpha=0.4,
            color=C_QR_COLOR, label="C qr (incr.)")
    ax.plot(cases, r_inc["time_ns"].values, ".", markersize=2, alpha=0.4,
            color=RUST_COLOR, label="Rust (incr.)")

    ax.set_xlabel("Test case index")
    ax.set_ylabel("Time per solve (ns)")
    ax.set_title("Per-Case Solve Timing")
    ax.legend(markerscale=5, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wls_per_case_timing.png"))
    plt.close(fig)
    print("  per_case_timing.png")


def print_summary(df_c, df_r):
    """Print a markdown-friendly summary table."""
    c_naive = df_c[df_c["solver"] == "qr_naive"]["time_ns"]
    c_qr = df_c[df_c["solver"] == "qr"]["time_ns"]
    r_inc = df_r["time_ns"]

    print("\n## Summary\n")
    print("| Metric | C qr_naive | C qr (incr.) | Rust (incr.) |")
    print("|--------|-----------|-------------|-------------|")
    print(f"| Median (ns) | {c_naive.median():.0f} | {c_qr.median():.0f} | {r_inc.median():.0f} |")
    print(f"| Mean (ns) | {c_naive.mean():.0f} | {c_qr.mean():.0f} | {r_inc.mean():.0f} |")
    print(f"| P95 (ns) | {c_naive.quantile(0.95):.0f} | {c_qr.quantile(0.95):.0f} | {r_inc.quantile(0.95):.0f} |")
    print(f"| P99 (ns) | {c_naive.quantile(0.99):.0f} | {c_qr.quantile(0.99):.0f} | {r_inc.quantile(0.99):.0f} |")
    print(f"| Max (ns) | {c_naive.max():.0f} | {c_qr.max():.0f} | {r_inc.max():.0f} |")
    throughputs = [1e9 / c_naive.median(), 1e9 / c_qr.median(), 1e9 / r_inc.median()]
    print(f"| Throughput | {throughputs[0]/1e6:.2f}M/s | {throughputs[1]/1e6:.2f}M/s | {throughputs[2]/1e6:.2f}M/s |")


def main():
    parser = argparse.ArgumentParser(description="Plot flight-solver CLS benchmarks")
    parser.add_argument("--dir", default=".", help="Directory with CSV files")
    args = parser.parse_args()

    outdir = os.path.join(args.dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    print("Loading data...")
    df_c, df_r = load_data(args.dir)
    print(f"  C: {len(df_c)} rows, Rust: {len(df_r)} rows\n")

    print("Generating plots:")
    plot_throughput(df_c, df_r, outdir)
    plot_timing_distribution(df_c, df_r, outdir)
    plot_accuracy(df_c, df_r, outdir)
    plot_timing_ratio_cdf(df_c, df_r, outdir)
    plot_per_case_timing(df_c, df_r, outdir)

    print_summary(df_c, df_r)
    print(f"\nPlots saved to {outdir}/")


if __name__ == "__main__":
    main()
