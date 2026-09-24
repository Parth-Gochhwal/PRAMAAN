#!/usr/bin/env python3
"""
rolling_horizon.py — PRAMAAN P1 Rolling-Horizon Warm-Start Benchmark

30-Day Rolling-Horizon Re-Solve: Cold Restart vs PRAMAAN Warm Start

This script:
  1. Invokes the rolling_horizon_mps_bench C++ binary which uses the ACTUAL
     DualSimplex::captureBasis / DualSimplex::warmSolve API in-process.
  2. Processes JSON results from the C++ binary.
  3. Generates rolling_horizon_results.csv with per-day metrics.
  4. Generates rolling_horizon.png — a publication-quality cumulative timing plot.

The C++ binary does ALL solve work. This Python script does NO solving.
Both cold and warm paths use the same in-process solver API — the only
difference is whether the previous day's simplex basis is reused.

Timing boundary: wall-clock time around RevisedSimplex::solve() (cold)
and DualSimplex::warmSolve() (warm) calls, measured in C++ with
std::chrono::steady_clock. Excludes scenario generation and MPS parsing.

Usage:
    python benchmark/rolling_horizon.py
    python benchmark/rolling_horizon.py --days 30 --seed 42
    python benchmark/rolling_horizon.py --instance benchmark/mrpl_crude_blend.mps --days 3

Prerequisites:
    The rolling_horizon_mps_bench binary must be built:
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build -j
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_MPS = os.path.join(SCRIPT_DIR, "mrpl_crude_blend.mps")
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "rolling_horizon_results.csv")
DEFAULT_PLOT = os.path.join(SCRIPT_DIR, "rolling_horizon.png")


def find_binary():
    """Locate the rolling_horizon_mps_bench binary."""
    # Check common build directories
    candidates = [
        os.path.join(PROJECT_ROOT, "build", "rolling_horizon_mps_bench"),
        os.path.join(PROJECT_ROOT, "build", "rolling_horizon_mps_bench.exe"),
        os.path.join(PROJECT_ROOT, "build", "Release", "rolling_horizon_mps_bench"),
        os.path.join(PROJECT_ROOT, "build", "Release", "rolling_horizon_mps_bench.exe"),
        os.path.join(PROJECT_ROOT, "build", "Debug", "rolling_horizon_mps_bench"),
        os.path.join(PROJECT_ROOT, "build", "Debug", "rolling_horizon_mps_bench.exe"),
        os.path.join(PROJECT_ROOT, "build-cpu-final", "rolling_horizon_mps_bench"),
        os.path.join(PROJECT_ROOT, "build-gpu-final", "rolling_horizon_mps_bench"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Run the C++ benchmark binary
# ---------------------------------------------------------------------------

def run_benchmark(binary, mps_path, days, seed, perturb_rhs, perturb_bounds,
                  tolerance):
    """Invoke the C++ benchmark and return parsed JSON results."""
    cmd = [
        binary, mps_path,
        "--days", str(days),
        "--seed", str(seed),
        "--perturb-rhs", str(perturb_rhs),
        "--perturb-bounds", str(perturb_bounds),
        "--tolerance", str(tolerance),
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out after 600 seconds.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    # Print stderr (progress info) to console
    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0:
        print(f"\nERROR: Benchmark exited with code {proc.returncode}",
              file=sys.stderr)
        if proc.stdout:
            print("STDOUT:", proc.stdout[:2000], file=sys.stderr)
        sys.exit(1)

    # Parse JSON from stdout
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON output: {e}", file=sys.stderr)
        print(f"Raw stdout (first 2000 chars): {proc.stdout[:2000]}",
              file=sys.stderr)
        sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Generate CSV
# ---------------------------------------------------------------------------

def write_csv(data, csv_path):
    """Write per-day benchmark results to CSV."""
    fieldnames = [
        "day",
        "perturbation_description",
        "cold_time_ms",
        "warm_time_ms",
        "cold_cumulative_ms",
        "warm_cumulative_ms",
        "cold_objective",
        "warm_objective",
        "objective_abs_diff",
        "cold_iterations",
        "warm_iterations",
        "cold_status",
        "warm_status",
        "cold_primal_residual",
        "warm_primal_residual",
        "basis_reused",
    ]

    cold_cum = 0.0
    warm_cum = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for day_data in data["per_day"]:
            cold_cum += day_data["cold_time_ms"]
            warm_cum += day_data["warm_time_ms"]

            writer.writerow({
                "day": day_data["day"],
                "perturbation_description": day_data["perturbation"],
                "cold_time_ms": f"{day_data['cold_time_ms']:.6f}",
                "warm_time_ms": f"{day_data['warm_time_ms']:.6f}",
                "cold_cumulative_ms": f"{cold_cum:.6f}",
                "warm_cumulative_ms": f"{warm_cum:.6f}",
                "cold_objective": f"{day_data['cold_objective']:.10f}",
                "warm_objective": f"{day_data['warm_objective']:.10f}",
                "objective_abs_diff": f"{day_data['objective_abs_diff']:.12f}",
                "cold_iterations": day_data["cold_iterations"],
                "warm_iterations": day_data["warm_iterations"],
                "cold_status": day_data["cold_status"],
                "warm_status": day_data["warm_status"],
                "cold_primal_residual": f"{day_data['cold_primal_residual']:.12f}",
                "warm_primal_residual": f"{day_data['warm_primal_residual']:.12f}",
                "basis_reused": day_data["basis_reused"],
            })

    print(f"CSV written: {csv_path}")


# ---------------------------------------------------------------------------
# Generate plot
# ---------------------------------------------------------------------------

def generate_plot(data, plot_path):
    """Generate a publication-quality cumulative timing plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("ERROR: matplotlib is required for plot generation.",
              file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    per_day = data["per_day"]
    days = [d["day"] + 1 for d in per_day]  # 1-indexed for display

    # Compute cumulative times
    cold_cum = []
    warm_cum = []
    cold_running = 0.0
    warm_running = 0.0
    for d in per_day:
        cold_running += d["cold_time_ms"]
        warm_running += d["warm_time_ms"]
        cold_cum.append(cold_running)
        warm_cum.append(warm_running)

    # Compute iteration counts per day for secondary info
    cold_iters = [d["cold_iterations"] for d in per_day]
    warm_iters = [d["warm_iterations"] for d in per_day]
    basis_reused = [d["basis_reused"] for d in per_day]

    num_days = data["days"]
    ratio = data["cumulative_ratio"]
    total_cold_ms = data["cold_cumulative_ms"]
    total_warm_ms = data["warm_cumulative_ms"]
    ws_successes = data["warm_start_successes"]
    ws_attempts = data["warm_start_attempts"]

    # --- Create figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                     gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")

    # --- Main plot: Cumulative timing ---
    ax1.set_facecolor("#0d1117")

    # Plot curves
    ax1.plot(days, cold_cum, color="#ff6b6b", linewidth=2.5, marker="o",
             markersize=4, label="Cold restart every day", zorder=3)
    ax1.plot(days, warm_cum, color="#51cf66", linewidth=2.5, marker="s",
             markersize=4, label="PRAMAAN warm-start", zorder=3)

    # Fill between
    ax1.fill_between(days, warm_cum, cold_cum, alpha=0.15, color="#51cf66",
                     zorder=1)

    # Annotate final values
    ax1.annotate(
        f"Cold: {total_cold_ms:.1f} ms",
        xy=(days[-1], cold_cum[-1]),
        xytext=(days[-1] - num_days * 0.25, cold_cum[-1] * 1.05),
        fontsize=10, color="#ff6b6b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5),
    )
    ax1.annotate(
        f"Warm: {total_warm_ms:.1f} ms",
        xy=(days[-1], warm_cum[-1]),
        xytext=(days[-1] - num_days * 0.25, warm_cum[-1] * 0.7),
        fontsize=10, color="#51cf66", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#51cf66", lw=1.5),
    )

    # Labels and title
    ax1.set_xlabel("Simulated Day", fontsize=12, color="white")
    ax1.set_ylabel("Cumulative Wall-Clock Time (ms)", fontsize=12, color="white")
    ax1.set_title(
        "30-Day Rolling-Horizon Re-Solve:\nCold Restart vs PRAMAAN Warm Start",
        fontsize=14, fontweight="bold", color="white", pad=15,
    )

    # Subtitle
    ax1.text(
        0.5, 1.02,
        "Same LP structure, small daily RHS/bounds changes, cached simplex basis reused",
        transform=ax1.transAxes, fontsize=9, color="#8b949e",
        ha="center", va="bottom",
    )

    ax1.legend(loc="upper left", fontsize=10, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#30363d")
    ax1.spines["left"].set_color("#30363d")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.15, color="white")

    # Summary box
    summary_text = (
        f"Day {num_days}\n"
        f"Cold:  {total_cold_ms:.1f} ms  ({data['cold_total_iters']} iters)\n"
        f"Warm:  {total_warm_ms:.1f} ms  ({data['warm_total_iters']} iters)\n"
        f"Measured ratio: {ratio:.2f}×\n"
        f"Basis reuse: {ws_successes}/{ws_attempts} days"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                 edgecolor="#51cf66", alpha=0.9)
    ax1.text(0.02, 0.95, summary_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", color="white", bbox=props,
             family="monospace")

    # --- Secondary plot: Per-day iterations ---
    ax2.set_facecolor("#0d1117")

    bar_width = 0.35
    x_pos = [d - bar_width / 2 for d in days]
    x_pos_warm = [d + bar_width / 2 for d in days]

    ax2.bar(x_pos, cold_iters, bar_width, color="#ff6b6b", alpha=0.7,
            label="Cold iterations")
    ax2.bar(x_pos_warm, warm_iters, bar_width, color="#51cf66", alpha=0.7,
            label="Warm iterations")

    # Mark basis reuse
    for i, reused in enumerate(basis_reused):
        if reused:
            ax2.annotate("✓", xy=(days[i] + bar_width / 2, warm_iters[i]),
                         fontsize=7, color="#51cf66", ha="center",
                         va="bottom")

    ax2.set_xlabel("Simulated Day", fontsize=10, color="white")
    ax2.set_ylabel("Simplex Iterations", fontsize=10, color="white")
    ax2.legend(loc="upper right", fontsize=8, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.15, color="white", axis="y")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close()

    print(f"Plot written: {plot_path}")


# ---------------------------------------------------------------------------
# Print summary report
# ---------------------------------------------------------------------------

def print_report(data):
    """Print a human-readable summary report."""
    print()
    print("=" * 72)
    print("  PRAMAAN Rolling-Horizon Warm-Start Benchmark Report")
    print("=" * 72)
    print()
    print(f"Instance:           {data['instance']}")
    print(f"Variables:          {data['num_vars']}")
    print(f"Constraints:        {data['num_rows']}")
    print(f"Days:               {data['days']}")
    print(f"Seed:               {data['seed']}")
    print(f"Perturbation:       ±{data['perturb_rhs']*100:.0f}% RHS / "
          f"±{data['perturb_bounds']*100:.0f}% bounds")
    print()
    print("--- Cumulative Timing ---")
    print(f"Cold cumulative:    {data['cold_cumulative_ms']:.3f} ms")
    print(f"Warm cumulative:    {data['warm_cumulative_ms']:.3f} ms")
    print(f"Measured ratio:     {data['cumulative_ratio']:.2f}×")
    print()
    print("--- Iteration Counts ---")
    print(f"Cold total iters:   {data['cold_total_iters']}")
    print(f"Warm total iters:   {data['warm_total_iters']}")
    print(f"Iteration ratio:    {data['iteration_ratio']:.2f}×")
    print()
    print("--- Warm-Start Statistics ---")
    print(f"Warm-start attempts:   {data['warm_start_attempts']}")
    print(f"Warm-start successes:  {data['warm_start_successes']}")
    print(f"Warm-start fallbacks:  {data['warm_start_fallbacks']}")
    print()
    print("--- Correctness ---")
    print(f"All days correct:   {'YES' if data['correctness'] else 'NO'}")
    print(f"Cold optimal days:  {data['cold_optimal_days']}/{data['days']}")
    print(f"Warm optimal days:  {data['warm_optimal_days']}/{data['days']}")
    print(f"Benchmark valid:    {'YES' if data['benchmark_valid'] else 'NO'}")
    print()

    # Per-day table
    print("--- Per-Day Detail ---")
    header = (f"{'Day':>4}  {'Cold(ms)':>10}  {'Warm(ms)':>10}  "
              f"{'Cold It':>8}  {'Warm It':>8}  {'Basis':>6}  "
              f"{'Cold Obj':>14}  {'Warm Obj':>14}  {'Diff':>12}")
    print(header)
    print("-" * len(header))

    for d in data["per_day"]:
        basis_str = "reuse" if d["basis_reused"] else ("cold" if d["day"] == 0 else "fall")
        print(f"{d['day']:4d}  {d['cold_time_ms']:10.4f}  {d['warm_time_ms']:10.4f}  "
              f"{d['cold_iterations']:8d}  {d['warm_iterations']:8d}  {basis_str:>6}  "
              f"{d['cold_objective']:14.4f}  {d['warm_objective']:14.4f}  "
              f"{d['objective_abs_diff']:12.2e}")

    print()
    print("=" * 72)
    print(f"  Measured cumulative ratio: {data['cumulative_ratio']:.2f}×")
    print(f"  (cold_total / warm_total, measured from wall-clock data)")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PRAMAAN P1 Rolling-Horizon Warm-Start Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This benchmark measures the ACTUAL cumulative cold-vs-warm-start performance
of PRAMAAN's DualSimplex::warmSolve implementation over a simulated 30-day
rolling-horizon crude-blending LP trace.

The C++ binary does all solve work using the real PRAMAAN solver APIs.
This Python script invokes it, processes results, and generates outputs.
""")

    parser.add_argument("--days", type=int, default=30,
                        help="Number of simulated days (default: 30)")
    parser.add_argument("--instance", type=str, default=DEFAULT_MPS,
                        help=f"Path to base MPS file (default: {DEFAULT_MPS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic perturbations (default: 42)")
    parser.add_argument("--perturb-rhs", type=float, default=0.08,
                        help="RHS perturbation magnitude (default: 0.08)")
    parser.add_argument("--perturb-bounds", type=float, default=0.06,
                        help="Bounds perturbation magnitude (default: 0.06)")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="Solver tolerance (default: 1e-8)")
    parser.add_argument("--output", type=str, default=DEFAULT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--plot", type=str, default=DEFAULT_PLOT,
                        help=f"Output plot path (default: {DEFAULT_PLOT})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")

    args = parser.parse_args()

    # --- Locate binary ---
    binary = find_binary()
    if binary is None:
        print("ERROR: rolling_horizon_mps_bench binary not found.")
        print()
        print("Build it with:")
        print(f"  cmake -S {PROJECT_ROOT} -B {os.path.join(PROJECT_ROOT, 'build')} "
              f"-DCMAKE_BUILD_TYPE=Release")
        print(f"  cmake --build {os.path.join(PROJECT_ROOT, 'build')} -j")
        sys.exit(1)

    print(f"Binary: {binary}")

    # --- Verify MPS file exists ---
    if not os.path.isfile(args.instance):
        print(f"ERROR: MPS file not found: {args.instance}")
        sys.exit(1)

    print(f"Instance: {args.instance}")
    print(f"Days: {args.days}")
    print(f"Seed: {args.seed}")
    print(f"Perturbation: ±{args.perturb_rhs*100:.0f}% RHS / "
          f"±{args.perturb_bounds*100:.0f}% bounds")
    print()

    # --- Run benchmark ---
    data = run_benchmark(
        binary=binary,
        mps_path=args.instance,
        days=args.days,
        seed=args.seed,
        perturb_rhs=args.perturb_rhs,
        perturb_bounds=args.perturb_bounds,
        tolerance=args.tolerance,
    )

    # --- Write CSV ---
    write_csv(data, args.output)

    # --- Generate plot ---
    if not args.no_plot:
        generate_plot(data, args.plot)

    # --- Print report ---
    print_report(data)

    # --- Exit code reflects correctness ---
    if not data["benchmark_valid"]:
        print("WARNING: Benchmark is NOT valid — see correctness failures above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())