#!/usr/bin/env python3
import sys
import os

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def plot_tradeoff(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib or pandas not installed. Cannot generate tradeoff plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter out baselines
    baselines = df[df['threshold'].isin([0.0, float('inf')])]
    sweep = df[~df['threshold'].isin([0.0, float('inf')])]

    if sweep.empty:
        print("Error: No valid sweep data found.", file=sys.stderr)
        sys.exit(1)

    # GPU evidence check: optimal GPU sweep rows
    gpu_sweep_optimal = sweep[(sweep['stage'] == 'GPU_FP32') & (sweep['final_optimal'] == 1)]
    if gpu_sweep_optimal.empty:
        print("Error: No optimal GPU_FP32 runs found in sweep. A valid CUDA GPU run is required.", file=sys.stderr)
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Split into GPU accepted vs CPU polished
    gpu_mask = (sweep['stage'] == 'GPU_FP32') & (sweep['final_optimal'] == 1)
    cpu_mask = sweep['stage'] == 'CPU_FP64_POLISH'

    # PANEL 1: Total Time
    ax1.plot(sweep['threshold'], sweep['total_time_ms'], 'k-', alpha=0.5, label='Precision Ladder')
    
    if gpu_mask.any():
        ax1.scatter(sweep[gpu_mask]['threshold'], sweep[gpu_mask]['total_time_ms'], 
                    color='green', marker='o', s=100, label='GPU FP32 Accepted')
    
    if cpu_mask.any():
        ax1.scatter(sweep[cpu_mask]['threshold'], sweep[cpu_mask]['total_time_ms'], 
                    color='red', marker='x', s=100, label='CPU FP64 Polished')

    # Baselines
    cpu_only = baselines[baselines['stage'] == 'CPU_ONLY']
    if not cpu_only.empty:
        ax1.axhline(y=cpu_only['total_time_ms'].iloc[0], color='blue', linestyle='--', label='Direct CPU FP64 baseline')
    
    gpu_only = baselines[(baselines['threshold'] == float('inf')) & (baselines['stage'] == 'GPU_FP32') & (baselines['final_optimal'] == 1)]
    if not gpu_only.empty:
        ax1.axhline(y=gpu_only['total_time_ms'].iloc[0], color='orange', linestyle='--', label='Direct GPU FP32 baseline')

    ax1.set_ylabel('Total Time (ms)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # PANEL 2: Residuals
    ax2.plot(sweep['threshold'], sweep['gpu_primal_residual'], color='purple', marker='^', linestyle=':', label='GPU Primal Residual')
    ax2.plot(sweep['threshold'], sweep['gpu_dual_residual'], color='brown', marker='v', linestyle=':', label='GPU Dual Residual')
    ax2.plot(sweep['threshold'], sweep['threshold'], color='black', linestyle='--', alpha=0.7, label='Acceptance Threshold')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Residual Threshold (log scale, loose to strict $\\rightarrow$)')
    ax2.set_ylabel('Residual Magnitude')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.invert_xaxis()  # loose threshold on left, strict on right

    plt.suptitle(f"Precision Ladder Tradeoff: {sweep['instance'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark/precision_ladder_bench_cpu_only.csv"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "benchmark/precision_ladder_tradeoff.png"
    plot_tradeoff(csv_file, out_file)
