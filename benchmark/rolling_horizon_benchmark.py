#!/usr/bin/env python3
"""
rolling_horizon_benchmark.py — PRAMAAN P1 Step 8 Rolling-Horizon Benchmark

Thin Python wrapper around the C++ rolling_horizon_bench executable.
Handles build verification, argument forwarding, and provides a familiar
Python CLI interface consistent with the existing benchmark/run_benchmark.py.

Usage:
    python benchmark/rolling_horizon_benchmark.py [--days 30] [--seed 12345] [--json]

This script does NOT implement the benchmark logic — it invokes the C++
binary which uses the actual DualSimplex::warmSolve API in-process.
"""
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BENCHMARK_BINARY = os.path.join(BUILD_DIR, "rolling_horizon_bench")


def find_binary():
    """Locate the benchmark binary, checking common build directories."""
    candidates = [
        BENCHMARK_BINARY,
        os.path.join(PROJECT_ROOT, "build-cpu-final", "rolling_horizon_bench"),
        os.path.join(PROJECT_ROOT, "build-gpu-final", "rolling_horizon_bench"),
        os.path.join(PROJECT_ROOT, "build", "Release", "rolling_horizon_bench"),
        os.path.join(PROJECT_ROOT, "build", "Debug", "rolling_horizon_bench"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main():
    binary = find_binary()

    if binary is None:
        print(f"ERROR: rolling_horizon_bench binary not found.")
        print(f"  Expected at: {BENCHMARK_BINARY}")
        print()
        print("Build it with:")
        print(f"  cmake -S {PROJECT_ROOT} -B {BUILD_DIR} -DCMAKE_BUILD_TYPE=Release")
        print(f"  cmake --build {BUILD_DIR} -j$(nproc)")
        sys.exit(1)

    # Forward all arguments to the C++ binary
    cmd = [binary] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, timeout=600)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out after 600 seconds.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
