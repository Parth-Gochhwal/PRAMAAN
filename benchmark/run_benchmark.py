#!/usr/bin/env python3
"""
run_benchmark.py

Runs pramaan-solve and HiGHS (via `pip install highspy`, or the HiGHS CLI)
on the same set of instances in benchmark/data/, compares objective values
and solve times, and prints a table.

IMPORTANT: HiGHS is invoked here as an external, separate process/library
call purely for COMPARISON. Nothing in src/ links against HiGHS -- keep it
that way, this script is the only place HiGHS is even installed.

FIRST TASK: get this running on a single instance (afiro) comparing just
objective value and wall-clock time. Add the degeneracy/ill-conditioning
stress-test set and the residual/robustness columns once the basic
harness works.
"""
import subprocess
import time
import sys


def run_pramaan(mps_path: str):
    # TODO: subprocess.run(["../build/pramaan-solve", mps_path], ...)
    raise NotImplementedError


def run_highs(mps_path: str):
    # TODO: import highspy; h = highspy.Highs(); h.readModel(mps_path); ...
    raise NotImplementedError


if __name__ == "__main__":
    print("run_benchmark.py: not yet implemented")
