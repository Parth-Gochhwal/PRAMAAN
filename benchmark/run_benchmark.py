#!/usr/bin/env python3
"""
run_benchmark.py — PRAMAAN P0 Benchmark Runner

Runs pramaan-solve and HiGHS (via highspy) on the same set of Netlib LP
instances, compares objective values and wall-clock solve times, verifies
PRAMAAN certificates, and prints a comparison table.

IMPORTANT:
- HiGHS is invoked here ONLY as an external benchmark reference via highspy.
- PRAMAAN is invoked via subprocess execution of the actual pramaan-solve binary.
- Nothing in src/ links against HiGHS.
- Both solvers read the same .mps file independently.

Timing methodology:
- PRAMAAN: wall-clock time of subprocess execution (includes process startup).
- HiGHS: wall-clock time around h.run() call only (excludes model read).
  Note: PRAMAAN timing includes process startup + MPS parse + presolve +
  scale + solve + unscale + postsolve + certificate write, while HiGHS
  timing is solve-only. This is disclosed in the output.

Usage:
    cd ~/pramaan
    source venv/bin/activate
    python benchmark/run_benchmark.py
"""
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Find the project root (parent of benchmark/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")

PRAMAAN_SOLVE = os.path.join(BUILD_DIR, "pramaan-solve")
PRAMAAN_VERIFY = os.path.join(BUILD_DIR, "pramaan-verify")

# Objective agreement tolerance. PRAMAAN uses a PFI-based primal simplex
# without production-grade numerical refinement, so we use a relatively
# generous tolerance for agreement checking.
OBJ_TOLERANCE = 1e-4

# ---------------------------------------------------------------------------
# PRAMAAN solver (subprocess)
# ---------------------------------------------------------------------------

def run_pramaan(mps_path: str, cert_path: str) -> dict:
    """Run pramaan-solve on an MPS file, return status/objective/time."""
    result = {
        "status": "ERROR",
        "objective": None,
        "time_s": None,
        "iterations": None,
        "error": None,
    }

    if not os.path.isfile(PRAMAAN_SOLVE):
        result["error"] = f"Binary not found: {PRAMAAN_SOLVE}"
        return result

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [PRAMAAN_SOLVE, mps_path, cert_path],
            capture_output=True, text=True, timeout=120
        )
        t1 = time.perf_counter()
        result["time_s"] = t1 - t0

        output = proc.stdout + proc.stderr

        # Parse status
        m = re.search(r"Status:\s+(\S+)", output)
        if m:
            result["status"] = m.group(1)

        # Parse objective
        m = re.search(r"Objective:\s+([\d.eE+\-]+)", output)
        if m:
            result["objective"] = float(m.group(1))

        # Parse iterations
        m = re.search(r"Iterations:\s+(\d+)", output)
        if m:
            result["iterations"] = int(m.group(1))

        if proc.returncode != 0 and result["status"] == "ERROR":
            result["error"] = output.strip()[-200:] if output.strip() else f"exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["time_s"] = 120.0
        result["status"] = "TIMEOUT"
        result["error"] = "Timed out after 120s"
    except Exception as e:
        result["error"] = str(e)

    return result


def verify_certificate(mps_path: str, cert_path: str) -> str:
    """Run pramaan-verify on an MPS+certificate pair, return VALID/INVALID/ERROR."""
    if not os.path.isfile(PRAMAAN_VERIFY):
        return "ERROR(binary not found)"
    if not os.path.isfile(cert_path):
        return "NO_CERT"

    try:
        proc = subprocess.run(
            [PRAMAAN_VERIFY, mps_path, cert_path],
            capture_output=True, text=True, timeout=30
        )
        output = proc.stdout + proc.stderr
        if "CERTIFICATE VALID" in output:
            return "VALID"
        elif "CERTIFICATE INVALID" in output:
            return "INVALID"
        else:
            return f"ERROR(exit={proc.returncode})"
    except Exception as e:
        return f"ERROR({e})"


# ---------------------------------------------------------------------------
# HiGHS solver (highspy)
# ---------------------------------------------------------------------------

def run_highs(mps_path: str) -> dict:
    """Run HiGHS on an MPS file via highspy, return status/objective/time."""
    result = {
        "status": "ERROR",
        "objective": None,
        "time_s": None,
        "error": None,
    }

    try:
        import highspy
    except ImportError:
        result["error"] = "highspy not installed"
        return result

    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)  # suppress HiGHS output

        # Read model (not timed — we time only the solve)
        read_status = h.readModel(mps_path)
        if read_status != highspy.HighsStatus.kOk:
            result["error"] = f"HiGHS readModel failed: {read_status}"
            return result

        # Solve (timed)
        t0 = time.perf_counter()
        run_status = h.run()
        t1 = time.perf_counter()
        result["time_s"] = t1 - t0

        model_status = h.getModelStatus()
        if model_status == highspy.HighsModelStatus.kOptimal:
            result["status"] = "OPTIMAL"
            info = h.getInfoValue("objective_function_value")
            result["objective"] = info[1]
        elif model_status == highspy.HighsModelStatus.kInfeasible:
            result["status"] = "INFEASIBLE"
        elif model_status == highspy.HighsModelStatus.kUnbounded:
            result["status"] = "UNBOUNDED"
        else:
            result["status"] = str(model_status)

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main():
    # Check prerequisites
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Run ./benchmark/fetch_benchmarks.sh first.")
        sys.exit(1)

    # EXACT 10-instance Stage 8 benchmark dataset
    expected_instances = [
        "afiro",
        "adlittle",
        "kb2",
        "sc50a",
        "sc205",
        "share2b",
        "lotfi",
        "israel",
        "beaconfd",
        "scorpion",
    ]

    mps_files = []
    for name in expected_instances:
        path = os.path.join(DATA_DIR, f"{name}.mps")
        if not os.path.isfile(path):
            print(f"ERROR: Expected benchmark instance missing: {path}")
            print("Run ./benchmark/fetch_benchmarks.sh first to download the correct dataset.")
            sys.exit(1)
        mps_files.append(path)

    if not os.path.isfile(PRAMAAN_SOLVE):
        print(f"ERROR: pramaan-solve not found at {PRAMAAN_SOLVE}")
        print("Run: cmake --build build -j")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import highspy
        highspy_version = highspy.Highs().version()
    except Exception as e:
        highspy_version = f"ERROR: {e}"

    print("=" * 80)
    print("PRAMAAN P0 BENCHMARK — PRAMAAN vs HiGHS on Netlib LP Instances")
    print("=" * 80)
    print()
    print(f"Data directory:   {DATA_DIR}")
    print(f"Instances found:  {len(mps_files)} (strictly verified against canonical 10-instance dataset)")
    print(f"PRAMAAN binary:   {PRAMAAN_SOLVE}")
    print(f"Verifier binary:  {PRAMAAN_VERIFY}")
    print(f"Python env:       {sys.executable}")
    print(f"highspy version:  {highspy_version}")
    print()
    print("Timing Methodology Note:")
    print("  The scopes of the two timing measurements are NOT equivalent.")
    print("  PRAMAAN E2E Time: includes subprocess startup, MPS parsing, presolve,")
    print("                    scaling, simplex solve, unscaling, postsolve, and")
    print("                    certificate generation/writing.")
    print("  HiGHS Solve Time: includes only the `h.run()` call (solve only, excludes")
    print("                    MPS read and process startup).")
    print("  These timings do NOT represent an apples-to-apples solver-speed comparison.")
    print()
    print(f"Objective Agreement criterion:")
    print(f"  relative objective difference <= {OBJ_TOLERANCE}")
    print()

    results = []

    for mps_path in mps_files:
        name = os.path.splitext(os.path.basename(mps_path))[0]
        cert_path = os.path.join(OUTPUT_DIR, f"{name}.cert")

        print(f"  [{name}] ", end="", flush=True)

        # Run PRAMAAN
        pramaan_result = run_pramaan(mps_path, cert_path)

        # Verify certificate if OPTIMAL
        cert_status = "N/A"
        if pramaan_result["status"] == "OPTIMAL":
            cert_status = verify_certificate(mps_path, cert_path)

        # Run HiGHS
        highs_result = run_highs(mps_path)

        # Compute objective difference
        obj_diff = None
        agreement = "N/A"
        if (pramaan_result["status"] == "OPTIMAL" and
                highs_result["status"] == "OPTIMAL" and
                pramaan_result["objective"] is not None and
                highs_result["objective"] is not None):
            obj_diff = abs(pramaan_result["objective"] - highs_result["objective"])
            # Use relative tolerance for large objectives
            ref = max(abs(highs_result["objective"]), 1.0)
            if obj_diff / ref <= OBJ_TOLERANCE:
                agreement = "WITHIN_TOL"
            else:
                agreement = "NO"

        results.append({
            "name": name,
            "pramaan": pramaan_result,
            "highs": highs_result,
            "obj_diff": obj_diff,
            "agreement": agreement,
            "cert_status": cert_status,
        })

        # Progress indicator
        p_status = pramaan_result["status"]
        h_status = highs_result["status"]
        print(f"PRAMAAN={p_status}  HiGHS={h_status}  cert={cert_status}  agree={agreement}")

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 168)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 168)

    header = (
        f"{'Instance':<12} | "
        f"{'PRAMAAN Status':<16} | "
        f"{'PRAMAAN Obj':>16} | "
        f"{'PRAMAAN E2E Time(s)':>19} | "
        f"{'HiGHS Status':<14} | "
        f"{'HiGHS Obj':>16} | "
        f"{'HiGHS Solve Time(s)':>19} | "
        f"{'Obj Diff':>12} | "
        f"{'Agree':>10} | "
        f"{'Cert':>7}"
    )
    print(header)
    print("-" * 168)

    for r in results:
        p = r["pramaan"]
        h = r["highs"]

        p_obj = f"{p['objective']:.6f}" if p['objective'] is not None else "—"
        h_obj = f"{h['objective']:.6f}" if h['objective'] is not None else "—"
        p_time = f"{p['time_s']:.4f}" if p['time_s'] is not None else "—"
        h_time = f"{h['time_s']:.4f}" if h['time_s'] is not None else "—"
        obj_diff = f"{r['obj_diff']:.2e}" if r['obj_diff'] is not None else "—"

        row = (
            f"{r['name']:<12} | "
            f"{p['status']:<16} | "
            f"{p_obj:>16} | "
            f"{p_time:>19} | "
            f"{h['status']:<14} | "
            f"{h_obj:>16} | "
            f"{h_time:>19} | "
            f"{obj_diff:>12} | "
            f"{r['agreement']:>10} | "
            f"{r['cert_status']:>7}"
        )
        print(row)

    print("-" * 168)
    print()
    print("Methodology Note: PRAMAAN E2E Time includes parsing, presolve, and certificate generation.")
    print("HiGHS Solve Time includes solver execution only. DO NOT calculate speedups from these numbers.")
    print("-" * 168)

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    pramaan_optimal = sum(1 for r in results if r["pramaan"]["status"] == "OPTIMAL")
    highs_optimal = sum(1 for r in results if r["highs"]["status"] == "OPTIMAL")
    both_optimal = sum(1 for r in results
                       if r["pramaan"]["status"] == "OPTIMAL"
                       and r["highs"]["status"] == "OPTIMAL")
    agreements = sum(1 for r in results if r["agreement"] == "WITHIN_TOL")
    disagreements = sum(1 for r in results if r["agreement"] == "NO")
    pramaan_failures = sum(1 for r in results if r["pramaan"]["status"] not in ("OPTIMAL",))
    highs_failures = sum(1 for r in results if r["highs"]["status"] not in ("OPTIMAL",))
    cert_valid = sum(1 for r in results if r["cert_status"] == "VALID")
    cert_invalid = sum(1 for r in results if r["cert_status"] == "INVALID")

    print()
    print("SUMMARY")
    print(f"  Total instances:          {len(results)}")
    print(f"  PRAMAAN OPTIMAL:          {pramaan_optimal}")
    print(f"  HiGHS OPTIMAL:            {highs_optimal}")
    print(f"  Both OPTIMAL:             {both_optimal}")
    print(f"  Objective values within relative tolerance: {agreements}/{len(results)}")
    print(f"  Objective disagreements:   {disagreements}")
    print(f"  PRAMAAN failures:          {pramaan_failures}")
    print(f"  HiGHS failures:            {highs_failures}")
    print(f"  Certificates VALID:        {cert_valid}")
    print(f"  Certificates INVALID:      {cert_invalid}")
    print()

    # Report any disagreements
    if disagreements > 0:
        print("WARNING: Objective disagreements detected!")
        for r in results:
            if r["agreement"] == "NO":
                p = r["pramaan"]
                h = r["highs"]
                print(f"  {r['name']}: PRAMAAN={p['objective']}  HiGHS={h['objective']}  diff={r['obj_diff']}")
        print()

    # Report any errors
    errors = [r for r in results if r["pramaan"].get("error")]
    if errors:
        print("PRAMAAN ERRORS:")
        for r in errors:
            print(f"  {r['name']}: {r['pramaan']['error']}")
        print()

    print("Benchmark complete.")
    return 0 if disagreements == 0 and pramaan_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
