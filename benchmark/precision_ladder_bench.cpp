// precision_ladder_bench.cpp
// P2 Step 3 — Precision Ladder Threshold Sweep Benchmark
//
// PURPOSE
// -------
// This benchmark demonstrates the accuracy/speed tradeoff of the Precision
// Ladder by sweeping the residual_threshold over a range and recording:
//   - which stage was selected (GPU_FP32 or CPU_FP64_POLISH)
//   - GPU solve time and residual
//   - CPU polish time (if escalated)
//   - total ladder time
//   - final objective value
//
// The benchmark does NOT claim GPU is faster — timing results are reported
// honestly. The purpose is to demonstrate the intended controller behavior:
//
//   Looser thresholds permit more GPU results to pass the acceptance gate.
//   Stricter thresholds require stronger residual evidence before accepting.
//
// OUTPUT
// ------
// Produces machine-readable CSV to stdout and a human-readable summary.
// Usage:
//   ./precision_ladder_bench [afiro|adlittle|tiny] > bench.csv
//
// INSTANCE
// --------
// Uses the Netlib benchmark instances already in tests/data/, the same
// ones used by test_gpu_pdhg.cpp.  A small synthetic LP is also available
// for environments without GPU hardware.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pramaan/gpu/precision_ladder.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/simplex.hpp"

#ifdef PRAMAAN_ENABLE_CUDA
#include "pramaan/gpu/pdhg_solver.hpp"
#endif

using namespace pramaan;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense,
                             CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> rp, ci;
    std::vector<double> vals;
    rp.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < num_cols; ++c) {
            if (row[static_cast<std::size_t>(c)] != 0.0) {
                ci.push_back(c);
                vals.push_back(row[static_cast<std::size_t>(c)]);
            }
        }
        rp.push_back(static_cast<CSRMatrix::Index>(ci.size()));
    }
    return CSRMatrix(std::move(rp), std::move(ci), std::move(vals), num_cols);
}

// Wyndor Glass Co. (tiny synthetic LP)
static ModelIR makeTiny() {
    return ModelIR(ObjSense::kMaximize, 0.0, {3.0, 5.0},
                   denseToCSR({{1,0},{0,2},{3,2}}, 2),
                   {-kInfinity,-kInfinity,-kInfinity}, {4,12,18},
                   {"r1","r2","r3"}, {0,0}, {kInfinity,kInfinity},
                   {VarType::kContinuous,VarType::kContinuous}, {"x1","x2"});
}

// ---------------------------------------------------------------------------
// BenchRow: one row of the benchmark output
// ---------------------------------------------------------------------------
struct BenchRow {
    std::string instance;
    double      threshold;
    std::string stage;
    std::string escalation_reason;
    double      gpu_time_ms;
    double      cpu_time_ms;
    double      total_time_ms;
    double      gpu_primal_residual;
    double      gpu_dual_residual;
    double      objective;
    int         gpu_iterations;
    bool        final_optimal;
};

// ---------------------------------------------------------------------------
// Run one threshold sweep step
// ---------------------------------------------------------------------------
static BenchRow runOneThreshold(const std::string& instance_name,
                                const ModelIR& model,
                                double threshold,
                                int gpu_max_iter  = 200000,
                                int gpu_check_freq = 500,
                                double gpu_tol    = 1e-5) {
    PrecisionLadder::Options opts;
    opts.residual_threshold  = threshold;
    opts.gpu_max_iterations  = gpu_max_iter;
    opts.gpu_check_frequency = gpu_check_freq;
    opts.gpu_tolerance       = gpu_tol;

    PrecisionLadder ladder(opts);
    PrecisionLadderResult r = ladder.solve(model);

    BenchRow row;
    row.instance             = instance_name;
    row.threshold            = threshold;
    row.stage                = (r.stage == PrecisionLadderResult::Stage::kGpuFp32)
                                 ? "GPU_FP32" : "CPU_FP64_POLISH";
    row.escalation_reason    = r.escalation_reason_str();
    row.gpu_time_ms          = r.gpu_elapsed_ms;
    row.cpu_time_ms          = r.cpu_elapsed_ms;
    row.total_time_ms        = r.total_elapsed_ms;
    row.gpu_primal_residual  = r.gpu_primal_residual;
    row.gpu_dual_residual    = r.gpu_dual_residual;
    row.objective            = r.final_result.objective_value;
    row.gpu_iterations       = r.gpu_iterations;
    row.final_optimal        = (r.final_result.status == SolveStatus::kOptimal);
    return row;
}

// ---------------------------------------------------------------------------
// CPU-only baseline (no GPU stage)
// ---------------------------------------------------------------------------
static BenchRow runCpuOnly(const std::string& instance_name,
                           const ModelIR& model) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    RevisedSimplex solver;
    SolveResult r = solver.solve(model);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    BenchRow row;
    row.instance             = instance_name;
    row.threshold            = 0.0;  // N/A for CPU-only
    row.stage                = "CPU_ONLY";
    row.escalation_reason    = "n/a";
    row.gpu_time_ms          = 0.0;
    row.cpu_time_ms          = ms;
    row.total_time_ms        = ms;
    row.gpu_primal_residual  = std::numeric_limits<double>::infinity();
    row.gpu_dual_residual    = std::numeric_limits<double>::infinity();
    row.objective            = r.objective_value;
    row.gpu_iterations       = 0;
    row.final_optimal        = (r.status == SolveStatus::kOptimal);
    return row;
}

// ---------------------------------------------------------------------------
// Direct GPU baseline (Direct PdhgSolver result, bypasses ladder)
// ---------------------------------------------------------------------------
static BenchRow runGpuOnly(const std::string& instance_name,
                           const ModelIR& model,
                           int gpu_max_iter   = 200000,
                           int gpu_check_freq = 500,
                           double gpu_tol     = 1e-5) {
    BenchRow r;
    r.instance = instance_name;
    r.threshold = std::numeric_limits<double>::infinity();
#ifdef PRAMAAN_ENABLE_CUDA
    gpu::PdhgOptions opts;
    opts.max_iterations = gpu_max_iter;
    opts.check_frequency = gpu_check_freq;
    opts.tolerance = gpu_tol;
    
    gpu::PdhgSolver solver(opts);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    SolveResult res = solver.solve(model);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    r.gpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.cpu_time_ms = 0.0;
    r.total_time_ms = r.gpu_time_ms;
    
    r.stage = "GPU_FP32";
    r.escalation_reason = "n/a";
    r.gpu_primal_residual = res.primal_residual;
    r.gpu_dual_residual = res.dual_residual;
    r.objective = res.objective_value;
    r.gpu_iterations = res.iterations;
    r.final_optimal = (res.status == SolveStatus::kOptimal);
#else
    (void)model; (void)gpu_max_iter; (void)gpu_check_freq; (void)gpu_tol;
    r.stage = "GPU_UNAVAILABLE";
#endif
    return r;
}

// ---------------------------------------------------------------------------
// Print CSV header
// ---------------------------------------------------------------------------
static void printCsvHeader() {
    std::cout << "instance,threshold,stage,escalation_reason,"
                 "gpu_time_ms,cpu_time_ms,total_time_ms,"
                 "gpu_primal_residual,gpu_dual_residual,"
                 "objective,gpu_iterations,final_optimal\n";
}

static void printCsvRow(const BenchRow& row) {
    std::cout << std::setprecision(6) << std::scientific
              << row.instance         << ","
              << row.threshold        << ","
              << row.stage            << ","
              << row.escalation_reason << ","
              << row.gpu_time_ms      << ","
              << row.cpu_time_ms      << ","
              << row.total_time_ms    << ","
              << row.gpu_primal_residual << ","
              << row.gpu_dual_residual   << ","
              << row.objective        << ","
              << row.gpu_iterations   << ","
              << (row.final_optimal ? "1" : "0") << "\n";
}

// ---------------------------------------------------------------------------
// Human-readable summary
// ---------------------------------------------------------------------------
static void printSummary(const std::string& instance_name,
                         const ModelIR& model,
                         const std::vector<BenchRow>& rows) {
    std::cerr << "\n========== " << instance_name << " ==========\n";
    std::cerr << "  m=" << model.numRows() << " n=" << model.numVars()
              << " nnz=" << model.A.nnz() << "\n\n";

#ifndef PRAMAAN_ENABLE_CUDA
    std::cerr << "GPU residual data unavailable: CUDA runtime unavailable.\n";
    std::cerr << "Threshold sweep validates CPU fallback only.\n\n";
#endif

    std::cerr << std::left
              << std::setw(16) << "threshold"
              << std::setw(18) << "stage"
              << std::setw(12) << "total_ms"
              << std::setw(12) << "GPU_ms"
              << std::setw(12) << "CPU_ms"
              << std::setw(14) << "primal_res"
              << std::setw(14) << "dual_res"
              << "objective\n";
    std::cerr << std::string(110, '-') << "\n";

    int num_gpu = 0;
    int num_cpu = 0;
    for (const auto& row : rows) {
        if (row.stage == "GPU_FP32") num_gpu++;
        if (row.stage == "CPU_FP64_POLISH") num_cpu++;
        std::cerr << std::left << std::setprecision(2) << std::scientific
                  << std::setw(16) << row.threshold
                  << std::setw(18) << row.stage
                  << std::setw(12) << std::setprecision(1) << std::fixed << row.total_time_ms
                  << std::setw(12) << row.gpu_time_ms
                  << std::setw(12) << row.cpu_time_ms
                  << std::setprecision(2) << std::scientific
                  << std::setw(14) << row.gpu_primal_residual
                  << std::setw(14) << row.gpu_dual_residual
                  << std::setprecision(8) << row.objective << "\n";
    }
    std::cerr << "\n";

#ifdef PRAMAAN_ENABLE_CUDA
    std::cerr << "Tradeoff summary:\n"
              << "  - GPU accepted runs: " << num_gpu << "\n"
              << "  - CPU escalated runs: " << num_cpu << "\n"
              << "  - No speedup ratio is claimed; see measured times above.\n\n";
#endif
}

// ---------------------------------------------------------------------------
// Sweep a set of thresholds over one instance
// ---------------------------------------------------------------------------
static std::vector<BenchRow> sweepInstance(const std::string& name,
                                           const ModelIR& model,
                                           const std::vector<double>& thresholds) {
    std::vector<BenchRow> rows;

    // CPU-only baseline
    rows.push_back(runCpuOnly(name, model));

    // GPU-only baseline (threshold = +inf)
    rows.push_back(runGpuOnly(name, model));

    // Adaptive sweep
    for (double t : thresholds) {
        rows.push_back(runOneThreshold(name, model, t));
    }
    return rows;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string instance = "afiro";
    if (argc >= 2) instance = argv[1];

    // ---- Load or build the LP instance ----
    ModelIR model;
    if (instance == "tiny") {
        model = makeTiny();
    } else {
        std::string path = "tests/data/" + instance + ".mps";
        try {
            model = parse_mps(path);
        } catch (const std::exception& e) {
            std::cerr << "Cannot load " << path << ": " << e.what()
                      << "\nFalling back to tiny synthetic LP.\n";
            model    = makeTiny();
            instance = "tiny";
        }
    }

    // ---- Threshold sweep values ----
    // These are chosen in decades spanning the range from very loose (accept
    // almost any GPU result) to very strict (force CPU polish).  The actual
    // GPU residual for a well-converged PDHG run is typically ~1e-5 to 1e-7;
    // thresholds around that range demonstrate the acceptance/escalation boundary.
    const std::vector<double> thresholds = {
        1e0,    // loose: accept any kOptimal GPU result with residual <= 1.0
        1e-3,   // medium-loose
        1e-5,   // medium (near GPU tolerance)
        1e-6,   // default (matches PdhgOptions::tolerance)
        1e-8,   // medium-strict
        1e-10,  // very strict threshold; intended to force CPU polish in typical FP32 runs.
        1e-12,  // extremely strict
    };

    // ---- Run sweep ----
    auto rows = sweepInstance(instance, model, thresholds);

    // ---- CSV output (to stdout, so benchmark can be piped to a file) ----
    printCsvHeader();
    for (const auto& row : rows) {
        printCsvRow(row);
    }

    // ---- Human-readable summary (to stderr, so it doesn't pollute CSV) ----
    printSummary(instance, model, rows);

    // ---- Verify at least one row was optimal ----
    bool any_optimal = false;
    for (const auto& row : rows) {
        if (row.final_optimal) { any_optimal = true; break; }
    }
    if (!any_optimal) {
        std::cerr << "WARNING: no benchmark run reached kOptimal status.\n";
        return 1;
    }
    return 0;
}

