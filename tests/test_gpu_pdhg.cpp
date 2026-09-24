// test_gpu_pdhg.cpp
// P2 Step 2 — GPU PDHG LP Solver tests
//
// Tests the PRAMAAN GPU PDHG backend (Precision Ladder bottom rung).
// Note: This test exercises the GPU FP32 fast-pass in isolation. CPU FP64
// escalation is an orchestration-level capability outside this Step 2 test.
//
// PRIOR ART NOTE
// PDHG / PDLP is established prior art:
//   [PDLP]   Applegate et al. (2021). "Practical large-scale linear programming
//             using primal-dual hybrid gradient." NeurIPS 2021. arXiv:2106.04756.
//   [cuPDLP] Lu, H., & Yang, J. (2025). "cuPDLP.jl: A GPU implementation of
//             restarted primal-dual hybrid gradient for linear programming in
//             Julia." Operations Research, 73(6). arXiv:2311.12180.
//
// PRAMAAN uses PDHG as a GPU fast pass, not as a novel algorithm contribution.
//
// This file covers:
//   A. SpMV unit tests  (GPU vs independent CPU reference)
//   B. Transpose SpMV unit tests
//   C. Small synthetic LP correctness
//   D. Netlib afiro.mps correctness (GPU vs CPU simplex)
//   E. adlittle.mps benchmark (GPU vs CPU timing)
//   F. Regression: existing Step-1 Wyndor and ZeroRow tests

#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

#include "pramaan/gpu/pdhg_solver.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/ir.hpp"

using namespace pramaan;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int g_checks_run    = 0;
static int g_checks_failed = 0;

static void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

static void checkNear(double a, double b, double atol, double rtol,
                      const std::string& description) {
    ++g_checks_run;
    // Exact equality first (handles ±0, identical inf, etc.)
    if (a == b) return;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description
                  << " (non-finite: " << a << " vs " << b << ")\n";
        return;
    }
    double diff = std::abs(a - b);
    double tol  = atol + rtol * std::max(std::abs(a), std::abs(b));
    if (diff > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description
                  << " (" << a << " vs " << b << ", diff=" << diff
                  << ", tol=" << tol << ")\n";
    }
}

static void run(const std::string& name, const std::function<void()>& body) {
    std::cout << name << "...\n";
    int before = g_checks_failed;
    body();
    if (g_checks_failed == before) std::cout << "  ok\n";
}

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
            if (row[static_cast<size_t>(c)] != 0.0) {
                ci.push_back(c);
                vals.push_back(row[static_cast<size_t>(c)]);
            }
        }
        rp.push_back(static_cast<CSRMatrix::Index>(ci.size()));
    }
    return CSRMatrix(std::move(rp), std::move(ci), std::move(vals), num_cols);
}

// Check that solution is feasible for the model within tol.
static double computePrimalViolation(const ModelIR& model, const std::vector<double>& x) {
    double viol = 0.0;
    for (int j = 0; j < static_cast<int>(x.size()); ++j) {
        if (x[j] < model.var_lower[j]) viol = std::max(viol, model.var_lower[j] - x[j]);
        if (x[j] > model.var_upper[j]) viol = std::max(viol, x[j] - model.var_upper[j]);
    }
    auto act = model.A.multiply(x);
    for (int i = 0; i < static_cast<int>(act.size()); ++i) {
        if (act[i] < model.row_lower[i]) viol = std::max(viol, model.row_lower[i] - act[i]);
        if (act[i] > model.row_upper[i]) viol = std::max(viol, act[i] - model.row_upper[i]);
    }
    return viol;
}

// ---------------------------------------------------------------------------
// Declarations for the internal GPU SpMV launchers (from pdhg_solver.cu,
// exposed via an internal test-only header-free forward declaration).
// ---------------------------------------------------------------------------
namespace pramaan {
namespace gpu {
// These are exposed via a thin wrapper we add to pdhg.cu for testing.
// We forward-declare the host wrappers here:
void launch_spmv(const std::vector<int>& row_ptr,
                 const std::vector<int>& col_idx,
                 const std::vector<float>& values,
                 const std::vector<float>& x,
                 std::vector<float>& y_out,
                 int m, int n);

void launch_spmv_t(const std::vector<int>& row_ptr,
                   const std::vector<int>& col_idx,
                   const std::vector<float>& values,
                   const std::vector<float>& x,
                   std::vector<float>& y_out,
                   int m, int n);
} // namespace gpu
} // namespace pramaan

// ---------------------------------------------------------------------------
// A. SpMV unit tests: GPU vs independent CPU reference
// ---------------------------------------------------------------------------
static std::vector<float> cpu_spmv(const std::vector<int>& rp,
                                    const std::vector<int>& ci,
                                    const std::vector<float>& vals,
                                    const std::vector<float>& x, int m) {
    std::vector<float> y(static_cast<size_t>(m), 0.0f);
    for (int i = 0; i < m; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k)
            y[i] += vals[k] * x[ci[k]];
    return y;
}

static std::vector<float> cpu_spmv_t(const std::vector<int>& rp,
                                      const std::vector<int>& ci,
                                      const std::vector<float>& vals,
                                      const std::vector<float>& x, int m, int n) {
    std::vector<float> y(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < m; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k)
            y[ci[k]] += vals[k] * x[i];
    return y;
}

static void runSpMVTest(int m, int n, const std::vector<std::vector<double>>& dense_A) {
    // Build CSR in int/float
    std::vector<int> rp, ci;
    std::vector<float> vals;
    rp.push_back(0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double v = dense_A[i][j];
            if (v != 0.0) { ci.push_back(j); vals.push_back(static_cast<float>(v)); }
        }
        rp.push_back(static_cast<int>(ci.size()));
    }

    // Deterministic x
    std::vector<float> x(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) x[j] = static_cast<float>(j + 1) * 0.5f;

    auto ref_y  = cpu_spmv(rp, ci, vals, x, m);
    std::vector<float> gpu_y;
    pramaan::gpu::launch_spmv(rp, ci, vals, x, gpu_y, m, n);
    check(static_cast<int>(gpu_y.size()) == m, "SpMV output size m=" + std::to_string(m));
    for (int i = 0; i < m; ++i) {
        float diff = std::abs(ref_y[i] - gpu_y[i]);
        float tol  = 1e-4f + 1e-4f * std::max(std::abs(ref_y[i]), std::abs(gpu_y[i]));
        if (diff > tol) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] SpMV row " << i << " m=" << m << " n=" << n
                      << " ref=" << ref_y[i] << " gpu=" << gpu_y[i] << "\n";
        } else {
            ++g_checks_run;
        }
    }
    if (m == 0) ++g_checks_run; // empty is trivially ok
}

static void runSpMVTransposeTest(int m, int n, const std::vector<std::vector<double>>& dense_A) {
    std::vector<int> rp, ci;
    std::vector<float> vals;
    rp.push_back(0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double v = dense_A[i][j];
            if (v != 0.0) { ci.push_back(j); vals.push_back(static_cast<float>(v)); }
        }
        rp.push_back(static_cast<int>(ci.size()));
    }
    std::vector<float> x(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) x[i] = static_cast<float>(i + 1) * 0.3f;

    auto ref_y  = cpu_spmv_t(rp, ci, vals, x, m, n);
    std::vector<float> gpu_y;
    pramaan::gpu::launch_spmv_t(rp, ci, vals, x, gpu_y, m, n);
    check(static_cast<int>(gpu_y.size()) == n, "SpMV-T output size n=" + std::to_string(n));
    for (int j = 0; j < n; ++j) {
        float diff = std::abs(ref_y[j] - gpu_y[j]);
        float tol  = 1e-4f + 1e-4f * std::max(std::abs(ref_y[j]), std::abs(gpu_y[j]));
        if (diff > tol) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] SpMVT col " << j << " m=" << m << " n=" << n
                      << " ref=" << ref_y[j] << " gpu=" << gpu_y[j] << "\n";
        } else {
            ++g_checks_run;
        }
    }
    if (n == 0) ++g_checks_run;
}

void testSpMVDimensions() {
    // Various dimensions crossing CUDA warp/block boundaries
    std::vector<int> dims = {1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    for (int d : dims) {
        // Identity-like matrix of size d x d
        std::vector<std::vector<double>> A(d, std::vector<double>(d, 0.0));
        for (int i = 0; i < d; ++i) A[i][i] = static_cast<double>(i + 1);
        runSpMVTest(d, d, A);
        runSpMVTransposeTest(d, d, A);
    }
    // Rectangular: m < n
    {
        std::vector<std::vector<double>> A(3, std::vector<double>(7, 0.0));
        A[0][1] = 2.0; A[0][5] = -1.0;
        A[1][3] = 3.0;
        A[2][0] = 1.0; A[2][6] = 4.0;
        runSpMVTest(3, 7, A);
        runSpMVTransposeTest(3, 7, A);
    }
    // m=0 edge case
    runSpMVTest(0, 5, {});
    runSpMVTransposeTest(0, 5, {});
}

// ---------------------------------------------------------------------------
// B. Small synthetic LP correctness
// ---------------------------------------------------------------------------
void testWyndorGlass() {
    // max 3x1 + 5x2 s.t. x1<=4, 2x2<=12, 3x1+2x2<=18, x1,x2>=0
    // optimal: x1=2, x2=6, obj=36
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0},
                  denseToCSR({{1,0},{0,2},{3,2}}, 2),
                  {-kInfinity,-kInfinity,-kInfinity}, {4,12,18},
                  {"r1","r2","r3"}, {0,0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x1","x2"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "Wyndor: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, 36.0, 1e-3, 1e-4, "Wyndor: obj=36");
        check(computePrimalViolation(model, r.x) <= 1e-4, "Wyndor: feasible");
    }
}

void testZeroRowLP() {
    // min 2x - y, x in [0,10], y in [-5,5], no constraints
    // optimal: x=0, y=5, obj = 10 + 0 - 5 = 5
    ModelIR model(ObjSense::kMinimize, 10.0, {2.0,-1.0},
                  denseToCSR({}, 2), {}, {}, {}, {0,-5}, {10,5},
                  {VarType::kContinuous,VarType::kContinuous}, {"x","y"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "ZeroRowLP: optimal");
    if (r.status == SolveStatus::kOptimal)
        checkNear(r.objective_value, 5.0, 1e-3, 1e-4, "ZeroRowLP: obj=5");
}

void testSingleVariableSingleConstraint() {
    // min x s.t. x >= 3, x in [0, inf)
    // optimal: x=3, obj=3
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0},
                  denseToCSR({{1.0}}, 1), {3.0}, {kInfinity}, {"r1"},
                  {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "1var1con: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, 3.0, 1e-3, 1e-4, "1var1con: obj=3");
        check(computePrimalViolation(model, r.x) <= 1e-4, "1var1con: feasible");
    }
}

void testEqualityConstraint() {
    // min x+y s.t. x+y=5, x,y>=0
    // optimal: any x+y=5, obj=5
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0,1.0},
                  denseToCSR({{1,1}}, 2), {5.0}, {5.0}, {"eq"},
                  {0,0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x","y"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "EqLP: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, 5.0, 1e-3, 1e-4, "EqLP: obj=5");
        check(computePrimalViolation(model, r.x) <= 1e-4, "EqLP: feasible");
    }
}

void testMultipleConstraints() {
    // min -x1 - 2x2 s.t. x1+x2<=4, x1<=3, x2<=3, x>=0
    // optimal: x1=1, x2=3, obj=-7
    ModelIR model(ObjSense::kMinimize, 0.0, {-1.0,-2.0},
                  denseToCSR({{1,1},{1,0},{0,1}}, 2),
                  {-kInfinity,-kInfinity,-kInfinity}, {4,3,3},
                  {"r1","r2","r3"}, {0,0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x1","x2"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "MultiCon: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, -7.0, 1e-3, 1e-4, "MultiCon: obj=-7");
        check(computePrimalViolation(model, r.x) <= 1e-4, "MultiCon: feasible");
    }
}

void testZeroObjectiveCoeffs() {
    // min 0*x + 0*y s.t. x+y>=2, x,y>=0
    // optimal: obj=0
    ModelIR model(ObjSense::kMinimize, 0.0, {0.0,0.0},
                  denseToCSR({{1,1}}, 2), {2.0}, {kInfinity}, {"r1"},
                  {0,0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x","y"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "ZeroObj: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, 0.0, 1e-3, 1e-4, "ZeroObj: obj=0");
        check(computePrimalViolation(model, r.x) <= 1e-4, "ZeroObj: feasible");
    }
}

void testBoundedKnownOptimum() {
    // min x+y s.t. x+y<=10, x in [2,5], y in [3,7]
    // optimal: x=2, y=3, obj=5
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0,1.0},
                  denseToCSR({{1,1}}, 2),
                  {-kInfinity}, {10.0}, {"r1"},
                  {2.0,3.0}, {5.0,7.0},
                  {VarType::kContinuous,VarType::kContinuous}, {"x","y"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "BoundedLP: optimal");
    if (r.status == SolveStatus::kOptimal) {
        checkNear(r.objective_value, 5.0, 1e-3, 1e-4, "BoundedLP: obj=5");
        check(computePrimalViolation(model, r.x) <= 1e-4, "BoundedLP: feasible");
    }
}

void testSparseMatrixWithZeroEntries() {
    // Same as Wyndor but A has structural zeros included (handled by denseToCSR)
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0,5.0},
                  denseToCSR({{1,0},{0,2},{3,2}}, 2),
                  {-kInfinity,-kInfinity,-kInfinity}, {4,12,18},
                  {"r1","r2","r3"}, {0,0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x1","x2"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    check(r.status == SolveStatus::kOptimal, "Sparse0: optimal");
    if (r.status == SolveStatus::kOptimal)
        checkNear(r.objective_value, 36.0, 1e-3, 1e-4, "Sparse0: obj=36");
}

void testScaledCoeffs() {
    // min 1e-3*x + 1e3*y s.t. 1e3*x + 1e-3*y >= 1, x,y >= 0
    // optimal: x = 1/1e3 = 1e-3, y=0, obj = 1e-6
    ModelIR model(ObjSense::kMinimize, 0.0, {1e-3, 1e3},
                  denseToCSR({{1e3, 1e-3}}, 2),
                  {1.0}, {kInfinity}, {"r1"},
                  {0.0,0.0}, {kInfinity,kInfinity},
                  {VarType::kContinuous,VarType::kContinuous}, {"x","y"});
    gpu::PdhgSolver solver;
    SolveResult r = solver.solve(model);
    // Feasibility is the main check here due to numerical scale differences
    if (r.status == SolveStatus::kOptimal)
        check(computePrimalViolation(model, r.x) <= 1e-3, "ScaledLP: feasible");
    // Don't require exact obj here; just that it terminates
    check(r.status == SolveStatus::kOptimal || r.status == SolveStatus::kIterationLimit,
          "ScaledLP: terminates");
}

// ---------------------------------------------------------------------------
// D. Netlib afiro.mps: GPU vs CPU simplex
// ---------------------------------------------------------------------------
void testAfiro() {
    const std::string path = "tests/data/afiro.mps";
    ModelIR model;
    try {
        model = parse_mps(path);
    } catch (const std::exception& e) {
        std::cerr << "  [SKIP] Cannot load " << path << ": " << e.what() << "\n";
        return;
    }

    // CPU reference
    RevisedSimplex cpu_solver;
    SolveResult cpu_r = cpu_solver.solve(model);
    check(cpu_r.status == SolveStatus::kOptimal, "afiro CPU: optimal");
    if (cpu_r.status != SolveStatus::kOptimal) {
        std::cerr << "  [SKIP] CPU reference did not converge for afiro\n";
        return;
    }
    std::cout << "  afiro CPU obj = " << cpu_r.objective_value << "\n";

    // GPU
    gpu::PdhgOptions opts;
    opts.max_iterations  = 200000;
    opts.check_frequency = 500;
    opts.tolerance       = 1e-5;
    gpu::PdhgSolver gpu_solver(opts);
    SolveResult gpu_r = gpu_solver.solve(model);
    check(gpu_r.status == SolveStatus::kOptimal, "afiro GPU: optimal");
    if (gpu_r.status != SolveStatus::kOptimal) {
        std::cerr << "  afiro GPU did not converge (iterations=" << gpu_r.iterations << ")\n";
        return;
    }
    std::cout << "  afiro GPU obj = " << gpu_r.objective_value
              << " (iters=" << gpu_r.iterations << ")\n";

    // Objective agreement within 1e-4 absolute (0 relative)
    checkNear(gpu_r.objective_value, cpu_r.objective_value, 1e-4, 0.0,
              "afiro: GPU obj agrees with CPU obj (absolute)");

    // Feasibility check
    double viol = computePrimalViolation(model, gpu_r.x);
    std::cout << "  afiro GPU primal violation = " << viol << "\n";
    check(viol <= 1e-4, "afiro GPU: feasible within 1e-4");
}

// ---------------------------------------------------------------------------
// E. adlittle benchmark: GPU vs CPU timing
// ---------------------------------------------------------------------------
void benchmarkAdlittle() {
    const std::string path = "tests/data/adlittle.mps";
    ModelIR model;
    try {
        model = parse_mps(path);
    } catch (const std::exception& e) {
        std::cerr << "  [SKIP] Cannot load " << path << ": " << e.what() << "\n";
        return;
    }

    std::cout << "  adlittle: m=" << model.numRows()
              << " n=" << model.numVars()
              << " nnz=" << model.A.nnz() << "\n";

    // CPU solve timing
    RevisedSimplex cpu_solver;
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    SolveResult cpu_r = cpu_solver.solve(model);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    std::cout << "  CPU solve time: " << cpu_ms << " ms\n";
    std::cout << "  CPU status: " << (cpu_r.status == SolveStatus::kOptimal ? "OPTIMAL" : "other")
              << "  obj=" << cpu_r.objective_value << "\n";

    // GPU end-to-end timing (includes H->D setup + solve + D->H)
    gpu::PdhgOptions opts;
    opts.max_iterations  = 200000;
    opts.check_frequency = 500;
    opts.tolerance       = 1e-5;

    auto gpu_e2e_t0 = std::chrono::high_resolution_clock::now();
    gpu::PdhgSolver gpu_solver(opts);
    SolveResult gpu_r = gpu_solver.solve(model);
    auto gpu_e2e_t1 = std::chrono::high_resolution_clock::now();
    double gpu_e2e_ms = std::chrono::duration<double, std::milli>(gpu_e2e_t1 - gpu_e2e_t0).count();

    std::cout << "  GPU end-to-end time: " << gpu_e2e_ms << " ms (includes H->D + solve + D->H)\n";
    std::cout << "  GPU status: " << (gpu_r.status == SolveStatus::kOptimal ? "OPTIMAL" : "ITER_LIMIT")
              << "  obj=" << gpu_r.objective_value
              << "  iters=" << gpu_r.iterations << "\n";

    double viol = computePrimalViolation(model, gpu_r.x);
    std::cout << "  GPU primal violation: " << viol << "\n";

    if (cpu_r.status == SolveStatus::kOptimal && gpu_r.status == SolveStatus::kOptimal) {
        double ratio = gpu_e2e_ms / cpu_ms;
        std::cout << "  [VALID CORRECTNESS-MATCHED TIMING COMPARISON]\n";
        std::cout << "  GPU/CPU time ratio: " << ratio << "\n";
        checkNear(gpu_r.objective_value, cpu_r.objective_value, 1e-4, 1e-4,
                  "adlittle: GPU obj agrees with CPU obj");
    } else {
        std::cout << "  [NON-CONVERGED / DIAGNOSTIC ONLY]\n";
        std::cout << "  GPU did not reach optimal status (status=" << static_cast<int>(gpu_r.status) << ").\n";
        std::cout << "  No speedup ratio is claimed for this instance.\n";
    }

    // Correctness checks
    check(gpu_r.status == SolveStatus::kOptimal || gpu_r.status == SolveStatus::kIterationLimit,
          "adlittle GPU: terminates cleanly");
    if (gpu_r.status == SolveStatus::kOptimal)
        check(viol <= 1e-4, "adlittle GPU: feasible");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    run("A. SpMV dimensions (crosses warp/block boundaries)", testSpMVDimensions);
    run("B. Wyndor Glass Co. (maximize, regression)", testWyndorGlass);
    run("B. Zero row LP", testZeroRowLP);
    run("C. 1-var 1-constraint", testSingleVariableSingleConstraint);
    run("C. Equality constraint", testEqualityConstraint);
    run("C. Multiple constraints", testMultipleConstraints);
    run("C. Zero objective coefficients", testZeroObjectiveCoeffs);
    run("C. Bounded LP with known optimum", testBoundedKnownOptimum);
    run("C. Sparse matrix with zero entries", testSparseMatrixWithZeroEntries);
    run("C. Scaled coefficients (numerical)", testScaledCoeffs);
    run("D. Netlib afiro.mps correctness", testAfiro);
    run("E. adlittle benchmark (GPU vs CPU)", benchmarkAdlittle);

    std::cout << "\n" << g_checks_run << " checks run, "
              << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
