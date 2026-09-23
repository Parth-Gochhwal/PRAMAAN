// test_backend_cpu.cpp
// P2 Step 1 — CpuBackend integration test suite
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include "pramaan/gpu/backend.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

using namespace pramaan;

static int g_checks_run = 0;
static int g_checks_failed = 0;

static void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

static void checkNear(double a, double b, double atol, double rtol, const std::string& description) {
    ++g_checks_run;
    if (a == b) {
        return; // Exact equality (handles +0/-0 and same infinity)
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (" << a << " vs " << b << ")\n"
                  << "    Reason: Non-finite value without exact equality.\n";
        return;
    }
    double diff = std::abs(a - b);
    double tol = atol + rtol * std::max(std::abs(a), std::abs(b));
    if (diff > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (" << a << " vs " << b << ")\n"
                  << "    diff: " << diff << " > tol: " << tol << "\n";
    }
}

static void checkVectorsNear(const std::vector<double>& a, const std::vector<double>& b, double atol, double rtol, const std::string& desc) {
    check(a.size() == b.size(), desc + " (sizes match)");
    if (a.size() != b.size()) return;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) {
            continue;
        }
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] " << desc << " at index " << i << "\n"
                      << "    expected: " << a[i] << "\n"
                      << "    actual:   " << b[i] << "\n"
                      << "    Reason: Non-finite value without exact equality.\n";
            return;
        }
        double diff = std::abs(a[i] - b[i]);
        double max_val = std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > atol + rtol * max_val) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] " << desc << " at index " << i << "\n"
                      << "    expected: " << a[i] << "\n"
                      << "    actual:   " << b[i] << "\n"
                      << "    diff:     " << diff << "\n"
                      << "    tol:      " << (atol + rtol * max_val) << "\n";
            return;
        }
    }
    ++g_checks_run; // Count as 1 check for the vector if all elements pass
}

static void checkBoolsMatch(const std::vector<bool>& a, const std::vector<bool>& b, const std::string& desc) {
    check(a.size() == b.size(), desc + " (sizes match)");
    if (a.size() != b.size()) return;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] " << desc << " at index " << i << " (expected " << a[i] << " got " << b[i] << ")\n";
            return;
        }
    }
    ++g_checks_run;
}

static void compareSolveResults(const SolveResult& direct, const SolveResult& via_backend) {
    check(direct.status == via_backend.status, "status matches");
    checkNear(direct.objective_value, via_backend.objective_value, 1e-9, 1e-9, "objective values match");
    check(direct.iterations == via_backend.iterations, "iterations match");
    checkVectorsNear(direct.x, via_backend.x, 1e-9, 1e-9, "primal solutions match");
    checkVectorsNear(direct.row_activity, via_backend.row_activity, 1e-9, 1e-9, "row activities match");
    checkBoolsMatch(direct.is_basic, via_backend.is_basic, "basis statuses match");
}

static void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

// Helper: build a CSR matrix from a dense 2D vector.
static CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < num_cols; ++c) {
            if (row[static_cast<std::size_t>(c)] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[static_cast<std::size_t>(c)]);
            }
        }
        row_ptr.push_back(static_cast<CSRMatrix::Index>(col_idx.size()));
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), num_cols);
}

// Compare two models for equality (Input preservation)
static bool modelsEqual(const ModelIR& m1, const ModelIR& m2) {
    return m1.obj_sense == m2.obj_sense &&
           m1.obj_offset == m2.obj_offset &&
           m1.obj_coeffs == m2.obj_coeffs &&
           m1.row_lower == m2.row_lower &&
           m1.row_upper == m2.row_upper &&
           m1.row_names == m2.row_names &&
           m1.var_lower == m2.var_lower &&
           m1.var_upper == m2.var_upper &&
           m1.var_types == m2.var_types &&
           m1.var_names == m2.var_names &&
           m1.A.numRows() == m2.A.numRows() &&
           m1.A.numCols() == m2.A.numCols() &&
           m1.A.nnz() == m2.A.nnz() &&
           m1.A.rowPtr() == m2.A.rowPtr() &&
           m1.A.colIdx() == m2.A.colIdx() &&
           m1.A.values() == m2.A.values();
}

static void solveAndCompare(const ModelIR& model, const RevisedSimplex::Options& options, bool expect_optimal = true) {
    ModelIR original_model = model;
    
    RevisedSimplex direct_solver(options);
    SolveResult direct_result = direct_solver.solve(model);
    
    auto backend = make_cpu_backend();
    SolveResult backend_result = backend->solve_lp_relaxation(model, options);
    
    compareSolveResults(direct_result, backend_result);
    if (expect_optimal) {
        check(backend_result.status == SolveStatus::kOptimal, "Expected optimal status");
    }
    
    check(modelsEqual(model, original_model), "ModelIR was mutated by backend!");
}

// ---- A. Factory/API tests ----
void testFactoryAndAPI() {
    auto b1 = make_cpu_backend();
    check(b1 != nullptr, "make_cpu_backend() returns non-null");
    auto b2 = make_cpu_backend();
    check(b2 != nullptr, "second make_cpu_backend() call returns non-null");
    check(b1.get() != b2.get(), "independent instances created");
    // Destructor is tested by them going out of scope safely
}

// ---- B, C. Meaningful Collection of LP Structures ----

void testWyndorMax() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0},
                  denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0},
                  {"r1", "r2", "r3"}, {0.0, 0.0}, {kInfinity, kInfinity},
                  {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testSimpleMin() {
    ModelIR model(ObjSense::kMinimize, 0.0, {2.0, 3.0},
                  denseToCSR({{1.0, 1.0}}, 2), {1.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous},
                  {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testZeroRowLP() {
    // 0 rows, 2 variables
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0, 1.0},
                  denseToCSR({}, 2), {}, {}, {}, {0.0, -5.0},
                  {kInfinity, 5.0}, {VarType::kContinuous, VarType::kContinuous},
                  {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testZeroColumnLP() {
    // 2 rows, 0 variables
    ModelIR model(ObjSense::kMinimize, 0.0, {},
                  denseToCSR({{}, {}}, 0), {-1.0, -1.0}, {1.0, 1.0}, {"r1", "r2"}, {},
                  {}, {}, {});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testSingleVariableLP() {
    ModelIR model(ObjSense::kMinimize, 0.0, {-1.0},
                  denseToCSR({{1.0}}, 1), {0.0}, {10.0}, {"r1"}, {0.0},
                  {kInfinity}, {VarType::kContinuous}, {"x"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testSingleConstraintLP() {
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0, 2.0, 3.0},
                  denseToCSR({{1.0, 1.0, 1.0}}, 3), {-kInfinity}, {10.0}, {"r1"}, {0.0, 0.0, 0.0},
                  {kInfinity, kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous}, {"x", "y", "z"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testRedundantConstraints() {
    ModelIR model(ObjSense::kMinimize, 0.0, {-1.0, -1.0},
                  denseToCSR({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {2.0, 4.0, 6.0}, {"r1", "r2", "r3"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testEqualityConstraints() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0, 1.0},
                  denseToCSR({{1.0, 1.0}}, 2), {5.0}, {5.0}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testInequalityBounds() { // Ranged rows
    ModelIR model(ObjSense::kMaximize, 0.0, {2.0, 1.0},
                  denseToCSR({{1.0, -1.0}}, 2), {-1.0}, {1.0}, {"r1"}, {0.0, 0.0},
                  {5.0, 5.0}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testVariableBounds() {
    ModelIR model(ObjSense::kMinimize, 0.0, {-1.0, -1.0},
                  denseToCSR({{1.0, 1.0}}, 2), {-kInfinity}, {10.0}, {"r1"}, {2.0, -3.0},
                  {5.0, 4.0}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testZeroObjCoeffs() {
    ModelIR model(ObjSense::kMinimize, 5.0, {0.0, 0.0},
                  denseToCSR({{1.0, 1.0}}, 2), {2.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testOptimumContainsZeros() {
    ModelIR model(ObjSense::kMinimize, 0.0, {10.0, 1.0},
                  denseToCSR({{1.0, 1.0}}, 2), {5.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    // Optimal: x=0, y=5
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testDegenerateLP() { // Multiple active constraints at origin
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0, 1.0},
                  denseToCSR({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {0.0, 0.0, 0.0}, {"r1", "r2", "r3"}, {-kInfinity, -kInfinity},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

void testNumericallyAwkward() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1e-7, 1e7},
                  denseToCSR({{1e7, 1e-7}}, 2), {1.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options);
}

// ---- D. Boundary/status behavior ----
void testInfeasible() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0},
                  denseToCSR({{1.0}, {1.0}}, 1), {5.0, -kInfinity}, {kInfinity, 2.0}, {"r1", "r2"}, {0.0},
                  {kInfinity}, {VarType::kContinuous}, {"x"}); // x >= 5 and x <= 2
    RevisedSimplex::Options options;
    solveAndCompare(model, options, false);
}

void testUnbounded() {
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0},
                  denseToCSR({{1.0}}, 1), {-kInfinity}, {kInfinity}, {"r1"}, {0.0},
                  {kInfinity}, {VarType::kContinuous}, {"x"});
    RevisedSimplex::Options options;
    solveAndCompare(model, options, false);
}

void testIterationLimit() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0},
                  denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0},
                  {"r1", "r2", "r3"}, {0.0, 0.0}, {kInfinity, kInfinity},
                  {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    RevisedSimplex::Options options;
    options.max_iterations = 0; // Force iteration limit
    solveAndCompare(model, options, false);
}

// ---- E. Options propagation ----
void testOptionsPropagation() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0},
                  denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0},
                  {"r1", "r2", "r3"}, {0.0, 0.0}, {kInfinity, kInfinity},
                  {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    RevisedSimplex::Options options1;
    options1.max_iterations = 0;
    
    RevisedSimplex::Options options2;
    options2.max_iterations = 1000;
    options2.tolerance = 1e-3;

    auto backend = make_cpu_backend();
    SolveResult r1 = backend->solve_lp_relaxation(model, options1);
    SolveResult r2 = backend->solve_lp_relaxation(model, options2);
    
    check(r1.status == SolveStatus::kIterationLimit, "options1 propagated (iter limit)");
    check(r2.status == SolveStatus::kOptimal, "options2 propagated (optimal)");
}

// ---- G. Repeated/concurrent-safe behavior ----
void testRepeatedSequentialCalls() {
    auto backend = make_cpu_backend();
    ModelIR model(ObjSense::kMinimize, 0.0, {2.0, 3.0},
                  denseToCSR({{1.0, 1.0}}, 2), {1.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous},
                  {"x", "y"});
    RevisedSimplex::Options options;
    
    SolveResult r1 = backend->solve_lp_relaxation(model, options);
    SolveResult r2 = backend->solve_lp_relaxation(model, options);
    SolveResult r3 = backend->solve_lp_relaxation(model, options);
    
    check(r1.status == SolveStatus::kOptimal, "Run 1 OK");
    check(r2.status == SolveStatus::kOptimal, "Run 2 OK");
    check(r3.status == SolveStatus::kOptimal, "Run 3 OK");
    checkNear(r1.objective_value, r2.objective_value, 1e-9, 1e-9, "Run 1 matches Run 2");
    checkNear(r2.objective_value, r3.objective_value, 1e-9, 1e-9, "Run 2 matches Run 3");
}

void testConcurrentBehavior() {
    // If standard C++17 thread is available
    const int NUM_THREADS = 4;
    std::vector<std::thread> threads;
    std::vector<SolveResult> results(NUM_THREADS);
    
    auto backend = make_cpu_backend(); // shared backend instance
    
    ModelIR model(ObjSense::kMinimize, 0.0, {2.0, 3.0},
                  denseToCSR({{1.0, 1.0}}, 2), {1.0}, {kInfinity}, {"r1"}, {0.0, 0.0},
                  {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous},
                  {"x", "y"});
    RevisedSimplex::Options options;
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            results[i] = backend->solve_lp_relaxation(model, options);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        check(results[i].status == SolveStatus::kOptimal, "Thread " + std::to_string(i) + " OK");
        if (i > 0) {
            checkNear(results[0].objective_value, results[i].objective_value, 1e-9, 1e-9, "Thread outputs match");
        }
    }
}

int main() {
    run("A. Factory/API tests", testFactoryAndAPI);
    run("B, C. Wyndor Max", testWyndorMax);
    run("B, C. Simple Min", testSimpleMin);
    run("B, C. Zero Row LP", testZeroRowLP);
    run("B, C. Zero Column LP", testZeroColumnLP);
    run("B, C. Single Variable LP", testSingleVariableLP);
    run("B, C. Single Constraint LP", testSingleConstraintLP);
    run("B, C. Redundant Constraints", testRedundantConstraints);
    run("B, C. Equality Constraints", testEqualityConstraints);
    run("B, C. Inequality Bounds", testInequalityBounds);
    run("B, C. Variable Bounds", testVariableBounds);
    run("B, C. Zero Obj Coeffs", testZeroObjCoeffs);
    run("B, C. Optimum Contains Zeros", testOptimumContainsZeros);
    run("B, C. Degenerate LP", testDegenerateLP);
    run("B, C. Numerically Awkward", testNumericallyAwkward);
    run("D. Boundary Infeasible", testInfeasible);
    run("D. Boundary Unbounded", testUnbounded);
    run("D. Boundary Iteration Limit", testIterationLimit);
    run("E. Options Propagation", testOptionsPropagation);
    run("G. Repeated Sequential Calls", testRepeatedSequentialCalls);
    run("G. Concurrent Behavior", testConcurrentBehavior);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}