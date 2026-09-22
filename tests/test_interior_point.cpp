#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>

#include "pramaan/ir.hpp"
#include "pramaan/ipm/interior_point.hpp"

using namespace pramaan;

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

void checkNear(double a, double b, double tol, const std::string& description) {
    ++g_checks_run;
    if (std::abs(a - b) > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (" << a << " vs " << b << ")\n";
    }
}

CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < num_cols; ++c) {
            if (row[c] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[c]);
            }
        }
        row_ptr.push_back(col_idx.size());
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), num_cols);
}

void checkFeasible(const ModelIR& model, const SolveResult& result, double tol, const std::string& case_name) {
    check(std::isfinite(result.objective_value), case_name + ": objective is finite");
    for (int j = 0; j < model.numVars(); ++j) {
        const double lo = model.var_lower[j];
        const double up = model.var_upper[j];
        const double xj = result.x[j];
        check(std::isfinite(xj), case_name + ": x[" + std::to_string(j) + "] is finite");
        check((lo <= -kInfinity || xj >= lo - tol) && (up >= kInfinity || xj <= up + tol), case_name + ": var bound");
    }
    auto activity = model.A.multiply(result.x);
    for (int r = 0; r < model.numRows(); ++r) {
        const double lo = model.row_lower[r];
        const double up = model.row_upper[r];
        const double ar = activity[r];
        check(std::isfinite(ar), case_name + ": row_activity[" + std::to_string(r) + "] is finite");
        check((lo <= -kInfinity || ar >= lo - tol) && (up >= kInfinity || ar <= up + tol), case_name + ": row bound");
    }
}

void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

void testWyndorGlass() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0}, denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0}, {"r1", "r2", "r3"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Wyndor: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 36.0, 1e-6, "Wyndor: obj");
    checkFeasible(model, result, 1e-6, "Wyndor");
}

void testWyndorGlassAsMinimize() {
    ModelIR model(ObjSense::kMinimize, 0.0, {-3.0, -5.0}, denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0}, {"r1", "r2", "r3"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "WyndorMin: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, -36.0, 1e-6, "WyndorMin: obj");
    checkFeasible(model, result, 1e-6, "WyndorMin");
}

void testMixedConstraints() {
    ModelIR model(ObjSense::kMinimize, 0.0, {4.0, 3.0}, denseToCSR({{1.0, 1.0}, {2.0, 1.0}, {1.0, -1.0}}, 2),
                  {10.0, -kInfinity, 2.0}, {kInfinity, 24.0, 2.0}, {"d", "c", "r"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Mixed: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 36.0, 1e-6, "Mixed: obj");
    checkFeasible(model, result, 1e-6, "Mixed");
}

void testFreeVariable() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0, 0.0}, denseToCSR({{1.0, 1.0}}, 2),
                  {10.0}, {10.0}, {"s"}, {-kInfinity, 0.0}, {kInfinity, 4.0},
                  {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "FreeVar: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 6.0, 1e-6, "FreeVar: obj");
    checkFeasible(model, result, 1e-6, "FreeVar");
}

void testNegativeLowerBound() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0}, denseToCSR({{1.0}}, 1),
                  {-3.0}, {kInfinity}, {"r"}, {-5.0}, {10.0}, {VarType::kContinuous}, {"x"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "NegLB: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, -3.0, 1e-6, "NegLB: obj");
    checkFeasible(model, result, 1e-6, "NegLB");
}

void testRangedRow() {
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0, 1.0}, denseToCSR({{1.0, 1.0}}, 2),
                  {2.0}, {6.0}, {"r"}, {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Ranged: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 6.0, 1e-6, "Ranged: obj");
    checkFeasible(model, result, 1e-6, "Ranged");
}

void testBealeCyclingExample() {
    ModelIR model(ObjSense::kMinimize, 0.0, {-0.75, 150.0, -0.02, 6.0},
                  denseToCSR({{0.25, -60.0, -0.04, 9.0}, {0.5, -90.0, -0.02, 3.0}, {0.0, 0.0, 1.0, 0.0}}, 4),
                  {-kInfinity, -kInfinity, -kInfinity}, {0.0, 0.0, 1.0}, {"r1", "r2", "r3"},
                  {0.0, 0.0, 0.0, 0.0}, {kInfinity, kInfinity, kInfinity, kInfinity},
                  {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous, VarType::kContinuous}, {"x4", "x5", "x6", "x7"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Beale: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkFeasible(model, result, 1e-6, "Beale");
}

void testSolverIsReusable() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0}, denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0}, {"p1", "p2", "p3"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    InteriorPointSolver solver;
    const SolveResult first = solver.solve(model);
    const SolveResult second = solver.solve(model);
    checkNear(first.objective_value, second.objective_value, 1e-9, "Reusable: objective");
}

void testObjectiveOffset() {
    ModelIR model(ObjSense::kMinimize, 100.0, {1.0}, denseToCSR({{1.0}}, 1),
                  {5.0}, {kInfinity}, {"lb"}, {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "ObjOffset: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 105.0, 1e-6, "ObjOffset: obj");
    checkFeasible(model, result, 1e-6, "ObjOffset");
}

void testMaxIterationsZero() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0}, denseToCSR({{1.0}}, 1), {5.0}, {kInfinity}, {"lb"}, {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});
    InteriorPointSolver::Options opts;
    opts.max_iterations = 0;
    InteriorPointSolver solver(opts);
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kIterationLimit, "MaxIterations=0: status should be IterationLimit");
}

void testInvalidOptions() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0}, denseToCSR({{1.0}}, 1), {5.0}, {kInfinity}, {"lb"}, {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});

    auto checkOpt = [&](InteriorPointSolver::Options opts, const std::string& name) {
        InteriorPointSolver solver(opts);
        const SolveResult result = solver.solve(model);
        check(result.status == SolveStatus::kNumericalFailure, "Invalid option " + name + ": should fail");
    };

    InteriorPointSolver::Options opts;
    opts = InteriorPointSolver::Options(); opts.max_iterations = -1; checkOpt(opts, "max_iterations < 0");
    opts = InteriorPointSolver::Options(); opts.tolerance = 0.0; checkOpt(opts, "tolerance <= 0");
    opts = InteriorPointSolver::Options(); opts.tolerance = -1e-6; checkOpt(opts, "tolerance < 0");
    opts = InteriorPointSolver::Options(); opts.barrier_reduction = -0.1; checkOpt(opts, "barrier_reduction < 0");
    opts = InteriorPointSolver::Options(); opts.barrier_reduction = 1.1; checkOpt(opts, "barrier_reduction > 1");
    opts = InteriorPointSolver::Options(); opts.initial_bound_slack = 0.0; checkOpt(opts, "initial_bound_slack <= 0");
    opts = InteriorPointSolver::Options(); opts.initial_bound_slack = -1.0; checkOpt(opts, "initial_bound_slack < 0");
}

void testZeroRowLP() {
    ModelIR model(ObjSense::kMinimize, 10.0, {2.0, -1.0}, denseToCSR({}, 2), {}, {}, {}, {0.0, -5.0}, {10.0, 5.0}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "ZeroRowLP: optimal");
    checkNear(result.objective_value, 10.0 + 2.0 * 0.0 + (-1.0) * 5.0, 1e-6, "ZeroRowLP: objective");
}

void testMaxZeroRowLP() {
    // maximize 3x + 4y + 5
    // 1 <= x <= 10
    // -2 <= y <= 8
    // No rows.
    // Optimal: x=10, y=8. Obj = 3(10) + 4(8) + 5 = 67
    ModelIR model(ObjSense::kMaximize, 5.0, {3.0, 4.0}, denseToCSR({}, 2), {}, {}, {}, {1.0, -2.0}, {10.0, 8.0}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "MaxZeroRowLP: optimal");
    check(result.x.size() == 2, "MaxZeroRowLP: solution dimension");

    if (result.status != SolveStatus::kOptimal || result.x.size() != 2) {
        return;
    }

    double dot_prod = 0.0;
    for (int j = 0; j < 2; ++j) {
        dot_prod += model.obj_coeffs[j] * result.x[j];
    }
    double recomputed_obj = model.obj_offset + dot_prod;
    checkNear(result.objective_value, 67.0, 1e-6, "MaxZeroRowLP: objective matches known");
    checkNear(result.objective_value, recomputed_obj, 1e-6, "MaxZeroRowLP: objective matches recomputed dot product");
}

void testMaxZeroRowLP_Bug() {
    // maximize -x + 5
    // x >= 0
    // no structural rows
    // Expected: x = 0, objective = 5
    ModelIR model(ObjSense::kMaximize, 5.0, {-1.0}, denseToCSR({}, 1), {}, {}, {}, {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "MaxZeroRowLP_Bug: optimal");
    check(result.x.size() == 1, "MaxZeroRowLP_Bug: solution dimension");

    if (result.status != SolveStatus::kOptimal || result.x.size() != 1) {
        return;
    }

    checkNear(result.x[0], 0.0, 1e-6, "MaxZeroRowLP_Bug: x is 0");
    checkNear(result.objective_value, 5.0, 1e-6, "MaxZeroRowLP_Bug: objective is 5");

    double recomputed_obj = model.obj_offset + model.obj_coeffs[0] * result.x[0];
    checkNear(result.objective_value, recomputed_obj, 1e-6, "MaxZeroRowLP_Bug: objective matches recomputed dot product");
}

void testMaxZeroRowLP_Unbounded() {
    // maximize x
    // x >= 0
    // no structural rows
    // Expected: kUnbounded
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0}, denseToCSR({}, 1), {}, {}, {}, {0.0}, {kInfinity}, {VarType::kContinuous}, {"x"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kUnbounded, "MaxZeroRowLP_Unbounded: unbounded");
}

void testIntegerVariableRejection() {
    ModelIR model(ObjSense::kMinimize, 0.0, {1.0}, denseToCSR({{1.0}}, 1), {5.0}, {kInfinity}, {"lb"}, {0.0}, {kInfinity}, {VarType::kInteger}, {"x"});
    InteriorPointSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kNumericalFailure, "Integer var: should fail");
}

int main() {
    run("Wyndor Glass Co. (maximize)", testWyndorGlass);
    run("Wyndor Glass Co. (minimize equivalent)", testWyndorGlassAsMinimize);
    run("Mixed <=, >=, = constraints", testMixedConstraints);
    run("Free variable with a bounded partner", testFreeVariable);
    run("Negative lower bound tightened by a row", testNegativeLowerBound);
    run("Genuinely ranged row", testRangedRow);
    run("Beale's cycling example", testBealeCyclingExample);
    run("Solver instance is reusable across solves", testSolverIsReusable);
    run("Objective offset handling", testObjectiveOffset);
    run("Max iterations = 0", testMaxIterationsZero);
    run("Invalid options", testInvalidOptions);
    run("Zero row LP", testZeroRowLP);
    run("Max zero row LP", testMaxZeroRowLP);
    run("Max zero row LP (Bug)", testMaxZeroRowLP_Bug);
    run("Max zero row LP (Unbounded)", testMaxZeroRowLP_Unbounded);
    run("Integer variable rejection", testIntegerVariableRejection);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
