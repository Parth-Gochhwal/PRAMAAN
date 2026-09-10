// test_simplex_tiny_lp.cpp
//
// Correctness milestone for RevisedSimplex: a hand-solvable 2-variable LP
// with a known textbook answer, plus a battery of smaller cases that each
// exercise one feature of ModelIR (min vs max, <=/>=/=/ranged rows,
// negative and free variable bounds, infeasibility, unboundedness, and a
// classic degenerate instance that historically defeats naive pivoting
// rules). No external test framework is used, to keep the solver's test
// binary dependency-free like the rest of PRAMAAN -- failures are reported
// with enough context to debug directly from the console output.
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::RevisedSimplex;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::VarType;

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

void checkNear(double actual, double expected, double tol, const std::string& description) {
    ++g_checks_run;
    if (std::abs(actual - expected) > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (expected " << expected << ", got " << actual
                   << ", |diff| = " << std::abs(actual - expected) << ")\n";
    }
}

// Builds a CSRMatrix from a dense row-major matrix, for test readability.
CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < static_cast<CSRMatrix::Index>(row.size()); ++c) {
            if (row[static_cast<std::size_t>(c)] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[static_cast<std::size_t>(c)]);
            }
        }
        row_ptr.push_back(static_cast<CSRMatrix::Index>(col_idx.size()));
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), num_cols);
}

// Asserts that a reported-optimal result is genuinely primal feasible for
// `model`: every variable within its bounds and every row activity within
// its row bounds. This is recomputed independently of SolveResult's own
// `row_activity` field so it also catches a solver bug that filled that
// field incorrectly.
void checkFeasible(const ModelIR& model, const SolveResult& result, double tol,
                    const std::string& case_name) {
    check(result.x.size() == static_cast<std::size_t>(model.numVars()), case_name + ": x has num_vars entries");
    for (CSRMatrix::Index j = 0; j < model.numVars(); ++j) {
        const double lo = model.var_lower[static_cast<std::size_t>(j)];
        const double up = model.var_upper[static_cast<std::size_t>(j)];
        const double xj = result.x[static_cast<std::size_t>(j)];
        std::ostringstream desc;
        desc << case_name << ": x[" << j << "] = " << xj << " within [" << lo << ", " << up << "]";
        check((lo <= -kInfinity || xj >= lo - tol) && (up >= kInfinity || xj <= up + tol), desc.str());
    }
    const std::vector<double> activity = model.A.multiply(result.x);
    for (CSRMatrix::Index r = 0; r < model.numRows(); ++r) {
        const double lo = model.row_lower[static_cast<std::size_t>(r)];
        const double up = model.row_upper[static_cast<std::size_t>(r)];
        const double ar = activity[static_cast<std::size_t>(r)];
        std::ostringstream desc;
        desc << case_name << ": row[" << r << "] activity " << ar << " within [" << lo << ", " << up << "]";
        check((lo <= -kInfinity || ar >= lo - tol) && (up >= kInfinity || ar <= up + tol), desc.str());
    }
}

void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    const int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

}  // namespace

// ---------------------------------------------------------------------
// Test 1: Wyndor Glass Co. -- the classic hand-solved 2-variable LP used
// throughout introductory LP/OR teaching (Hillier & Lieberman, and the
// MIT 15.053 recitation notes that cover graphical LP solving use the
// same problem). Textbook answer: x1 = 2, x2 = 6, objective = 36.
//
//   maximize   3 x1 + 5 x2
//   subject to      x1      <= 4
//                  2 x2     <= 12
//              3 x1 + 2 x2  <= 18
//              x1, x2 >= 0
// ---------------------------------------------------------------------
void testWyndorGlass() {
    ModelIR model(
        ObjSense::kMaximize,
        /*obj_offset=*/0.0,
        /*obj_coeffs=*/{3.0, 5.0},
        denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
        /*row_lower=*/{-kInfinity, -kInfinity, -kInfinity},
        /*row_upper=*/{4.0, 12.0, 18.0},
        /*row_names=*/{"plant1", "plant2", "plant3"},
        /*var_lower=*/{0.0, 0.0},
        /*var_upper=*/{kInfinity, kInfinity},
        /*var_types=*/{VarType::kContinuous, VarType::kContinuous},
        /*var_names=*/{"x1", "x2"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "Wyndor: status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.x[0], 2.0, 1e-6, "Wyndor: x1");
    checkNear(result.x[1], 6.0, 1e-6, "Wyndor: x2");
    checkNear(result.objective_value, 36.0, 1e-6, "Wyndor: objective");
    checkFeasible(model, result, 1e-6, "Wyndor");
}

// Same problem restated as an equivalent minimization to check ObjSense
// handling: minimize -3x1 - 5x2 has the same optimal point, negated value.
void testWyndorGlassAsMinimize() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {-3.0, -5.0},
        denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
        {-kInfinity, -kInfinity, -kInfinity},
        {4.0, 12.0, 18.0},
        {"plant1", "plant2", "plant3"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "Wyndor(min): status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.x[0], 2.0, 1e-6, "Wyndor(min): x1");
    checkNear(result.x[1], 6.0, 1e-6, "Wyndor(min): x2");
    checkNear(result.objective_value, -36.0, 1e-6, "Wyndor(min): objective");
    checkFeasible(model, result, 1e-6, "Wyndor(min)");
}

// ---------------------------------------------------------------------
// Test 2: mixed <=, >=, = constraints (a small diet-style LP), to
// exercise every row kind other than "ranged" in one model.
//
//   minimize   4 x1 + 3 x2
//   subject to  x1 +  x2 >= 10     (>=)
//              2 x1 +  x2 <= 24     (<=)
//                x1 -  x2  = 2      (=)
//              x1, x2 >= 0
//
// Solving x1 - x2 = 2 => x1 = x2 + 2. Substituting: feasible x2 in
// [4, ...] from x1+x2>=10 => 2x2+2>=10 => x2>=4; and 2x1+x2<=24 =>
// 2(x2+2)+x2<=24 => 3x2<=20 => x2<=20/3. Cost = 4(x2+2)+3x2 = 7x2+8,
// increasing in x2, so the minimum is at x2=4, x1=6, cost = 7*4+8 = 36.
// ---------------------------------------------------------------------
void testMixedConstraints() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {4.0, 3.0},
        denseToCSR({{1.0, 1.0}, {2.0, 1.0}, {1.0, -1.0}}, 2),
        {10.0, -kInfinity, 2.0},
        {kInfinity, 24.0, 2.0},
        {"demand", "capacity", "ratio"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "Mixed: status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.x[0], 6.0, 1e-6, "Mixed: x1");
    checkNear(result.x[1], 4.0, 1e-6, "Mixed: x2");
    checkNear(result.objective_value, 36.0, 1e-6, "Mixed: objective");
    checkFeasible(model, result, 1e-6, "Mixed");
}

// ---------------------------------------------------------------------
// Test 3: infeasible model (x >= 5 and x <= 2 simultaneously via two
// separate rows on the same variable).
// ---------------------------------------------------------------------
void testInfeasible() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1.0},
        denseToCSR({{1.0}, {1.0}}, 1),
        {5.0, -kInfinity},
        {kInfinity, 2.0},
        {"lo", "hi"},
        {0.0},
        {kInfinity},
        {VarType::kContinuous},
        {"x"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kInfeasible, "Infeasible: status infeasible");
}

// ---------------------------------------------------------------------
// Test 4: unbounded model (maximize x with no upper bound and no
// constraint that would cap it).
// ---------------------------------------------------------------------
void testUnbounded() {
    ModelIR model(
        ObjSense::kMaximize,
        0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {-kInfinity},
        {kInfinity},
        {"free_row"},
        {0.0},
        {kInfinity},
        {VarType::kContinuous},
        {"x"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kUnbounded, "Unbounded: status unbounded");
}

// Same idea but with zero rows at all (exercises the num_rows_std == 0
// fast path directly).
void testUnboundedNoRows() {
    ModelIR model(
        ObjSense::kMaximize,
        0.0,
        {1.0},
        denseToCSR({}, 1),
        {},
        {},
        {},
        {0.0},
        {kInfinity},
        {VarType::kContinuous},
        {"x"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kUnbounded, "UnboundedNoRows: status unbounded");
}

// ---------------------------------------------------------------------
// Test 5: a free variable (no lower bound) combined with a finite upper
// bound on the other variable.
//
//   minimize x
//   subject to x + y = 10
//              0 <= y <= 4
//              x free
//
// x = 10 - y is minimized by maximizing y, so y = 4, x = 6.
// ---------------------------------------------------------------------
void testFreeVariable() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1.0, 0.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {10.0},
        {10.0},
        {"sum"},
        {-kInfinity, 0.0},
        {kInfinity, 4.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x", "y"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "FreeVar: status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.x[0], 6.0, 1e-6, "FreeVar: x");
    checkNear(result.x[1], 4.0, 1e-6, "FreeVar: y");
    checkNear(result.objective_value, 6.0, 1e-6, "FreeVar: objective");
    checkFeasible(model, result, 1e-6, "FreeVar");
}

// ---------------------------------------------------------------------
// Test 6: a variable with a finite negative lower bound and a finite
// positive upper bound (exercises the shift path without splitting).
//
//   minimize x
//   subject to x >= -3   (a row, tighter than the variable's own bound)
//              -5 <= x <= 10
//
// Optimal x = -3.
// ---------------------------------------------------------------------
void testNegativeLowerBound() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {-3.0},
        {kInfinity},
        {"floor"},
        {-5.0},
        {10.0},
        {VarType::kContinuous},
        {"x"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "NegLB: status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.x[0], -3.0, 1e-6, "NegLB: x");
    checkNear(result.objective_value, -3.0, 1e-6, "NegLB: objective");
    checkFeasible(model, result, 1e-6, "NegLB");
}

// ---------------------------------------------------------------------
// Test 7: a genuinely ranged row (both a finite lower and finite upper
// bound, distinct), which internally splits into one <= row and one >=
// row.
//
//   maximize x + y
//   subject to 2 <= x + y <= 6
//              x, y >= 0
//
// Optimal objective is 6 (multiple optimal points on the segment
// x + y = 6); only the objective value and feasibility are checked.
// ---------------------------------------------------------------------
void testRangedRow() {
    ModelIR model(
        ObjSense::kMaximize,
        0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {2.0},
        {6.0},
        {"range"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x", "y"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "Ranged: status optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 6.0, 1e-6, "Ranged: objective");
    checkFeasible(model, result, 1e-6, "Ranged");
}

// ---------------------------------------------------------------------
// Test 8: Beale's classic cycling example. Under the textbook largest-
// coefficient (Dantzig) entering-variable rule with no anti-cycling
// safeguard, this instance cycles forever. It exists here purely as an
// anti-cycling regression test for Bland's rule: the assertion is that
// the solver terminates at optimality within the iteration budget at
// all, not any specific numeric answer.
//
//   minimize -0.75 x4 + 150 x5 - 0.02 x6 + 6 x7
//   subject to 0.25 x4 - 60 x5 - 0.04 x6 + 9 x7 <= 0
//              0.5  x4 - 90 x5 - 0.02 x6 + 3 x7 <= 0
//                                       x6       <= 1
//              x4, x5, x6, x7 >= 0
// ---------------------------------------------------------------------
void testBealeCyclingExample() {
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {-0.75, 150.0, -0.02, 6.0},
        denseToCSR({{0.25, -60.0, -0.04, 9.0}, {0.5, -90.0, -0.02, 3.0}, {0.0, 0.0, 1.0, 0.0}}, 4),
        {-kInfinity, -kInfinity, -kInfinity},
        {0.0, 0.0, 1.0},
        {"r1", "r2", "r3"},
        {0.0, 0.0, 0.0, 0.0},
        {kInfinity, kInfinity, kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x4", "x5", "x6", "x7"});

    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal, "Beale: status optimal (did not cycle)");
    if (result.status != SolveStatus::kOptimal) return;
    checkFeasible(model, result, 1e-6, "Beale");
    std::cout << "  (terminated in " << result.iterations << " pivots)\n";
}

// ---------------------------------------------------------------------
// Test 9: solving the same model twice with one RevisedSimplex instance
// (no mutable state should leak between calls).
// ---------------------------------------------------------------------
void testSolverIsReusable() {
    ModelIR model(
        ObjSense::kMaximize,
        0.0,
        {3.0, 5.0},
        denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
        {-kInfinity, -kInfinity, -kInfinity},
        {4.0, 12.0, 18.0},
        {"plant1", "plant2", "plant3"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    RevisedSimplex solver;
    const SolveResult first = solver.solve(model);
    const SolveResult second = solver.solve(model);

    checkNear(first.objective_value, second.objective_value, 1e-9, "Reusable: consistent objective");
    checkNear(first.x[0], second.x[0], 1e-9, "Reusable: consistent x1");
    checkNear(first.x[1], second.x[1], 1e-9, "Reusable: consistent x2");
}

void testPhase1BasisCleanup() {
    // Redundant equality constraint (tests redundant equality / zero-artificial)
    ModelIR model1(
        ObjSense::kMinimize,
        0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}, {1.0, 1.0}}, 2),
        {0.0, 0.0},
        {0.0, 0.0},
        {"eq1", "eq2"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    RevisedSimplex solver1;
    SolveResult res1 = solver1.solve(model1);
    check(res1.status == SolveStatus::kOptimal, "Cleanup: redundant equalities solve to optimal");
    if (res1.status == SolveStatus::kOptimal) {
        checkNear(res1.x[0], 0.0, 1e-9, "Cleanup: x1=0");
        checkNear(res1.x[1], 0.0, 1e-9, "Cleanup: x2=0");
    }

    // Infeasible equalities (tests infeasible equality)
    ModelIR model2(
        ObjSense::kMinimize,
        0.0,
        {1.0},
        denseToCSR({{1.0}, {1.0}}, 1),
        {5.0, 6.0},
        {5.0, 6.0},
        {"eq1", "eq2"},
        {0.0},
        {kInfinity},
        {VarType::kContinuous},
        {"x1"});

    RevisedSimplex solver2;
    SolveResult res2 = solver2.solve(model2);
    check(res2.status == SolveStatus::kInfeasible, "Cleanup: conflicting equalities are infeasible");

    // Artificial basic variable with a nonbasic replacement
    ModelIR model3(
        ObjSense::kMinimize,
        0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {0.0},
        {0.0},
        {"eq1"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    RevisedSimplex solver3;
    SolveResult res3 = solver3.solve(model3);
    check(res3.status == SolveStatus::kOptimal, "Cleanup: simple equality replacement solves");
    if (res3.status == SolveStatus::kOptimal) {
        checkNear(res3.x[0] + res3.x[1], 0.0, 1e-9, "Cleanup: x1+x2=0");
    }
}

int main() {
    run("Wyndor Glass Co. (MIT 15.053-style hand-solved LP)", testWyndorGlass);
    run("Wyndor Glass Co., restated as a minimize", testWyndorGlassAsMinimize);
    run("Mixed <=, >=, = constraints", testMixedConstraints);
    run("Infeasible model", testInfeasible);
    run("Unbounded model", testUnbounded);
    run("Unbounded model with zero rows", testUnboundedNoRows);
    run("Free variable with a bounded partner", testFreeVariable);
    run("Negative lower bound tightened by a row", testNegativeLowerBound);
    run("Genuinely ranged row", testRangedRow);
    run("Beale's cycling example (anti-cycling regression)", testBealeCyclingExample);
    run("Solver instance is reusable across solves", testSolverIsReusable);
    run("Phase 1 basis cleanup (artificials, redundant/infeasible equalities)", testPhase1BasisCleanup);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}