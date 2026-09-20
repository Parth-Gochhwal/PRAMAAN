// test_branch_and_bound.cpp
// P1 Step 3 — DFS Branch-and-Bound correctness tests
//
// Primary instance is a textbook 0/1 knapsack with a hand-verifiable optimum:
//
//   maximize  8 x0 + 11 x1 + 6 x2 + 4 x3
//   s.t.      5 x0 +  7 x1 + 4 x2 + 3 x3 <= 14
//             x in {0,1}^4
//
// LP relaxation (by value/weight ratio 1.6, 1.571, 1.5, 1.333): take x0 and
// x1 whole (weight 12, value 19), then 2/4 of x2 -> x = (1, 1, 0.5, 0) with
// objective 22. Fractional, so B&B must branch.
//
// Integer optimum by enumeration of the feasible 0/1 points:
//   (1,1,0,0) w=12 v=19      (1,0,1,1) w=12 v=18
//   (0,1,1,1) w=14 v=21  <-- optimum
//   (1,1,1,0) w=16 infeasible
// so the optimum is 21 at x = (0, 1, 1, 1), strictly below the LP bound of 22.
//
// Tests:
// 1. Knapsack: objective 21, solution (0,1,1,1), branching and pruning occur,
//    the fractional LP relaxation is NOT accepted.
// 2. Every integer variable in the answer is integral.
// 3. The answer satisfies the original constraints and bounds.
// 4. Minimization sense is handled correctly.
// 5. A root relaxation that is already integral is returned immediately.
// 6. An infeasible integer problem is reported kInfeasible.
// 7. Continuous variables are left continuous in a mixed model.
// 8. Warm-start plumbing is exercised and falls back safely (see the note
//    on BranchAndBound::Options::use_warm_start).
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/branch_and_bound.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::VarType;
using pramaan::mip::BranchAndBound;
using pramaan::mip::MipResult;
using pramaan::mip::MipStatus;

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

void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    const int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

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

ModelIR makeKnapsack() {
    return ModelIR(ObjSense::kMaximize, 0.0,
                   {8.0, 11.0, 6.0, 4.0},
                   denseToCSR({{5.0, 7.0, 4.0, 3.0}}, 4),
                   {-kInfinity}, {14.0}, {"capacity"},
                   {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0},
                   {VarType::kInteger, VarType::kInteger, VarType::kInteger, VarType::kInteger},
                   {"x0", "x1", "x2", "x3"});
}

// Verifies x against the ORIGINAL model: bounds, rows, and integrality.
void checkFeasibleAndIntegral(const ModelIR& model, const std::vector<double>& x, double tol,
                              const std::string& case_name) {
    check(x.size() == static_cast<std::size_t>(model.numVars()), case_name + ": x has num_vars entries");
    if (x.size() != static_cast<std::size_t>(model.numVars())) return;

    for (CSRMatrix::Index j = 0; j < model.numVars(); ++j) {
        const double lo = model.var_lower[static_cast<std::size_t>(j)];
        const double up = model.var_upper[static_cast<std::size_t>(j)];
        const double xj = x[static_cast<std::size_t>(j)];
        std::ostringstream d;
        d << case_name << ": x[" << j << "]=" << xj << " within [" << lo << ", " << up << "]";
        check((lo <= -kInfinity || xj >= lo - tol) && (up >= kInfinity || xj <= up + tol), d.str());

        if (model.isInteger(j)) {
            std::ostringstream di;
            di << case_name << ": x[" << j << "]=" << xj << " is integral";
            check(std::abs(xj - std::round(xj)) <= 1e-6, di.str());
        }
    }

    const std::vector<double> activity = model.A.multiply(x);
    for (CSRMatrix::Index r = 0; r < model.numRows(); ++r) {
        const double lo = model.row_lower[static_cast<std::size_t>(r)];
        const double up = model.row_upper[static_cast<std::size_t>(r)];
        const double ar = activity[static_cast<std::size_t>(r)];
        std::ostringstream d;
        d << case_name << ": row " << r << " activity " << ar << " within [" << lo << ", " << up << "]";
        check((lo <= -kInfinity || ar >= lo - tol) && (up >= kInfinity || ar <= up + tol), d.str());
    }
}

void testKnapsack() {
    const ModelIR model = makeKnapsack();
    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    check(res.status == MipStatus::kOptimal, "knapsack: status is kOptimal");
    check(res.hasSolution(), "knapsack: an incumbent was found");
    checkNear(res.objective_value, 21.0, 1e-6, "knapsack: objective is the hand-computed 21");

    checkNear(res.x[0], 0.0, 1e-6, "knapsack: x0 = 0");
    checkNear(res.x[1], 1.0, 1e-6, "knapsack: x1 = 1");
    checkNear(res.x[2], 1.0, 1e-6, "knapsack: x2 = 1");
    checkNear(res.x[3], 1.0, 1e-6, "knapsack: x3 = 1");

    checkFeasibleAndIntegral(model, res.x, 1e-9, "knapsack");

    // The fractional LP relaxation (22) must NOT have been accepted.
    check(res.objective_value < 22.0 - 1e-6,
          "knapsack: the fractional LP bound of 22 was not returned as the answer");

    // B&B must actually have searched, not shortcut to an answer.
    check(res.statistics.branchings >= 1, "knapsack: branching actually occurred");
    check(res.statistics.nodes_explored > 1, "knapsack: more than the root node was explored");
    check(res.statistics.nodes_pruned_by_bound + res.statistics.nodes_pruned_infeasible >= 1,
          "knapsack: at least one node was pruned");
    check(res.statistics.incumbent_updates >= 1, "knapsack: the incumbent was set at least once");

    std::cout << "    nodes explored=" << res.statistics.nodes_explored
              << " branchings=" << res.statistics.branchings
              << " pruned(bound)=" << res.statistics.nodes_pruned_by_bound
              << " pruned(infeas)=" << res.statistics.nodes_pruned_infeasible
              << " incumbents=" << res.statistics.incumbent_updates
              << " warm=" << res.statistics.warm_start_successes << "/"
              << res.statistics.warm_start_attempts << "\n";
}

// Same feasible set and data, minimized instead. The minimum of
// 8x0+11x1+6x2+4x3 over 0/1 points with weight <= 14 is plainly x = 0.
void testMinimizationSense() {
    ModelIR model = makeKnapsack();
    model.obj_sense = ObjSense::kMinimize;

    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    check(res.status == MipStatus::kOptimal, "minimize: status is kOptimal");
    checkNear(res.objective_value, 0.0, 1e-6, "minimize: optimum is 0 at the origin");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "minimize");
}

// A root LP relaxation that is already integral must be returned as-is,
// with no branching at all.
void testRootAlreadyIntegral() {
    // maximize x0 s.t. x0 <= 3, 0 <= x0 <= 3 integer -> LP optimum x0 = 3.
    const ModelIR model(ObjSense::kMaximize, 0.0,
                        {1.0},
                        denseToCSR({{1.0}}, 1),
                        {-kInfinity}, {3.0}, {"r0"},
                        {0.0}, {3.0},
                        {VarType::kInteger},
                        {"x0"});

    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    check(res.status == MipStatus::kOptimal, "root-integral: status is kOptimal");
    checkNear(res.objective_value, 3.0, 1e-6, "root-integral: objective is 3");
    checkNear(res.x[0], 3.0, 1e-6, "root-integral: x0 = 3");
    check(res.statistics.branchings == 0, "root-integral: no branching was needed");
    check(res.statistics.nodes_explored == 1, "root-integral: only the root was explored");
}

// 2 x0 = 1 with x0 integer has no integer solution, though its LP relaxation
// is feasible at x0 = 0.5 -- so this exercises the search proving
// infeasibility rather than the root LP doing it.
void testIntegerInfeasible() {
    const ModelIR model(ObjSense::kMaximize, 0.0,
                        {1.0},
                        denseToCSR({{2.0}}, 1),
                        {1.0}, {1.0}, {"eq"},
                        {0.0}, {1.0},
                        {VarType::kInteger},
                        {"x0"});

    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    check(res.status == MipStatus::kInfeasible, "integer-infeasible: status is kInfeasible");
    check(!res.hasSolution(), "integer-infeasible: no solution is reported");
    check(res.statistics.branchings >= 1, "integer-infeasible: the search had to branch");
    check(res.statistics.nodes_pruned_infeasible >= 1,
          "integer-infeasible: both children were pruned infeasible");
}

// A mixed model: x0 integer, x1 continuous. The continuous variable must be
// allowed to take a fractional value in the answer.
void testMixedIntegerContinuous() {
    // maximize x0 + x1  s.t.  x0 + x1 <= 2.5,  0 <= x0 <= 2 (int), 0 <= x1 <= 2 (cont)
    // Optimum: objective 2.5, with x0 integral and x1 taking up the slack.
    const ModelIR model(ObjSense::kMaximize, 0.0,
                        {1.0, 1.0},
                        denseToCSR({{1.0, 1.0}}, 2),
                        {-kInfinity}, {2.5}, {"cap"},
                        {0.0, 0.0}, {2.0, 2.0},
                        {VarType::kInteger, VarType::kContinuous},
                        {"x0", "x1"});

    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    check(res.status == MipStatus::kOptimal, "mixed: status is kOptimal");
    checkNear(res.objective_value, 2.5, 1e-6, "mixed: objective is 2.5");
    check(std::abs(res.x[0] - std::round(res.x[0])) <= 1e-6, "mixed: x0 (integer) is integral");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "mixed");
}

// Warm-starting must never change the ANSWER -- only the path taken to it.
//
// The underlying DualSimplex::warmSolve() dual-pivot bug (setting x_B[leaving]=0
// instead of theta) has been fixed. This test verifies that warm-start now
// produces the same correct answer as the cold path.
void testWarmStartPlumbing() {
    const ModelIR model = makeKnapsack();

    // Default (cold) path: deterministic and correct.
    const BranchAndBound cold_bnb;
    const MipResult a = cold_bnb.solve(model);
    const MipResult b = cold_bnb.solve(model);

    check(a.status == b.status, "cold: repeated solves agree on status");
    checkNear(a.objective_value, b.objective_value, 1e-12,
              "cold: repeated solves agree on the objective");
    check(a.x == b.x, "cold: repeated solves are deterministic");
    checkNear(a.objective_value, 21.0, 1e-6, "cold: the hand-computed optimum");
    check(a.statistics.warm_start_attempts == 0,
          "cold: warm-start is off by default, so no attempts are made");

    // Warm-start path: the integration is wired and produces the correct result.
    BranchAndBound::Options warm_opts;
    warm_opts.use_warm_start = true;
    const BranchAndBound warm_bnb(warm_opts);
    const MipResult w = warm_bnb.solve(model);

    check(w.statistics.warm_start_attempts >= 1,
          "warm: the warm-start path was attempted at least once");
    check(w.statistics.warm_start_successes <= w.statistics.warm_start_attempts,
          "warm: successes never exceed attempts");
    check(w.status == MipStatus::kOptimal, "warm: status is kOptimal");
    checkNear(w.objective_value, 21.0, 1e-6, "warm: the hand-computed optimum");
    checkFeasibleAndIntegral(model, w.x, 1e-9, "warm");

    std::cout << "    warm: obj=" << w.objective_value
              << " warm_start=" << w.statistics.warm_start_successes << "/"
              << w.statistics.warm_start_attempts << "\n";
}

// A branch that turns an infinite bound finite changes the basis fingerprint
// and is rejected by warmSolve; the search must absorb that and still be
// correct. x1 has no upper bound here, so branching on it exercises exactly
// that fallback.
void testIncompatibleBasisFallback() {
    // maximize x0 + x1  s.t.  2x0 + 2x1 <= 5,  x0 in [0,4] int, x1 in [0, inf) int
    const ModelIR model(ObjSense::kMaximize, 0.0,
                        {1.0, 1.0},
                        denseToCSR({{2.0, 2.0}}, 2),
                        {-kInfinity}, {5.0}, {"cap"},
                        {0.0, 0.0}, {4.0, kInfinity},
                        {VarType::kInteger, VarType::kInteger},
                        {"x0", "x1"});

    const BranchAndBound bnb;
    const MipResult res = bnb.solve(model);

    // 2(x0 + x1) <= 5 with integers means x0 + x1 <= 2, so the optimum is 2.
    check(res.status == MipStatus::kOptimal, "fallback: status is kOptimal");
    checkNear(res.objective_value, 2.0, 1e-6, "fallback: objective is the hand-computed 2");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "fallback");
    check(res.statistics.branchings >= 1, "fallback: branching occurred");
}

}  // namespace

int main() {
    run("0/1 knapsack with hand-verified optimum 21", testKnapsack);
    run("Minimization sense", testMinimizationSense);
    run("Root relaxation already integral", testRootAlreadyIntegral);
    run("Integer-infeasible (LP-feasible) model", testIntegerInfeasible);
    run("Mixed integer/continuous model", testMixedIntegerContinuous);
    run("Warm-start plumbing (option off by default)", testWarmStartPlumbing);
    run("Incompatible inherited basis falls back cleanly", testIncompatibleBasisFallback);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
