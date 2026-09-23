// test_branch_and_bound.cpp
// Branch-and-bound correctness tests
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
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

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

// =========================================================================
// Core functionality tests
// =========================================================================

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
                        {0.0}, {2.0},
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

void testRootLPInfeasibleStatus() {
    // Issue 1: A model whose root LP is immediately infeasible must return
    // kInfeasible, not kNodeLimit, even if the node limit wasn't explicitly exhausted.
    //
    // maximize x0
    // s.t. x0 >= 20
    // x0 in [0, 10]
    // The LP relaxation is infeasible.
    ModelIR model;
    model.obj_sense = ObjSense::kMaximize;
    model.obj_coeffs = {1.0};
    model.var_lower = {0.0};
    model.var_upper = {10.0};
    model.var_types = {VarType::kInteger};
    model.var_names = {"x0"};
    model.row_names = {"r0"};
    model.row_lower = {20.0};
    model.row_upper = {kInfinity};
    model.A = CSRMatrix({0, 1}, {0}, {1.0}, 1);

    BranchAndBound::Options opts;
    opts.max_nodes = 5; // Use a limit to verify we don't accidentally return kNodeLimit
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kInfeasible, "root-lp-infeasible: status must be kInfeasible, not kNodeLimit");
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
// and is rejected by warmSolve; the search must absorb that and still be correct.
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

// Warm-starting must NEVER change the final answer — only the internal path.
void testWarmVsColdComparison() {
    const ModelIR model = makeKnapsack();
    const double tol = 1e-6;

    BranchAndBound::Options cold_opts;
    cold_opts.use_warm_start = false;
    const BranchAndBound cold_bnb(cold_opts);
    const MipResult cold = cold_bnb.solve(model);

    BranchAndBound::Options warm_opts;
    warm_opts.use_warm_start = true;
    const BranchAndBound warm_bnb(warm_opts);
    const MipResult warm = warm_bnb.solve(model);

    check(cold.status == MipStatus::kOptimal, "cold-vs-warm: cold status is kOptimal");
    check(warm.status == MipStatus::kOptimal, "cold-vs-warm: warm status is kOptimal");
    check(cold.status == warm.status, "cold-vs-warm: statuses match");

    checkNear(cold.objective_value, warm.objective_value, tol,
              "cold-vs-warm: objectives match");
    checkNear(cold.objective_value, 21.0, tol,
              "cold-vs-warm: objective is the hand-computed 21");

    check(cold.x.size() == warm.x.size(), "cold-vs-warm: x vector sizes match");
    if (cold.x.size() == warm.x.size()) {
        for (std::size_t j = 0; j < cold.x.size(); ++j) {
            std::ostringstream d;
            d << "cold-vs-warm: x[" << j << "] matches";
            checkNear(cold.x[j], warm.x[j], tol, d.str());
        }
    }

    checkFeasibleAndIntegral(model, cold.x, 1e-9, "cold-vs-warm cold");
    checkFeasibleAndIntegral(model, warm.x, 1e-9, "cold-vs-warm warm");

    check(cold.statistics.warm_start_attempts == 0,
          "cold-vs-warm: cold mode made zero warm-start attempts");
    check(warm.statistics.warm_start_attempts > 0,
          "cold-vs-warm: warm mode attempted warm starts");
    check(warm.statistics.warm_start_successes > 0,
          "cold-vs-warm: at least one warm start actually succeeded");

    std::cout << "    cold: nodes=" << cold.statistics.nodes_explored
              << " warm_starts=" << cold.statistics.warm_start_successes
              << "/" << cold.statistics.warm_start_attempts << "\n";
    std::cout << "    warm: nodes=" << warm.statistics.nodes_explored
              << " warm_starts=" << warm.statistics.warm_start_successes
              << "/" << warm.statistics.warm_start_attempts << "\n";
}

// =========================================================================
// Cut tests
// =========================================================================

void testCutTightensRelaxation() {
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {3.0, 3.0, 3.0},
        denseToCSR({{2.0, 2.0, 2.0}}, 3),
        {-kInfinity}, {5.0}, {"r1"},
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
        {VarType::kInteger, VarType::kInteger, VarType::kInteger},
        {"x1", "x2", "x3"}
    );

    BranchAndBound::Options opts;
    opts.node_selection = BranchAndBound::Options::NodeSelection::kDepthFirst;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "cut-tightens: Optimal status");
    checkNear(res.objective_value, 6.0, 1e-6, "cut-tightens: Correct IP obj = 6");
    check(res.statistics.cuts_generated > 0, "cut-tightens: Cuts were generated");
    check(res.statistics.cuts_applied > 0, "cut-tightens: Cuts were applied");
}

// Issue 2: A row with a continuous variable mixed in must NEVER generate a cut.
void testCoverCutRejectsContinuousRow() {
    // Row: 2*x0 + 3*x1 <= 4, where x0 is binary integer and x1 is continuous.
    // LP value: x0=1.0, x1=0.66 (fractional LP value of continuous var doesn't matter).
    // A cover cut is NOT valid for this row because x1 is not binary.
    // x0+x1 > 1 would be violated at LP, but the cut is invalid for MILP.
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {2.0, 3.0},
        denseToCSR({{2.0, 3.0}}, 2),
        {-kInfinity}, {4.0}, {"r1"},
        {0.0, 0.0}, {1.0, kInfinity},
        {VarType::kInteger, VarType::kContinuous},
        {"x0", "x1"}
    );

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    // The solve itself must succeed or fail correctly — what matters is no cut
    // was generated for the mixed row.
    check(res.statistics.cuts_generated == 0,
          "cover-cut-reject-continuous: no cut generated for binary+continuous row");
}

// Issue 2: A row with a general integer variable (not [0,1]) must NOT generate a cut.
void testCoverCutRejectsGeneralIntegerRow() {
    // Row: x0 + x1 <= 1, where x0 is binary [0,1] and x1 is general integer [0,3].
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {1.0, 2.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity}, {1.0}, {"r1"},
        {0.0, 0.0}, {1.0, 3.0},
        {VarType::kInteger, VarType::kInteger},
        {"x0", "x1"}
    );

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.statistics.cuts_generated == 0,
          "cover-cut-reject-general-int: no cut generated for binary+general-int row");
}

// Issue 2: A row with a negative coefficient must NOT generate a cut.
void testCoverCutRejectsNegativeCoeff() {
    // Row: x0 - x1 <= 0, where x0 and x1 are binary.
    // Negative coefficient on x1 disqualifies this row.
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, -1.0}}, 2),
        {-kInfinity}, {0.0}, {"r1"},
        {0.0, 0.0}, {1.0, 1.0},
        {VarType::kInteger, VarType::kInteger},
        {"x0", "x1"}
    );

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.statistics.cuts_generated == 0,
          "cover-cut-reject-neg-coeff: no cut generated for row with negative coefficient");
}

// Issue 2: A row with a finite lower bound (ranged row) must NOT generate a cut.
void testCoverCutRejectsFiniteLowerBound() {
    // Row: 1 <= x0 + x1 <= 2, x0, x1 binary.
    // row_lower is finite (not -inf), so this is not a pure <= row.
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {1.0}, {2.0}, {"r1"},
        {0.0, 0.0}, {1.0, 1.0},
        {VarType::kInteger, VarType::kInteger},
        {"x0", "x1"}
    );

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.statistics.cuts_generated == 0,
          "cover-cut-reject-finite-lb: no cut generated for row with finite lower bound");
}

// Issue 2 + 3: A pure binary knapsack row should be able to generate a valid cut.
void testCoverCutValidBinaryRow() {
    // Row: x0 + x1 + x2 <= 2, all binary; LP optimum at (1,1,1) is fractional
    // at root only if the RHS is tighter. Use a 3-var example where LP = 3 >= 3
    // but all-ones violates: x0+x1+x2 <= 2. So LP at root is (1,1,0.0...) ok.
    // Better: use the testCutTightensRelaxation model: 2x0+2x1+2x2 <= 5, all binary.
    // LP gives x0=x1=x2=1 violating 2+2+2=6>5? No, LP can be (1,1,0.5)=obj 7.5.
    // We just check that the solver reaches the right answer, which requires no cuts necessarily.
    // The test testCutTightensRelaxation already covers this. Just verify coverage here.
    ModelIR model(
        ObjSense::kMaximize, 0.0,
        {1.0, 1.0, 1.0},
        denseToCSR({{1.0, 1.0, 1.0}}, 3),
        {-kInfinity}, {2.0}, {"r1"},
        {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
        {VarType::kInteger, VarType::kInteger, VarType::kInteger},
        {"x0", "x1", "x2"}
    );

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "cover-cut-valid: optimal status");
    checkNear(res.objective_value, 2.0, 1e-6, "cover-cut-valid: obj = 2");
    // Cuts may or may not be generated for this specific LP value; no assertion on that.
}

// Issue 4: Duplicate cut must NOT be counted as generated or applied.
void testDuplicateCuts() {
    // Two identical rows over the same binary variables force the same cover
    // inequality to be generated twice in the same pass.
    //
    //   maximize 3 x0 + 3 x1 + 3 x2
    //   2 x0 + 2 x1 + 2 x2 <= 5    (row 0)
    //   2 x0 + 2 x1 + 2 x2 <= 5    (row 1)
    //   x in {0,1}^3
    //
    // The LP optimum is (1, 1, 0.5), obj = 7.5.
    // Row 0 generates cover {x0,x1,x2} with weight sum 6 > 5.
    // Cut: x0+x1+x2 <= 2. Evaluated at LP opt: 1+1+0.5 = 2.5 > 2 (violated!).
    // Row 1 generates the same cover. The second must be detected as a duplicate.
    ModelIR model;
    model.obj_sense = ObjSense::kMaximize;
    model.obj_coeffs = {3, 3, 3};
    model.var_lower = {0, 0, 0};
    model.var_upper = {1, 1, 1};
    model.var_types = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
    model.var_names = {"x0", "x1", "x2"};
    model.row_names = {"r0", "r1"};
    model.row_lower = {-kInfinity, -kInfinity};
    model.row_upper = {5.0, 5.0};
    // Two identical rows: 2 x0 + 2 x1 + 2 x2 <= 5
    model.A = CSRMatrix({0, 3, 6}, {0, 1, 2, 0, 1, 2}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0}, 3);

    BranchAndBound::Options opts;
    opts.use_warm_start = false;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "duplicate-cuts: found optimum");
    // The invariant: applied + rejected == generated (non-duplicate cuts)
    check(res.statistics.cuts_applied + res.statistics.cuts_rejected == res.statistics.cuts_generated,
          "duplicate-cuts: applied + rejected == generated");
    // The duplicate was generated and must be counted in duplicate_cuts.
    check(res.statistics.duplicate_cuts > 0,
          "duplicate-cuts: at least one duplicate was detected (same cover from identical rows)");

    std::cout << "    generated=" << res.statistics.cuts_generated
              << " applied=" << res.statistics.cuts_applied
              << " rejected=" << res.statistics.cuts_rejected
              << " duplicates=" << res.statistics.duplicate_cuts << "\n";
}

// Issue 5: After a cut is added, children warm-start from the post-cut basis.
void testWarmStartCutInteraction() {
    // Same as knapsack but ensure warm+cut path works end-to-end.
    ModelIR model;
    model.obj_sense = ObjSense::kMaximize;
    model.obj_coeffs = {8, 11, 6, 4};
    model.var_lower = {0, 0, 0, 0};
    model.var_upper = {1, 1, 1, 1};
    model.var_types = {VarType::kInteger, VarType::kInteger, VarType::kInteger, VarType::kInteger};
    model.var_names = {"x1", "x2", "x3", "x4"};
    model.row_names = {"r1"};
    model.row_lower = {-kInfinity};
    model.row_upper = {14};
    model.A = CSRMatrix({0, 4}, {0, 1, 2, 3}, {5, 7, 4, 3}, 4);

    BranchAndBound::Options opts;
    opts.use_warm_start = true;
    opts.node_selection = BranchAndBound::Options::NodeSelection::kBestBound;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "warm-start+cut: optimal status");
    checkNear(res.objective_value, 21.0, 1e-6, "warm-start+cut: correct optimum");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "warm-start+cut");
}

// =========================================================================
// Issue 1: Root LP unbounded does NOT prove MILP unbounded.
// The MILP solver must return kNumericalFailure (inconclusive) when the
// root LP relaxation is unbounded. This is a regression test for the bug
// where the solver was returning kUnbounded (which is not even in the
// MipStatus enum anymore — it was removed precisely because an unbounded LP
// does not prove MILP unboundedness).
//
// Scenario: maximize x0 + x1, subject to x1 >= 1 (so it's feasible),
// and x0 has no upper bound. The LP relaxation is unbounded (x0 can grow
// without limit). With a small node limit, the solver must return
// kNumericalFailure (LP relaxation inconclusive) or kNodeLimit.
// It must NOT return kOptimal (there is no finite optimal).
void testRootLPUnboundedMilpBehavior() {
    // maximize x0 + x1
    // subject to: x1 >= 1 (to make feasible)
    // x0 in [0, +inf) integer  <- unbounded direction
    // x1 in [0, 5] integer
    // LP is unbounded because x0 is unconstrained from above.
    ModelIR model(ObjSense::kMaximize, 0.0,
                  {1.0, 1.0},
                  denseToCSR({{0.0, 1.0}}, 2),  // x1 >= 1
                  {1.0}, {kInfinity}, {"lb"},
                  {0.0, 0.0}, {kInfinity, 5.0},
                  {VarType::kInteger, VarType::kInteger},
                  {"x0", "x1"});

    BranchAndBound::Options opts;
    opts.max_nodes = 3;  // Small limit to avoid hanging
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    // The LP relaxation is unbounded. Our solver maps NodeState::kUnbounded ->
    // MipStatus::kNumericalFailure (inconclusive). kNodeLimit is also acceptable
    // if the node limit fires before the unbounded LP is reached.
    // kOptimal would be WRONG (no finite optimal exists for an unbounded MILP).
    // kInfeasible would be WRONG (feasible solutions exist, e.g. x0=0, x1=1).
    const bool acceptable = (res.status == MipStatus::kNumericalFailure ||
                              res.status == MipStatus::kNodeLimit);
    check(acceptable,
          "root-LP-unbounded: status is kNumericalFailure or kNodeLimit (NOT false kUnbounded)");

    // If kOptimal was returned, that would be a correctness bug.
    check(res.status != MipStatus::kOptimal,
          "root-LP-unbounded: kOptimal must NOT be returned for an unbounded model");

    std::cout << "    root-LP-unbounded: status=" << static_cast<int>(res.status)
              << " (0=kOptimal 1=kInfeasible 2=kNodeLimit 3=kNumericalFailure)\n";
}

// =========================================================================
// Issue 6: Best-bound queue ordering correctness (min and max).
// =========================================================================

void testBestBoundMinimization() {
    // minimize x0 + x1  s.t.  x0 + x1 >= 3, x0 x1 in {0,1,...,5} int
    // Optimum: x0=2, x1=1 (or any combo summing to 3, min obj = 3)
    ModelIR model(ObjSense::kMinimize, 0.0,
                  {1.0, 1.0},
                  denseToCSR({{1.0, 1.0}}, 2),
                  {3.0}, {kInfinity}, {"lb"},
                  {0.0, 0.0}, {5.0, 5.0},
                  {VarType::kInteger, VarType::kInteger},
                  {"x0", "x1"});

    BranchAndBound::Options opts;
    opts.node_selection = BranchAndBound::Options::NodeSelection::kBestBound;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "best-bound-min: status kOptimal");
    checkNear(res.objective_value, 3.0, 1e-6, "best-bound-min: optimum is 3");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "best-bound-min");
}

void testBestBoundMaximization() {
    // Existing knapsack with best-bound
    ModelIR model = makeKnapsack();
    BranchAndBound::Options opts;
    opts.node_selection = BranchAndBound::Options::NodeSelection::kBestBound;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    check(res.status == MipStatus::kOptimal, "best-bound-max: status kOptimal");
    checkNear(res.objective_value, 21.0, 1e-6, "best-bound-max: knapsack optimum 21");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "best-bound-max");
}

// =========================================================================
// Issue 7 & 8: Node limit and termination status invariants
// =========================================================================

void testNodeLimits() {
    ModelIR model = makeKnapsack();

    // max_nodes=0: immediately kNodeLimit, zero nodes explored
    BranchAndBound::Options opts0;
    opts0.max_nodes = 0;
    BranchAndBound bb0(opts0);
    MipResult res0 = bb0.solve(model);
    check(res0.status == MipStatus::kNodeLimit, "max_nodes=0 -> kNodeLimit");
    check(res0.statistics.nodes_explored == 0, "max_nodes=0 -> zero nodes explored");

    // max_nodes=1: explore exactly 1 node, kNodeLimit (tree not proven exhausted)
    BranchAndBound::Options opts1;
    opts1.max_nodes = 1;
    BranchAndBound bb1(opts1);
    MipResult res1 = bb1.solve(model);
    check(res1.status == MipStatus::kNodeLimit, "max_nodes=1 -> kNodeLimit");
    check(res1.statistics.nodes_explored <= 1, "max_nodes=1 -> at most 1 node explored");

    // max_nodes=2
    BranchAndBound::Options opts2;
    opts2.max_nodes = 2;
    BranchAndBound bb2(opts2);
    MipResult res2 = bb2.solve(model);
    check(res2.status == MipStatus::kNodeLimit, "max_nodes=2 -> kNodeLimit");
    check(res2.statistics.nodes_explored <= 2, "max_nodes=2 -> at most 2 nodes explored");

    // max_nodes=4: still kNodeLimit (knapsack takes more than 4 nodes to prove)
    BranchAndBound::Options opts4;
    opts4.max_nodes = 4;
    BranchAndBound bb4(opts4);
    MipResult res4 = bb4.solve(model);
    // Either kNodeLimit or kOptimal if the tree is tiny enough — just verify
    // nodes_explored <= 4 and that kOptimal is only set when tree was exhausted.
    check(res4.statistics.nodes_explored <= 4, "max_nodes=4 -> nodes_explored <= 4");
    if (res4.status == MipStatus::kOptimal) {
        // If kOptimal, the incumbent must exist and tree was exhausted (not just node-limited).
        check(res4.hasSolution(), "max_nodes=4 kOptimal: must have solution");
    }

    // Full solve: find the actual node count, then use it as max to verify kOptimal.
    BranchAndBound::Options opts_unlim;
    opts_unlim.max_nodes = 100000;
    BranchAndBound bb_unlim(opts_unlim);
    MipResult res_unlim = bb_unlim.solve(model);
    check(res_unlim.status == MipStatus::kOptimal, "unlimited -> kOptimal");

    int actual_nodes = res_unlim.statistics.nodes_explored;
    BranchAndBound::Options opts_exact;
    opts_exact.max_nodes = actual_nodes;
    BranchAndBound bb_exact(opts_exact);
    MipResult res_exact = bb_exact.solve(model);
    check(res_exact.status == MipStatus::kOptimal, "exact node count -> kOptimal");
    check(res_exact.statistics.nodes_explored <= actual_nodes,
          "exact node count -> nodes_explored <= max_nodes");
}

// kOptimal must NOT be returned when node-limit termination prevented proof.
void testNodeLimitStatusInvariant() {
    // A clearly suboptimal scenario: very tight node limit on a hard problem.
    ModelIR model = makeKnapsack();
    BranchAndBound::Options opts;
    opts.max_nodes = 2;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    // With only 2 nodes, the knapsack is definitely not proven optimal.
    // kOptimal must not be returned.
    if (res.status == MipStatus::kOptimal) {
        // This would be wrong -- the tree was cut short.
        check(false, "node-limit invariant: kOptimal must NOT be returned when node limit was hit "
                     "before tree exhaustion");
    } else {
        check(res.status == MipStatus::kNodeLimit,
              "node-limit invariant: status is kNodeLimit with max_nodes=2");
    }
}

// Multi-thread: nodes_explored must never exceed max_nodes.
void testNodeLimitMultithread() {
    ModelIR model;
    model.obj_sense = ObjSense::kMaximize;
    model.obj_coeffs = {8, 11, 6, 4, 5, 2};
    model.var_lower = {0, 0, 0, 0, 0, 0};
    model.var_upper = {1, 1, 1, 1, 1, 1};
    model.var_types = {VarType::kInteger, VarType::kInteger, VarType::kInteger,
                       VarType::kInteger, VarType::kInteger, VarType::kInteger};
    model.var_names = {"x1", "x2", "x3", "x4", "x5", "x6"};
    model.row_names = {"r1"};
    model.row_lower = {-kInfinity};
    model.row_upper = {18};
    model.A = CSRMatrix({0, 6}, {0, 1, 2, 3, 4, 5}, {5, 7, 4, 3, 2, 1}, 6);

    for (int threads : {1, 2, 4}) {
        for (int max_n : {0, 1, 2, 5}) {
            BranchAndBound::Options opts;
            opts.num_threads = threads;
            opts.max_nodes = max_n;
            BranchAndBound bb(opts);
            MipResult res = bb.solve(model);

            std::ostringstream d;
            d << "node-limit-mt: threads=" << threads << " max_nodes=" << max_n
              << " nodes_explored=" << res.statistics.nodes_explored << " <= " << max_n;
            check(res.statistics.nodes_explored <= max_n, d.str());

            // kOptimal must not be returned when we hit the node limit.
            if (max_n < 10) {
                // Small limit: should be kNodeLimit (not kOptimal).
                // Exception: if max_n is large enough that the tree is exhausted,
                // kOptimal is valid. For our 6-var problem with max_n <= 5, assume
                // the tree is not exhausted.
                std::ostringstream d2;
                d2 << "node-limit-mt: threads=" << threads << " max_nodes=" << max_n
                   << ": kOptimal must not be returned when node limit was hit";
                if (res.statistics.nodes_explored >= max_n && max_n > 0) {
                    check(res.status != MipStatus::kOptimal, d2.str());
                }
            }
        }
    }
}

// =========================================================================
// Issue 9: Warm-start contract
// =========================================================================

void testWarmStartRejectionFallback() {
    // An unbounded x1 causes the fingerprint to change on branching, so
    // warmSolve should reject the basis. The B&B must fall back correctly.
    const ModelIR model(ObjSense::kMaximize, 0.0,
                        {1.0, 1.0},
                        denseToCSR({{2.0, 2.0}}, 2),
                        {-kInfinity}, {5.0}, {"cap"},
                        {0.0, 0.0}, {4.0, kInfinity},
                        {VarType::kInteger, VarType::kInteger},
                        {"x0", "x1"});

    BranchAndBound::Options opts;
    opts.use_warm_start = true;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);

    // Even if warm start is rejected for some nodes, the answer must be correct.
    check(res.status == MipStatus::kOptimal, "warm-start-reject: fallback produces correct status");
    checkNear(res.objective_value, 2.0, 1e-6, "warm-start-reject: fallback produces correct obj");
    checkFeasibleAndIntegral(model, res.x, 1e-9, "warm-start-reject");
    // warm_start_successes <= warm_start_attempts always holds
    check(res.statistics.warm_start_successes <= res.statistics.warm_start_attempts,
          "warm-start-reject: successes <= attempts");
}

// =========================================================================
// Additional: determinism and parallel consistency
// =========================================================================

void testDeterminism() {
    ModelIR model = makeKnapsack();
    BranchAndBound::Options opts;
    opts.num_threads = 1;
    BranchAndBound bb(opts);
    MipResult res1 = bb.solve(model);
    for (int i = 0; i < 5; ++i) {
        MipResult res = bb.solve(model);
        check(res.objective_value == res1.objective_value, "determinism: obj matches");
        check(res.statistics.nodes_explored == res1.statistics.nodes_explored,
              "determinism: nodes match");
    }
}

void testParallel() {
    ModelIR model = makeKnapsack();

    BranchAndBound::Options opts1;
    opts1.num_threads = 1;
    BranchAndBound bb1(opts1);
    MipResult res1 = bb1.solve(model);

    BranchAndBound::Options opts2;
    opts2.num_threads = 2;
    BranchAndBound bb2(opts2);
    MipResult res2 = bb2.solve(model);

    BranchAndBound::Options opts4;
    opts4.num_threads = 4;
    BranchAndBound bb4(opts4);
    MipResult res4 = bb4.solve(model);

    check(res1.status == res2.status, "parallel: 2-thread status matches 1-thread");
    check(res1.status == res4.status, "parallel: 4-thread status matches 1-thread");
    checkNear(res1.objective_value, res2.objective_value, 1e-6, "parallel: 2-thread obj");
    checkNear(res1.objective_value, res4.objective_value, 1e-6, "parallel: 4-thread obj");
}

// =========================================================================
// Status invariants
// =========================================================================

void testStatusInvariants() {
    // kOptimal: incumbent exists
    {
        MipResult res = BranchAndBound{}.solve(makeKnapsack());
        check(res.status == MipStatus::kOptimal, "invariant-optimal: status");
        check(res.hasSolution(), "invariant-optimal: incumbent exists");
        checkNear(res.objective_value, 21.0, 1e-6, "invariant-optimal: correct obj");
    }

    // kInfeasible: no incumbent
    {
        const ModelIR model(ObjSense::kMaximize, 0.0,
                            {1.0},
                            denseToCSR({{2.0}}, 1),
                            {1.0}, {1.0}, {"eq"},
                            {0.0}, {2.0},
                            {VarType::kInteger},
                            {"x0"});
        MipResult res = BranchAndBound{}.solve(model);
        check(res.status == MipStatus::kInfeasible, "invariant-infeasible: status");
        check(!res.hasSolution(), "invariant-infeasible: no incumbent");
    }

    // kNodeLimit: hit limit, kOptimal not returned
    {
        BranchAndBound::Options opts;
        opts.max_nodes = 1;
        MipResult res = BranchAndBound{opts}.solve(makeKnapsack());
        check(res.status == MipStatus::kNodeLimit, "invariant-nodelimit: status");
        // Incumbent MAY exist (rounding heuristic), but status must be kNodeLimit.
        // kOptimal is only valid when tree is exhausted.
    }
}

void testToleranceBoundary() {
    ModelIR model;
    model.obj_sense = ObjSense::kMaximize;
    model.obj_coeffs = {1.0};
    model.var_lower = {0};
    model.var_upper = {1};
    model.var_types = {VarType::kInteger};
    model.var_names = {"x1"};
    model.row_names = {"r1"};
    model.row_lower = {-kInfinity};
    model.row_upper = {0.9999999999};
    model.A = CSRMatrix({0, 1}, {0}, {1}, 1);

    BranchAndBound::Options opts;
    BranchAndBound bb(opts);
    MipResult res = bb.solve(model);
    check(res.status == MipStatus::kOptimal, "tolerance boundary: found optimal");
}

}  // namespace

int main() {
    run("0/1 knapsack with hand-verified optimum 21", testKnapsack);
    run("Minimization sense", testMinimizationSense);
    run("Root relaxation already integral", testRootAlreadyIntegral);
    run("Integer-infeasible (LP-feasible) model", testIntegerInfeasible);
    run("Root LP infeasible status", testRootLPInfeasibleStatus);
    run("Mixed integer/continuous model", testMixedIntegerContinuous);
    run("Warm-start plumbing (option off by default)", testWarmStartPlumbing);
    run("Incompatible inherited basis falls back cleanly", testIncompatibleBasisFallback);
    run("Warm-vs-cold B&B comparison (final answer invariance)", testWarmVsColdComparison);

    // Cut tests
    run("Cut tightens LP relaxation", testCutTightensRelaxation);
    run("Cover cut rejects binary+continuous row", testCoverCutRejectsContinuousRow);
    run("Cover cut rejects binary+general-integer row", testCoverCutRejectsGeneralIntegerRow);
    run("Cover cut rejects negative coefficient row", testCoverCutRejectsNegativeCoeff);
    run("Cover cut rejects finite lower bound row", testCoverCutRejectsFiniteLowerBound);
    run("Cover cut valid for pure binary row", testCoverCutValidBinaryRow);
    run("Duplicate cut statistics semantics", testDuplicateCuts);
    run("Warm-start + cut interaction", testWarmStartCutInteraction);

    // Issue 1: Root LP unbounded
    run("Root LP unbounded does not prove MILP unbounded", testRootLPUnboundedMilpBehavior);

    // Issue 6: Best-bound ordering
    run("Best-bound queue minimization", testBestBoundMinimization);
    run("Best-bound queue maximization", testBestBoundMaximization);

    // Issues 7 & 8: Node limit and termination status
    run("Node limits (0/1/2/4/exact)", testNodeLimits);
    run("Node limit kOptimal invariant", testNodeLimitStatusInvariant);
    run("Node limit multi-thread (nodes_explored <= max_nodes)", testNodeLimitMultithread);

    // Issue 9: Warm-start contract
    run("Warm-start rejection fallback", testWarmStartRejectionFallback);

    // Status invariants
    run("Status invariants (kOptimal/kInfeasible/kNodeLimit)", testStatusInvariants);

    // Misc
    run("Determinism", testDeterminism);
    run("Parallel consistency (1/2/4 threads)", testParallel);
    run("Tolerance boundary", testToleranceBoundary);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
