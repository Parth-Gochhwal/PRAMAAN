// test_dual_simplex.cpp
// P1 Step 1 — Dual Simplex + Basis Warm-Start correctness tests
//
// Tests:
// 1. Wyndor warm-start (mandatory): cold solve, capture basis, tighten bound,
//    warm-solve, cold-solve the modified LP, verify identical objectives
//    and solutions, and verify warm pivots < cold pivots.
// 2. Unchanged LP + optimal basis: warm resolve recognizes the existing
//    optimum with zero pivots.
// 3. Small RHS perturbation: warm and cold solutions/objectives agree.
// 4. Small bound tightening: warm and cold solutions/objectives agree.
// 5. Incompatible cached basis: reject safely.
// 6. Infeasible modification: verify correct infeasibility detection.
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/dual_simplex.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::RevisedSimplex;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::VarType;
using pramaan::BasisState;
using pramaan::DualSimplex;

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

// Wyndor Glass Co. LP:
//   maximize   3 x1 + 5 x2
//   subject to      x1      <= 4
//                  2 x2     <= 12
//              3 x1 + 2 x2  <= 18
//              0 <= x1 <= 10,  0 <= x2 <= 10
//
// The finite var_upper values ensure the standard-form includes bound rows
// from the start, so tightening bounds in warm-start tests doesn't change
// the standard-form dimension.  The bounds are loose enough not to affect
// the original optimal: x1=2, x2=6, obj=36.
ModelIR makeWyndor() {
    return ModelIR(
        ObjSense::kMaximize, 0.0, {3.0, 5.0},
        denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
        {-kInfinity, -kInfinity, -kInfinity},
        {4.0, 12.0, 18.0},
        {"plant1", "plant2", "plant3"},
        {0.0, 0.0}, {10.0, 10.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});
}

}  // namespace

// =========================================================================
// Test 1 (Mandatory): Wyndor warm-start
// =========================================================================
void testWyndorWarmStart() {
    ModelIR original = makeWyndor();

    // Step 1: Cold-solve the original
    DualSimplex ds;
    BasisState basis = ds.captureBasis(original);
    check(!basis.empty(), "Wyndor: captured basis is non-empty");

    // Also get the cold solution for reference
    RevisedSimplex cold_solver;
    SolveResult cold_original = cold_solver.solve(original);
    check(cold_original.status == SolveStatus::kOptimal, "Wyndor original: cold solve optimal");
    checkNear(cold_original.objective_value, 36.0, 1e-6, "Wyndor original: obj = 36");

    std::cout << "  Original cold obj = " << cold_original.objective_value
              << ", iters = " << cold_original.iterations << "\n";

    // Step 2: Tighten x1 upper bound from 10 to 1.5
    // This changes the optimal from x1=2,x2=6,obj=36 to something new
    // that requires dual simplex pivots.
    // With x1 <= 1.5:  x1=1.5, and 3(1.5)+2*x2 <= 18 => x2 <= 6.75,
    //   but 2*x2 <= 12 => x2 <= 6, so x2=6.
    //   obj = 3*1.5 + 5*6 = 4.5 + 30 = 34.5.
    ModelIR modified = original;
    modified.var_upper[0] = 1.5;

    // Step 3: Warm-solve from inherited basis
    SolveResult warm_result = ds.warmSolve(modified, basis);
    check(warm_result.status == SolveStatus::kOptimal, "Wyndor warm: status optimal");

    // Step 4: Independent cold-solve the modified LP
    SolveResult cold_modified = cold_solver.solve(modified);
    check(cold_modified.status == SolveStatus::kOptimal, "Wyndor cold modified: status optimal");

    if (warm_result.status == SolveStatus::kOptimal && cold_modified.status == SolveStatus::kOptimal) {
        // Verify objectives agree
        checkNear(warm_result.objective_value, cold_modified.objective_value, 1e-6,
                  "Wyndor: warm obj == cold obj");
        checkNear(warm_result.objective_value, 34.5, 1e-6, "Wyndor: modified obj = 34.5");

        // Verify solutions agree
        checkNear(warm_result.x[0], cold_modified.x[0], 1e-6, "Wyndor: warm x1 == cold x1");
        checkNear(warm_result.x[1], cold_modified.x[1], 1e-6, "Wyndor: warm x2 == cold x2");
        checkNear(warm_result.x[0], 1.5, 1e-6, "Wyndor: x1 = 1.5");
        checkNear(warm_result.x[1], 6.0, 1e-6, "Wyndor: x2 = 6.0");

        // Verify feasibility
        checkFeasible(modified, warm_result, 1e-6, "Wyndor warm");
        checkFeasible(modified, cold_modified, 1e-6, "Wyndor cold modified");

        // Verify warm pivots < cold pivots
        std::cout << "  Modified cold obj = " << cold_modified.objective_value
                  << ", iters = " << cold_modified.iterations << "\n";
        std::cout << "  Warm obj = " << warm_result.objective_value
                  << ", iters = " << warm_result.iterations << "\n";
        check(warm_result.iterations < cold_modified.iterations,
              "Wyndor: warm pivots (" + std::to_string(warm_result.iterations) +
              ") < cold pivots (" + std::to_string(cold_modified.iterations) + ")");

        // Verify warm solve actually used dual pivots (not just returned
        // the old solution). Since x1 was tightened from 10 to 1.5,
        // the new solution is different.
        check(std::abs(warm_result.x[0] - cold_original.x[0]) > 0.1,
              "Wyndor: warm solve actually changed the solution (not a silent fallback)");
    }
}

// =========================================================================
// Test 2: Unchanged LP + optimal basis → zero pivots
// =========================================================================
void testUnchangedLP() {
    ModelIR model = makeWyndor();

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "Unchanged: captured basis is non-empty");

    // Warm-solve the exact same model — should be zero pivots.
    SolveResult warm = ds.warmSolve(model, basis);
    check(warm.status == SolveStatus::kOptimal, "Unchanged: warm solve optimal");
    if (warm.status == SolveStatus::kOptimal) {
        checkNear(warm.objective_value, 36.0, 1e-6, "Unchanged: obj = 36");
        checkNear(warm.x[0], 2.0, 1e-6, "Unchanged: x1 = 2");
        checkNear(warm.x[1], 6.0, 1e-6, "Unchanged: x2 = 6");
        check(warm.iterations == 0,
              "Unchanged: zero pivots (got " + std::to_string(warm.iterations) + ")");
        checkFeasible(model, warm, 1e-6, "Unchanged");
    }
}

// =========================================================================
// Test 3: Small RHS perturbation
// =========================================================================
void testRHSPerturbation() {
    ModelIR model = makeWyndor();

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "RHS: captured basis is non-empty");

    // Tighten row 2 (plant3: 3x1 + 2x2 <= 18) to 3x1 + 2x2 <= 16
    ModelIR modified = model;
    modified.row_upper[2] = 16.0;

    SolveResult warm = ds.warmSolve(modified, basis);
    check(warm.status == SolveStatus::kOptimal, "RHS: warm solve optimal");

    RevisedSimplex cold_solver;
    SolveResult cold = cold_solver.solve(modified);
    check(cold.status == SolveStatus::kOptimal, "RHS: cold solve optimal");

    if (warm.status == SolveStatus::kOptimal && cold.status == SolveStatus::kOptimal) {
        checkNear(warm.objective_value, cold.objective_value, 1e-6, "RHS: warm obj == cold obj");
        for (size_t j = 0; j < warm.x.size(); ++j) {
            checkNear(warm.x[j], cold.x[j], 1e-6,
                      "RHS: warm x[" + std::to_string(j) + "] == cold x[" + std::to_string(j) + "]");
        }
        checkFeasible(modified, warm, 1e-6, "RHS warm");
        checkFeasible(modified, cold, 1e-6, "RHS cold");
        std::cout << "  RHS warm obj = " << warm.objective_value << ", iters = " << warm.iterations << "\n";
        std::cout << "  RHS cold obj = " << cold.objective_value << ", iters = " << cold.iterations << "\n";
    }
}

// =========================================================================
// Test 4: Small bound tightening (different from Test 1's Wyndor)
// =========================================================================
void testBoundTightening() {
    // A 3-variable LP:
    //   minimize  2x1 + 3x2 + x3
    //   s.t.      x1 + x2 + x3  >= 10
    //            2x1 + x2       <= 20
    //                  x2 + 2x3 <= 16
    //            0 <= x1,x2,x3 <= 20
    ModelIR original(
        ObjSense::kMinimize, 0.0, {2.0, 3.0, 1.0},
        denseToCSR({{1.0, 1.0, 1.0}, {2.0, 1.0, 0.0}, {0.0, 1.0, 2.0}}, 3),
        {10.0, -kInfinity, -kInfinity},
        {kInfinity, 20.0, 16.0},
        {"demand", "cap1", "cap2"},
        {0.0, 0.0, 0.0}, {20.0, 20.0, 20.0},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(original);
    check(!basis.empty(), "BoundTighten: captured basis is non-empty");

    // Tighten x3 <= 3
    ModelIR modified = original;
    modified.var_upper[2] = 3.0;

    SolveResult warm = ds.warmSolve(modified, basis);
    check(warm.status == SolveStatus::kOptimal, "BoundTighten: warm solve optimal");

    RevisedSimplex cold_solver;
    SolveResult cold = cold_solver.solve(modified);
    check(cold.status == SolveStatus::kOptimal, "BoundTighten: cold solve optimal");

    if (warm.status == SolveStatus::kOptimal && cold.status == SolveStatus::kOptimal) {
        checkNear(warm.objective_value, cold.objective_value, 1e-5, "BoundTighten: warm obj == cold obj");
        checkFeasible(modified, warm, 1e-6, "BoundTighten warm");
        checkFeasible(modified, cold, 1e-6, "BoundTighten cold");
        std::cout << "  Bound warm obj = " << warm.objective_value << ", iters = " << warm.iterations << "\n";
        std::cout << "  Bound cold obj = " << cold.objective_value << ", iters = " << cold.iterations << "\n";
    }
}

// =========================================================================
// Test 5: Incompatible cached basis (Dimensions, Objective, Matrix)
// =========================================================================
void testIncompatibleBasis() {
    ModelIR model = makeWyndor();

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "IncompatBasis: captured basis is non-empty");

    // Dimension mismatch
    ModelIR different_dim(
        ObjSense::kMinimize, 0.0, {1.0},
        denseToCSR({{1.0}}, 1),
        {-kInfinity}, {5.0},
        {"row1"},
        {0.0}, {10.0},
        {VarType::kContinuous},
        {"x"});
    bool threw = false;
    try {
        ds.warmSolve(different_dim, basis);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "IncompatBasis: throws on dimension mismatch");

    // Objective coefficient mismatch
    ModelIR different_obj = model;
    different_obj.obj_coeffs[0] = 99.0;
    threw = false;
    try {
        ds.warmSolve(different_obj, basis);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "IncompatBasis: throws on objective mismatch");

    // Matrix coefficient mismatch
    ModelIR different_mat = model;
    different_mat.A = denseToCSR({{1.0, 0.0}, {0.0, 99.0}, {3.0, 2.0}}, 2);
    threw = false;
    try {
        ds.warmSolve(different_mat, basis);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "IncompatBasis: throws on matrix mismatch");

    // Empty basis
    BasisState empty;
    bool threw2 = false;
    try {
        ds.warmSolve(model, empty);
    } catch (const std::invalid_argument&) {
        threw2 = true;
    }
    check(threw2, "IncompatBasis: throws on empty basis");
}

// =========================================================================
// Test 6: Infeasible modification
// =========================================================================
void testInfeasibleModification() {
    // Use a model where infeasibility can be created by tightening bounds
    // without changing the standard-form structure.
    //
    //   maximize x1 + x2
    //   s.t.  x1 + x2 >= 5    (row_lower = 5, row_upper = inf)
    //         x1 + x2 <= 10   (row_lower = -inf, row_upper = 10)
    //         0 <= x1 <= 8,   0 <= x2 <= 8
    //
    // Optimal: x1+x2 = 10, obj = 10.
    ModelIR model(
        ObjSense::kMaximize, 0.0, {1.0, 1.0},
        denseToCSR({{1.0, 1.0}, {1.0, 1.0}}, 2),
        {5.0, -kInfinity},
        {kInfinity, 10.0},
        {"floor", "cap"},
        {0.0, 0.0}, {8.0, 8.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "Infeasible: captured basis is non-empty");

    // Make it infeasible: x1 <= 2 AND x2 <= 2 => x1+x2 <= 4 < 5 (the floor).
    ModelIR infeasible = model;
    infeasible.var_upper[0] = 2.0;
    infeasible.var_upper[1] = 2.0;

    SolveResult warm = ds.warmSolve(infeasible, basis);
    check(warm.status == SolveStatus::kInfeasible,
          "Infeasible: warm solve detects infeasibility (got " +
          std::to_string(static_cast<int>(warm.status)) + ")");
}

// =========================================================================
// Test 7: Redundant Equality (Artificial Variable safely handled)
// =========================================================================
void testRedundantEquality() {
    // minimize x1
    // s.t. x1 = 2
    //      x1 = 2 (redundant)
    ModelIR model(
        ObjSense::kMinimize, 0.0, {1.0},
        denseToCSR({{1.0}, {1.0}}, 1),
        {2.0, 2.0}, {2.0, 2.0},
        {"eq1", "eq2"},
        {0.0}, {10.0},
        {VarType::kContinuous},
        {"x1"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    // Because the second equality is redundant and identical, an artificial
    // variable will remain in the basis at 0 after Phase 1.
    // The implementation should explicitly and safely reject this basis rather
    // than risk using an unresolved artificial variable during warm start.
    check(basis.empty(), "RedundantEquality: basis with unresolved artificial is safely rejected");
}

// =========================================================================
// Test 8: Finite <-> Infinite bound change rejection
// =========================================================================
void testBoundTypeChangeRejection() {
    ModelIR model = makeWyndor(); // has finite upper bounds (10)
    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);

    // Change finite bound to infinite (adds/removes a row in standard form)
    ModelIR modified = model;
    modified.var_upper[0] = kInfinity;

    bool threw = false;
    try {
        ds.warmSolve(modified, basis);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "BoundTypeChange: throws on finite -> infinite bound change");
}

// =========================================================================
// Test 9: Dual Infeasible Basis Rejection
// =========================================================================
void testDualInfeasibleBasisRejection() {
    // We deterministically craft a structurally valid but dual-INFEASIBLE basis.
    // Dual simplex requires a dual-feasible basis.
    // When it detects dual infeasibility at initialization, it must safely
    // abort with kNumericalFailure.
    // Note: This exercises the initial dual-feasibility check.
    // Exercising the dynamic pivot-recovery path (`u_leaving >= -tol`) via
    // purely deterministic round-off cancellation without memory corruption (NaNs)
    // is highly compiler/architecture dependent, thus we omit a separate
    // fragile regression for it per requirements.
    ModelIR model(
        ObjSense::kMinimize, 0.0, {-1.0, 1.0}, // x1 is profitable
        denseToCSR({{1.0, 0.0}, {0.0, 1.0}}, 2),
        {-kInfinity, -kInfinity}, {10.0, 10.0}, // inequality rows!
        {"r1", "r2"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    DualSimplex ds;
    BasisState bad_basis = ds.captureBasis(model);
    check(!bad_basis.empty(), "InfeasibleBasis: captured basis");

    // We tamper with the columns to make it the slack basis {2, 3}.
    // Since rows are inequalities, slacks are NOT artificial.
    // This allows it to pass the initial artificial check and hit the
    // dual-feasibility check, which should return kNumericalFailure.
    bad_basis.basis_columns = {2, 3};

    SolveResult warm = ds.warmSolve(model, bad_basis);
    check(warm.status == SolveStatus::kNumericalFailure,
          "InfeasibleBasis: dual-infeasible inherited basis safely triggers kNumericalFailure");
}

// =========================================================================
// Test 10: Artificial column rejection in warmSolve
// =========================================================================
void testArtificialBasisRejectionWarmSolve() {
    ModelIR model(
        ObjSense::kMinimize, 0.0, {1.0, 1.0},
        denseToCSR({{1.0, 0.0}, {0.0, 1.0}}, 2),
        {0.0, 0.0}, {0.0, 0.0}, // equality rows
        {"r1", "r2"},
        {0.0, 0.0}, {10.0, 10.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    DualSimplex ds;
    BasisState bad_basis = ds.captureBasis(model);
    check(!bad_basis.empty(), "ArtificialReject: captured basis");

    // Force an artificial variable into the basis (slacks for equalities are artificial)
    bad_basis.basis_columns = {0, 2};

    bool threw = false;
    try {
        ds.warmSolve(model, bad_basis);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "ArtificialReject: warmSolve rejects artificial columns");
}

// =========================================================================
// Test 11: Knapsack warm-start regression (the exact 19-vs-21 failure)
// =========================================================================
// This is the B&B regression: a 0/1 knapsack LP relaxation cold-solved,
// then warm-solved with a tightened variable upper bound. Previously,
// warmSolve returned a suboptimal kOptimal result because the dual-simplex
// pivot formula incorrectly set x_B[leaving]=0 instead of theta.
void testKnapsackWarmStartRegression() {
    // maximize 8x0 + 11x1 + 6x2 + 4x3  s.t.  5x0+7x1+4x2+3x3 <= 14,  0<=xi<=1
    ModelIR model(
        ObjSense::kMaximize, 0.0, {8.0, 11.0, 6.0, 4.0},
        denseToCSR({{5.0, 7.0, 4.0, 3.0}}, 4),
        {-kInfinity}, {14.0}, {"capacity"},
        {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x0", "x1", "x2", "x3"});

    DualSimplex ds;
    SolveResult cold_original;
    BasisState basis = ds.captureBasis(model, &cold_original);
    check(!basis.empty(), "Knapsack: captured basis is non-empty");
    check(cold_original.status == SolveStatus::kOptimal, "Knapsack: root LP optimal");
    checkNear(cold_original.objective_value, 22.0, 1e-6, "Knapsack: root LP obj=22");

    RevisedSimplex rs;

    // Case A: x0 <= 0  (down branch, was the B&B path that gave 19 instead of 21)
    {
        ModelIR mod = model;
        mod.var_upper[0] = 0.0;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal, "Knapsack x0<=0: warm optimal");
        check(cold.status == SolveStatus::kOptimal, "Knapsack x0<=0: cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "Knapsack x0<=0: warm obj == cold obj");
        checkNear(warm.objective_value, 21.0, 1e-6, "Knapsack x0<=0: obj=21");
        checkFeasible(mod, warm, 1e-6, "Knapsack x0<=0 warm");
        for (size_t j = 0; j < warm.x.size(); ++j) {
            checkNear(warm.x[j], cold.x[j], 1e-6,
                      "Knapsack x0<=0: warm x[" + std::to_string(j) + "] == cold");
        }
        std::cout << "  x0<=0: warm obj=" << warm.objective_value
                  << ", iters=" << warm.iterations << "\n";
    }

    // Case B: x1 <= 0  (previously returned kInfeasible incorrectly)
    {
        ModelIR mod = model;
        mod.var_upper[1] = 0.0;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal, "Knapsack x1<=0: warm optimal");
        check(cold.status == SolveStatus::kOptimal, "Knapsack x1<=0: cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "Knapsack x1<=0: warm obj == cold obj");
        checkNear(warm.objective_value, 18.0, 1e-6, "Knapsack x1<=0: obj=18");
        checkFeasible(mod, warm, 1e-6, "Knapsack x1<=0 warm");
        std::cout << "  x1<=0: warm obj=" << warm.objective_value
                  << ", iters=" << warm.iterations << "\n";
    }

    // Case C: x2 <= 0.25 (the minimal reproducer from B&B comments)
    {
        ModelIR mod = model;
        mod.var_upper[2] = 0.25;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal, "Knapsack x2<=0.25: warm optimal");
        check(cold.status == SolveStatus::kOptimal, "Knapsack x2<=0.25: cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "Knapsack x2<=0.25: warm obj == cold obj");
        checkFeasible(mod, warm, 1e-6, "Knapsack x2<=0.25 warm");
        std::cout << "  x2<=0.25: warm obj=" << warm.objective_value
                  << ", iters=" << warm.iterations << "\n";
    }

    // Case D: x3 <= 0
    {
        ModelIR mod = model;
        mod.var_upper[3] = 0.0;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal, "Knapsack x3<=0: warm optimal");
        check(cold.status == SolveStatus::kOptimal, "Knapsack x3<=0: cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "Knapsack x3<=0: warm obj == cold obj");
        checkFeasible(mod, warm, 1e-6, "Knapsack x3<=0 warm");
        std::cout << "  x3<=0: warm obj=" << warm.objective_value
                  << ", iters=" << warm.iterations << "\n";
    }

    // Case E: Multiple bound tightenings (x0<=0.5 AND x3<=0.5)
    {
        ModelIR mod = model;
        mod.var_upper[0] = 0.5;
        mod.var_upper[3] = 0.5;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal, "Knapsack x0<=0.5,x3<=0.5: warm optimal");
        check(cold.status == SolveStatus::kOptimal, "Knapsack x0<=0.5,x3<=0.5: cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "Knapsack x0<=0.5,x3<=0.5: warm obj == cold obj");
        checkFeasible(mod, warm, 1e-6, "Knapsack x0<=0.5,x3<=0.5 warm");
        std::cout << "  x0<=0.5,x3<=0.5: warm obj=" << warm.objective_value
                  << ", iters=" << warm.iterations << "\n";
    }
}

// =========================================================================
// Test 12: Warm-start chaining
// =========================================================================
// modified model A -> warm solve -> capture resulting basis -> modified model B -> warm solve
void testWarmStartChaining() {
    ModelIR original = makeWyndor();

    DualSimplex ds;
    SolveResult cold_orig;
    BasisState basis_A = ds.captureBasis(original, &cold_orig);
    check(!basis_A.empty(), "Chain: initial basis non-empty");

    // Chain step 1: tighten x1 upper to 1.5
    ModelIR model_A = original;
    model_A.var_upper[0] = 1.5;

    BasisState basis_B;
    SolveResult warm_A = ds.warmSolve(model_A, basis_A, &basis_B);
    check(warm_A.status == SolveStatus::kOptimal, "Chain step 1: optimal");
    checkNear(warm_A.objective_value, 34.5, 1e-6, "Chain step 1: obj=34.5");
    check(!basis_B.empty(), "Chain step 1: output basis non-empty");
    std::cout << "  Step 1: obj=" << warm_A.objective_value
              << ", iters=" << warm_A.iterations << "\n";

    // Chain step 2: further tighten x1 upper to 0.5 using basis from step 1
    ModelIR model_B = original;
    model_B.var_upper[0] = 0.5;

    SolveResult warm_B = ds.warmSolve(model_B, basis_B);
    RevisedSimplex rs;
    SolveResult cold_B = rs.solve(model_B);
    check(warm_B.status == SolveStatus::kOptimal, "Chain step 2: warm optimal");
    check(cold_B.status == SolveStatus::kOptimal, "Chain step 2: cold optimal");
    checkNear(warm_B.objective_value, cold_B.objective_value, 1e-6,
              "Chain step 2: warm obj == cold obj");
    checkFeasible(model_B, warm_B, 1e-6, "Chain step 2 warm");
    std::cout << "  Step 2: obj=" << warm_B.objective_value
              << ", iters=" << warm_B.iterations << "\n";
}

// =========================================================================
// Test 13: Warm result matches cold result
// =========================================================================
// Comprehensive check across multiple perturbations of the Wyndor model.
void testWarmMatchesCold() {
    ModelIR original = makeWyndor();
    DualSimplex ds;
    BasisState basis = ds.captureBasis(original);
    check(!basis.empty(), "WarmCold: basis non-empty");
    RevisedSimplex rs;

    // Several bound perturbations
    struct Case { int var; double new_ub; std::string label; };
    std::vector<Case> cases = {
        {0, 1.0, "x1<=1"},
        {0, 0.5, "x1<=0.5"},
        {1, 5.0, "x2<=5"},
        {1, 3.0, "x2<=3"},
    };

    for (const auto& c : cases) {
        ModelIR mod = original;
        mod.var_upper[static_cast<std::size_t>(c.var)] = c.new_ub;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal,
              "WarmCold " + c.label + ": warm optimal");
        check(cold.status == SolveStatus::kOptimal,
              "WarmCold " + c.label + ": cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "WarmCold " + c.label + ": objectives match");
        checkFeasible(mod, warm, 1e-6, "WarmCold " + c.label + " warm");
        for (size_t j = 0; j < warm.x.size(); ++j) {
            checkNear(warm.x[j], cold.x[j], 1e-6,
                      "WarmCold " + c.label + " x[" + std::to_string(j) + "]");
        }
    }

    // Several RHS perturbations
    struct RCase { int row; double new_ub; std::string label; };
    std::vector<RCase> rcases = {
        {0, 3.0, "r0<=3"},
        {1, 10.0, "r1<=10"},
        {2, 15.0, "r2<=15"},
    };

    for (const auto& c : rcases) {
        ModelIR mod = original;
        mod.row_upper[static_cast<std::size_t>(c.row)] = c.new_ub;
        SolveResult warm = ds.warmSolve(mod, basis);
        SolveResult cold = rs.solve(mod);
        check(warm.status == SolveStatus::kOptimal,
              "WarmCold " + c.label + ": warm optimal");
        check(cold.status == SolveStatus::kOptimal,
              "WarmCold " + c.label + ": cold optimal");
        checkNear(warm.objective_value, cold.objective_value, 1e-6,
                  "WarmCold " + c.label + ": objectives match");
        checkFeasible(mod, warm, 1e-6, "WarmCold " + c.label + " warm");
    }
}

// =========================================================================
// Test 14: RHS Sign Flip Rejection
// =========================================================================
// Changing a finite RHS across zero can cause buildSparseStandardForm to flip
// the row's standard-form sign and RowKind (e.g., kLessEqual -> kGreaterEqual),
// adding new structural columns (slack -> surplus + artificial).
// The warm path must detect this layout change and cleanly reject the basis.
void testRHSSignFlipRejection() {
    // x0 - x1 <= 5 (RHS > 0, standard form: x0 - x1 + s = 5)
    ModelIR model(ObjSense::kMaximize, 0.0,
                   {1.0, 1.0},
                   denseToCSR({{1.0, -1.0}}, 2),
                   {-kInfinity}, {5.0}, {"cap"},
                   {0.0, 0.0}, {10.0, 10.0},
                   {VarType::kContinuous, VarType::kContinuous},
                   {"x0", "x1"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "RHSSignFlip: original basis captured");

    // Modify RHS to -5. x0 - x1 <= -5 is equivalent to -x0 + x1 >= 5.
    // Standard form: -x0 + x1 - surplus + artificial = 5.
    ModelIR modified = model;
    modified.row_upper[0] = -5.0;

    bool threw = false;
    try {
        ds.warmSolve(modified, basis);
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        if (msg.find("layout mismatch") != std::string::npos) {
            threw = true;
        }
    }

    check(threw, "RHSSignFlip: warmSolve cleanly rejected the structurally shifted standard-form matrix");
}

// =========================================================================
// Test 15: Equality RHS Sign Flip Rejection
// =========================================================================
// For equality rows, changing RHS across zero negates the standard-form A
// coefficients but does NOT change the number of columns (adds exactly 1 artificial).
// The full std_layout_fingerprint (which hashes A.values) must catch this and reject.
void testEqualityRHSSignFlipRejection() {
    ModelIR model(ObjSense::kMaximize, 0.0,
                   {1.0, 1.0},
                   denseToCSR({{1.0, 1.0}}, 2),
                   {5.0}, {5.0}, {"eq"},
                   {0.0, 0.0}, {10.0, 10.0},
                   {VarType::kContinuous, VarType::kContinuous},
                   {"x0", "x1"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "EqRHSSignFlip: original basis captured");

    // Modify RHS to -5. The coefficients in sf.A will be negated.
    ModelIR modified = model;
    modified.row_lower[0] = -5.0;
    modified.row_upper[0] = -5.0;

    bool threw = false;
    try {
        ds.warmSolve(modified, basis);
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        if (msg.find("layout mismatch") != std::string::npos) {
            threw = true;
        }
    }

    check(threw, "EqRHSSignFlip: warmSolve cleanly rejected the structurally shifted (negated) standard-form matrix");
}

// =========================================================================
// Test 16: Ordinary RHS Perturbation (No Sign Flip)
// =========================================================================
// Verify that an ordinary RHS perturbation that does NOT cross zero preserves
// the layout fingerprint and solves successfully.
void testOrdinaryRHSPerturbation() {
    ModelIR model(ObjSense::kMaximize, 0.0,
                   {1.0, 1.0},
                   denseToCSR({{1.0, 1.0}}, 2),
                   {-kInfinity}, {5.0}, {"cap"},
                   {0.0, 0.0}, {10.0, 10.0},
                   {VarType::kContinuous, VarType::kContinuous},
                   {"x0", "x1"});

    DualSimplex ds;
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "OrdRHS: original basis captured");

    // Modify RHS to 2. No sign flip occurs (2 > 0).
    ModelIR modified = model;
    modified.row_upper[0] = 2.0;

    RevisedSimplex rs;
    SolveResult cold = rs.solve(modified);
    SolveResult warm = ds.warmSolve(modified, basis);

    check(warm.status == SolveStatus::kOptimal, "OrdRHS: warm optimal");
    check(cold.status == SolveStatus::kOptimal, "OrdRHS: cold optimal");
    checkNear(warm.objective_value, cold.objective_value, 1e-6, "OrdRHS: objectives match");
    checkFeasible(modified, warm, 1e-6, "OrdRHS warm");
}

int main() {
    run("Wyndor warm-start (mandatory)", testWyndorWarmStart);
    run("Unchanged LP + optimal basis → zero pivots", testUnchangedLP);
    run("Small RHS perturbation", testRHSPerturbation);
    run("Small bound tightening", testBoundTightening);
    run("Incompatible cached basis", testIncompatibleBasis);
    run("Infeasible modification", testInfeasibleModification);
    run("Redundant Equality", testRedundantEquality);
    run("Finite <-> Infinite bound change rejection", testBoundTypeChangeRejection);
    run("Dual Infeasible Basis Rejection", testDualInfeasibleBasisRejection);
    run("Artificial Rejection in warmSolve", testArtificialBasisRejectionWarmSolve);
    run("Knapsack warm-start regression (19 vs 21)", testKnapsackWarmStartRegression);
    run("Warm-start chaining", testWarmStartChaining);
    run("Warm result matches cold result", testWarmMatchesCold);
    run("RHS Sign Flip Rejection", testRHSSignFlipRejection);
    run("Equality RHS Sign Flip Rejection", testEqualityRHSSignFlipRejection);
    run("Ordinary RHS Perturbation (No Sign Flip)", testOrdinaryRHSPerturbation);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
