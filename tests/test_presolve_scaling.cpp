// test_presolve_scaling.cpp
//
// Stage 6 correctness tests for:
//   - fixed-variable presolve + ledger recording + postsolve reconstruction
//   - Ruiz scaling + unscaling
//   - full pipeline: presolve -> scale -> solve -> unscale -> postsolve
//   - AFIRO end-to-end with presolve+scaling ON vs OFF comparison
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/presolve.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/transformation_ledger.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::ColumnScaled;
using pramaan::CumulativeScaling;
using pramaan::FixedVariableRemoved;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::Reduction;
using pramaan::RevisedSimplex;
using pramaan::RowScaled;
using pramaan::ScalingFactors;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::TransformationLedger;
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

// =========================================================================
// Test: fixed-variable elimination
// =========================================================================
void testFixedVariableElimination() {
    std::cout << "testFixedVariableElimination...\n";

    // min  3*x1 + 2*x2 + 5*x3
    // s.t. x1 + x2 + x3 <= 10
    //      x1 = 2 (fixed), x2 in [0, inf), x3 in [0, inf)
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {3.0, 2.0, 5.0},
        denseToCSR({{1.0, 1.0, 1.0}}, 3),
        {-kInfinity},
        {10.0},
        {"row1"},
        {2.0, 0.0, 0.0},
        {2.0, kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3"});

    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model, ledger);

    // Check that x1 was removed.
    check(reduced.numVars() == 2, "FixVar: reduced has 2 vars");
    check(reduced.numRows() == 1, "FixVar: reduced has 1 row");

    // Check ledger.
    check(ledger.size() == 1, "FixVar: ledger has 1 entry");
    const auto* rec = std::get_if<FixedVariableRemoved>(&ledger.reductions()[0]);
    check(rec != nullptr, "FixVar: ledger entry is FixedVariableRemoved");
    if (rec) {
        check(rec->original_index == 0, "FixVar: recorded original index == 0");
        checkNear(rec->fixed_value, 2.0, 1e-15, "FixVar: recorded fixed value == 2.0");
        check(rec->var_name == "x1", "FixVar: recorded var name == x1");
        checkNear(rec->obj_coeff, 3.0, 1e-15, "FixVar: recorded obj coeff == 3.0");
    }

    // Check objective offset: original 0 + 3*2 = 6.
    checkNear(reduced.obj_offset, 6.0, 1e-15, "FixVar: obj_offset == 6.0");

    // Check remaining objective coefficients: [2.0, 5.0].
    check(reduced.obj_coeffs.size() == 2, "FixVar: 2 obj coeffs");
    if (reduced.obj_coeffs.size() == 2) {
        checkNear(reduced.obj_coeffs[0], 2.0, 1e-15, "FixVar: obj_coeff[0] == 2.0");
        checkNear(reduced.obj_coeffs[1], 5.0, 1e-15, "FixVar: obj_coeff[1] == 5.0");
    }

    // Check row upper bound: 10 - 1*2 = 8.
    checkNear(reduced.row_upper[0], 8.0, 1e-15, "FixVar: row_upper[0] == 8.0");

    // Check variable names: x2, x3.
    check(reduced.var_names.size() == 2, "FixVar: 2 var names");
    if (reduced.var_names.size() == 2) {
        check(reduced.var_names[0] == "x2", "FixVar: var_names[0] == x2");
        check(reduced.var_names[1] == "x3", "FixVar: var_names[1] == x3");
    }

    // Solve the reduced model.
    RevisedSimplex solver;
    SolveResult result = solver.solve(reduced);
    check(result.status == SolveStatus::kOptimal, "FixVar: reduced model optimal");

    // Postsolve: reconstruct original-space solution.
    std::vector<double> full_x = pramaan::postsolve(result.x, ledger);
    check(full_x.size() == 3, "FixVar: postsolve gives 3 vars");
    if (full_x.size() == 3) {
        checkNear(full_x[0], 2.0, 1e-9, "FixVar: postsolve x1 == 2.0");
    }

    // Verify the objective.
    double obj_val = model.obj_offset;
    for (int j = 0; j < 3; ++j) {
        obj_val += model.obj_coeffs[static_cast<std::size_t>(j)] * full_x[static_cast<std::size_t>(j)];
    }
    checkNear(obj_val, result.objective_value, 1e-9, "FixVar: postsolve objective consistent");

    std::cout << "  ok\n";
}

// =========================================================================
// Test: ledger recording details
// =========================================================================
void testLedgerRecording() {
    std::cout << "testLedgerRecording...\n";

    // Two fixed variables: x1=3, x3=7
    // min x1 + x2 + x3
    // s.t. 2*x1 + 3*x2 + 4*x3 <= 100
    ModelIR model(
        ObjSense::kMinimize,
        10.0,  // existing offset
        {1.0, 1.0, 1.0},
        denseToCSR({{2.0, 3.0, 4.0}}, 3),
        {-kInfinity},
        {100.0},
        {"con1"},
        {3.0, 0.0, 7.0},
        {3.0, kInfinity, 7.0},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3"});

    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model, ledger);

    check(ledger.size() == 2, "Ledger: 2 entries for 2 fixed vars");
    check(reduced.numVars() == 1, "Ledger: reduced has 1 var");

    // obj_offset: 10 + 1*3 + 1*7 = 20
    checkNear(reduced.obj_offset, 20.0, 1e-15, "Ledger: obj_offset == 20.0");

    // row_upper: 100 - 2*3 - 4*7 = 100 - 6 - 28 = 66
    checkNear(reduced.row_upper[0], 66.0, 1e-15, "Ledger: row_upper == 66.0");

    // Remaining variable is x2 with coeff 3.0 in row.
    check(reduced.var_names[0] == "x2", "Ledger: remaining var is x2");

    // Row coefficient for x2 in the reduced model should be 3.0.
    auto rv = reduced.A.row(0);
    double a_val = 0.0;
    for (auto e : rv) a_val = e.value;
    checkNear(a_val, 3.0, 1e-15, "Ledger: A[0,0] == 3.0 (x2 coeff)");

    // Postsolve with a fake reduced solution x2=5.
    std::vector<double> reduced_x = {5.0};
    std::vector<double> full_x = pramaan::postsolve(reduced_x, ledger);
    check(full_x.size() == 3, "Ledger: postsolve gives 3 vars");
    if (full_x.size() == 3) {
        checkNear(full_x[0], 3.0, 1e-15, "Ledger: postsolve x1 == 3.0");
        checkNear(full_x[1], 5.0, 1e-15, "Ledger: postsolve x2 == 5.0");
        checkNear(full_x[2], 7.0, 1e-15, "Ledger: postsolve x3 == 7.0");
    }

    std::cout << "  ok\n";
}

// =========================================================================
// Test: model with no fixed variables (presolve is a no-op)
// =========================================================================
void testNoFixedVariables() {
    std::cout << "testNoFixedVariables...\n";

    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1.0, 2.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity},
        {5.0},
        {"row1"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model, ledger);

    check(ledger.empty(), "NoFix: ledger is empty");
    check(reduced.numVars() == 2, "NoFix: reduced still has 2 vars");
    check(reduced.numRows() == 1, "NoFix: reduced still has 1 row");
    checkNear(reduced.obj_offset, 0.0, 1e-15, "NoFix: obj_offset unchanged");

    // Postsolve with no entries should return x unchanged.
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> full_x = pramaan::postsolve(x, ledger);
    check(full_x.size() == 2, "NoFix: postsolve size == 2");
    if (full_x.size() == 2) {
        checkNear(full_x[0], 1.0, 1e-15, "NoFix: postsolve x[0] == 1.0");
        checkNear(full_x[1], 2.0, 1e-15, "NoFix: postsolve x[1] == 2.0");
    }

    std::cout << "  ok\n";
}

// =========================================================================
// Test: Ruiz scaling and unscaling
// =========================================================================
void testRuizScaling() {
    std::cout << "testRuizScaling...\n";

    // A model with badly scaled coefficients.
    // min 1e6*x1 + 1e-3*x2
    // s.t. 1e6*x1 + 1e-3*x2 <= 1e6
    //      x1, x2 >= 0
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1e6, 1e-3},
        denseToCSR({{1e6, 1e-3}}, 2),
        {-kInfinity},
        {1e6},
        {"row1"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    // Save original model for comparison.
    ModelIR original = model;

    TransformationLedger ledger;
    ScalingFactors factors = pramaan::ruiz_scale(model, ledger, 10);

    // Check that scaling factors are positive.
    check(factors.row_scale.size() == 1, "Ruiz: row_scale size == 1");
    check(factors.col_scale.size() == 2, "Ruiz: col_scale size == 2");
    for (double rs : factors.row_scale) {
        check(rs > 0.0, "Ruiz: row scale factor > 0");
    }
    for (double cs : factors.col_scale) {
        check(cs > 0.0, "Ruiz: col scale factor > 0");
    }

    // The scaled matrix should have better-conditioned coefficients.
    // Check that the infinity-norm ratio improved.
    double max_val = 0.0, min_val = 1e30;
    for (int r = 0; r < model.numRows(); ++r) {
        auto rv = model.A.row(r);
        for (auto e : rv) {
            double av = std::abs(e.value);
            if (av > max_val) max_val = av;
            if (av > 0 && av < min_val) min_val = av;
        }
    }
    double ratio_after = (min_val > 0) ? max_val / min_val : 1e30;
    check(ratio_after < 1e9, "Ruiz: coefficient ratio improved (< 1e9 vs original 1e9)");

    // Solve the scaled model.
    RevisedSimplex solver;
    SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Ruiz: scaled model optimal");

    // Unscale.
    if (result.status == SolveStatus::kOptimal) {
        pramaan::unscale_solution(result.x, result.objective_value, original, factors);

        // x1 = 0 (expensive), x2 can be anything. Min is at x1=0, x2=0.
        checkNear(result.x[0], 0.0, 1e-6, "Ruiz: unscaled x1 == 0");
        checkNear(result.x[1], 0.0, 1e-6, "Ruiz: unscaled x2 == 0");
        checkNear(result.objective_value, 0.0, 1e-6, "Ruiz: unscaled obj == 0");
    }

    std::cout << "  ok\n";
}

// =========================================================================
// Test: ruiz_scale records exactly one CumulativeScaling in the ledger
// =========================================================================
void testCumulativeScalingLedger() {
    std::cout << "testCumulativeScalingLedger...\n";

    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {1e4, 1e-2},
        denseToCSR({{1e4, 1e-2}}, 2),
        {-kInfinity},
        {1e4},
        {"row1"},
        {0.0, 0.0},
        {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    TransformationLedger ledger;
    ScalingFactors factors = pramaan::ruiz_scale(model, ledger, 10);

    // Exactly one ledger entry.
    check(ledger.size() == 1, "CumScale: ledger has exactly 1 entry");

    // The entry is a CumulativeScaling.
    const auto* cs = std::get_if<CumulativeScaling>(&ledger.reductions()[0]);
    check(cs != nullptr, "CumScale: entry is CumulativeScaling");

    if (cs) {
        // Sizes match the model dimensions.
        check(cs->row_scale.size() == 1, "CumScale: row_scale size == 1");
        check(cs->col_scale.size() == 2, "CumScale: col_scale size == 2");

        // Factors match the returned ScalingFactors exactly.
        checkNear(cs->row_scale[0], factors.row_scale[0], 1e-15,
                  "CumScale: row_scale[0] matches returned factors");
        checkNear(cs->col_scale[0], factors.col_scale[0], 1e-15,
                  "CumScale: col_scale[0] matches returned factors");
        checkNear(cs->col_scale[1], factors.col_scale[1], 1e-15,
                  "CumScale: col_scale[1] matches returned factors");

        // Factors are positive and non-trivial (scaling actually happened).
        check(cs->row_scale[0] > 0.0, "CumScale: row_scale[0] > 0");
        check(cs->col_scale[0] > 0.0, "CumScale: col_scale[0] > 0");
        check(cs->col_scale[1] > 0.0, "CumScale: col_scale[1] > 0");
    }

    std::cout << "  ok\n";
}

// =========================================================================
// Test: full pipeline with a small hand-built model
// =========================================================================
void testFullPipeline() {
    std::cout << "testFullPipeline...\n";

    // min  3*x1 + 2*x2 + 5*x3 + x4
    // s.t. x1 + x2 + x3 + x4 <= 10
    //      x1 = 2 (fixed), x2 in [0,10], x3 in [0,10], x4 in [0,10]
    // Optimal without presolve: x1=2, x2=8, x3=0, x4=0, obj = 6+16 = 22
    // With presolve: remove x1, solve reduced:
    //   min 2*x2 + 5*x3 + x4 + 6
    //   s.t. x2 + x3 + x4 <= 8
    //   Optimal: x2=0, x3=0, x4=0, obj = 6
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {3.0, 2.0, 5.0, 1.0},
        denseToCSR({{1.0, 1.0, 1.0, 1.0}}, 4),
        {-kInfinity},
        {10.0},
        {"row1"},
        {2.0, 0.0, 0.0, 0.0},
        {2.0, 10.0, 10.0, 10.0},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3", "x4"});

    ModelIR original = model;

    // Step 1: Presolve.
    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model, ledger);

    check(reduced.numVars() == 3, "Pipeline: 3 vars after presolve");

    // Step 2: Scale.
    ModelIR pre_scale_reduced = reduced;  // save for unscaling
    ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);

    // Step 3: Solve.
    RevisedSimplex solver;
    SolveResult result = solver.solve(reduced);
    check(result.status == SolveStatus::kOptimal, "Pipeline: scaled reduced model optimal");

    if (result.status == SolveStatus::kOptimal) {
        // Step 4: Unscale.
        pramaan::unscale_solution(result.x, result.objective_value, pre_scale_reduced, factors);

        // Step 5: Postsolve.
        std::vector<double> full_x = pramaan::postsolve(result.x, ledger);
        check(full_x.size() == 4, "Pipeline: postsolve gives 4 vars");

        if (full_x.size() == 4) {
            checkNear(full_x[0], 2.0, 1e-9, "Pipeline: x1 == 2.0 (fixed)");
            // x2, x3, x4 should all be 0 at optimal.
            checkNear(full_x[1], 0.0, 1e-9, "Pipeline: x2 == 0.0");
            checkNear(full_x[2], 0.0, 1e-9, "Pipeline: x3 == 0.0");
            checkNear(full_x[3], 0.0, 1e-9, "Pipeline: x4 == 0.0");
        }

        // Recompute objective in original space.
        double obj = original.obj_offset;
        for (int j = 0; j < 4; ++j) {
            obj += original.obj_coeffs[static_cast<std::size_t>(j)] * full_x[static_cast<std::size_t>(j)];
        }
        checkNear(obj, 6.0, 1e-9, "Pipeline: original-space obj == 6.0");

        // Verify feasibility in original model.
        std::vector<double> activity = original.A.multiply(full_x);
        for (int r = 0; r < original.numRows(); ++r) {
            double lo = original.row_lower[static_cast<std::size_t>(r)];
            double hi = original.row_upper[static_cast<std::size_t>(r)];
            double ar = activity[static_cast<std::size_t>(r)];
            check((lo <= -kInfinity || ar >= lo - 1e-9) && (hi >= kInfinity || ar <= hi + 1e-9),
                  "Pipeline: original row " + std::to_string(r) + " feasible");
        }
    }

    std::cout << "  ok\n";
}

// =========================================================================
// Test: regression for issue where scaling indices are skewed by presolve
// =========================================================================
void testPresolveScalingIndices() {
    std::cout << "testPresolveScalingIndices...\n";

    // min  10*x1 + 2*x2 + 3*x3
    // s.t. x1 + x2 + x3 >= 5
    //      x1 = 2 (fixed)
    //      x2 >= 0
    //      x3 >= 0
    // This places a fixed variable (x1) BEFORE non-fixed variables (x2, x3).
    // Original optimal solution: x1=2. The remaining constraint is x2 + x3 >= 3.
    // To minimize 2*x2 + 3*x3, we choose x2=3, x3=0.
    // Original optimal objective = 10*2 + 2*3 + 3*0 = 26.
    ModelIR model(
        ObjSense::kMinimize,
        0.0,
        {10.0, 2.0, 3.0},
        denseToCSR({{1.0, 1.0, 1.0}}, 3),
        {5.0},
        {kInfinity},
        {"row1"},
        {2.0, 0.0, 0.0},
        {2.0, kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3"});

    ModelIR original = model;

    // Solve original directly to get the reference objective.
    RevisedSimplex solver;
    SolveResult result_ref = solver.solve(original);
    check(result_ref.status == SolveStatus::kOptimal, "PresolveIndexBug: original model optimal");
    checkNear(result_ref.objective_value, 26.0, 1e-9, "PresolveIndexBug: reference obj == 26");

    // Pipeline
    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model, ledger);
    ModelIR pre_scale_reduced = reduced;
    ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);

    SolveResult result = solver.solve(reduced);
    check(result.status == SolveStatus::kOptimal, "PresolveIndexBug: reduced scaled model optimal");

    if (result.status == SolveStatus::kOptimal) {
        pramaan::unscale_solution(result.x, result.objective_value, pre_scale_reduced, factors);

        // The unscaled objective value should exactly match the original objective,
        // because the reduced model's obj_offset absorbed the fixed variables' contribution.
        checkNear(result.objective_value, result_ref.objective_value, 1e-9,
                  "PresolveIndexBug: unscaled objective matches reference exactly");

        std::vector<double> full_x = pramaan::postsolve(result.x, ledger);
        check(full_x.size() == 3, "PresolveIndexBug: postsolve gives 3 vars");
        if (full_x.size() == 3) {
            checkNear(full_x[0], 2.0, 1e-9, "PresolveIndexBug: x1 == 2");
            checkNear(full_x[1], 3.0, 1e-9, "PresolveIndexBug: x2 == 3");
            checkNear(full_x[2], 0.0, 1e-9, "PresolveIndexBug: x3 == 0");
        }
    }
    std::cout << "  ok\n";
}

// =========================================================================
// Test: AFIRO end-to-end with presolve + scaling ON vs OFF
// =========================================================================
void testAfiroEndToEnd() {
    std::cout << "testAfiroEndToEnd...\n";

    std::string afiro_path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps",
             "../../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            afiro_path = candidate;
            break;
        }
    }
    if (afiro_path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found\n";
        return;
    }

    // --- Solve WITHOUT presolve/scaling (reference) ---
    ModelIR model_off = pramaan::parse_mps(afiro_path);
    RevisedSimplex solver;
    SolveResult result_off = solver.solve(model_off);
    check(result_off.status == SolveStatus::kOptimal, "AFIRO-OFF: status optimal");

    // --- Solve WITH presolve + scaling ---
    ModelIR model_on = pramaan::parse_mps(afiro_path);
    ModelIR original = model_on;  // keep a copy of original

    // Step 1: Presolve (fixed-variable removal).
    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(model_on, ledger);

    int vars_removed = model_on.numVars() - reduced.numVars();
    std::cout << "  AFIRO presolve: " << vars_removed << " fixed vars removed, "
              << reduced.numVars() << " vars remaining\n";

    // Step 2: Scale.
    ModelIR pre_scale = reduced;  // save unscaled reduced for unscaling
    ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);

    // Step 3: Solve.
    SolveResult result_on = solver.solve(reduced);
    check(result_on.status == SolveStatus::kOptimal, "AFIRO-ON: status optimal");

    if (result_on.status == SolveStatus::kOptimal && result_off.status == SolveStatus::kOptimal) {
        // Step 4: Unscale.
        pramaan::unscale_solution(result_on.x, result_on.objective_value, pre_scale, factors);

        // Step 5: Postsolve.
        std::vector<double> full_x = pramaan::postsolve(result_on.x, ledger);
        check(full_x.size() == static_cast<std::size_t>(original.numVars()),
              "AFIRO-ON: postsolve gives original var count");

        // Recompute objective in original space.
        double obj_on = original.obj_offset;
        for (int j = 0; j < original.numVars(); ++j) {
            obj_on += original.obj_coeffs[static_cast<std::size_t>(j)] * full_x[static_cast<std::size_t>(j)];
        }

        std::cout << "  AFIRO-OFF objective = " << result_off.objective_value << "\n";
        std::cout << "  AFIRO-ON  objective = " << obj_on << "\n";

        // Compare objectives.
        checkNear(obj_on, result_off.objective_value, 1e-4,
                  "AFIRO: ON vs OFF objective match");

        // Compare solution vectors.
        if (full_x.size() == result_off.x.size()) {
            double max_diff = 0.0;
            for (std::size_t j = 0; j < full_x.size(); ++j) {
                double diff = std::abs(full_x[j] - result_off.x[j]);
                if (diff > max_diff) max_diff = diff;
            }
            std::cout << "  AFIRO solution max |diff| = " << max_diff << "\n";
            check(max_diff < 1e-4, "AFIRO: ON vs OFF solution vectors close");
        }

        // Verify original-space feasibility.
        std::vector<double> activity = original.A.multiply(full_x);
        bool feasible = true;
        for (int r = 0; r < original.numRows(); ++r) {
            double lo = original.row_lower[static_cast<std::size_t>(r)];
            double hi = original.row_upper[static_cast<std::size_t>(r)];
            double ar = activity[static_cast<std::size_t>(r)];
            if ((lo > -kInfinity && ar < lo - 1e-6) || (hi < kInfinity && ar > hi + 1e-6)) {
                feasible = false;
                std::cerr << "  [FAIL] AFIRO-ON: row " << r << " activity " << ar
                           << " outside [" << lo << ", " << hi << "]\n";
            }
        }
        check(feasible, "AFIRO-ON: all original rows feasible");

        // Verify variable bounds.
        bool var_feasible = true;
        for (int j = 0; j < original.numVars(); ++j) {
            double lo = original.var_lower[static_cast<std::size_t>(j)];
            double hi = original.var_upper[static_cast<std::size_t>(j)];
            double xj = full_x[static_cast<std::size_t>(j)];
            if ((lo > -kInfinity && xj < lo - 1e-6) || (hi < kInfinity && xj > hi + 1e-6)) {
                var_feasible = false;
            }
        }
        check(var_feasible, "AFIRO-ON: all original variable bounds satisfied");

        // All values finite.
        bool all_finite = true;
        for (std::size_t j = 0; j < full_x.size(); ++j) {
            if (!std::isfinite(full_x[j])) all_finite = false;
        }
        check(all_finite, "AFIRO-ON: all solution values finite");
    }

    std::cout << "  ok\n";
}

}  // namespace

int main() {
    testFixedVariableElimination();
    testLedgerRecording();
    testNoFixedVariables();
    testRuizScaling();
    testCumulativeScalingLedger();
    testFullPipeline();
    testPresolveScalingIndices();
    testAfiroEndToEnd();

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed
               << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
