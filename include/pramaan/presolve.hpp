// presolve.hpp
// PILLAR 1 (Structure) -- Zone 2 "Reversible Presolve"
//
// Purpose: bound tightening, singleton-variable elimination, Ruiz scaling.
// CRITICAL: every reduction applied here must be logged to the
// TransformationLedger (see transformation_ledger.hpp) so the solution can
// be mapped back to original-variable space and independently verified.
//
// Stage 6 implements exactly one reduction: fixed-variable removal
// (lb == ub). Ruiz scaling is provided as a separate, independent step.
#pragma once
#include "pramaan/ir.hpp"
#include "pramaan/transformation_ledger.hpp"

namespace pramaan {

// --- Fixed-variable presolve ---------------------------------------------

// Removes all fixed variables (lb == ub) from the model, substituting their
// fixed values into the objective offset and RHS vectors. Records each
// removal in the ledger for exact postsolve reversal.
//
// Returns a new, reduced ModelIR with fewer columns. The original model is
// not modified.
ModelIR presolve_fixed_variables(const ModelIR& model, TransformationLedger& ledger);

// Reconstructs the original-space solution vector by replaying the ledger
// in reverse order. `reduced_x` is the solution of the reduced model;
// returns a vector sized to the original model's variable count.
std::vector<double> postsolve(const std::vector<double>& reduced_x,
                              const TransformationLedger& ledger);

// --- Ruiz scaling --------------------------------------------------------

// Holds the row and column scaling factors computed by Ruiz equilibration.
struct ScalingFactors {
    std::vector<double> row_scale;   // size = num_rows; row r is multiplied by row_scale[r]
    std::vector<double> col_scale;   // size = num_cols; col j is multiplied by col_scale[j]
};

// Applies Ruiz equilibration to the model, returning the scaled model and
// the scaling factors needed to undo the transformation. Also records
// the scaling factors in the TransformationLedger for auditability.
//
// max_iterations: number of Ruiz equilibration passes (typically 10-20).
ScalingFactors ruiz_scale(ModelIR& model, TransformationLedger& ledger, int max_iterations = 10);

// Reverses Ruiz scaling on a solution vector obtained from solving the
// scaled model. `pre_scaled_model` is the reduced model BEFORE Ruiz scaling
// was applied (i.e. after presolve but before scaling). Its obj_coeffs and
// obj_offset are used to recompute the correct objective value.
void unscale_solution(std::vector<double>& x, double& objective_value,
                      const ModelIR& pre_scaled_model,
                      const ScalingFactors& factors);

}  // namespace pramaan
