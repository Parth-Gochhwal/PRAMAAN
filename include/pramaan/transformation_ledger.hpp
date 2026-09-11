// transformation_ledger.hpp
// PILLAR 4 (Trust) -- supports Zone 7 Certificate Generator
//
// Purpose: append-only log of every reversible presolve step. This is what
// makes postsolve (mapping the presolved solution back to the ORIGINAL
// model) and independent verification possible at all.
//
// Each reduction is recorded as a tagged variant (Reduction). Postsolve
// replays the ledger vector in REVERSE order, reconstructing the original-
// space solution vector from the reduced solution.
#pragma once
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pramaan {

// --- Reduction records ---------------------------------------------------

// A variable was fixed (lb == ub) and removed from the reduced model.
// Postsolve inserts it back at position `original_index` with value
// `fixed_value`, shifting later variable indices accordingly.
struct FixedVariableRemoved {
    int original_index;          // column index in the model BEFORE this reduction
    double fixed_value;          // lb == ub value
    std::string var_name;        // for diagnostics
    double obj_coeff;            // objective coefficient of the removed variable
    // Per-row coefficients of the removed variable (sparse: only nonzeros).
    // These are needed to verify the RHS adjustment during certificate
    // generation, but postsolve itself only needs original_index + fixed_value.
    struct RowCoeff {
        int row;
        double value;
    };
    std::vector<RowCoeff> row_coeffs;
};

// Row scaling: row r was multiplied by `factor`.
// Postsolve divides the row activity by `factor`.
struct RowScaled {
    int row;
    double factor;
};

// Column scaling: column j was multiplied by `factor`.
// Postsolve divides x[j] by `factor` (and multiplies the reduced cost).
struct ColumnScaled {
    int col;
    double factor;
};

// Cumulative scaling: rows and columns were multiplied by the respective factors.
// Postsolve divides x[j] by col_scale[j] (wait, x_orig = x_scaled * col_scale, so we multiply by col_scale).
// Actually, earlier we fixed unscale_solution to: x_orig = x_scaled * col_scale.
struct CumulativeScaling {
    std::vector<double> row_scale;
    std::vector<double> col_scale;
};

// Tagged union of all reduction types.
using Reduction = std::variant<FixedVariableRemoved, RowScaled, ColumnScaled, CumulativeScaling>;

// --- TransformationLedger ------------------------------------------------

class TransformationLedger {
public:
    // Append a reduction record.
    void record(Reduction r) { reductions_.push_back(std::move(r)); }

    // Number of recorded reductions.
    std::size_t size() const noexcept { return reductions_.size(); }

    bool empty() const noexcept { return reductions_.empty(); }

    // Access (const) for iteration.
    const std::vector<Reduction>& reductions() const noexcept { return reductions_; }

    // Clear all records (for reuse).
    void clear() { reductions_.clear(); }

private:
    std::vector<Reduction> reductions_;
};

}  // namespace pramaan
