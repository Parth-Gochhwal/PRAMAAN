// presolve.cpp -- see include/pramaan/presolve.hpp
//
// Stage 6: Fixed-variable removal + postsolve via ledger replay.
#include "pramaan/presolve.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace pramaan {

// ---------------------------------------------------------------------------
// presolve_fixed_variables
// ---------------------------------------------------------------------------
// Scans for variables with lb == ub, records them in the ledger, and builds
// a reduced ModelIR without those columns. The fixed value is substituted
// into the objective offset and into the row bounds (RHS adjustment).

ModelIR presolve_fixed_variables(const ModelIR& model, TransformationLedger& ledger) {
    const int n = model.numVars();
    const int m = model.numRows();

    // Identify fixed variables.
    std::vector<bool> is_fixed(static_cast<std::size_t>(n), false);
    int num_fixed = 0;
    for (int j = 0; j < n; ++j) {
        if (model.var_lower[static_cast<std::size_t>(j)] == model.var_upper[static_cast<std::size_t>(j)]) {
            is_fixed[static_cast<std::size_t>(j)] = true;
            ++num_fixed;
        }
    }

    if (num_fixed == 0) {
        // Nothing to do; return a copy of the original model.
        return model;
    }

    // Build mapping: old column index -> new column index (-1 if fixed).
    std::vector<int> old_to_new(static_cast<std::size_t>(n), -1);
    int new_n = 0;
    for (int j = 0; j < n; ++j) {
        if (!is_fixed[static_cast<std::size_t>(j)]) {
            old_to_new[static_cast<std::size_t>(j)] = new_n++;
        }
    }

    // Compute RHS adjustments: for each row r, subtract sum of A[r,j]*fixed_val
    // for every fixed variable j.
    std::vector<double> rhs_adjust(static_cast<std::size_t>(m), 0.0);
    double obj_adjust = 0.0;

    // Record each fixed variable in the ledger (in original column order).
    for (int j = 0; j < n; ++j) {
        if (!is_fixed[static_cast<std::size_t>(j)]) continue;

        double val = model.var_lower[static_cast<std::size_t>(j)];

        FixedVariableRemoved rec;
        rec.original_index = j;
        rec.fixed_value = val;
        rec.var_name = model.var_names[static_cast<std::size_t>(j)];
        rec.obj_coeff = model.obj_coeffs[static_cast<std::size_t>(j)];

        // Accumulate objective offset contribution.
        obj_adjust += model.obj_coeffs[static_cast<std::size_t>(j)] * val;

        // Record row coefficients and accumulate RHS adjustment.
        // Iterate over rows (CSR provides efficient row iteration).
        for (int r = 0; r < m; ++r) {
            auto rv = model.A.row(r);
            for (auto e : rv) {
                if (e.index == j) {
                    rec.row_coeffs.push_back({r, e.value});
                    rhs_adjust[static_cast<std::size_t>(r)] += e.value * val;
                    break;
                }
            }
        }

        ledger.record(std::move(rec));
    }

    // Build the reduced model.
    ModelIR reduced;
    reduced.obj_sense = model.obj_sense;
    reduced.obj_offset = model.obj_offset + obj_adjust;

    // Objective coefficients: only non-fixed variables.
    reduced.obj_coeffs.reserve(static_cast<std::size_t>(new_n));
    for (int j = 0; j < n; ++j) {
        if (!is_fixed[static_cast<std::size_t>(j)]) {
            reduced.obj_coeffs.push_back(model.obj_coeffs[static_cast<std::size_t>(j)]);
        }
    }

    // Variable bounds, types, names: only non-fixed variables.
    reduced.var_lower.reserve(static_cast<std::size_t>(new_n));
    reduced.var_upper.reserve(static_cast<std::size_t>(new_n));
    reduced.var_types.reserve(static_cast<std::size_t>(new_n));
    reduced.var_names.reserve(static_cast<std::size_t>(new_n));
    for (int j = 0; j < n; ++j) {
        if (!is_fixed[static_cast<std::size_t>(j)]) {
            reduced.var_lower.push_back(model.var_lower[static_cast<std::size_t>(j)]);
            reduced.var_upper.push_back(model.var_upper[static_cast<std::size_t>(j)]);
            reduced.var_types.push_back(model.var_types[static_cast<std::size_t>(j)]);
            reduced.var_names.push_back(model.var_names[static_cast<std::size_t>(j)]);
        }
    }

    // Row bounds: adjusted by the fixed-variable contributions.
    reduced.row_lower.resize(static_cast<std::size_t>(m));
    reduced.row_upper.resize(static_cast<std::size_t>(m));
    reduced.row_names = model.row_names;
    for (int r = 0; r < m; ++r) {
        double adj = rhs_adjust[static_cast<std::size_t>(r)];
        double lo = model.row_lower[static_cast<std::size_t>(r)];
        double hi = model.row_upper[static_cast<std::size_t>(r)];
        // row_lower <= Ax <= row_upper  becomes
        // row_lower - adj <= A_reduced * x_reduced <= row_upper - adj
        reduced.row_lower[static_cast<std::size_t>(r)] = (lo <= -kInfinity) ? lo : lo - adj;
        reduced.row_upper[static_cast<std::size_t>(r)] = (hi >= kInfinity) ? hi : hi - adj;
    }

    // Constraint matrix: drop fixed-variable columns, remap column indices.
    const auto& old_row_ptr = model.A.rowPtr();
    const auto& old_col_idx = model.A.colIdx();
    const auto& old_values = model.A.values();

    std::vector<CSRMatrix::Index> new_row_ptr;
    std::vector<CSRMatrix::Index> new_col_idx;
    std::vector<double> new_values;
    new_row_ptr.reserve(static_cast<std::size_t>(m + 1));
    new_row_ptr.push_back(0);

    for (int r = 0; r < m; ++r) {
        for (CSRMatrix::Index k = old_row_ptr[r]; k < old_row_ptr[r + 1]; ++k) {
            int j = old_col_idx[static_cast<std::size_t>(k)];
            if (!is_fixed[static_cast<std::size_t>(j)]) {
                new_col_idx.push_back(old_to_new[static_cast<std::size_t>(j)]);
                new_values.push_back(old_values[static_cast<std::size_t>(k)]);
            }
        }
        new_row_ptr.push_back(static_cast<CSRMatrix::Index>(new_col_idx.size()));
    }

    reduced.A = CSRMatrix(std::move(new_row_ptr), std::move(new_col_idx),
                          std::move(new_values), new_n);

    return reduced;
}

// ---------------------------------------------------------------------------
// postsolve
// ---------------------------------------------------------------------------
// Replays the ledger in reverse, inserting fixed variables back into the
// solution vector at their original positions.

std::vector<double> postsolve(const std::vector<double>& reduced_x,
                              const TransformationLedger& ledger) {
    // Start with the reduced solution.
    std::vector<double> x = reduced_x;

    // Replay in reverse order.
    const auto& recs = ledger.reductions();
    for (auto it = recs.rbegin(); it != recs.rend(); ++it) {
        if (const auto* fix = std::get_if<FixedVariableRemoved>(&*it)) {
            // Insert the fixed variable back at its original position.
            auto pos = static_cast<std::size_t>(fix->original_index);
            if (pos > x.size()) pos = x.size();
            x.insert(x.begin() + static_cast<std::ptrdiff_t>(pos), fix->fixed_value);
        }
        // RowScaled and ColumnScaled are not used by the ledger-based
        // postsolve path (scaling is reversed separately). But if they were
        // recorded, they would be handled here.
    }

    return x;
}

}  // namespace pramaan
