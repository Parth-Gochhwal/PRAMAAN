// scaling.cpp
// PILLAR 1 (Structure) -- part of Zone 2 "Reversible Presolve"
//
// Purpose: Ruiz equilibration -- iteratively rescale rows and columns of A
// so the infinity-norm of each row and column converges toward 1. This
// reduces the condition number of the constraint matrix and improves
// numerical stability of the simplex solver.
//
// The implementation is independent of presolve.cpp's fixed-variable logic.
// Scaling factors are returned for explicit reversal via unscale_solution().
#include "pramaan/presolve.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace pramaan {

// ---------------------------------------------------------------------------
// ruiz_scale
// ---------------------------------------------------------------------------
// Ruiz equilibration: for each iteration, compute the infinity-norm of each
// row and column, then scale rows/columns by 1/sqrt(norm). After several
// iterations, row and column norms converge close to 1.
//
// Modifies the model in place (A, obj_coeffs, row_lower, row_upper,
// var_lower, var_upper) and returns the cumulative scaling factors.

ScalingFactors ruiz_scale(ModelIR& model, TransformationLedger& ledger, int max_iterations) {
    const int m = model.numRows();
    const int n = model.numVars();

    ScalingFactors factors;
    factors.row_scale.assign(static_cast<std::size_t>(m), 1.0);
    factors.col_scale.assign(static_cast<std::size_t>(n), 1.0);

    if (m == 0 || n == 0) return factors;

    // We need mutable access to the CSR internals. Since CSRMatrix exposes
    // only const accessors, we rebuild the matrix after scaling. To avoid
    // repeated rebuilds, we work with raw arrays extracted once, scale them,
    // and rebuild at the end.

    // Extract CSR arrays (copies -- we'll modify and put them back).
    std::vector<CSRMatrix::Index> row_ptr = model.A.rowPtr();
    std::vector<CSRMatrix::Index> col_idx = model.A.colIdx();
    std::vector<double> values = model.A.values();
    const CSRMatrix::Index nnz = static_cast<CSRMatrix::Index>(values.size());

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute infinity-norm of each row.
        std::vector<double> row_norm(static_cast<std::size_t>(m), 0.0);
        for (int r = 0; r < m; ++r) {
            for (CSRMatrix::Index k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                double av = std::abs(values[static_cast<std::size_t>(k)]);
                if (av > row_norm[static_cast<std::size_t>(r)]) {
                    row_norm[static_cast<std::size_t>(r)] = av;
                }
            }
        }

        // Compute infinity-norm of each column.
        std::vector<double> col_norm(static_cast<std::size_t>(n), 0.0);
        for (std::size_t k = 0; k < static_cast<std::size_t>(nnz); ++k) {
            int j = col_idx[k];
            double av = std::abs(values[k]);
            if (av > col_norm[static_cast<std::size_t>(j)]) {
                col_norm[static_cast<std::size_t>(j)] = av;
            }
        }

        // Also consider objective coefficients as part of column norms.
        for (int j = 0; j < n; ++j) {
            double av = std::abs(model.obj_coeffs[static_cast<std::size_t>(j)]);
            if (av > col_norm[static_cast<std::size_t>(j)]) {
                col_norm[static_cast<std::size_t>(j)] = av;
            }
        }

        // Compute row scale factors: 1 / sqrt(row_norm).
        std::vector<double> r_scale(static_cast<std::size_t>(m), 1.0);
        for (int r = 0; r < m; ++r) {
            if (row_norm[static_cast<std::size_t>(r)] > 0.0) {
                r_scale[static_cast<std::size_t>(r)] = 1.0 / std::sqrt(row_norm[static_cast<std::size_t>(r)]);
            }
        }

        // Compute column scale factors: 1 / sqrt(col_norm).
        std::vector<double> c_scale(static_cast<std::size_t>(n), 1.0);
        for (int j = 0; j < n; ++j) {
            if (col_norm[static_cast<std::size_t>(j)] > 0.0) {
                c_scale[static_cast<std::size_t>(j)] = 1.0 / std::sqrt(col_norm[static_cast<std::size_t>(j)]);
            }
        }

        // Apply row and column scaling to A values.
        for (int r = 0; r < m; ++r) {
            double rs = r_scale[static_cast<std::size_t>(r)];
            for (CSRMatrix::Index k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                int j = col_idx[static_cast<std::size_t>(k)];
                values[static_cast<std::size_t>(k)] *= rs * c_scale[static_cast<std::size_t>(j)];
            }
        }

        // Apply column scaling to objective coefficients.
        for (int j = 0; j < n; ++j) {
            model.obj_coeffs[static_cast<std::size_t>(j)] *= c_scale[static_cast<std::size_t>(j)];
        }

        // Apply row scaling to row bounds.
        for (int r = 0; r < m; ++r) {
            double rs = r_scale[static_cast<std::size_t>(r)];
            double lo = model.row_lower[static_cast<std::size_t>(r)];
            double hi = model.row_upper[static_cast<std::size_t>(r)];
            if (lo > -kInfinity) model.row_lower[static_cast<std::size_t>(r)] = lo * rs;
            if (hi < kInfinity) model.row_upper[static_cast<std::size_t>(r)] = hi * rs;
        }

        // Apply column scaling to variable bounds.
        // x_scaled[j] = x_orig[j] / col_scale[j], so:
        //   lo[j] <= x_orig[j]  becomes  lo[j] / col_scale[j] <= x_scaled[j]
        //   x_orig[j] <= hi[j]  becomes  x_scaled[j] <= hi[j] / col_scale[j]
        for (int j = 0; j < n; ++j) {
            double cs = c_scale[static_cast<std::size_t>(j)];
            double lo = model.var_lower[static_cast<std::size_t>(j)];
            double hi = model.var_upper[static_cast<std::size_t>(j)];
            if (lo > -kInfinity) model.var_lower[static_cast<std::size_t>(j)] = lo / cs;
            if (hi < kInfinity) model.var_upper[static_cast<std::size_t>(j)] = hi / cs;
        }

        // Accumulate into cumulative factors.
        for (int r = 0; r < m; ++r) {
            factors.row_scale[static_cast<std::size_t>(r)] *= r_scale[static_cast<std::size_t>(r)];
        }
        for (int j = 0; j < n; ++j) {
            factors.col_scale[static_cast<std::size_t>(j)] *= c_scale[static_cast<std::size_t>(j)];
        }
    }

    // Rebuild the CSRMatrix with scaled values.
    model.A = CSRMatrix(std::move(row_ptr), std::move(col_idx),
                        std::move(values), n);

    CumulativeScaling rec;
    rec.row_scale = factors.row_scale;
    rec.col_scale = factors.col_scale;
    ledger.record(std::move(rec));

    return factors;
}

// ---------------------------------------------------------------------------
// unscale_solution
// ---------------------------------------------------------------------------
// Reverses Ruiz scaling on the solution vector.
//
// `pre_scaled_model` is the reduced model BEFORE Ruiz scaling was applied
// (i.e. after fixed-variable presolve but before scaling). Its obj_coeffs
// and obj_offset are used to recompute the correct objective value.
//
// Scaling convention used in ruiz_scale:
//   A_scaled[r,j] = A[r,j] * row_scale[r] * col_scale[j]
//   c_scaled[j]   = c[j] * col_scale[j]
//   b_scaled[r]   = b[r] * row_scale[r]
//   var_bounds_scaled[j] = var_bounds[j] / col_scale[j]
//
// The scaled variable satisfies: x_scaled[j] = x_orig[j] / col_scale[j]
// (from A_scaled * x_scaled = b_scaled expanding to R*A*C * C^-1 * x = R*b).
// Therefore: x_orig[j] = x_scaled[j] * col_scale[j].

void unscale_solution(std::vector<double>& x, double& objective_value,
                      const ModelIR& pre_scaled_model,
                      const ScalingFactors& factors) {
    const int n = static_cast<int>(x.size());

    // x_orig[j] = x_scaled[j] * col_scale[j]
    for (int j = 0; j < n; ++j) {
        x[static_cast<std::size_t>(j)] *= factors.col_scale[static_cast<std::size_t>(j)];
    }

    // Recompute objective from pre-scaled (unscaled) coefficients and unscaled x.
    double obj = pre_scaled_model.obj_offset;
    for (int j = 0; j < n; ++j) {
        obj += pre_scaled_model.obj_coeffs[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    }
    objective_value = obj;
}

}  // namespace pramaan
