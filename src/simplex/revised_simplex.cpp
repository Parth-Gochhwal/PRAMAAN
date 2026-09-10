// revised_simplex.cpp
//
// Implements a sparse revised primal simplex solver.
// Replaces the dense tableau implementation with a sparse product-form
// inverse (PFI) based revised simplex, as described in the Huangfu & Hall
// papers (e.g. "Parallelizing the dual revised simplex method").
//
// 1. Converts ModelIR into a sparse standard form (Ax = b, x >= 0) using CSC.
// 2. Maintains an explicit basis and a sequence of Eta matrices for B^-1.
// 3. Performs FTRAN and BTRAN entirely using sparse data structures.
// 4. Periodically reinverts the basis by rebuilding the Product Form of the
//    Inverse (PFI) representation using transformed basis columns.
// 5. Extracts the solution in terms of the original variables without
//    requiring a dense m x n allocation at any point.
//
// Note: This is a Stage-5 sequential PFI foundation. The Eta vectors may
// densify over many iterations. This is not yet a production implementation
// using Sparse-LU (e.g. Suhl-Suhl) or Forrest-Tomlin basis updates.
#include "pramaan/simplex.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pramaan {

namespace {

using Index = CSRMatrix::Index;

struct ColumnMap {
    Index original_var;
    double sign;
};

enum class RowKind { kLessEqual, kGreaterEqual, kEqual };

struct CSCMatrix {
    Index num_rows = 0;
    Index num_cols = 0;
    std::vector<Index> col_ptr;
    std::vector<Index> row_idx;
    std::vector<double> values;
};

struct SparseStandardForm {
    CSCMatrix A;
    std::vector<double> b;
    std::vector<double> cost;
    std::vector<ColumnMap> col_map;
    std::vector<double> var_shift;
    std::vector<bool> is_artificial;
    std::vector<Index> initial_basis;
};

struct Triplet { Index row; Index col; double val; };

CSCMatrix buildCSC(Index num_rows, Index num_cols, const std::vector<Triplet>& triplets) {
    CSCMatrix csc;
    csc.num_rows = num_rows;
    csc.num_cols = num_cols;
    csc.col_ptr.assign(static_cast<std::size_t>(num_cols + 1), 0);

    for (const auto& t : triplets) {
        csc.col_ptr[static_cast<std::size_t>(t.col + 1)]++;
    }
    for (Index j = 0; j < num_cols; ++j) {
        csc.col_ptr[static_cast<std::size_t>(j + 1)] += csc.col_ptr[static_cast<std::size_t>(j)];
    }
    csc.row_idx.resize(triplets.size());
    csc.values.resize(triplets.size());
    std::vector<Index> cursor = csc.col_ptr;
    for (const auto& t : triplets) {
        Index pos = cursor[static_cast<std::size_t>(t.col)]++;
        csc.row_idx[static_cast<std::size_t>(pos)] = t.row;
        csc.values[static_cast<std::size_t>(pos)] = t.val;
    }
    return csc;
}

SparseStandardForm buildSparseStandardForm(const ModelIR& model, double tol) {
    const Index n = model.numVars();
    const Index m = model.numRows();

    SparseStandardForm sf;
    sf.var_shift.assign(static_cast<std::size_t>(n), 0.0);

    std::vector<Index> single_col(static_cast<std::size_t>(n), -1);
    std::vector<Index> plus_col(static_cast<std::size_t>(n), -1);
    std::vector<Index> minus_col(static_cast<std::size_t>(n), -1);
    std::vector<char> is_free(static_cast<std::size_t>(n), 0);

    for (Index j = 0; j < n; ++j) {
        const double lo = model.var_lower[static_cast<std::size_t>(j)];
        const double ic = (model.obj_sense == ObjSense::kMaximize)
                               ? -model.obj_coeffs[static_cast<std::size_t>(j)]
                               : model.obj_coeffs[static_cast<std::size_t>(j)];
        if (lo <= -kInfinity) {
            is_free[static_cast<std::size_t>(j)] = 1;
            plus_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.col_map.size());
            sf.col_map.push_back(ColumnMap{j, 1.0});
            sf.cost.push_back(ic);
            minus_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.col_map.size());
            sf.col_map.push_back(ColumnMap{j, -1.0});
            sf.cost.push_back(-ic);
        } else {
            single_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.col_map.size());
            sf.col_map.push_back(ColumnMap{j, 1.0});
            sf.cost.push_back(ic);
            sf.var_shift[static_cast<std::size_t>(j)] = lo;
        }
    }

    const Index num_structural = static_cast<Index>(sf.col_map.size());

    std::vector<Triplet> triplets;
    std::vector<double> rhs_vals;
    std::vector<RowKind> row_kinds;

    auto pushRow = [&](std::vector<std::pair<Index, double>> row_nz, double rhs, RowKind kind) {
        if (rhs < -tol) {
            for (auto& nz : row_nz) nz.second = -nz.second;
            rhs = -rhs;
            if (kind == RowKind::kLessEqual) kind = RowKind::kGreaterEqual;
            else if (kind == RowKind::kGreaterEqual) kind = RowKind::kLessEqual;
        }
        if (rhs < 0.0) rhs = 0.0;
        Index r = static_cast<Index>(rhs_vals.size());
        for (const auto& nz : row_nz) {
            if (std::abs(nz.second) > 1e-14) {
                triplets.push_back(Triplet{r, nz.first, nz.second});
            }
        }
        rhs_vals.push_back(rhs);
        row_kinds.push_back(kind);
    };

    auto rowFromModelRow = [&](Index r) -> std::pair<std::vector<std::pair<Index, double>>, double> {
        std::vector<std::pair<Index, double>> row_nz;
        double rhs_adjust = 0.0;
        for (const auto& e : model.A.row(r)) {
            const Index j = e.index;
            const double a = e.value;
            if (is_free[static_cast<std::size_t>(j)]) {
                row_nz.push_back({plus_col[static_cast<std::size_t>(j)], a});
                row_nz.push_back({minus_col[static_cast<std::size_t>(j)], -a});
            } else {
                row_nz.push_back({single_col[static_cast<std::size_t>(j)], a});
                rhs_adjust += a * sf.var_shift[static_cast<std::size_t>(j)];
            }
        }
        return {row_nz, rhs_adjust};
    };

    for (Index r = 0; r < m; ++r) {
        if (model.isFreeRow(r)) continue;
        const double lo = model.row_lower[static_cast<std::size_t>(r)];
        const double up = model.row_upper[static_cast<std::size_t>(r)];
        auto row_pair = rowFromModelRow(r);
        const auto& row_nz = row_pair.first;
        const double rhs_adjust = row_pair.second;

        if (model.isEqualityRow(r)) {
            pushRow(row_nz, lo - rhs_adjust, RowKind::kEqual);
        } else if (lo <= -kInfinity) {
            pushRow(row_nz, up - rhs_adjust, RowKind::kLessEqual);
        } else if (up >= kInfinity) {
            pushRow(row_nz, lo - rhs_adjust, RowKind::kGreaterEqual);
        } else {
            pushRow(row_nz, up - rhs_adjust, RowKind::kLessEqual);
            pushRow(row_nz, lo - rhs_adjust, RowKind::kGreaterEqual);
        }
    }

    for (Index j = 0; j < n; ++j) {
        const double up = model.var_upper[static_cast<std::size_t>(j)];
        if (up >= kInfinity) continue;
        std::vector<std::pair<Index, double>> row_nz;
        double rhs;
        if (is_free[static_cast<std::size_t>(j)]) {
            row_nz.push_back({plus_col[static_cast<std::size_t>(j)], 1.0});
            row_nz.push_back({minus_col[static_cast<std::size_t>(j)], -1.0});
            rhs = up;
        } else {
            row_nz.push_back({single_col[static_cast<std::size_t>(j)], 1.0});
            rhs = up - sf.var_shift[static_cast<std::size_t>(j)];
        }
        pushRow(row_nz, rhs, RowKind::kLessEqual);
    }

    Index num_rows_std = static_cast<Index>(rhs_vals.size());
    sf.b = rhs_vals;
    sf.initial_basis.assign(static_cast<std::size_t>(num_rows_std), -1);

    Index next_col = num_structural;
    for (Index i = 0; i < num_rows_std; ++i) {
        switch (row_kinds[static_cast<std::size_t>(i)]) {
            case RowKind::kLessEqual:
                triplets.push_back(Triplet{i, next_col, 1.0});
                sf.initial_basis[static_cast<std::size_t>(i)] = next_col;
                next_col++;
                break;
            case RowKind::kGreaterEqual:
                triplets.push_back(Triplet{i, next_col, -1.0});
                next_col++;
                triplets.push_back(Triplet{i, next_col, 1.0});
                sf.initial_basis[static_cast<std::size_t>(i)] = next_col;
                next_col++;
                break;
            case RowKind::kEqual:
                triplets.push_back(Triplet{i, next_col, 1.0});
                sf.initial_basis[static_cast<std::size_t>(i)] = next_col;
                next_col++;
                break;
        }
    }

    Index num_cols_std = next_col;
    sf.cost.resize(static_cast<std::size_t>(num_cols_std), 0.0);
    sf.is_artificial.assign(static_cast<std::size_t>(num_cols_std), false);
    for (Index i = 0; i < num_rows_std; ++i) {
        if (row_kinds[static_cast<std::size_t>(i)] == RowKind::kEqual || row_kinds[static_cast<std::size_t>(i)] == RowKind::kGreaterEqual) {
            sf.is_artificial[static_cast<std::size_t>(sf.initial_basis[static_cast<std::size_t>(i)])] = true;
        }
    }

    sf.A = buildCSC(num_rows_std, num_cols_std, triplets);
    return sf;
}

struct EtaMatrix {
    Index p;
    std::vector<Index> indices;
    std::vector<double> values;
};

void ftran(const std::vector<EtaMatrix>& etas, std::vector<double>& x) {
    for (const auto& eta : etas) {
        double x_p = x[static_cast<std::size_t>(eta.p)];
        if (x_p != 0.0) {
            x[static_cast<std::size_t>(eta.p)] = 0.0;
            for (size_t k = 0; k < eta.indices.size(); ++k) {
                x[static_cast<std::size_t>(eta.indices[k])] += x_p * eta.values[k];
            }
        }
    }
}

void btran(const std::vector<EtaMatrix>& etas, std::vector<double>& y) {
    for (auto it = etas.rbegin(); it != etas.rend(); ++it) {
        const auto& eta = *it;
        double dot = 0.0;
        for (size_t k = 0; k < eta.indices.size(); ++k) {
            dot += y[static_cast<std::size_t>(eta.indices[k])] * eta.values[k];
        }
        y[static_cast<std::size_t>(eta.p)] = dot;
    }
}

// Rebuilds the Product Form of the Inverse (PFI) representation from scratch.
// It applies the currently accumulated etas to each basis column, and uses
// the transformed basis columns to create a fresh, minimal set of Eta matrices.
void reinvert(std::vector<EtaMatrix>& etas, std::vector<Index>& basis, const CSCMatrix& A) {
    etas.clear();
    Index m = A.num_rows;
    std::vector<bool> row_used(static_cast<std::size_t>(m), false);
    std::vector<Index> new_basis(static_cast<std::size_t>(m), -1);

    for (Index k = 0; k < m; ++k) {
        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        Index bcol = basis[static_cast<std::size_t>(k)];
        for(Index idx = A.col_ptr[static_cast<std::size_t>(bcol)]; idx < A.col_ptr[static_cast<std::size_t>(bcol+1)]; ++idx) {
            u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] = A.values[static_cast<std::size_t>(idx)];
        }
        ftran(etas, u);

        Index best_p = -1;
        double max_val = 0.0;
        for (Index i = 0; i < m; ++i) {
            if (!row_used[static_cast<std::size_t>(i)]) {
                double abs_val = std::abs(u[static_cast<std::size_t>(i)]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    best_p = i;
                }
            }
        }
        if (best_p == -1 || max_val < 1e-12) {
            throw std::runtime_error("Singular basis in reinvert()");
        }
        row_used[static_cast<std::size_t>(best_p)] = true;
        new_basis[static_cast<std::size_t>(best_p)] = basis[static_cast<std::size_t>(k)];

        EtaMatrix eta;
        eta.p = best_p;
        double inv_u_p = 1.0 / u[static_cast<std::size_t>(best_p)];
        for (Index i = 0; i < m; ++i) {
            if (std::abs(u[static_cast<std::size_t>(i)]) > 1e-14 || i == best_p) {
                eta.indices.push_back(i);
                eta.values.push_back(i == best_p ? inv_u_p : -u[static_cast<std::size_t>(i)] * inv_u_p);
            }
        }
        etas.push_back(std::move(eta));
    }
    basis = std::move(new_basis);
}

void driveOutArtificials(const CSCMatrix& A, std::vector<Index>& basis,
                         std::vector<double>& x_B, std::vector<EtaMatrix>& etas,
                         const std::vector<bool>& is_artificial, double tol) {
    Index m = A.num_rows;
    Index n = A.num_cols;

    std::vector<bool> is_basic(static_cast<std::size_t>(n), false);
    for (Index i = 0; i < m; ++i) {
        is_basic[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] = true;
    }

    for (Index i = 0; i < m; ++i) {
        Index bcol = basis[static_cast<std::size_t>(i)];
        if (!is_artificial[static_cast<std::size_t>(bcol)]) continue;
        
        std::vector<double> y(static_cast<std::size_t>(m), 0.0);
        y[static_cast<std::size_t>(i)] = 1.0;
        btran(etas, y);
        
        Index entering = -1;
        for (Index j = 0; j < n; ++j) {
            if (is_artificial[static_cast<std::size_t>(j)] || is_basic[static_cast<std::size_t>(j)]) continue;
            double a_ij = 0.0;
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)]; idx < A.col_ptr[static_cast<std::size_t>(j+1)]; ++idx) {
                a_ij += y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] * A.values[static_cast<std::size_t>(idx)];
            }
            if (std::abs(a_ij) > tol && std::isfinite(a_ij)) {
                entering = j;
                break;
            }
        }
        
        if (entering != -1) {
            std::vector<double> u(static_cast<std::size_t>(m), 0.0);
            for (Index idx = A.col_ptr[static_cast<std::size_t>(entering)]; idx < A.col_ptr[static_cast<std::size_t>(entering+1)]; ++idx) {
                u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] = A.values[static_cast<std::size_t>(idx)];
            }
            ftran(etas, u);
            
            double u_p = u[static_cast<std::size_t>(i)];
            if (!std::isfinite(u_p) || std::abs(u_p) <= tol) continue;

            double theta = x_B[static_cast<std::size_t>(i)] / u_p;
            for (Index k = 0; k < m; ++k) {
                if (k == i) x_B[static_cast<std::size_t>(k)] = theta;
                else {
                    x_B[static_cast<std::size_t>(k)] -= theta * u[static_cast<std::size_t>(k)];
                    if (x_B[static_cast<std::size_t>(k)] < 0.0 && x_B[static_cast<std::size_t>(k)] > -tol) {
                        x_B[static_cast<std::size_t>(k)] = 0.0;
                    }
                }
            }

            is_basic[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] = false;
            basis[static_cast<std::size_t>(i)] = entering;
            is_basic[static_cast<std::size_t>(entering)] = true;

            EtaMatrix eta;
            eta.p = i;
            double inv_u_p = 1.0 / u_p;
            for (Index k = 0; k < m; ++k) {
                if (std::abs(u[static_cast<std::size_t>(k)]) > 1e-14 || k == i) {
                    eta.indices.push_back(k);
                    eta.values.push_back(k == i ? inv_u_p : -u[static_cast<std::size_t>(k)] * inv_u_p);
                }
            }
            etas.push_back(std::move(eta));
        }
    }
}

enum class IterateStatus { kOptimal, kUnbounded, kIterationLimit };

IterateStatus iterate(const CSCMatrix& A, const std::vector<double>& b,
                      const std::vector<double>& cost, std::vector<Index>& basis,
                      std::vector<double>& x_B, std::vector<EtaMatrix>& etas,
                      const std::vector<bool>& blocked, double tol,
                      int max_iterations, int& iterations_used) {
    Index m = A.num_rows;
    Index n = A.num_cols;
    int reinvert_freq = 50;
    int iters_since_reinvert = 0;

    std::vector<bool> is_basic(static_cast<std::size_t>(n), false);
    for (Index i = 0; i < m; ++i) {
        is_basic[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] = true;
    }

    while (iterations_used < max_iterations) {
        std::vector<double> c_B(static_cast<std::size_t>(m), 0.0);
        for (Index i = 0; i < m; ++i) c_B[static_cast<std::size_t>(i)] = cost[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])];

        std::vector<double> y = c_B;
        btran(etas, y);

        Index entering = -1;
        double min_rc = -tol;
        for (Index j = 0; j < n; ++j) {
            if (is_basic[static_cast<std::size_t>(j)] || blocked[static_cast<std::size_t>(j)]) continue;
            double z_j = cost[static_cast<std::size_t>(j)];
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)]; idx < A.col_ptr[static_cast<std::size_t>(j+1)]; ++idx) {
                z_j -= y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] * A.values[static_cast<std::size_t>(idx)];
            }
            if (z_j < min_rc) {
                entering = j;
                break; // Bland's rule: first eligible
            }
        }

        if (entering == -1) return IterateStatus::kOptimal;

        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        for (Index idx = A.col_ptr[static_cast<std::size_t>(entering)]; idx < A.col_ptr[static_cast<std::size_t>(entering+1)]; ++idx) {
            u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] = A.values[static_cast<std::size_t>(idx)];
        }
        ftran(etas, u);

        Index leaving = -1;
        double best_ratio = 1e30;
        for (Index i = 0; i < m; ++i) {
            if (u[static_cast<std::size_t>(i)] > tol) {
                double ratio = x_B[static_cast<std::size_t>(i)] / u[static_cast<std::size_t>(i)];
                bool strictly_better = (leaving < 0) || (ratio < best_ratio - tol);
                bool tied = (leaving >= 0) && (std::abs(ratio - best_ratio) <= tol);
                bool lower_index = tied && (basis[static_cast<std::size_t>(i)] < basis[static_cast<std::size_t>(leaving)]);
                if (strictly_better || lower_index) {
                    leaving = i;
                    best_ratio = ratio;
                }
            }
        }

        if (leaving == -1) return IterateStatus::kUnbounded;

        double theta = x_B[static_cast<std::size_t>(leaving)] / u[static_cast<std::size_t>(leaving)];
        for (Index i = 0; i < m; ++i) {
            if (i == leaving) x_B[static_cast<std::size_t>(i)] = theta;
            else {
                x_B[static_cast<std::size_t>(i)] -= theta * u[static_cast<std::size_t>(i)];
                if (x_B[static_cast<std::size_t>(i)] < 0.0 && x_B[static_cast<std::size_t>(i)] > -tol) {
                    x_B[static_cast<std::size_t>(i)] = 0.0;
                }
            }
        }

        is_basic[static_cast<std::size_t>(basis[static_cast<std::size_t>(leaving)])] = false;
        basis[static_cast<std::size_t>(leaving)] = entering;
        is_basic[static_cast<std::size_t>(entering)] = true;

        EtaMatrix eta;
        eta.p = leaving;
        double inv_u_p = 1.0 / u[static_cast<std::size_t>(leaving)];
        for (Index i = 0; i < m; ++i) {
            if (std::abs(u[static_cast<std::size_t>(i)]) > 1e-14 || i == leaving) {
                eta.indices.push_back(i);
                eta.values.push_back(i == leaving ? inv_u_p : -u[static_cast<std::size_t>(i)] * inv_u_p);
            }
        }
        etas.push_back(std::move(eta));

        ++iterations_used;
        ++iters_since_reinvert;

        if (iters_since_reinvert >= reinvert_freq) {
            reinvert(etas, basis, A);
            x_B = b;
            ftran(etas, x_B);
            iters_since_reinvert = 0;
        }
    }
    return IterateStatus::kIterationLimit;
}

}  // namespace

SolveResult RevisedSimplex::solve(const ModelIR& model) const {
    model.validate();

    const double tol = options_.tolerance;
    const Index n = model.numVars();

    SolveResult result;

    const SparseStandardForm sf = buildSparseStandardForm(model, tol);
    const Index num_structural = static_cast<Index>(sf.col_map.size());
    const Index num_rows_std = sf.A.num_rows;

    if (num_rows_std == 0) {
        bool bounded = true;
        for (double c : sf.cost) {
            if (c < -tol) {
                bounded = false;
                break;
            }
        }
        if (!bounded) {
            result.status = SolveStatus::kUnbounded;
            return result;
        }
        result.status = SolveStatus::kOptimal;
        result.x = sf.var_shift;
        result.is_basic.assign(static_cast<std::size_t>(n), false);
        result.row_activity = model.A.multiply(result.x);
        result.objective_value = model.obj_offset;
        for (Index j = 0; j < n; ++j) {
            result.objective_value +=
                model.obj_coeffs[static_cast<std::size_t>(j)] * result.x[static_cast<std::size_t>(j)];
        }
        result.iterations = 0;
        return result;
    }

    std::vector<Index> basis = sf.initial_basis;
    std::vector<double> x_B = sf.b;
    std::vector<EtaMatrix> etas;

    int iterations_used = 0;

    bool need_phase1 = false;
    for (bool f : sf.is_artificial) {
        if (f) {
            need_phase1 = true;
            break;
        }
    }

    if (need_phase1) {
        std::vector<double> cost1(static_cast<std::size_t>(sf.A.num_cols), 0.0);
        for (Index j = 0; j < sf.A.num_cols; ++j) {
            if (sf.is_artificial[static_cast<std::size_t>(j)]) cost1[static_cast<std::size_t>(j)] = 1.0;
        }

        std::vector<bool> none_blocked(static_cast<std::size_t>(sf.A.num_cols), false);
        const IterateStatus st = iterate(sf.A, sf.b, cost1, basis, x_B, etas, none_blocked, tol, options_.max_iterations, iterations_used);

        if (st == IterateStatus::kIterationLimit) {
            result.status = SolveStatus::kIterationLimit;
            result.iterations = iterations_used;
            return result;
        }
        if (st == IterateStatus::kUnbounded) {
            throw std::logic_error("RevisedSimplex: phase 1 reported unbounded, which should be impossible");
        }

        double obj1 = 0.0;
        for (Index i = 0; i < num_rows_std; ++i) {
            obj1 += cost1[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] * x_B[static_cast<std::size_t>(i)];
        }
        if (obj1 > tol) {
            result.status = SolveStatus::kInfeasible;
            result.iterations = iterations_used;
            return result;
        }

        driveOutArtificials(sf.A, basis, x_B, etas, sf.is_artificial, tol);
    }

    const IterateStatus st2 = iterate(sf.A, sf.b, sf.cost, basis, x_B, etas, sf.is_artificial, tol, options_.max_iterations, iterations_used);
    result.iterations = iterations_used;

    if (st2 == IterateStatus::kIterationLimit) {
        result.status = SolveStatus::kIterationLimit;
        return result;
    }
    if (st2 == IterateStatus::kUnbounded) {
        result.status = SolveStatus::kUnbounded;
        return result;
    }

    std::vector<double> x_std(static_cast<std::size_t>(num_structural), 0.0);
    std::vector<bool> col_is_basic(static_cast<std::size_t>(num_structural), false);
    for (Index i = 0; i < num_rows_std; ++i) {
        const Index bcol = basis[static_cast<std::size_t>(i)];
        if (bcol < num_structural) {
            x_std[static_cast<std::size_t>(bcol)] = x_B[static_cast<std::size_t>(i)];
            col_is_basic[static_cast<std::size_t>(bcol)] = true;
        }
    }

    result.x = sf.var_shift;
    result.is_basic.assign(static_cast<std::size_t>(n), false);
    for (Index col = 0; col < num_structural; ++col) {
        const ColumnMap& cm = sf.col_map[static_cast<std::size_t>(col)];
        result.x[static_cast<std::size_t>(cm.original_var)] += cm.sign * x_std[static_cast<std::size_t>(col)];
        if (col_is_basic[static_cast<std::size_t>(col)]) {
            result.is_basic[static_cast<std::size_t>(cm.original_var)] = true;
        }
    }

    result.status = SolveStatus::kOptimal;
    result.row_activity = model.A.multiply(result.x);
    result.objective_value = model.obj_offset;
    for (Index j = 0; j < n; ++j) {
        result.objective_value +=
            model.obj_coeffs[static_cast<std::size_t>(j)] * result.x[static_cast<std::size_t>(j)];
    }
    return result;
}

}  // namespace pramaan