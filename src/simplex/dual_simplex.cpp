// dual_simplex.cpp
// P1 Step 1 — Dual Simplex + Basis Warm-Start
//
// Implements:
//   DualSimplex::captureBasis()  — cold-solves and captures the optimal basis
//   DualSimplex::warmSolve()     — warm-solves from a captured basis using
//                                  genuine dual-simplex pivots
//
// The standard-form transformation is replicated from revised_simplex.cpp
// (both live in anonymous namespaces, so sharing requires either a refactor
// to a shared internal header or duplication — we choose duplication to avoid
// touching the verified P0 primal-simplex code).
//
// Dual simplex algorithm:
//   Given a dual-feasible basis B (which any optimal primal basis is),
//   after a RHS/bound change the basic solution x_B = B^{-1} b may have
//   negative components.  The dual simplex picks the most-infeasible
//   basic variable as the leaving variable (row r with x_B[r] < 0),
//   then performs a ratio test on the reduced costs to find the entering
//   variable that maintains dual feasibility after the pivot.  Each pivot
//   reduces primal infeasibility while preserving dual feasibility, until
//   all x_B >= 0 (optimal) or no valid entering variable exists (infeasible).
#include "pramaan/dual_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>

namespace pramaan {

namespace {

using Index = CSRMatrix::Index;

// --- Standard-form data structures (same as revised_simplex.cpp) ----------

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



uint64_t computeLayoutFingerprint(const SparseStandardForm& sf) {
    uint64_t hash = 8469598103934665603ULL;
    auto add_double = [&](double v) {
        uint64_t bits;
        if (v == 0.0) v = 0.0;
        std::memcpy(&bits, &v, sizeof(double));
        bits ^= bits >> 33;
        bits *= 0xff51afd7ed558ccdULL;
        bits ^= bits >> 33;
        hash ^= bits;
        hash *= 1099511628211ULL;
    };
    auto add_int = [&](uint64_t v) {
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccdULL;
        v ^= v >> 33;
        hash ^= v;
        hash *= 1099511628211ULL;
    };
    add_int(sf.A.num_rows);
    add_int(sf.A.num_cols);
    for (auto v : sf.A.col_ptr) add_int(v);
    for (auto v : sf.A.row_idx) add_int(v);
    for (auto v : sf.A.values) add_double(v);
    for (bool b : sf.is_artificial) {
        add_int(b ? 1 : 0);
    }
    return hash;
}

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
        if (row_kinds[static_cast<std::size_t>(i)] == RowKind::kEqual ||
            row_kinds[static_cast<std::size_t>(i)] == RowKind::kGreaterEqual) {
            sf.is_artificial[static_cast<std::size_t>(sf.initial_basis[static_cast<std::size_t>(i)])] = true;
        }
    }

    sf.A = buildCSC(num_rows_std, num_cols_std, triplets);
    return sf;
}

// --- PFI / Eta infrastructure (same as revised_simplex.cpp) ---------------

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

void reinvert(std::vector<EtaMatrix>& etas, std::vector<Index>& basis, const CSCMatrix& A) {
    etas.clear();
    Index m = A.num_rows;
    std::vector<bool> row_used(static_cast<std::size_t>(m), false);
    std::vector<Index> new_basis(static_cast<std::size_t>(m), -1);

    for (Index k = 0; k < m; ++k) {
        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        Index bcol = basis[static_cast<std::size_t>(k)];
        for (Index idx = A.col_ptr[static_cast<std::size_t>(bcol)];
             idx < A.col_ptr[static_cast<std::size_t>(bcol + 1)]; ++idx) {
            u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] =
                A.values[static_cast<std::size_t>(idx)];
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

// --- Primal simplex iterate (same as revised_simplex.cpp) -----------------
// Needed by captureBasis() to cold-solve the original LP.

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
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)];
                 idx < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++idx) {
                a_ij += y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] *
                        A.values[static_cast<std::size_t>(idx)];
            }
            if (std::abs(a_ij) > tol && std::isfinite(a_ij)) {
                entering = j;
                break;
            }
        }

        if (entering != -1) {
            std::vector<double> u(static_cast<std::size_t>(m), 0.0);
            for (Index idx = A.col_ptr[static_cast<std::size_t>(entering)];
                 idx < A.col_ptr[static_cast<std::size_t>(entering + 1)]; ++idx) {
                u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] =
                    A.values[static_cast<std::size_t>(idx)];
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

IterateStatus iteratePrimal(const CSCMatrix& A, const std::vector<double>& b,
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
        for (Index i = 0; i < m; ++i)
            c_B[static_cast<std::size_t>(i)] = cost[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])];

        std::vector<double> y = c_B;
        btran(etas, y);

        Index entering = -1;
        double min_rc = -tol;
        for (Index j = 0; j < n; ++j) {
            if (is_basic[static_cast<std::size_t>(j)] || blocked[static_cast<std::size_t>(j)]) continue;
            double z_j = cost[static_cast<std::size_t>(j)];
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)];
                 idx < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++idx) {
                z_j -= y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] *
                       A.values[static_cast<std::size_t>(idx)];
            }
            if (z_j < min_rc) {
                entering = j;
                break;
            }
        }

        if (entering == -1) return IterateStatus::kOptimal;

        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        for (Index idx = A.col_ptr[static_cast<std::size_t>(entering)];
             idx < A.col_ptr[static_cast<std::size_t>(entering + 1)]; ++idx) {
            u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] =
                A.values[static_cast<std::size_t>(idx)];
        }
        ftran(etas, u);

        Index leaving = -1;
        double best_ratio = 1e30;
        for (Index i = 0; i < m; ++i) {
            if (u[static_cast<std::size_t>(i)] > tol) {
                double ratio = x_B[static_cast<std::size_t>(i)] / u[static_cast<std::size_t>(i)];
                bool strictly_better = (leaving < 0) || (ratio < best_ratio - tol);
                bool tied = (leaving >= 0) && (std::abs(ratio - best_ratio) <= tol);
                bool lower_index = tied && (basis[static_cast<std::size_t>(i)] <
                                            basis[static_cast<std::size_t>(leaving)]);
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

// --- Dual simplex iterate ------------------------------------------------
// Starting from a dual-feasible basis (all reduced costs >= 0 for
// non-basic variables), restores primal feasibility (all x_B >= 0)
// through dual-simplex pivots.
//
// Leaving variable selection: most negative x_B[r] (most infeasible).
// Entering variable selection: among non-basic j with u_r[j] < 0
//   (where u_r = e_r^T B^{-1} A_j = row r of B^{-1} A), find j that
//   minimizes |reduced_cost_j / u_r[j]| — this is the dual ratio test
//   that maintains dual feasibility after the pivot.

enum class DualIterateStatus { kOptimal, kInfeasible, kIterationLimit, kNumericalFailure };

DualIterateStatus iterateDual(const CSCMatrix& A, const std::vector<double>& b,
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

    // Verify initial dual feasibility
    std::vector<double> init_c_B(static_cast<std::size_t>(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        init_c_B[static_cast<std::size_t>(i)] = cost[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])];
    }
    std::vector<double> init_y = init_c_B;
    btran(etas, init_y);
    for (Index j = 0; j < n; ++j) {
        if (is_basic[static_cast<std::size_t>(j)] || blocked[static_cast<std::size_t>(j)]) continue;
        double d_j = cost[static_cast<std::size_t>(j)];
        for (Index idx = A.col_ptr[static_cast<std::size_t>(j)];
             idx < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++idx) {
            d_j -= init_y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] *
                   A.values[static_cast<std::size_t>(idx)];
        }
        if (d_j < -tol) {
            return DualIterateStatus::kNumericalFailure;
        }
    }

    while (iterations_used < max_iterations) {
        // --- Leaving variable: most negative x_B[r] ---
        Index leaving = -1;
        double most_neg = -tol;
        for (Index i = 0; i < m; ++i) {
            if (x_B[static_cast<std::size_t>(i)] < most_neg) {
                most_neg = x_B[static_cast<std::size_t>(i)];
                leaving = i;
            }
        }

        // All x_B >= -tol: primal feasible => optimal.
        if (leaving == -1) {
            // Force strict non-negativity for exact feasibility extraction
            for (Index i = 0; i < m; ++i) {
                if (x_B[static_cast<std::size_t>(i)] < 0.0 && x_B[static_cast<std::size_t>(i)] >= -tol) {
                    x_B[static_cast<std::size_t>(i)] = 0.0;
                }
            }
            return DualIterateStatus::kOptimal;
        }

        // --- Compute row `leaving` of B^{-1} A for the ratio test ---
        std::vector<double> rho(static_cast<std::size_t>(m), 0.0);
        rho[static_cast<std::size_t>(leaving)] = 1.0;
        btran(etas, rho);

        // Compute reduced costs for all non-basic columns.
        std::vector<double> c_B(static_cast<std::size_t>(m), 0.0);
        for (Index i = 0; i < m; ++i)
            c_B[static_cast<std::size_t>(i)] = cost[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])];
        std::vector<double> y = c_B;
        btran(etas, y);

        // --- Dual ratio test ---
        Index entering = -1;
        double best_ratio = kInfinity;
        for (Index j = 0; j < n; ++j) {
            if (is_basic[static_cast<std::size_t>(j)] || blocked[static_cast<std::size_t>(j)]) continue;

            double alpha_j = 0.0;
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)];
                 idx < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++idx) {
                alpha_j += rho[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] *
                           A.values[static_cast<std::size_t>(idx)];
            }

            if (alpha_j > -tol) continue;  // need alpha_j < 0 for dual-simplex

            double d_j = cost[static_cast<std::size_t>(j)];
            for (Index idx = A.col_ptr[static_cast<std::size_t>(j)];
                 idx < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++idx) {
                d_j -= y[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] *
                       A.values[static_cast<std::size_t>(idx)];
            }

            double ratio = d_j / (-alpha_j);
            bool strictly_better = (entering < 0) || (ratio < best_ratio - tol);
            bool tied = (entering >= 0) && (std::abs(ratio - best_ratio) <= tol);
            bool lower_index = tied && (j < basis[static_cast<std::size_t>(entering)]);
            if (strictly_better || lower_index) {
                entering = j;
                best_ratio = ratio;
            }
        }

        if (entering == -1) return DualIterateStatus::kInfeasible;

        // --- Perform the pivot ---
        std::vector<double> u(static_cast<std::size_t>(m), 0.0);
        for (Index idx = A.col_ptr[static_cast<std::size_t>(entering)];
             idx < A.col_ptr[static_cast<std::size_t>(entering + 1)]; ++idx) {
            u[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(idx)])] =
                A.values[static_cast<std::size_t>(idx)];
        }
        ftran(etas, u);

        double u_leaving = u[static_cast<std::size_t>(leaving)];

        // Critical Fix: Never fabricate primal feasibility. If a valid pivot
        // cannot be performed, safely recompute/reinvert and retry.
        if (!std::isfinite(u_leaving) || u_leaving >= -tol) {
            if (iters_since_reinvert == 0) {
                return DualIterateStatus::kNumericalFailure;
            }
            reinvert(etas, basis, A);
            x_B = b;
            ftran(etas, x_B);
            iters_since_reinvert = 0;
            continue;
        }

        double theta = x_B[static_cast<std::size_t>(leaving)] / u_leaving;
        for (Index i = 0; i < m; ++i) {
            if (i == leaving) {
                x_B[static_cast<std::size_t>(i)] = theta;
            } else {
                x_B[static_cast<std::size_t>(i)] -= theta * u[static_cast<std::size_t>(i)];
            }
        }

        is_basic[static_cast<std::size_t>(basis[static_cast<std::size_t>(leaving)])] = false;
        basis[static_cast<std::size_t>(leaving)] = entering;
        is_basic[static_cast<std::size_t>(entering)] = true;

        EtaMatrix eta;
        eta.p = leaving;
        double inv_u_p = 1.0 / u_leaving;
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
    return DualIterateStatus::kIterationLimit;
}

// --- Extract original-space solution from standard-form -------------------

SolveResult extractSolution(const ModelIR& model, const SparseStandardForm& sf,
                            const std::vector<Index>& basis, const std::vector<double>& x_B,
                            SolveStatus status, int iterations) {
    const Index n = model.numVars();
    const Index num_structural = static_cast<Index>(sf.col_map.size());
    const Index num_rows_std = sf.A.num_rows;

    SolveResult result;
    result.status = status;
    result.iterations = iterations;

    if (status != SolveStatus::kOptimal) return result;

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

    result.row_activity = model.A.multiply(result.x);
    result.objective_value = model.obj_offset;
    for (Index j = 0; j < n; ++j) {
        result.objective_value +=
            model.obj_coeffs[static_cast<std::size_t>(j)] * result.x[static_cast<std::size_t>(j)];
    }
    return result;
}

}  // namespace

// =========================================================================
// DualSimplex::captureBasis()
// =========================================================================

// `out_result`, when non-null, receives the SolveResult of the cold solve
// this function already performs internally. This lets a caller (notably the
// B&B engine) obtain both the LP result and the warm-startable basis from a
// single solve instead of solving the same relaxation twice.
//
// Note the two outputs are independent: a solve can reach optimality while
// still yielding no usable basis (e.g. an artificial remains basic on a
// redundant row). In that case *out_result is the optimal result and the
// returned BasisState is empty, which callers must treat as "no warm start
// available", not as a failed solve.
BasisState DualSimplex::captureBasis(const ModelIR& model, SolveResult* out_result) const {
    auto emit = [&](SolveStatus status, const SparseStandardForm* sf,
                    const std::vector<Index>* basis, const std::vector<double>* x_B,
                    int iters) {
        if (out_result == nullptr) return;
        if (sf == nullptr || basis == nullptr || x_B == nullptr) {
            *out_result = SolveResult{};
            out_result->status = status;
            out_result->iterations = iters;
        } else {
            *out_result = extractSolution(model, *sf, *basis, *x_B, status, iters);
        }
    };

    model.validate();
    const double tol = options_.tolerance;
    const Index n = model.numVars();

    const SparseStandardForm sf = buildSparseStandardForm(model, tol);
    const Index num_rows_std = sf.A.num_rows;

    if (num_rows_std == 0) {
        // Trivial model: no constraints survive standard-form conversion, so
        // there is no basis to capture, but the LP itself is solved.
        if (out_result != nullptr) {
            *out_result = SolveResult{};
            out_result->status = SolveStatus::kOptimal;
            out_result->x = sf.var_shift;
            out_result->is_basic.assign(static_cast<std::size_t>(n), false);
            out_result->row_activity = model.A.multiply(out_result->x);
            out_result->objective_value = model.obj_offset;
            for (Index j = 0; j < n; ++j) {
                out_result->objective_value += model.obj_coeffs[static_cast<std::size_t>(j)] *
                                               out_result->x[static_cast<std::size_t>(j)];
            }
        }
        return BasisState{};
    }

    std::vector<Index> basis = sf.initial_basis;
    std::vector<double> x_B = sf.b;
    std::vector<EtaMatrix> etas;
    int iterations_used = 0;

    // Phase 1 if needed
    bool need_phase1 = false;
    for (bool f : sf.is_artificial) {
        if (f) { need_phase1 = true; break; }
    }

    if (need_phase1) {
        std::vector<double> cost1(static_cast<std::size_t>(sf.A.num_cols), 0.0);
        for (Index j = 0; j < sf.A.num_cols; ++j) {
            if (sf.is_artificial[static_cast<std::size_t>(j)]) cost1[static_cast<std::size_t>(j)] = 1.0;
        }
        std::vector<bool> none_blocked(static_cast<std::size_t>(sf.A.num_cols), false);
        const IterateStatus st = iteratePrimal(sf.A, sf.b, cost1, basis, x_B, etas, none_blocked,
                                               tol, options_.max_iterations, iterations_used);
        if (st != IterateStatus::kOptimal) {
            emit(st == IterateStatus::kUnbounded ? SolveStatus::kNumericalFailure
                                                 : SolveStatus::kIterationLimit,
                 nullptr, nullptr, nullptr, iterations_used);
            return BasisState{};
        }

        double obj1 = 0.0;
        for (Index i = 0; i < num_rows_std; ++i) {
            obj1 += cost1[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] *
                    x_B[static_cast<std::size_t>(i)];
        }
        if (obj1 > tol) {  // infeasible
            emit(SolveStatus::kInfeasible, nullptr, nullptr, nullptr, iterations_used);
            return BasisState{};
        }

        driveOutArtificials(sf.A, basis, x_B, etas, sf.is_artificial, tol);
    }

    // Phase 2
    const IterateStatus st2 = iteratePrimal(sf.A, sf.b, sf.cost, basis, x_B, etas,
                                            sf.is_artificial, tol, options_.max_iterations, iterations_used);
    if (st2 != IterateStatus::kOptimal) {
        emit(st2 == IterateStatus::kUnbounded ? SolveStatus::kUnbounded
                                              : SolveStatus::kIterationLimit,
             nullptr, nullptr, nullptr, iterations_used);
        return BasisState{};
    }

    // FIX 2: Reject basis if any artificial variables are still basic.
    // If driveOutArtificials could not pivot them out, they remain at 0 (e.g. redundant constraints).
    // Warm-starting from such a basis is unsafe because the redundant constraint's RHS might change.
    for (Index i = 0; i < num_rows_std; ++i) {
        if (sf.is_artificial[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])]) {
            // The LP is solved to optimality; only the basis is unusable for
            // a later warm start.
            emit(SolveStatus::kOptimal, &sf, &basis, &x_B, iterations_used);
            return BasisState{};
        }
    }

    emit(SolveStatus::kOptimal, &sf, &basis, &x_B, iterations_used);

    // Capture the optimal basis
    BasisState state;
    state.basis_columns.resize(static_cast<std::size_t>(num_rows_std));
    for (Index i = 0; i < num_rows_std; ++i) {
        state.basis_columns[static_cast<std::size_t>(i)] = basis[static_cast<std::size_t>(i)];
    }
    state.structural_fingerprint = computeStructuralFingerprint(model);
    state.std_layout_fingerprint = computeLayoutFingerprint(sf);
    state.orig_num_vars = n;
    state.orig_num_rows = model.numRows();
    state.orig_obj_sense = model.obj_sense;

    return state;
}

// =========================================================================
// DualSimplex::warmSolve()
// =========================================================================

// `out_basis`, when non-null, receives the basis this solve ends on, so the
// caller can warm-start a further modification (in B&B: a grandchild node)
// without re-solving. It is left empty unless the solve reached optimality
// with no artificial column basic, matching captureBasis()'s safety rule.
SolveResult DualSimplex::warmSolve(const ModelIR& model, const BasisState& basis_state,
                                   BasisState* out_basis) const {
    if (out_basis != nullptr) *out_basis = BasisState{};
    model.validate();

    if (basis_state.empty()) {
        SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
    }

    // Compatibility checks
    if (model.numVars() != basis_state.orig_num_vars ||
        model.numRows() != basis_state.orig_num_rows ||
        model.obj_sense != basis_state.orig_obj_sense) {
        SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
    }

    if (computeStructuralFingerprint(model) != basis_state.structural_fingerprint) {
        SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
    }

    const double tol = options_.tolerance;
    const Index n = model.numVars();

    // Rebuild standard form with the new RHS/bounds.
    const SparseStandardForm sf = buildSparseStandardForm(model, tol);
    const Index num_rows_std = sf.A.num_rows;

    if (num_rows_std == 0) {
        // Trivial: no constraints left after standard-form conversion.
        SolveResult result;
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

    // The standard-form dimensions must match. If the number of standard-form
    // rows changed (e.g. because a variable's finite bound became infinite or
    // vice versa, which adds/removes bound rows), the cached basis is invalid.
    if (static_cast<Index>(basis_state.basis_columns.size()) != num_rows_std) {
        SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
    }

    uint64_t sf_hash = computeLayoutFingerprint(sf);
    if (sf_hash != basis_state.std_layout_fingerprint) {
        SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
    }

    // Validate that all basis column indices are in range and non-duplicate.
    // Must also explicitly reject any artificial columns.
    {
        std::vector<bool> seen(static_cast<std::size_t>(sf.A.num_cols), false);
        for (Index i = 0; i < num_rows_std; ++i) {
            Index col = basis_state.basis_columns[static_cast<std::size_t>(i)];
            if (col < 0 || col >= sf.A.num_cols) {
                SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
            }
            if (seen[static_cast<std::size_t>(col)]) {
                SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
            }
            if (sf.is_artificial[static_cast<std::size_t>(col)]) {
                SolveResult rej; rej.status = SolveStatus::kWarmStartRejected; return rej;
            }
            seen[static_cast<std::size_t>(col)] = true;
        }
    }

    // Inject the inherited basis.
    std::vector<Index> basis(static_cast<std::size_t>(num_rows_std));
    for (Index i = 0; i < num_rows_std; ++i) {
        basis[static_cast<std::size_t>(i)] = basis_state.basis_columns[static_cast<std::size_t>(i)];
    }

    // Build PFI for the inherited basis via reinvert.
    std::vector<EtaMatrix> etas;
    reinvert(etas, basis, sf.A);

    // Recompute x_B = B^{-1} b for the new RHS.
    std::vector<double> x_B = sf.b;
    ftran(etas, x_B);

    int iterations_used = 0;

    // Run dual-simplex pivots to restore primal feasibility.
    const DualIterateStatus dst = iterateDual(sf.A, sf.b, sf.cost, basis, x_B, etas,
                                              sf.is_artificial, tol,
                                              options_.max_iterations, iterations_used);

    SolveStatus status;
    switch (dst) {
        case DualIterateStatus::kOptimal:          status = SolveStatus::kOptimal; break;
        case DualIterateStatus::kInfeasible:       status = SolveStatus::kInfeasible; break;
        case DualIterateStatus::kNumericalFailure: status = SolveStatus::kNumericalFailure; break;
        default:                                   status = SolveStatus::kIterationLimit; break;
    }

    if (out_basis != nullptr && status == SolveStatus::kOptimal) {
        bool artificial_basic = false;
        for (Index i = 0; i < num_rows_std; ++i) {
            if (sf.is_artificial[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])]) {
                artificial_basic = true;
                break;
            }
        }
        if (!artificial_basic) {
            out_basis->basis_columns.assign(basis.begin(), basis.end());
            out_basis->structural_fingerprint = computeStructuralFingerprint(model);
            out_basis->std_layout_fingerprint = computeLayoutFingerprint(sf);
            out_basis->orig_num_vars = n;
            out_basis->orig_num_rows = model.numRows();
            out_basis->orig_obj_sense = model.obj_sense;
        }
    }

    return extractSolution(model, sf, basis, x_B, status, iterations_used);
}

}  // namespace pramaan
