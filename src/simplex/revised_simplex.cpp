// revised_simplex.cpp -- see include/pramaan/simplex.hpp
//
// Implements a classic two-phase, dense, tableau-based primal simplex.
// This file does three things, in order:
//
//   1. Transform the general ModelIR (ranged rows, arbitrary finite/
//      infinite variable bounds, min or max) into standard form:
//          minimize   internal_c^T t
//          subject to  M t {<=,=,>=} rhs   (rhs >= 0)
//                      t >= 0
//      via the textbook substitutions: shift each bounded variable so its
//      lower bound becomes 0, split each free variable into the difference
//      of two non-negative parts, and turn a finite upper bound into an
//      explicit "<=" row. A row that's already ranged (both a finite lower
//      and upper bound) is split into one "<=" row and one ">=" row.
//
//   2. Build the initial dense tableau (structural columns + one slack per
//      "<=" row + one surplus/artificial pair per ">=" row + one artificial
//      per "=" row) and run phase 1 (minimize the sum of artificials) to
//      find a feasible basis, then phase 2 (minimize the real cost) from
//      there. Both phases share the same pivot() / iterate() machinery and
//      use Bland's rule (smallest eligible index, both for entering and for
//      breaking ratio-test ties) so the method is guaranteed to terminate
//      in finitely many pivots -- no cycling heuristics needed for an
//      implementation this size.
//
//   3. Map the standard-form solution back onto the original ModelIR
//      variables, and recompute the objective value and row activity
//      directly from that solution and the model's own (untouched) data,
//      so the reported numbers never depend on getting the internal sign
//      bookkeeping right.
#include "pramaan/simplex.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace pramaan {

namespace {

using Index = CSRMatrix::Index;

// Maps one structural (post-substitution) column back to the original
// ModelIR variable it represents, and the sign with which it contributes:
// +1 for a shifted variable or the positive half of a split free variable,
// -1 for the negative half of a split free variable.
struct ColumnMap {
    Index original_var;
    double sign;
};

enum class RowKind { kLessEqual, kGreaterEqual, kEqual };

struct StdRow {
    std::vector<double> coeffs;  // size == number of structural columns
    double rhs;                   // always >= 0 by construction
    RowKind kind;
};

struct StandardForm {
    std::vector<ColumnMap> columns;      // structural columns
    std::vector<double> internal_cost;   // size == columns.size(); sign-adjusted for kMaximize
    std::vector<double> var_shift;       // size == model.numVars(); added back to recover x
    std::vector<StdRow> rows;
};

// Builds the standard-form LP described in the file header comment.
StandardForm buildStandardForm(const ModelIR& model, double tol) {
    const Index n = model.numVars();
    const Index m = model.numRows();

    StandardForm sf;
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
            plus_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.columns.size());
            sf.columns.push_back(ColumnMap{j, 1.0});
            sf.internal_cost.push_back(ic);
            minus_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.columns.size());
            sf.columns.push_back(ColumnMap{j, -1.0});
            sf.internal_cost.push_back(-ic);
        } else {
            single_col[static_cast<std::size_t>(j)] = static_cast<Index>(sf.columns.size());
            sf.columns.push_back(ColumnMap{j, 1.0});
            sf.internal_cost.push_back(ic);
            sf.var_shift[static_cast<std::size_t>(j)] = lo;
        }
    }

    const Index num_structural = static_cast<Index>(sf.columns.size());

    // Expands one ModelIR row into a dense structural-column coefficient
    // vector, applying the free-variable split and returning how much the
    // row's rhs must be reduced by to account for the shifted variables'
    // constant offsets (A(t + shift) = A t + A*shift, so A t = rhs - A*shift).
    auto rowFromModelRow = [&](Index r) -> std::pair<std::vector<double>, double> {
        std::vector<double> coeffs(static_cast<std::size_t>(num_structural), 0.0);
        double rhs_adjust = 0.0;
        for (const auto& e : model.A.row(r)) {
            const Index j = e.index;
            const double a = e.value;
            if (is_free[static_cast<std::size_t>(j)]) {
                coeffs[static_cast<std::size_t>(plus_col[static_cast<std::size_t>(j)])] += a;
                coeffs[static_cast<std::size_t>(minus_col[static_cast<std::size_t>(j)])] += -a;
            } else {
                coeffs[static_cast<std::size_t>(single_col[static_cast<std::size_t>(j)])] += a;
                rhs_adjust += a * sf.var_shift[static_cast<std::size_t>(j)];
            }
        }
        return {coeffs, rhs_adjust};
    };

    // Appends a row, normalizing its sign so rhs >= 0 (flipping <= <-> >=
    // as needed; = stays =). Every downstream step assumes this invariant.
    auto pushRow = [&](std::vector<double> coeffs, double rhs, RowKind kind) {
        if (rhs < -tol) {
            for (double& v : coeffs) v = -v;
            rhs = -rhs;
            if (kind == RowKind::kLessEqual) kind = RowKind::kGreaterEqual;
            else if (kind == RowKind::kGreaterEqual) kind = RowKind::kLessEqual;
        }
        if (rhs < 0.0) rhs = 0.0;  // clamp residual floating noise from the flip above
        sf.rows.push_back(StdRow{std::move(coeffs), rhs, kind});
    };

    for (Index r = 0; r < m; ++r) {
        if (model.isFreeRow(r)) continue;  // no restriction at all -- nothing to encode
        const double lo = model.row_lower[static_cast<std::size_t>(r)];
        const double up = model.row_upper[static_cast<std::size_t>(r)];
        auto row_pair = rowFromModelRow(r);
        const std::vector<double>& coeffs = row_pair.first;
        const double rhs_adjust = row_pair.second;

        if (model.isEqualityRow(r)) {
            pushRow(coeffs, lo - rhs_adjust, RowKind::kEqual);
        } else if (lo <= -kInfinity) {
            pushRow(coeffs, up - rhs_adjust, RowKind::kLessEqual);
        } else if (up >= kInfinity) {
            pushRow(coeffs, lo - rhs_adjust, RowKind::kGreaterEqual);
        } else {
            pushRow(coeffs, up - rhs_adjust, RowKind::kLessEqual);
            pushRow(coeffs, lo - rhs_adjust, RowKind::kGreaterEqual);
        }
    }

    // Explicit upper-bound rows introduced by shifting/splitting above.
    for (Index j = 0; j < n; ++j) {
        const double up = model.var_upper[static_cast<std::size_t>(j)];
        if (up >= kInfinity) continue;
        std::vector<double> coeffs(static_cast<std::size_t>(num_structural), 0.0);
        double rhs;
        if (is_free[static_cast<std::size_t>(j)]) {
            coeffs[static_cast<std::size_t>(plus_col[static_cast<std::size_t>(j)])] = 1.0;
            coeffs[static_cast<std::size_t>(minus_col[static_cast<std::size_t>(j)])] = -1.0;
            rhs = up;
        } else {
            coeffs[static_cast<std::size_t>(single_col[static_cast<std::size_t>(j)])] = 1.0;
            rhs = up - sf.var_shift[static_cast<std::size_t>(j)];
        }
        pushRow(coeffs, rhs, RowKind::kLessEqual);
    }

    return sf;
}

// Full dense tableau: A is num_rows x num_cols (already reduced so that
// column basis[i] is the i-th unit vector), b is the current basic
// solution, cost holds the current reduced costs (0 on every basic
// column), and obj_value == the current value of whichever objective
// `cost` was built from.
struct Tableau {
    Index num_rows = 0;
    Index num_cols = 0;
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    std::vector<double> cost;
    double obj_value = 0.0;
    std::vector<Index> basis;
};

// Recomputes cost/obj_value for `raw_cost` against the tableau's CURRENT
// basis: cost = raw_cost - c_B^T * A, obj_value = c_B^T * b. Used once to
// start each phase (phase 2 reuses phase 1's already-reduced A/b).
void recomputeReducedCosts(Tableau& t, const std::vector<double>& raw_cost) {
    t.cost = raw_cost;
    t.obj_value = 0.0;
    for (Index i = 0; i < t.num_rows; ++i) {
        const double cb = raw_cost[static_cast<std::size_t>(t.basis[static_cast<std::size_t>(i)])];
        if (cb == 0.0) continue;
        const std::vector<double>& row = t.A[static_cast<std::size_t>(i)];
        for (Index j = 0; j < t.num_cols; ++j) {
            t.cost[static_cast<std::size_t>(j)] -= cb * row[static_cast<std::size_t>(j)];
        }
        t.obj_value += cb * t.b[static_cast<std::size_t>(i)];
    }
}

// Standard Gauss-Jordan pivot on (r, q): normalize row r so A[r][q] == 1,
// then eliminate column q from every other row and from the cost row.
void pivot(Tableau& t, Index r, Index q) {
    std::vector<double>& prow = t.A[static_cast<std::size_t>(r)];
    const double piv = prow[static_cast<std::size_t>(q)];
    for (Index j = 0; j < t.num_cols; ++j) prow[static_cast<std::size_t>(j)] /= piv;
    t.b[static_cast<std::size_t>(r)] /= piv;

    for (Index i = 0; i < t.num_rows; ++i) {
        if (i == r) continue;
        std::vector<double>& row = t.A[static_cast<std::size_t>(i)];
        const double factor = row[static_cast<std::size_t>(q)];
        if (factor == 0.0) continue;
        for (Index j = 0; j < t.num_cols; ++j) {
            row[static_cast<std::size_t>(j)] -= factor * prow[static_cast<std::size_t>(j)];
        }
        t.b[static_cast<std::size_t>(i)] -= factor * t.b[static_cast<std::size_t>(r)];
    }

    const double cfactor = t.cost[static_cast<std::size_t>(q)];
    if (cfactor != 0.0) {
        for (Index j = 0; j < t.num_cols; ++j) {
            t.cost[static_cast<std::size_t>(j)] -= cfactor * prow[static_cast<std::size_t>(j)];
        }
        // obj_value = c_B^T b under the convention used by
        // recomputeReducedCosts(); a pivot changes it by exactly
        // (entering variable's reduced cost) * (amount it increases by),
        // i.e. cfactor * theta, where theta is the entering variable's new
        // value -- which is precisely prow's post-normalization b-entry.
        // (This is a genuine "+=": unlike every other row/column update in
        // this function, obj_value is not itself being eliminated against
        // the pivot row, it is accumulating the entering variable's
        // contribution to the objective.)
        t.obj_value += cfactor * t.b[static_cast<std::size_t>(r)];
    }

    t.basis[static_cast<std::size_t>(r)] = q;
}

enum class IterateStatus { kOptimal, kUnbounded, kIterationLimit };

// Runs primal simplex pivots (Bland's rule) until optimal, unbounded, or
// `max_iterations` total pivots (shared across phase 1 + phase 2, tracked
// via `iterations_used`) is hit. Columns with blocked[j] == true are never
// selected as the entering variable (used in phase 2 to permanently lock
// out artificial columns).
IterateStatus iterate(Tableau& t, const std::vector<bool>& blocked, double tol,
                       int max_iterations, int& iterations_used) {
    while (iterations_used < max_iterations) {
        Index entering = -1;
        for (Index j = 0; j < t.num_cols; ++j) {
            if (blocked[static_cast<std::size_t>(j)]) continue;
            if (t.cost[static_cast<std::size_t>(j)] < -tol) {
                entering = j;
                break;
            }
        }
        if (entering < 0) return IterateStatus::kOptimal;

        Index leaving = -1;
        double best_ratio = 0.0;
        for (Index i = 0; i < t.num_rows; ++i) {
            const double a = t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(entering)];
            if (a <= tol) continue;
            const double ratio = t.b[static_cast<std::size_t>(i)] / a;
            const bool strictly_better = (leaving < 0) || (ratio < best_ratio - tol);
            const bool tied_but_lower_index =
                (leaving >= 0) && (ratio < best_ratio + tol) &&
                (t.basis[static_cast<std::size_t>(i)] < t.basis[static_cast<std::size_t>(leaving)]);
            if (strictly_better || tied_but_lower_index) {
                leaving = i;
                best_ratio = ratio;
            }
        }
        if (leaving < 0) return IterateStatus::kUnbounded;

        pivot(t, leaving, entering);
        ++iterations_used;
    }
    return IterateStatus::kIterationLimit;
}

// After phase 1 reaches a feasible (objective ~0) basis, any artificial
// variable still basic must be sitting at value ~0 (else phase 1 wouldn't
// be optimal). Try to pivot each one out in favor of a genuine column so
// phase 2 starts from a clean basis; if a row has no genuine column with a
// nonzero entry, that row is linearly dependent on the others (a redundant
// constraint) and is simply left with its artificial basic at 0 -- phase 2
// permanently blocks artificials from re-entering, so this is harmless.
void driveOutArtificials(Tableau& t, const std::vector<bool>& is_artificial, double tol) {
    for (Index i = 0; i < t.num_rows; ++i) {
        const Index bcol = t.basis[static_cast<std::size_t>(i)];
        if (!is_artificial[static_cast<std::size_t>(bcol)]) continue;
        for (Index j = 0; j < t.num_cols; ++j) {
            if (is_artificial[static_cast<std::size_t>(j)]) continue;
            if (std::abs(t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) > tol) {
                pivot(t, i, j);
                break;
            }
        }
    }
}

}  // namespace

SolveResult RevisedSimplex::solve(const ModelIR& model) const {
    model.validate();

    const double tol = options_.tolerance;
    const Index n = model.numVars();

    SolveResult result;

    const StandardForm sf = buildStandardForm(model, tol);
    const Index num_structural = static_cast<Index>(sf.columns.size());
    const Index num_rows_std = static_cast<Index>(sf.rows.size());

    // No rows at all: every original row was free and every variable had a
    // finite lower / infinite upper bound with no "<=" constraint to pivot
    // against. The feasible region is just {t >= 0}, so the LP is bounded
    // iff every structural cost is already non-negative, and if so t = 0
    // (i.e. every variable sits at its lower bound) is optimal.
    if (num_rows_std == 0) {
        bool bounded = true;
        for (double c : sf.internal_cost) {
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
        result.x = sf.var_shift;  // size n; t == 0 everywhere, so x == shift
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

    // --- lay out slack / surplus / artificial columns, one group per row,
    // appended after the structural columns ---
    std::vector<Index> slack_or_surplus_col(static_cast<std::size_t>(num_rows_std), -1);
    std::vector<Index> artificial_col(static_cast<std::size_t>(num_rows_std), -1);
    Index next_col = num_structural;
    for (Index i = 0; i < num_rows_std; ++i) {
        switch (sf.rows[static_cast<std::size_t>(i)].kind) {
            case RowKind::kLessEqual:
                slack_or_surplus_col[static_cast<std::size_t>(i)] = next_col++;
                break;
            case RowKind::kGreaterEqual:
                slack_or_surplus_col[static_cast<std::size_t>(i)] = next_col++;
                artificial_col[static_cast<std::size_t>(i)] = next_col++;
                break;
            case RowKind::kEqual:
                artificial_col[static_cast<std::size_t>(i)] = next_col++;
                break;
        }
    }
    const Index num_cols = next_col;

    Tableau t;
    t.num_rows = num_rows_std;
    t.num_cols = num_cols;
    t.A.assign(static_cast<std::size_t>(num_rows_std),
               std::vector<double>(static_cast<std::size_t>(num_cols), 0.0));
    t.b.assign(static_cast<std::size_t>(num_rows_std), 0.0);
    t.basis.assign(static_cast<std::size_t>(num_rows_std), -1);

    std::vector<bool> is_artificial(static_cast<std::size_t>(num_cols), false);

    for (Index i = 0; i < num_rows_std; ++i) {
        const StdRow& row = sf.rows[static_cast<std::size_t>(i)];
        for (Index j = 0; j < num_structural; ++j) {
            t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = row.coeffs[static_cast<std::size_t>(j)];
        }
        t.b[static_cast<std::size_t>(i)] = row.rhs;

        switch (row.kind) {
            case RowKind::kLessEqual: {
                const Index s = slack_or_surplus_col[static_cast<std::size_t>(i)];
                t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(s)] = 1.0;
                t.basis[static_cast<std::size_t>(i)] = s;
                break;
            }
            case RowKind::kGreaterEqual: {
                const Index s = slack_or_surplus_col[static_cast<std::size_t>(i)];
                const Index a = artificial_col[static_cast<std::size_t>(i)];
                t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(s)] = -1.0;
                t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(a)] = 1.0;
                is_artificial[static_cast<std::size_t>(a)] = true;
                t.basis[static_cast<std::size_t>(i)] = a;
                break;
            }
            case RowKind::kEqual: {
                const Index a = artificial_col[static_cast<std::size_t>(i)];
                t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(a)] = 1.0;
                is_artificial[static_cast<std::size_t>(a)] = true;
                t.basis[static_cast<std::size_t>(i)] = a;
                break;
            }
        }
    }

    int iterations_used = 0;

    bool need_phase1 = false;
    for (bool f : is_artificial) {
        if (f) {
            need_phase1 = true;
            break;
        }
    }

    if (need_phase1) {
        std::vector<double> raw_cost1(static_cast<std::size_t>(num_cols), 0.0);
        for (Index j = 0; j < num_cols; ++j) {
            if (is_artificial[static_cast<std::size_t>(j)]) raw_cost1[static_cast<std::size_t>(j)] = 1.0;
        }
        recomputeReducedCosts(t, raw_cost1);

        const std::vector<bool> none_blocked(static_cast<std::size_t>(num_cols), false);
        const IterateStatus st = iterate(t, none_blocked, tol, options_.max_iterations, iterations_used);

        if (st == IterateStatus::kIterationLimit) {
            result.status = SolveStatus::kIterationLimit;
            result.iterations = iterations_used;
            return result;
        }
        if (st == IterateStatus::kUnbounded) {
            // Phase 1 minimizes a sum of non-negative artificial variables,
            // which is bounded below by 0 -- this outcome is unreachable
            // for a correct implementation, so surface it loudly rather
            // than mislabel the result.
            throw std::logic_error(
                "RevisedSimplex: phase 1 reported unbounded, which should be impossible");
        }
        if (t.obj_value > tol) {
            result.status = SolveStatus::kInfeasible;
            result.iterations = iterations_used;
            return result;
        }

        driveOutArtificials(t, is_artificial, tol);
    }

    std::vector<double> raw_cost2(static_cast<std::size_t>(num_cols), 0.0);
    for (Index j = 0; j < num_structural; ++j) {
        raw_cost2[static_cast<std::size_t>(j)] = sf.internal_cost[static_cast<std::size_t>(j)];
    }
    recomputeReducedCosts(t, raw_cost2);

    const IterateStatus st2 = iterate(t, is_artificial, tol, options_.max_iterations, iterations_used);
    result.iterations = iterations_used;

    if (st2 == IterateStatus::kIterationLimit) {
        result.status = SolveStatus::kIterationLimit;
        return result;
    }
    if (st2 == IterateStatus::kUnbounded) {
        result.status = SolveStatus::kUnbounded;
        return result;
    }

    // --- extract the standard-form solution, then map it back onto the
    // original ModelIR variables ---
    std::vector<double> x_std(static_cast<std::size_t>(num_structural), 0.0);
    std::vector<bool> col_is_basic(static_cast<std::size_t>(num_structural), false);
    for (Index i = 0; i < num_rows_std; ++i) {
        const Index bcol = t.basis[static_cast<std::size_t>(i)];
        if (bcol < num_structural) {
            x_std[static_cast<std::size_t>(bcol)] = t.b[static_cast<std::size_t>(i)];
            col_is_basic[static_cast<std::size_t>(bcol)] = true;
        }
    }

    result.x = sf.var_shift;  // size n; starts each var at its lower bound
    result.is_basic.assign(static_cast<std::size_t>(n), false);
    for (Index col = 0; col < num_structural; ++col) {
        const ColumnMap& cm = sf.columns[static_cast<std::size_t>(col)];
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