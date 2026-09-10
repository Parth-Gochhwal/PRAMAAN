// simplex.hpp
// PILLAR 2 (Speed) -- Zone 4 "Heterogeneous Continuous Engine", CPU half
//
// Purpose: LP solver entry point. This is the first real solver algorithm --
// everything before this file was plumbing (sparse storage, IR).
//
// SCOPE OF THIS MILESTONE
// ------------------------
// Per the roadmap, this is deliberately the DENSE, TEXTBOOK, tableau-based
// simplex -- not the sparse revised (basis-inverse) formulation. Internally
// it runs the classic two-phase primal simplex method on a full dense
// tableau (see revised_simplex.cpp), using Bland's rule throughout for a
// guaranteed-finite-termination, easy-to-hand-verify implementation. It is
// correct for any ModelIR (arbitrary finite/infinite variable bounds, any
// mix of <=, >=, =, ranged and free rows, min or max), which makes it a
// solid reference oracle for the sparse/revised engine that replaces it
// later -- but it is O(rows * cols) per pivot with no exploitation of
// sparsity, so it is only intended for the small, hand-checkable instances
// this milestone targets, not for Netlib-scale models.
//
// NOT YET DONE (left for later milestones, intentionally):
//   - sparse / revised (basis-inverse) formulation
//   - dual simplex / warm start from a prior basis
//   - presolve integration
//   - numerical stability machinery beyond a single fixed tolerance
//     (e.g. dynamic scaling, Harris ratio test)
#pragma once

#include <vector>

#include "pramaan/ir.hpp"

namespace pramaan {

// Outcome of a solve attempt. This mirrors the vocabulary every LP solver
// (HiGHS, CPLEX, Gurobi) uses, so downstream code (branch-and-bound,
// certificates) can switch on it without solver-specific translation.
enum class SolveStatus {
    kOptimal,          // solved to optimality within tolerance
    kInfeasible,        // phase 1 proved no feasible point exists
    kUnbounded,          // objective is unbounded on the feasible region
    kIterationLimit,   // stopped without a conclusive result (see SolveResult::iterations)
};

// Result of RevisedSimplex::solve().
//
// `x`, `objective_value`, and `row_activity` are only meaningful when
// `status == SolveStatus::kOptimal`; for other statuses they are left at
// their default-constructed (empty/zero) values.
struct SolveResult {
    SolveStatus status = SolveStatus::kIterationLimit;

    // Primal solution, in ORIGINAL ModelIR variable order/units --
    // size == model.numVars() when status == kOptimal, empty otherwise.
    std::vector<double> x;

    // c^T x + obj_offset, computed directly from `x` and the model's
    // original (un-transformed) objective, in the model's own sense
    // (i.e. this is a maximum when obj_sense == kMaximize, not a negated
    // internal minimum) -- so callers never need to know how the solver
    // represented the problem internally.
    double objective_value = 0.0;

    // A * x, size == model.numRows() -- handy for verifying a solution
    // against the original row bounds without recomputing it.
    std::vector<double> row_activity;

    // Total simplex pivots performed across both phases.
    int iterations = 0;

    // True basic/nonbasic status per structural variable, in ORIGINAL
    // ModelIR variable order -- size == model.numVars() when status ==
    // kOptimal. This is a placeholder for warm-starting: the internal
    // standard-form tableau (with its slack/surplus/artificial columns and
    // free-variable splits) is not exposed, since that layout is an
    // implementation detail that will change completely once this becomes
    // a revised (basis-inverse) simplex. `is_basic[j]` is the one piece of
    // that information that IS stable across that future rewrite, so it's
    // exposed now and the rest is deferred.
    std::vector<bool> is_basic;
};

// Solves a general linear program expressed as a ModelIR:
//
//     optimize   obj_sense: c^T x + obj_offset
//     subject to row_lower <= A x <= row_upper
//                var_lower <= x   <= var_upper
//
// via a dense, two-phase, tableau-based primal simplex (see simplex.hpp's
// header comment for exactly what this does and doesn't handle yet).
//
// Thread-safety: a RevisedSimplex instance holds no mutable state between
// calls -- `solve()` builds and discards its own tableau -- so the same
// instance may be reused (even concurrently) across multiple models.
class RevisedSimplex {
public:
    struct Options {
        // Hard cap on total pivots across both phases. A tiny hand-solved
        // LP should never come close to this; it exists so a bug (or a
        // model this milestone isn't meant to handle) fails as
        // kIterationLimit instead of hanging.
        int max_iterations = 10000;

        // Absolute tolerance used for every "is this zero / non-negative /
        // improving" comparison in the tableau (reduced costs, ratio test,
        // feasibility of bounds). 1e-9 is appropriate for the well-scaled,
        // small, hand-built instances this milestone targets.
        double tolerance = 1e-9;
    };

    RevisedSimplex() = default;
    explicit RevisedSimplex(Options options) : options_(options) {}

    SolveResult solve(const ModelIR& model) const;

    const Options& options() const noexcept { return options_; }

private:
    Options options_{};
};

}  // namespace pramaan