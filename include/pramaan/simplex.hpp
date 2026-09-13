// simplex.hpp
// PILLAR 2 (Speed) -- Zone 4 "Heterogeneous Continuous Engine", CPU half
//
// Purpose: LP solver entry point. This is the first real solver algorithm --
// everything before this file was plumbing (sparse storage, IR).
//
// SCOPE OF THIS MILESTONE
// ------------------------
// This implements a full sparse revised primal simplex method using the Product
// Form of the Inverse (PFI). It supports arbitrary finite/infinite variable bounds,
// any mix of <=, >=, =, ranged and free rows, min or max. It is designed to be
// the primary robust solver engine for the project, capable of solving Netlib
// benchmark instances accurately and efficiently using sparse matrix operations.
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
    kNumericalFailure, // stopped due to unrecoverable numerical instability
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
    // kOptimal. The full internal standard-form basis (including
    // slack/surplus/artificial columns and free-variable splits) is not
    // exposed through SolveResult. `is_basic[j]` reports whether structural
    // variable j is basic in the final basis.
    std::vector<bool> is_basic;
};

// Solves a general linear program expressed as a ModelIR:
//
//     optimize   obj_sense: c^T x + obj_offset
//     subject to row_lower <= A x <= row_upper
//                var_lower <= x   <= var_upper
//
// via a sparse, two-phase revised primal simplex.
//
// Thread-safety: a RevisedSimplex instance holds no mutable state between
// calls -- `solve()` builds and discards its own internal state -- so the same
// instance may be reused (even concurrently) across multiple models.
class RevisedSimplex {
public:
    struct Options {
        // Hard cap on total pivots across both phases. A tiny hand-solved
        // LP should never come close to this; it exists so a bug (or a
        // model this milestone isn't meant to handle) fails as
        // kIterationLimit instead of hanging.
        int max_iterations = 10000;

        // Solver-level tolerance used for feasibility, reduced-cost,
        // ratio-test, and related numerical comparisons.
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