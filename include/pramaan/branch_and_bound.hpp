// branch_and_bound.hpp
// P1 Step 3 — MILP Branch-and-Bound search (first implementation)
//
// A deliberately basic but correct DFS branch-and-bound built on PRAMAAN's
// existing LP layer:
//   - node relaxations are solved by DualSimplex::warmSolve() from the
//     parent's basis where the basis is compatible, and by a cold solve
//     otherwise;
//   - branching is on the first fractional integer variable in model order;
//   - the search is a LIFO stack (no best-bound ordering yet).
//
// Scope actually implemented here: root processing, fractional branching,
// floor/ceil children, incumbent tracking and bound-based pruning, for both
// minimization and maximization. NOT implemented: cutting planes, primal
// heuristics, presolve inside the tree, best-bound/pseudocost selection,
// parallel search, MILP certificates.
#pragma once

#include <vector>

#include "pramaan/node.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {
namespace mip {

enum class MipStatus {
    kOptimal,           // incumbent found and the tree was exhausted -> proven optimal
    kInfeasible,        // no integer-feasible solution exists
    kUnbounded,         // the root LP relaxation is unbounded
    kNodeLimit,         // node limit hit; any incumbent is NOT proven optimal
    kNumericalFailure,  // an LP relaxation was inconclusive; see the note in the .cpp
};

// Search counters. These are the quantities needed to tell whether the
// search actually behaved as intended (did it branch? did it prune? did
// warm-starting apply?), which the tests assert on -- not decoration.
struct MipStatistics {
    int nodes_created = 0;
    int nodes_explored = 0;          // nodes popped and evaluated
    int nodes_pruned_by_bound = 0;   // pruned against the incumbent
    int nodes_pruned_infeasible = 0; // relaxation infeasible or bounds empty
    int branchings = 0;              // nodes that produced two children
    int incumbent_updates = 0;
    int warm_start_attempts = 0;
    int warm_start_successes = 0;    // attempts that were not rejected as incompatible
};

struct MipResult {
    MipStatus status = MipStatus::kNodeLimit;

    // Best integer-feasible solution found, in ORIGINAL model variable order.
    // Empty unless an incumbent was found.
    std::vector<double> x;

    // Objective of `x` in the model's own sense. Meaningful only when `x` is
    // non-empty.
    double objective_value = 0.0;

    MipStatistics statistics{};

    bool hasSolution() const noexcept { return !x.empty(); }
};

class BranchAndBound {
public:
    struct Options {
        // Hard cap on nodes explored, so a bug or a model outside this
        // implementation's scope terminates as kNodeLimit instead of hanging.
        int max_nodes = 100000;

        // |x[j] - round(x[j])| <= integrality_tolerance counts as integral.
        // 1e-6 matches the tolerance the certificate layer uses for the same
        // question.
        double integrality_tolerance = 1e-6;

        // Options forwarded to the LP solvers. Its `tolerance` is also used
        // for incumbent comparison and bound-based pruning.
        RevisedSimplex::Options lp_options{};

        // Warm-start child relaxations from the parent basis via
        // DualSimplex::warmSolve().
        //
        // The underlying dual-simplex pivot formula bug (x_B[leaving]=0
        // instead of theta) has been fixed. Warm-start now produces correct
        // results on the knapsack test and other verification cases.
        // Defaults to false until sufficient production validation.
        bool use_warm_start = false;
    };

    BranchAndBound() = default;
    explicit BranchAndBound(Options options) : options_(options) {}

    // Solves `model` as a MILP. Variables flagged VarType::kInteger are
    // constrained to integral values; VarType::kContinuous variables are
    // not. `model` is never mutated: each node builds its own relaxation.
    //
    // A model with no integer variables is solved by the root LP alone and
    // reported kOptimal, which makes this a safe entry point for either.
    MipResult solve(const ModelIR& model) const;

    const Options& options() const noexcept { return options_; }

private:
    Options options_{};
};

// Exposed for testing and for callers that need the same integrality
// predicate the search uses. Only VarType::kInteger variables are checked.
bool isIntegerFeasible(const ModelIR& model, const std::vector<double>& x, double tolerance);

// Index of the first fractional integer variable in model order, or -1 if
// every integer variable is integral within `tolerance`. This is the
// branching rule this implementation uses.
CSRMatrix::Index firstFractionalInteger(const ModelIR& model, const std::vector<double>& x,
                                        double tolerance);

}  // namespace mip
}  // namespace pramaan
