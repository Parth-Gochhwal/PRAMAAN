// branch_and_bound.hpp
// PRAMAAN MILP Branch-and-Bound search
//
// Implements a parallel, multi-strategy branch-and-bound MILP solver built on
// PRAMAAN's existing LP layer. Currently supported features:
//   - Node relaxations solved by DualSimplex::warmSolve() (warm start from
//     parent basis when enabled) or by a cold solve otherwise.
//   - Branching on the first fractional integer variable in model order.
//   - Two node-selection strategies: depth-first (LIFO stack) and
//     best-bound (priority queue ordered by LP relaxation bound).
//   - Parallel search via a shared work queue and configurable worker threads.
//   - Knapsack cover cuts generated at each node and added to the LP
//     relaxation before branching (only for pure binary rows; see
//     generateCoverCuts() documentation for validity criteria).
//   - A basic rounding heuristic to find incumbents faster.
//   - Node-limit termination: returns kNodeLimit when the limit is reached
//     before the tree is exhausted; kOptimal only when proof is complete.
//
// NOT implemented: Gomory/MIR cuts, primal heuristics beyond simple rounding,
// presolve inside the tree, pseudocost branching, MILP optimality certificates.
#pragma once

#include <vector>

#include "pramaan/node.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {
namespace mip {

enum class MipStatus {
    kOptimal,          // incumbent found AND tree exhausted/pruned (proven optimal)
    kInfeasible,       // complete tree exhausted with no integer-feasible solution
    kNodeLimit,        // search stopped because node limit was reached before proof;
                       // any incumbent found is NOT proven optimal
    kNumericalFailure, // an LP relaxation failed inconclusively (iteration limit,
                       // numerical instability, or an unbounded LP relaxation whose
                       // MILP status could not be determined); result is inconclusive
};

// Search statistics. These document exactly what each counter measures so that
// test assertions on them are unambiguous.
struct MipStatistics {
    int nodes_created = 0;
    int nodes_explored = 0;          // nodes popped from the queue and evaluated
    int nodes_pruned_by_bound = 0;   // nodes pruned because LP bound >= incumbent
    int nodes_pruned_infeasible = 0; // nodes pruned because relaxation was infeasible
                                     // or bounds were inconsistent by construction
    int branchings = 0;              // nodes whose relaxation was fractional, leading
                                     // to two child nodes being pushed

    int incumbent_updates = 0;       // times the best known integer-feasible solution
                                     // was improved
    int warm_start_attempts = 0;     // times DualSimplex::warmSolve() was called
    int warm_start_successes = 0;    // times warmSolve() returned kOptimal

    // Cut counters — semantics:
    //   cuts_generated:   non-duplicate cuts whose violation check was attempted
    //   cuts_applied:     cuts whose violation was > threshold and were inserted
    //                     into the node relaxation (cuts_applied <= cuts_generated)
    //   cuts_rejected:    cuts whose violation was <= threshold and were discarded
    //                     (cuts_rejected = cuts_generated - cuts_applied)
    //   duplicate_cuts:   cuts identical to an already-inserted cut; these are
    //                     NOT counted in cuts_generated, cuts_applied, or cuts_rejected
    int cuts_generated = 0;
    int cuts_applied = 0;
    int cuts_rejected = 0;
    int duplicate_cuts = 0;
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
        // IMPORTANT: after a cut is added to a node, the inherited basis is
        // invalidated and the augmented LP is cold-solved. Only the final
        // post-cut-round basis is forwarded to children.
        bool use_warm_start = false;

        // Number of worker threads for parallel branch-and-bound.
        // 1 = single-threaded sequential execution.
        int num_threads = 1;

        enum class NodeSelection {
            kDepthFirst,
            kBestBound
        };
        // Node selection strategy.
        // kDepthFirst: LIFO stack (good for quickly finding incumbents).
        // kBestBound:  priority queue ordered by LP relaxation bound
        //              (good for tight dual bounds and early pruning).
        //
        // Queued children inherit the strongest currently available valid
        // bound from their parent until their own relaxation is evaluated.
        // This is a valid lower (minimization) / upper (maximization) bound
        // because the child's feasible region is a subset of the parent's.
        NodeSelection node_selection = NodeSelection::kDepthFirst;
    };

    BranchAndBound() = default;
    explicit BranchAndBound(Options options) : options_(options) {}

    // Solves `model` as a MILP. Variables flagged VarType::kInteger are
    // constrained to integral values; VarType::kContinuous variables are
    // not. `model` is never mutated: each node builds its own relaxation.
    //
    // A model with no integer variables is solved by the root LP alone and
    // reported kOptimal, which makes this a safe entry point for either.
    //
    // Status guarantees:
    //   kOptimal:         incumbent exists; tree exhausted; no node limit hit;
    //                     no inconclusive LP failures.
    //   kInfeasible:      tree exhausted with no incumbent.
    //   kNodeLimit:       search stopped at node limit; kOptimal was NOT returned.
    //   kNumericalFailure: an LP relaxation was inconclusive (includes the case
    //                     where the LP relaxation was unbounded -- an unbounded LP
    //                     relaxation does NOT by itself prove MILP unboundedness).
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
