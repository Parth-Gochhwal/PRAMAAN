// node.hpp
// P1 Step 2 — Branch-and-Bound node representation
//
// A Node is one subproblem in the B&B search tree. It records:
//   1. the single bound change that distinguishes it from its parent,
//   2. the accumulated effective variable bounds for its own LP relaxation,
//   3. the parent's optimal BasisState, for Dual Simplex warm-starting,
//   4. the LP relaxation result once evaluated, plus explicit node state.
//
// This type deliberately holds no solver logic: it is the data the B&B
// engine (branch_and_bound.hpp) pushes/pops and inspects. Keeping search
// state out of the LP layer preserves the spec's separation between LP
// solving and MILP search.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_set>

#include "pramaan/dual_simplex.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {
namespace mip {

// Which side of a fractional LP value a child node takes.
//   kDown: x_j <= bound   (bound == floor(lp_value))
//   kUp:   x_j >= bound   (bound == ceil(lp_value))
enum class BranchDirection { kDown, kUp };

// Lifecycle of a node's LP relaxation. A node is created kUnevaluated and
// moves to exactly one terminal state when its relaxation is solved.
//
// Integer-feasibility is NOT a state here: it is a property of the
// relaxation's solution vector, derived on demand by the B&B engine, so it
// cannot fall out of sync with `relaxation()`.
enum class NodeState {
    kUnevaluated,   // created, LP relaxation not yet solved
    kOptimal,       // relaxation solved to optimality
    kInfeasible,    // relaxation infeasible (or bounds are inconsistent by construction)
    kUnbounded,     // relaxation unbounded
    kFailed,        // iteration limit or numerical failure -- result is inconclusive
};

// The bound change this node applies relative to its parent.
// `variable < 0` marks the root, which applies no change.

struct Cut {
    std::vector<CSRMatrix::Index> cols;
    std::vector<double> vals;
    double rhs = 0.0;
};

// Validates a cut for structural correctness. Throws std::invalid_argument if:
//   - cols and vals have different sizes
//   - any column index is out of range [0, num_vars)
//   - any coefficient is not finite or is zero
//   - rhs is not finite
//   - there are duplicate column indices
// num_vars must be the number of variables in the model.
inline void validateCut(const Cut& cut, int num_vars) {
    if (cut.cols.size() != cut.vals.size()) {
        throw std::invalid_argument("validateCut: cols and vals size mismatch");
    }
    if (!std::isfinite(cut.rhs)) {
        throw std::invalid_argument("validateCut: rhs is not finite");
    }
    std::unordered_set<CSRMatrix::Index> seen;
    for (std::size_t i = 0; i < cut.cols.size(); ++i) {
        const CSRMatrix::Index c = cut.cols[i];
        if (c < 0 || c >= static_cast<CSRMatrix::Index>(num_vars)) {
            throw std::invalid_argument(
                "validateCut: column index " + std::to_string(c) +
                " out of range [0, " + std::to_string(num_vars) + ")");
        }
        if (!std::isfinite(cut.vals[i]) || cut.vals[i] == 0.0) {
            throw std::invalid_argument("validateCut: coefficient must be finite and nonzero");
        }
        if (!seen.insert(c).second) {
            throw std::invalid_argument(
                "validateCut: duplicate column index " + std::to_string(c));
        }
    }
}

// Returns the violation of the cut (sum(a_j x_j) - rhs). Positive means violated.
// Bounds-checks every column index against x.size().
inline double cutViolation(const Cut& cut, const std::vector<double>& x) {
    double lhs = 0.0;
    for (std::size_t i = 0; i < cut.cols.size(); ++i) {
        const std::size_t idx = static_cast<std::size_t>(cut.cols[i]);
        if (idx >= x.size()) {
            throw std::out_of_range(
                "cutViolation: column index " + std::to_string(cut.cols[i]) +
                " >= x.size() " + std::to_string(x.size()));
        }
        lhs += cut.vals[i] * x[idx];
    }
    return lhs - cut.rhs;
}

struct BranchDecision {
    CSRMatrix::Index variable = -1;
    BranchDirection direction = BranchDirection::kDown;
    double bound = 0.0;

    bool isRoot() const noexcept { return variable < 0; }
};

class Node {
public:
    // Root node: effective bounds are the original model's own bounds, and
    // there is no inherited basis (the root LP is always cold-solved).
    static Node makeRoot(const ModelIR& model);

    // Derives a child that additionally constrains `variable`:
    //   kDown -> var_upper[variable] = min(parent_upper, bound)
    //   kUp   -> var_lower[variable] = max(parent_lower, bound)
    //
    // The child starts from THIS node's accumulated bounds, so the child's
    // effective bounds are exactly "parent bounds + this one change".
    // `inherited_basis` should be the basis captured from the FINAL post-cut
    // solve of this node (i.e. after all cuts have been incorporated). Pass
    // an empty BasisState to force the child to cold-solve.
    Node makeChild(CSRMatrix::Index variable,
                   BranchDirection direction,
                   double bound,
                   const BasisState& inherited_basis) const;

    // --- Structure -------------------------------------------------------

    const BranchDecision& branch() const noexcept { return branch_; }
    int depth() const noexcept { return depth_; }

    // Accumulated effective bounds for this subproblem, indexed by original
    // model variable order.
    const std::vector<double>& varLower() const noexcept { return var_lower_; }
    const std::vector<double>& varUpper() const noexcept { return var_upper_; }

    // True when a branch produced an empty box (lower > upper) for some
    // variable. Such a node is infeasible by construction and must NOT be
    // handed to the LP solver: ModelIR::validate() throws on inverted
    // bounds, so the B&B engine checks this first and marks the node
    // infeasible directly.
    bool hasInconsistentBounds(double tolerance) const;

    // Builds this node's LP relaxation: `original` with this node's
    // accumulated variable bounds substituted in.
    //
    // var_types are copied through unchanged. PRAMAAN's LP layer
    // (RevisedSimplex / DualSimplex) does not read var_types at all, so the
    // returned model IS the continuous relaxation; preserving the flags
    // keeps the integrality test in one place (the B&B engine) rather than
    // splitting it across a mutated copy.
    ModelIR buildRelaxation(const ModelIR& original) const;

    // --- Warm-start basis ------------------------------------------------

    const BasisState& inheritedBasis() const noexcept { return inherited_basis_; }
    bool hasInheritedBasis() const noexcept { return !inherited_basis_.empty(); }
    void clearInheritedBasis() noexcept { inherited_basis_ = BasisState{}; }

    // --- Evaluation ------------------------------------------------------

    NodeState state() const noexcept { return state_; }
    bool isEvaluated() const noexcept { return state_ != NodeState::kUnevaluated; }

    // Records the relaxation result and derives `state_` from its status.
    // Note: kWarmStartRejected should be handled at the B&B layer (by
    // falling back to a cold solve) before calling setRelaxation, so
    // receiving it here maps to kFailed as a safety net.
    void setRelaxation(SolveResult result, BasisState optimal_basis = BasisState{});

    // Marks a node infeasible without solving (used for inconsistent bounds).
    void markInfeasible();

    const SolveResult& relaxation() const noexcept { return relaxation_; }
    const BasisState& optimalBasis() const noexcept { return optimal_basis_; }

    // LP relaxation objective, in the model's own objective sense. Only
    // meaningful when state() == kOptimal; the B&B engine checks state
    // before using this as a bound.
    double relaxationObjective() const noexcept { return relaxation_.objective_value; }

    // Adds a cut to this node after validating its structure.
    // Invariants enforced:
    //   - cols.size() == vals.size()
    //   - all column indices valid for `num_vars`
    //   - all coefficients finite and nonzero
    //   - rhs finite
    //   - no duplicate columns
    // Throws std::invalid_argument if any invariant is violated.
    void addCut(Cut cut, int num_vars);

    const std::vector<Cut>& cuts() const noexcept { return cuts_; }

private:
    std::vector<Cut> cuts_;

    Node() = default;

    BranchDecision branch_{};
    int depth_ = 0;
    std::vector<double> var_lower_;
    std::vector<double> var_upper_;
    BasisState inherited_basis_{};
    NodeState state_ = NodeState::kUnevaluated;
    SolveResult relaxation_{};
    BasisState optimal_basis_{};
};

}  // namespace mip
}  // namespace pramaan
