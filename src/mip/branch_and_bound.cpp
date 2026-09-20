// branch_and_bound.cpp -- see src/mip/branch_and_bound.hpp
#include "pramaan/branch_and_bound.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pramaan/dual_simplex.hpp"

namespace pramaan {
namespace mip {

namespace {

using Index = CSRMatrix::Index;

// One node's LP outcome: the relaxation result plus a basis for its children
// (empty when no safely reusable basis was produced).
struct Relaxation {
    SolveResult result;
    BasisState basis;
    bool warm_attempted = false;
    bool warm_succeeded = false;
};

// Solves a node's relaxation, warm-starting from the inherited basis where
// possible.
//
// DualSimplex::warmSolve() throws std::invalid_argument when the inherited
// basis is structurally incompatible with the modified model. In B&B this is
// an expected, routine event rather than an error: the basis fingerprint
// encodes which variable bounds are finite vs. infinite, so the FIRST branch
// on a variable with an infinite bound necessarily changes it and is
// rejected. Catching it and cold-solving keeps the search correct; the final
// answer never depends on whether a warm start was taken.
Relaxation solveRelaxation(const ModelIR& node_model, const Node& node,
                           const RevisedSimplex::Options& lp_options, bool use_warm_start) {
    const DualSimplex dual(lp_options);
    Relaxation out;

    if (use_warm_start && node.hasInheritedBasis()) {
        out.warm_attempted = true;
        try {
            out.result = dual.warmSolve(node_model, node.inheritedBasis(), &out.basis);
            out.warm_succeeded = true;
            return out;
        } catch (const std::invalid_argument&) {
            // Incompatible basis -- fall through to a cold solve.
            out.basis = BasisState{};
        }
    }

    // Cold path. captureBasis() performs the cold solve and reports both the
    // result and a reusable basis, so a cold node costs one LP solve, not two.
    out.basis = dual.captureBasis(node_model, &out.result);
    return out;
}

}  // namespace

bool isIntegerFeasible(const ModelIR& model, const std::vector<double>& x, double tolerance) {
    return firstFractionalInteger(model, x, tolerance) < 0;
}

Index firstFractionalInteger(const ModelIR& model, const std::vector<double>& x, double tolerance) {
    if (x.size() != static_cast<std::size_t>(model.numVars())) return -1;
    for (Index j = 0; j < model.numVars(); ++j) {
        if (!model.isInteger(j)) continue;
        const double v = x[static_cast<std::size_t>(j)];
        if (std::abs(v - std::round(v)) > tolerance) return j;
    }
    return -1;
}

MipResult BranchAndBound::solve(const ModelIR& model) const {
    model.validate();  // house convention: solver entry points validate first

    MipResult out;
    const double tol = options_.lp_options.tolerance;
    const double int_tol = options_.integrality_tolerance;
    const bool minimize = (model.obj_sense == ObjSense::kMinimize);

    // Strict improvement over the incumbent, in the model's own sense.
    const auto improves = [&](double candidate, double incumbent) {
        return minimize ? (candidate < incumbent - tol) : (candidate > incumbent + tol);
    };
    // A node whose relaxation bound cannot strictly beat the incumbent can be
    // discarded: for minimization the LP value is a lower bound on anything
    // in that subtree, for maximization an upper bound.
    const auto cannotImprove = [&](double bound, double incumbent) {
        return minimize ? (bound >= incumbent - tol) : (bound <= incumbent + tol);
    };

    bool have_incumbent = false;
    double incumbent_obj = 0.0;

    std::vector<Node> stack;
    Node root = Node::makeRoot(model);
    ++out.statistics.nodes_created;
    stack.push_back(std::move(root));

    while (!stack.empty()) {
        if (out.statistics.nodes_explored >= options_.max_nodes) {
            out.status = MipStatus::kNodeLimit;
            return out;
        }

        Node node = std::move(stack.back());
        stack.pop_back();

        // A branch can produce an empty box (e.g. x <= 2 on a variable already
        // bounded below by 3). ModelIR::validate() throws on inverted bounds,
        // so this is detected here rather than handed to the LP solver.
        if (node.hasInconsistentBounds(tol)) {
            node.markInfeasible();
            ++out.statistics.nodes_explored;
            ++out.statistics.nodes_pruned_infeasible;
            continue;
        }

        const ModelIR node_model = node.buildRelaxation(model);
        Relaxation rel =
            solveRelaxation(node_model, node, options_.lp_options, options_.use_warm_start);
        node.setRelaxation(std::move(rel.result));
        ++out.statistics.nodes_explored;
        if (rel.warm_attempted) ++out.statistics.warm_start_attempts;
        if (rel.warm_succeeded) ++out.statistics.warm_start_successes;

        const bool is_root = node.branch().isRoot();

        switch (node.state()) {
            case NodeState::kInfeasible:
                if (is_root) {
                    out.status = MipStatus::kInfeasible;
                    return out;
                }
                ++out.statistics.nodes_pruned_infeasible;
                continue;

            case NodeState::kUnbounded:
                // Only reachable at the root: every child is strictly more
                // constrained than its parent, so a bounded root cannot have
                // an unbounded descendant.
                out.status = MipStatus::kUnbounded;
                return out;

            case NodeState::kFailed:
                // The relaxation is inconclusive, so this subtree's bound is
                // unknown. Dropping it could silently discard the true
                // optimum, so the whole solve is reported as failed rather
                // than returning an unproven answer.
                out.status = MipStatus::kNumericalFailure;
                return out;

            case NodeState::kOptimal:
                break;

            case NodeState::kUnevaluated:
                out.status = MipStatus::kNumericalFailure;
                return out;
        }

        const double node_bound = node.relaxationObjective();
        if (have_incumbent && cannotImprove(node_bound, incumbent_obj)) {
            ++out.statistics.nodes_pruned_by_bound;
            continue;
        }

        const std::vector<double>& x = node.relaxation().x;
        const Index branch_var = firstFractionalInteger(model, x, int_tol);

        if (branch_var < 0) {
            // Integer-feasible: a candidate incumbent, not a node to branch.
            if (!have_incumbent || improves(node_bound, incumbent_obj)) {
                have_incumbent = true;
                incumbent_obj = node_bound;
                out.x = x;
                out.objective_value = node_bound;
                ++out.statistics.incumbent_updates;
            }
            continue;
        }

        const double v = x[static_cast<std::size_t>(branch_var)];
        const double floor_v = std::floor(v);
        const double ceil_v = std::ceil(v);

        // Child push order is fixed for determinism: the UP child is pushed
        // first so the DOWN child (x_j <= floor(v)) is popped first, giving a
        // deterministic depth-first dive down the floor branch.
        Node up_child = node.makeChild(branch_var, BranchDirection::kUp, ceil_v, rel.basis);
        Node down_child = node.makeChild(branch_var, BranchDirection::kDown, floor_v, rel.basis);
        out.statistics.nodes_created += 2;
        ++out.statistics.branchings;

        stack.push_back(std::move(up_child));
        stack.push_back(std::move(down_child));
    }

    out.status = have_incumbent ? MipStatus::kOptimal : MipStatus::kInfeasible;
    return out;
}

}  // namespace mip
}  // namespace pramaan
