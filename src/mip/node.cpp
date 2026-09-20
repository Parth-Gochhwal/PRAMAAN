// node.cpp -- see src/mip/node.hpp
#include "pramaan/node.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace pramaan {
namespace mip {

Node Node::makeRoot(const ModelIR& model) {
    model.validate();

    Node node;
    node.branch_ = BranchDecision{};  // variable == -1 marks the root
    node.depth_ = 0;
    node.var_lower_ = model.var_lower;
    node.var_upper_ = model.var_upper;
    node.inherited_basis_ = BasisState{};
    node.state_ = NodeState::kUnevaluated;
    return node;
}

Node Node::makeChild(CSRMatrix::Index variable,
                     BranchDirection direction,
                     double bound,
                     const BasisState& inherited_basis) const {
    if (variable < 0 || static_cast<std::size_t>(variable) >= var_lower_.size()) {
        throw std::invalid_argument("Node::makeChild: branch variable index out of range");
    }

    Node child;
    child.branch_ = BranchDecision{variable, direction, bound};
    child.depth_ = depth_ + 1;
    child.var_lower_ = var_lower_;
    child.var_upper_ = var_upper_;
    child.inherited_basis_ = inherited_basis;
    child.state_ = NodeState::kUnevaluated;

    const auto j = static_cast<std::size_t>(variable);
    if (direction == BranchDirection::kDown) {
        // x_j <= bound, never loosening the parent's existing upper bound.
        child.var_upper_[j] = std::min(child.var_upper_[j], bound);
    } else {
        // x_j >= bound, never loosening the parent's existing lower bound.
        child.var_lower_[j] = std::max(child.var_lower_[j], bound);
    }
    return child;
}

bool Node::hasInconsistentBounds(double tolerance) const {
    for (std::size_t j = 0; j < var_lower_.size(); ++j) {
        if (var_lower_[j] > var_upper_[j] + tolerance) return true;
    }
    return false;
}

ModelIR Node::buildRelaxation(const ModelIR& original) const {
    if (var_lower_.size() != static_cast<std::size_t>(original.numVars()) ||
        var_upper_.size() != static_cast<std::size_t>(original.numVars())) {
        throw std::invalid_argument(
            "Node::buildRelaxation: node bounds do not match the original model's variable count");
    }

    ModelIR relaxation = original;
    relaxation.var_lower = var_lower_;
    relaxation.var_upper = var_upper_;
    return relaxation;
}

void Node::setRelaxation(SolveResult result) {
    switch (result.status) {
        case SolveStatus::kOptimal:          state_ = NodeState::kOptimal; break;
        case SolveStatus::kInfeasible:       state_ = NodeState::kInfeasible; break;
        case SolveStatus::kUnbounded:        state_ = NodeState::kUnbounded; break;
        case SolveStatus::kIterationLimit:
        case SolveStatus::kNumericalFailure: state_ = NodeState::kFailed; break;
    }
    relaxation_ = std::move(result);
}

void Node::markInfeasible() {
    state_ = NodeState::kInfeasible;
    relaxation_ = SolveResult{};
    relaxation_.status = SolveStatus::kInfeasible;
}

}  // namespace mip
}  // namespace pramaan
