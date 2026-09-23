#include <cmath>
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
    if (!std::isfinite(bound)) {
        throw std::invalid_argument("Node::makeChild: branch bound must be finite");
    }

    Node child;
    child.branch_ = BranchDecision{variable, direction, bound};
    child.depth_ = depth_ + 1;
    child.var_lower_ = var_lower_;
    child.var_upper_ = var_upper_;
    child.inherited_basis_ = inherited_basis;
    child.cuts_ = cuts_;
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

    // Apply cuts
    if (!cuts_.empty()) {
        std::vector<CSRMatrix::Index> row_ptr(relaxation.A.rowPtr().begin(), relaxation.A.rowPtr().end());
        std::vector<CSRMatrix::Index> col_idx(relaxation.A.colIdx().begin(), relaxation.A.colIdx().end());
        std::vector<double> values(relaxation.A.values().begin(), relaxation.A.values().end());

        for (const auto& cut : cuts_) {
            for (std::size_t i = 0; i < cut.cols.size(); ++i) {
                col_idx.push_back(cut.cols[i]);
                values.push_back(cut.vals[i]);
            }
            row_ptr.push_back(static_cast<CSRMatrix::Index>(col_idx.size()));

            relaxation.row_lower.push_back(-kInfinity);
            relaxation.row_upper.push_back(cut.rhs);
            relaxation.row_names.push_back("cut_" + std::to_string(relaxation.row_names.size()));
        }

        relaxation.A = CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), relaxation.numVars());
    }

    return relaxation;
}

void Node::setRelaxation(SolveResult result, BasisState optimal_basis) {
    switch (result.status) {
        case SolveStatus::kOptimal:          state_ = NodeState::kOptimal; break;
        case SolveStatus::kInfeasible:       state_ = NodeState::kInfeasible; break;
        case SolveStatus::kUnbounded:        state_ = NodeState::kUnbounded; break;
        case SolveStatus::kIterationLimit:
        case SolveStatus::kNumericalFailure:
        case SolveStatus::kWarmStartRejected: state_ = NodeState::kFailed; break;
    }
    relaxation_ = std::move(result);
    optimal_basis_ = std::move(optimal_basis);
}

void Node::markInfeasible() {
    state_ = NodeState::kInfeasible;
    relaxation_ = SolveResult{};
    relaxation_.status = SolveStatus::kInfeasible;
}

void Node::addCut(Cut cut, int num_vars) {
    validateCut(cut, num_vars);
    cuts_.push_back(std::move(cut));
}

}  // namespace mip
}  // namespace pramaan
