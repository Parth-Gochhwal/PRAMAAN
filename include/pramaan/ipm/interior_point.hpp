// interior_point.hpp
// PILLAR 2 (Speed) -- Interior-point method for LP.
//
// Dense textbook primal-dual path-following interior-point solver.
// Complements the RevisedSimplex as the second required solver method.
// LP only, continuous variables only, dense linear algebra.
#pragma once

#include <vector>
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"  // for SolveResult, SolveStatus

namespace pramaan {

class InteriorPointSolver {
public:
    struct Options {
        int max_iterations = 100;
        double tolerance = 1e-8;
        double barrier_reduction = 0.2;  // mu *= barrier_reduction each outer iter
        double initial_bound_slack = 1.0; // initial distance from variable bounds
    };

    InteriorPointSolver() = default;
    explicit InteriorPointSolver(Options options) : options_(options) {}

    SolveResult solve(const ModelIR& model) const;

    const Options& options() const noexcept { return options_; }

private:
    Options options_{};
};

}  // namespace pramaan
