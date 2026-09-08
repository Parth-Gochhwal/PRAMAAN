// simplex.hpp
// PILLAR 2 (Speed) -- Zone 4 "Heterogeneous Continuous Engine", CPU half
//
// Purpose: revised/dual simplex solver for LP. This is your first REAL
// solver algorithm -- everything before this file was plumbing.
//
// FIRST TASK: do NOT start with a general sparse revised simplex. Start with
// a dense, textbook, tableau-based simplex that only needs to solve a
// 2-3 variable LP you write by hand and verify on paper. Get that working
// end to end (including reading the answer back out correctly) before
// touching sparsity, the revised (basis-inverse) formulation, or Netlib
// instances. This mirrors MIT 15.053's lecture examples on purpose --
// use one of those as your first hand-checkable test case.
#pragma once
#include "pramaan/ir.hpp"

namespace pramaan {

struct SolveResult {
    // TODO: solution vector, objective value, status (optimal/infeasible/
    // unbounded), basis info (needed later for warm-start)
};

class RevisedSimplex {
    // TODO
};

}  // namespace pramaan
