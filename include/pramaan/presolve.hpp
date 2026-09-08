// presolve.hpp
// PILLAR 1 (Structure) -- Zone 2 "Reversible Presolve"
//
// Purpose: bound tightening, singleton-variable elimination, Ruiz scaling.
// CRITICAL: every reduction applied here must be logged to the
// TransformationLedger (see transformation_ledger.hpp) so the solution can
// be mapped back to original-variable space and independently verified.
//
// FIRST TASK: start with the SIMPLEST possible reduction --
//   "remove a variable whose lower bound == upper bound (a fixed variable)"
// and log it to the ledger. Get that one reduction round-tripping correctly
// (presolve -> solve -> postsolve back to original space) before adding
// any other reduction. This is the hardest correctness bug source in the
// whole codebase if done out of order.
#pragma once
#include "pramaan/ir.hpp"

namespace pramaan {

class Presolver {
    // TODO
};

}  // namespace pramaan
