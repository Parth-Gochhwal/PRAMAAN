// transformation_ledger.hpp
// PILLAR 4 (Trust) -- supports Zone 7 Certificate Generator
//
// Purpose: append-only log of every reversible presolve step. This is what
// makes postsolve (mapping the presolved solution back to the ORIGINAL
// model) and independent verification possible at all.
//
// FIRST TASK: define a simple tagged-union / variant of "reduction records"
// (e.g. FixedVariableRemoved{var_id, value}, RowScaled{row_id, factor}) and
// a std::vector<Reduction> that presolve.cpp appends to in order. Postsolve
// just replays this vector in REVERSE.
#pragma once
#include <vector>

namespace pramaan {

class TransformationLedger {
    // TODO
};

}  // namespace pramaan
