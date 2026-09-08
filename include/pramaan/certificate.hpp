// certificate.hpp
// PILLAR 4 (Trust) -- Zone 7, shared between pramaan-solve and pramaan-verify
//
// Purpose: the data format for a solution certificate -- primal residual
// (||Ax-b||), dual residual, complementarity residual, objective value,
// and a reference to the transformation ledger for reconstructing the
// original-space solution. THIS FILE is the contract between the solver
// and the verifier -- keep it solver-logic-free, pure data + serialization.
//
// FIRST TASK: define the Certificate struct and a simple text/JSON
// serialization (write to disk, read back) -- no solver logic here at all.
#pragma once
#include <string>

namespace pramaan {

struct Certificate {
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double objective_value = 0.0;
    bool is_valid = false;
    // TODO: serialize()/deserialize() to/from a file
};

}  // namespace pramaan
