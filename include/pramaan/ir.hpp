// ir.hpp
// PILLAR 1 (Structure) -- Zone 1/2 of the architecture diagram
//
// Purpose: PRAMAAN-IR, the canonical in-memory model representation every
// parsed input gets converted into. Every other component (presolve,
// structural fingerprint, solvers) reads/writes THIS type, never the raw
// .mps file directly.
//
// FIRST TASK: define the struct/class holding:
//   - objective coefficients (dense vector, one per variable)
//   - constraint matrix A (a CSRMatrix, see sparse_matrix.hpp)
//   - row bounds (lhs <= Ax <= rhs form, or separate senses -- your choice)
//   - variable bounds (lower/upper per variable)
//   - variable types (continuous / integer, for LP vs MILP)
//   - a name/id per variable and constraint (for debugging + certificates)
#pragma once
#include "pramaan/sparse_matrix.hpp"

namespace pramaan {

struct ModelIR {
    // TODO
};

}  // namespace pramaan
