// sparse_matrix.hpp
// PILLAR: foundation (used by every other component)
//
// Purpose: CSR/CSC sparse matrix storage. Every constraint matrix in PRAMAAN
// flows through this type -- never store a dense matrix anywhere.
//
// FIRST TASK: implement a minimal CSRMatrix class with:
//   - construction from (row_ptr, col_idx, values) triplet arrays
//   - a matrix-vector multiply (Ax) -- this is the single most-used operation
//     in the entire codebase (every simplex iteration, every PDHG iteration)
//   - a way to iterate non-zeros of a given row/column
//
// Test this file FIRST, in isolation, before writing any solver code.
#pragma once
#include <vector>

namespace pramaan {

class CSRMatrix {
    // TODO: row_ptr_, col_idx_, values_, num_rows_, num_cols_
};

}  // namespace pramaan
