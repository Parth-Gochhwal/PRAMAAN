// fingerprint.hpp
// PILLAR 1 (Structure) -- Structural Fingerprint Engine
//
// Purpose: Decompose a constraint matrix into independent blocks and identify
// coupling constraints (rows whose nonzero pattern links otherwise-separate
// variable groups).  This drives block-angular decomposition, parallel
// sub-problem dispatch, and diagnostic reporting.
#pragma once
#include <vector>

#include "pramaan/sparse_matrix.hpp"

namespace pramaan {
namespace structure {

using Index = CSRMatrix::Index;

// One connected component (block) of the bipartite variable/constraint graph.
struct Block {
    std::vector<Index> variables;   // sorted
    std::vector<Index> constraints; // sorted
};

// Result of structural analysis.
struct StructuralFingerprint {
    // Independent blocks of the matrix after all coupling constraints
    // are conceptually removed. Each block is a disjoint connected component
    // of variables and the strictly internal constraints that bind them.
    // Sorted by first variable index for determinism.
    std::vector<Block> blocks;

    // Constraints (rows) whose removal strictly increases the number of
    // independent variable components (where a valid component must contain
    // at least one internal constraint). These are constraint-node articulation points
    // between otherwise independent macroscopic blocks.
    std::vector<Index> coupling_constraints;

    int numBlocks() const { return static_cast<int>(blocks.size()); }
};

// Build a bipartite graph from the sparsity pattern of A and find
// the block-angular decomposition.
//
// Algorithm (O(M+N+NNZ)):
// 1. A constraint is a "coupling constraint" if its removal separates
//    the bipartite graph into more components, excluding trivial components
//    that have no constraints (i.e. isolating a degree-1 variable).
//    This is equivalent to finding constraint-node articulation points in the bipartite graph
//    restricted to variables of degree >= 2.
// 2. Variables and non-coupling constraints form disjoint independent blocks
//    (connected components).
// 3. Blocks are sorted by their first variable index for deterministic output.
//
// Edge cases:
//   - 0 rows + N columns => N singleton variable blocks
//   - 0 columns + M rows => zero blocks
//   - 0 rows + 0 columns => empty fingerprint
//   - isolated variables => singleton blocks
//   - empty rows => ignored
//   - one-variable rows => treated as internal to whatever block the variable is in.
StructuralFingerprint analyze(const CSRMatrix& A);

}  // namespace structure
}  // namespace pramaan
