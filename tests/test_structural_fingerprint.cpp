// test_structural_fingerprint.cpp
//
// Tests for pramaan::structure::analyze() -- the structural fingerprint engine.
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/structure/fingerprint.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::VarType;
using pramaan::structure::Block;
using pramaan::structure::StructuralFingerprint;
using pramaan::structure::analyze;

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense,
                     CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0;
             c < static_cast<CSRMatrix::Index>(row.size()); ++c) {
            if (row[static_cast<std::size_t>(c)] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[static_cast<std::size_t>(c)]);
            }
        }
        row_ptr.push_back(static_cast<CSRMatrix::Index>(col_idx.size()));
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values),
                     num_cols);
}

// =========================================================================
// Test 1: Two independent blocks, no coupling.
// Block A: vars {0,1}, rows {0,1}
// Block B: vars {2,3}, rows {2}
//
// Row 0: x0 + x1 <= 10
// Row 1: x0      <= 5
// Row 2: x2 + x3 <= 8
// =========================================================================
void testTwoIndependentBlocks() {
    std::cout << "testTwoIndependentBlocks...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0},  // row 0: x0, x1
        {1.0, 0.0, 0.0, 0.0},  // row 1: x0
        {0.0, 0.0, 1.0, 1.0},  // row 2: x2, x3
    }, 4);

    auto fp = analyze(A);
    check(fp.numBlocks() == 2, "Two blocks detected");
    check(fp.coupling_constraints.empty(), "No coupling constraints");

    // Block 0: vars {0,1}, constraints {0,1}
    check(fp.blocks[0].variables ==
              std::vector<CSRMatrix::Index>{0, 1},
          "Block 0 variables");
    check(fp.blocks[0].constraints ==
              std::vector<CSRMatrix::Index>{0, 1},
          "Block 0 constraints");

    // Block 1: vars {2,3}, constraints {2}
    check(fp.blocks[1].variables ==
              std::vector<CSRMatrix::Index>{2, 3},
          "Block 1 variables");
    check(fp.blocks[1].constraints ==
              std::vector<CSRMatrix::Index>{2},
          "Block 1 constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 2: Two blocks with a coupling constraint.
// Row 0: x0 + x1 <= 10  (block A)
// Row 1: x2 + x3 <= 8   (block B)
// Row 2: x1 + x2 <= 6   (couples blocks A and B)
// =========================================================================
void testCouplingConstraint() {
    std::cout << "testCouplingConstraint...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0},  // row 0: x0, x1
        {0.0, 0.0, 1.0, 1.0},  // row 1: x2, x3
        {0.0, 1.0, 1.0, 0.0},  // row 2: x1, x2 (coupling)
    }, 4);

    auto fp = analyze(A);
    // Blocks remain separate when coupling constraint is correctly identified.
    check(fp.numBlocks() == 2, "Two blocks");
    check(fp.coupling_constraints.size() == 1, "One coupling constraint");
    check(fp.coupling_constraints[0] == 2, "Row 2 is the coupling constraint");

    check(fp.blocks[0].variables == std::vector<CSRMatrix::Index>{0, 1}, "Block 0 vars");
    check(fp.blocks[0].constraints == std::vector<CSRMatrix::Index>{0}, "Block 0 constraints");

    check(fp.blocks[1].variables == std::vector<CSRMatrix::Index>{2, 3}, "Block 1 vars");
    check(fp.blocks[1].constraints == std::vector<CSRMatrix::Index>{1}, "Block 1 constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 3: Fully connected model (single block, no coupling).
// =========================================================================
void testFullyConnected() {
    std::cout << "testFullyConnected...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
    }, 3);

    auto fp = analyze(A);
    check(fp.numBlocks() == 1, "One block");
    check(fp.coupling_constraints.empty(), "No coupling constraints");
    check(fp.blocks[0].variables.size() == 3, "All 3 vars");
    check(fp.blocks[0].constraints.size() == 2, "Both constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 4: Empty matrix.
// =========================================================================
void testEmptyMatrix() {
    std::cout << "testEmptyMatrix...\n";
    CSRMatrix A;  // default: 0 rows, 0 cols
    auto fp = analyze(A);
    check(fp.numBlocks() == 0, "No blocks");
    check(fp.coupling_constraints.empty(), "No coupling");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 5: Isolated variables (columns with no nonzeros).
// =========================================================================
void testIsolatedVariables() {
    std::cout << "testIsolatedVariables...\n";
    // 3 vars, row 0 only touches x0.  x1 and x2 are isolated.
    CSRMatrix A = denseToCSR({
        {1.0, 0.0, 0.0},
    }, 3);

    auto fp = analyze(A);
    check(fp.numBlocks() == 3,
          "Three blocks (one per isolated var)");
    check(fp.coupling_constraints.empty(), "No coupling");
    // Block ordering: by first variable index.
    check(fp.blocks[0].variables ==
              std::vector<CSRMatrix::Index>{0},
          "Block 0: var 0");
    check(fp.blocks[0].constraints ==
              std::vector<CSRMatrix::Index>{0},
          "Block 0: row 0");
    check(fp.blocks[1].variables ==
              std::vector<CSRMatrix::Index>{1},
          "Block 1: var 1");
    check(fp.blocks[1].constraints.empty(), "Block 1: no constraints");
    check(fp.blocks[2].variables ==
              std::vector<CSRMatrix::Index>{2},
          "Block 2: var 2");
    check(fp.blocks[2].constraints.empty(), "Block 2: no constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 6: Zero rows, nonzero cols.
// =========================================================================
void testZeroRows() {
    std::cout << "testZeroRows...\n";
    // 0 rows, 3 cols.
    CSRMatrix A = denseToCSR({}, 3);
    auto fp = analyze(A);
    check(fp.numBlocks() == 3, "Three singleton blocks");
    check(fp.coupling_constraints.empty(), "No coupling");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 7: Deterministic result -- run the same analysis twice.
// =========================================================================
void testDeterminism() {
    std::cout << "testDeterminism...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 1.0},
    }, 4);

    auto fp1 = analyze(A);
    auto fp2 = analyze(A);
    check(fp1.numBlocks() == fp2.numBlocks(), "Same block count");
    for (int b = 0; b < fp1.numBlocks(); ++b) {
        check(fp1.blocks[static_cast<std::size_t>(b)].variables ==
                  fp2.blocks[static_cast<std::size_t>(b)].variables,
              "Same var membership");
        check(fp1.blocks[static_cast<std::size_t>(b)].constraints ==
                  fp2.blocks[static_cast<std::size_t>(b)].constraints,
              "Same constraint membership");
    }
    check(fp1.coupling_constraints == fp2.coupling_constraints,
          "Same coupling");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 8: Three independent blocks, then one row couples two of them.
// =========================================================================
void testPartialCoupling() {
    std::cout << "testPartialCoupling...\n";
    // Block A: x0     (row 0)
    // Block B: x1     (row 1)
    // Block C: x2     (row 2)
    // Coupling: row 3 connects x0 and x1.
    CSRMatrix A = denseToCSR({
        {1.0, 0.0, 0.0},  // row 0: x0
        {0.0, 1.0, 0.0},  // row 1: x1
        {0.0, 0.0, 1.0},  // row 2: x2
        {1.0, 1.0, 0.0},  // row 3: x0, x1 (coupling)
    }, 3);

    auto fp = analyze(A);
    check(fp.numBlocks() == 3, "Three separate blocks");
    check(fp.coupling_constraints.size() == 1, "One coupling constraint");
    check(fp.coupling_constraints[0] == 3, "Row 3 is coupling");

    check(fp.blocks[0].variables == std::vector<CSRMatrix::Index>{0}, "Block A vars");
    check(fp.blocks[0].constraints == std::vector<CSRMatrix::Index>{0}, "Block A constraints");

    check(fp.blocks[1].variables == std::vector<CSRMatrix::Index>{1}, "Block B vars");
    check(fp.blocks[1].constraints == std::vector<CSRMatrix::Index>{1}, "Block B constraints");

    check(fp.blocks[2].variables == std::vector<CSRMatrix::Index>{2}, "Block C vars");
    check(fp.blocks[2].constraints == std::vector<CSRMatrix::Index>{2}, "Block C constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 9: N rows, 0 columns
// =========================================================================
void testZeroCols() {
    std::cout << "testZeroCols...\n";
    CSRMatrix A(std::vector<CSRMatrix::Index>{0, 0, 0}, std::vector<CSRMatrix::Index>{}, std::vector<double>{}, 0);
    auto fp = analyze(A);
    check(fp.numBlocks() == 0, "No blocks for 0 columns");
    check(fp.coupling_constraints.empty(), "No coupling constraints");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 10: One-variable rows
// =========================================================================
void testOneVariableRows() {
    std::cout << "testOneVariableRows...\n";
    // 3 variables. Row 0 has x0. Row 1 has x0, x1. Row 2 has x2.
    // x0 and x1 form a block. x2 is its own block.
    // Neither row is coupling.
    CSRMatrix A = denseToCSR({
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }, 3);
    auto fp = analyze(A);
    check(fp.numBlocks() == 2, "Two blocks");
    check(fp.coupling_constraints.empty(), "No coupling constraints");
    check(fp.blocks[0].variables == std::vector<CSRMatrix::Index>{0, 1}, "Block 0 vars");
    check(fp.blocks[0].constraints == std::vector<CSRMatrix::Index>{0, 1}, "Block 0 constraints");
    check(fp.blocks[1].variables == std::vector<CSRMatrix::Index>{2}, "Block 1 vars");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 11: Chain of three blocks
// =========================================================================
void testChainOfBlocks() {
    std::cout << "testChainOfBlocks...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // Row 0 (Block A)
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // Row 1 (Block A) - makes it biconnected
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0}, // Row 2 (Block B)
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0}, // Row 3 (Block B)
        {0.0, 0.0, 0.0, 0.0, 1.0, 1.0}, // Row 4 (Block C)
        {0.0, 0.0, 0.0, 0.0, 1.0, 1.0}, // Row 5 (Block C)
        {0.0, 1.0, 1.0, 0.0, 0.0, 0.0}, // Row 6 couples A and B
        {0.0, 0.0, 0.0, 1.0, 1.0, 0.0}, // Row 7 couples B and C
    }, 6);
    auto fp = analyze(A);
    check(fp.numBlocks() == 3, "Three blocks");
    check(fp.coupling_constraints.size() == 2, "Two coupling constraints");
    check(fp.coupling_constraints[0] == 6, "Row 6");
    check(fp.coupling_constraints[1] == 7, "Row 7");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 12: Cycle of coupling relationships
// =========================================================================
void testCycleOfCoupling() {
    std::cout << "testCycleOfCoupling...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // Row 0 (Block A)
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0}, // Row 1 (Block B)
        {0.0, 0.0, 0.0, 0.0, 1.0, 1.0}, // Row 2 (Block C)
        {0.0, 1.0, 1.0, 0.0, 0.0, 0.0}, // Row 3 couples A and B
        {0.0, 0.0, 0.0, 1.0, 1.0, 0.0}, // Row 4 couples B and C
        {1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, // Row 5 couples A and C
    }, 6);
    auto fp = analyze(A);
    // In a cycle, removing any single coupling row leaves the other blocks connected!
    // Therefore, NO row is an articulation point. The entire thing merges into 1 block.
    // This is mathematically correct: they do not form a strict block-angular tree.
    check(fp.numBlocks() == 1, "Cycle merges into 1 block");
    check(fp.coupling_constraints.empty(), "No articulation points in a cycle");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 13: Multiple coupling rows between SAME blocks (duplicate/repeated)
// =========================================================================
void testMultipleCouplingSameBlocks() {
    std::cout << "testMultipleCouplingSameBlocks...\n";
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0}, // Row 0 (Block A)
        {0.0, 0.0, 1.0, 1.0}, // Row 1 (Block B)
        {0.0, 1.0, 1.0, 0.0}, // Row 2 couples A and B
        {1.0, 0.0, 0.0, 1.0}, // Row 3 ALSO couples A and B
    }, 4);
    auto fp = analyze(A);
    // Two rows bridge A and B. Removing one does not separate them.
    // So neither is an AP. Merges into 1 block.
    check(fp.numBlocks() == 1, "Multiple parallel bridges merge into 1 block");
    check(fp.coupling_constraints.empty(), "No single AP");
    std::cout << "  ok\n";
}

// =========================================================================
// Test 14: Permutations and Coefficient changes
// =========================================================================
void testPermutations() {
    std::cout << "testPermutations...\n";
    // Original A:
    // Block 1 (x0, x1) in Row 0
    // Block 2 (x2, x3) in Row 1
    // Coupling: Row 2 (x1, x2)
    CSRMatrix A = denseToCSR({
        {1.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 1.0, 0.0},
    }, 4);
    auto fpA = analyze(A);
    // After coupling merge, 1 block, coupling row = 2.
    check(fpA.coupling_constraints.size() == 1 && fpA.coupling_constraints[0] == 2, "Original coupling");

    // Permute rows: 2->0, 0->1, 1->2
    CSRMatrix A_row = denseToCSR({
        {0.0, 1.0, 1.0, 0.0}, // old row 2
        {1.0, 1.0, 0.0, 0.0}, // old row 0
        {0.0, 0.0, 1.0, 1.0}, // old row 1
    }, 4);
    auto fpR = analyze(A_row);
    check(fpR.coupling_constraints.size() == 1 && fpR.coupling_constraints[0] == 0, "Row permuted coupling");

    // Permute columns: 0->3, 1->2, 2->1, 3->0 (reverse)
    CSRMatrix A_col = denseToCSR({
        {0.0, 0.0, 1.0, 1.0}, // old row 0
        {1.0, 1.0, 0.0, 0.0}, // old row 1
        {0.0, 1.0, 1.0, 0.0}, // old row 2
    }, 4);
    auto fpC = analyze(A_col);
    check(fpC.coupling_constraints.size() == 1 && fpC.coupling_constraints[0] == 2, "Col permuted coupling");
    std::cout << "  ok\n";
}

}  // namespace

void testRandomizedBruteForce();
int main() {
    testTwoIndependentBlocks();
    testCouplingConstraint();
    testFullyConnected();
    testEmptyMatrix();
    testIsolatedVariables();
    testZeroRows();
    testDeterminism();
    testPartialCoupling();
    testZeroCols();
    testOneVariableRows();
    testChainOfBlocks();
    testCycleOfCoupling();
    testMultipleCouplingSameBlocks();
    testPermutations();
    testRandomizedBruteForce();

    std::cout << "\n"
              << g_checks_run << " checks run, " << g_checks_failed
              << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}

#include <random>

namespace {
class FuzzUnionFind {
public:
    explicit FuzzUnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

int countNonTrivialComponentsBruteForce(const CSRMatrix& A, CSRMatrix::Index exclude_row) {
    int n = A.numCols();
    FuzzUnionFind uf(n);
    std::vector<bool> has_row(n, false);
    for (CSRMatrix::Index i = 0; i < A.numRows(); ++i) {
        if (i == exclude_row) continue;
        auto rv = A.row(i);
        if (rv.size() == 0) continue;
        CSRMatrix::Index first_col = -1;
        for (auto entry : rv) {
            has_row[entry.index] = true;
            if (first_col < 0) {
                first_col = entry.index;
            } else {
                uf.unite(first_col, entry.index);
            }
        }
    }
    int count = 0;
    for (int j = 0; j < n; ++j) {
        if (has_row[j] && uf.find(j) == j) {
            count++;
        }
    }
    return count;
}

std::vector<CSRMatrix::Index> getCouplingBruteForce(const CSRMatrix& A) {
    std::vector<CSRMatrix::Index> coupling;
    int base_components = countNonTrivialComponentsBruteForce(A, -1);
    for (CSRMatrix::Index i = 0; i < A.numRows(); ++i) {
        if (A.row(i).size() > 0) {
            int comps = countNonTrivialComponentsBruteForce(A, i);
            if (comps > base_components) {
                coupling.push_back(i);
            }
        }
    }
    return coupling;
}

} // namespace

void testRandomizedBruteForce() {
    std::cout << "testRandomizedBruteForce...\n";
    std::mt19937 gen(42);
    for (int iter = 0; iter < 500; ++iter) {
        std::uniform_int_distribution<> dist_m(1, 25);
        std::uniform_int_distribution<> dist_n(1, 25);
        int m = dist_m(gen);
        int n = dist_n(gen);
        std::uniform_real_distribution<> dist_density(0.05, 0.4);
        double density = dist_density(gen);

        std::vector<std::vector<double>> dense(m, std::vector<double>(n, 0.0));
        std::uniform_real_distribution<> dist_val(1.0, 10.0);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < density) {
                    dense[i][j] = dist_val(gen);
                }
            }
        }
        CSRMatrix A = denseToCSR(dense, n);

        auto fp = analyze(A);
        auto brute = getCouplingBruteForce(A);

        check(fp.coupling_constraints == brute, "Randomized brute-force mismatch");
        if (fp.coupling_constraints != brute) {
            std::cerr << "Mismatch on iter " << iter << " m=" << m << " n=" << n << "\n";
            std::cerr << "Optimized size: " << fp.coupling_constraints.size() << " Brute size: " << brute.size() << "\n";
            break;
        }
    }
    std::cout << "  ok\n";
}
