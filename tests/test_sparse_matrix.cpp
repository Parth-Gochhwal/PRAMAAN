#include "pramaan/sparse_matrix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using pramaan::CSRMatrix;

void test_basic_shape_and_multiply() {
    CSRMatrix A({0, 2, 3, 5}, {0, 2, 1, 0, 2}, {1, 2, 3, 4, 5}, 3);
    assert(A.numRows() == 3);
    assert(A.numCols() == 3);
    assert(A.nnz() == 5);

    std::vector<CSRMatrix::Scalar> x = {1, 1, 1};
    auto y = A.multiply(x);
    assert(std::abs(y[0] - 3) < 1e-12);
    assert(std::abs(y[1] - 3) < 1e-12);
    assert(std::abs(y[2] - 9) < 1e-12);

    int count = 0;
    for (auto entry : A.row(2)) { (void)entry; ++count; }
    assert(count == 2);

    auto col0 = A.column(0);
    assert(col0.size() == 2);
    assert(col0[0].index == 0 && col0[0].value == 1);
    assert(col0[1].index == 2 && col0[1].value == 4);

    std::vector<CSRMatrix::Scalar> y2;
    A.multiplyInto(x, y2);
    assert(y2 == y);

    bool threw = false;
    try { A.multiply({1, 2}); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    std::cout << "  test_basic_shape_and_multiply passed\n";
}

void test_non_square() {
    // 2 rows x 3 cols -- your actual LP shape: constraints x variables
    CSRMatrix A({0, 2, 3}, {0, 2, 1}, {1, 2, 3}, 3);
    assert(A.numRows() == 2);
    assert(A.numCols() == 3);
    auto y = A.multiply({1, 1, 1});
    assert(y.size() == 2);
    assert(std::abs(y[0] - 3) < 1e-12);
    assert(std::abs(y[1] - 3) < 1e-12);
    std::cout << "  test_non_square passed\n";
}

void test_empty_row() {
    // A = [[1,0],[0,0],[0,2]] -- middle row entirely empty
    CSRMatrix A({0, 1, 1, 2}, {0, 1}, {1, 2}, 2);
    int count = 0;
    for (auto entry : A.row(1)) { (void)entry; ++count; }
    assert(count == 0);
    auto y = A.multiply({5, 5});
    assert(std::abs(y[0] - 5) < 1e-12);
    assert(std::abs(y[1] - 0) < 1e-12);
    assert(std::abs(y[2] - 10) < 1e-12);
    std::cout << "  test_empty_row passed\n";
}

void test_transpose() {
    CSRMatrix A({0, 2, 3}, {0, 2, 1}, {1, 2, 3}, 3);   // 2x3
    auto At = A.transpose();
    assert(At.numRows() == 3);
    assert(At.numCols() == 2);
    auto y = At.multiply({1, 1});
    assert(std::abs(y[0] - 1) < 1e-12);
    assert(std::abs(y[1] - 3) < 1e-12);
    assert(std::abs(y[2] - 2) < 1e-12);

    auto Att = At.transpose();   // round-trip
    for (auto v : {std::vector<CSRMatrix::Scalar>{1, 0, 0},
                    std::vector<CSRMatrix::Scalar>{0, 1, 0},
                    std::vector<CSRMatrix::Scalar>{0, 0, 1}}) {
        assert(A.multiply(v) == Att.multiply(v));
    }
    std::cout << "  test_transpose passed\n";
}

void test_invalid_construction_throws() {
    auto expect_throw = [](auto build_fn, const char* label) {
        bool threw = false;
        try { build_fn(); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) std::cerr << "  FAILED to throw: " << label << "\n";
        assert(threw);
    };

    expect_throw([]{ CSRMatrix(std::vector<CSRMatrix::Index>{0, 2},
                                std::vector<CSRMatrix::Index>{0, 1},
                                std::vector<CSRMatrix::Scalar>{1}, 2); },
                 "col_idx/values mismatch");

    expect_throw([]{ CSRMatrix(std::vector<CSRMatrix::Index>{0, 1},
                                std::vector<CSRMatrix::Index>{5},
                                std::vector<CSRMatrix::Scalar>{1}, 2); },
                 "col_idx out of range");

    expect_throw([]{ CSRMatrix(std::vector<CSRMatrix::Index>{0, 3, 1},
                                std::vector<CSRMatrix::Index>{0, 1, 0},
                                std::vector<CSRMatrix::Scalar>{1, 2, 3}, 2); },
                 "non-decreasing row_ptr violated");

    std::cout << "  test_invalid_construction_throws passed\n";
}

void test_1x1_matrix() {
    CSRMatrix A({0, 1}, {0}, {7.0}, 1);
    auto y = A.multiply({3.0});
    assert(std::abs(y[0] - 21.0) < 1e-12);
    std::cout << "  test_1x1_matrix passed\n";
}

int main() {
    test_basic_shape_and_multiply();
    test_non_square();
    test_empty_row();
    test_transpose();
    test_invalid_construction_throws();
    test_1x1_matrix();
    std::cout << "All sparse_matrix tests passed.\n";
    return 0;
}