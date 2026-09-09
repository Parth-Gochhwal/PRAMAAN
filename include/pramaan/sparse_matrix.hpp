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
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pramaan {

// Compressed Sparse Row matrix.
//
// Storage: row_ptr_ has num_rows_+1 entries; the non-zeros of row r live in
// col_idx_[row_ptr_[r] .. row_ptr_[r+1]) with matching entries in values_.
// Within a row, column indices are assumed sorted and unique (standard CSR
// invariant) -- callers building this matrix are responsible for that.
class CSRMatrix {
public:
    using Index = std::int32_t;
    using Scalar = double;

    CSRMatrix() = default;

    // Construct directly from CSR triplet arrays. row_ptr must have
    // num_rows+1 entries; col_idx/values must have matching sizes equal to
    // row_ptr.back() (the nnz count). num_cols is not derivable from the
    // arrays alone, so it's passed explicitly.
    CSRMatrix(std::vector<Index> row_ptr,
              std::vector<Index> col_idx,
              std::vector<Scalar> values,
              Index num_cols)
        : row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          values_(std::move(values)),
          num_cols_(num_cols) {
        num_rows_ = row_ptr_.empty() ? 0 : static_cast<Index>(row_ptr_.size()) - 1;
        validate();
    }

    // --- Shape / storage accessors ---------------------------------------

    Index numRows() const noexcept { return num_rows_; }
    Index numCols() const noexcept { return num_cols_; }
    Index nnz() const noexcept { return static_cast<Index>(values_.size()); }

    const std::vector<Index>& rowPtr() const noexcept { return row_ptr_; }
    const std::vector<Index>& colIdx() const noexcept { return col_idx_; }
    const std::vector<Scalar>& values() const noexcept { return values_; }

    // --- Matrix-vector multiply: y = A * x --------------------------------
    // O(nnz) -- the hot path for simplex and PDHG iterations.

    std::vector<Scalar> multiply(const std::vector<Scalar>& x) const {
        std::vector<Scalar> y(static_cast<std::size_t>(num_rows_));
        multiplyInto(x, y);
        return y;
    }

    // Same as multiply(), but writes into a caller-supplied buffer to avoid
    // an allocation per call. Use this inside hot solver loops.
    void multiplyInto(const std::vector<Scalar>& x, std::vector<Scalar>& y) const {
        if (static_cast<Index>(x.size()) != num_cols_) {
            throw std::invalid_argument("CSRMatrix::multiply: x size != num_cols");
        }
        y.assign(static_cast<std::size_t>(num_rows_), Scalar{0});
        for (Index r = 0; r < num_rows_; ++r) {
            Scalar sum = 0;
            const Index row_end = row_ptr_[r + 1];
            for (Index k = row_ptr_[r]; k < row_end; ++k) {
                sum += values_[k] * x[col_idx_[k]];
            }
            y[r] = sum;
        }
    }

    // --- Row iteration ------------------------------------------------------
    // O(1) to obtain, O(row nnz) to iterate -- this is CSR's native access
    // pattern.

    struct Entry {
        Index index;   // column index for a row view, row index for a column view
        Scalar value;
    };

    class RowView {
    public:
        RowView(const Index* cols, const Scalar* vals, Index count)
            : cols_(cols), vals_(vals), count_(count) {}

        struct Iterator {
            const Index* cols;
            const Scalar* vals;
            Index pos;
            Entry operator*() const { return Entry{cols[pos], vals[pos]}; }
            Iterator& operator++() { ++pos; return *this; }
            bool operator!=(const Iterator& other) const { return pos != other.pos; }
        };

        Iterator begin() const { return Iterator{cols_, vals_, 0}; }
        Iterator end() const { return Iterator{cols_, vals_, count_}; }
        Index size() const noexcept { return count_; }

    private:
        const Index* cols_;
        const Scalar* vals_;
        Index count_;
    };

    RowView row(Index r) const {
        assert(r >= 0 && r < num_rows_);
        const Index start = row_ptr_[r];
        const Index end = row_ptr_[r + 1];
        return RowView(col_idx_.data() + start, values_.data() + start, end - start);
    }

    // --- Column iteration -----------------------------------------------
    // CSR carries no column index, so this is an O(nnz) scan over every row.
    // Fine for occasional use (e.g. pivoting, debugging); if a solver needs
    // repeated column access, build a companion CSCMatrix instead of calling
    // this in a hot loop.

    std::vector<Entry> column(Index c) const {
        assert(c >= 0 && c < num_cols_);
        std::vector<Entry> result;
        for (Index r = 0; r < num_rows_; ++r) {
            const Index row_end = row_ptr_[r + 1];
            for (Index k = row_ptr_[r]; k < row_end; ++k) {
                if (col_idx_[k] == c) {
                    result.push_back(Entry{r, values_[k]});
                    break;  // at most one entry per (row, col) under the CSR invariant
                }
            }
        }
        return result;
    }
    // Returns A^T as a new CSRMatrix (whose rows are A's columns).
    // O(nnz + num_cols) counting-sort transpose. Defined out-of-line in
    // sparse_matrix.cpp since it's not on any hot loop.
    CSRMatrix transpose() const;

private:
    void validate() const {
        if (static_cast<Index>(row_ptr_.size()) != num_rows_ + 1) {
            throw std::invalid_argument("CSRMatrix: row_ptr must have num_rows + 1 entries");
        }
        if (col_idx_.size() != values_.size()) {
            throw std::invalid_argument("CSRMatrix: col_idx and values size mismatch");
        }
        if (!row_ptr_.empty() &&
            static_cast<std::size_t>(row_ptr_.back()) != col_idx_.size()) {
            throw std::invalid_argument("CSRMatrix: row_ptr.back() must equal nnz");
        }
        for (Index r = 0; r + 1 < static_cast<Index>(row_ptr_.size()); ++r) {
            if (row_ptr_[r] > row_ptr_[r + 1]) {
                throw std::invalid_argument("CSRMatrix: row_ptr must be non-decreasing");
            }
        }
        for (Index c : col_idx_) {
            if (c < 0 || c >= num_cols_) {
                throw std::invalid_argument("CSRMatrix: col_idx entry out of bounds");
            }
        }
    }

    std::vector<Index> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Scalar> values_;
    Index num_rows_ = 0;
    Index num_cols_ = 0;
};

}  // namespace pramaan