// sparse_matrix.cpp
// See include/pramaan/sparse_matrix.hpp.
//
// The header keeps everything except transpose() inline (constructor,
// multiply/multiplyInto, row(), column() are all defined in the class body).
// This file implements only CSRMatrix::transpose().
#include "pramaan/sparse_matrix.hpp"

#include <utility>

namespace pramaan {

CSRMatrix CSRMatrix::transpose() const {
    // Index and Scalar are CSRMatrix::Index / CSRMatrix::Scalar, already
    // visible unqualified inside this member function.
    //
    // Standard counting-sort transpose, O(nnz + num_cols):
    //   1. count nonzeros per (original) column -> becomes transposed row_ptr
    //   2. prefix-sum the counts into offsets
    //   3. scatter each nonzero into its transposed slot using a per-row
    //      cursor that starts at each row's offset and advances by one each
    //      time that row is written to.
    //
    // Because step 3 walks original rows r = 0, 1, 2, ... in order, and for
    // a fixed original column c every write lands in transposed row c with
    // strictly increasing r, the result satisfies the same "sorted, unique
    // column indices per row" invariant the header documents for input
    // matrices -- so transpose() output is itself a valid CSRMatrix to feed
    // back into row()/column()/multiply() or transpose() again.
    std::vector<Index> t_row_ptr(static_cast<std::size_t>(num_cols_) + 1, 0);
    for (Index c : col_idx_) {
        ++t_row_ptr[static_cast<std::size_t>(c) + 1];
    }
    for (Index c = 0; c < num_cols_; ++c) {
        t_row_ptr[static_cast<std::size_t>(c) + 1] += t_row_ptr[static_cast<std::size_t>(c)];
    }

    std::vector<Index> t_col_idx(col_idx_.size());
    std::vector<Scalar> t_values(values_.size());
    std::vector<Index> cursor = t_row_ptr;  // next free slot per transposed row

    for (Index r = 0; r < num_rows_; ++r) {
        const Index row_end = row_ptr_[r + 1];
        for (Index k = row_ptr_[r]; k < row_end; ++k) {
            const Index c = col_idx_[k];
            const Index dest = cursor[static_cast<std::size_t>(c)]++;
            t_col_idx[static_cast<std::size_t>(dest)] = r;
            t_values[static_cast<std::size_t>(dest)] = values_[k];
        }
    }

    // New matrix: num_rows(new) == num_cols(old), num_cols(new) == num_rows(old).
    // The constructor derives num_rows from row_ptr.size() - 1, which is
    // num_cols_ here, so we only need to pass num_rows_ as the explicit
    // num_cols argument.
    return CSRMatrix(std::move(t_row_ptr), std::move(t_col_idx), std::move(t_values), num_rows_);
}

}  // namespace pramaan