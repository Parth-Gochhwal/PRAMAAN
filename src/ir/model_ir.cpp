#include <cstring>
#include "pramaan/ir.hpp"

#include <stdexcept>
#include <utility>

namespace pramaan {

ModelIR::ModelIR(
    ObjSense obj_sense_in,
    double obj_offset_in,
    std::vector<double> obj_coeffs_in,
    CSRMatrix A_in,
    std::vector<double> row_lower_in,
    std::vector<double> row_upper_in,
    std::vector<std::string> row_names_in,
    std::vector<double> var_lower_in,
    std::vector<double> var_upper_in,
    std::vector<VarType> var_types_in,
    std::vector<std::string> var_names_in)

    : obj_sense(obj_sense_in),
      obj_offset(obj_offset_in),
      obj_coeffs(std::move(obj_coeffs_in)),
      A(std::move(A_in)),
      row_lower(std::move(row_lower_in)),
      row_upper(std::move(row_upper_in)),
      row_names(std::move(row_names_in)),
      var_lower(std::move(var_lower_in)),
      var_upper(std::move(var_upper_in)),
      var_types(std::move(var_types_in)),
      var_names(std::move(var_names_in)) {

    validate();
}


void ModelIR::validate() const {

    const auto n = static_cast<std::size_t>(numVars());
    const auto m = static_cast<std::size_t>(numRows());

    if (obj_coeffs.size() != n) {
        throw std::invalid_argument(
            "ModelIR: obj_coeffs size != num_vars");
    }

    if (var_lower.size() != n ||
        var_upper.size() != n) {

        throw std::invalid_argument(
            "ModelIR: var_lower/var_upper size != num_vars");
    }

    if (var_types.size() != n) {
        throw std::invalid_argument(
            "ModelIR: var_types size != num_vars");
    }

    if (var_names.size() != n) {
        throw std::invalid_argument(
            "ModelIR: var_names size != num_vars");
    }

    if (row_lower.size() != m ||
        row_upper.size() != m) {

        throw std::invalid_argument(
            "ModelIR: row_lower/row_upper size != num_rows");
    }

    if (row_names.size() != m) {
        throw std::invalid_argument(
            "ModelIR: row_names size != num_rows");
    }

    for (std::size_t j = 0; j < n; ++j) {

        if (var_lower[j] > var_upper[j]) {
            throw std::invalid_argument(
                "ModelIR: var_lower > var_upper for some variable");
        }

    }

    for (std::size_t r = 0; r < m; ++r) {

        if (row_lower[r] > row_upper[r]) {
            throw std::invalid_argument(
                "ModelIR: row_lower > row_upper for some row");
        }

    }
}

uint64_t computeStructuralFingerprint(const ModelIR& model) {
    uint64_t hash = 14695981039346656037ULL;
    auto add_int = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ULL;
    };

    add_int(model.numVars());
    add_int(model.numRows());
    add_int(static_cast<uint64_t>(model.obj_sense));

    for (double c : model.obj_coeffs) {
        uint64_t bits;
        if (c == 0.0) c = 0.0;
        std::memcpy(&bits, &c, sizeof(double));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }

    for (auto v : model.A.rowPtr()) add_int(v);
    for (auto v : model.A.colIdx()) add_int(v);

    for (double v : model.A.values()) {
        uint64_t bits;
        if (v == 0.0) v = 0.0;
        std::memcpy(&bits, &v, sizeof(double));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    for (CSRMatrix::Index j = 0; j < model.numVars(); ++j) {
        add_int(model.var_lower[j] <= -kInfinity ? 1 : 0);
        add_int(model.var_upper[j] >= kInfinity ? 1 : 0);
    }
    for (CSRMatrix::Index r = 0; r < model.numRows(); ++r) {
        add_int(model.isEqualityRow(r) ? 1 : 0);
        add_int(model.isFreeRow(r) ? 1 : 0);
        add_int(model.row_lower[r] <= -kInfinity ? 1 : 0);
        add_int(model.row_upper[r] >= kInfinity ? 1 : 0);
    }
    return hash;
}

uint64_t computeModelFingerprint(const ModelIR& model) {
    uint64_t hash = 14695981039346656037ULL;
    auto add_double = [&](double v) {
        uint64_t bits;
        if (v == 0.0) v = 0.0; // normalize -0.0
        std::memcpy(&bits, &v, sizeof(double));
        hash ^= bits;
        hash *= 1099511628211ULL;
    };
    auto add_int = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ULL;
    };

    add_int(model.numVars());
    add_int(model.numRows());
    add_double(model.obj_offset);
    add_int(static_cast<uint64_t>(model.obj_sense));
    for (double c : model.obj_coeffs) add_double(c);

    for (auto v : model.A.rowPtr()) add_int(v);
    for (auto v : model.A.colIdx()) add_int(v);
    for (auto v : model.A.values()) add_double(v);

    for (CSRMatrix::Index j = 0; j < model.numVars(); ++j) {
        add_double(model.var_lower[j]);
        add_double(model.var_upper[j]);
        add_int(static_cast<uint64_t>(model.var_types[j]));
    }
    for (CSRMatrix::Index r = 0; r < model.numRows(); ++r) {
        add_double(model.row_lower[r]);
        add_double(model.row_upper[r]);
    }
    return hash;
}

}  // namespace pramaan
