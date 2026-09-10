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
#include <stdexcept>
#include <string>
#include <vector>

#include "pramaan/sparse_matrix.hpp"

namespace pramaan {

// Whether the objective is to be minimized or maximized. Not in the
// original TODO list, but every LP needs this to interpret obj_coeffs
// correctly -- defaults to minimize (the MPS/CPLEX-LP convention).
enum class ObjSense {
    kMinimize,
    kMaximize,
};

enum class VarType {
    kContinuous,
    kInteger,
};

// Sentinel for an unbounded row/variable bound. Deliberately a large finite
// value rather than std::numeric_limits<double>::infinity(): IEEE infinity
// is correct in isolation, but the moment a downstream solver multiplies a
// zero dual/reduced-cost against an "infinite" bound (0 * inf == NaN), the
// NaN silently poisons everything after it. A large finite sentinel avoids
// that whole class of bug. 1e30 matches the convention HiGHS uses; CPLEX
// uses 1e20 and Gurobi 1e100 -- the exact magnitude doesn't matter much as
// long as it's used consistently and stays well outside any real problem's
// coefficient range.
inline constexpr double kInfinity = 1.0e30;

// PRAMAAN-IR: the canonical in-memory model.
//
// Row convention: row_lower[r] <= (A x)[r] <= row_upper[r]. This single
// ranged form subsumes <=, >=, =, and ranged rows without a separate sense
// enum:
//   <=  rhs   -> row_lower = -kInfinity, row_upper = rhs
//   >=  rhs   -> row_lower = rhs,        row_upper = kInfinity
//   =   rhs   -> row_lower = row_upper = rhs
//   lo <= .. <= hi (ranged) -> row_lower = lo, row_upper = hi
//
// This is a plain data holder, not an encapsulated class: presolve,
// fingerprinting, and the solvers all read and mutate it directly, so
// getters/setters would just add ceremony without adding safety.
struct ModelIR {
    ModelIR() = default;

    ModelIR(ObjSense obj_sense_in,
            double obj_offset_in,
            std::vector<double> obj_coeffs_in,
            CSRMatrix A_in,
            std::vector<double> row_lower_in,
            std::vector<double> row_upper_in,
            std::vector<std::string> row_names_in,
            std::vector<double> var_lower_in,
            std::vector<double> var_upper_in,
            std::vector<VarType> var_types_in,
            std::vector<std::string> var_names_in);
    ObjSense obj_sense = ObjSense::kMinimize;
    double obj_offset = 0.0;              // constant added to c^T x for reporting
    std::vector<double> obj_coeffs;        // size num_vars, dense (c)

    CSRMatrix A;                           // num_rows x num_vars

    std::vector<double> row_lower;         // size num_rows
    std::vector<double> row_upper;         // size num_rows
    std::vector<std::string> row_names;    // size num_rows

    std::vector<double> var_lower;         // size num_vars
    std::vector<double> var_upper;         // size num_vars
    std::vector<VarType> var_types;        // size num_vars
    std::vector<std::string> var_names;    // size num_vars

    // --- Convenience accessors --------------------------------------------

    CSRMatrix::Index numVars() const noexcept { return A.numCols(); }
    CSRMatrix::Index numRows() const noexcept { return A.numRows(); }

    bool isInteger(CSRMatrix::Index j) const { return var_types[j] == VarType::kInteger; }

    // A model with no integer variables can go straight to an LP solver;
    // otherwise it needs branch-and-bound (or similar) around the LP core.
    bool isPureLP() const {
        for (VarType t : var_types) {
            if (t == VarType::kInteger) return false;
        }
        return true;
    }

    bool isEqualityRow(CSRMatrix::Index r) const { return row_lower[r] == row_upper[r]; }

    bool isFreeRow(CSRMatrix::Index r) const {
        return row_lower[r] <= -kInfinity && row_upper[r] >= kInfinity;
    }

    // Checks internal consistency: every array is sized against A's shape,
    // and every bound pair is non-inverted. Throws std::invalid_argument on
    // the first problem found, naming which invariant broke, so a bad
    // parser or presolve pass fails loudly instead of corrupting downstream
    // solves.
    void validate() const;
};

}  // namespace pramaan