// dual_simplex.hpp
// P1 Step 1 — Dual Simplex + Basis Warm-Start
//
// Provides:
// - BasisState:   a cached basis from a solved LP, sufficient to warm-start
//                 a modified LP (same structure, changed RHS/bounds).
// - DualSimplex:  a dual-simplex solver that takes an inherited basis from
//                 an optimal primal solve, recomputes the basic solution for
//                 modified RHS/bounds, and re-optimizes using dual pivots.
//
// The dual simplex starts from a dual-feasible basis (which an optimal primal
// basis is) and restores primal feasibility through dual-simplex pivots.
// This is the foundation for P1's warm-started B&B node relaxations.
#pragma once

#include <cstdint>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {

// Captured basis from a solved LP.  Stores enough information to
// reconstruct the simplex state when the LP structure is unchanged
// but RHS or variable bounds have been modified.
//
// Supported:
// - RHS value changes
// - Changes to already-existing finite variable bounds, provided they preserve
//   the standard-form structural dimensions/representation.
//
// Not currently supported:
// - Changing objective coefficients
// - Changing matrix coefficients
// - Changing number/order of variables
// - Changing number/order of constraints
// - Changes that add/remove standard-form rows/columns (e.g. changing a
//   variable bound from finite to infinite)
// - Changing the structural representation of the model (e.g. changing an
//   inequality constraint to an equality)
struct BasisState {
    // Standard-form basis column indices, size == num_std_rows.
    std::vector<int> basis_columns;

    // Structural fingerprint for compatibility checks.
    // Hashes objective coefficients, matrix structure/values, and row/var bound
    // types (finite/infinite, equality/inequality). Does NOT hash RHS or bound values.
    std::uint64_t structural_fingerprint = 0;

    // Fingerprint of the standard-form layout (number of columns and artificials).
    // Detects when an RHS value change causes buildSparseStandardForm to flip
    // a row's sign and alter the standard-form matrix architecture.
    std::uint64_t std_layout_fingerprint = 0;

    int orig_num_vars = 0;
    int orig_num_rows = 0;
    ObjSense orig_obj_sense = ObjSense::kMinimize;

    bool empty() const noexcept { return basis_columns.empty(); }
};

// Dual simplex solver with warm-start from a previously captured basis.
//
// Usage:
//   1. Cold-solve the original LP with RevisedSimplex.
//   2. Capture its basis via DualSimplex::captureBasis().
//   3. Modify the model's RHS or variable bounds.
//   4. Call DualSimplex::warmSolve() with the modified model and cached basis.
//
// The warm-start path:
//   - Rebuilds the standard form with the new RHS/bounds.
//   - Injects the inherited basis (instead of an artificial-variable basis).
//   - Recomputes the basic solution x_B = B^{-1} b for the new RHS.
//   - If x_B >= 0 already, the basis is still optimal → zero pivots.
//   - Otherwise, performs dual-simplex pivots until primal feasibility
//     is restored or infeasibility is detected.
//
// Thread-safety: same as RevisedSimplex — no mutable state between calls.
class DualSimplex {
public:
    using Options = RevisedSimplex::Options;

    DualSimplex() = default;
    explicit DualSimplex(Options options) : options_(options) {}

    // Captures the internal standard-form basis from a cold solve of
    // `model`.  The returned BasisState can be passed to warmSolve()
    // after modifying the model's RHS or variable bounds.
    //
    // Returns an empty BasisState if the cold solve does not reach
    // optimality, or if it reaches optimality on a basis that is not safe to
    // warm-start from (an artificial column still basic on a redundant row).
    //
    // `out_result`, when non-null, additionally receives the SolveResult of
    // the cold solve performed internally, so a caller needing both the LP
    // solution and a reusable basis pays for one solve rather than two.
    // An empty returned basis therefore does NOT imply a failed solve --
    // check *out_result for that.
    BasisState captureBasis(const ModelIR& model, SolveResult* out_result = nullptr) const;

    // Warm-solves `model` using the inherited `basis`.
    // The model must have the same structure (same number of variables
    // and rows, same constraint matrix, same objective sense) as the
    // model that produced the basis; only RHS and variable bounds may
    // differ.
    //
    // Returns a SolveResult identical to what a cold solve would
    // produce, but typically with fewer pivots.
    //
    // Throws std::invalid_argument if `basis` is empty or structurally
    // incompatible with `model`; callers that cannot guarantee compatibility
    // (such as the B&B engine, where a branch may turn an infinite bound
    // finite) must catch it and fall back to a cold solve.
    //
    // `out_basis`, when non-null, receives the basis this solve ends on, for
    // warm-starting a further modification without re-solving. It is left
    // empty unless the solve is optimal on a safely reusable basis.
    SolveResult warmSolve(const ModelIR& model, const BasisState& basis,
                          BasisState* out_basis = nullptr) const;

    const Options& options() const noexcept { return options_; }

private:
    Options options_{};
};

}  // namespace pramaan

