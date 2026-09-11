// certificate.hpp
// PILLAR 4 (Trust) -- Zone 7, shared between pramaan-solve and pramaan-verify
//
// Purpose: the data format for a solution certificate -- primal residual,
// bound violation, integrality violation, objective value, and a reference
// to the transformation ledger for demonstrating the solution corresponds
// to the original-space model.
//
// THIS FILE is the contract between the solver and the verifier -- keep it
// solver-logic-free, pure data + serialization.
//
// Certificate format: PRAMAAN_CERTIFICATE_V1, a deterministic plain-text
// format designed for human readability, manual corruption testing, and
// independent machine parsing. No JSON or external library dependencies.
#pragma once
#include <optional>
#include <string>
#include <vector>

namespace pramaan {

// Forward declarations -- the certificate does not depend on solver internals.
struct ModelIR;

// ---------------------------------------------------------------------------
// Certificate data model
// ---------------------------------------------------------------------------
struct Certificate {
    // Format/version.
    std::string format_id = "PRAMAAN_CERTIFICATE_V1";

    // Solver status string ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", etc.).
    std::string status;

    // Dimensions from the original model (for sanity-checking).
    int num_rows = 0;
    int num_cols = 0;

    // Objective value: c^T x + obj_offset, recomputed from original model.
    double objective_value = 0.0;

    // Numerical tolerance used for feasibility checking.
    double tolerance = 1e-6;

    // Primal feasibility: max( max_row_violation, 0 ) over all rows.
    // Rows are checked as: row_lower[r] <= (Ax)[r] <= row_upper[r].
    double primal_residual = 0.0;

    // Dual and complementarity residuals. The current primal simplex does not
    // produce independent dual information, so these are explicitly marked
    // as unavailable rather than fabricating a 0.0.
    std::optional<double> dual_residual;
    std::optional<double> complementarity_residual;

    // Variable bound violation: max over j of max(lb[j]-x[j], x[j]-ub[j], 0).
    double bound_violation = 0.0;

    // Integrality violation: max over integer vars of |x[j] - round(x[j])|.
    // For pure-LP models (Stage 7 scope), this is always 0.0.
    double integrality_violation = 0.0;

    // Objective bound gap. For the current LP scope, the primal simplex
    // does not produce an independent dual bound. This is left empty.
    std::optional<double> objective_bound_gap;

    // Number of transformation ledger entries applied (for auditability).
    int ledger_entries = 0;

    // Original-space primal solution vector.
    std::vector<double> x;
};

// ---------------------------------------------------------------------------
// Certificate generation (solver-side)
// ---------------------------------------------------------------------------

// Generates a certificate from the original model and the original-space
// solution vector. Independently recomputes all residuals from scratch.
// `ledger_entries` records how many transformation ledger entries were used.
Certificate generate_certificate(const ModelIR& original_model,
                                 const std::vector<double>& x,
                                 const std::string& status,
                                 int ledger_entries);

// ---------------------------------------------------------------------------
// Certificate serialization
// ---------------------------------------------------------------------------

// Writes a certificate to a file in PRAMAAN_CERTIFICATE_V1 format.
// Throws std::runtime_error on I/O failure.
void write_certificate(const Certificate& cert, const std::string& filepath);

// Reads a certificate from a file in PRAMAAN_CERTIFICATE_V1 format.
// Throws std::runtime_error on I/O or parse failure.
Certificate read_certificate(const std::string& filepath);

}  // namespace pramaan
