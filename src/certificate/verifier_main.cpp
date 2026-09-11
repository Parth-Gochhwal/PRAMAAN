// verifier_main.cpp
// PILLAR 4 (Trust) -- Zone 7's "pramaan-verify"
//
// Independent verifier: reads an original MPS model and a certificate file,
// recomputes all residuals from scratch, and validates the certificate.
//
// This program does NOT call RevisedSimplex or any solver routine.
// It does NOT trust residual values written into the certificate.
// It recomputes everything independently from the model and certificate x.
//
// Usage: pramaan-verify <model.mps> <certificate.cert>
//
// Output:
//   CERTIFICATE VALID    (exit code 0)
//   CERTIFICATE INVALID  (exit code 1)
//
// Links against pramaan_core only for: MPS parsing, certificate I/O,
// ModelIR data types, and CSRMatrix::multiply(). No solver code is used.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "pramaan/certificate.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: pramaan-verify <model.mps> <certificate.cert>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string cert_path = argv[2];

    // --- Parse the original model ---
    pramaan::ModelIR model;
    try {
        model = pramaan::parse_mps(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing model: " << e.what() << "\n";
        std::cout << "CERTIFICATE INVALID\n";
        return 1;
    }

    // --- Parse the certificate ---
    pramaan::Certificate cert;
    try {
        cert = pramaan::read_certificate(cert_path);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing certificate: " << e.what() << "\n";
        std::cout << "CERTIFICATE INVALID\n";
        return 1;
    }

    // --- Validation ---
    bool valid = true;
    const double tol = cert.tolerance;
    const int m = model.numRows();
    const int n = model.numVars();

    auto fail = [&](const std::string& reason) {
        std::cerr << "  FAIL: " << reason << "\n";
        valid = false;
    };

    // 0. Tolerance security check
    const double max_trusted_tolerance = 1e-4; // or 1e-3, enough for AFIRO
    if (tol > max_trusted_tolerance) {
        fail("Tolerance too large: " + std::to_string(tol) + " > " + std::to_string(max_trusted_tolerance));
    }

    // 1. Format check.
    if (cert.format_id != "PRAMAAN_CERTIFICATE_V1") {
        fail("Unknown certificate format: " + cert.format_id);
    }

    // 2. Status check.
    if (cert.status != "OPTIMAL") {
        fail("Certificate status is not OPTIMAL: " + cert.status);
    }

    // 3. Dimension checks.
    if (cert.num_rows != m) {
        fail("Row count mismatch: certificate=" + std::to_string(cert.num_rows)
             + " model=" + std::to_string(m));
    }
    if (cert.num_cols != n) {
        fail("Column count mismatch: certificate=" + std::to_string(cert.num_cols)
             + " model=" + std::to_string(n));
    }
    if (static_cast<int>(cert.x.size()) != n) {
        fail("Solution vector size mismatch: x.size()=" + std::to_string(cert.x.size())
             + " num_cols=" + std::to_string(n));
    }

    // 4. Check all x values are finite.
    for (int j = 0; j < static_cast<int>(cert.x.size()); ++j) {
        if (std::isnan(cert.x[static_cast<std::size_t>(j)]) ||
            std::isinf(cert.x[static_cast<std::size_t>(j)])) {
            fail("x[" + std::to_string(j) + "] is NaN or Inf");
        }
    }

    // If dimensions don't match, we can't proceed with further checks.
    if (static_cast<int>(cert.x.size()) != n) {
        std::cout << "CERTIFICATE INVALID\n";
        return 1;
    }

    // 5. Independently recompute objective.
    double recomputed_obj = model.obj_offset;
    for (int j = 0; j < n; ++j) {
        recomputed_obj += model.obj_coeffs[static_cast<std::size_t>(j)]
                        * cert.x[static_cast<std::size_t>(j)];
    }
    double obj_diff = std::abs(recomputed_obj - cert.objective_value);
    if (obj_diff > tol) {
        fail("Objective mismatch: recomputed=" + std::to_string(recomputed_obj)
             + " certificate=" + std::to_string(cert.objective_value)
             + " |diff|=" + std::to_string(obj_diff));
    }

    // 6. Independently recompute row activities and primal residual.
    std::vector<double> activity = model.A.multiply(cert.x);
    double recomputed_primal_residual = 0.0;
    for (int r = 0; r < m; ++r) {
        double ar = activity[static_cast<std::size_t>(r)];
        double lo = model.row_lower[static_cast<std::size_t>(r)];
        double hi = model.row_upper[static_cast<std::size_t>(r)];

        if (lo > -pramaan::kInfinity) {
            double viol = lo - ar;
            if (viol > recomputed_primal_residual) recomputed_primal_residual = viol;
        }
        if (hi < pramaan::kInfinity) {
            double viol = ar - hi;
            if (viol > recomputed_primal_residual) recomputed_primal_residual = viol;
        }
    }

    // Check that the recomputed primal residual is within tolerance.
    if (recomputed_primal_residual > tol) {
        fail("Primal infeasible: max row violation=" + std::to_string(recomputed_primal_residual));
    }

    // Check that the certificate's claimed primal residual matches.
    double pr_diff = std::abs(recomputed_primal_residual - cert.primal_residual);
    if (pr_diff > tol) {
        fail("Primal residual mismatch: recomputed=" + std::to_string(recomputed_primal_residual)
             + " certificate=" + std::to_string(cert.primal_residual)
             + " |diff|=" + std::to_string(pr_diff));
    }

    // 7. Independently recompute variable bound violations.
    double recomputed_bound_violation = 0.0;
    for (int j = 0; j < n; ++j) {
        double xj = cert.x[static_cast<std::size_t>(j)];
        double lo = model.var_lower[static_cast<std::size_t>(j)];
        double hi = model.var_upper[static_cast<std::size_t>(j)];

        if (lo > -pramaan::kInfinity) {
            double viol = lo - xj;
            if (viol > recomputed_bound_violation) recomputed_bound_violation = viol;
        }
        if (hi < pramaan::kInfinity) {
            double viol = xj - hi;
            if (viol > recomputed_bound_violation) recomputed_bound_violation = viol;
        }
    }

    if (recomputed_bound_violation > tol) {
        fail("Bound violation: max=" + std::to_string(recomputed_bound_violation));
    }

    double bv_diff = std::abs(recomputed_bound_violation - cert.bound_violation);
    if (bv_diff > tol) {
        fail("Bound violation mismatch: recomputed=" + std::to_string(recomputed_bound_violation)
             + " certificate=" + std::to_string(cert.bound_violation)
             + " |diff|=" + std::to_string(bv_diff));
    }

    // 8. Integrality check (currently all continuous).
    double recomputed_int_viol = 0.0;
    for (int j = 0; j < n; ++j) {
        if (model.var_types[static_cast<std::size_t>(j)] == pramaan::VarType::kInteger) {
            double xj = cert.x[static_cast<std::size_t>(j)];
            double rounded = std::round(xj);
            double viol = std::abs(xj - rounded);
            if (viol > recomputed_int_viol) recomputed_int_viol = viol;
        }
    }
    if (recomputed_int_viol > tol) {
        fail("Integrality violation: max=" + std::to_string(recomputed_int_viol));
    }
    double iv_diff = std::abs(recomputed_int_viol - cert.integrality_violation);
    if (iv_diff > tol) {
        fail("Integrality violation mismatch: recomputed=" + std::to_string(recomputed_int_viol)
             + " certificate=" + std::to_string(cert.integrality_violation)
             + " |diff|=" + std::to_string(iv_diff));
    }

    // 9. Unsupported fields check (Stage 7 LP scope)
    if (cert.dual_residual.has_value()) {
        fail("Dual residual is currently unsupported, must be UNAVAILABLE");
    }
    if (cert.complementarity_residual.has_value()) {
        fail("Complementarity residual is currently unsupported, must be UNAVAILABLE");
    }
    if (cert.objective_bound_gap.has_value()) {
        fail("Objective bound gap is currently unsupported, must be UNAVAILABLE");
    }

    // --- Result ---
    if (valid) {
        std::cout << "CERTIFICATE VALID\n";
        return 0;
    } else {
        std::cout << "CERTIFICATE INVALID\n";
        return 1;
    }
}
