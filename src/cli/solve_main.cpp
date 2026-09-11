// solve_main.cpp
// The pramaan-solve entry point. Wires together: mps_parser -> ModelIR ->
// Presolver -> RevisedSimplex -> Certificate -> print result.
//
// Usage: pramaan-solve <model.mps> [certificate.cert]
//
// If a certificate path is provided, the solver writes a PRAMAAN_CERTIFICATE_V1
// certificate file after solving.
#include <iostream>
#include <string>

#include "pramaan/certificate.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/presolve.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/transformation_ledger.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pramaan-solve <model.mps> [certificate.cert]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string cert_path = (argc >= 3) ? argv[2] : "";

    // --- Parse ---
    pramaan::ModelIR original;
    try {
        original = pramaan::parse_mps(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Model: " << model_path << "\n";
    std::cout << "  Rows: " << original.numRows()
              << "  Cols: " << original.numVars() << "\n";

    // --- Presolve ---
    pramaan::TransformationLedger ledger;
    pramaan::ModelIR reduced = pramaan::presolve_fixed_variables(original, ledger);
    int vars_removed = original.numVars() - reduced.numVars();
    if (vars_removed > 0) {
        std::cout << "  Presolve: " << vars_removed << " fixed vars removed\n";
    }

    // --- Scale ---
    pramaan::ModelIR pre_scale = reduced;
    pramaan::ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);

    // --- Solve ---
    pramaan::RevisedSimplex solver;
    pramaan::SolveResult result = solver.solve(reduced);

    // --- Map status to string ---
    std::string status_str;
    switch (result.status) {
        case pramaan::SolveStatus::kOptimal:       status_str = "OPTIMAL"; break;
        case pramaan::SolveStatus::kInfeasible:    status_str = "INFEASIBLE"; break;
        case pramaan::SolveStatus::kUnbounded:     status_str = "UNBOUNDED"; break;
        case pramaan::SolveStatus::kIterationLimit: status_str = "ITERATION_LIMIT"; break;
    }
    std::cout << "  Status: " << status_str << "\n";

    if (result.status == pramaan::SolveStatus::kOptimal) {
        // --- Unscale ---
        pramaan::unscale_solution(result.x, result.objective_value, pre_scale, factors);

        // --- Postsolve ---
        std::vector<double> full_x = pramaan::postsolve(result.x, ledger);

        // --- Recompute objective in original space ---
        double obj = original.obj_offset;
        for (int j = 0; j < original.numVars(); ++j) {
            obj += original.obj_coeffs[static_cast<std::size_t>(j)]
                 * full_x[static_cast<std::size_t>(j)];
        }
        std::cout << "  Objective: " << obj << "\n";
        std::cout << "  Iterations: " << result.iterations << "\n";

        // --- Certificate ---
        if (!cert_path.empty()) {
            int ledger_count = static_cast<int>(ledger.size());
            pramaan::Certificate cert = pramaan::generate_certificate(
                original, full_x, status_str, ledger_count);
            try {
                pramaan::write_certificate(cert, cert_path);
                std::cout << "  Certificate written: " << cert_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error writing certificate: " << e.what() << "\n";
                return 1;
            }
        }
    } else {
        std::cout << "  (no solution available)\n";
    }

    return 0;
}
