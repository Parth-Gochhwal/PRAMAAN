// solve_main.cpp
// The pramaan-solve entry point. Wires together: mps_parser -> ModelIR ->
// Presolver -> RevisedSimplex -> Certificate -> print result.
//
// Usage: pramaan-solve <model.mps> [certificate.cert] [--gpu|--cpu|--auto]
//
// If a certificate path is provided, the solver writes a PRAMAAN_CERTIFICATE_V1
// certificate file after solving.
//
// GPU path: runs GPU PDHG candidate generation followed by a full CPU FP64
// final solve. The GPU candidate is not used to warm-start the CPU simplex
// (no basis is available from PDHG). The CPU solve always runs to completion
// and is the source of the final result. No GPU speedup is claimed.
#include <iostream>
#include <string>

#include "pramaan/certificate.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/presolve.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/branch_and_bound.hpp"
#include "pramaan/transformation_ledger.hpp"

#ifdef PRAMAAN_ENABLE_CUDA
#include "pramaan/gpu/pdhg_solver.hpp"
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pramaan-solve <model.mps> [certificate.cert] [--gpu|--cpu|--auto]\n";
        return 1;
    }

    std::string model_path;
    std::string cert_path;
    std::string mode = "";
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu" || arg == "--cpu" || arg == "--auto") {
            if (mode_set) {
                std::cerr << "Error: multiple backend flags.\n";
                return 1;
            }
            mode = arg;
            mode_set = true;
        } else if (arg.rfind("--", 0) == 0 || arg.rfind("-", 0) == 0) {
            std::cerr << "Error: unknown option " << arg << "\n";
            return 1;
        } else if (model_path.empty()) {
            model_path = arg;
        } else if (cert_path.empty()) {
            cert_path = arg;
        } else {
            std::cerr << "Error: too many positional arguments.\n";
            return 1;
        }
    }
    if (!mode_set) {
        mode = "--auto";
    }

    if (model_path.empty()) {
        std::cerr << "Usage: pramaan-solve <model.mps> [certificate.cert] [--gpu|--cpu|--auto]\n";
        return 1;
    }

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

    bool is_milp = false;
    for (auto type : reduced.var_types) {
        if (type == pramaan::VarType::kInteger) {
            is_milp = true;
            break;
        }
    }

    // --- Scale ---
    pramaan::ModelIR pre_scale = reduced;
    pramaan::ScalingFactors factors;
    if (!is_milp) {
        // Ruiz scaling is only applied to pure LP problems.
        // It is NOT applied to MILP because it changes variable bounds in ways
        // that are not safe for integer variables.
        factors = pramaan::ruiz_scale(reduced, ledger, 10);
    }

    // --- Solver Path Selection ---
    bool use_gpu = false;
    bool force_gpu = false;
    if (mode == "--gpu") { use_gpu = true; force_gpu = true; }
    else if (mode == "--cpu") { use_gpu = false; }
    else if (mode == "--auto") {
        use_gpu = !is_milp && (reduced.numRows() > 1000 || reduced.numVars() > 1000);
    }

    if (is_milp && force_gpu) {
        std::cerr << "Error: GPU solver (--gpu) does not support MILP (integer variables). Please use --cpu or --auto.\n";
        return 1;
    }

#ifndef PRAMAAN_ENABLE_CUDA
    if (use_gpu) {
        std::cerr << "Warning: GPU solver requested but CUDA is not enabled. Falling back to CPU.\n";
        use_gpu = false;
    }
#endif

    pramaan::SolveResult result;
    if (is_milp) {
        std::cout << "  Backend: CPU (MILP Branch-and-Bound)\n";
        pramaan::mip::BranchAndBound::Options bb_options;
        bb_options.num_threads = 4;
        bb_options.node_selection = pramaan::mip::BranchAndBound::Options::NodeSelection::kBestBound;
        bb_options.use_warm_start = true;
        pramaan::mip::BranchAndBound bb(bb_options);
        pramaan::mip::MipResult mip_res = bb.solve(reduced);

        if (mip_res.status == pramaan::mip::MipStatus::kOptimal) {
            result.status = pramaan::SolveStatus::kOptimal;
            result.x = mip_res.x;
            result.objective_value = mip_res.objective_value;
            result.iterations = mip_res.statistics.nodes_explored;
        } else if (mip_res.status == pramaan::mip::MipStatus::kInfeasible) {
            result.status = pramaan::SolveStatus::kInfeasible;
        } else if (mip_res.status == pramaan::mip::MipStatus::kNumericalFailure) {
            result.status = pramaan::SolveStatus::kNumericalFailure;
        } else {
            // kNodeLimit or any other inconclusive status
            result.status = pramaan::SolveStatus::kIterationLimit;
        }
    } else if (use_gpu) {
        // GPU path: generate a PDHG candidate, then run a full CPU FP64 solve.
        // The GPU result is not used to warm-start the CPU simplex (PDHG does
        // not produce a simplex basis). The CPU solve always runs to completion
        // and provides the final certified result.
        std::cout << "  Backend: GPU PDHG candidate generation followed by CPU FP64 final solve\n";
#ifdef PRAMAAN_ENABLE_CUDA
        pramaan::gpu::PdhgSolver gpu_solver;
        // GPU candidate (approximate, not certified):
        pramaan::SolveResult gpu_candidate = gpu_solver.solve(reduced);
        (void)gpu_candidate;  // candidate not used to warm-start; reserved for future work

        // CPU FP64 final solve (certified result):
        std::cout << "  Precision Ladder: running CPU FP64 final solve\n";
        pramaan::RevisedSimplex cpu_solver;
        result = cpu_solver.solve(reduced);
#endif
    } else {
        std::cout << "  Backend: CPU (Revised Simplex)\n";
        pramaan::RevisedSimplex solver;
        result = solver.solve(reduced);
    }

    // --- Map status to string ---
    std::string status_str;
    switch (result.status) {
        case pramaan::SolveStatus::kOptimal:       status_str = "OPTIMAL"; break;
        case pramaan::SolveStatus::kInfeasible:    status_str = "INFEASIBLE"; break;
        case pramaan::SolveStatus::kUnbounded:     status_str = "UNBOUNDED"; break;
        case pramaan::SolveStatus::kIterationLimit: status_str = "ITERATION_LIMIT"; break;
        case pramaan::SolveStatus::kNumericalFailure: status_str = "NUMERICAL_FAILURE"; break;
        default:                                   status_str = "UNKNOWN"; break;
    }
    std::cout << "  Status: " << status_str << "\n";

    if (result.status == pramaan::SolveStatus::kOptimal) {
        // --- Unscale ---
        if (!is_milp) {
            pramaan::unscale_solution(result.x, result.objective_value, pre_scale, factors);
        }

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
