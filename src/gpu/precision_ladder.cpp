// precision_ladder.cpp
// P2 Step 3 — Precision Ladder Controller Implementation
//
// This file is pure C++17 with no CUDA headers. It is compiled as its own
// pramaan_precision_ladder library and works in both CPU-only and CUDA builds.
//
// When PRAMAAN_ENABLE_CUDA is defined (CUDA build), the GPU stage calls
// PdhgSolver::solve() and populates GPU metadata.  When it is not defined
// (CPU-only build), the GPU stage is skipped and the controller always
// escalates to the CPU FP64 solver.
//
// RESIDUAL DEFINITIONS USED
// --------------------------
// The Precision Ladder uses the actual primal and dual residuals computed by
// PdhgSolver. PdhgSolver computes these residuals in FP64 on the host using the
// final returned iterate before returning them in SolveResult.

#include "pramaan/gpu/precision_ladder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

#ifdef PRAMAAN_ENABLE_CUDA
#include "pramaan/gpu/pdhg_solver.hpp"
#endif

namespace pramaan {

// ---------------------------------------------------------------------------
// PrecisionLadderResult helpers
// ---------------------------------------------------------------------------
std::string PrecisionLadderResult::escalation_reason_str() const {
    switch (escalation_reason) {
        case EscalationReason::kNone:               return "none";
        case EscalationReason::kHighPrimalResidual: return "high_primal_residual";
        case EscalationReason::kHighDualResidual:   return "high_dual_residual";
        case EscalationReason::kGpuIterationLimit:  return "gpu_iteration_limit";
        case EscalationReason::kGpuNumericalFailure:return "gpu_numerical_failure";
        case EscalationReason::kGpuNotOptimal:      return "gpu_not_optimal";
        case EscalationReason::kGpuDisabled:        return "gpu_disabled";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Residual computation (FP64, host-side)
// ---------------------------------------------------------------------------


PrecisionLadderResult PrecisionLadder::solve(const ModelIR& model) const {
    using Clock = std::chrono::high_resolution_clock;

    PrecisionLadderResult result;
    const auto t_start = Clock::now();

    // ========================================================================
    // RUNG 1: GPU FP32 PDHG fast pass
    // ========================================================================
    // When CUDA is not compiled, the GPU stage is skipped entirely and the
    // controller immediately escalates to CPU FP64.
#ifdef PRAMAAN_ENABLE_CUDA
    {

        gpu::PdhgOptions gpu_opts;
        gpu_opts.tolerance       = options_.gpu_tolerance;
        gpu_opts.max_iterations  = options_.gpu_max_iterations;
        gpu_opts.check_frequency = options_.gpu_check_frequency;

        const auto t_gpu0 = Clock::now();
        gpu::PdhgSolver gpu_solver(gpu_opts);
        SolveResult gpu_res = gpu_solver.solve(model);
        const auto t_gpu1 = Clock::now();

        result.gpu_elapsed_ms =
            std::chrono::duration<double, std::milli>(t_gpu1 - t_gpu0).count();
        result.gpu_status     = gpu_res.status;
        result.gpu_iterations = gpu_res.iterations;
        result.gpu_objective  = gpu_res.objective_value;

        // Compute residuals from the returned GPU primal point (FP64).
        // (Now uses the actual residuals computed by PdhgSolver in its final convergence check)
        result.gpu_primal_residual = gpu_res.primal_residual;
        result.gpu_dual_residual   = gpu_res.dual_residual;

        // ---- Acceptance decision ----
        // The GPU result is accepted only when:
        //   (a) status is kOptimal
        //   (b) primal_residual <= residual_threshold
        //   (c) dual_residual   <= residual_threshold
        //
        // Any other condition triggers escalation to CPU FP64 polish.
        const double T = options_.residual_threshold;
        bool accept = false;

        if (gpu_res.status == SolveStatus::kOptimal) {
            if (result.gpu_primal_residual <= T && result.gpu_dual_residual <= T) {
                accept = true;
            } else if (result.gpu_primal_residual > T) {
                result.escalation_reason =
                    PrecisionLadderResult::EscalationReason::kHighPrimalResidual;
            } else {
                result.escalation_reason =
                    PrecisionLadderResult::EscalationReason::kHighDualResidual;
            }
        } else if (gpu_res.status == SolveStatus::kIterationLimit) {
            result.escalation_reason =
                PrecisionLadderResult::EscalationReason::kGpuIterationLimit;
        } else if (gpu_res.status == SolveStatus::kNumericalFailure) {
            result.escalation_reason =
                PrecisionLadderResult::EscalationReason::kGpuNumericalFailure;
        } else {
            result.escalation_reason =
                PrecisionLadderResult::EscalationReason::kGpuNotOptimal;
        }

        if (accept) {
            // GPU result satisfies residual criteria: return it directly.
            result.stage        = PrecisionLadderResult::Stage::kGpuFp32;
            result.escalated    = false;
            result.final_result = std::move(gpu_res);
            result.total_elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - t_start).count();
            return result;
        }

        // GPU rejected: fall through to CPU FP64 polish.
        // Pass the GPU x as a warm-start hint to the CPU solver.
        result.escalated = true;
        result.stage     = PrecisionLadderResult::Stage::kCpuFp64Polish;

        // ---- CPU FP64 polish with GPU primal initialization hint ----
        // The GPU primal vector x is passed to RevisedSimplex to construct a
        // crash-basis initialization. This attempts to pivot likely-basic
        // variables into the basis before Phase 1 begins.
        RevisedSimplex::Options cpu_opts = options_.cpu_options;
        if (!gpu_res.x.empty() &&
            static_cast<int>(gpu_res.x.size()) == model.numVars()) {
            cpu_opts.primal_start_hint = gpu_res.x;  // GPU x → CPU warm-start
        }

        const auto t_cpu0 = Clock::now();
        RevisedSimplex cpu_solver(cpu_opts);
        result.final_result = cpu_solver.solve(model);
        const auto t_cpu1 = Clock::now();
        result.cpu_elapsed_ms =
            std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();
    }
#else
    // ---- CUDA disabled: GPU stage skipped, always use CPU FP64 ----
    // When compiled without CUDA, no GPU hardware is available.
    // The controller escalates immediately with reason kGpuDisabled.
    result.escalated         = true;
    result.stage             = PrecisionLadderResult::Stage::kCpuFp64Polish;
    result.escalation_reason =
        PrecisionLadderResult::EscalationReason::kGpuDisabled;

    {
        const auto t_cpu0 = Clock::now();
        RevisedSimplex cpu_solver(options_.cpu_options);
        result.final_result = cpu_solver.solve(model);
        const auto t_cpu1 = Clock::now();
        result.cpu_elapsed_ms =
            std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();
    }
#endif

    result.total_elapsed_ms =
        std::chrono::duration<double, std::milli>(
            Clock::now() - t_start).count();
    return result;
}

}  // namespace pramaan
