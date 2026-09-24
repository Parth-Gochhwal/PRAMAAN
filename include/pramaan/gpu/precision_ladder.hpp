// precision_ladder.hpp
// P2 Step 3 — Precision Ladder Controller
//
// OVERVIEW
// --------
// The Precision Ladder is a thin orchestrator implementing the two-rung solve
// strategy at the heart of PRAMAAN's heterogeneous engine:
//
//   Rung 1 (fast pass):  FP32 GPU PDHG — high throughput, approximate
//   Rung 2 (polish):     FP64 CPU Revised Simplex — high accuracy, deterministic
//
// The controller runs the GPU fast pass, evaluates its primal and dual
// residuals, and decides whether the result meets the configured accuracy
// threshold.  Only if it does is the GPU result returned directly.  Otherwise
// (high residual, iteration-limit stop, or numerical failure) the GPU primal
// point is passed to the CPU solver as a warm-start hint, and the CPU FP64
// result becomes the final answer.
//
// WHY FP32 GPU FIRST?
//   GPU FP32 PDHG can process many iterations cheaply and often converges to
//   LP solutions that are "good enough" for practical use.  It is treated as a
//   fast approximate pass, not a certified solver.
//
// WHY RESIDUAL GATING?
//   FP32 arithmetic introduces rounding errors that FP64 does not.  A GPU
//   result is only accepted when its measured primal feasibility violation and
//   dual stationarity violation are both within the configured threshold.
//   Accepting without this check would silently return an inaccurate solution.
//
// WHY CPU FP64 POLISH?
//   The FP64 Revised Simplex produces a basis whose basic solution is optimal
//   within FP64 numerical tolerances (residuals at machine-double precision,
//   ~1e-9 or better).  When the GPU result is rejected, the CPU solver is
//   seeded with the GPU primal point as a primal_start_hint.
//
// HOW THE GPU SOLUTION INITIALISES THE CPU SOLVER
//   PDHG returns a primal point x, not a simplex basis. The hint is passed to
//   the Revised Simplex solver, which uses it in a crash-basis heuristic.
//   It attempts to pivot likely-basic variables into the basis using strict
//   primal ratio tests before Phase 1 begins, displacing artificial variables.
//   Phase 1 artificial variables still run for any remaining infeasibilities,
//   but the number of pivots may be greatly reduced.
//
// THRESHOLD SEMANTICS
//   residual_threshold T means:
//     GPU result is accepted  ⟺  primal_residual ≤ T  AND  dual_residual ≤ T
//                                 AND  gpu_status ∈ {kOptimal}
//   Smaller T → stricter acceptance → more escalations → higher final accuracy.
//   Larger  T → more GPU results accepted → less CPU polish → lower total cost.
//
// ESCALATION
//   Any of the following triggers CPU polish regardless of T:
//     - GPU status != kOptimal (e.g. kIterationLimit, kNumericalFailure)
//     - primal_residual > T
//     - dual_residual   > T
//
// DESIGN CONSTRAINTS
//   - PrecisionLadder is a THIN CONTROLLER.  It does not duplicate PDHG
//     kernels, simplex logic, or residual engines.
//   - It calls PdhgSolver::solve() and RevisedSimplex::solve() unchanged.
//   - The residuals are computed host-side from the returned x using
//     model.A.multiply() from pramaan_core (FP64, already available).
//   - No CUDA headers appear in this file; it compiles in CPU-only builds.
//     When CUDA is disabled, the GPU stage is skipped and the CPU stage always runs.
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

// PdhgOptions is declared in pramaan/gpu/pdhg_solver.hpp which requires CUDA
// headers; we forward-declare only what we need here to keep this header
// CUDA-free and compilable in CPU-only builds.
namespace pramaan {
namespace gpu {
struct PdhgOptions;
}  // namespace gpu
}  // namespace pramaan

namespace pramaan {

// ---------------------------------------------------------------------------
// PrecisionLadderResult
//
// Wraps the final SolveResult with metadata that allows callers to
// distinguish which solve stage produced the result and inspect solver health.
// The existing SolveResult type is not modified; this wrapper keeps
// controller-specific metadata isolated from core solver types.
// ---------------------------------------------------------------------------
struct PrecisionLadderResult {
    // Which stage produced the final result.
    enum class Stage {
        kGpuFp32,         // GPU FP32 PDHG result met the residual threshold
        kCpuFp64Polish,   // GPU result was rejected; CPU FP64 simplex provided the final result
    };

    // Reason for escalation (empty if GPU was accepted).
    enum class EscalationReason {
        kNone,              // GPU was accepted; no escalation
        kHighPrimalResidual,// primal_residual > threshold
        kHighDualResidual,  // dual_residual   > threshold
        kGpuIterationLimit, // GPU stopped at max_iterations without convergence
        kGpuNumericalFailure, // GPU reported kNumericalFailure
        kGpuNotOptimal,     // GPU status is not kOptimal (catch-all)
        kGpuDisabled,       // CUDA not compiled in; GPU stage skipped
    };

    // ---- Final answer ----
    SolveResult final_result;
    Stage stage = Stage::kCpuFp64Polish;

    // ---- GPU metadata ----
    SolveStatus gpu_status      = SolveStatus::kIterationLimit;
    double gpu_primal_residual  = std::numeric_limits<double>::infinity();
    double gpu_dual_residual    = std::numeric_limits<double>::infinity();
    double gpu_objective        = 0.0;
    int    gpu_iterations       = 0;
    double gpu_elapsed_ms       = 0.0;

    // ---- CPU polish metadata ----
    // cpu_elapsed_ms is 0 when GPU was accepted (no CPU solve performed).
    double cpu_elapsed_ms       = 0.0;

    // ---- Escalation ----
    bool             escalated          = false;
    EscalationReason escalation_reason  = EscalationReason::kNone;

    // ---- Timing ----
    double total_elapsed_ms = 0.0;

    // Human-readable escalation reason string.
    std::string escalation_reason_str() const;
};

// ---------------------------------------------------------------------------
// PrecisionLadder
//
// Thin two-rung LP solve controller.
//
// Preferred usage:
//
//   PrecisionLadder::Options opts;
//   opts.residual_threshold = 1e-6;  // accept GPU if residual ≤ this
//   PrecisionLadder ladder(opts);
//   PrecisionLadderResult r = ladder.solve(model);
//
//   if (r.escalated) {
//       // r.final_result is from CPU FP64 polish
//   } else {
//       // r.final_result is from GPU FP32 fast pass
//   }
// ---------------------------------------------------------------------------
class PrecisionLadder {
public:
    struct Options {
        // Residual threshold T.
        // GPU result is accepted only when:
        //   gpu_status == kOptimal
        //   AND primal_residual <= T
        //   AND dual_residual   <= T
        //
        // Default: 1e-6, matching PdhgOptions::tolerance (the GPU solver's own
        // convergence criterion).  A result that passed the GPU's internal check
        // should also pass this threshold under the same definition.
        //
        // To accept any kOptimal GPU result, set a large value (e.g., infinity).
        // To force CPU polish for any non-zero residual, set 0.0. Note that
        // a perfect 0.0 residual is possible, though rare, in FP32, so this does
        // not mathematically guarantee escalation, but practically ensures it.
        double residual_threshold = 1e-6;

        // GPU PDHG solver options.  The default tolerance (1e-6) and iteration
        // limit (100 000) are inherited from PdhgOptions defaults.
        // max_iterations controls when GPU escalation occurs.
        int  gpu_max_iterations   = 100000;
        int  gpu_check_frequency  = 100;
        double gpu_tolerance      = 1e-6;

        // CPU FP64 Revised Simplex options.
        RevisedSimplex::Options cpu_options;
    };

    PrecisionLadder() = default;
    explicit PrecisionLadder(Options options) : options_(options) {}

    // Solves `model` via the Precision Ladder.
    // Thread-safety: no mutable state is stored between calls — the same
    // PrecisionLadder instance may be reused across models.
    PrecisionLadderResult solve(const ModelIR& model) const;

    const Options& options() const noexcept { return options_; }

private:
    Options options_{};

};

}  // namespace pramaan

