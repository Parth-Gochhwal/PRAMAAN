// backend.hpp
// P2 Step 1 — Compute Backend Interface
//
// Purpose: hardware-sovereign abstraction over LP relaxation solvers.
// The optimization layer depends on this interface, never on CUDA directly.
//
// Current implementations:
//   CpuBackend  — delegates to RevisedSimplex (src/gpu/backend_cpu.cpp)
//
// Future implementations (P2 Step 2+):
//   CudaBackend — GPU PDHG solver
//   HipBackend  — AMD GPU (roadmap)
//   SyclBackend — Intel GPU (roadmap)
//
// IMPORTANT: this header must remain compilable by any standard C++17
// compiler without CUDA headers, CUDA runtime, or GPU-specific types.
#pragma once

#include <memory>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {

// Abstract compute backend for LP relaxation solving.
//
// Each backend encapsulates a complete LP solve path (algorithm choice,
// hardware dispatch, precision strategy) behind a single virtual call.
// The caller never needs to know whether the solve ran on CPU, GPU, or
// a future accelerator.
//
// Thread-safety: implementations should document their own guarantees.
// As a baseline, concurrent calls to solve_lp_relaxation() on the same
// instance with different models must be safe (stateless dispatch).
class Backend {
public:
    virtual ~Backend() = default;

    // Solves the continuous LP relaxation of `model`:
    //
    //     optimize   obj_sense: c^T x + obj_offset
    //     subject to row_lower <= A x <= row_upper
    //                var_lower <= x   <= var_upper
    //
    // Integer variable types in `model` are ignored — this solves the
    // continuous relaxation only. Callers needing integrality enforcement
    // (branch-and-bound) wrap this in their own discrete logic.
    //
    // `options` controls iteration limits and numerical tolerances.
    // Backends that use a different internal algorithm (e.g. PDHG) may
    // interpret these as guidelines rather than exact simplex parameters.
    //
    // Returns a SolveResult with the same semantics as RevisedSimplex::solve().
    virtual SolveResult solve_lp_relaxation(const ModelIR& model,
                                            const RevisedSimplex::Options& options) = 0;
};

// Factory function: creates the default CPU backend (RevisedSimplex).
// Defined in backend_cpu.cpp.
std::unique_ptr<Backend> make_cpu_backend();

}  // namespace pramaan