// backend_cpu.cpp
// P2 Step 1 — CpuBackend implementation
//
// Adapts the existing RevisedSimplex solver to the Backend interface.
// This is a transparent pass-through: no additional logic, no model
// copies, no altered numerical behavior. The result is byte-identical
// to calling RevisedSimplex::solve() directly.
//
// Compiled as part of pramaan_core (host C++ only, no CUDA dependency).
#include "pramaan/gpu/backend.hpp"

#include "pramaan/simplex.hpp"

namespace pramaan {

namespace {

// CpuBackend: delegates LP relaxation solving to RevisedSimplex.
//
// Thread-safety: RevisedSimplex::solve() is stateless (builds and
// discards its own internal state per call), so concurrent calls on
// the same CpuBackend instance with different models are safe.
class CpuBackend final : public Backend {
public:
    SolveResult solve_lp_relaxation(const ModelIR& model,
                                    const RevisedSimplex::Options& options) override {
        const RevisedSimplex solver(options);
        return solver.solve(model);
    }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace pramaan