// pdhg.cu
// P2 Step 1 — CUDA PDHG infrastructure stub
//
// Purpose: prove that the CUDA compilation toolchain works and establish
// the kernel/data-flow foundation for the full GPU PDHG solver in P2 Step 2.
//
// This file contains:
//   1. A single __global__ kernel performing one PDHG primal-dual iteration
//      on dense vectors (x, y) for a problem of the form:
//          min  c^T x
//          s.t. Ax = b,  x >= 0
//      using the update rules:
//          x_next = max(0, x - tau * (c + A^T y))
//          y_next = y + sigma * (A * x_bar - b)
//      where x_bar = 2 * x_next - x  (extrapolation).
//
//   2. A host-side launcher function that allocates device memory, copies
//      data, runs one iteration, and copies the result back.
//
// This is NOT a complete LP solver. It intentionally:
//   - Operates on dense vectors only (no CSR SpMV yet)
//   - Runs exactly one iteration (no convergence loop)
//   - Does not expose a Backend-compatible interface
//   - Does not claim to produce a correct LP solution
//
// P2 Step 2 will extend this into a full CudaBackend with CSR SpMV,
// convergence checking, and proper integration with the Backend interface.
//
// Compiled by NVCC as part of pramaan_gpu. Not included in pramaan_core.

#include "cuda_utils.cuh"

#include <cstddef>
#include <vector>

namespace pramaan {
namespace gpu {

// ---------------------------------------------------------------------------
// Kernel: one PDHG primal update step
//
// For each coordinate j in [0, n):
//   x_next[j] = max(0, x[j] - tau * gradient[j])
//   x_bar[j]  = 2 * x_next[j] - x[j]   (extrapolation)
//
// `gradient` is expected to contain c + A^T y, precomputed by the caller.
// The non-negativity projection (max(0, ...)) handles the x >= 0 constraint.
// ---------------------------------------------------------------------------
__global__ void pdhg_primal_update_kernel(const float* __restrict__ x,
                                          const float* __restrict__ gradient,
                                          float* __restrict__ x_next,
                                          float* __restrict__ x_bar,
                                          const int n,
                                          const float tau) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }

    const float xj = x[j];
    const float xj_next = fmaxf(0.0f, xj - tau * gradient[j]);

    x_next[j] = xj_next;
    x_bar[j] = 2.0f * xj_next - xj;  // extrapolation
}

// ---------------------------------------------------------------------------
// Kernel: one PDHG dual update step
//
// For each coordinate i in [0, m):
//   y_next[i] = y[i] + sigma * (Ax_bar[i] - b[i])
//
// `Ax_bar` is expected to contain A * x_bar, precomputed by the caller.
// No projection — dual variables are unconstrained for equality constraints.
// ---------------------------------------------------------------------------
__global__ void pdhg_dual_update_kernel(const float* __restrict__ y,
                                        const float* __restrict__ Ax_bar,
                                        const float* __restrict__ b,
                                        float* __restrict__ y_next,
                                        const int m,
                                        const float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) {
        return;
    }

    y_next[i] = y[i] + sigma * (Ax_bar[i] - b[i]);
}

// ---------------------------------------------------------------------------
// Host-side launcher: runs one complete PDHG iteration on dense data.
//
// This is an internal function for infrastructure validation. It is NOT
// part of the Backend interface and NOT exposed through any public header.
//
// Parameters:
//   h_x        — primal variables (size n), modified in-place
//   h_y        — dual variables (size m), modified in-place
//   h_gradient — c + A^T y (size n), precomputed by caller
//   h_Ax_bar   — A * x_bar (size m), precomputed by caller
//   h_b        — RHS vector (size m)
//   n          — number of primal variables
//   m          — number of dual variables / constraints
//   tau        — primal step size
//   sigma      — dual step size
//
// After return, h_x and h_y contain the updated primal/dual iterates.
// h_gradient and h_Ax_bar are not modified (they would need recomputation
// with the new x_bar for the next iteration, but that requires SpMV which
// is deferred to P2 Step 2).
// ---------------------------------------------------------------------------
void launch_pdhg_iteration(float* h_x,
                           float* h_y,
                           const float* h_gradient,
                           const float* h_Ax_bar,
                           const float* h_b,
                           const int n,
                           const int m,
                           const float tau,
                           const float sigma) {
    if (n <= 0 && m <= 0) {
        return;  // nothing to do
    }

    constexpr int kBlockSize = 256;

    // --- Device allocations ---
    float* d_x = nullptr;
    float* d_x_next = nullptr;
    float* d_x_bar = nullptr;
    float* d_gradient = nullptr;
    float* d_y = nullptr;
    float* d_y_next = nullptr;
    float* d_Ax_bar = nullptr;
    float* d_b = nullptr;

    const std::size_t n_bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t m_bytes = static_cast<std::size_t>(m) * sizeof(float);

    // Primal allocations (only if n > 0)
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_x, n_bytes));
        CUDA_CHECK(cudaMalloc(&d_x_next, n_bytes));
        CUDA_CHECK(cudaMalloc(&d_x_bar, n_bytes));
        CUDA_CHECK(cudaMalloc(&d_gradient, n_bytes));

        CUDA_CHECK(cudaMemcpy(d_x, h_x, n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gradient, h_gradient, n_bytes, cudaMemcpyHostToDevice));
    }

    // Dual allocations (only if m > 0)
    if (m > 0) {
        CUDA_CHECK(cudaMalloc(&d_y, m_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_next, m_bytes));
        CUDA_CHECK(cudaMalloc(&d_Ax_bar, m_bytes));
        CUDA_CHECK(cudaMalloc(&d_b, m_bytes));

        CUDA_CHECK(cudaMemcpy(d_y, h_y, m_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ax_bar, h_Ax_bar, m_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, m_bytes, cudaMemcpyHostToDevice));
    }

    // --- Primal update ---
    if (n > 0) {
        const int grid_n = (n + kBlockSize - 1) / kBlockSize;
        pdhg_primal_update_kernel<<<grid_n, kBlockSize>>>(d_x, d_gradient, d_x_next, d_x_bar, n,
                                                          tau);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Dual update ---
    if (m > 0) {
        const int grid_m = (m + kBlockSize - 1) / kBlockSize;
        pdhg_dual_update_kernel<<<grid_m, kBlockSize>>>(d_y, d_Ax_bar, d_b, d_y_next, m, sigma);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Synchronize and copy back ---
    CUDA_CHECK(cudaDeviceSynchronize());

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(h_x, d_x_next, n_bytes, cudaMemcpyDeviceToHost));
    }
    if (m > 0) {
        CUDA_CHECK(cudaMemcpy(h_y, d_y_next, m_bytes, cudaMemcpyDeviceToHost));
    }

    // --- Cleanup ---
    if (d_x) cudaFree(d_x);
    if (d_x_next) cudaFree(d_x_next);
    if (d_x_bar) cudaFree(d_x_bar);
    if (d_gradient) cudaFree(d_gradient);
    if (d_y) cudaFree(d_y);
    if (d_y_next) cudaFree(d_y_next);
    if (d_Ax_bar) cudaFree(d_Ax_bar);
    if (d_b) cudaFree(d_b);
}

}  // namespace gpu
}  // namespace pramaan