// pdhg.cu
// P2 Step 1 — CUDA PDHG infrastructure scaffold  (retained for backward compat)
// P2 Step 2 — Adds SpMV test-harness launchers (launch_spmv / launch_spmv_t)
//
// ─── STEP 1 CONTENTS ────────────────────────────────────────────────────────
// launch_pdhg_iteration: dense-vector single-iteration launcher used by
//     tests/test_pdhg_kernel.cpp.  NOT a complete LP solver.
//
//   The primal/dual kernels here are intentionally a simplified formulation
//   (scalar τ/σ, nonneg-only primal projection, equality-row dual only).
//   They exist solely to validate CUDA compilation + device memory flow.
//
// ─── STEP 2 ADDITIONS ───────────────────────────────────────────────────────
// launch_spmv:   thin test-harness wrapper that calls the production
//     csr_spmv_kernel from pdhg_kernels.cuh, for SpMV unit tests.
// launch_spmv_t: same for transposed SpMV.
//
// ─── PRIOR ART NOTE ─────────────────────────────────────────────────────────
// PDHG / PDLP are established prior art:
//   Applegate et al. (2021) arXiv:2106.04756  (PDLP, NeurIPS 2021)
//   Lu & Yang (2025) arXiv:2311.12180         (cuPDLP.jl, Operations Research)
// PRAMAAN uses PDHG as a GPU computational foundation, not as a novel
// algorithmic contribution.

#include "cuda_utils.cuh"
#include "pdhg_kernels.cuh"   // shared kernel definitions (Step 2)

#include <cstddef>
#include <vector>

namespace pramaan {
namespace gpu {

// ---------------------------------------------------------------------------
// Step 1 local kernels (dense-vector, simplified update equations)
// These are intentionally separate from the production sparse kernels in
// pdhg_kernels.cuh so that the Step 1 test baseline is independent.
// ---------------------------------------------------------------------------

// Primal update: x_next[j] = max(0, x[j] - tau * gradient[j])
//                x_bar[j]  = 2 * x_next[j] - x[j]
__global__ static void pdhg_primal_update_kernel(const float* __restrict__ x,
                                                  const float* __restrict__ gradient,
                                                  float* __restrict__ x_next,
                                                  float* __restrict__ x_bar,
                                                  const int n,
                                                  const float tau) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const float xj      = x[j];
    const float xj_next = fmaxf(0.0f, xj - tau * gradient[j]);
    x_next[j] = xj_next;
    x_bar[j]  = 2.0f * xj_next - xj;
}

// Dual update: y_next[i] = y[i] + sigma * (Ax_bar[i] - b[i])
__global__ static void pdhg_dual_update_kernel(const float* __restrict__ y,
                                                const float* __restrict__ Ax_bar,
                                                const float* __restrict__ b,
                                                float* __restrict__ y_next,
                                                const int m,
                                                const float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    y_next[i] = y[i] + sigma * (Ax_bar[i] - b[i]);
}

// ---------------------------------------------------------------------------
// launch_pdhg_iteration — Step 1 infrastructure validator
//
// Runs ONE complete primal-dual iteration on dense host vectors.
// h_x and h_y are updated in-place.
// h_gradient = c + Aᵀy  (precomputed by caller)
// h_Ax_bar   = A * x̄    (precomputed by caller)
// h_b        = RHS vector
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
    if (n <= 0 && m <= 0) return;

    constexpr int kBlockSize = 256;

    float* d_x        = nullptr;
    float* d_x_next   = nullptr;
    float* d_x_bar    = nullptr;
    float* d_gradient = nullptr;
    float* d_y        = nullptr;
    float* d_y_next   = nullptr;
    float* d_Ax_bar   = nullptr;
    float* d_b        = nullptr;

    const std::size_t n_bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t m_bytes = static_cast<std::size_t>(m) * sizeof(float);

    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_x,        n_bytes));
        CUDA_CHECK(cudaMalloc(&d_x_next,   n_bytes));
        CUDA_CHECK(cudaMalloc(&d_x_bar,    n_bytes));
        CUDA_CHECK(cudaMalloc(&d_gradient, n_bytes));
        CUDA_CHECK(cudaMemcpy(d_x,        h_x,        n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gradient, h_gradient, n_bytes, cudaMemcpyHostToDevice));
    }
    if (m > 0) {
        CUDA_CHECK(cudaMalloc(&d_y,      m_bytes));
        CUDA_CHECK(cudaMalloc(&d_y_next, m_bytes));
        CUDA_CHECK(cudaMalloc(&d_Ax_bar, m_bytes));
        CUDA_CHECK(cudaMalloc(&d_b,      m_bytes));
        CUDA_CHECK(cudaMemcpy(d_y,      h_y,      m_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ax_bar, h_Ax_bar, m_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,      h_b,      m_bytes, cudaMemcpyHostToDevice));
    }

    if (n > 0) {
        const int grid_n = (n + kBlockSize - 1) / kBlockSize;
        pdhg_primal_update_kernel<<<grid_n, kBlockSize>>>(
            d_x, d_gradient, d_x_next, d_x_bar, n, tau);
        CUDA_CHECK(cudaGetLastError());
    }
    if (m > 0) {
        const int grid_m = (m + kBlockSize - 1) / kBlockSize;
        pdhg_dual_update_kernel<<<grid_m, kBlockSize>>>(
            d_y, d_Ax_bar, d_b, d_y_next, m, sigma);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (n > 0) CUDA_CHECK(cudaMemcpy(h_x, d_x_next, n_bytes, cudaMemcpyDeviceToHost));
    if (m > 0) CUDA_CHECK(cudaMemcpy(h_y, d_y_next, m_bytes, cudaMemcpyDeviceToHost));

    if (d_x)        cudaFree(d_x);
    if (d_x_next)   cudaFree(d_x_next);
    if (d_x_bar)    cudaFree(d_x_bar);
    if (d_gradient) cudaFree(d_gradient);
    if (d_y)        cudaFree(d_y);
    if (d_y_next)   cudaFree(d_y_next);
    if (d_Ax_bar)   cudaFree(d_Ax_bar);
    if (d_b)        cudaFree(d_b);
}

// ---------------------------------------------------------------------------
// Step 2 SpMV test-harness launchers
//
// These wrappers call the production CSR SpMV kernels from pdhg_kernels.cuh,
// allowing tests to validate GPU SpMV directly without going through
// PdhgSolver::solve().  They are NOT used in the solver hot path.
// ---------------------------------------------------------------------------

void launch_spmv(const std::vector<int>& row_ptr,
                 const std::vector<int>& col_idx,
                 const std::vector<float>& values,
                 const std::vector<float>& x,
                 std::vector<float>& y_out,
                 int m, int n) {
    if (m == 0) { y_out.clear(); return; }
    const int nnz = static_cast<int>(values.size());

    int*   d_rp = nullptr;  int*   d_ci = nullptr;
    float* d_v  = nullptr;  float* d_x  = nullptr;  float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rp, static_cast<size_t>(m + 1) * sizeof(int)));
    if (nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_ci, static_cast<size_t>(nnz) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_v,  static_cast<size_t>(nnz) * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(m) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rp, row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_ci, col_idx.data(), nnz * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v,  values.data(),  nnz * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int kBlock = 256;
    csr_spmv_kernel<<<(m + kBlock - 1) / kBlock, kBlock>>>(m, d_rp, d_ci, d_v, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    y_out.resize(static_cast<size_t>(m));
    CUDA_CHECK(cudaMemcpy(y_out.data(), d_y, static_cast<size_t>(m) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_rp);
    if (d_ci) cudaFree(d_ci);
    if (d_v)  cudaFree(d_v);
    cudaFree(d_x);
    cudaFree(d_y);
}

void launch_spmv_t(const std::vector<int>& row_ptr,
                   const std::vector<int>& col_idx,
                   const std::vector<float>& values,
                   const std::vector<float>& x,
                   std::vector<float>& y_out,
                   int m, int n) {
    y_out.assign(static_cast<size_t>(n), 0.0f);
    if (n == 0) return;
    const int nnz = static_cast<int>(values.size());

    int*   d_rp = nullptr;  int*   d_ci = nullptr;
    float* d_v  = nullptr;  float* d_x  = nullptr;  float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rp, static_cast<size_t>(m + 1) * sizeof(int)));
    if (nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_ci, static_cast<size_t>(nnz) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_v,  static_cast<size_t>(nnz) * sizeof(float)));
    }
    if (m > 0) CUDA_CHECK(cudaMalloc(&d_x, static_cast<size_t>(m) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rp, row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_ci, col_idx.data(), nnz * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v,  values.data(),  nnz * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (m > 0) CUDA_CHECK(cudaMemcpy(d_x, x.data(), static_cast<size_t>(m) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, static_cast<size_t>(n) * sizeof(float)));

    if (m > 0) {
        constexpr int kBlock = 256;
        csr_spmv_t_kernel<<<(m + kBlock - 1) / kBlock, kBlock>>>(m, d_rp, d_ci, d_v, d_x, d_y);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y_out.data(), d_y, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_rp);
    if (d_ci) cudaFree(d_ci);
    if (d_v)  cudaFree(d_v);
    if (d_x)  cudaFree(d_x);
    cudaFree(d_y);
}

}  // namespace gpu
}  // namespace pramaan