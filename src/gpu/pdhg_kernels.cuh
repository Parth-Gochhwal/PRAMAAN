// pdhg_kernels.cuh
// P2 Step 2 — Shared CUDA kernel definitions for GPU PDHG solver
//
// This header is included by both pdhg_solver.cu (production solver) and
// pdhg.cu (test harness wrappers).  Defining __global__ kernels in a .cuh
// header and including it in multiple TUs is safe in NVCC because each .cu
// file gets its own kernel instantiation; the linker correctly deduplicates
// the PTX via the CUDA separable compilation model.
//
// Kernels defined here:
//   csr_spmv_kernel   — CSR row-per-thread  y = A x
//   csr_spmv_t_kernel — Atomic scatter      y = Aᵀ x
//   dual_update_kernel
//   primal_update_kernel
//   copy_kernel
#pragma once
#include <cuda_runtime.h>

namespace pramaan {
namespace gpu {

// CSR SpMV: y[i] = sum_{k in row i} A[k] * x[col[k]]
// One CUDA thread per row.
static __global__ void csr_spmv_kernel(const int m,
                                        const int* __restrict__ row_ptr,
                                        const int* __restrict__ col_idx,
                                        const float* __restrict__ values,
                                        const float* __restrict__ x,
                                        float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float sum = 0.0f;
    const int end = row_ptr[i + 1];
    for (int k = row_ptr[i]; k < end; ++k)
        sum += values[k] * x[col_idx[k]];
    y[i] = sum;
}

// Transposed SpMV: y[col[k]] += A[k] * x[i] for each (i, k)
// One thread per row of A; atomic scatter into y (over n columns).
// Caller must zero y before launch.
static __global__ void csr_spmv_t_kernel(const int m,
                                          const int* __restrict__ row_ptr,
                                          const int* __restrict__ col_idx,
                                          const float* __restrict__ values,
                                          const float* __restrict__ x,
                                          float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float xi = x[i];
    const int end = row_ptr[i + 1];
    for (int k = row_ptr[i]; k < end; ++k)
        atomicAdd(&y[col_idx[k]], values[k] * xi);
}

// Dual update with ranged-row projection (Chambolle-Pock):
//
// We solve:  min_x max_y  cᵀx + yᵀ(Ax - Π_{[l_r,u_r]}(Ax))
// The proximal dual update for the indicator f(Ax) = δ(Ax ∈ [l_r, u_r]) is:
//
//   y_new = y + σ·(A x̄) - σ·Π_{[l_r, u_r]}( y/σ + (A x̄) )
//
// Equivalently:
//   w       = y/σ + Ax̄[i]
//   w_proj  = clamp(w, l_r[i], u_r[i])
//   y_new   = y + σ·(Ax̄ - w_proj)   = y + σ·(w - w_proj)·σ/σ ... see below
//           = σ · (w - w_proj) + y   = y + σ·Ax̄ - σ·w_proj
//
// Note: when l_r = -inf and u_r = +inf (free row), w_proj = w, so y_new = y.
// When l_r = u_r = b (equality), w_proj = b always, driving the residual to 0.
// For one-sided inequality (l_r = -inf, u_r = b), w_proj = min(w, b).
static __global__ void dual_update_kernel(const int m,
                                           const float* __restrict__ Ax_bar,
                                           const float* __restrict__ sigma,
                                           const float* __restrict__ l_r,
                                           const float* __restrict__ u_r,
                                           float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float sig = sigma[i];
    if (sig <= 0.0f) return;
    const float w      = y[i] / sig + Ax_bar[i];
    const float w_proj = fminf(fmaxf(w, l_r[i]), u_r[i]);
    // y_new = y + σ*(Ax̄ - w_proj)
    //       = y + σ*(y/σ + Ax̄ - w_proj)·... simplify:
    // y/σ = w - Ax̄  =>  y = σ*(w - Ax̄)
    // y_new = σ*(w - Ax̄) + σ*(Ax̄ - w_proj) = σ*(w - w_proj)
    y[i] = sig * (w - w_proj);
}

// Primal update with variable-box projection and over-relaxation:
//   x_new[j] = clamp(x[j] - τ[j]*(c[j] + ATy[j]), l_x[j], u_x[j])
//   x_bar[j] = 2*x_new[j] - x[j]
//   x[j]     = x_new[j]
static __global__ void primal_update_kernel(const int n,
                                             const float* __restrict__ ATy,
                                             const float* __restrict__ tau,
                                             const float* __restrict__ c,
                                             const float* __restrict__ l_x,
                                             const float* __restrict__ u_x,
                                             float* __restrict__ x,
                                             float* __restrict__ x_bar) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const float x_prev = x[j];
    const float x_new  = fminf(fmaxf(x_prev - tau[j] * (c[j] + ATy[j]), l_x[j]), u_x[j]);
    x_bar[j] = 2.0f * x_new - x_prev;
    x[j]     = x_new;
}

// Element-wise copy: dst[j] = src[j]
static __global__ void copy_kernel(const int n,
                                    const float* __restrict__ src,
                                    float* __restrict__ dst) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    dst[j] = src[j];
}

} // namespace gpu
} // namespace pramaan
