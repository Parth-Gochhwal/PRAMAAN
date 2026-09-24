// pdhg_solver.cu
// P2 Step 2 — FP32-first GPU PDHG LP Solver (Precision Ladder bottom rung)
//
// PRIOR ART ACKNOWLEDGEMENT
// --------------------------
// GPU PDHG for LP is established prior art.  Key references:
//   [PDLP]   Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M.,
//             O'Donoghue, B., & Schudy, W. (2021). "Practical large-scale
//             linear programming using primal-dual hybrid gradient."
//             NeurIPS 2021. arXiv:2106.04756.
//   [cuPDLP] Lu, H., & Yang, J. (2025). "cuPDLP.jl: A GPU implementation of
//             restarted primal-dual hybrid gradient for linear programming in
//             Julia." Operations Research, 73(6), 3440–3452.
//             arXiv:2311.12180.
//
// PRAMAAN is implementing PDHG as established prior art because its sparse
// linear-algebra workload is a natural GPU target; PDHG/PDLP itself is not
// claimed as PRAMAAN's novel contribution.
//
// LP CANONICALIZATION
// --------------------
// ModelIR:  optimize obj_sense: cᵀx + offset
//           row_lower ≤ A x ≤ row_upper,  var_lower ≤ x ≤ var_upper
//
// Internally we minimize by negating c when obj_sense == kMaximize.
//
// PDHG ITERATION (Chambolle-Pock / primal-dual proximal splitting)
// ----------------------------------------------------------------
// Saddle-point form:
//   min_{x ∈ [l_x, u_x]}  cᵀx + max_y { yᵀAx : Ax ∈ [l_r, u_r] }
//
// The proximal dual update for the range constraint indicator is:
//   w     = y/σ + A x̄        (lifted dual variable)
//   y_new = σ * (w - clamp(w, l_r, u_r))
//         = y + σ*(A x̄) - σ*clamp(y/σ + Ax̄, l_r, u_r)
//
// Equivalently: the dual variable is the "residual from projection":
//   y[i] measures the signed violation of Ax̄[i] ∈ [l_r[i], u_r[i]]
//
// The primal update:
//   x_new = clamp(x - τ*(c + Aᵀy), l_x, u_x)
//   x̄_new = 2*x_new - x   (over-relaxation / extrapolation)
//
// STEP SIZES
// -----------
// We use a conservative diagonal step-size scaling (Ruiz-style):
//   τ_j = α / max(Σ_i |A_ij|, ε)
//   σ_i = α / max(Σ_j |A_ij|, ε)
// with α = 0.95. This enforces the heuristic τ_j * σ_i * |A_ij|² ≤ 0.95² < 1,
// which promotes stability but does not formally guarantee rapid convergence
// on all ill-conditioned models without additional preconditioning.
//
// CONVERGENCE
// -----------
// Every `check_frequency` iterations we compute (in FP64 on host):
//   primal_viol: max violation of Ax ∈ [l_r,u_r] and x ∈ [l_x,u_x]
//   dual_viol:   relative max projected reduced-cost violation / ‖c‖_∞
// Report kOptimal when both ≤ tolerance.
//
// RESTART
// -------
// A restart is triggered when the combined residual (primal + dual) has not
// decreased by 1% over the last kRestartPeriod checks AND the residual is
// non-zero (we never restart from a feasible point; we need dual convergence
// not a feasibility reset).
// On restart: reset y=0, reset x̄=x, reset step-size counters.
// At most kMaxRestarts total restarts.
//
// FP32/FP64 BOUNDARY
// -------------------
// All GPU kernels operate in FP32. Host-side residuals, objective, and
// Aᵀy computations use FP64 via model.A.multiply() and double arithmetic.
// Note: This file implements the GPU fast-pass (the "bottom rung" of the
// Precision Ladder). Automated escalation to the CPU FP64 polishing solver
// is implemented at the caller/orchestrator level, not inside this GPU component.

#include "pramaan/gpu/pdhg_solver.hpp"
#include "cuda_utils.cuh"
#include "pdhg_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace pramaan {
namespace gpu {

// ---------------------------------------------------------------------------
// RAII device-array wrapper
// ---------------------------------------------------------------------------
namespace {
template <typename T>
struct CudaDeleter { void operator()(T* p) const { if (p) cudaFree(p); } };
template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
CudaPtr<T> make_dev(size_t n) {
    if (n == 0) return CudaPtr<T>(nullptr);
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return CudaPtr<T>(p);
}
} // anonymous namespace

// ---------------------------------------------------------------------------
PdhgSolver::PdhgSolver(const PdhgOptions& opts) : options_(opts) {}
PdhgSolver::~PdhgSolver() {}

SolveResult PdhgSolver::solve(const ModelIR& model) {
    const int m   = static_cast<int>(model.numRows());
    const int n   = static_cast<int>(model.numVars());
    const int nnz = static_cast<int>(model.A.nnz());

    SolveResult result;
    result.status    = SolveStatus::kIterationLimit;
    result.iterations = 0;

    // ---- Trivial: no variables ----
    if (n == 0) {
        result.status          = SolveStatus::kOptimal;
        result.objective_value = model.obj_offset;
        result.row_activity.assign(static_cast<size_t>(m), 0.0);
        return result;
    }

    // ---- Guard: NaN/Inf in objective ----
    for (int j = 0; j < n; ++j) {
        if (!std::isfinite(model.obj_coeffs[j])) {
            result.status = SolveStatus::kNumericalFailure;
            return result;
        }
    }

    // ---- Minimisation sign convention ----
    // Internally always minimize: negate c for maximization problems.
    const float obj_sign = (model.obj_sense == ObjSense::kMaximize) ? -1.0f : 1.0f;

    // ---- Build FP32 host vectors ----
    // Use 1e30 as FP32 infinity sentinel (safely representable as float).
    constexpr float kFltInf = 1e30f;

    std::vector<float> h_c(n);
    for (int j = 0; j < n; ++j)
        h_c[j] = static_cast<float>(obj_sign * model.obj_coeffs[j]);

    std::vector<float> h_l_r(m), h_u_r(m);
    for (int i = 0; i < m; ++i) {
        h_l_r[i] = (model.row_lower[i] <= -1e29) ? -kFltInf
                   : static_cast<float>(model.row_lower[i]);
        h_u_r[i] = (model.row_upper[i] >=  1e29) ?  kFltInf
                   : static_cast<float>(model.row_upper[i]);
    }

    std::vector<float> h_l_x(n), h_u_x(n);
    for (int j = 0; j < n; ++j) {
        h_l_x[j] = (model.var_lower[j] <= -1e29) ? -kFltInf
                   : static_cast<float>(model.var_lower[j]);
        h_u_x[j] = (model.var_upper[j] >=  1e29) ?  kFltInf
                   : static_cast<float>(model.var_upper[j]);
    }

    // CSR in int/float
    const int rp_size = m + 1;
    std::vector<int>   h_rp(rp_size);
    std::vector<int>   h_ci(nnz);
    std::vector<float> h_vl(nnz);
    for (int i = 0; i < rp_size; ++i) h_rp[i] = static_cast<int>(model.A.rowPtr()[i]);
    for (int k = 0; k < nnz; ++k)    h_ci[k]  = static_cast<int>(model.A.colIdx()[k]);
    for (int k = 0; k < nnz; ++k)    h_vl[k]  = static_cast<float>(model.A.values()[k]);

    // ---- Diagonal preconditioning (1-norm scaled) ----
    // τ_j = α / Σ_i |A_ij|,  σ_i = α / Σ_j |A_ij|
    // α = 0.95 provides conservative diagonal step-size scaling.
    // This controls the per-entry τ_j σ_i |A_ij|² product and promotes
    // numerical stability; it does not by itself constitute a formal
    // convergence guarantee for every ill-conditioned model.
    const float alpha = 0.95f;
    std::vector<float> h_sigma(m, alpha), h_tau(n, alpha);
    {
        std::vector<double> rn(m, 0.0), cn(n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int k = h_rp[i]; k < h_rp[i + 1]; ++k) {
                double av = std::abs(static_cast<double>(h_vl[k]));
                rn[i] += av;  cn[h_ci[k]] += av;
            }
        for (int i = 0; i < m; ++i)
            if (rn[i] > 1e-12) h_sigma[i] = alpha / static_cast<float>(rn[i]);
        for (int j = 0; j < n; ++j)
            if (cn[j] > 1e-12) h_tau[j]   = alpha / static_cast<float>(cn[j]);
    }

    // ---- Allocate device memory (once per solve) ----
    auto d_rp_a  = make_dev<int>(rp_size);   auto d_ci_a = make_dev<int>(nnz);
    auto d_vl_a  = make_dev<float>(nnz);     auto d_c_a  = make_dev<float>(n);
    auto d_l_r_a = make_dev<float>(m);       auto d_u_r_a= make_dev<float>(m);
    auto d_l_x_a = make_dev<float>(n);       auto d_u_x_a= make_dev<float>(n);
    auto d_sig_a = make_dev<float>(m);       auto d_tau_a= make_dev<float>(n);
    auto d_x_a   = make_dev<float>(n);       auto d_y_a  = make_dev<float>(m);
    auto d_xb_a  = make_dev<float>(n);       // x̄ (extrapolated)
    auto d_Axb_a = make_dev<float>(m);       // A x̄
    auto d_ATy_a = make_dev<float>(n);       // Aᵀ y

    int*   d_rp  = d_rp_a.get();   int*   d_ci  = d_ci_a.get();
    float* d_vl  = d_vl_a.get();   float* d_c   = d_c_a.get();
    float* d_l_r = d_l_r_a.get();  float* d_u_r = d_u_r_a.get();
    float* d_l_x = d_l_x_a.get();  float* d_u_x = d_u_x_a.get();
    float* d_sig = d_sig_a.get();  float* d_tau = d_tau_a.get();
    float* d_x   = d_x_a.get();   float* d_y   = d_y_a.get();
    float* d_xb  = d_xb_a.get();  float* d_Axb = d_Axb_a.get();
    float* d_ATy = d_ATy_a.get();

    // ---- H → D (one-time) ----
    CUDA_CHECK(cudaMemcpy(d_rp, h_rp.data(), rp_size * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_ci, h_ci.data(), nnz * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vl, h_vl.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_c,   h_c.data(),   n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l_x, h_l_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_x, h_u_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, h_tau.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    if (m > 0) {
        CUDA_CHECK(cudaMemcpy(d_l_r, h_l_r.data(),   m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_r, h_u_r.data(),   m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sig, h_sigma.data(),  m * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---- Initialise iterates ----
    // x₀ = clamp(0, l_x, u_x),  y₀ = 0,  x̄₀ = x₀
    {
        std::vector<float> x0(n);
        for (int j = 0; j < n; ++j)
            x0[j] = std::max(h_l_x[j], std::min(0.0f, h_u_x[j]));
        CUDA_CHECK(cudaMemcpy(d_x,  x0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xb, x0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (m > 0) CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));

    // ---- Launch parameters ----
    constexpr int kBlock = 256;
    const int gm = (m > 0) ? (m + kBlock - 1) / kBlock : 1;
    const int gn = (n + kBlock - 1) / kBlock;

    // ---- Host buffers for convergence checks ----
    std::vector<float>  h_x(n, 0.0f), h_y(m, 0.0f);

    // ---- Restart state ----
    // We restart when the combined residual (primal+dual) stagnates,
    // but ONLY when the residual is > 0 (don't restart from a good feasible point).
    double best_res     = std::numeric_limits<double>::infinity();
    int    stagnation   = 0;
    int    restarts     = 0;
    const  int kRestP   = 8;   // checks without 1% improvement → restart
    const  int kMaxR    = 10;  // max restarts

    // ====================================================================
    // MAIN PDHG LOOP
    // ====================================================================
    for (int iter = 0; iter < options_.max_iterations; ++iter) {

        // 1. Dual update:  y ← σ*(y/σ + Ax̄ - clamp(y/σ + Ax̄, l_r, u_r))
        if (m > 0) {
            // 1a. Compute A x̄
            csr_spmv_kernel<<<gm, kBlock>>>(m, d_rp, d_ci, d_vl, d_xb, d_Axb);
            CUDA_CHECK(cudaGetLastError());
            // 1b. Apply dual proximal update
            dual_update_kernel<<<gm, kBlock>>>(m, d_Axb, d_sig, d_l_r, d_u_r, d_y);
            CUDA_CHECK(cudaGetLastError());
        }

        // 2. Primal update: x ← clamp(x - τ(c+Aᵀy), l_x, u_x),  x̄ = 2x-x_prev
        CUDA_CHECK(cudaMemset(d_ATy, 0, n * sizeof(float)));
        if (m > 0) {
            csr_spmv_t_kernel<<<gm, kBlock>>>(m, d_rp, d_ci, d_vl, d_y, d_ATy);
            CUDA_CHECK(cudaGetLastError());
        }
        primal_update_kernel<<<gn, kBlock>>>(n, d_ATy, d_tau, d_c, d_l_x, d_u_x, d_x, d_xb);
        CUDA_CHECK(cudaGetLastError());

        // ---- Convergence check every check_frequency iterations ----
        if ((iter + 1) % options_.check_frequency == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
            if (m > 0)
                CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, m * sizeof(float), cudaMemcpyDeviceToHost));

            // ---- FP64 residual computation ----
            std::vector<double> x_d(n), y_d(m);
            for (int j = 0; j < n; ++j) x_d[j] = static_cast<double>(h_x[j]);
            for (int i = 0; i < m; ++i) y_d[i] = static_cast<double>(h_y[i]);

            // Primal feasibility: max violation of Ax ∈ [l_r,u_r] and x ∈ [l_x,u_x]
            auto act = model.A.multiply(x_d);
            double pv = 0.0;
            for (int i = 0; i < m; ++i) {
                if (act[i] < model.row_lower[i]) pv = std::max(pv, model.row_lower[i] - act[i]);
                if (act[i] > model.row_upper[i]) pv = std::max(pv, act[i] - model.row_upper[i]);
            }
            for (int j = 0; j < n; ++j) {
                if (x_d[j] < model.var_lower[j]) pv = std::max(pv, model.var_lower[j] - x_d[j]);
                if (x_d[j] > model.var_upper[j]) pv = std::max(pv, x_d[j] - model.var_upper[j]);
            }

            // Dual stationarity: relative max projected reduced-cost violation
            // (computed in FP64 using host ATy for accuracy)
            std::vector<double> ATy_d(n, 0.0);
            for (int i = 0; i < m; ++i)
                for (int k = h_rp[i]; k < h_rp[i + 1]; ++k)
                    ATy_d[h_ci[k]] += static_cast<double>(h_vl[k]) * y_d[i];

            double c_norm = 1.0;
            for (int j = 0; j < n; ++j)
                c_norm = std::max(c_norm, static_cast<double>(std::abs(h_c[j])));

            double dv = 0.0;
            for (int j = 0; j < n; ++j) {
                double rc   = static_cast<double>(h_c[j]) + ATy_d[j];
                bool at_lb  = (x_d[j] <= model.var_lower[j] + options_.tolerance);
                bool at_ub  = (x_d[j] >= model.var_upper[j] - options_.tolerance);
                double viol = 0.0;
                if (!at_lb && !at_ub)     viol = std::abs(rc);
                else if (at_lb && !at_ub) viol = std::max(0.0, -rc);
                else if (at_ub && !at_lb) viol = std::max(0.0,  rc);
                dv = std::max(dv, viol);
            }
            dv /= c_norm;

            // ---- Optimal? ----
            if (pv <= options_.tolerance && dv <= options_.tolerance) {
                result.status    = SolveStatus::kOptimal;
                result.iterations = iter + 1;
                break;
            }

            // ---- Restart detection: stagnation of combined residual ----
            // Only restart when combined residual is non-trivially large
            double combined = pv + dv;
            if (combined > options_.tolerance * 10.0) {
                if (combined < best_res * 0.99) {
                    best_res   = combined;
                    stagnation = 0;
                } else {
                    ++stagnation;
                }
                if (stagnation >= kRestP && restarts < kMaxR) {
                    if (m > 0) CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));
                    copy_kernel<<<gn, kBlock>>>(n, d_x, d_xb);
                    CUDA_CHECK(cudaGetLastError());
                    stagnation = 0;
                    best_res   = std::numeric_limits<double>::infinity();
                    ++restarts;
                }
            }
        }
    }

    // ---- Final sync + D → H ----
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (result.status != SolveStatus::kOptimal)
        result.iterations = options_.max_iterations;

    // ---- Construct SolveResult ----
    result.x.resize(n);
    double obj = model.obj_offset;
    for (int j = 0; j < n; ++j) {
        result.x[j] = static_cast<double>(h_x[j]);
        obj += model.obj_coeffs[j] * result.x[j];
    }
    result.objective_value = obj;
    result.row_activity    = model.A.multiply(result.x);
    return result;
}

} // namespace gpu
} // namespace pramaan
